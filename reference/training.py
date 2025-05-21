import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from datasets import load_metric
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LanguageModelTrainer:
    """
    A class for training and fine-tuning language models using LoRA.
    
    Attributes:
        model_name (str): Name or path of the pre-trained model
        data_path (str): Path to the training data CSV file
        output_dir (str): Directory to save the trained model
    """
    
    def __init__(self, model_name: str, data_path: str, output_dir: str):
        """
        Initialize the LanguageModelTrainer.

        Args:
            model_name (str): Name or path of the pre-trained model
            data_path (str): Path to the training data CSV file
            output_dir (str): Directory to save the trained model
        """
        self.model_name = model_name
        self.data_path = data_path
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self.config: Optional[LoraConfig] = None
        self.train_data = None
        self.eval_data = None
        self.trainer = None

    def setup_model(self) -> None:
        """Set up the tokenizer, model, and LoRA configuration."""
        try:
            logger.info("Setting up tokenizer and model...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding=True,
                return_tensors="pt"
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",  # Changed to "auto" for better device management
                quantization_config=bnb_config
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Configuring LoRA...")
            self.config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj", "lm_head"
                ],
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            self.model = get_peft_model(self.model, self.config)
            
        except Exception as e:
            logger.error(f"Error in setting up model: {str(e)}")
            raise

    def load_data(self) -> None:
        """Load and preprocess training and evaluation data."""
        try:
            logger.info(f"Loading data from {self.data_path}...")
            df = pd.read_csv(self.data_path)
            
            if 'conversation' not in df.columns:
                raise ValueError("CSV file must contain a 'conversation' column")
            
            # Using 80-20 split instead of fixed numbers
            train_size = int(0.8 * len(df))
            train_texts = df['conversation'][:train_size]
            eval_texts = df['conversation'][train_size:]
            
            self.train_data = self._tokenize_texts(train_texts, "train")
            self.eval_data = self._tokenize_texts(eval_texts, "eval")
            
        except Exception as e:
            logger.error(f"Error in loading data: {str(e)}")
            raise

    def _tokenize_texts(self, texts: List[str], split_name: str) -> List[Dict[str, torch.Tensor]]:
        """
        Tokenize a list of texts.
        
        Args:
            texts (List[str]): List of text strings to tokenize
            split_name (str): Name of the split (train/eval) for logging
            
        Returns:
            List[Dict[str, torch.Tensor]]: List of tokenized texts
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call setup_model first.")
        
        return [
            self.tokenizer(
                text,
                truncation=True,
                #max_length=1024,
                padding=True,
                return_tensors="pt"
            ) for text in tqdm(texts, desc=f"Tokenizing {split_name} data")
        ]

    @staticmethod
    def evaluate_metrics(eval_pred: Any, tokenizer: AutoTokenizer) -> Dict[str, float]:
        """
        Compute evaluation metrics including ROUGE and accuracy.
        
        Args:
            eval_pred: Evaluation predictions object
            tokenizer: Tokenizer for decoding predictions
            
        Returns:
            Dict[str, float]: Dictionary containing evaluation metrics
        """
        try:
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=-1)
            
            # Decode predictions and labels
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            # Load metrics
            rouge = load_metric("rouge")
            glue = load_metric("glue", "sst2")
            
            # Compute ROUGE scores
            rouge_scores = rouge.compute(
                predictions=decoded_preds,
                references=decoded_labels,
                use_aggregator=True
            )
            
            # Compute accuracy
            accuracy = sum(
                1 for pred, label in zip(decoded_preds, decoded_labels)
                if pred.strip() == label.strip()
            ) / len(decoded_preds)
            
            return {
                "accuracy": accuracy,
                "rouge1": rouge_scores["rouge1"].mid.fmeasure,
                "rouge2": rouge_scores["rouge2"].mid.fmeasure,
                "rougeL": rouge_scores["rougeL"].mid.fmeasure,
            }
            
        except Exception as e:
            logger.error(f"Error in computing metrics: {str(e)}")
            return {"error": 0.0}

    def train(self) -> None:
        """Configure and run the training process."""
        try:
            logger.info("Setting up training arguments...")
            training_args = Seq2SeqTrainingArguments(
                output_dir=self.output_dir,
                do_train=True,
                do_eval=True,
                eval_steps=20,
                logging_steps=20,
                save_steps=20,
                evaluation_strategy="steps",
                save_strategy="steps",
                logging_strategy="steps",
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=4,  # Increased for better stability
                eval_accumulation_steps=1,
                learning_rate=2.5e-4,
                save_total_limit=3,
                bf16=True,
                load_best_model_at_end=True,
                num_train_epochs=3,
                optim="paged_adamw_8bit",
                predict_with_generate=True,
                generation_max_length=1000,
                report_to=None,  # Changed to enable tensorboard logging
                remove_unused_columns=False,  # Added to prevent column removal issues
            )

            logger.info("Initializing trainer...")
            self.trainer = Seq2SeqTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                args=training_args,
                data_collator=DataCollatorForLanguageModeling(
                    tokenizer=self.tokenizer,
                    mlm=False
                ),
                train_dataset=self.train_data,
                eval_dataset=self.eval_data,
                compute_metrics=lambda pred: self.evaluate_metrics(pred, self.tokenizer)
            )

            logger.info("Starting training...")
            self.trainer.train()
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    def save_model(self) -> None:
        """Save the trained model and tokenizer."""
        try:
            logger.info("Saving model...")
            if self.trainer is None:
                raise ValueError("Trainer not initialized. Complete training first.")
            
            self.trainer.save_model(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            logger.info(f"Model saved successfully to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

def main():
    """Main function to run the training pipeline."""
    try:
        model_name = "/project/aaz/ashmitam/Qwen2/Qwen2Base"
        data_path = "Cleaned_CEO_Text.csv"
        output_dir = "./outputs"

        trainer = LanguageModelTrainer(model_name, data_path, output_dir)
        trainer.setup_model()
        trainer.load_data()
        trainer.train()
        trainer.save_model()
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()