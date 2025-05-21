import os
import json
import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import logging
import wandb
import evaluate 
import re
from nltk.tokenize import sent_tokenize

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration - changed from class to dictionary
CONFIG = {
    "MODEL_NAME": "/project/aaz/leo/models/Llama-3.1-8B",  # or your preferred LLaMA variant
    "OUTPUT_DIR": "./result/",
    "DATA_PATH": "./dataset/Cleaned_CEO_Text.csv",
    "NUM_EPOCHS": 3,
    "BATCH_SIZE": 8,
    "GRADIENT_ACCUMULATION_STEPS": 4,
    "LEARNING_RATE": 3e-4,
    "WARMUP_STEPS": 100,
    "MAX_SEQ_LENGTH": 512,
    "EVAL_STEPS": 200,
    "SAVE_STEPS": 500,
    "FP16": True,
    "SEED": 42,
    
    # LoRA specific config
    "LORA_R": 16,
    "LORA_ALPHA": 32,
    "LORA_DROPOUT": 0.05,
    "TARGET_MODULES": ["q_proj", "v_proj", "k_proj", "o_proj"]
}

# Set seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
set_seed(CONFIG["SEED"])

# Data preparation
class ConferenceCallDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format the input text with a clear instruction and structure
        formatted_text = f"""
        [INST] Extract important CEO insights from the following conference call excerpt. 
        
        Company: {item['metadata']['company']}
        Quarter: {item['metadata']['quarter']}
        Fiscal Year: {item['metadata']['fiscal_year']}
        
        Context: {item['context']}
        
        CEO Statement: {item['statement']} [/INST]
        
        The CEO is highlighting the following key insights:
        1. {item.get('insights', {}).get('insight1', '')}
        2. {item.get('insights', {}).get('insight2', '')}
        3. {item.get('insights', {}).get('insight3', '')}
        
        The strategic focus areas mentioned are: {item.get('insights', {}).get('strategic_areas', '')}
        
        Financial implications: {item.get('insights', {}).get('financial_implications', '')}
        """
        
        # Tokenize inputs
        encodings = self.tokenizer(
            formatted_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Create label tensors (for causal LM, labels are the same as input_ids)
        encodings["labels"] = encodings["input_ids"].clone()
        
        # Process to return tensors without the batch dimension that the tokenizer adds
        return {key: val.squeeze(0) for key, val in encodings.items()}

def load_and_prepare_data(data_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load conference call data from CSV and split into train/eval sets"""
    
    logger.info(f"Loading data from CSV file: {data_path}")
    df = pd.read_csv(data_path)
    
    logger.info(f"CSV loaded with {len(df)} rows and columns: {', '.join(df.columns)}")
    
    # Initialize cleaned_df as an empty DataFrame
    cleaned_df = pd.DataFrame()
    
   # Process each transcript
    for _, row in df.iterrows():
        conversation = row['conversation']
        
        # Extract Q&A pairs
        qa_pairs = re.split(r'Question :|Answer :', conversation)
        qa_pairs = [p.strip() for p in qa_pairs if p.strip()] #remove empty strings
        
        # Process pairs and deduplicate
        processed_pairs = []
        for i in range(0, len(qa_pairs)-1, 2):
            if i+1 < len(qa_pairs): #end loop
                q = qa_pairs[i]
                a = qa_pairs[i+1]
                
                # # Remove duplicated content
                # a = remove_duplicates(a)
                # q = remove_duplicates(q)
                
                # # Extract topic using NLP
                # topic = extract_topic(q)
                
                # Add to processed pairs if meaningful exchange
                if len(q.split()) > 1 and len(a.split()) > 1: #minimum length
                    processed_pairs.append({
                        'Transscript_id': f"{row['transcript_id']}_{i//2}",
                        'Company_id': row['company_id'],
                        'Date': row['processed_fs'],
                        'Question': q,
                        'Answer': a,
                        # 'topic': topic
                    })
        
        # Add to cleaned dataframe
        if processed_pairs: 
            cleaned_df = pd.concat([cleaned_df, pd.DataFrame(processed_pairs)], ignore_index=True)
    
    logger.info(f"Processed {len(cleaned_df)} valid examples from CSV")
    
    # Split into train and eval
    train_data, eval_data = train_test_split(
        cleaned_df, test_size=0.1, random_state=CONFIG["SEED"]
    )
    
    logger.info(f"Split into {len(train_data)} train and {len(eval_data)} eval examples")
    return train_data, eval_data

def prepare_model_and_tokenizer():
    """Load and prepare the LLaMA model and tokenizer with LoRA configuration"""
    
    logger.info(f"Loading base model: {CONFIG['MODEL_NAME']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["MODEL_NAME"],return_tensors = "pt")
                                              #padding = True,truncation = True,max_length = 10)
    bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
    # self.model = AutoModelForCausalLM.from_pretrained(self.model_id,device_map = "cuda:0",quantization_config = self.bnb)
    model = AutoModelForCausalLM.from_pretrained(
                    CONFIG['MODEL_NAME'],
                    device_map="auto",  # Let the library decide optimal placement
                    quantization_config=bnb,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True
                    # offload_folder="offload",
                )
    print("model loaded")
    
    # Ensure the tokenizer has pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure LoRA
    logger.info("Configuring LoRA adapters")
    lora_config = LoraConfig(
        r=CONFIG["LORA_R"],
        lora_alpha=CONFIG["LORA_ALPHA"],
        lora_dropout=CONFIG["LORA_DROPOUT"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=CONFIG["TARGET_MODULES"],
    )
    
    # Prepare model for LoRA training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    logger.info(f"Trainable parameters: {model.print_trainable_parameters()}")
    
    return model, tokenizer

def compute_metrics(eval_preds, tokenizer):
    """Compute evaluation metrics for the model"""
    predictions, labels = eval_preds
    
    # Remove padding from predictions and labels
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Decode predictions and references
    pred_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Calculate metrics - can use ROUGE, BLEU, or custom metrics
    rouge = evaluate.load("rouge")
    rouge_output = rouge.compute(predictions=pred_str, references=label_str)
    
    return {
        "rouge1": rouge_output["rouge1"],
        "rouge2": rouge_output["rouge2"],
        "rougeL": rouge_output["rougeL"]
    }

def extract_embeddings(model, tokenizer, text, pooling="mean"):
    """Extract embeddings from the model for a given text"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Extract the hidden states from the last layer
    last_hidden_states = outputs.hidden_states[-1]
    
    if pooling == "mean":
        # Mean pooling - take average of all token embeddings
        attention_mask = inputs['attention_mask']
        embedding = torch.sum(last_hidden_states * attention_mask.unsqueeze(-1), dim=1) / torch.sum(attention_mask, dim=1, keepdim=True)
    elif pooling == "cls":
        # CLS pooling - use the embedding of the first token
        embedding = last_hidden_states[:, 0]
    
    return embedding.cpu().numpy()

def main():
    # Initialize wandb for experiment tracking - fixed to use the dictionary
    #wandb.init(project="llama-ceo-insights", config=CONFIG)
    
    # Load and prepare data
    train_data, eval_data = load_and_prepare_data(CONFIG["DATA_PATH"])
    
    # Prepare model and tokenizer
    model, tokenizer = prepare_model_and_tokenizer()
    
    # Create datasets
    train_dataset = ConferenceCallDataset(train_data, tokenizer, CONFIG["MAX_SEQ_LENGTH"])
    eval_dataset = ConferenceCallDataset(eval_data, tokenizer, CONFIG["MAX_SEQ_LENGTH"])
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=CONFIG["OUTPUT_DIR"],
        num_train_epochs=CONFIG["NUM_EPOCHS"],
        per_device_train_batch_size=CONFIG["BATCH_SIZE"],
        per_device_eval_batch_size=CONFIG["BATCH_SIZE"],
        gradient_accumulation_steps=CONFIG["GRADIENT_ACCUMULATION_STEPS"],
        learning_rate=CONFIG["LEARNING_RATE"],
        warmup_steps=CONFIG["WARMUP_STEPS"],
        evaluation_strategy="steps",
        eval_steps=CONFIG["EVAL_STEPS"],
        save_strategy="steps",
        save_steps=CONFIG["SAVE_STEPS"],
        save_total_limit=3,
        fp16=CONFIG["FP16"],
        logging_steps=50,
        report_to="wandb",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=CONFIG["SEED"],
        remove_unused_columns=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        ##compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer),
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm=False  # Not using masked language modeling
        ),
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    model.save_pretrained(os.path.join(CONFIG["OUTPUT_DIR"], "final_model"))
    tokenizer.save_pretrained(os.path.join(CONFIG["OUTPUT_DIR"], "final_model"))
    
    # Example of using the model to extract embeddings
    logger.info("Generating sample embeddings...")
    sample_text = """
    Our Q3 results highlight the success of our digital transformation initiatives. 
    We've seen a 22% increase in user engagement on our platform, which has directly 
    translated to a 15% revenue growth year-over-year. Looking ahead, we're 
    investing heavily in AI-driven solutions that we believe will revolutionize 
    our product offerings in the coming quarters.
    """
    
    embedding = extract_embeddings(model, tokenizer, sample_text)
    logger.info(f"Generated embedding shape: {embedding.shape}")
    
    wandb.finish()
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
