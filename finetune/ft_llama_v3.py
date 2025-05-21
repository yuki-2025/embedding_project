import torch
import bitsandbytes as bnb
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import pandas as pd 
import os
from datetime import datetime

class LlamaFineTuner:
    def __init__(self, base_model_path, lora_model_path, training_mode=False):
        self.base_model_path = base_model_path
        self.lora_model_path = lora_model_path
        self.training_mode = training_mode
        self.tokenizer = None
        self.model = None
        self.timenow = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Disable wandb
        os.environ["WANDB_DISABLED"] = "true"
        
        # Configure bitsandbytes quantization
        self.quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,              # Enable 8-bit quantization
            llm_int8_threshold=6.0,         # Threshold for outlier detection
            llm_int8_has_fp16_weight=False, # Whether to use fp16 weights for int8 tensors
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )

    def train_model(self):
        """Set up and train the model"""
        # Load tokenizer and model with bitsandbytes (8-bit) support
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            quantization_config=self.quantization_config,
            device_map="auto",
            output_hidden_states=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        # --- LoRA Configuration ---
        lora_config = LoraConfig(
            r=8,                     # Low-rank dimension.
            lora_alpha=32,           # Scaling factor.
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.1,        # Dropout to mitigate overfitting.
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # Wrap the model with LoRA adapters
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # --- Data Preparation ---
        df = pd.read_csv("./dataset/Cleaned_CEO_Text.csv")
        df = df[["conversation"]]

        def tokenize_function(examples):
            return self.tokenizer(examples["conversation"], truncation=True, max_length=2048)

        dataset = Dataset.from_pandas(df)
        tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        # --- Training Arguments ---
        training_args = TrainingArguments(
            output_dir=f"./results/train_ck_{self.timenow}",
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            save_steps=500,
            save_total_limit=2,
            logging_steps=50,
            fp16=True,
            report_to=None,
        )
        
        # Free up memory
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        # Set up the Trainer for causal language modeling
        trainer = Trainer(
            model=self.model,
            train_dataset=tokenized_datasets,
            args=training_args,
            data_collator=data_collator,
        )

        # Start fine-tuning
        trainer.train()
        
        # Save the final checkpoint
        self.model.save_pretrained(self.lora_model_path)
        self.tokenizer.save_pretrained(self.lora_model_path)

    def load_fine_tuned_model(self):
        """Load the fine-tuned model for inference"""
        # Load base model with quantization
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            quantization_config=self.quantization_config,
            device_map="auto",
            output_hidden_states=True,
        )
        
        # Set pad token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load the LoRA weights on top of the base model
        self.model = PeftModel.from_pretrained(self.model, self.lora_model_path)
        
        return self.tokenizer, self.model

    def get_embedding_and_text(self, text):
        """Given an input text, return both the embedding and generated text"""
        # Prepare input for the model
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        # Generate text
        with torch.no_grad():
            # For hidden states and embeddings
            outputs_for_embedding = self.model(**inputs, output_hidden_states=True)
            
            # For text generation
            generation_outputs = self.model.generate(
                inputs.input_ids,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        
        # Get the last hidden state
        last_hidden_state = outputs_for_embedding.hidden_states[-1]
        
        # Mean pooling across sequence length dimension
        embedding = last_hidden_state.mean(dim=1)
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(generation_outputs[0], skip_special_tokens=True)
        
        return embedding, generated_text, last_hidden_state
    
    def run_inference(self):
        """Run inference with the loaded model"""
        print("Loading fine-tuned model...")
        self.load_fine_tuned_model()
        print("Model loaded successfully!")

        # Test examples
        test_examples = [
            "The CEO emphasized future market trends and strategic growth.",
        ]

        # If CSV is available, add a sample from there
        try:
            df = pd.read_csv("./dataset/Cleaned_CEO_Text.csv")
            test_examples.append(df["conversation"][0])
        except Exception as e:
            print(f"Could not load CSV sample: {e}")

        # Process each test example
        for i, example in enumerate(test_examples):
            print(f"\n\n***** Example {i+1} *****")
            print(f"Input: {example}")
            print("-----------------------Processing...----------------------------------")
            
            # Get embedding and generated text
            embedding, generated_text, last_hidden = self.get_embedding_and_text(example)
            
            # Print results
            print(f"\nEmbedding shape: {embedding.shape}")
            print(f"Last hidden state shape: {last_hidden.shape}")
            print(f"\nGenerated text:\n{generated_text}")
            
            # Print some values from the embedding for verification
            print(f"\nSample embedding values (first 5): {embedding[0, :5].tolist()}")

        print("\nInference complete!")
    
    def run(self):
        """Main method to run either training or inference"""
        if self.training_mode:
            print("Starting training mode...")
            self.train_model()
        
        # Always run inference
        self.run_inference()


if __name__ == "__main__":
    # Specify model paths
    base_model_path = "/project/aaz/leo/models/Llama-3.1-8B"
    lora_model_path = "./results/lora_llama3.1_8b_raw_0615"
    
    # Create and run the fine-tuner
    fine_tuner = LlamaFineTuner(
        base_model_path=base_model_path,
        lora_model_path=lora_model_path,
        training_mode=True  # Set to True to enable training
    )
    
    fine_tuner.run()  
    
    # Example usage:
    # Call the function to read parquet data
    # folder_path = "dataset/ceo_scripts_export_20250405_193052"  # Replace with the actual path to your folder
    # df, total_rows = read_parquet_data(folder_path) 
    
    # print(f"Successfully loaded {len(df)} dataframes with a total of {total_rows} rows")
    
    # dataset = get_first_rows(df, 10000)
    # print(f"Successfully loaded {len(dataset)} rows od dataset with a total of {total_rows} rows")