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
timenow = datetime.now().strftime("%Y%m%d_%H%M%S")

os.environ["WANDB_DISABLED"] = "true"

# Specify your model path or identifier
base_model_path  = "/project/aaz/leo/models/Llama-3.1-8B"
training_mode = False  # Set this to False if you only want to load and test the model

# Configure bitsandbytes quantization
quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,              # Enable 8-bit quantization
        llm_int8_threshold=6.0,         # Threshold for outlier detection
        llm_int8_has_fp16_weight=False, # Whether to use fp16 weights for int8 tensors
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

if training_mode:

    # Load tokenizer and model with bitsandbytes (8-bit) support.
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=quantization_config,  # Use the bitsandbytes quantization config
        device_map="auto",
        # Ensure hidden states are output for embedding extraction.
        output_hidden_states=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # --- LoRA Configuration ---
    lora_config = LoraConfig(
        r=8,                     # Low-rank dimension.
        lora_alpha=32,           # Scaling factor.
        target_modules= ["q_proj", "v_proj", "k_proj", "o_proj"],  # Typically, attention projection layers.
        lora_dropout=0.1,        # Dropout to mitigate overfitting.
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Wrap the model with LoRA adapters.
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Data Preparation ---
    # Assume a CSV file with a "text" column containing conference call transcripts.
    #dataset = load_dataset("csv", data_files={"train": "./dataset/Cleaned_CEO_Text.csv"})

    df = pd.read_csv("./dataset/Cleaned_CEO_Text.csv")
    df = df[["conversation"]]

    def tokenize_function(examples):
        return tokenizer(examples["conversation"], truncation=True, max_length=2048)

    dataset = Dataset.from_pandas(df) # creates HF Dataset from a pandas dataframe


    # tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_datasets = dataset.map(tokenize_function, batched=True,remove_columns=dataset.column_names)

    #print(df["conversation"][0])
    #print(tokenized_datasets[0])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # --- Training Arguments ---
    training_args = TrainingArguments(
        output_dir="./results/train_ck_"+timenow,  # Directory for saving checkpoints.
        num_train_epochs=3,                      # Adjust epochs as needed.
        per_device_train_batch_size=2,           # Adjust batch size according to your GPU memory.
        gradient_accumulation_steps=4, # Accumulate gradients over 4 steps (effective batch size is 4)
        learning_rate=2e-4,
        save_steps=500,                          # Save intermediate checkpoints.
        save_total_limit=2,
        logging_steps=50,
        fp16=True,                               # Mixed precision training.
        report_to=None,
    )
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Set up the Trainer for causal language modeling.
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_datasets,
        args=training_args,
        data_collator=data_collator,
    )

    # Start fine-tuning.
    trainer.train() 
    # Save the final checkpoint.
    model.save_pretrained("./results/lora_llama3.1_8b_raw_0605")
    tokenizer.save_pretrained("./results/lora_llama3.1_8b_raw_0605")

lora_model_path = "./results/lora_llama3.1_8b_raw_0605"  # Where fine-tuned model is saved

# --- Embedding Extraction Helper Function ---
# Helper function to load the fine-tuned model
def load_fine_tuned_model():
    # Load base model with quantization
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=quantization_config,
        device_map="auto",
        output_hidden_states=True,
    )
    
    # Set pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load the LoRA weights on top of the base model
    model = PeftModel.from_pretrained(model, lora_model_path)
    
    # Merge LoRA weights with base model for better inference (optional)
    # model = model.merge_and_unload()  # Uncomment if you want to merge weights
    
    return tokenizer, model

# Helper function to get embeddings and model output
def get_embedding_and_text(model, tokenizer, text):
    """
    Given an input text, return both the embedding and generated text
    """
    # Prepare input for the model
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Generate text
    with torch.no_grad():
        # For hidden states and embeddings
        outputs_for_embedding = model(**inputs, output_hidden_states=True)
        
        # For text generation
        generation_outputs = model.generate(
            inputs.input_ids,
            #max_length=200,  # Adjust as needed
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    # Get the last hidden state
    last_hidden_state = outputs_for_embedding.hidden_states[-1]
    
    # Mean pooling across sequence length dimension
    embedding = last_hidden_state.mean(dim=1)
    
    # Decode the generated text
    generated_text = tokenizer.decode(generation_outputs[0], skip_special_tokens=True)
    
    return embedding, generated_text, last_hidden_state

# INFERENCE SECTION - Always runs
# Load the fine-tuned model
print("Loading fine-tuned model...")
tokenizer, model = load_fine_tuned_model()
print("Model loaded successfully!")

# Test with some examples
test_examples = [
    "The CEO emphasized future market trends and strategic growth.",
    # "In the last quarterly earnings call, the executives discussed",
    # Add more test examples as needed
]

# If CSV is available, add a sample from there
try:
    df = pd.read_csv("./dataset/Cleaned_CEO_Text.csv")
    test_examples.append(df["conversation"][0] )  # Add truncated first conversation
except Exception as e:
    print(f"Could not load CSV sample: {e}")

# Process each test example
for i, example in enumerate(test_examples):
    print(f"\n\n***** Example {i+1} *****")
    print(f"Input: {example}")
    print("-----------------------Processing...----------------------------------")
    
    # Get embedding and generated text
    embedding, generated_text, last_hidden = get_embedding_and_text(model, tokenizer, example)
    
    # Print results
    print(f"\nEmbedding shape: {embedding.shape}")
    print(f"Last hidden state shape: {last_hidden.shape}")
    print(f"\nGenerated text:\n{generated_text}")
    
    # Print some values from the embedding for verification
    print(f"\nSample embedding values (first 5): {embedding[0, :5].tolist()}")

print("\nInference complete!")