# prepare.py
import torch
import bitsandbytes as bnb
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    AutoModel,  # Import AutoModel for BERTScore and BGE
    AutoModelForSequenceClassification #For BGE, though AutoModel is also fine
)
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import pandas as pd 
import os
from datetime import datetime
import evaluate  # Added for ROUGE evaluation
from rouge_score import rouge_scorer  # Import rouge_scorer
from tqdm import tqdm  # For progress bar
import pickle  # Import the pickle library
import numpy as np


print("-----starting----------")

# Set offline mode *before* any evaluate.load() calls.
os.environ["HF_EVALUATE_OFFLINE"] = "1"  # Force offline mode

timenow = datetime.now().strftime("%Y%m%d_%H%M%S")

os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_EVALUATE_OFFLINE"] = "1"  # Force offline mode

# Specify your model path or identifier
base_model_path  = "/project/aaz/leo/models/Llama-3.1-8B"
training_mode = False  # Set this to False if you only want to load and test the model
lora_model_path = "./results/lora_llama3.1_8b_raw_0605"  # Where fine-tuned model is saved
output_dir = "./results/embedding/"

# BERTScore setup (using BGE)
bertscore_model_name = "/project/aaz/leo/models/bge-large-en-v1.5"
bertscore_tokenizer = AutoTokenizer.from_pretrained(bertscore_model_name)
bertscore_model = AutoModel.from_pretrained(bertscore_model_name).to("cuda") # Move to GPU
print("----------loaded bert for evaluation--------------")

# Configure bitsandbytes quantizationn
quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,              # Enable 8-bit quantization
        llm_int8_threshold=6.0,         # Threshold for outlier detection
        llm_int8_has_fp16_weight=False, # Whether to use fp16 weights for int8 tensors
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )
# Initialize the rouge_scorer.  This doesn't require any downloads.
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
 
#  If CSV is available, add a sample from there
# Helper function to get embeddings and model output

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
    last_hidden_state = outputs_for_embedding.hidden_states[-1].detach()
    
    # Mean pooling across sequence length dimension
    embedding = last_hidden_state.mean(dim=1)
    
    # Decode the generated text
    generated_text = tokenizer.decode(generation_outputs[0], skip_special_tokens=True)
    
    return embedding, generated_text, last_hidden_state

# Test with some examples
test_examples = [
    "The CEO emphasized future market trends and strategic growth.",
    # "In the last quarterly earnings call, the executives discussed",
    # Add more test examples as needed
]

# INFERENCE SECTION - Always runs
# Load the fine-tuned model
print("Loading fine-tuned model...")
tokenizer, model = load_fine_tuned_model()
print("Model loaded successfully!")

# Test with some examples

try:
    df = pd.read_csv("./dataset/Cleaned_CEO_Text.csv")
    # df = df.head(3)
    # test_examples.append(df["conversation"][0] )  # Add truncated first conversation
except Exception as e:
    print(f"Could not load CSV sample: {e}")

print("loaded data")

generated_texts = []
all_results = []  # List to store results for all examples
all_embeddings = []

# Process each test example
# for i, example in enumerate(test_examples):
for index, row in df.iterrows():
    # print(f"\n\n***** Example {i+1} *****")
    # print(f"Input: {example}")
    transcript_id = row['transcript_id']
    example = row['conversation']
    print(f"\n\n***** Transcript ID: {transcript_id} *****")
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
     
    # --- ROUGE Evaluation ---
    scores = scorer.score(example, generated_text)  # target, prediction
    print("\nROUGE Scores:")
    rouge_scores_dict = {}
    
    for key, value in scores.items():
        print(f"  {key}:")
        print(f"    Precision: {value.precision:.4f}")
        print(f"    Recall:    {value.recall:.4f}")
        print(f"    F-measure: {value.fmeasure:.4f}")
        rouge_scores_dict[key] = {
                "precision": value.precision,
                "recall": value.recall,
                "fmeasure": value.fmeasure
            }

    generated_texts.append(generated_text)
    # Append embedding to the list
    all_embeddings.append(embedding.cpu())  # Keep as tensor for now
     # Save embedding and last_hidden_state as .npy files
    # embedding_filename = os.path.join(output_dir, f"embedding_{i+1}.npy")
    # last_hidden_filename = os.path.join(output_dir, f"last_hidden_{i+1}.npy")

    # np.save(embedding_filename, embedding.cpu().numpy())
    # np.save(last_hidden_filename, last_hidden.cpu().numpy())

    results = {
            "example_id": transcript_id,
            "input": example,
            "generated_text": generated_text, 
            "rouge_scores": rouge_scores_dict,
        }

    all_results.append(results)

# Combine all embeddings into a single NumPy array
combined_embeddings = torch.cat(all_embeddings, dim=0).numpy()

# Save the combined embeddings
embedding_filename = os.path.join(output_dir, "embeddings_all.npy")  # Single file
np.save(embedding_filename, combined_embeddings)

# Add the embedding filename to each result entry in the metadata
for result in all_results:
      result["embedding_file"] = embedding_filename


    # Save metadata (including filenames) to a single pickle file
metadata_file = os.path.join(output_dir, "metadata.pkl")  # Changed to .pkl
with open(metadata_file, "wb") as f:  # Open in binary write mode ("wb")
        pickle.dump(all_results, f) # Use pickle.dump

print(f"Metadata saved to: {metadata_file}")
print(f"Combined embeddings saved to: {embedding_filename}") 
    
def calculate_bertscore(predictions, references):
    """Calculates BERTScore using the specified model."""
    bertscore_results = []
    for pred, ref in tqdm(zip(predictions, references), total=len(predictions), desc="Calculating BERTScore"):
        # Tokenize and move to GPU
        encoded_predictions = bertscore_tokenizer(pred, padding=True, truncation=True, return_tensors='pt').to("cuda")
        encoded_references = bertscore_tokenizer(ref, padding=True, truncation=True, return_tensors='pt').to("cuda")

        with torch.no_grad():
            # Get embeddings
            predictions_embeddings = bertscore_model(**encoded_predictions).last_hidden_state
            references_embeddings = bertscore_model(**encoded_references).last_hidden_state

        # CLS POOLING
        predictions_embeddings = predictions_embeddings[:, 0, :]  # Use CLS token embedding
        references_embeddings = references_embeddings[:, 0, :]      # Use CLS token embedding

        # Calculate cosine similarity (BERTScore F1)
        cosine_similarity = torch.nn.functional.cosine_similarity(predictions_embeddings, references_embeddings, dim=1)
        
        # For each value in the batch, create a dictionary entry for the metrics
        for val in cosine_similarity.tolist():
            bertscore_results.append({'precision': val, 'recall': val, 'f1': val})
    
    return bertscore_results

# # --- BERTScore Evaluation ---
# print("\nCalculating BERTScore...")
# bert_scores = calculate_bertscore(generated_texts, test_examples)

# # Print average BERTScore results
# print("\nBERTScore Results (Average):")
# avg_precision = sum(score['precision'] for score in bert_scores) / len(bert_scores)
# avg_recall = sum(score['recall'] for score in bert_scores) / len(bert_scores)
# avg_f1 = sum(score['f1'] for score in bert_scores) / len(bert_scores)
# print(f"  Precision: {avg_precision:.4f}")  # Average precision
# print(f"  Recall: {avg_recall:.4f}")          # Average recall
# print(f"  F1: {avg_f1:.4f}")                  # Average F1

# # Print detailed BERTScore results per example
# print("\nBERTScore Results (Detailed):")
# for i, score in enumerate(bert_scores):
#     print(f"\nExample {i+1}:")
#     print(f"  Precision: {score['precision']:.4f}")
#     print(f"  Recall: {score['recall']:.4f}")
#     print(f"  F1: {score['f1']:.4f}")
