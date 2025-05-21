import torch
import os
import numpy as np
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel 
from warnings import filterwarnings
from torch.utils.data import DataLoader
import pandas as pd

filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false" 
 
# Option B: Load parquet files directly
train_path = "/project/aaz/leo/dataset/imdb/plain_text/train-00000-of-00001.parquet"
test_path = "/project/aaz/leo/dataset/imdb/plain_text/test-00000-of-00001.parquet"

# Load parquet files using pandas and convert to Dataset
train_df = pd.read_parquet(train_path)
print("train_df loaded")
print(train_df.head())
test_df = pd.read_parquet(test_path)
print("test_df loaded")
    
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
    
    # Create DatasetDict 
dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })

# First, inspect your dataset to understand its structure
print("Dataset keys:", dataset.keys())  # This shows the splits like 'train', 'test'
print("Train features:", dataset["train"].features)  # This shows column names
print("First example:", dataset["train"][0])  # See what a sample looks like
print(dataset)
model_name = "models/bge-large-en-v1.5"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

print(tokenizer)
# calculate the model parameters
params = sum(p.numel() for p in model.parameters())
print(f"The LLM has {params} parameters") # Question: how much memory does this model require? 

# move the model to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


#--------------------------
# Define batch size
BATCH_SIZE = 256

# This is the most memory-efficient way to tokenize a large dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        truncation=True, 
        max_length=512, 
        padding="max_length",
        return_tensors=None  # Don't return tensors here
    )

# Apply tokenization to the entire dataset
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=BATCH_SIZE,
    desc="Tokenizing dataset"
)

# Now you can use the tokenized datasets
train_dataset = tokenized_datasets["train"]
print("---------------------------train_encodings--------------------------",train_dataset)
test_dataset = tokenized_datasets["test"]
print("---------------------------train_encodings--------------------------",test_dataset)
# Create DataLoader for batching

# Create DataLoaders
from torch.utils.data import DataLoader

# Function to format batches for the model
def collate_fn(batch):
    # Process input_ids, attention_mask, etc.
    input_ids = torch.tensor([item["input_ids"] for item in batch])
    attention_mask = torch.tensor([item["attention_mask"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch])
    return {
        "input_ids": input_ids, 
        "attention_mask": attention_mask, 
        "labels": labels
    }

train_dataloader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    collate_fn=collate_fn
)

test_dataloader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    collate_fn=collate_fn
)

# Extract CLS embeddings for training encodings in batches
train_cls_embeddings = []
train_labels = []

print("Extracting CLS embeddings for train data...")

model.eval()  # Set model to evaluation mode
for batch in tqdm(train_dataloader):
    # Correctly access the dictionary keys from the batch
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"]  # Keep labels on CPU for now torch.Size([256]) tensor([1, 1, 1, 1, 1, 1, 0, 

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state # Shape: (batch_size, seq_length, hidden_size) torch.Size([256, 512, 1024])

        # Extract CLS token (first token) embeddings from each sequence in the batch 
        cls_embeddings = last_hidden_states[:, 0, :]  # Shape: (batch_size, hidden_size) torch.Size([256, 1024])
        train_cls_embeddings.extend(cls_embeddings.cpu().numpy()) # Move to CPU and convert to NumPy
        train_labels.extend(labels.numpy())

# Convert to NumPy arrays for easier downstream processing
train_cls_embeddings = np.array(train_cls_embeddings)
train_labels = np.array(train_labels)

print(f"Extracted {len(train_cls_embeddings)} CLS embeddings for train data.")
print(f"The shape of the extracted embeddings is: {train_cls_embeddings.shape}")
print(f"The shape of the labels is: {train_labels.shape}")