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
import sys
import re
import unicodedata
from tqdm import tqdm
import time

sys.path.append('/project/aaz/leo')
from util.read_parquet import read_parquet_data, get_first_rows, extract_speaker_text

# Pre-compile regex patterns for better performance
CLEANING_PATTERNS = [
    # Conference call introductions and greetings
    re.compile(r'^(good (day|morning|afternoon|evening)|hello|hi).*?conference call\.', re.IGNORECASE),
    re.compile(r'^welcome to.*?conference call\.', re.IGNORECASE),
    
    # Legal disclaimers
    re.compile(r'forward[- ]looking statements?.*?(?:\.\s|$)', re.IGNORECASE),
    re.compile(r'safe harbor.*?(?:\.\s|$)', re.IGNORECASE),
    
    # Operator instructions
    re.compile(r'press \*?\s?1.*?(?:\.\s|$)', re.IGNORECASE),
    re.compile(r'touchtone.*?(?:\.\s|$)', re.IGNORECASE),
    re.compile(r'mute function.*?(?:\.\s|$)', re.IGNORECASE),
    re.compile(r'we will pause.*?(?:\.\s|$)', re.IGNORECASE),
    
    # Brackets and placeholder text
    re.compile(r'\[.*?\]', re.IGNORECASE),
    
    # Filler words
    re.compile(r'\b(uh|um|you know|kind of|sort of)\b', re.IGNORECASE),
    
    # Call transitions
    re.compile(r'turn the call|hand the call|operator', re.IGNORECASE),
    re.compile(r'.*?I\'ll turn (the|this|it) over to.*?(?:\.\s|$)', re.IGNORECASE),
    re.compile(r'.*?I will turn (the|this|it) over to.*?(?:\.\s|$)', re.IGNORECASE),
    re.compile(r'thank you,? operator\.?', re.IGNORECASE),
    
    # Page numbers and headers
    re.compile(r'Page \d+ of \d+.*', re.IGNORECASE),
    re.compile(r'\d+Q\d{2}', re.IGNORECASE),
    
    # Self-introductions with titles
    re.compile(r'I am [^,]+, Vice President.*?(?:\.\s|$)', re.IGNORECASE),
]

WHITESPACE_PATTERN = re.compile(r'\s+')

class Extractor:
    def __init__(self, base_model_path, lora_model_path, training_mode=False):
        self.base_model_path = base_model_path
        self.lora_model_path = lora_model_path
        self.training_mode = training_mode
        self.tokenizer = None
        self.model = None
        self.timenow = datetime.now().strftime("%Y%m%d_%H%M%S") 
        self.dataset = None
        
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
        
    def load_fine_tuned_model(self):
        """Load the fine-tuned model with LoRA weights"""
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            trust_remote_code=True,
        )
        
        # Load the base model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            quantization_config=self.quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
        # Set pad token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        # Load and merge the LoRA adapter weights
        if self.lora_model_path:
            print(f"Loading LoRA weights from {self.lora_model_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                self.lora_model_path,
                torch_dtype=torch.float16,
            )
            
        # Set model to evaluation mode
        self.model.eval()
        
    def data_massage(self, data, full_transcript=True):
        """Process the dataframe: sort, filter, and add CEO tags""" 
        
        df = data
        # Sort by new_id and transcript_id in ascending order
        df = df.sort_values(by=['new_id', 'transcript_id'], ascending=True)
        
        # Filter out rows where context_id is 1
        if full_transcript == False:
            df = df[df['context_id'] != 1]
        
        print(f"Starting data cleaning for {len(df)} rows.....")
        
        # Clean the text data with progress indicator
        tqdm.pandas(desc="Cleaning text")
        #df['speaker_text'] = df['speaker_text'].progress_apply(self.clean_data)
        
        # Create the formatted dataset with the combined data
        print("Creating formatted transcript dataset...")
        formatted_dataset = self.create_structured_dataset(df)
        self.dataset = formatted_dataset  # Store the processed dataframe
        
        print(f"Extracting 100 samples for checking.....")
        # Extract sample of speaker text for checking
        extract_speaker_text(formatted_dataset.head(100), 100)
        
        return self.dataset
    
    def create_structured_dataset(self, df):
        """
        Create a structured dataset from the processed dataframe.
        Returns a dataframe with formatted dialogues that include transcript ID, session type,
        and speaker tags for CEOs and analysts in a single column.
        """
        print("Structuring dialogues by transcript and context...")
        
        # Group by transcript_id and context_id (for SESSION type)
        grouped_df = df.groupby(['transcript_id', 'context_id'])
        
        structured_conversations = []
        
        for (transcript_id, context_id), group in grouped_df:
            # Sort by new_id to maintain conversation order
            group = group.sort_values('new_id')
            
            # Determine session type
            session_type = "MANAGEMENT DISCUSSION" if context_id == 1 else "Q&A"
            
            # Start conversation with headers
            conversation = f"<TRANSCRIPT_ID={transcript_id}> <SESSION={session_type}>\n"
            
            # Process all speakers sequentially according to new_id
            for _, row in group.iterrows():
                # Add appropriate tags based on speaker type
                if row['is_ceo']:
                    # Add <CEO> </CEO> tags
                    text = self.split_long_text(f"<CEO>{row['speaker_text']}</CEO>", 1500)
                    conversation += f"{text}\n"
                else:
                    # For non-CEO speakers (analysts or others)
                    speaker_type = row.get('speaker_type', 'Unknown')
                    speaker_id = row.get('speaker_id', 'Unknown')
                    
                    # Add <ANALYST> tags or other appropriate tags
                    text = self.split_long_text(f"<ANALYST id={speaker_id}> {row['speaker_text']} </ANALYST>", 1500)
                    conversation += f"{text}\n"
            
            # End the conversation
            conversation += "<END>"
            structured_conversations.append(conversation)
        
        # Create a dataframe with the structured conversations
        result_df = pd.DataFrame({"transcript": structured_conversations})
         
        return result_df
    
    def split_long_text(self, text, max_length=1500):
        """
        Split text that exceeds max_length into multiple chunks while preserving tags.
        For CEO text, maintains <CEO> at start and </CEO> at end of each chunk.
        For ANALYST text, maintains <ANALYST id=X> at start and </ANALYST> at end of each chunk.
        """
        # Check if text needs splitting
        if len(text) <= max_length:
            return text
            
        result = ""
        
        # Extract tag information
        if "<CEO>" in text:
            opening_tag = "<CEO>"
            closing_tag = "</CEO>"
            content = text.replace(opening_tag, "").replace(closing_tag, "")
        elif "<ANALYST" in text:
            # Extract the full analyst tag with ID
            opening_tag_end = text.find(">") + 1
            opening_tag = text[:opening_tag_end]
            closing_tag = "</ANALYST>"
            content = text[opening_tag_end:text.rfind(closing_tag)]
        else:
            # If no recognized tags, just split the text
            opening_tag = ""
            closing_tag = ""
            content = text
            
        # Split content into chunks
        words = content.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for the space
            if current_length + word_length > max_length and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
                
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        # Reassemble with appropriate tags
        for i, chunk in enumerate(chunks):
            if i < len(chunks) - 1:
                # For all chunks except the last one
                result += f"{opening_tag}{chunk}{closing_tag}\n"
            else:
                # For the last chunk
                result += f"{opening_tag}{chunk}{closing_tag}"
                
        return result
    
    def clean_data(self, text: str) -> str:
        """
        Clean CEO transcript data by removing common noise patterns.
        """
        # Normalize Unicode characters
        text = unicodedata.normalize('NFKC', text)
        
        # Apply each pre-compiled pattern
        for pattern in CLEANING_PATTERNS:
            text = pattern.sub(' ', text)
        
        # Replace typographical variants
        text = text.replace('"', '"').replace('"', '"').replace('–', '-')
        
        # Clean up extra whitespace
        text = WHITESPACE_PATTERN.sub(' ', text).strip()
         
        return text
        
    def run(self, dataframe, text_column='transcript', max_texts=None):
        """
        Process texts from a dataframe and extract embeddings
        
        Args:
            dataframe (pd.DataFrame): DataFrame containing text data
            text_column (str): Column name containing text to extract embeddings from
            max_texts (int, optional): Maximum number of texts to process
            
        Returns:
            pd.DataFrame: DataFrame with original data and added embeddings
        """
        if self.model is None:
            self.load_fine_tuned_model()
            print("Model loaded successfully!")
        
        # Create a copy of the dataframe to avoid modifying the original
        result_df = dataframe.copy()
        
        # Add columns for embeddings and generated text
        result_df['embedding'] = None
        # result_df['generated_text'] = None
        
        # Limit the number of texts if specified
        if max_texts and max_texts < len(result_df):
            process_df = result_df.head(max_texts)
        else:
            process_df = result_df
        
        # Process each text and extract embeddings
        for idx, row in tqdm(process_df.iterrows(), total=len(process_df), desc="Extracting embeddings"):
            if text_column in row and pd.notna(row[text_column]):
                text = str(row[text_column])
                try:
                    embedding, generated_text, _ = self.get_embedding_and_text(text)
                    
                    # Convert embedding tensor to list and store in dataframe
                    result_df.at[idx, 'embedding'] = embedding.cpu().numpy().tolist()
                    # result_df.at[idx, 'generated_text'] = generated_text
                except Exception as e:
                    print(f"Error processing text at index {idx}: {e}")
        
        print(f"Successfully processed {len(process_df)} texts") 
        
        # Check if there are any valid embeddings to display shape information
        if len(result_df) > 0 and 'embedding' in result_df.columns:
            # Find first non-None embedding
            for idx, row in result_df.iterrows():
                if row['embedding'] is not None:
                    embedding_array = row['embedding']
                    print(f"\nEmbedding shape: {torch.tensor(embedding_array).shape}")
                    break
            else:
                print("No valid embeddings found to display shape information")
        else:
            print("No valid embeddings found to display shape information")
            
        return result_df
    
    def get_embedding_and_text(self, text):
        """Given an input text, return both the embedding and generated text"""
        # Prepare input for the model with proper padding and truncation
        # Use a larger max_length for tokenization to avoid issues when input is exactly at the limit
        inputs = self.tokenizer(
            text, 
            return_tensors="pt",
            # padding="max_length",
            # truncation=True,
            # max_length=1500  # Input tokenization length
        ).to(self.model.device)
        
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
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Get the last hidden state
        last_hidden_state = outputs_for_embedding.hidden_states[-1]
        
        # Mean pooling across sequence length dimension
        embedding = last_hidden_state.mean(dim=1)
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(generation_outputs[0], skip_special_tokens=True)
        
        return embedding, generated_text, last_hidden_state 
            
    def save_embeddings(self, df, output_path=None):
        """
        Save the dataframe with embeddings to a file
        
        Args:
            df (pd.DataFrame): DataFrame containing embeddings
            output_path (str, optional): Path to save the embeddings
        """
        if output_path is None:
            output_path = f"./embeddings/embeddings_{self.timenow}.parquet"
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
        # Save to parquet file
        df.to_parquet(output_path)
        print(f"Embeddings saved to {output_path}")
        
        # Also save a CSV version without the embedding column (which might be too large)
        csv_df = df.drop(columns=['embedding'])
        csv_path = output_path.replace('.parquet', '.csv')
        csv_df.to_csv(csv_path, index=False)
        print(f"Generated text saved to {csv_path}")


if __name__ == "__main__":
    start_time = time.time()
    print(f"Process started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Specify model paths
    base_model_path = "/project/aaz/leo/models/Llama-3.1-8B"
    lora_model_path = "./results/lora_llama3.1_8b_0645"
    parquet_folder = "/project/aaz/leo/dataset/ceo_scripts_export_20250405_193052"
    
    # Create the extractor
    extractor = Extractor(
        base_model_path=base_model_path,
        lora_model_path=lora_model_path,
    )
    print(f"Successfully initiated model extractor.")
    
    # Read parquet data
    # print(f"Reading parquet data from {parquet_folder}...")
    # df, total_rows = read_parquet_data(parquet_folder)
    # print(f"Successfully loaded parquet data with {total_rows} total rows")
    
    df = pd.read_parquet('/project/aaz/leo/dataset/ceo_scripts_export_20250405_193052/page_1.parquet')
    # # Print confirmation
    print(f"Successfully loaded {len(df)} rows of dataset with a total of rows")
    
    # Wrap the dataframe in a list since get_first_rows expects a list of dataframes 
    df = get_first_rows(df, 10000) 
    # Process the data to extract embeddings
    print(f"Extracting embeddings from {len(df)} texts...")
    df = extractor.data_massage(data=df, full_transcript=False)
    print(f"Successfully data processed {len(df)} rows of dataset for extraction")
    result_df = extractor.run(df, text_column='transcript')
    print(f"Successfully data processed {len(df)} rows of dataset for extraction")

    # Save the embeddings
    extractor.save_embeddings(result_df)
    
    # Print execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Process completed in {execution_time:.2f} seconds")