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

class LlamaFineTunerWithAttention:
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

    def build_attention_mask(self, token_ids, tag_ids):
        """
        Build an attention mask that gives higher weight to CEO statements.
        
        Args:
            token_ids (torch.Tensor): The input token IDs
            tag_ids (dict): A dictionary with the start and end indices of CEO statements
            
        Returns:
            torch.Tensor: The attention mask with weights
        """
        mask = torch.ones_like(token_ids)
        mask[tag_ids['CEO_start']:tag_ids['CEO_end']] = 2  # CEO statements get double weight
        return mask

    def find_ceo_tag_positions(self, text):
        """
        Find positions of CEO tags in the text to create tag indices for attention masking.
        
        Args:
            text (str): The input text with CEO tags
            
        Returns:
            dict: A dictionary with start and end indices for CEO statements
        """
        tag_ids = {'CEO_start': [], 'CEO_end': []}
        
        # Process the input text to find CEO tag positions
        ceo_start_pattern = re.compile(r'<CEO>')
        ceo_end_pattern = re.compile(r'</CEO>')
        
        # Find all CEO tag positions
        for match in ceo_start_pattern.finditer(text):
            tag_ids['CEO_start'].append(match.start())
        
        for match in ceo_end_pattern.finditer(text):
            tag_ids['CEO_end'].append(match.end())
            
        return tag_ids

    def tokenize_with_attention(self, text):
        """
        Tokenize text and create attention masks based on CEO tags.
        
        Args:
            text (str): Input text with CEO tags
            
        Returns:
            dict: Tokenized inputs with attention mask
        """
        # Find CEO tag positions
        tag_ids = self.find_ceo_tag_positions(text)
        
        # Tokenize the text
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        )
        
        # Add attention mask with CEO weighting
        inputs['attention_mask'] = self.build_attention_mask(
            inputs['input_ids'],
            tag_ids
        )
        
        return inputs

    def train_model(self):
        """Set up and train the model with attention masking for CEO statements"""
        train_start_time = time.time()
        print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load tokenizer and model with bitsandbytes (8-bit) support
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            quantization_config=self.quantization_config,
            device_map="auto",
            output_hidden_states=True,
            use_cache=False,
            # attn_implementation="flash_attention_2",  # Enable Flash Attention 2
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
        df = self.dataset[["speaker_text"]]

        def tokenize_function(examples):
            """Tokenize function that adds attention masking"""
            result = {}
            for i, text in enumerate(examples["speaker_text"]):
                # Find CEO tag positions in the text
                tag_positions = self.extract_tag_positions(text)
                
                # Tokenize the text
                tokens = self.tokenizer(
                    text, 
                    padding="max_length",
                    truncation=True, 
                    max_length=2048,
                    return_tensors="pt"
                )
                
                # Create attention mask for CEO statements
                attention_mask = self.create_ceo_attention_mask(tokens.input_ids[0], tag_positions)
                
                # Add to result dictionary
                if i == 0:
                    result["input_ids"] = tokens.input_ids
                    result["attention_mask"] = attention_mask.unsqueeze(0)
                else:
                    result["input_ids"] = torch.cat([result["input_ids"], tokens.input_ids])
                    result["attention_mask"] = torch.cat([result["attention_mask"], attention_mask.unsqueeze(0)])
            
            return result

        def extract_tag_positions(self, text):
            """Extract CEO tag positions from text"""
            positions = {"CEO_start": [], "CEO_end": []}
            
            # Find CEO tag positions - simplified version for example
            start_tag = "<CEO>"
            end_tag = "</CEO>"
            
            current_pos = 0
            while True:
                start_pos = text.find(start_tag, current_pos)
                if start_pos == -1:
                    break
                    
                end_pos = text.find(end_tag, start_pos)
                if end_pos == -1:
                    break
                
                positions["CEO_start"].append(start_pos)
                positions["CEO_end"].append(end_pos + len(end_tag))
                current_pos = end_pos + len(end_tag)
                
            return positions
            
        def create_ceo_attention_mask(self, token_ids, tag_positions):
            """Create attention mask that emphasizes CEO statements"""
            mask = torch.ones_like(token_ids)
            
            # Map character positions to token positions
            for start, end in zip(tag_positions["CEO_start"], tag_positions["CEO_end"]):
                # This is a simplified approach - in a real implementation,
                # you would need to map character positions to token positions
                start_token = self.tokenizer.encode_plus(text[:start], add_special_tokens=False).input_ids.size
                end_token = self.tokenizer.encode_plus(text[:end], add_special_tokens=False).input_ids.size
                
                # Set higher weights for CEO statements
                mask[start_token:end_token] = 2
                
            return mask

        dataset = Dataset.from_pandas(df)
        # Map the tokenization function to the dataset
        tokenized_datasets = dataset.map(
            tokenize_function, 
            batched=True, 
            remove_columns=dataset.column_names,
            desc="Tokenizing text data with attention masking"
        )

        # Use DataCollatorForLanguageModeling to handle padding and attention masks
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, 
            mlm=False
        )

        # --- Training Arguments ---
        training_args = TrainingArguments(
            output_dir=f"./results/train_att_ck_{self.timenow}",
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
        
        train_end_time = time.time()
        training_duration = train_end_time - train_start_time
        print(f"Training finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total training time: {training_duration:.2f} seconds ({training_duration / 60:.2f} minutes)")

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

        # If CSV is available, add a sample from there
        try:
            df = pd.read_csv("./dataset/Cleaned_CEO_Text.csv") 
        except Exception as e:
            print(f"Could not load CSV sample: {e}")

        # Process each test example
        for i, example in enumerate(df):
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
            print("Starting training mode with attention masking...")
            self.train_model()
        
        # Always run inference
        self.run_inference()
        
    
    def data_massage(self, data, full_transcript):
        """Process the dataframe: sort, filter, and add CEO tags""" 
        
        df = data
        # Concatenate all dataframes if read_parquet_data returns multiple
        # if isinstance(data, list):
        #     df = pd.concat(data, ignore_index=True)
        
        # Sort by new_id and transcript_id in ascending order
        df = df.sort_values(by=['new_id', 'transcript_id'], ascending=True)
        
        # Filter out rows where context_id is 1
        if full_transcript == False:
            df = df[df['context_id'] != 1]
        
        print(f"Starting data cleaning for {len(df)} rows.....")
        
        # Clean the text data with progress indicator
        tqdm.pandas(desc="Cleaning text")
        df['speaker_text'] = df['speaker_text'].progress_apply(self.clean_data)
        
        # Note: CEO tag application moved to create_structured_dataset function
        
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
        
        Example format in the dataframe:
        <TRANSCRIPT_ID=4503> <SESSION=Q&A>
        <CEO> ...CEO sentence1... </CEO>
        <ANALYST id=123> ...Analyst sentence... </ANALYST>
        <CEO> ...CEO sentence2... </CEO>
        <END>
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
        result_df = pd.DataFrame({"speaker_text": structured_conversations})
         
        return result_df
    
    def split_long_text(self, text, max_length=1500):
        """
        Split text that exceeds max_length into multiple chunks while preserving tags.
        For CEO text, maintains <CEO> at start and </CEO> at end of each chunk.
        For ANALYST text, maintains <ANALYST id=X> at start and </ANALYST> at end of each chunk.
        
        Args:
            text: Text with tags to split
            max_length: Maximum length of each chunk
            
        Returns:
            Formatted text with appropriate splits and tags
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
        
        Args:
            text: Raw transcript text
        
        Returns:
            Cleaned text with noise patterns removed
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

if __name__ == "__main__":
    start_time = time.time()
    print(f"Process started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # Specify model paths
    base_model_path = "/project/aaz/leo/models/Llama-3.1-8B"
    lora_model_path = "./results/lora_llama3.1_8b_att_0655"
    
    # Create and run the fine-tuner with attention masking
    fine_tuner = LlamaFineTunerWithAttention(
        base_model_path=base_model_path,
        lora_model_path=lora_model_path,
        training_mode=True  # Set to True to enable training
    )
    print(f"Successfully initiated llama fine_tuner with attention masking.")
    
    
    # Example usage:
    # Call the function to read parquet data
    folder_path = "dataset/ceo_scripts_export_20250405_193052"  # Replace with the actual path to your folder
    df, total_rows = read_parquet_data(folder_path)
    print(f"Successfully loaded {len(df)} dataframes with a total of {total_rows} rows")
    
    # Wrap the dataframe in a list since get_first_rows expects a list of dataframes 
    df = pd.read_parquet('/project/aaz/leo/dataset/ceo_scripts_export_20250405_193052/page_1.parquet')
    # Print confirmation
    print(f"Successfully loaded {len(df)} rows of dataset with a total of rows")
    
    # Get first 200,000 rows for fine-tuning
    df = get_first_rows(df, 200000)
    fine_tuner.data_massage(data=df, full_transcript=False)
    print(f"Successfully data processed {len(df)} rows of dataset for fine-tuning")
    print(fine_tuner.dataset.head(10))
    fine_tuner.train_model()
    
    
    end_time = time.time()
    total_duration = end_time - start_time
    print(f"Process completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total process time: {total_duration:.2f} seconds ({total_duration / 60:.2f} minutes)")