import torch
import bitsandbytes as bnb
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoModel
)
from datasets import Dataset
from peft import PeftModel
import pandas as pd
import os
from datetime import datetime
from rouge_score import rouge_scorer
from tqdm import tqdm
import pickle
import numpy as np

class LlamaEvaluator:
    def __init__(self, base_model_path, lora_model_path, output_dir, bertscore_model_name=None):
        """Initialize the evaluator with model paths and configurations"""
        # Set environment variables
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["HF_EVALUATE_OFFLINE"] = "1"  # Force offline mode
        
        # Store paths and configurations
        self.base_model_path = base_model_path
        self.lora_model_path = lora_model_path
        self.output_dir = output_dir
        self.bertscore_model_name = bertscore_model_name
        self.timenow = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize models and tokenizers
        self.tokenizer = None
        self.model = None
        self.bertscore_tokenizer = None
        self.bertscore_model = None
        
        # Configure quantization
        self.quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )
        
        # Initialize rouge scorer
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
        
        # Results storage
        self.generated_texts = []
        self.all_results = []
        self.all_embeddings = []
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def load_fine_tuned_model(self):
        """Load the fine-tuned model with LoRA adapters"""
        print("Loading fine-tuned model...")
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
        print("Model loaded successfully!")
        
        return self.tokenizer, self.model

    def load_bertscore_model(self):
        """Load the model for BERTScore evaluation"""
        if self.bertscore_model_name:
            print(f"Loading BERTScore model: {self.bertscore_model_name}")
            self.bertscore_tokenizer = AutoTokenizer.from_pretrained(self.bertscore_model_name)
            self.bertscore_model = AutoModel.from_pretrained(self.bertscore_model_name).to("cuda")
            print("BERTScore model loaded successfully!")
        else:
            print("No BERTScore model specified, skipping BERTScore evaluation.")

    def get_embedding_and_text(self, text):
        """Generate text and extract embeddings from the fine-tuned model"""
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
        last_hidden_state = outputs_for_embedding.hidden_states[-1].detach()
        
        # Mean pooling across sequence length dimension
        embedding = last_hidden_state.mean(dim=1)
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(generation_outputs[0], skip_special_tokens=True)
        
        return embedding, generated_text, last_hidden_state

    def load_test_data(self, csv_path):
        """Load test data from CSV file"""
        try:
            df = pd.read_csv(csv_path)
            print(f"Loaded {len(df)} examples from {csv_path}")
            return df
        except Exception as e:
            print(f"Could not load CSV data: {e}")
            return None

    def calculate_rouge_scores(self, reference, prediction):
        """Calculate ROUGE scores for a given reference and prediction"""
        scores = self.scorer.score(reference, prediction)
        rouge_scores_dict = {}
        
        for key, value in scores.items():
            rouge_scores_dict[key] = {
                "precision": value.precision,
                "recall": value.recall,
                "fmeasure": value.fmeasure
            }
        
        return rouge_scores_dict

    def calculate_bertscore(self, predictions, references):
        """Calculates BERTScore using the specified model"""
        if not self.bertscore_model or not self.bertscore_tokenizer:
            print("BERTScore model not loaded. Run load_bertscore_model() first.")
            return []
            
        bertscore_results = []
        for pred, ref in tqdm(zip(predictions, references), total=len(predictions), desc="Calculating BERTScore"):
            # Tokenize and move to GPU
            encoded_predictions = self.bertscore_tokenizer(pred, padding=True, truncation=True, return_tensors='pt').to("cuda")
            encoded_references = self.bertscore_tokenizer(ref, padding=True, truncation=True, return_tensors='pt').to("cuda")

            with torch.no_grad():
                # Get embeddings
                predictions_embeddings = self.bertscore_model(**encoded_predictions).last_hidden_state
                references_embeddings = self.bertscore_model(**encoded_references).last_hidden_state

            # CLS POOLING
            predictions_embeddings = predictions_embeddings[:, 0, :]  # Use CLS token embedding
            references_embeddings = references_embeddings[:, 0, :]    # Use CLS token embedding

            # Calculate cosine similarity (BERTScore F1)
            cosine_similarity = torch.nn.functional.cosine_similarity(predictions_embeddings, references_embeddings, dim=1)
            
            # For each value in the batch, create a dictionary entry for the metrics
            for val in cosine_similarity.tolist():
                bertscore_results.append({'precision': val, 'recall': val, 'f1': val})
        
        return bertscore_results

    def process_examples(self, df):
        """Process each example, generate text, and calculate metrics"""
        print(f"Processing {len(df)} examples...")
        
        for index, row in df.iterrows():
            transcript_id = row['transcript_id']
            example = row['conversation']
            print(f"\n\n***** Transcript ID: {transcript_id} *****")
            print(f"Input: {example}")
            print("-----------------------Processing...----------------------------------")
            
            # Generate text and get embeddings
            embedding, generated_text, last_hidden = self.get_embedding_and_text(example)
            
            # Print results
            print(f"\nEmbedding shape: {embedding.shape}")
            print(f"Last hidden state shape: {last_hidden.shape}")
            print(f"\nGenerated text:\n{generated_text}")
            print(f"\nSample embedding values (first 5): {embedding[0, :5].tolist()}")
            
            # Calculate ROUGE scores
            rouge_scores_dict = self.calculate_rouge_scores(example, generated_text)
            
            # Print ROUGE scores
            print("\nROUGE Scores:")
            for key, value in rouge_scores_dict.items():
                print(f"  {key}:")
                print(f"    Precision: {value['precision']:.4f}")
                print(f"    Recall:    {value['recall']:.4f}")
                print(f"    F-measure: {value['fmeasure']:.4f}")
            
            # Store results
            self.generated_texts.append(generated_text)
            self.all_embeddings.append(embedding.cpu())
            
            results = {
                "example_id": transcript_id,
                "input": example,
                "generated_text": generated_text,
                "rouge_scores": rouge_scores_dict,
            }
            
            self.all_results.append(results)
        
        print(f"Processed {len(df)} examples successfully.")

    def save_results(self):
        """Save embeddings and results metadata"""
        if not self.all_embeddings or not self.all_results:
            print("No results to save. Run process_examples() first.")
            return
            
        # Combine all embeddings into a single NumPy array
        combined_embeddings = torch.cat(self.all_embeddings, dim=0).numpy()
        
        # Save the combined embeddings
        embedding_filename = os.path.join(self.output_dir, "embeddings_all.npy")
        np.save(embedding_filename, combined_embeddings)
        
        # Add the embedding filename to each result entry in the metadata
        for result in self.all_results:
            result["embedding_file"] = embedding_filename
        
        # Save metadata to a pickle file
        metadata_file = os.path.join(self.output_dir, "metadata.pkl")
        with open(metadata_file, "wb") as f:
            pickle.dump(self.all_results, f)
        
        print(f"Metadata saved to: {metadata_file}")
        print(f"Combined embeddings saved to: {embedding_filename}")

    def evaluate_bertscore(self):
        """Evaluate with BERTScore and print results"""
        if not self.bertscore_model or not self.generated_texts:
            print("Either BERTScore model not loaded or no generated texts available.")
            return
            
        input_texts = [result["input"] for result in self.all_results]
        
        print("\nCalculating BERTScore...")
        bert_scores = self.calculate_bertscore(self.generated_texts, input_texts)
        
        # Print average BERTScore results
        print("\nBERTScore Results (Average):")
        avg_precision = sum(score['precision'] for score in bert_scores) / len(bert_scores)
        avg_recall = sum(score['recall'] for score in bert_scores) / len(bert_scores)
        avg_f1 = sum(score['f1'] for score in bert_scores) / len(bert_scores)
        print(f"  Precision: {avg_precision:.4f}")
        print(f"  Recall: {avg_recall:.4f}")
        print(f"  F1: {avg_f1:.4f}")
        
        # Print detailed BERTScore results per example
        print("\nBERTScore Results (Detailed):")
        for i, score in enumerate(bert_scores):
            print(f"\nExample {i+1}:")
            print(f"  Precision: {score['precision']:.4f}")
            print(f"  Recall: {score['recall']:.4f}")
            print(f"  F1: {score['f1']:.4f}")
            
        # Add BERTScore to results
        for i, result in enumerate(self.all_results):
            result["bertscore"] = bert_scores[i]
            
        return bert_scores

    def run_evaluation(self, csv_path):
        """Run the complete evaluation pipeline"""
        # Load models
        self.load_fine_tuned_model()
        if self.bertscore_model_name:
            self.load_bertscore_model()
        
        # Load test data
        df = self.load_test_data(csv_path)
        if df is None:
            print("No data to evaluate. Exiting.")
            return
        
        # Process examples
        self.process_examples(df)
        
        # Evaluate BERTScore if model is loaded
        if self.bertscore_model:
            self.evaluate_bertscore()
        
        # Save results
        self.save_results()
        
        print("Evaluation complete!")


if __name__ == "__main__":
    # Example usage
    base_model_path = "/project/aaz/leo/models/Llama-3.1-8B"
    lora_model_path = "./results/lora_llama3.1_8b_raw_0605"
    output_dir = "./results/embedding/"
    bertscore_model_name = "/project/aaz/leo/models/bge-large-en-v1.5"  # Optional, set to None to skip BERTScore
    csv_path = "./dataset/Cleaned_CEO_Text.csv"
    
    # Create evaluator instance
    evaluator = LlamaEvaluator(
        base_model_path=base_model_path,
        lora_model_path=lora_model_path,
        output_dir=output_dir,
        bertscore_model_name=bertscore_model_name
    )
    
    # OPTION 1: Run the complete evaluation pipeline
    evaluator.run_evaluation(csv_path)
    
    # OPTION 2: Run individual steps manually
    # Step 1: Load the fine-tuned model
    # tokenizer, model = evaluator.load_fine_tuned_model()
    
    # Step 2: Load BERTScore model (optional)
    # evaluator.load_bertscore_model()
    
    # Step 3: Load test data
    # df = evaluator.load_test_data(csv_path)
    
    # Step 4: Process examples
    # evaluator.process_examples(df)
    
    # Step 5: Evaluate with BERTScore (optional)
    # bert_scores = evaluator.evaluate_bertscore()
    
    # Step 6: Save results
    # evaluator.save_results()
    
    # Example of getting embedding and text for a single example
    # embedding, generated_text, last_hidden = evaluator.get_embedding_and_text("Your example text here")
    
    # Example of calculating ROUGE scores for a single example
    # rouge_scores = evaluator.calculate_rouge_scores("Reference text", "Generated text")