#!/usr/bin/env python3
# Script to read and display embeddings from a parquet file

import pandas as pd
import numpy as np
import torch
import os
import sys
from datetime import datetime

def load_embeddings(file_path):
    """
    Load embeddings from a parquet file
    
    Args:
        file_path (str): Path to the parquet file containing embeddings
        
    Returns:
        pd.DataFrame: DataFrame with the loaded embeddings
    """
    print(f"Loading embeddings from {file_path}...")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        sys.exit(1)
        
    # Load the parquet file
    try:
        df = pd.read_parquet(file_path)
        print(df[1000:1020])
        print(f"Successfully loaded embeddings file with {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error loading parquet file: {e}")
        sys.exit(1)
        
def analyze_embeddings(df):
    """
    Analyze and display information about the embeddings
    
    Args:
        df (pd.DataFrame): DataFrame containing embeddings
    """
    print("\n===== Embedding Analysis =====")
    
    # Display dataframe info
    print("\nDataFrame Information:")
    print(f"- Rows: {len(df)}")
    print(f"- Columns: {', '.join(df.columns)}")
    
    # Check embedding shape if available
    if 'embedding' in df.columns:
        # Find first non-null embedding
        for idx, row in df.iterrows():
            if row['embedding'] is not None:
                try:
                    embedding_array = row['embedding']
                    print(f"\nEmbedding Details:")
                    print(f"- Type: {type(embedding_array)}")
                    
                    # Convert to proper numpy array if it's a nested object
                    if isinstance(embedding_array, (list, np.ndarray)):
                        if isinstance(embedding_array, list):
                            embedding_np = np.array(embedding_array, dtype=np.float32)
                        else:
                            # Handle case where it's already a numpy array but might be object type
                            embedding_np = np.array(embedding_array.tolist(), dtype=np.float32)
                            
                        print(f"- Shape: {embedding_np.shape}")
                        print(f"- First 5 values: {embedding_np[:5]}")
                        
                        # Basic statistics
                        print(f"- Min: {embedding_np.min():.6f}")
                        print(f"- Max: {embedding_np.max():.6f}")
                        print(f"- Mean: {embedding_np.mean():.6f}")
                        print(f"- Std: {embedding_np.std():.6f}")
                    else:
                        print("Embedding is not a list or array type")
                        
                    break
                except Exception as e:
                    print(f"Error analyzing embedding: {e}")
                    print(f"Embedding data structure: {type(embedding_array)}")
                    
                    # Try to provide more details about the structure
                    if isinstance(embedding_array, list):
                        print(f"List length: {len(embedding_array)}")
                        if embedding_array and isinstance(embedding_array[0], list):
                            print(f"First element type: {type(embedding_array[0])}, length: {len(embedding_array[0])}")
                    elif isinstance(embedding_array, np.ndarray):
                        print(f"Array shape: {embedding_array.shape}")
                        print(f"Array dtype: {embedding_array.dtype}")
                    
                    # Try an alternative approach
                    try:
                        print("\nAttempting alternative analysis:")
                        # Convert to standard Python list first
                        if hasattr(embedding_array, 'tolist'):
                            embedding_list = embedding_array.tolist()
                        else:
                            embedding_list = list(embedding_array)
                            
                        print(f"- List length: {len(embedding_list)}")
                        print(f"- First 5 values: {embedding_list[:5]}")
                    except Exception as e2:
                        print(f"Alternative analysis failed: {e2}")
        else:
            print("No valid embeddings found.")
    else:
        print("No 'embedding' column found in the dataframe.")
    
    # Display sample of transcript text if available
    if 'transcript' in df.columns:
        print("\nSample Transcript (truncated):")
        sample_text = df.iloc[0]['transcript']
        print(sample_text[:500] + "..." if len(sample_text) > 500 else sample_text)
        
    print("\n=========================")

def check_embedding_completeness(df):
    """
    Check for empty or null embeddings in the dataframe
    
    Args:
        df (pd.DataFrame): DataFrame containing embeddings
    """
    print("\n===== Embedding Completeness Check =====")
    
    if 'embedding' not in df.columns:
        print("No 'embedding' column found in the dataframe.")
        return
    
    # Count different types of empty/null values
    null_count = df['embedding'].isna().sum()
    empty_count = 0
    zero_length_count = 0
    all_zeros_count = 0
    valid_count = 0
    total_rows = len(df)
    
    # Check each row
    for idx, row in df.iterrows():
        embedding = row['embedding']
        
        if pd.isna(embedding):
            continue  # Already counted in null_count
        
        try:
            if isinstance(embedding, (list, np.ndarray)):
                # Check if it's an empty list/array
                if hasattr(embedding, 'size') and embedding.size == 0:
                    zero_length_count += 1
                elif len(embedding) == 0:
                    zero_length_count += 1
                # Check if it's all zeros - safely handle different array types
                elif isinstance(embedding, np.ndarray):
                    # Convert to a simple numpy array with a single data type if needed
                    try:
                        # First try a safe conversion to float array
                        if embedding.dtype == np.dtype('O'):
                            np_emb = np.array(embedding.tolist(), dtype=np.float32)
                        else:
                            np_emb = embedding
                            
                        if not np.any(np_emb):
                            all_zeros_count += 1
                        else:
                            valid_count += 1
                    except (ValueError, TypeError):
                        # If conversion fails, check manually
                        try:
                            if all(float(v) == 0 for v in embedding):
                                all_zeros_count += 1
                            else:
                                valid_count += 1
                        except (ValueError, TypeError):
                            print(f"Cannot process embedding at index {idx}: unusual data structure")
                            empty_count += 1
                elif all(float(v) == 0 for v in embedding):
                    all_zeros_count += 1
                else:
                    valid_count += 1
            elif embedding == '':
                empty_count += 1
            else:
                # Something else that's not null but also not a proper embedding
                print(f"Unexpected embedding type at index {idx}: {type(embedding)}")
                empty_count += 1
        except Exception as e:
            print(f"Error processing embedding at index {idx}: {e}")
            empty_count += 1
    
    # Print results
    print(f"Total rows: {total_rows}")
    print(f"Null embeddings: {null_count} ({null_count/total_rows*100:.2f}%)")
    print(f"Empty string embeddings: {empty_count} ({empty_count/total_rows*100:.2f}%)")
    print(f"Zero-length embeddings: {zero_length_count} ({zero_length_count/total_rows*100:.2f}%)")
    print(f"All-zero embeddings: {all_zeros_count} ({all_zeros_count/total_rows*100:.2f}%)")
    print(f"Valid embeddings: {valid_count} ({valid_count/total_rows*100:.2f}%)")
    
    # Optional: List some indices with missing embeddings
    if null_count + empty_count + zero_length_count + all_zeros_count > 0:
        print("\nSample indices with missing embeddings:")
        count = 0
        for idx, row in df.iterrows():
            embedding = row['embedding']
            
            try:
                is_missing = pd.isna(embedding)
                
                if not is_missing and isinstance(embedding, (list, np.ndarray)):
                    if len(embedding) == 0:
                        is_missing = True
                    elif isinstance(embedding, np.ndarray):
                        try:
                            if embedding.dtype == np.dtype('O'):
                                np_emb = np.array(embedding.tolist(), dtype=np.float32)
                            else:
                                np_emb = embedding
                            
                            is_missing = not np.any(np_emb)
                        except (ValueError, TypeError):
                            try:
                                is_missing = all(float(v) == 0 for v in embedding)
                            except:
                                is_missing = False
                    else:
                        try:
                            is_missing = all(float(v) == 0 for v in embedding)
                        except:
                            is_missing = False
                elif not is_missing:
                    is_missing = (embedding == '')
                
                if is_missing:
                    print(f"Row {idx}: Missing embedding")
                    count += 1
                    if count >= 5:  # Show at most 5 examples
                        break
            except Exception as e:
                print(f"Error checking embedding at index {idx}: {e}")
                
    print("\n==========================")

def main():
    start_time = datetime.now()
    print(f"Process started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define the path to the embeddings file
    #embeddings_file = "/project/aaz/leo/embeddings/embeddings_20250430_094656._100k_full.parquet"
    #embeddings_file = "/project/aaz/leo/embeddings/embeddings_20250430_122845_100k_q&a.parquet"
    embeddings_file = "/project/aaz/leo/embeddings/embeddings_20250429_235128_10k_full.parquet"
    embeddings_file = "/project/aaz/leo/embeddings/embeddings_20250430_171840_10k_q&a.parquet"
    embeddings_file = "/project/aaz/leo/embeddings/embeddings_20250501_202056.parquet"
    # Load and analyze the embeddings
    df = load_embeddings(embeddings_file)
    #analyze_embeddings(df)
    check_embedding_completeness(df)
    
    # Print execution time
    end_time = datetime.now()
    print(f"\nProcess completed in {(end_time - start_time).total_seconds():.2f} seconds")

if __name__ == "__main__":
    main()