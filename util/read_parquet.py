import os
import glob
import pandas as pd

def read_parquet_data(folder_path):
    """
    Reads all parquet files in a folder in lexicographical order and returns a combined dataframe.

    Args:
        folder_path (str): The path to the folder containing the parquet files.
        
    Returns:
        pandas.DataFrame: A combined DataFrame with data from all parquet files.
        int: Total number of rows in the combined dataframe.
    """
    # Get a list of all parquet files in the folder, sorted lexicographically
    parquet_files = sorted(glob.glob(os.path.join(folder_path, "*.parquet")))

    total_rows = 0
    dataframes = []  # Store dataframes to concatenate later

    for file_path in parquet_files:
        try:
            df = pd.read_parquet(file_path)
            num_rows = len(df)
            total_rows += num_rows
            dataframes.append(df)  # append df to the dataframes list
            print(f"File: {file_path}, Rows: {num_rows}")  # Print rows of individual file
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            continue  # Continue to next file instead of returning

    print(f"\nTotal rows in all files: {total_rows}")
    
    # Combine all dataframes into a single dataframe
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        return combined_df, total_rows
    else:
        print("No valid dataframes to process")
        return None, 0

def get_first_rows(dataframe, row=10000):
    """
    Gets the first N rows from a dataframe.
    
    Args:
        dataframe (pandas.DataFrame): A pandas DataFrame.
        row (int): Number of rows to return (default: 10000).
        
    Returns:
        pandas.DataFrame: Dataframe with first N rows.
    """ 
    # Check if dataframe is empty
    if dataframe is None or len(dataframe) == 0:
        print("No data to process")
        return None
    
    try:
        # Get first N rows
        first_rows = dataframe.head(row)
        
        print(f"Retrieved {len(first_rows)} rows from the dataframe")
        return first_rows
        
    except Exception as e:
        print(f"Error processing dataframe: {e}")
        return None
 

def extract_speaker_text(dataframe, num_rows=10):
    """
    Extracts data from all columns in a dataframe and writes the first num_rows to a text file.
    
    Args:
        dataframe (pandas.DataFrame): A pandas DataFrame.
        num_rows (int): Number of rows to extract and write (default: 10).
        
    Returns:
        bool: True if successful, False otherwise.
    """
    output_file = "/project/aaz/leo/speaker_text.txt"
    
    # Check if dataframe is empty
    if dataframe is None or len(dataframe) == 0:
        print("No data to process")
        return False
    
    # Take only the first num_rows
    try:
        if len(dataframe) > num_rows:
            sample_df = dataframe.head(num_rows)
        else:
            sample_df = dataframe
        
        # Print a preview
        print(f"\nPreview of first 5 rows:")
        print(sample_df.head(5))
        
        # Write to file with good formatting
        with open(output_file, "w", encoding="utf-8") as f:
            # Write header
            f.write("="*80 + "\n")
            f.write("EXTRACTED DATA\n")
            f.write("="*80 + "\n\n")
            
            # Write each row with all columns
            for i, (idx, row) in enumerate(sample_df.iterrows()):
                f.write(f"--- Record {i+1} ---\n")
                for col in sample_df.columns:
                    f.write(f"{col}: {row[col]}\n")
                f.write("\n")
                
        print(f"\nFirst {len(sample_df)} rows with all columns written to {output_file}")
        return True
        
    except Exception as e:
        print(f"Error processing dataframe: {e}")
        return False


# # Example usage:
# if __name__ == "__main__":
#     folder_path = "dataset/ceo_scripts_export_20250405_193052"  # Replace with the actual path to your folder
#     combined_df, total_rows = read_parquet_data(folder_path)
#     if combined_df is not None:
#         extract_speaker_text(combined_df, 100)  # Extract 10 rows