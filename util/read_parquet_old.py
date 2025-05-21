import os
import glob
import pandas as pd

def read_parquet_data(folder_path):
    """
    Reads all parquet files in a folder in lexicographical order,
    calculates the total number of rows, and prints the row count of a specific file.

    Args:
        folder_path (str): The path to the folder containing the parquet files.
    """

    # Get a list of all parquet files in the folder, sorted lexicographically
    parquet_files = sorted(glob.glob(os.path.join(folder_path, "*.parquet")))

    total_rows = 0
    dataframes = []  # Store dataframes to potentially concatenate later

    for file_path in parquet_files:
        try:
            df = pd.read_parquet(file_path)
            num_rows = len(df)
            total_rows += num_rows
            dataframes.append(df)  # append df to the dataframes list
            print(f"File: {file_path}, Rows: {num_rows}")  # Print rows of individual file
            break
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return  # Stop processing if there's an error

    print(f"\nTotal rows in all files: {total_rows}")
    
    speaker_texts = []
    output_file = "speaker_text.txt"
    if 'speaker_text' in df.columns:
        print(f"\nFirst 5 rows of 'speaker_text' in {file_path}:")
        print(df['speaker_text'].head(5))
        speaker_texts.extend(df['speaker_text'].head(20).tolist())
        with open(output_file, "w", encoding="utf-8") as f:
            for i in range(min(10, len(speaker_texts))):
                f.write(f"{speaker_texts[i]}\n")

        print(f"\nFirst 10 speaker_text entries written to {output_file}")
    else:
        print(f"\n'speaker_text' column not found in {file_path}")
    

    # # Example of accessing and printing the row count for the specified file:
    # specific_file_path = os.path.join(folder_path, "ceo_scripts_export_20250405_193052", "page_1.parquet")

    # if os.path.exists(specific_file_path):
    #     try:
    #         df_specific = pd.read_parquet(specific_file_path)
    #         num_rows_specific = len(df_specific)
    #         print(f"\nRows in {specific_file_path}: {num_rows_specific}")
    #     except Exception as e:
    #          print(f"Error reading file {specific_file_path}: {e}")
    # else:
    #     print(f"\nFile {specific_file_path} not found.")


# Example usage:
folder_path = "dataset/ceo_scripts_export_20250405_193052"  # Replace with the actual path to your folder
read_parquet_data(folder_path)