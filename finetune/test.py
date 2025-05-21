# file.py
import pandas as pd

try:
    # Load the CSV file
    df1 = pd.read_csv("./dataset/Cleaned_CEO_Text.csv")
    
    df = df1.tail(5)

    # Truncate the 'conversation' column
    df['conversation'] = df['conversation'].str[:50] + '...'  # Truncate to 50 chars + ellipsis

    # Print the DataFrame as a table
    print(df.head(5)) 
    
    # Check for duplicate transcript_ids
    duplicates = df['transcript_id'].duplicated(keep=False)

    # Print duplicate transcript_ids (if any)
    if duplicates.any():
        print("Duplicate transcript_ids found:")
        print(df[duplicates]['transcript_id'])

        # Option 1:  Print the entire rows with duplicates
        print("\nRows with duplicate transcript_ids:")
        print(df[duplicates])

        # Option 2:  Count the occurrences of each duplicate
        print("\nDuplicate counts:")
        print(df['transcript_id'].value_counts()[df['transcript_id'].value_counts() > 1])
        
        # Option 3: Get a list of the *unique* duplicate IDs
        duplicate_ids = df.loc[df['transcript_id'].duplicated(keep=False), 'transcript_id'].unique()
        print("\nUnique Duplicate Transcript IDs:", duplicate_ids)
except FileNotFoundError:
    print("Error: CSV file not found. Please check the file path.")
except pd.errors.EmptyDataError:
    print("Error: CSV file is empty.")
except pd.errors.ParserError:
    print("Error: Could not parse the CSV file. Check for formatting issues.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")