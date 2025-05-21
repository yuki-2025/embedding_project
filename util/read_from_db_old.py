import psycopg2
from psycopg2 import sql

import pandas as pd
from sqlalchemy import create_engine
 

# Create an engine to connect to your PostgreSQL database.
# Replace 'your_password' with your actual password.
engine = create_engine('postgresql://yukileong:temp_03292025@aaz2.chicagobooth.edu:54320/postgres')

# Load data from the two tables.
ceo_nor_data = pd.read_sql_query("SELECT * FROM llms.ceo_nor_data", engine)
speaker_data = pd.read_sql_query("SELECT * FROM fs_core.speaker_data", engine)

# Define the join columns based on the R inner_join:
join_cols = ['transcript_id', 'version_id', 'transcript_type', 'processed_fs', 'context_id', 'speaker_number']

# In speaker_data, select only the columns needed.
speaker_data_subset = speaker_data[join_cols + ['speaker_type', 'speaker_text']]

# Perform an inner join to create speaker_data_alt.
speaker_data_alt = pd.merge(ceo_nor_data, speaker_data_subset, on=join_cols, how='inner')

# Check if counts are identical.
ceo_count = len(ceo_nor_data)
speaker_alt_count = len(speaker_data_alt)
print(f"ceo_nor_data count: {ceo_count}")
print(f"speaker_data_alt count: {speaker_alt_count}")

# Descriptives: Group by 'speaker_type' and count the rows.
speaker_type_counts = speaker_data_alt.groupby('speaker_type').size().reset_index(name='count')
print("\nCounts by speaker_type:")
print(speaker_type_counts)

# Group by 'is_ceo' and count the rows.
# Note: Ensure that the column 'is_ceo' exists in the joined data.
if 'is_ceo' in speaker_data_alt.columns:
    is_ceo_counts = speaker_data_alt.groupby('is_ceo').size().reset_index(name='count')
    print("\nCounts by is_ceo:")
    print(is_ceo_counts)
else:
    print("\nColumn 'is_ceo' not found in the data.")

# Group by 'is_qa' and count the rows.
# Note: Ensure that the column 'is_qa' exists in the joined data.
if 'is_qa' in speaker_data_alt.columns:
    is_qa_counts = speaker_data_alt.groupby('is_qa').size().reset_index(name='count')
    print("\nCounts by is_qa:")
    print(is_qa_counts)
else:
    print("\nColumn 'is_qa' not found in the data.")

# Close the database connection.
engine.dispose()