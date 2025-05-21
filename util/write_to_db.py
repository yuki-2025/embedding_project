import psycopg2
from psycopg2 import sql
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import time
import sys
import numpy as np

sys.path.append('/project/aaz/leo')
from util.read_parquet import read_parquet_data, get_first_rows, extract_speaker_text

def connect_to_db():
    """
    Connect to the PostgreSQL database using psycopg2.
    Returns the connection object if successful, otherwise None.
    """
    # It's better practice to get sensitive info like passwords from environment variables
    # or a config file, but using the provided password for this example.
    # password = "temp_0423025" # Replace with your actual password or secure method
    password = "123123"
    conn = None  # Initialize conn to None
    try:
        # Define your connection parameters
        conn = psycopg2.connect(
            host="aaz2.chicagobooth.edu",
            port="54320",
            dbname="postgres",
            user="yukileong",
            password=password
        )
        print("Connection to PostgreSQL successful!")
        
        # Optional: Check DB version
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            db_version = cur.fetchone()
            print(f"Database version: {db_version[0]}")
        
        return conn # Return the connection object

    except Exception as e:
        print("Error connecting to PostgreSQL:", e)
        if conn:
            conn.close() # Ensure connection is closed on error if partially opened
        return None

def perform_keyset_pagination_with_parquet_export(conn, output_dir="parquet_output", page_size=3000000, max_pages=None): #3000000
    """
    Performs keyset pagination on the large database table and exports each page to a parquet file.
    
    Args:
        conn: Database connection
        output_dir: Directory to save parquet files
        page_size: Number of records to fetch per page
        max_pages: Maximum number of pages to fetch (None for all pages)
        
    Returns:
        The total number of records exported
    """
    if not conn:
        print("No valid database connection provided.")
        return 0
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nStarting keyset pagination with page size: {page_size}")
    print(f"Data will be exported to: {output_dir}")
    
    # Variables to track pagination
    total_records = 0
    page_count = 0
    last_new_id = 0
    has_more = True
    
    start_time = time.time()
    
    try:
        # First, get column names to use in our DataFrame
        with conn.cursor() as cur:
            cur.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_schema = 'llms' AND table_name = 'ceo_scripts'
                ORDER BY ordinal_position
            """)
            columns = [row[0] for row in cur.fetchall()]
        
        print(f"Columns in the table: {columns}")
        
        while has_more and (max_pages is None or page_count < max_pages):
            # Build the query based on whether this is the first page or not
            query = """
            SELECT *
            FROM llms.ceo_scripts
            WHERE new_id > %s 
            ORDER BY new_id
            LIMIT %s
            """
            params = (last_new_id, page_size)
            
            with conn.cursor() as cur:
                cur.execute(query, params)
                page_results = cur.fetchall()
                
                # If we got fewer results than page_size, we've reached the end
                if len(page_results) < page_size:
                    has_more = False
                
                # If we got results, process and export them
                if page_results:
                    # Get the new_id of the last row for next page's query
                    last_row = page_results[-1]
                    last_new_id = last_row[0]  # Assuming new_id is first column
                    
                    page_count += 1
                    total_records += len(page_results)
                    
                    # Convert to pandas DataFrame
                    df = pd.DataFrame(page_results, columns=columns)
                    
                    # Export to parquet
                    file_path = os.path.join(output_dir, f"page_{page_count}.parquet")
                    df.to_parquet(file_path, engine='pyarrow', index=False)
                    
                    elapsed = time.time() - start_time
                    rate = total_records / elapsed if elapsed > 0 else 0
                    
                    print(f"Exported page {page_count} with {len(page_results)} records to {file_path}")
                    print(f"Last ID: {last_new_id}, Total records: {total_records}, Avg rate: {rate:.2f} records/sec")
                    
                    # For the first page, print sample results
                    if page_count == 1:
                        print("\nSample Results (first 5 rows):")
                        print(df.head())
                else:
                    has_more = False
        
        elapsed_time = time.time() - start_time
        print(f"\nExport complete. Exported {total_records} records in {page_count} pages.")
        print(f"Total time: {elapsed_time:.2f} seconds, Average rate: {total_records/elapsed_time:.2f} records/sec")
        
    except Exception as e:
        print(f"Error during keyset pagination: {e}")
    
    return total_records

def save_embeddings_to_postgres(conn, dataframe, table_name='llms.embeddings', batch_size=1000):
    """
    Reads a parquet file containing embedding vectors and saves them to PostgreSQL.
    
    Args:
        conn: Database connection
        parquet_file_path: Path to the parquet file containing embeddings
        table_name: Target table in PostgreSQL (schema.table)
        batch_size: Number of records to insert in each batch
        
    Returns:
        Number of records saved to the database
    """
    if not conn:
        print("No valid database connection provided.")
        return 0
    
    try: 
        df = dataframe
        # Print dataframe info
        # print(f"Loaded dataframe with {len(df)} rows and {len(df.columns)} columns")
        # print("Column names:", df.columns.tolist())
        
        # Check if the table exists, create it if it doesn't
        schema, table = table_name.split('.')
        with conn.cursor() as cur:
            # # Check if schema exists
            # cur.execute("""
            #     SELECT schema_name FROM information_schema.schemata 
            #     WHERE schema_name = %s
            # """, (schema,))
            
            # if cur.fetchone() is None:
            #     print(f"Creating schema: {schema}")
            #     cur.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
            
            # Check if table exists
            cur.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = %s AND table_name = %s
            """, (schema, table))
            
            if cur.fetchone() is None:
                print(f"Creating table: {table_name}")
                
                # Dynamically generate CREATE TABLE statement based on dataframe structure
                columns = []
                for col_name, dtype in df.dtypes.items():
                    if col_name == 'embedding' or 'vector' in col_name or isinstance(df[col_name].iloc[0], (list, np.ndarray)):
                        # Assuming embeddings are stored as arrays/lists
                        #vector_dim = len(df[col_name].iloc[0]) if len(df) > 0 else 1536  # Default to 1536 if empty
                        col_type = f"vector({4096})"
                    elif np.issubdtype(dtype, np.integer):
                        col_type = "INTEGER"
                    elif np.issubdtype(dtype, np.floating):
                        col_type = "FLOAT"
                    elif np.issubdtype(dtype, np.datetime64):
                        col_type = "TIMESTAMP"
                    else:
                        col_type = "TEXT"
                    
                    columns.append(f"\"{col_name}\" {col_type}")
                
                create_table_query = f"""
                CREATE TABLE {table_name} (
                    id SERIAL PRIMARY KEY,
                    {', '.join(columns)}
                )
                """
                cur.execute(create_table_query)
                print(f"Table created with schema: {create_table_query}")
        
        # Commit the table creation
        conn.commit()
        
        # Insert data in batches
        total_inserted = 0
        num_batches = (len(df) + batch_size - 1) // batch_size
        
        print(f"Inserting {len(df)} records in {num_batches} batches of size {batch_size}")
        
        for batch_num in range(num_batches):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]
            
            if len(batch_df) == 0:
                continue
            
            # Generate column names and placeholders for INSERT statement
            columns = batch_df.columns
            placeholders = ', '.join(['%s'] * len(columns))
            column_names = ', '.join([f'"{col}"' for col in columns])
            insert_stmt = f"INSERT INTO {table_name} ({column_names}) VALUES ({placeholders})"
            
            # Convert data to list of tuples for executemany
            # Need to handle embedding vectors specially (convert numpy arrays to lists)
            rows_data = []
            for _, row in batch_df.iterrows():
                row_data = []
                for col in columns:
                    val = row[col]
                    if isinstance(val, np.ndarray):
                        val = val.tolist()
                    row_data.append(val)
                rows_data.append(tuple(row_data))
            
            with conn.cursor() as cur:
                cur.executemany(insert_stmt, rows_data)
            
            # Commit after each batch
            conn.commit()
            
            total_inserted += len(batch_df)
            print(f"Inserted batch {batch_num+1}/{num_batches}: {len(batch_df)} records. Total: {total_inserted}/{len(df)}")
        
        print(f"Successfully saved {total_inserted} embedding records to {table_name}")
        return total_inserted
        
    except Exception as e:
        print(f"Error saving embeddings to PostgreSQL: {e}")
        # Roll back if an error occurred
        conn.rollback()
        return 0
     

if __name__ == "__main__":
    conn = connect_to_db()

    if not conn:
        print("Failed to connect to database. Exiting.")
        exit()
    
    try: 
         
        embedding_file_path = 'embeddings/embeddings_20250428_174429.parquet'
        table_name = "llms.embeddings"
        batch_size = 1000
        
        if not os.path.exists(embedding_file_path):
            print(f"File not found: {embedding_file_path}") 
            exit()
        
        # Load the parquet file for preview
        print("Loading parquet file for preview...")
        df_preview = pd.read_parquet(embedding_file_path)
        print(f"\nParquet file contains {len(df_preview)} rows and {len(df_preview.columns)} columns")
        print("Column names:", df_preview.columns.tolist())
        
        if len(df_preview) > 0:
            # Check if there's an embedding column by looking for vector-like columns
            embedding_cols = [col for col in df_preview.columns if 
                             'embedding' in col.lower() or 
                             'vector' in col.lower() or
                             (hasattr(df_preview[col].iloc[0], '__len__') and len(df_preview[col].iloc[0]) > 10)]
            
            if embedding_cols:
                print(f"\nDetected potential embedding columns: {embedding_cols}")
                print(f"First embedding shape: {len(df_preview[embedding_cols[0]].iloc[0])}")
            else:
                print("\nWarning: No obvious embedding columns detected!")
         
        # Confirm before proceeding
        print(f"\nReady to import embeddings:")
        print(f"  - Source: {embedding_file_path}")
        print(f"  - Destination: {table_name}")
        print(f"  - Batch size: {batch_size}")
        
        # Start timer
        start_time = time.time()
        
        # Import the embeddings
        records_saved = save_embeddings_to_postgres(
            conn=conn,
            dataframe=df_preview[embedding_cols],
            table_name=table_name,
            batch_size=batch_size
        )
        
        # Report results
        elapsed_time = time.time() - start_time
        print(f"\nImport completed in {elapsed_time:.2f} seconds")
        print(f"Total records saved: {records_saved}")
     
    except Exception as e:
        print(f"Error during import: {e}")
    
    finally:
        # Close the database connection 
        try:
            conn.close()
            print("\nDatabase connection closed.")
        except Exception as e:
            print(f"Error closing connection: {e}")

