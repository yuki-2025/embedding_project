import psycopg2
from psycopg2 import sql
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import time

def connect_to_db(username,password):
    """
    Connect to the PostgreSQL database using psycopg2.
    Returns the connection object if successful, otherwise None.
    """
    # It's better practice to get sensitive info like passwords from environment variables
    # or a config file, but using the provided password for this example.
    # password = "temp_03292025" # Replace with your actual password or secure method
    # password = "123123"
    conn = None  # Initialize conn to None
    try:
        # Define your connection parameters
        conn = psycopg2.connect(
            host="aaz2.chicagobooth.edu",
            port="54320",
            dbname="postgres",
            user=username,
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

if __name__ == "__main__":
    db_connection = connect_to_db(username=' ',password='')

    if db_connection:
        # Create a timestamped output directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = f"dataset/ceo_scripts_export_{timestamp}"
        
        # Export data in pages of 10,000 records
        # Set max_pages=None to fetch all data 
        total_exported = perform_keyset_pagination_with_parquet_export(
            db_connection, 
            output_dir=output_dir,
            page_size=1000000, 
            max_pages=None
        )
        
        print(f"\nTotal records exported: {total_exported}")
        
        # Close the connection when done
        try:
            db_connection.close()
            print("\nDatabase connection closed.")
        except Exception as e:
            print(f"Error closing connection: {e}")