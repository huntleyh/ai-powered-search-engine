import os
import pyodbc
import csv
import sys
import struct
import pandas as pd
from azure.identity import DefaultAzureCredential
import time

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Azure SQL connection settings
conn_str_base = os.getenv('AZURE_SQL_CONNECTION_STRING')

# Data file path
FILE_NAME = 'dataset_full.xlsx' # can be CSV or XLSX (SRCExport.xlsx)
table_name = 'EnforcementActionsFull'

def get_azure_sql_token():
    """Get Azure AD access token for SQL Database"""
    try:
        credential = DefaultAzureCredential()
        # The scope for Azure SQL Database
        token = credential.get_token("https://database.windows.net/.default")
        return token.token
    except Exception as e:
        print(f"ERROR getting Azure AD token: {e}")
        return None

def create_connection_string_with_token():
    """Create connection string using Azure AD token"""
    if not conn_str_base:
        print("ERROR: AZURE_SQL_CONNECTION_STRING environment variable not set")
        return None
        
    token = get_azure_sql_token()
    if not token:
        return None
    
    # Convert token to the format pyodbc expects
    token_bytes = token.encode('utf-16-le')
    token_struct = struct.pack(f'<I{len(token_bytes)}s', len(token_bytes), token_bytes)
    
    return conn_str_base, token_struct

def validate_data_file():
    """Check if data file exists and is readable (CSV or XLSX)"""
    if not os.path.exists(FILE_NAME):
        print(f"ERROR: Data file '{FILE_NAME}' not found in current directory")
        return False
    try:
        if FILE_NAME.lower().endswith('.csv'):
            with open(FILE_NAME, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                headers = next(reader)
                if not headers:
                    print("ERROR: CSV file appears to be empty")
                    return False
            print(f"✓ CSV file validated: {len(headers)} columns found")
        elif FILE_NAME.lower().endswith('.xlsx'):
            df = pd.read_excel(FILE_NAME, engine='openpyxl')
            headers = list(df.columns)
            if not headers:
                print("ERROR: Excel file appears to be empty")
                return False
            print(f"✓ Excel file validated: {len(headers)} columns found")
        else:
            print("ERROR: Only .csv and .xlsx files are supported")
            return False
        return True
    except Exception as e:
        print(f"ERROR reading data file: {e}")
        return False

def validate_sql_connection():
    """Test SQL Server connection using Azure AD"""
    try:
        conn_info = create_connection_string_with_token()
        if not conn_info:
            return False
        
        conn_str, token_struct = conn_info
        
        # Connect using the token
        conn = pyodbc.connect(conn_str, attrs_before={1256: token_struct})
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        conn.close()
        
        print("✓ SQL Server connection successful (Azure AD)")
        return True
    except Exception as e:
        print(f"ERROR connecting to SQL Server with Azure AD: {e}")
        print("Make sure you're logged in with 'az login' or have proper Azure credentials configured")
        return False

def create_or_truncate_table(cursor):
    """Create table if not exists, otherwise truncate existing table"""
    # Check if table exists
    cursor.execute(f"""
        SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_NAME = '{table_name}'
    """)
    table_exists = cursor.fetchone()[0] > 0
    
    if table_exists:
        print(f"Table '{table_name}' exists. Truncating...")
        cursor.execute(f"TRUNCATE TABLE {table_name}")
    else:
        print(f"Creating table '{table_name}'...")
        cursor.execute(f'''
            CREATE TABLE {table_name} (
                ID INT PRIMARY KEY,
                Title NVARCHAR(MAX),
                BrowserFile NVARCHAR(MAX),
                Ordinal FLOAT NULL,
                DateIssued DATETIME,
                Published BIT,
                DocumentTypes NVARCHAR(MAX),
                KeyFacts NVARCHAR(MAX),
                DocumentText NVARCHAR(MAX),
                Commentary NVARCHAR(MAX),
                NumberOfViolations INT NULL,
                SettlementAmount FLOAT NULL,
                OfacPenalty NVARCHAR(50) NULL,
                AggregatePenalty NVARCHAR(50) NULL,
                BasePenalty NVARCHAR(50) NULL,
                StatutoryMaximum NVARCHAR(50) NULL,
                VSD NVARCHAR(10) NULL,
                Egregious NVARCHAR(10) NULL,
                WillfulOrReckless NVARCHAR(50) NULL,
                Criminal NVARCHAR(10) NULL,
                RegulatoryProvisions NVARCHAR(MAX) NULL,
                LegalIssues NVARCHAR(MAX) NULL,
                SanctionPrograms NVARCHAR(MAX) NULL,
                EnforcementCharacterizations NVARCHAR(MAX) NULL,
                Industries NVARCHAR(MAX) NULL,
                AggravatingFactors NVARCHAR(MAX) NULL,
                MitigatingFactors NVARCHAR(MAX) NULL,
                ReferenceCount INT NULL
            )
        ''')

def prepare_batch_data(headers, rows):
    """Prepare data for batch insert, handling int columns, empty strings, and NaN values"""
    int_columns = {'ID', 'NumberOfViolations'}
    filtered_headers = [col for col in headers if col.strip()]
    header_indices = [i for i, col in enumerate(headers) if col.strip()]
    int_indices = [i for i, col in enumerate(filtered_headers) if col in int_columns]

    batch_rows = []
    for row in rows:
        filtered_row = []
        for idx, i in enumerate(header_indices):
            val = row[i]
            # Treat empty string or NaN as None
            if val == '' or pd.isna(val):
                filtered_row.append(None)
            elif idx in int_indices:
                try:
                    if isinstance(val, float):
                        filtered_row.append(int(val))
                    elif isinstance(val, int):
                        filtered_row.append(val)
                    elif isinstance(val, str):
                        float_val = float(val)
                        filtered_row.append(int(float_val))
                    else:
                        filtered_row.append(None)
                except Exception:
                    filtered_row.append(None)
            else:
                filtered_row.append(val)
        batch_rows.append(filtered_row)
    return filtered_headers, batch_rows

def batch_insert_with_retry(cursor_factory, headers, batch_rows, max_retries=3, retry_delay=5):
    """Insert multiple rows in a single batch with retry logic."""
    if not batch_rows:
        return
    filtered_headers, processed_rows = prepare_batch_data(headers, batch_rows)
    placeholders = ','.join(['?'] * len(filtered_headers))
    columns = ','.join(f'[{col}]' for col in filtered_headers)
    sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
    attempt = 0
    while attempt < max_retries:
        try:
            cursor = cursor_factory()
            cursor.executemany(sql, processed_rows)
            cursor.connection.commit()
            return
        except Exception as e:
            attempt += 1
            print(f"Batch insert failed (attempt {attempt}/{max_retries}): {e}")
            if attempt >= max_retries:
                print("Max retries reached. Skipping this batch.")
                return
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            # Reconnect for next attempt
            cursor_factory(reconnect=True)

def check_schema_simple(headers):
    expected = [
        'ID', 'Title', 'BrowserFile', 'Ordinal', 'DateIssued', 'Published', 'DocumentTypes',
        'KeyFacts', 'DocumentText', 'Commentary', 'NumberOfViolations',
        'SettlementAmount', 'OfacPenalty', 'AggregatePenalty', 'BasePenalty',
        'StatutoryMaximum', 'VSD', 'Egregious', 'WillfulOrReckless', 'Criminal',
        'RegulatoryProvisions', 'LegalIssues', 'SanctionPrograms',
        'EnforcementCharacterizations', 'Industries', 'AggravatingFactors',
        'MitigatingFactors', 'ReferenceCount'
    ]
    if headers != expected:
        print("ERROR: Data file schema has changed (columns added, removed, renamed, or reordered). Please review your data file.")
        sys.exit(1)

def preflight_scan(headers, rows):
    """Scan all rows for int conversion, type, and truncation issues before import."""
    # Define schema: column_name: (type, max_length or None)
    schema = {
        'ID': ('int', None),
        'Title': ('str', None),  # NVARCHAR(MAX)
        'BrowserFile': ('str', None),  # NVARCHAR(MAX)
        'Ordinal': ('float', None),
        'DateIssued': ('datetime', None),
        'Published': ('bit', None),
        'DocumentTypes': ('str', None),
        'KeyFacts': ('str', None),
        'DocumentText': ('str', None),
        'Commentary': ('str', None),
        'NumberOfViolations': ('int', None),
        'SettlementAmount': ('float', None),
        'OfacPenalty': ('str', 50),
        'AggregatePenalty': ('str', 50),
        'BasePenalty': ('str', 50),
        'StatutoryMaximum': ('str', 50),
        'VSD': ('str', 10),
        'Egregious': ('str', 10),
        'WillfulOrReckless': ('str', 50),
        'Criminal': ('str', 10),
        'RegulatoryProvisions': ('str', None),
        'LegalIssues': ('str', None),
        'SanctionPrograms': ('str', None),
        'EnforcementCharacterizations': ('str', None),
        'Industries': ('str', None),
        'AggravatingFactors': ('str', None),
        'MitigatingFactors': ('str', None),
        'ReferenceCount': ('int', None)
    }
    import pandas as pd
    from datetime import datetime
    errors = []
    filtered_headers = [col for col in headers if col.strip()]
    header_indices = [i for i, col in enumerate(headers) if col.strip()]
    for row_num, row in enumerate(rows, start=2):  # start=2 to account for header row
        for idx, i in enumerate(header_indices):
            col = filtered_headers[idx]
            val = row[i]
            col_type, max_len = schema.get(col, (None, None))
            # Accept empty string or NaN as valid (will be NULL in DB)
            if val == '' or pd.isna(val):
                continue
            try:
                if col_type == 'int':
                    int(float(val))
                elif col_type == 'float':
                    float(val)
                elif col_type == 'bit':
                    # Accept 0, 1, True, False, '0', '1', 'true', 'false'
                    if isinstance(val, (int, float)):
                        if val not in (0, 1):
                            raise ValueError('bit out of range')
                    elif isinstance(val, str):
                        if val.lower() not in ('0', '1', 'true', 'false'):
                            raise ValueError('bit string invalid')
                    else:
                        raise ValueError('bit type invalid')
                elif col_type == 'datetime':
                    # Accept ISO, Excel, or pandas Timestamp
                    if isinstance(val, datetime):
                        pass
                    elif isinstance(val, str):
                        pd.to_datetime(val)
                    else:
                        raise ValueError('datetime type invalid')
                elif col_type == 'str':
                    sval = str(val)
                    if max_len is not None and len(sval) > max_len:
                        raise ValueError(f'string too long: {len(sval)} > {max_len}')
                # else: skip
            except Exception as e:
                errors.append((row_num, col, val, str(e)))
    if errors:
        print(f"Preflight scan found {len(errors)} problematic values:")
        for row_num, col, val, msg in errors[:10]:
            print(f"  Row {row_num}, Column '{col}': Value '{val}' | Error: {msg}")
        if len(errors) > 10:
            print(f"  ...and {len(errors)-10} more.")
        print("Aborting import. Please fix these values in your data file.")
        sys.exit(1)
    else:
        print("✓ Preflight scan passed: No type or truncation issues found.")

def main():
    print('Validating prerequisites...')
    
    # Validate data file (CSV or XLSX)
    if not validate_data_file():
        sys.exit(1)
    
    # Check schema
    if FILE_NAME.lower().endswith('.csv'):
        with open(FILE_NAME, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)
            check_schema_simple(headers)
    elif FILE_NAME.lower().endswith('.xlsx'):
        df = pd.read_excel(FILE_NAME, engine='openpyxl')
        headers = list(df.columns)
        check_schema_simple(headers)
    else:
        print("ERROR: Only .csv and .xlsx files are supported for schema check.")
        sys.exit(1)
    
    # Preflight scan for int conversion issues before import
    if FILE_NAME.lower().endswith('.csv'):
        with open(FILE_NAME, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)
            rows = list(reader)
            preflight_scan(headers, rows)
    elif FILE_NAME.lower().endswith('.xlsx'):
        df = pd.read_excel(FILE_NAME, engine='openpyxl')
        headers = list(df.columns)
        rows = df.values.tolist()
        preflight_scan(headers, rows)
    else:
        print("ERROR: Only .csv and .xlsx files are supported for preflight scan.")
        sys.exit(1)
    
    # Validate SQL connection  
    if not validate_sql_connection():
        sys.exit(1)
    print('Validation passed. Starting truncate and reload...')
   
    # Get connection info with token
    conn_info = create_connection_string_with_token()
    if not conn_info:
        print("ERROR: Could not establish Azure AD connection")
        sys.exit(1)
    conn_str, token_struct = conn_info

    # Connection/cursor factory for retry logic
    conn = pyodbc.connect(conn_str, attrs_before={1256: token_struct})
    def cursor_factory(reconnect=False):
        nonlocal conn
        if reconnect:
            try:
                conn.close()
            except Exception:
                pass
            # Re-acquire token for reconnect
            conn_info = create_connection_string_with_token()
            if not conn_info:
                raise Exception("Could not re-establish Azure AD connection")
            new_conn_str, new_token_struct = conn_info
            conn = pyodbc.connect(new_conn_str, attrs_before={1256: new_token_struct})
        return conn.cursor()

    try:
        cursor = cursor_factory()
        create_or_truncate_table(cursor)
        conn.commit()
        # Import data in batches
        batch_size = 2000  # Process 2000 rows at a time
        batch_rows = []
        total_rows = 0
        if FILE_NAME.lower().endswith('.csv'):
            with open(FILE_NAME, encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                headers = next(reader)  # Get header
                for row in reader:
                    batch_rows.append(row)
                    total_rows += 1
                    if len(batch_rows) >= batch_size:
                        batch_insert_with_retry(cursor_factory, headers, batch_rows)
                        print(f"Processed {total_rows} rows...")
                        batch_rows = []
                if batch_rows:
                    batch_insert_with_retry(cursor_factory, headers, batch_rows)
                    print(f"Processed final {len(batch_rows)} rows...")
        elif FILE_NAME.lower().endswith('.xlsx'):
            df = pd.read_excel(FILE_NAME, engine='openpyxl')
            headers = list(df.columns)
            rows = df.values.tolist()
            for row in rows:
                batch_rows.append([str(cell) if pd.notnull(cell) else '' for cell in row])
                total_rows += 1
                if len(batch_rows) >= batch_size:
                    batch_insert_with_retry(cursor_factory, headers, batch_rows)
                    print(f"Processed {total_rows} rows...")
                    batch_rows = []
            if batch_rows:
                batch_insert_with_retry(cursor_factory, headers, batch_rows)
                print(f"Processed final {len(batch_rows)} rows...")
        else:
            print("ERROR: Only .csv and .xlsx files are supported for import")
            sys.exit(1)
        print(f'Truncate and reload complete. Total rows loaded: {total_rows}')
    finally:
        conn.close()

if __name__ == '__main__':
    main()