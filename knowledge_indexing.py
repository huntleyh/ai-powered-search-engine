# This script reads all rows from the Azure SQL database, generates embeddings for the main text fields, and uploads them to Azure AI Search, recreating the index each run

from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SimpleField,
    SearchFieldDataType,
    SearchableField,
    SearchField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
    SearchIndex
)
from datetime import datetime, timezone
import json
import hashlib
import struct
import sys
from typing import Any

from azure.core.credentials import AzureKeyCredential  
from azure.search.documents import SearchClient  
from datetime import datetime
import os  
from dotenv import load_dotenv  
from azure.core.credentials import AzureKeyCredential  
from azure.identity import DefaultAzureCredential

import openai
from openai import AzureOpenAI

import pyodbc

load_dotenv()

# Azure AI Search settings
ai_search_endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
ai_search_key = os.environ["AZURE_SEARCH_KEY"]
ai_search_index = os.environ["AZURE_SEARCH_INDEX"]

# Azure OpenAI settings
aoai_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
aoai_key = os.getenv("AZURE_OPENAI_API_KEY")
aoai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

# Azure SQL connection settings
conn_str_base = os.getenv('AZURE_SQL_CONNECTION_STRING')

# Table configuration - update these as needed
table_name = 'EnforcementActionsFull'

# Initialize Azure clients
search_index_client = SearchIndexClient(
    ai_search_endpoint, 
    AzureKeyCredential(ai_search_key)
)

search_client = SearchClient(
    ai_search_endpoint, 
    ai_search_index, 
    AzureKeyCredential(ai_search_key)
)

# Initialize Azure OpenAI client
openai_client = AzureOpenAI(
    api_key=aoai_key,
    azure_endpoint=aoai_endpoint,
    api_version="2024-02-15-preview"
)

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

def validate_table_exists():
    """Check if the source table exists and has data"""
    try:
        conn_info = create_connection_string_with_token()
        if not conn_info:
            return False
        
        conn_str, token_struct = conn_info
        conn = pyodbc.connect(conn_str, attrs_before={1256: token_struct})
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute(f"""
            SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_NAME = '{table_name}'
        """)
        table_exists = cursor.fetchone()[0] > 0
        
        if not table_exists:
            print(f"ERROR: Table '{table_name}' does not exist in the database")
            conn.close()
            return False
        
        # Count records
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        record_count = cursor.fetchone()[0]
        print(f"✓ Table '{table_name}' found with {record_count} records")
        
        conn.close()
        return record_count > 0
        
    except Exception as e:
        print(f"ERROR checking table: {e}")
        return False

def generate_embeddings(text, model=None):
    """Generate embeddings for given text"""
    if not text or not text.strip():
        # Return a zero vector if text is empty (3072 dimensions for text-embedding-3-large)
        return [0.0] * 3072
    try:
        deployment = model or aoai_deployment
        response = openai_client.embeddings.create(input=[text], model=deployment)
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding generation failed: {e}")
        return [0.0] * 3072

def create_index():
    """Create or recreate the Azure AI Search index"""
    # Always delete the index if it exists, to fully overwrite schema and data
    try:
        search_index_client.delete_index(ai_search_index)
        print(f"Deleted existing index: {ai_search_index}")
    except Exception as e:
        print(f"Index {ai_search_index} did not exist or could not be deleted: {e}")

    fields = [
        SimpleField(name="ID", type=SearchFieldDataType.String, key=True, filterable=True),
        SearchableField(name="BrowserFile", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="Title", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="Ordinal", type=SearchFieldDataType.Double, filterable=True, facetable=True, sortable=True),
        SimpleField(name="DateIssued", type=SearchFieldDataType.DateTimeOffset, filterable=True, facetable=True),
        SimpleField(name="Published", type=SearchFieldDataType.Boolean, filterable=True),
        SimpleField(name="DocumentTypes", type=SearchFieldDataType.String, filterable=True),
        # Embedding fields (vector search) - 3072 dimensions for text-embedding-3-large
        SearchField(
            name="KeyFactsVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=3072,
            vector_search_profile_name="myHnswProfile"
        ),
        SearchField(
            name="DocumentTextVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=3072,
            vector_search_profile_name="myHnswProfile"
        ),
        SearchField(
            name="CommentaryVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=3072,
            vector_search_profile_name="myHnswProfile"
        ),
        # Text fields for embedding
        SearchableField(name="KeyFacts", type=SearchFieldDataType.String),
        SearchableField(name="DocumentText", type=SearchFieldDataType.String),
        SearchableField(name="Commentary", type=SearchFieldDataType.String),
        SimpleField(name="NumberOfViolations", type=SearchFieldDataType.Int32, filterable=True, facetable=True),
        SimpleField(name="SettlementAmount", type=SearchFieldDataType.Double, filterable=True, facetable=True),
        SearchableField(name="OfacPenalty", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="AggregatePenalty", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="BasePenalty", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="StatutoryMaximum", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="VSD", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="Egregious", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="WillfulOrReckless", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="Criminal", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="RegulatoryProvisions", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="LegalIssues", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="SanctionPrograms", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="EnforcementCharacterizations", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="Industries", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="AggravatingFactors", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="MitigatingFactors", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="ReferenceCount", type=SearchFieldDataType.Int32, filterable=True, facetable=True)
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="myHnsw"
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="myHnswProfile",
                algorithm_configuration_name="myHnsw",
            )
        ]
    )

    index = SearchIndex(
        name=ai_search_index,
        fields=fields,
        vector_search=vector_search
    )
    result = search_index_client.create_or_update_index(index)
    print("✓ Index has been created")

def fetch_enforcement_actions():
    """Fetch all enforcement actions from Azure SQL using Azure AD authentication"""
    conn_info = create_connection_string_with_token()
    if not conn_info:
        raise Exception("Could not establish Azure AD connection")
    
    conn_str, token_struct = conn_info
    conn = pyodbc.connect(conn_str, attrs_before={1256: token_struct})
    
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {table_name}")
        columns = [column[0] for column in cursor.description]
        rows = cursor.fetchall()
        return [dict(zip(columns, row)) for row in rows]
    finally:
        conn.close()

def populate_index(batch_size=25):
    """Populate the search index with data from SQL database"""
    print("Populating index from SQL...")
    rows = fetch_enforcement_actions()
    print(f"Fetched {len(rows)} rows from SQL.")
    
    batch = []
    for i, row in enumerate(rows):
        # Convert ID to string for AI Search
        row["ID"] = str(row["ID"])
        
        # Generate embeddings for KeyFacts, DocumentText, Commentary
        print(f"Generating embeddings for row {i+1}/{len(rows)}...")
        for field, vec_field in [("KeyFacts", "KeyFactsVector"), ("DocumentText", "DocumentTextVector"), ("Commentary", "CommentaryVector")]:
            text = row.get(field) or ""
            row[vec_field] = generate_embeddings(text)
        
        batch.append(row)
        
        # Upload batch when it reaches batch_size or is the last row
        if len(batch) >= batch_size or i == len(rows)-1:
            try:
                search_client.upload_documents(documents=batch)
                print(f"✓ Uploaded batch {i+1-len(batch)+1} to {i+1}")
            except Exception as e:
                print(f"ERROR uploading batch ending at row {row['ID']}: {e}")
            batch = []
    
    print(f"✓ Index population complete. Processed {len(rows)} records.")

def main():
    """Main function that orchestrates the indexing process"""
    print("AZURE AI SEARCH INDEXING TOOL")
    print("=" * 50)
    
    print("Validating prerequisites...")
    
    # Validate SQL connection
    if not validate_sql_connection():
        print("❌ SQL validation failed. Please check your connection and authentication.")
        sys.exit(1)
    
    # Validate table exists and has data
    if not validate_table_exists():
        print("❌ Table validation failed. Please ensure the table exists and has data.")
        sys.exit(1)
    
    print("✓ All validations passed. Starting indexing process...")
    
    try:
        # Create the search index
        create_index()
        
        # Populate the index with data
        populate_index()
        
        print("\n🎉 Indexing process completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Indexing process failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()