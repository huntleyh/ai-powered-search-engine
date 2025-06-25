# AI-Powered Search Engine

A sophisticated legal document search engine powered by Azure AI Search, Azure OpenAI, and advanced RAG (Retrieval-Augmented Generation) capabilities. This system enables intelligent search across legal enforcement documents with AI-generated answers and comprehensive tracing.

## Features

- **Intelligent Query Routing**: Automatically analyzes user questions and routes them to the most appropriate search method using Azure AI Foundry agents
- **Advanced Search**: Hybrid semantic and vector search across multiple document fields (KeyFacts, DocumentText, Commentary)
- **Basic Keyword Search with Filters**: Structured search with specific criteria like date ranges, programs, industries, etc.
- **AI-Generated Answers**: Uses Azure OpenAI (o3-mini) to generate comprehensive answers based on search results
- **Query Classification**: Azure AI Foundry agents classify queries into basic search, advanced search, statistical queries, or clarification requests
- **FastAPI Web Interface**: RESTful API for integration with web applications with modern lifespan management
- **Command Line Interface**: Direct Python execution for testing and development
- **OpenTelemetry Tracing**: Full observability with Azure Monitor integration for monitoring AI operations
- **Robust Data Import**: Optimized scripts for importing CSV/Excel data to Azure SQL

## Architecture

- **Frontend**: FastAPI web framework with automatic API documentation and intelligent query routing
- **Query Classification**: Azure AI Foundry agents (GPT-4o) for intelligent query analysis and routing
- **Search Engine**: Azure AI Search with vector embeddings using text-embedding-3-large
- **AI Model**: Azure OpenAI o3-mini for answer generation and query processing
- **Database**: Azure SQL Server for data storage
- **Monitoring**: OpenTelemetry with Azure Monitor for tracing and observability

## Pre-reqs
- Create all Azure resources in South Central US region
- Services needed:
   - Azure SQL server and a database
   - Azure OpenAI
   - Azure AI search (standard tier)
   - Azure AI Foundry project for intelligent query routing
- Install **ODBC Driver 18 for SQL Server** on your local machine
  - Download from: https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server

## Setup

### 1. Azure OpenAI
- Need a text-embedding-3-large model deployed
- Need access to GPT-4o and o3-mini models for query classification and answer generation

### 2. Azure AI Foundry
- Create an Azure AI Foundry project
- Ensure you have access to create and manage agents
- Copy the project endpoint URL for configuration

### 3. Azure AI Search
- May need system-assigned identity turned on in your search service
- Grant the Azure identity (eg your user id) "Search Index Data Contributor" role on your Search Service. If using Azure AD authentication, you may also need to add the "Search Service Contributor" role.

### 4. Azure SQL Server
- **Settings > Microsoft Entra ID**: 
  - Set yourself as the Microsoft Entra admin
  - **Enable Microsoft Entra authentication** - The scripts use Azure AD authentication exclusively
- **Important**: Ensure your user account has appropriate database permissions for creating/modifying tables

### 5. Local Development Setup
1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

2. **Authenticate with Azure:**
   ```sh
   az login
   ```

3. **Environment Configuration:**
   - Copy `sample.env` to `.env` and fill in your Azure credentials and configuration values
   - Key environment variables needed:
     ```
     AZURE_SQL_CONNECTION_STRING=Driver={ODBC Driver 18 for SQL Server};Server=tcp:<your-server>.database.windows.net,1433;Database=<your-database>;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30
     AZURE_SEARCH_ENDPOINT=https://<your-search-service>.search.windows.net
     AZURE_SEARCH_KEY=<your-search-admin-key>
     AZURE_SEARCH_INDEX=<your-index-name>
     AZURE_OPENAI_ENDPOINT=https://<your-openai-service>.openai.azure.com/
     AZURE_FOUNDRY_PROJECT_ENDPOINT=<your-azure-foundry-project-endpoint-url> # Found in your Azure AI Foundry Project Overview page. 
     AZURE_OPENAI_API_KEY=<your-openai-key>
     ENABLE_TRACING=false
     AZURE_OPENAI_EMBEDDING_DEPLOYMENT=<your-embedding-model-deployment-name>
     AZURE_OPENAI_SIMPLE_DEPLOYMENT=<your-deployment-name>
     AZURE_OPENAI_REASONING_DEPLOYMENT=<your-deployment-name>
     APPLICATIONINSIGHTS_CONNECTION_STRING=<your-app-insights-connection-string>
     ```

## Usage

### Running the API Server
Start the FastAPI server for web-based access with intelligent query routing:
```sh
python app.py
```
The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`

### Direct Command Line Usage
Test individual components:

**Query Orchestrator with Intelligent Routing:**
```sh
python orchestrator_agent.py
```

**Advanced Document Search:**
```sh
python document_rag_agent.py
```

**Basic Search with Filters:**
```sh
python simple_search_agent.py
```

### API Endpoints
- **POST /chat**: Submit questions and receive AI-generated answers with intelligent routing to the most appropriate search method
- **POST /classify**: Development endpoint to test query classification without executing search
- **GET /query-types**: Information about supported query types and examples
- **POST /test-request**: Simple test endpoint to verify request model functionality
- **GET /health**: Health check endpoint
- **GET /**: Root endpoint with service status

### Query Types and Intelligent Routing

The system automatically analyzes your questions and routes them to the most appropriate search method:

#### **Basic Keyword Search with Filters**
For structured queries with specific criteria:
- Date ranges: "Find OFAC violations from 2020 to 2023"
- Programs: "Show me Iran sanctions cases"
- Industries: "Search financial services violations"
- Penalty amounts: "Cases with penalties over $1 million"

#### **Advanced Document Search**
For complex questions requiring semantic understanding:
- Legal interpretations: "Can Iranian origin banknotes be imported into the U.S.?"
- Compliance requirements: "What are OFAC compliance requirements?"
- Procedural questions: "How does OFAC determine penalty amounts?"

#### **Statistical Queries (Coming Soon)**
For aggregate questions and statistics:
- "How many violations were there in 2023?"
- "What's the average penalty amount for financial institutions?"
- "Which industry had the most violations?"

### Tracing and Monitoring
The application includes comprehensive OpenTelemetry tracing that captures:
- OpenAI API calls (inputs and outputs)
- Azure AI Foundry agent interactions
- Query classification and routing decisions
- Search operations
- Error tracking

Tracing is automatically initialized and configured for both API and command-line usage.

To see Tracing in Azure AI Foundry:
1. In your Azure portal, create an Azure Application Insights Resource and copy the connection string (can be found on the resource overview page)
2. Go to Tracing in your Azure AI Foundry Project and click Manage Data Source. Paste in the connection string.
3. Now, when you run your application, you will see tracing metrics in the Tracing tab. 
> **Warning:** There is a 32KB limit on traces. Anything above this limit will likely not show up in Foundry Tracing Portal.  
> [Learn more about trace size limits and possible workarounds.](https://learn.microsoft.com/en-us/answers/questions/543396/how-to-increase-the-size-of-messages-logged-in-app)

## Data Import Scripts

### CSV to Azure SQL Import
The project includes scripts for importing data:

1. **Prepare your data:**
   - Place your CSV file in the project directory
   - Update the `CSV_FILE` and `table_name` variables in the import script as needed

2. **Run the import:**
   ```sh
   python import_sql_data.py
   ```

**Features:**
- ✅ **Azure AD Authentication** - Uses your `az login` credentials (no passwords stored)
- ✅ **Batch Processing** - Processes 1000 rows at a time for optimal performance
- ✅ **Truncate & Reload** - Each run clears existing data and loads fresh data
- ✅ **Validation** - Checks CSV file and SQL connection before importing
- ✅ **Progress Tracking** - Shows import progress in real-time

### Generate Embeddings and Upload to AI Search
After importing data to SQL, generate embeddings and upload to Azure AI Search:

```sh
python <embedding_script_name>.py
```

This script:
- Reads data from the Azure SQL database
- Generates embeddings for text fields using Azure OpenAI
- Uploads the data with embeddings to Azure AI Search
- Recreates the search index on each run