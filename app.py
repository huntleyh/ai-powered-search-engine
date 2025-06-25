"""
app.py

FastAPI application with intelligent query routing using the orchestrator.
Routes user questions to appropriate search methods based on query analysis.
"""

import os

# Initialize tracing FIRST at application level
ENABLE_TRACING = os.environ.get("ENABLE_TRACING")
if ENABLE_TRACING and ENABLE_TRACING.lower() == "true":
    from tracing_setup import setup_tracing
    setup_tracing()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn

# Import the orchestrator AFTER tracing setup
from orchestrator_agent import process_query_with_routing, cleanup_orchestrator

app = FastAPI(
    title="Legal Search Engine API with Intelligent Routing",
    description="API for searching legal enforcement documents with intelligent query routing to optimal search methods",
    version="2.0.0"
)

# Shutdown event handler for cleanup
@app.on_event("shutdown")
async def shutdown_event():
    """
    Clean up resources when the FastAPI application shuts down.
    """
    print("🛑 Application shutting down, cleaning up resources...")
    try:
        cleanup_orchestrator()
        print("✅ Cleanup completed successfully")
    except Exception as e:
        print(f"❌ Error during shutdown cleanup: {e}")

# Request model
class ChatRequest(BaseModel):
    question: str
    thread_id: Optional[str] = None # Optional thread ID for conversation context

# Enhanced response models
class Document(BaseModel):
    id: str
    content: str
    title: str
    browser_file: str
    date_issued: str
    document_types: str
    settlement_amount: Any
    sanction_programs: str
    industries: str | None
    score: float

class QueryClassificationInfo(BaseModel):
    query_type: str
    confidence: float
    reasoning: str
    clarification_question: Optional[str] = None

class ChatResponse(BaseModel):
    question: str
    query_type: str
    classification: Optional[Dict[str, Any]] = None
    documents: List[Document] = []
    answer: str
    search_parameters: Optional[Dict[str, Any]] = None
    clarification_question: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Legal Search Engine API with Intelligent Routing is running",
        "version": "2.0.0",
        "features": [
            "Basic Keyword Search with Filters",
            "Advanced Document Search with RAG",
            "NL2SQL (placeholder)",
            "Intelligent Query Routing"
        ]
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Enhanced chat endpoint with intelligent query routing.
    
    Analyzes the user's question and routes it to the most appropriate search method:
    - Basic Keyword Search: For structured queries with specific filters
    - Advanced Document Search: For complex questions requiring semantic analysis  
    - NL2SQL: For statistical/aggregate questions (placeholder)
    - Clarification: When the query needs more context
    
    Args:
        request: ChatRequest containing the user's question
        
    Returns:
        ChatResponse with routing information and appropriate search results
    """
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        print(f"📝 Received question: {request.question}")
        
        # Use the orchestrator to process the query with intelligent routing
        result = await process_query_with_routing(request.question)
        
        # Prepare documents list (handle different result formats)
        documents = []
        if "documents" in result and result["documents"]:
            # Convert documents to the expected format
            for doc in result["documents"]:
                if isinstance(doc, dict):
                    # Ensure all required fields are present with defaults
                    document = Document(
                        id=doc.get("id", ""),
                        content=doc.get("content", ""),
                        title=doc.get("title", ""),
                        browser_file=doc.get("browser_file", ""),
                        date_issued=doc.get("date_issued", ""),
                        document_types=doc.get("document_types", ""),
                        settlement_amount=doc.get("settlement_amount", ""),
                        sanction_programs=doc.get("sanction_programs", ""),
                        industries=doc.get("industries", ""),
                        score=doc.get("score", 0.0)
                    )
                    documents.append(document)
        
        # Build the response
        response = ChatResponse(
            question=result["question"],
            query_type=result.get("query_type", "unknown"),
            classification=result.get("classification"),
            documents=documents,
            answer=result.get("answer", "No answer provided"),
            search_parameters=result.get("search_parameters"),
            clarification_question=result.get("clarification_question"),
            message=result.get("message"),
            error=result.get("error")
        )
        
        print(f"✅ Successfully processed query as: {response.query_type}")
        return response
        
    except Exception as e:
        print(f"❌ Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    return {
        "status": "healthy", 
        "service": "Legal Search Engine API",
        "version": "2.0.0",
        "routing_enabled": True
    }

@app.get("/query-types")
async def get_query_types():
    """
    Endpoint to get information about supported query types.
    Useful for frontend applications to show users what types of questions they can ask.
    """
    return {
        "supported_query_types": [
            {
                "type": "basic_search",
                "name": "Basic Keyword Search with Filters",
                "description": "For structured queries with specific criteria like date ranges, programs, industries, etc.",
                "examples": [
                    "Find OFAC violations related to Iran sanctions from 2020 to 2023",
                    "Show me voluntary disclosures in the financial services industry",
                    "Search for cases involving penalties over $1 million"
                ]
            },
            {
                "type": "advanced_search", 
                "name": "Advanced Document Search",
                "description": "For complex questions requiring semantic understanding and analysis of document content",
                "examples": [
                    "Can Iranian origin banknotes be imported into the U.S.?",
                    "What are the compliance requirements for financial institutions?",
                    "How does OFAC determine penalty amounts?"
                ]
            },
            {
                "type": "nl2sql",
                "name": "Statistical Queries",
                "description": "For aggregate questions and statistics (coming soon)",
                "examples": [
                    "How many violations were there in 2023?",
                    "What's the average penalty amount for financial institutions?",
                    "Which industry had the most violations?"
                ],
                "status": "placeholder"
            }
        ],
        "routing_info": {
            "automatic": True,
            "confidence_threshold": 0.5,
            "fallback_method": "advanced_search"
        }
    }

# Optional: Add endpoint for manual query classification (for debugging/testing)
@app.post("/classify")
async def classify_query_endpoint(request: ChatRequest):
    """
    Development endpoint to test query classification without executing search.
    Useful for debugging and understanding how queries are being classified.
    """
    try:
        from orchestrator_agent import classify_query
        
        classification = await classify_query(request.question)
        
        return {
            "question": request.question,
            "classification": {
                "query_type": classification.query_type.value,
                "confidence": classification.confidence,
                "reasoning": classification.reasoning,
                "clarification_question": classification.clarification_question
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

# Admin endpoint for manual cleanup (useful for testing)
@app.post("/admin/cleanup")
async def manual_cleanup():
    """
    Manually trigger cleanup of orchestrator resources.
    Useful for testing and administrative purposes.
    """
    try:
        print("🧹 Manual cleanup requested...")
        cleanup_orchestrator()
        return {
            "message": "Cleanup completed successfully",
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup error: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting Legal Search Engine API with Intelligent Routing...")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )