"""
app.py

FastAPI application with intelligent query routing using the orchestrator.
Routes user questions to appropriate search methods based on query analysis.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import os
import time
import asyncio


# Initialize tracing first
ENABLE_TRACING = os.environ.get("ENABLE_TRACING")
if ENABLE_TRACING and ENABLE_TRACING.lower() == "true":
    from tracing_setup import setup_tracing
    setup_tracing()


# Import the orchestrator
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
    Enhanced chat endpoint with intelligent query routing and session tracking.
    
    Analyzes the user's question and routes it to the most appropriate search method:
    - Basic Keyword Search: For structured queries with specific filters
    - Advanced Document Search: For complex questions requiring semantic analysis  
    - NL2SQL: For statistical/aggregate questions (placeholder)
    - Clarification: When the query needs more context
    
    Args:
        request: ChatRequest containing the user's question and optional thread_id
        
    Returns:
        ChatResponse with routing information and appropriate search results
    """
    session_id = None
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        print(f"📝 Received question: {request.question}")
        
        # Track session if thread_id is provided
        if request.thread_id:
            session_id = f"thread_{request.thread_id}"
            if session_id in active_sessions:
                active_sessions[session_id]["thread_id"] = request.thread_id
                active_sessions[session_id]["last_activity"] = time.time()
                print(f"🔗 Tracking session: {session_id}")
        
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
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}\n {str(e.__traceback__)}")

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

@app.post("/admin/terminate-session/{session_id}")
async def terminate_session_endpoint(session_id: str):
    """
    Manually terminate a specific user session and clean up its resources.
    
    Args:
        session_id: The session ID to terminate
        
    Returns:
        Success message or error
    """
    try:
        if session_id in active_sessions:
            await cleanup_user_session(session_id)
            return {"message": f"Session {session_id} terminated successfully"}
        else:
            return {"message": f"Session {session_id} not found or already terminated"}
    except Exception as e:
        print(f"❌ Error terminating session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to terminate session: {str(e)}")

@app.get("/admin/active-sessions")
async def get_active_sessions():
    """
    Get information about all currently active sessions.
    
    Returns:
        Dict of active sessions with their information
    """
    try:
        current_time = time.time()
        session_info = {}
        
        for session_id, info in active_sessions.items():
            session_info[session_id] = {
                "start_time": info["start_time"],
                "last_activity": info["last_activity"],
                "duration_seconds": current_time - info["start_time"],
                "inactive_seconds": current_time - info["last_activity"],
                "thread_id": info.get("thread_id"),
                "status": "active" if current_time - info["last_activity"] < 1800 else "inactive"  # 30 min threshold
            }
        
        return {
            "total_sessions": len(active_sessions),
            "sessions": session_info
        }
    except Exception as e:
        print(f"❌ Error getting active sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get active sessions: {str(e)}")

# Session management for tracking active user sessions
active_sessions = {}  # Store active user sessions/threads

@app.middleware("http")
async def session_tracking_middleware(request, call_next):
    """
    Middleware to track user sessions and detect when they end.
    """
    # Generate or extract session ID (you can use various methods)
    session_id = request.headers.get("X-Session-ID") or request.client.host
    
    # Track the session
    active_sessions[session_id] = {
        "start_time": time.time(),
        "last_activity": time.time(),
        "thread_id": None  # Will be set when agent threads are created
    }
    
    try:
        response = await call_next(request)
        # Update last activity
        if session_id in active_sessions:
            active_sessions[session_id]["last_activity"] = time.time()
        return response
    except Exception as e:
        # Session ended with error - cleanup
        await cleanup_user_session(session_id)
        raise
    finally:
        # Optional: cleanup inactive sessions periodically
        await cleanup_inactive_sessions()

async def cleanup_user_session(session_id: str):
    """
    Clean up resources for a specific user session.
    """
    if session_id in active_sessions:
        session_info = active_sessions[session_id]
        print(f"🧹 Cleaning up session: {session_id}")
        
        # Cleanup any agent threads associated with this session
        if session_info.get("thread_id"):
            try:
                # Add thread cleanup logic here
                print(f"🗑️ Cleaning up thread: {session_info['thread_id']}")
            except Exception as e:
                print(f"⚠️ Error cleaning up thread: {e}")
        
        # Remove from active sessions
        del active_sessions[session_id]
        print(f"✅ Session cleanup completed for: {session_id}")

async def cleanup_inactive_sessions(timeout_minutes: int = 30):
    """
    Clean up sessions that have been inactive for too long.
    """
    current_time = time.time()
    timeout_seconds = timeout_minutes * 60
    
    inactive_sessions = [
        session_id for session_id, info in active_sessions.items()
        if current_time - info["last_activity"] > timeout_seconds
    ]
    
    for session_id in inactive_sessions:
        print(f"⏰ Session {session_id} timed out, cleaning up...")
        await cleanup_user_session(session_id)

# Background task for periodic session cleanup
async def periodic_session_cleanup():
    """
    Background task that runs periodically to clean up inactive sessions.
    """
    while True:
        try:
            await cleanup_inactive_sessions(timeout_minutes=30)
            await asyncio.sleep(300)  # Run every 5 minutes
        except Exception as e:
            print(f"❌ Error in periodic session cleanup: {e}")
            await asyncio.sleep(60)  # Wait 1 minute before retrying

# Start background task when app starts
@app.on_event("startup")
async def startup_event():
    """
    Start background tasks when the application starts.
    """
    print("🚀 Starting background session cleanup task...")
    asyncio.create_task(periodic_session_cleanup())

if __name__ == "__main__":
    print("🚀 Starting Legal Search Engine API with Intelligent Routing...")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )