"""
document_rag_agent.py

Azure AI Foundry agent-based document search with RAG functionality.
Uses Azure AI Foundry agents with the o3-mini model for generating answers from search results.
"""

import os
import json
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import AzureError
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from langchain_openai import AzureOpenAIEmbeddings

# Load environment variables
load_dotenv()

# Azure AI Foundry configuration
AZURE_FOUNDRY_PROJECT_ENDPOINT = os.environ.get("AZURE_FOUNDRY_PROJECT_ENDPOINT")
if not AZURE_FOUNDRY_PROJECT_ENDPOINT:
    raise ValueError("AZURE_FOUNDRY_PROJECT_ENDPOINT environment variable is required")

# Azure Search configuration
AZURE_SEARCH_ENDPOINT = os.environ.get("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.environ.get("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX")

if not all([AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY, AZURE_SEARCH_INDEX]):
    raise ValueError("Azure Search environment variables are required")

# Azure OpenAI configuration for embeddings
AOAI_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AOAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AOAI_EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
AOAI_REASONING_DEPLOYMENT = os.environ.get("AZURE_OPENAI_REASONING_DEPLOYMENT")

if not all([AOAI_KEY, AOAI_ENDPOINT]):
    raise ValueError("Azure OpenAI environment variables are required")

# Configuration
NUM_SEARCH_RESULTS = 15
K_NEAREST_NEIGHBORS = 30

# RAG System prompt for the Azure AI Foundry agent (from document_rag.py)
RAG_SYSTEM_PROMPT = """Review the provided documents and commentary to answer the user's question.

###Guidance###

1. From the list of provided documents, list out which are relevant to the user's question.
2. For each relevant document, explain how it addresses the user's question. Make sure to cite the document title and put the title in brackets. Always refer to the documents by [title], not by number.
3. If the commentary is relevant to the user's question, explain how it addresses the user's question. 
4. If there is no relevant information in the documents or commentary, say that you couldn't find any relevant information to answer the question. Under no circumstances should you answer with anything outside of the context of the search results. This is a legal search engine AI, accuracy is paramount. Do not make assumptions or inferences.
5. For each document, include the ReferenceCount field (number of times the document is cited in the commentary) in your answer when describing the document. When determining the importance or relevance of a document, prioritize those with higher ReferenceCount values, as discussed by Brian. If you use information from a document, mention its ReferenceCount value in your explanation.

###Output Format###

- Always start your answer by identifying which documents you're referencing (e.g., "According to [Document Title]..."). 
- When referencing information, clearly indicate which document it came from
- Use the document titles provided in the TITLE sections to identify sources
- If information comes from multiple documents, mention all relevant sources
- Be specific about which document contains which information
- Summarize the expert commentary at the end if relevant to the user's question.
- When describing a document, include its ReferenceCount value (e.g., "[Document Title] (ReferenceCount: 12)")
- When ranking or prioritizing documents, consider those with higher ReferenceCount as more important or relevant.

###Examples###

User: can iranian origin banknotes be imported into the U.S?
Assistant: According to [Document Title] (ReferenceCount: 12), Iranian origin banknotes cannot be imported into the U.S. This is backed up by supporting information in [Document Title 2] (ReferenceCount: 8). According to expert commentary, Iranian origin banknotes would require explicit authorization from OFAC."""

class DocumentRAGAgent:
    """
    Azure AI Foundry agent for document search and RAG-based answer generation.
    """
    
    def __init__(self):
        """Initialize the document RAG agent with Azure AI Foundry and search clients."""
        try:
            print("🔑 Initializing Document RAG Agent...")
            
            # Initialize Azure AI Foundry client with DefaultAzureCredential
            self.credential = DefaultAzureCredential()
            self.ai_client = AIProjectClient(
                endpoint=AZURE_FOUNDRY_PROJECT_ENDPOINT,
                credential=self.credential
            )
            
            # Initialize Azure Search client
            self.search_client = SearchClient(
                AZURE_SEARCH_ENDPOINT, 
                AZURE_SEARCH_INDEX, 
                AzureKeyCredential(AZURE_SEARCH_KEY)
            )
            
            # Initialize embeddings model
            self.embeddings_model = AzureOpenAIEmbeddings(
                azure_deployment=AOAI_EMBEDDING_DEPLOYMENT,
                api_key=AOAI_KEY,
                azure_endpoint=AOAI_ENDPOINT
            )
            
            # Create the RAG agent
            self.agent = self._create_rag_agent()
            
            print("✅ Document RAG Agent initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize Document RAG Agent: {e}")
            raise
    
    def _create_rag_agent(self):
        """
        Create the Azure AI Foundry agent for RAG operations.
        
        Returns:
            Agent instance for document analysis and answer generation
        """
        try:
            # Agent configuration for o3-mini model
            agent_config = {
                "model": AOAI_REASONING_DEPLOYMENT,  # Use o3-mini as specified
                "name": "document-rag-agent",
                "description": "Legal document analysis and RAG-based question answering agent",
                "instructions": RAG_SYSTEM_PROMPT,
                "tools": [],  # No additional tools needed
                #"temperature": 0.1,  # Low temperature for consistent, accurate responses
            }
            
            # Create agent using Azure AI Foundry
            agent = self.ai_client.agents.create_agent(**agent_config)
            
            print(f"✅ Created RAG agent: {agent.id}")
            return agent
            
        except AzureError as e:
            print(f"❌ Azure error creating RAG agent: {e}")
            raise
        except Exception as e:
            print(f"❌ Error creating RAG agent: {e}")
            raise
    
    def run_search(self, search_query: str) -> List[Dict[str, Any]]:
        """
        Perform a search using Azure Cognitive Search with both semantic and vector queries.
        Searches across KeyFacts, DocumentText, and Commentary vector fields.
        
        Args:
            search_query: The user's search query
            
        Returns:
            List of search results with combined content
        """
        try:
            print(f"🔍 Running search for: '{search_query}'")
            
            # Generate vector embedding for the query
            query_vector = self.embeddings_model.embed_query(search_query)
            
            # Create vector queries for all three vector fields
            vector_queries = [
                VectorizedQuery(
                    vector=query_vector,
                    k_nearest_neighbors=K_NEAREST_NEIGHBORS,
                    fields="KeyFactsVector"
                ),
                VectorizedQuery(
                    vector=query_vector,
                    k_nearest_neighbors=K_NEAREST_NEIGHBORS,
                    fields="DocumentTextVector"
                ),
                VectorizedQuery(
                    vector=query_vector,
                    k_nearest_neighbors=K_NEAREST_NEIGHBORS,
                    fields="CommentaryVector"
                )
            ]
            
            # Perform the search with all vector fields and corresponding text fields
            results = self.search_client.search(
                search_text=search_query,
                vector_queries=vector_queries,
                select=["ID", "BrowserFile", "Title", "KeyFacts", "DocumentText", "Commentary", 
                        "DateIssued", "Published", "DocumentTypes", "NumberOfViolations", 
                        "SettlementAmount", "SanctionPrograms", "Industries", "ReferenceCount"],
                top=NUM_SEARCH_RESULTS
            )
            
            search_results = []
            for result in results:
                # Combine all text content for the LLM with clear delineation
                content_parts = []
                
                # Always include title at the top
                if result.get("Title"):
                    content_parts.append(f"=== TITLE ===\n{result['Title']}\n=== END TITLE ===")
                
                if result.get("KeyFacts"):
                    content_parts.append(f"=== KEY FACTS ===\n{result['KeyFacts']}\n=== END KEY FACTS ===")
                
                if result.get("DocumentText"):
                    content_parts.append(f"=== DOCUMENT TEXT ===\n{result['DocumentText']}\n=== END DOCUMENT TEXT ===")
                
                if result.get("Commentary"):
                    content_parts.append(f"=== COMMENTARY ===\n{result['Commentary']}\n=== END COMMENTARY ===")
                
                # Add ReferenceCount to the content for prompt context
                if result.get("ReferenceCount") is not None:
                    content_parts.append(f"=== REFERENCE COUNT ===\n{result['ReferenceCount']}\n=== END REFERENCE COUNT ===")
                
                combined_content = "\n\n".join(content_parts)
                
                search_result = {
                    "id": result["ID"],
                    "content": combined_content,
                    "title": result.get("Title", ""),
                    "browser_file": result.get("BrowserFile", ""),
                    "date_issued": result.get("DateIssued", ""),
                    "document_types": result.get("DocumentTypes", ""),
                    "settlement_amount": result.get("SettlementAmount", ""),
                    "sanction_programs": result.get("SanctionPrograms", ""),
                    "industries": result.get("Industries", ""),
                    "reference_count": result.get("ReferenceCount", None),
                    "score": result["@search.score"]
                }
                search_results.append(search_result)
            
            print(f"✅ Found {len(search_results)} search results")
            return search_results
            
        except Exception as e:
            print(f"❌ Error during search: {e}")
            raise
    
    async def generate_answer(self, user_question: str, search_results: List[Dict[str, Any]]) -> str:
        """
        Generate an answer using Azure AI Foundry agent and search results.
        
        Args:
            user_question: The user's question
            search_results: List of search results from Azure Search
            
        Returns:
            Generated answer string
        """
        try:
            print(f"🤖 Generating answer for: '{user_question}'")
            
            # Create a thread for this conversation
            thread = self.ai_client.agents.threads.create()
            
            # Format search results for the agent
            formatted_results = []
            for i, result in enumerate(search_results, 1):
                formatted_results.append(f"DOCUMENT {i}:\n{result['content']}")
              # Create the user message with the exact format from document_rag.py
            user_message = f"""Create a comprehensive answer to the user's question using these search results.

User Question: {user_question}

Search Results:
{chr(10).join(formatted_results)}

Synthesize these results into a clear, complete answer. Remember to cite which documents contain the information you're referencing."""
            
            # Add message to thread
            self.ai_client.agents.messages.create(
                thread_id=thread.id,
                role="user",
                content=user_message
            )
            
            # Run the agent
            run = self.ai_client.agents.runs.create_and_process(
                thread_id=thread.id, 
                agent_id=self.agent.id
            )
            
            # Check if the run failed
            if run.status == "failed":
                raise Exception(f"Agent run failed: {run.last_error}")
            
            # Get the response messages
            messages = self.ai_client.agents.messages.list(thread_id=thread.id)
            
            # Find the latest assistant message
            assistant_message = None
            for message in messages:
                if message.role == "assistant":
                    assistant_message = message
                    break
            
            if not assistant_message:
                raise Exception("No response from agent")
            
            # Extract content from the message
            response_content = ""
            for content_item in assistant_message.content:
                if hasattr(content_item, 'text'):
                    response_content += content_item.text.value
            
            # Clean up thread (optional - could be kept for conversation history)
            try:
                self.ai_client.agents.threads.delete(thread_id=thread.id)
            except Exception as cleanup_error:
                print(f"⚠️ Warning: Failed to cleanup thread: {cleanup_error}")
            
            print("✅ Answer generated successfully")
            return response_content
            
        except AzureError as e:
            print(f"❌ Azure error during answer generation: {e}")
            raise
        except Exception as e:
            print(f"❌ Error during answer generation: {e}")
            raise
    
    async def advanced_search(self, question: str) -> Dict[str, Any]:
        """
        Main function for advanced document search with RAG using Azure AI Foundry agents.
        
        Args:
            question: The user's question
            
        Returns:
            dict: Contains the question, documents, and agent-generated answer
        """
        try:
            print(f"🔍 Starting advanced search with AI agent for: '{question}'")
            
            # Step 1: Perform semantic search
            documents = self.run_search(question)
            
            # Step 2: Generate answer using Azure AI Foundry agent
            answer = await self.generate_answer(question, documents)
            
            print(f"✅ Advanced search completed - found {len(documents)} documents")
            
            return {
                "question": question,
                "documents": documents,
                "answer": answer
            }
            
        except Exception as e:
            print(f"❌ Error in advanced search: {e}")
            raise
    
    def cleanup(self):
        """
        Clean up resources including the agent.
        Should be called during application shutdown.
        """
        try:
            print("🧹 Cleaning up Document RAG Agent...")
            
            # Clean up the agent
            if hasattr(self, 'agent') and self.agent:
                try:
                    print(f"🗑️ Cleaning up RAG agent: {self.agent.id}")
                    self.ai_client.agents.delete_agent(self.agent.id)
                except Exception as agent_error:
                    print(f"⚠️ Warning: Failed to cleanup RAG agent: {agent_error}")
            
            print("✅ Document RAG Agent cleanup completed")
            
        except Exception as e:
            print(f"❌ Error during Document RAG Agent cleanup: {e}")

# Global instance management
_rag_agent_instance = None

def get_rag_agent() -> DocumentRAGAgent:
    """
    Get or create the global DocumentRAGAgent instance.
    
    Returns:
        DocumentRAGAgent: The global RAG agent instance
    """
    global _rag_agent_instance
    if _rag_agent_instance is None:
        _rag_agent_instance = DocumentRAGAgent()
    return _rag_agent_instance

def cleanup_rag_agent():
    """
    Clean up the global DocumentRAGAgent instance.
    Should be called during application shutdown.
    """
    global _rag_agent_instance
    if _rag_agent_instance is not None:
        _rag_agent_instance.cleanup()
        _rag_agent_instance = None

# Convenience function to maintain compatibility with existing code
async def advanced_search(question: str) -> Dict[str, Any]:
    """
    Convenience function for advanced document search using Azure AI Foundry agents.
    Maintains compatibility with existing code that imports this function.
    
    Args:
        question: The user's question
        
    Returns:
        dict: Contains the question, documents, and answer
    """
    rag_agent = get_rag_agent()
    return await rag_agent.advanced_search(question)

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def main():
        """Example usage of the Document RAG Agent"""
        # Initialize tracing when running directly (for standalone testing)
        from tracing_setup import setup_tracing
        setup_tracing()
        
        # Get user question
        user_question = input("Enter your question: ")
        
        try:
            # Run advanced search with AI agent
            result = await advanced_search(user_question)
            
            print("\n" + "="*80)
            print("ADVANCED SEARCH AND ANSWER RESULT (AI AGENT)")
            print("="*80)
            
            print(f"\nQuestion: {result['question']}")
            
            print(f"\nDocuments Found ({len(result['documents'])}):")
            for i, doc in enumerate(result['documents'], 1):
                print(f"{i}. {doc['title']}")
            
            print(f"\nAI Agent Answer:")
            print(result['answer'])
            
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            # Clean up resources
            cleanup_rag_agent()
    
    # Run the async main function
    asyncio.run(main())