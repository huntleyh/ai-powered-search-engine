"""
orchestrator_agent.py

Query routing orchestrator using Azure AI Foundry agents that analyzes user questions and routes them to the appropriate search method:
1. Basic Keyword Search with Filters (simple_search.py)
2. Advanced Document Search (document_rag.py)  
3. NL2SQL (placeholder)

This refactored version uses Azure AI Foundry agents for query classification instead of direct OpenAI calls.
"""

import os
import json
import asyncio
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel
from enum import Enum
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import AzureError

# Import your existing modules
from simple_search_agent import basic_search_agent, cleanup_simple_search_agent
from document_rag_agent import advanced_search, cleanup_rag_agent

# Load environment variables
load_dotenv()

# Azure AI Foundry configuration
AZURE_FOUNDRY_PROJECT_ENDPOINT = os.environ.get("AZURE_FOUNDRY_PROJECT_ENDPOINT")
if not AZURE_FOUNDRY_PROJECT_ENDPOINT:
    raise ValueError("AZURE_FOUNDRY_PROJECT_ENDPOINT environment variable is required")

classification_llm_deployment = os.environ.get("AZURE_OPENAI_SIMPLE_DEPLOYMENT")
reasoning_llm_deployment = os.environ.get("AZURE_OPENAI_REASONING_DEPLOYMENT")
print(f"classification_llm_deployment: {classification_llm_deployment}")
print(f"reasoning_llm_deployment: {reasoning_llm_deployment}")

class QueryType(str, Enum):
    """Enumeration of supported query types"""
    BASIC_SEARCH = "basic_search"
    ADVANCED_SEARCH = "advanced_search"
    NL2SQL = "nl2sql"
    CLARIFICATION_NEEDED = "clarification_needed"

class QueryClassification(BaseModel):
    """
    Pydantic model for query classification results.
    """
    query_type: QueryType
    confidence: float  # 0.0 to 1.0
    reasoning: str
    clarification_question: Optional[str] = None
    thread_id: Optional[str] = None

# Orchestrator prompt for query classification (same as original)
ORCHESTRATOR_PROMPT = """You are a query classification expert for a legal enforcement document search system. Your job is to analyze user questions and classify them into one of these categories:

## QUERY TYPES:

### 1. BASIC_SEARCH (basic_search)
Use for queries that can be converted into structured search filters. These typically involve:
- Specific date ranges ("from 2020 to 2023", "in 2022")
- Specific programs/sanctions ("Iran sanctions", "OFAC violations", "Cuba program")
- Specific document types ("voluntary disclosures", "enforcement actions")
- Specific industries ("financial services", "shipping")
- Specific penalty amounts or ranges ("over $1 million", "penalties above $500k")
- Specific respondent characteristics ("US companies", "foreign entities")

Examples:
- "Find OFAC violations related to Iran sanctions from 2020 to 2023"
- "Show me voluntary disclosures in the financial services industry"
- "Search for cases involving penalties over $1 million in 2022"
- "Find enforcement actions against shipping companies for Cuba sanctions"

### 2. ADVANCED_SEARCH (advanced_search)
Use for complex questions that require semantic understanding and analysis of document content:
- Questions about legal interpretations or implications
- Questions asking "what", "how", "why" that need content analysis
- Questions requiring synthesis across multiple documents
- Questions about specific legal concepts or procedures
- Questions that need expert commentary analysis

Examples:
- "Can Iranian origin banknotes be imported into the U.S.?"
- "What are the compliance requirements for financial institutions dealing with sanctioned entities?"
- "How does OFAC determine penalty amounts?"
- "What constitutes a voluntary disclosure under OFAC regulations?"

### 3. NL2SQL (nl2sql)
Use for statistical or aggregate questions that would be better answered by database queries:
- Questions asking for counts, totals, averages, or statistics
- Questions comparing numbers across different categories
- Questions about trends over time (statistical trends, not interpretive)
- Questions asking for rankings or top/bottom lists

Examples:
- "How many violations were there in 2023?"
- "What's the average penalty amount for financial institutions?"
- "Which industry had the most violations last year?"
- "Show me the top 10 largest penalties by amount"

### 4. CLARIFICATION_NEEDED (clarification_needed)
Use when the query is too vague, ambiguous, or lacks sufficient context:
- Very short or unclear questions
- Questions that could apply to multiple categories
- Questions missing key context (time periods, specific topics, etc.)

Examples:
- "Tell me about sanctions"
- "What happened?"
- "Search for violations"

## INSTRUCTIONS:
1. Classify the query into one of the 4 types above
2. Provide a confidence score (0.0 to 1.0) for your classification
3. Explain your reasoning in 1-2 sentences
4. If classification is CLARIFICATION_NEEDED, provide a specific clarification question

## CONFIDENCE GUIDELINES:
- 0.9-1.0: Very clear classification, obvious category
- 0.7-0.8: Clear classification with minor ambiguity  
- 0.5-0.6: Moderate confidence, could potentially fit multiple categories
- 0.0-0.4: Low confidence, ambiguous or unclear

Be decisive but honest about confidence levels. When in doubt between BASIC_SEARCH and ADVANCED_SEARCH, prefer ADVANCED_SEARCH for better user experience.

## OUTPUT FORMAT:
Respond with a JSON object containing the following fields with no additional text:
{
    "query_type": "one of: basic_search, advanced_search, nl2sql, clarification_needed",
    "confidence": 0.85,
    "reasoning": "Brief explanation of why this classification was chosen",
    "clarification_question": "Only include if query_type is clarification_needed"
}"""

class OrchestratorAgent:
    """
    Azure AI Foundry agent-based orchestrator for query classification and routing.
    """
    def __init__(self):
        """Initialize the orchestrator agent with Azure AI Foundry client."""
        try:
            print("🤖 Initializing OrchestratorAgent...")
            
            # Use DefaultAzureCredential for authentication
            print("🔑 Using DefaultAzureCredential authentication")
            self.credential = DefaultAzureCredential()
            
            # Initialize Azure AI Projects client
            self.ai_client = AIProjectClient(
                endpoint=AZURE_FOUNDRY_PROJECT_ENDPOINT,
                credential=self.credential
            )
                
            print("✅ Azure AI Foundry client initialized successfully")
            
            # Create or get the agent for query classification
            self.agent = self._create_or_get_agent()
            
            print("✅ OrchestratorAgent initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize OrchestratorAgent: {e}")
            raise

    def cleanup(self):
        """
        Clean up resources including the agent.
        Should be called during application shutdown.
        """
        try:
            print("🧹 Starting orchestrator cleanup...")
            # Clean up the RAG agent first
            try:
                print("🧹 Cleaning up RAG agent...")
                cleanup_rag_agent()
            except Exception as rag_error:
                print(f"⚠️ Warning: Failed to cleanup RAG agent: {rag_error}")
            
            # Clean up the simple search agent
            try:
                print("🧹 Cleaning up simple search agent...")
                cleanup_simple_search_agent()
            except Exception as simple_error:
                print(f"⚠️ Warning: Failed to cleanup simple search agent: {simple_error}")
            
            # Clean up the orchestrator agent
            if hasattr(self, 'agent') and self.agent:
                try:
                    print(f"🗑️ Cleaning up orchestrator agent: {self.agent.id}")
                    self.ai_client.agents.delete_agent(self.agent.id)
                except Exception as agent_error:
                    print(f"⚠️ Warning: Failed to cleanup orchestrator agent: {agent_error}")
            
            print("✅ Orchestrator cleanup completed successfully")
            
        except Exception as e:
            print(f"❌ Error during orchestrator cleanup: {e}")

    def _create_or_get_agent(self):
        """
        Create or retrieve the query classification agent.
        
        Returns:
            Agent instance for query classification
        """
        try:
            # Agent configuration
            agent_config = {
                "model": classification_llm_deployment,  # Use GPT-4 for better classification accuracy
                "name": "query-classifier",
                "description": "Legal document search query classification agent",
                "instructions": ORCHESTRATOR_PROMPT,
                "tools": [],  # No additional tools needed for classification
                "temperature": 0.1,  # Low temperature for consistent classification
            }
            
            # Create agent using Azure AI Foundry
            agent = self.ai_client.agents.create_agent(**agent_config)
            
            print(f"✅ Created agent: {agent.id}")
            return agent
            
        except AzureError as e:
            print(f"❌ Azure error creating agent: {e}")
            raise
        except Exception as e:
            print(f"❌ Error creating agent: {e}")
            raise
        
    async def _get_or_create_thread(self, thread_id: Optional[str] = None) -> str:
        """
        Get or create a thread for the agent to communicate in.
        
        Args:
            thread_id: Optional existing thread ID to use
            
        Returns:
            str: The thread ID to use for communication
        """
        try:
            if thread_id:
                # For now, just create a new thread since we don't have thread persistence
                # In a production system, you might want to implement thread persistence
                print(f"📋 Note: Thread persistence not implemented, creating new thread")
            
            # Create a new thread
            thread = self.ai_client.agents.threads.create()
            print(f"✅ Created new thread: {thread.id}")
            return thread.id
            
        except AzureError as e:
            print(f"❌ Azure error getting/creating thread: {e}")
            raise
        except Exception as e:
            print(f"❌ Error getting/creating thread: {e}")
            raise
    
    async def classify_query(self, user_question: str, thread_id: Optional[str] = None) -> QueryClassification:
        """
        Classify user query into appropriate search type using Azure AI Foundry agent.
        
        Args:
            user_question: The user's input question
            thread_id: Optional existing thread ID to use
        Returns:
            QueryClassification: Classification result with type, confidence, and reasoning
        """
        try:
            print(f"🤔 Analyzing query type for: '{user_question}'")

            # Get or create thread
            thread_id = await self._get_or_create_thread(thread_id)
            
            # Add the classification request message
            classification_request = f"Classify this query: {user_question}"
            
            # Add a message to the thread
            message = self.ai_client.agents.messages.create(
                thread_id=thread_id,
                role="user",  # Role of the message sender
                content=classification_request,  # Message content
            )
            
            # Run the agent
            run = self.ai_client.agents.runs.create_and_process(thread_id=thread_id, agent_id=self.agent.id)

            # Check if the run failed
            if run.status == "failed":
                raise Exception(f"Run failed. Please check the agent configuration and try again: {str(run.last_error)}")

            # Get messages in the response
            messages = self.ai_client.agents.messages.list(thread_id=thread_id)
            
            # Parse the latest assistant message
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
            
            # Parse JSON response
            try:
                classification_data = json.loads(response_content)
                classification = QueryClassification(**classification_data)
                classification.thread_id = thread_id 
            except (json.JSONDecodeError, ValueError) as e:
                print(f"⚠️ Error parsing agent response as JSON: {e}")
                print(f"Raw response: {response_content}")
                # Fallback classification
                classification = QueryClassification(
                    query_type=QueryType.ADVANCED_SEARCH,
                    confidence=0.5,
                    reasoning="Error parsing agent response, defaulting to advanced search",
                    thread_id=thread_id
                )
            
            print(f"📊 Classification: {classification.query_type.value} (confidence: {classification.confidence:.2f})")
            print(f"💭 Reasoning: {classification.reasoning}")
            
            return classification
            
        except AzureError as e:
            print(f"❌ Azure error during query classification: {e}")
            # Default to advanced search on Azure error
            return QueryClassification(
                query_type=QueryType.ADVANCED_SEARCH,
                confidence=0.5,
                reasoning=f"Azure error occurred during classification: {str(e)}, defaulting to advanced search"
            )
        except Exception as e:
            print(f"❌ Error classifying query: {e}")
            # Default to advanced search on error
            return QueryClassification(
                query_type=QueryType.ADVANCED_SEARCH,
                confidence=0.5,
                reasoning="Error occurred during classification, defaulting to advanced search"
            )
    
    def nl2sql_placeholder(self, user_question: str) -> Dict[str, Any]:
        """
        Placeholder function for NL2SQL functionality.
        This would eventually convert natural language questions to SQL queries
        and execute them against a structured database.
        
        Args:
            user_question: The user's question
            
        Returns:
            Dict with placeholder response
        """
        print("🔧 NL2SQL functionality is not yet implemented")
        
        return {
            "question": user_question,
            "query_type": "nl2sql",
            "status": "not_implemented",
            "message": "Statistical and aggregate queries are not yet supported. This feature is coming soon!",
            "suggested_alternative": "Try rephrasing your question to search for specific documents or cases instead.",
            "documents": [],
            "answer": "I apologize, but I cannot process statistical queries yet. This feature is under development. Please try asking about specific documents, cases, or legal concepts instead."
        }
    
    async def process_query_with_routing(self, user_question: str) -> Dict[str, Any]:
        """
        Main orchestrator function that analyzes the query and routes to appropriate search method.
        
        Args:
            user_question: The user's input question
            
        Returns:
            Dict: Response from the selected search method, enhanced with routing metadata
        """
        print("🚀 Starting query orchestration...")
        print("="*60)
        
        # Step 1: Classify the query using Azure AI Foundry agent
        classification = await self.classify_query(user_question)
        
        # Step 2: Route to appropriate handler based on classification
        try:
            if classification.query_type == QueryType.CLARIFICATION_NEEDED:
                return {
                    "question": user_question,
                    "query_type": "clarification_needed",
                    "classification": classification.dict(),
                    "clarification_question": classification.clarification_question,
                    "message": "I need more information to help you effectively.",
                    "documents": [],
                    "answer": f"I need clarification to provide the best results. {classification.clarification_question}"
                }
                
            elif classification.query_type == QueryType.BASIC_SEARCH:
                print("📋 Routing to Basic Keyword Search with Filters...")
                result = await basic_search_agent(user_question)
                
                # Transform basic search result to match expected format
                return {
                    "question": user_question,
                    "query_type": "basic_search",
                    "classification": classification.dict(),
                    "search_parameters": result,
                    "documents": [],  # Basic search returns parameters, not documents
                    "answer": f"I've processed your query into structured search parameters. The system would search for documents matching these criteria: {json.dumps(result, indent=2)}"
                }
                
            elif classification.query_type == QueryType.ADVANCED_SEARCH:
                print("🔍 Routing to Advanced Document Search...")
                result = await advanced_search(user_question)
                
                # Enhance result with classification metadata
                result["query_type"] = "advanced_search"
                result["classification"] = classification.dict()
                return result
                
            elif classification.query_type == QueryType.NL2SQL:
                print("📊 Routing to NL2SQL...")
                result = self.nl2sql_placeholder(user_question)
                result["classification"] = classification.dict()
                return result
                
            else:
                # Fallback to advanced search
                print("⚠️ Unknown classification, defaulting to Advanced Document Search...")
                result = await advanced_search(user_question)
                result["query_type"] = "advanced_search_fallback"
                result["classification"] = classification.dict()
                return result
                
        except Exception as e:
            print(f"❌ Error during query processing: {e}")
            return {
                "question": user_question,
                "query_type": "error",
                "error": str(e),
                "message": "An error occurred while processing your query.",
                "documents": [],
                "answer": "I apologize, but I encountered an error while processing your question. Please try rephrasing your query."
            }

# Global orchestrator instance
_orchestrator_instance = None

def get_orchestrator() -> OrchestratorAgent:
    """
    Get or create the global orchestrator instance.
    
    Returns:
        OrchestratorAgent: The global orchestrator instance
    """
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = OrchestratorAgent()
    return _orchestrator_instance

def cleanup_orchestrator():
    """
    Clean up the global orchestrator instance.
    Should be called during application shutdown.
    """
    global _orchestrator_instance
    if _orchestrator_instance is not None:
        _orchestrator_instance.cleanup()
        _orchestrator_instance = None

async def classify_query(user_question: str) -> QueryClassification:
    """
    Convenience function for query classification using the global orchestrator.
    
    Args:
        user_question: The user's input question
        
    Returns:
        QueryClassification: Classification result
    """
    orchestrator = get_orchestrator()
    return await orchestrator.classify_query(user_question)

async def process_query_with_routing(user_question: str) -> Dict[str, Any]:
    """
    Convenience function for query processing using the global orchestrator.
    
    Args:
        user_question: The user's input question
        
    Returns:
        Dict: Response from the selected search method
    """
    orchestrator = get_orchestrator()
    return await orchestrator.process_query_with_routing(user_question)

async def example_usage():
    """Example usage showing different query types"""
    
    example_queries = [
        # Basic search examples
        "Find OFAC violations related to Iran sanctions from 2020 to 2023",
        "Show me voluntary disclosures in the financial services industry",
        
        # Advanced search examples  
        "Can Iranian origin banknotes be imported into the U.S.?",
        "What are the compliance requirements for financial institutions?",
        
        # NL2SQL examples
        "How many violations were there in 2023?",
        "What's the average penalty amount for financial institutions?",
        
        # Clarification needed examples
        "Tell me about sanctions",
        "What happened?"
    ]
    
    orchestrator = get_orchestrator()
    
    for query in example_queries:
        print("\n" + "="*80)
        print(f"Example Query: {query}")
        print("="*80)
        
        result = await orchestrator.process_query_with_routing(query)
        
        print("\n📋 Result Summary:")
        print(f"Query Type: {result.get('query_type', 'unknown')}")
        if 'classification' in result:
            print(f"Confidence: {result['classification']['confidence']:.2f}")
            print(f"Reasoning: {result['classification']['reasoning']}")

async def interactive_mode():
    """Interactive mode for testing the orchestrator"""
    print("\n" + "="*60) 
    print("💬 Interactive Mode")
    print("="*60)
    
    orchestrator = get_orchestrator()
    
    try:
        while True:
            user_input = input("\n🔍 Enter your question (or 'quit' to exit): ")
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
                
            result = await orchestrator.process_query_with_routing(user_input)
            
            print(f"\n📊 Query Type: {result.get('query_type', 'unknown')}")
            if 'classification' in result:
                print(f"🎯 Confidence: {result['classification']['confidence']:.2f}")
            
            print(f"\n💬 Answer:")
            print(result.get('answer', 'No answer available'))
            
            if result.get('query_type') == 'clarification_needed':
                print(f"\n❓ Clarification needed: {result.get('clarification_question', '')}")
                
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error in interactive mode: {e}")
    finally:
        # Always cleanup
        cleanup_orchestrator()

async def main():
    """Main async function"""
    print("🎛️ Query Orchestrator Agent - Intelligent Routing with Azure AI Foundry")
    print("="*60)
    
    # Initialize tracing when running directly (for standalone testing)
    from tracing_setup import setup_tracing
    setup_tracing()
    
    try:
        # Initialize orchestrator
        orchestrator = get_orchestrator()
        
        # Run examples (uncomment if desired)
        # await example_usage()
        
        # Run interactive mode
        await interactive_mode()
        
    except Exception as e:
        print(f"❌ Failed to initialize orchestrator: {e}")
        print("Please ensure AZURE_FOUNDRY_PROJECT_ENDPOINT is set correctly.")
    
    print("👋 Goodbye!")

if __name__ == "__main__":
    # Run the main async function
    asyncio.run(main())