"""
simple_search_agent.py

Agent-based basic search with structured outputs for search query processing using Azure AI Foundry agents.
This refactored version uses Azure AI Foundry agents instead of direct OpenAI calls to parse user queries 
into search parameters.

Replaces the direct OpenAI calls in simple_search.py with Azure AI Foundry agent calls while preserving
the original prompt, logic, and structured output format.
"""

import os
import json
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import AzureError
from prompts import simple_search_prompt

# Load environment variables
load_dotenv()

# Azure AI Foundry configuration
AZURE_FOUNDRY_PROJECT_ENDPOINT = os.environ.get("AZURE_FOUNDRY_PROJECT_ENDPOINT")
if not AZURE_FOUNDRY_PROJECT_ENDPOINT:
    raise ValueError("AZURE_FOUNDRY_PROJECT_ENDPOINT environment variable is required")

AOAI_SIMPLE_DEPLOYMENT = os.environ.get("AZURE_OPENAI_SIMPLE_DEPLOYMENT")

class SearchParameters(BaseModel):
    """
    Pydantic model for structured search parameters.
    """
    DateIssuedBegin: Optional[int] = None
    DateIssuedEnd: Optional[int] = None
    LegalIssue: List[str] = []
    Program: List[str] = []
    DocumentType: List[str] = []
    RegulatoryProvision: List[str] = []
    Published: Optional[bool] = None
    EnforcementCharacterization: List[str] = []
    NumberOfViolationsLow: Optional[int] = None
    NumberOfViolationsHigh: Optional[int] = None
    OFACPenalty: List[str] = []
    AggregatePenalty: List[str] = []
    Industry: List[str] = []
    RespondentNationality: List[str] = []
    VoluntaryDisclosure: List[str] = []
    EgregiousCase: List[str] = []
    KeyWords: str = ""
    ExcludeCommentaries: bool = False

class SimpleSearchAgent:
    """
    Azure AI Foundry agent-based simple search handler for structured query parsing.
    """
    def __init__(self):
        """Initialize the simple search agent with Azure AI Foundry client."""
        try:
            # Use DefaultAzureCredential for authentication
            print("🔑 Simple Search Agent: Using DefaultAzureCredential authentication")
            self.credential = DefaultAzureCredential()
            
            # Initialize Azure AI Projects client
            self.ai_client = AIProjectClient(
                endpoint=AZURE_FOUNDRY_PROJECT_ENDPOINT,
                credential=self.credential
            )
                
            print("✅ Simple Search Agent: Azure AI Foundry client initialized successfully")
            
            # Create or get the agent for structured output parsing
            self.agent = self._create_or_get_agent()
            
            print("✅ SimpleSearchAgent initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize SimpleSearchAgent: {e}")
            raise

    def cleanup(self):
        """
        Clean up resources including the agent.
        Should be called during application shutdown.
        """
        try:
            print("🧹 Starting simple search agent cleanup...")
            
            # Clean up the simple search agent
            if hasattr(self, 'agent') and self.agent:
                try:
                    print(f"🗑️ Cleaning up simple search agent: {self.agent.id}")
                    self.ai_client.agents.delete_agent(self.agent.id)
                except Exception as agent_error:
                    print(f"⚠️ Warning: Failed to cleanup simple search agent: {agent_error}")
            
            print("✅ Simple search agent cleanup completed successfully")
            
        except Exception as e:
            print(f"❌ Error during simple search agent cleanup: {e}")
    
    def _create_or_get_agent(self):
        """
        Create or retrieve the structured output parsing agent.
        
        Returns:
            Agent instance for structured output parsing
        """
        try:
            # Enhanced instructions for structured JSON output
            enhanced_instructions = f"""{simple_search_prompt}

CRITICAL: You must respond with ONLY a valid JSON object that matches the SearchParameters schema exactly. 
Do not include any explanatory text, markdown formatting, or code blocks. 
Return only the raw JSON object with these exact field names:

{{
    "DateIssuedBegin": null or integer,
    "DateIssuedEnd": null or integer,
    "LegalIssue": [],
    "Program": [],
    "DocumentType": [],
    "RegulatoryProvision": [],
    "Published": null or boolean,
    "EnforcementCharacterization": [],
    "NumberOfViolationsLow": null or integer,
    "NumberOfViolationsHigh": null or integer,
    "OFACPenalty": [],
    "AggregatePenalty": [],
    "Industry": [],
    "RespondentNationality": [],
    "VoluntaryDisclosure": [],
    "EgregiousCase": [],
    "KeyWords": "",
    "ExcludeCommentaries": false
}}

Ensure all array fields are arrays (even if empty) and all field names match exactly."""
            
            # Agent configuration
            agent_config = {
                "model": AOAI_SIMPLE_DEPLOYMENT,  # Use GPT-4 for better parsing accuracy
                "name": "simple-search-parser",
                "description": "Legal document search query parser for structured JSON outputs",
                "instructions": enhanced_instructions,
                "tools": [],  # No additional tools needed for parsing
                "temperature": 0.1,  # Low temperature for consistent parsing
            }
            
            # Create agent using Azure AI Foundry
            agent = self.ai_client.agents.create_agent(**agent_config)
            
            print(f"✅ Created simple search agent: {agent.id}")
            return agent
            
        except AzureError as e:
            print(f"❌ Azure error creating simple search agent: {e}")
            raise
        except Exception as e:
            print(f"❌ Error creating simple search agent: {e}")
            raise

    async def user_query_to_structured_outputs(self, user_input: str) -> Optional[SearchParameters]:
        """
        Convert user query to structured outputs using Azure AI Foundry agent.
        
        Parameters:
        - user_input (str): The raw user query
        
        Returns:
        - Optional[SearchParameters]: Structured output or None if error
        """
        thread_id = None
        try:
            print(f"Step 2: 🔄 Converting user query to structured outputs...")
            
            # Create a new thread for this request
            thread = self.ai_client.agents.threads.create()
            thread_id = thread.id
            print(f"✅ Created thread: {thread_id}")
            
            # Add the user query message to the thread
            message = self.ai_client.agents.messages.create(
                thread_id=thread_id,
                role="user",
                content=user_input,
            )
            
            # Run the agent to process the query
            run = self.ai_client.agents.runs.create_and_process(
                thread_id=thread_id, 
                agent_id=self.agent.id
            )

            # Check if the run failed
            if run.status == "failed":
                raise Exception(f"Agent run failed: {str(run.last_error)}")

            # Get messages from the thread
            messages = self.ai_client.agents.messages.list(thread_id=thread_id)
            
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
              # Parse the response as JSON to extract SearchParameters
            try:
                # Clean the response content - remove any markdown formatting
                clean_content = response_content.strip()
                
                # Remove markdown code blocks if present
                if clean_content.startswith('```json'):
                    clean_content = clean_content.replace('```json', '').replace('```', '').strip()
                elif clean_content.startswith('```'):
                    clean_content = clean_content.replace('```', '').strip()
                
                # Try to parse as JSON directly
                response_json = json.loads(clean_content)
                structured_output = SearchParameters(**response_json)
                
            except (json.JSONDecodeError, ValueError) as json_error:
                print(f"⚠️ Direct JSON parsing failed: {json_error}")
                
                # Fallback: Extract JSON object from text using regex
                try:
                    import re
                    # Look for JSON object pattern
                    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                    json_matches = re.findall(json_pattern, response_content, re.DOTALL)
                    
                    if json_matches:
                        # Try parsing each match until we find a valid one
                        for json_match in json_matches:
                            try:
                                response_json = json.loads(json_match)
                                structured_output = SearchParameters(**response_json)
                                print("✅ Successfully extracted JSON from agent response")
                                break
                            except (json.JSONDecodeError, ValueError):
                                continue
                        else:
                            raise Exception("Could not parse any JSON objects from agent response")
                    else:
                        raise Exception("No JSON objects found in agent response")
                        
                except Exception as extraction_error:
                    print(f"❌ JSON extraction also failed: {extraction_error}")
                    print(f"Raw agent response: {response_content}")
                    
                    # Final fallback: return a default/empty SearchParameters object
                    print("⚠️ Using fallback empty SearchParameters")
                    structured_output = SearchParameters(
                        KeyWords=user_input,  # At least preserve the original query as keywords
                        ExcludeCommentaries=False
                    )
            print("Step 2: ✅ Structured outputs received from LLM")
            
            # Validate the structured output (similar to OpenAI's built-in validation)
            self._validate_structured_output(structured_output)
            
            print(f"Structured Output: {structured_output}")
            return structured_output
            
        except AzureError as e:
            print(f"Step 2: ❌ Azure error getting structured outputs: {e}")
            return None
        except Exception as e:
            print(f"Step 2: ❌ Error getting structured outputs: {e}")
            return None
        finally:
            # Always clean up the thread after use
            if thread_id:
                try:
                    self.ai_client.agents.threads.delete(thread_id)
                    print(f"🗑️ Cleaned up thread: {thread_id}")
                except Exception as cleanup_error:
                    print(f"⚠️ Warning: Failed to cleanup thread {thread_id}: {cleanup_error}")

    def structured_outputs_mapping(self, search_params: SearchParameters) -> dict:
        """
        Map human-readable display values to ID codes.
        
        Parameters:
        - search_params (SearchParameters): The structured parameters from the LLM
        
        Returns:
        - dict: Mapped parameters with ID codes
        """
        try:
            print("Step 3: 🔄 Mapping display values to ID codes...")
            
            # Define the mapping dictionaries (these should match your database/API requirements)
            # Note: These mappings should be extracted from the actual system configuration
            document_type_mapping = {
                "Enforcement Action": "1",
                "Voluntary Disclosure": "2",
                "Advisory": "3",
                "FAQ": "4",
                "Guidance": "5",
                "General License": "6",
                "Specific License": "7",
                "Interpretive Guidance": "8",
                "Regulatory Provision": "9",
            }
            
            legal_issue_mapping = {
                "Iran Sanctions": "1",
                "Cuba Sanctions": "2", 
                "Syria Sanctions": "3",
                "Russia Sanctions": "4",
                "North Korea Sanctions": "5",
                "Counter-Terrorism": "6",
                "Anti-Money Laundering": "7",
                "Export Controls": "8",
                "Economic Sanctions": "9",
            }
            
            program_mapping = {
                "OFAC": "1",
                "Iran": "2",
                "Cuba": "3",
                "Syria": "4", 
                "Russia": "5",
                "North Korea": "6",
                "Counter-Terrorism": "7",
                "Narcotics": "8",
                "WMD": "9",
            }
            
            industry_mapping = {
                "Financial Services": "1",
                "Shipping": "2",
                "Energy": "3",
                "Technology": "4",
                "Manufacturing": "5",
                "Healthcare": "6",
                "Real Estate": "7",
                "Telecommunications": "8",
                "Transportation": "9",
            }
            
            # Map the parameters
            mapped_params = {}
            
            # Direct mappings
            if search_params.DateIssuedBegin:
                mapped_params["DateIssuedBegin"] = search_params.DateIssuedBegin
            if search_params.DateIssuedEnd:
                mapped_params["DateIssuedEnd"] = search_params.DateIssuedEnd
            if search_params.Published is not None:
                mapped_params["Published"] = search_params.Published
            if search_params.NumberOfViolationsLow:
                mapped_params["NumberOfViolationsLow"] = search_params.NumberOfViolationsLow
            if search_params.NumberOfViolationsHigh:
                mapped_params["NumberOfViolationsHigh"] = search_params.NumberOfViolationsHigh
            if search_params.KeyWords:
                mapped_params["KeyWords"] = search_params.KeyWords
            if search_params.ExcludeCommentaries:
                mapped_params["ExcludeCommentaries"] = search_params.ExcludeCommentaries
                
            # List mappings with ID conversion
            if search_params.DocumentType:
                mapped_params["DocumentType"] = [
                    document_type_mapping.get(dt, dt) for dt in search_params.DocumentType
                ]
            if search_params.LegalIssue:
                mapped_params["LegalIssue"] = [
                    legal_issue_mapping.get(li, li) for li in search_params.LegalIssue
                ]
            if search_params.Program:
                mapped_params["Program"] = [
                    program_mapping.get(p, p) for p in search_params.Program
                ]
            if search_params.Industry:
                mapped_params["Industry"] = [
                    industry_mapping.get(i, i) for i in search_params.Industry
                ]
                
            # Pass through other list fields as-is (or add more mappings as needed)
            if search_params.RegulatoryProvision:
                mapped_params["RegulatoryProvision"] = search_params.RegulatoryProvision
            if search_params.EnforcementCharacterization:
                mapped_params["EnforcementCharacterization"] = search_params.EnforcementCharacterization
            if search_params.OFACPenalty:
                mapped_params["OFACPenalty"] = search_params.OFACPenalty
            if search_params.AggregatePenalty:
                mapped_params["AggregatePenalty"] = search_params.AggregatePenalty
            if search_params.RespondentNationality:
                mapped_params["RespondentNationality"] = search_params.RespondentNationality
            if search_params.VoluntaryDisclosure:
                mapped_params["VoluntaryDisclosure"] = search_params.VoluntaryDisclosure
            if search_params.EgregiousCase:
                mapped_params["EgregiousCase"] = search_params.EgregiousCase
                
            print("Step 3: ✅ Display values mapped to ID codes successfully")
            return mapped_params
            
        except Exception as e:
            print(f"Step 3: ❌ Error mapping display values: {e}")
            return None

    def create_final_json_payload(self, mapped_params: dict) -> dict:
        """
        Create the final JSON payload for the search API.
        
        Parameters:
        - mapped_params (dict): The mapped search parameters
        
        Returns:
        - dict: Final JSON payload (the actual search parameters)
        """
        try:
            print("Step 4: 🔄 Creating final JSON payload...")
            
            # Return the actual JSON payload - just the search parameters
            print("Step 4: ✅ Final JSON payload created successfully")
            return mapped_params
            
        except Exception as e:
            print(f"Step 4: ❌ Error creating final payload: {e}")
            return None

    async def basic_search(self, user_input: str) -> dict:
        """
        Main function for basic search with filters - converts user query to structured search parameters.
        
        Parameters:
        - user_input (str): The raw user query
        
        Returns:
        - dict: Final JSON payload with search parameters or None if error
        """
        print(f"Step 1: 🚀 Starting basic search process for query: '{user_input}'")
        print("="*60)
        
        # Function 2: Get structured outputs from LLM
        structured_outputs = await self.user_query_to_structured_outputs(user_input)
        if not structured_outputs:
            print("Step 1: ❌ Failed at function 2 (structured outputs)")
            return None
        
        # Function 3: Map display values to IDs
        mapped_params = self.structured_outputs_mapping(structured_outputs)
        if not mapped_params:
            print("Step 1: ❌ Failed at function 3 (mapping)")
            return None
        
        # Function 4: Create final payload
        final_payload = self.create_final_json_payload(mapped_params)
        if not final_payload:
            print("Step 1: ❌ Failed at function 4 (final payload)")
            return None
        
        print("="*60)
        print("Step 1: 🎉 Basic search process completed successfully!")
        return final_payload

    def _validate_structured_output(self, structured_output: SearchParameters) -> None:
        """
        Validate the structured output to ensure it matches expected format.
        This mimics the validation that OpenAI's structured output parsing provides.
        
        Args:
            structured_output: The parsed SearchParameters object
            
        Raises:
            ValueError: If validation fails
        """
        try:
            # Validate date fields
            if structured_output.DateIssuedBegin is not None:
                if not isinstance(structured_output.DateIssuedBegin, int) or structured_output.DateIssuedBegin < 1900:
                    print(f"⚠️ Warning: DateIssuedBegin may be invalid: {structured_output.DateIssuedBegin}")
                    
            if structured_output.DateIssuedEnd is not None:
                if not isinstance(structured_output.DateIssuedEnd, int) or structured_output.DateIssuedEnd < 1900:
                    print(f"⚠️ Warning: DateIssuedEnd may be invalid: {structured_output.DateIssuedEnd}")
            
            # Validate list fields are actually lists
            list_fields = [
                'LegalIssue', 'Program', 'DocumentType', 'RegulatoryProvision',
                'EnforcementCharacterization', 'OFACPenalty', 'AggregatePenalty',
                'Industry', 'RespondentNationality', 'VoluntaryDisclosure', 'EgregiousCase'
            ]
            
            for field_name in list_fields:
                field_value = getattr(structured_output, field_name, [])
                if not isinstance(field_value, list):
                    print(f"⚠️ Warning: {field_name} should be a list but got {type(field_value)}")
            
            # Validate integer fields
            if structured_output.NumberOfViolationsLow is not None:
                if not isinstance(structured_output.NumberOfViolationsLow, int) or structured_output.NumberOfViolationsLow < 0:
                    print(f"⚠️ Warning: NumberOfViolationsLow may be invalid: {structured_output.NumberOfViolationsLow}")
                    
            if structured_output.NumberOfViolationsHigh is not None:
                if not isinstance(structured_output.NumberOfViolationsHigh, int) or structured_output.NumberOfViolationsHigh < 0:
                    print(f"⚠️ Warning: NumberOfViolationsHigh may be invalid: {structured_output.NumberOfViolationsHigh}")
            
            # Validate boolean fields
            if structured_output.Published is not None:
                if not isinstance(structured_output.Published, bool):
                    print(f"⚠️ Warning: Published should be boolean but got {type(structured_output.Published)}")
                    
            if not isinstance(structured_output.ExcludeCommentaries, bool):
                print(f"⚠️ Warning: ExcludeCommentaries should be boolean but got {type(structured_output.ExcludeCommentaries)}")
            
            # Validate KeyWords is a string
            if not isinstance(structured_output.KeyWords, str):
                print(f"⚠️ Warning: KeyWords should be string but got {type(structured_output.KeyWords)}")
            
            print("✅ Structured output validation completed")
            
        except Exception as e:
            print(f"⚠️ Warning during validation: {e}")
            # Don't raise - just warn, as the Pydantic model already did basic validation
# Global instance management
_simple_search_agent_instance = None

def get_simple_search_agent() -> SimpleSearchAgent:
    """
    Get or create the global SimpleSearchAgent instance.
    
    Returns:
        SimpleSearchAgent: The global agent instance
    """
    global _simple_search_agent_instance
    if _simple_search_agent_instance is None:
        _simple_search_agent_instance = SimpleSearchAgent()
    return _simple_search_agent_instance

async def basic_search_agent(user_input: str) -> dict:
    """
    Agent-based basic search function to replace the original basic_search.
    
    Parameters:
    - user_input (str): The raw user query
    
    Returns:
    - dict: Final JSON payload with search parameters or None if error
    """
    agent = get_simple_search_agent()
    return await agent.basic_search(user_input)

def cleanup_simple_search_agent():
    """
    Clean up the global SimpleSearchAgent instance.
    Should be called during application shutdown.
    """
    global _simple_search_agent_instance
    if _simple_search_agent_instance:
        try:
            _simple_search_agent_instance.cleanup()
        except Exception as e:
            print(f"❌ Error cleaning up simple search agent: {e}")
        finally:
            _simple_search_agent_instance = None

# Example usage function for testing
async def example_usage():
    """Example usage of the complete agent-based process."""
    
    # Example queries
    example_queries = [
        "Find OFAC violations related to Iran sanctions from 2020 to 2023",
        "Show me voluntary disclosures in the financial services industry",
        "Search for cases involving global distribution systems with penalties over $1 million"
    ]
    
    agent = get_simple_search_agent()
    
    for query in example_queries:
        print("\n" + "="*80)
        print(f"Example Query: {query}")
        print("="*80)
        
        # Run complete process
        final_json = await agent.basic_search(query)
        
        if final_json:
            print("\n📋 Final JSON Payload:")
            print(json.dumps(final_json, indent=2))
        else:
            print("❌ Process failed - Could not create final JSON payload")

if __name__ == "__main__":
    import asyncio
    
    print("🔍 Simple Search Agent - Azure AI Foundry Agent-Based Pipeline")
    print("="*60)
    
    # Run example usage
    asyncio.run(example_usage())
    
    print("\n" + "="*60)
    print("💬 Interactive Mode")
    print("="*60)
    
    async def interactive_mode():
        agent = get_simple_search_agent()
        
        while True:
            user_input = input("\n🔍 Enter your search query (or 'quit' to exit): ")
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
                
            # Run complete agent-based process
            final_json = await agent.basic_search(user_input)
            
            if final_json:
                print("\n📋 Final JSON Payload:")
                print(json.dumps(final_json, indent=2))
            else:
                print("❌ Process failed - Could not create final JSON payload")
                print("Please try again with a different query.")
        
        print("👋 Goodbye!")
        
        # Cleanup
        cleanup_simple_search_agent()
    
    # Run interactive mode
    asyncio.run(interactive_mode())
