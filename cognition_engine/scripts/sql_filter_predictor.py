import psycopg2
import psycopg2.extras
import os
import json
import asyncio
from typing import Dict, List, Any, Sequence, AsyncGenerator, Optional
from datetime import datetime, timedelta

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Import Autogen components
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import TextMessage, BaseChatMessage, UserMessage, BaseAgentEvent
from autogen_core import CancellationToken

# Import for Vector DB and Embeddings
import chromadb
# Import OllamaEmbeddings and ChatOllama from langchain_community or langchain_ollama
# Note: LangchainDeprecationWarning suggests using langchain-ollama.
# If you have `langchain-ollama` installed:
# from langchain_ollama import ChatOllama
# from langchain_ollama import OllamaEmbeddings
# If you only have `langchain_community` installed:
from langchain_community.chat_models import ChatOllama # Using ChatOllama for conversation capabilities
from langchain_community.embeddings import OllamaEmbeddings # Using OllamaEmbeddings for RAG

# Import OpenAI components
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings # Using OpenAIEmbeddings for RAG

from chromadb.api.types import Documents, QueryResult

# --- Database Configuration ---
# Ensure these environment variables are set or replace with your actual config
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://cognition_user:cognition_password@localhost:5432/cognition_db")
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "cognition_db"),
    "user": os.getenv("DB_USER", "cognition_user"),
    "password": os.getenv("DB_PASSWORD", "cognition_password"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
}

# --- Dimension Values and Threshold ---
# These are the possible values the agents can use for filtering
HS_ANALYTICS_SOURCES = [
    'OFFLINE', 'EMAIL_MARKETING', 'DIRECT_TRAFFIC',
    'PAID_SEARCH', 'ORGANIC_SEARCH'
]

JOB_TITLES = [
    'Salesperson', 'Head of Talent', 'CEO', 'CHRO', 'HR Director',
    'Executive Chairperson', 'VP People', 'People Analytics Lead',
    'Head of Business Development Europe'
]

INDUSTRIES = [
    'PHARMACEUTICALS', 'COMPUTER_SOFTWARE', 'ACCOUNTING',
    'INFORMATION_TECHNOLOGY_AND_SERVICES', 'RETAIL', 'FOOD_BEVERAGES'
]

# New dimensions
COUNTRIES = [
    'United States',
    'United Kingdom',
    'Netherlands',
    'Australia',
    'Germany',
    'UAE'
]

EMPLOYEE_BUCKETS = [
    '1-1000',
    '4001-5000',
    '2001-3000',
    '1001-2000'
]

MIN_DEAL_COUNT = 5 # Minimum deals for a segment to be considered statistically significant

# --- ChromaDB Initialization ---
CHROMA_DB_PATH = os.path.join(os.getcwd(), "chroma_db_deals")
COLLECTION_NAME: str = "all_deals_context"

# Global variables to hold initialized ChromaDB components
chroma_client: Optional[chromadb.PersistentClient] = None
all_deals_collection: Optional[chromadb.Collection] = None
ollama_embeddings: Optional[OllamaEmbeddings] = None

print("Attempting to initialize ChromaDB and Ollama embeddings...")
try:
    print(f"[INIT] Initializing ChromaDB client at {CHROMA_DB_PATH}...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    print("[INIT] ChromaDB client initialized successfully.")

    embeddings_model: Optional[object] = None # Will hold either OllamaEmbeddings or OpenAIEmbeddings
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Attempt to initialize OpenAI embeddings first
    if openai_api_key:
        openai_embedding_model_name = "text-embedding-ada-002" # Use the model you populated with
        print(f"[INIT] Attempting to initialize OpenAI embeddings with model '{openai_embedding_model_name}'...")
        try:
            embeddings_model = OpenAIEmbeddings(api_key=openai_api_key, model=openai_embedding_model_name)
            # Perform a small test invocation to check if it works
            embeddings_model.embed_documents(["test"]) # Use embed_documents for batch embedding test
            print("[INIT] Successfully initialized OpenAI embeddings.")
        except Exception as e:
            print(f"[INIT] Failed to initialize OpenAI embeddings: {e}")
            embeddings_model = None # Ensure it's None if OpenAI fails
    else:
        print("[INIT] OPENAI_API_KEY not found in environment variables. Skipping OpenAI embeddings initialization.")

    # If OpenAI initialization failed, attempt to initialize Ollama embeddings
    if embeddings_model is None:
        ollama_embedding_model_name = "qwen3:1.7b" # Ensure this matches populate_vector_db.py if you used Ollama
        print(f"[INIT] Attempting to initialize Ollama embeddings with model '{ollama_embedding_model_name}'...")
        try:
            embeddings_model = OllamaEmbeddings(model=ollama_embedding_model_name, base_url="http://localhost:11434")
            # Perform a small test invocation to check if it works
            embeddings_model.embed_documents(["test"]) # Use embed_documents for batch embedding test
            print("[INIT] Successfully initialized Ollama embeddings.")
        except Exception as e:
            print(f"[INIT] Failed to initialize Ollama embeddings: {e}")
            embeddings_model = None # Ensure it's None if Ollama fails

    # If neither LLM initialized, raise an error
    if embeddings_model is None:
        raise RuntimeError("Failed to initialize both OpenAI and Ollama embedding models for ChromaDB.")

    # Define a simple embedding function wrapper for ChromaDB
    class ChromaDBEmbeddingFunction:
        def __init__(self): # No need to pass embeddings during init
             pass # Embedding model will be accessed globally in __call__

        def __call__(self, input: Documents):
             # input is expected to be a list of strings (Documents type hint)
             print(f"[EMBED] Received input type: {type(input)}")
             
             # Access the globally initialized embeddings model
             global embeddings_model
             if embeddings_model is None:
                  # This case should ideally not happen if initialization check passes,
                  # but as a safeguard:
                  raise RuntimeError("Embeddings model is not initialized in ChromaDBEmbeddingFunction.__call__.")

             if isinstance(input, str):
                  print(f"[EMBED] Input is a string. Converting to list: [{input[:50]}...]")
                  input_list = [input]
             elif isinstance(input, list):
                  print(f"[EMBED] Input is a list. Checking contents...")
                  # Filter out non-string elements and log warnings
                  input_list = []
                  for i, item in enumerate(input):
                       if isinstance(item, str):
                            input_list.append(item)
                       else:
                            print(f"[EMBED] Warning: Item at index {i} is not a string (type: {type(item)}). Skipping.")
                  if not input_list:
                       print("[EMBED] Warning: List input contained no strings after filtering.")
                       # Depending on expected behavior, maybe raise error or return empty list
                       # For now, return empty list if no valid strings found
                       return [] # Or raise ValueError("Input list contains no valid strings.")
             else:
                  print(f"[EMBED] Error: Unexpected input type for embedding function: {type(input)}")
                  # Return empty list or raise error for unhandled types
                  return [] # Or raise TypeError("Input for embedding function must be a string or a list of strings.")

             # Langchain's embed_documents expects a list of strings
             print(f"[EMBED] Passing {len(input_list)} documents to embeddings_model.embed_documents...")
             try:
                  embeddings = embeddings_model.embed_documents(input_list)
                  print("[EMBED] Embeddings model.embed_documents successful.")
                  return embeddings
             except Exception as e:
                  print(f"[EMBED ERROR] Error during embeddings_model.embed_documents: {e}")
                  # Re-raise the exception after logging, as embedding failure is critical
                  raise


    # Instantiate the embedding function without passing the model instance
    chroma_embedding_function = ChromaDBEmbeddingFunction()
    print("[INIT] Custom embedding function created.")

    # Get the collection (do not create it here, it should be created by populate_vector_db.py)
    print(f"[INIT] Getting ChromaDB collection: {COLLECTION_NAME}...")
    # Use get_or_create_collection, but it won't re-create if it exists
    all_deals_collection = chroma_client.get_or_create_collection(
         name=COLLECTION_NAME,
         # Need to provide embedding_function when getting the collection if it was created with one
         embedding_function=chroma_embedding_function # Provide the embedding function here
    )
    print(f"[INIT] Successfully accessed ChromaDB collection '{COLLECTION_NAME}'.")

except Exception as e:
    print(f"[INIT ERROR] An unexpected error occurred during ChromaDB/Embeddings initialization: {e}")
    # Keep the global variables as None if initialization fails
    chroma_client = None
    all_deals_collection = None
    ollama_embeddings = None # Also set embeddings to None
    embeddings_model = None # Ensure embeddings_model is also None on error

# --- Helper function to build dynamic WHERE clause for a single filter combo ---
def build_single_filter_condition(filter_combo: Dict[str, List[str]]) -> str:
    """Builds SQL WHERE clause for a single filter combination."""
    conditions = []
    # Use .get() with an empty list default to handle missing keys in the combo
    if filter_combo.get('hs_analytics_source'):
        sources = ", ".join(f"'{s}'" for s in filter_combo['hs_analytics_source'])
        conditions.append(f"d.hs_analytics_source IN ({sources})")
    if filter_combo.get('job_title'):
        titles = ", ".join(f"'{t}'" for t in filter_combo['job_title'])
        conditions.append(f"ct.job_title IN ({titles})")
    if filter_combo.get('industry'):
        industries = ", ".join(f"'{i}'" for i in filter_combo['industry'])
        conditions.append(f"c.industry IN ({industries})")
    # Add conditions for new dimensions
    if filter_combo.get('country'):
        countries = ", ".join(f"'{c}'" for c in filter_combo['country'])
        conditions.append(f"c.country IN ({countries})")
    if filter_combo.get('employee_bucket'):
        buckets = ", ".join(f"'{b}'" for b in filter_combo['employee_bucket'])
        conditions.append(f"c.employee_bucket IN ({buckets})")

    return " AND ".join(conditions) if conditions else "TRUE" # Return TRUE if no filters specified

# Access global ChromaDB variables (assumed to be set by populate_vector_db.py or equivalent)
# These need to be imported or accessed carefully to avoid errors if not set.
# A more robust way is dependency injection, but for this structure, globals are used.
# REMOVED: import chromadb # Import chromadb here for access

# Define global variables (will be populated by populate_vector_db.py if run)
# REMOVED: Initialization block for ChromaDB clients and collection.
# The variables are declared here, but expected to be assigned values externally.
# REMOVED: chroma_client: Optional[chromadb.PersistentClient] = None
# REMOVED: all_deals_collection: Optional[chromadb.Collection] = None
# REMOVED: chroma_initialized: bool = False
# REMOVED: COLLECTION_NAME: str = "all_deals_context"
# REMOVED: CHROMA_DB_PATH = os.path.join(os.getcwd(), "chroma_db_deals")

# --- Agent 1: Filter Prediction Agent (LLM-backed, with RAG) ---
# This agent now assumes the vector database client and collection are available as global variables
# if the database population script has been run.
class FilterPredictionAgent(BaseChatAgent):
    def __init__(self, name: str, openai_model: str = "gpt-4o", ollama_model: str = "gemma3:1b"):
        super().__init__(name=name, description="Agent that predicts high-performing filter combinations using LLM reasoning and vector DB context")

        # Using ChatOllama for conversational capabilities
        # Ensure Ollama server is running and model is pulled (gemma3:1b for prediction, qwen3:1.7b for embeddings in populate_vector_db.py)
        self._llm = None
        openai_api_key = os.getenv("OPENAI_API_KEY")

        # Attempt to initialize OpenAI first
        if openai_api_key:
            print(f"Attempting to initialize OpenAI model '{openai_model}'...")
            try:
                self._llm = ChatOpenAI(api_key=openai_api_key, model=openai_model)
                print("Successfully initialized OpenAI model.")
            except Exception as e:
                print(f"Failed to initialize OpenAI model: {e}")
                self._llm = None # Ensure _llm is None if OpenAI initialization fails
        else:
            print("OPENAI_API_KEY not found in environment variables. Skipping OpenAI initialization.")

        # If OpenAI initialization failed, attempt to initialize Ollama
        if self._llm is None:
            print(f"Attempting to initialize Ollama model '{ollama_model}'...")
            try:
                # Check if Ollama server is running and model exists
                # A simple way to check might be to try invoking it immediately after init
                ollama_test_llm = ChatOllama(model=ollama_model, base_url="http://localhost:11434")
                ollama_test_llm.invoke("hello") # Simple test invocation
                self._llm = ollama_test_llm
                print("Successfully initialized Ollama model.")
            except Exception as e:
                print(f"Failed to initialize Ollama model: {e}")
                self._llm = None # Ensure _llm is None if Ollama initialization fails

        # If neither LLM initialized, raise an error
        if self._llm is None:
            raise RuntimeError("Failed to initialize both OpenAI and Ollama models.")

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    async def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        # This agent initiates the process by querying the vector DB
        try:
            # Check if ChromaDB collection and the embeddings model were successfully initialized globally
            global embeddings_model # Access the global embeddings model
            if all_deals_collection is None or embeddings_model is None:
                 yield Response(
                     chat_message=TextMessage(content=json.dumps({"status": "error", "message": f"Vector database or embeddings not initialized. ChromaDB Collection: {all_deals_collection is not None}, Embeddings Model: {embeddings_model is not None}. Check initialization logs."}), source=self.name),
                     inner_messages=[]
                 )
                 return

            rag_query = "Examples of deals with high win rate and good performance metrics."
            print(f"Querying vector DB for context: '{rag_query}'...")

            # Query the collection
            results: QueryResult = all_deals_collection.query(
                query_texts=[rag_query],
                n_results=20, # Retrieve a reasonable number of relevant deals for context
                include=['documents', 'metadatas']
            )

            retrieved_deals = results.get("documents", [[]])[0]
            retrieved_metadatas = results.get("metadatas", [[]])[0]


            if not retrieved_deals:
                 context = "No relevant deal examples found in the vector database."
                 print(context)
            else:
                 context = "Context from similar high-performing deals:\n\n"
                 for i, (doc, meta) in enumerate(zip(retrieved_deals, retrieved_metadatas)):
                    context += f"Deal {i+1}: {doc}\n"
                    # Optionally add some key metadata fields directly
                    # Ensure keys exist in metadata before accessing and explicitly convert to string
                    dealstage = str(meta.get('dealstage', 'N/A'))
                    amount = float(meta.get('amount', 0) or 0)
                    days_to_close = float(meta.get('days_to_close', 0) or 0)
                    industry = str(meta.get('industry', 'N/A'))
                    employees = str(meta.get('numberofemployees', 'N/A'))
                    job_title = str(meta.get('job_title', 'N/A'))
                    context += f"  - Dealstage: {dealstage}, Amount: ${amount:.2f}, Sales Cycle: {days_to_close:.2f} days\n"
                    # Include other metadata fields explicitly converted to string
                    context += f"  - Industry: {industry}, Employees: {employees}, Job Title: {job_title}\n"
                    context += "\n"
                    print(f"Retrieved {len(retrieved_deals)} deals for context.")


             # Step 2: Formulate prompt for the LLM with context and available dimensions
                    prompt = f"""
Analyze the following examples of high-performing deals provided as context.
Based on these examples and your general knowledge about sales dynamics, predict 5 to 10 specific filter combinations (segments) that are likely to have high revenue velocity. Revenue velocity is calculated as (Win Rate ÷ Sales Cycle in Days) × Average Contract Value (ACV).

Consider combinations of the following dimensions and their possible values:

- **Industries**: {', '.join(INDUSTRIES)}
- **Job Titles**: {', '.join(JOB_TITLES)}
- **Analytics Sources**: {', '.join(HS_ANALYTICS_SOURCES)}
- **Countries**: {', '.join(COUNTRIES)}
- **Employee Buckets**: {', '.join(EMPLOYEE_BUCKETS)}

For each predicted combination, provide a brief 'reasoning' field explaining *why* you predict this combination will have high revenue velocity, referencing patterns observed in the context examples or general sales principles. Also, provide the 'filter' field with the specific values chosen for each dimension in that combination as lists. Ensure each filter combination is a dictionary with 'filter' (a dictionary of lists) and 'reasoning' (a string) keys.

Example JSON format for the list of predictions:
[
  {{
    "filter": {{
      "industry": ["TECHNOLOGY"],
      "job_title": ["CEO", "VP People"],
      "hs_analytics_source": ["ORGANIC_SEARCH", "DIRECT_TRAFFIC"]
    }},
    "reasoning": "Based on the context, deals in the technology sector with C-level contacts from organic sources seem to close faster with higher amounts."
  }},
  {{
    "filter": {{
      "industry": ["PHARMACEUTICALS"],
      "job_title": ["Head of Talent"],
      "hs_analytics_source": ["EMAIL_MARKETING"]
    }},
    "reasoning": "Pharmaceutical deals with talent heads often represent specialized software needs, and email campaigns can effectively reach these niche roles."
  }},
  ...
]

Predicted Segments (JSON array):
"""
            # Add the context to the prompt for the LLM
            prompt_with_context = f"{context}\n\n{prompt}"

            # Print the full prompt being sent to the LLM for inspection
            print("--- START LLM PROMPT ---")
            print(prompt_with_context)
            print("--- END LLM PROMPT ---")

            # Use the LLM to generate the predicted filters based on context
            try:
                # Langchain ChatOllama uses invoke
                llm_response = self._llm.invoke(prompt_with_context) # Use the prompt with context again
                print("Successfully invoked LLM.")

                # Extract the string content from the LLM's response message
                llm_response_content = llm_response.content
                print("--- START RAW LLM RESPONSE CONTENT ---")
                print(llm_response_content)
                print("--- END RAW LLM RESPONSE CONTENT ---")

                # Attempt to extract JSON array from the response
                import re
                # Updated regex to be more robust, looking for the first occurrence of a list containing dicts
                match = re.search(r'(\[\s*\{.*?\}\s*(,\s*\{.*?\}\s*)*\])', llm_response_content, re.DOTALL)
                if match:
                    try:
                        # Take the first captured group, which should be the JSON array
                        json_string = match.group(1)
                        predicted_filters_with_reasoning = json.loads(json_string)
                        # Validate basic structure of the expected output
                        if isinstance(predicted_filters_with_reasoning, list) and \
                           all(isinstance(item, dict) and 'filter' in item and isinstance(item.get('filter'), dict) and 'reasoning' in item for item in predicted_filters_with_reasoning):

                             print(f"Successfully extracted {len(predicted_filters_with_reasoning)} predicted filters from LLM response.")
                             yield Response(
                                chat_message=TextMessage(content=json.dumps({"status": "success", "predicted_filters": predicted_filters_with_reasoning}), source=self.name),
                                inner_messages=[],
                             )
                        else:
                             print("LLM response did not match expected JSON structure.")
                             yield Response(
                                chat_message=TextMessage(content=json.dumps({"status": "error", "message": "LLM response format incorrect, expected list of dictionaries with 'filter' and 'reasoning'."}), source=self.name),
                                inner_messages=[],
                             )
                    except json.JSONDecodeError:
                        print("Could not parse JSON from LLM response.")
                        print(f"LLM response content was:\n{llm_response_content}") # Log response for debugging
                        yield Response(
                            chat_message=TextMessage(content=json.dumps({"status": "error", "message": "Could not parse JSON from LLM response."}), source=self.name),
                            inner_messages=[],
                        )
                else:
                    print("Could not find JSON array in LLM response.")
                    print(f"LLM response content was:\n{llm_response_content}") # Log response for debugging
                    yield Response(
                        chat_message=TextMessage(content=json.dumps({"status": "error", "message": "Could not extract filter combinations (JSON array) from LLM response."}), source=self.name),
                        inner_messages=[],
                    )
            except Exception as e:
                 # Catch any other exceptions during the LLM call or response processing
                 print(f"[LLM INVOKE ERROR] An unexpected error occurred during LLM invocation or initial processing: {e}")
                 yield Response(
                      chat_message=TextMessage(content=json.dumps({"status": "error", "message": f"LLM invocation or processing error: {e}"}), source=self.name),
                      inner_messages=[]
                 )
        except Exception as e:
            # Outer catch block for errors during ChromaDB query or context building
            print(f"[RAG ERROR] Error during vector database query or context building: {e}")
            yield Response(
                chat_message=TextMessage(content=json.dumps({"status": "error", "message": f"Vector database query or context building error: {e}"}), source=self.name),
                inner_messages=[],
            )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass

    async def on_messages(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        async for message in self.on_messages_stream(messages, cancellation_token):
            if isinstance(message, Response):
                return message
        raise RuntimeError("FilterPredictionAgent did not produce a response.")


# --- Agent 2: Database Query Agent (Measures performance for a single filter combo) ---
class DatabaseQueryAgent(BaseChatAgent):
    def __init__(self, name: str, db_config: Dict[str, Any]):
        super().__init__(name=name, description="Agent that queries the database to get deal metrics for a specific segment")
        self.db_config = db_config

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    async def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        latest_message = messages[-1]
        # The message content is expected to be a JSON string of a single filter combination dictionary
        try:
            filter_combo = json.loads(latest_message.content)

            if not isinstance(filter_combo, dict):
                 yield Response(
                     chat_message=TextMessage(content=json.dumps({"status": "error", "message": "Invalid input: Expected a filter combination dictionary."}), source=self.name),
                     inner_messages=[]
                 )
                 return

            where_condition = build_single_filter_condition(filter_combo)

            conn = None
            cursor = None
            try:
                conn = psycopg2.connect(**self.db_config)
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

                # Query to get metrics for this SPECIFIC filter combination
                # Note: We don't group by dimensions here, as the input already defines the group
                # The HAVING clause ensures we only return metrics if the segment has enough deals
                query = f"""
                    SELECT
                        COUNT(d.id) AS total_deals,
                        COUNT(CASE WHEN d.dealstage = 'closedwon' THEN d.id END) AS won_deals,
                        AVG(d.amount) AS avg_amount,
                        AVG(d.days_to_close) AS avg_days_to_close
                    FROM dim_deals d
                    JOIN dim_companies c ON d.company_id = c.company_id
                    LEFT JOIN dim_contacts ct ON d.company_id = ct.company_id
                    WHERE {where_condition}
                    GROUP BY TRUE -- Grouping by TRUE to get one row of aggregates for the entire filtered set
                    HAVING COUNT(d.id) >= {MIN_DEAL_COUNT}; -- Apply minimum deal count AFTER filtering

                """
                # print("Executing Query:", query) # Debug print
                cursor.execute(query)
                row = cursor.fetchone() # Fetch only one row of aggregates

                if row:
                    # Segment met minimum deal count and data was retrieved
                    result_data = {"status": "success", "metrics": row}
                else:
                    # No rows returned means the segment didn't meet MIN_DEAL_COUNT or has no deals matching the filter
                    result_data = {
                        "status": "success",
                        "metrics": None, # Explicitly None if no data/min deals not met
                         "message": f"Segment did not meet the minimum deal count ({MIN_DEAL_COUNT}) or had no deals matching the filter criteria."
                    }

            except Exception as e:
                result_data = {"status": "error", "message": str(e)}
            finally:
                if cursor:
                    cursor.close()
                if conn:
                    conn.close()

            # Always return a result_data dictionary, even on error
            yield Response(
                chat_message=TextMessage(content=json.dumps(result_data), source=self.name),
                inner_messages=[],
            )

        except json.JSONDecodeError:
            yield Response(
                 chat_message=TextMessage(content=json.dumps({"status": "error", "message": "Invalid JSON input"}), source=self.name),
                 inner_messages=[]
            )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass # No state to reset

    async def on_messages(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        async for message in self.on_messages_stream(messages, cancellation_token):
            if isinstance(message, Response):
                return message
        raise RuntimeError("DatabaseQueryAgent did not produce a response.")


# --- Agent 3: Revenue Velocity Analysis Agent (LLM-backed) ---
class RevenueVelocityAnalysisAgent(BaseChatAgent):
    def __init__(self, name: str, openai_model: str = "gpt-4o", ollama_model: str = "gemma3:1b"): # Use your preferred LLM
        super().__init__(name=name, description="Agent that analyzes predicted segment performance based on measured metrics")

        # Using ChatOllama for conversational capabilities
        # Ensure Ollama server is running and model is pulled (gemma3:1b for prediction, qwen3:1.7b for embeddings in populate_vector_db.py)
        self._llm = None
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Attempt to initialize OpenAI first
        if openai_api_key:
            print(f"Attempting to initialize OpenAI model '{openai_model}' for analysis...")
            try:
                self._llm = ChatOpenAI(api_key=openai_api_key, model=openai_model)
                print("Successfully initialized OpenAI model for analysis.")
            except Exception as e:
                print(f"Failed to initialize OpenAI model for analysis: {e}")
                self._llm = None
        else:
            print("OPENAI_API_KEY not found in environment variables. Skipping OpenAI initialization for analysis.")

        # If OpenAI initialization failed, attempt to initialize Ollama
        if self._llm is None:
            print(f"Attempting to initialize Ollama model '{ollama_model}' for analysis...")
            try:
                 # Check if Ollama server is running and model exists
                 # A simple way to check might be to try invoking it immediately after init
                 ollama_test_llm = ChatOllama(model=ollama_model, base_url="http://localhost:11434")
                 ollama_test_llm.invoke("hello") # Simple test invocation
                 self._llm = ollama_test_llm
                 print("Successfully initialized Ollama model for analysis.")
            except Exception as e:
                print(f"Failed to initialize Ollama model for analysis: {e}")
                self._llm = None

        # If neither LLM initialized, raise an error
        if self._llm is None:
            raise RuntimeError("Failed to initialize both OpenAI and Ollama models for analysis.")

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    def calculate_revenue_velocity(self, total_deals: int, won_deals: int, avg_amount: float, avg_days_to_close: float) -> float:
         """Calculates Revenue Velocity for a single segment."""
         win_rate = (won_deals / total_deals) if total_deals > 0 else 0
         acv = avg_amount or 0
         sales_cycle = avg_days_to_close or 0

         # Calculate Revenue Velocity: (Win Rate / Sales Cycle) * ACV
         # Avoid division by zero for sales cycle
         revenue_velocity = (win_rate / sales_cycle * acv) if sales_cycle > 0 else 0 # Handle sales_cycle = 0
         return round(revenue_velocity, 2)

    async def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        latest_message = messages[-1]
        # The message content is expected to be a JSON string containing a list of tested segments and their results
        try:
            tested_segments_results = json.loads(latest_message.content)

            if not isinstance(tested_segments_results, list):
                 yield Response(
                     chat_message=TextMessage(content=json.dumps({"status": "error", "message": "Invalid input: Expected a list of tested segment results."}), source=self.name),
                     inner_messages=[]
                 )
                 return

            analyzed_results = []
            for segment_result in tested_segments_results:
                filter_combo = segment_result.get("filter", {})
                prediction_reasoning = segment_result.get("reasoning", "No reasoning provided.")
                db_status = segment_result.get("db_status")
                db_message = segment_result.get("db_message")
                metrics = segment_result.get("metrics")

                measured_rv = 0
                if db_status == "success" and metrics is not None:
                    total_deals = metrics.get('total_deals', 0)
                    won_deals = metrics.get('won_deals', 0)
                    avg_amount = metrics.get('avg_amount', 0)
                    avg_days_to_close = metrics.get('avg_days_to_close', 0)

                    measured_rv = self.calculate_revenue_velocity(
                        total_deals, won_deals, avg_amount, avg_days_to_close
                    )

                    analyzed_results.append({
                        "filter": filter_combo,
                        "prediction_reasoning": prediction_reasoning,
                        "measured_metrics": {
                            "total_deals": total_deals,
                            "won_deals": won_deals,
                            "win_rate": round((won_deals / total_deals) if total_deals > 0 else 0, 2),
                            "acv": round(avg_amount or 0, 2),
                            "sales_cycle_days": round(avg_days_to_close or 0, 2),
                            "revenue_velocity": measured_rv
                        },
                        "status": "measured"
                    })
                else:
                    # Handle cases where DB query failed or no data for the segment
                     analyzed_results.append({
                         "filter": filter_combo,
                         "prediction_reasoning": prediction_reasoning,
                         "measured_metrics": None,
                         "status": "db_error" if db_status == "error" else "no_data",
                         "message": db_message if db_status == "error" else f"Segment did not meet the minimum deal count ({MIN_DEAL_COUNT}) or had no deals matching the filter criteria."
                     })

            # Sort segments by measured Revenue Velocity (highest first), putting segments with no data last
            # Use measured_metrics.get('revenue_velocity', -1) to handle None metrics and place them at the end
            analyzed_results.sort(key=lambda x: x.get("measured_metrics").get("revenue_velocity", -1) if x.get("measured_metrics") is not None else -1, reverse=True)


            if not analyzed_results:
                 yield Response(
                     chat_message=TextMessage(content=json.dumps({"status": "success", "analysis": "No segments were tested or none met the minimum deal count.", "results": []}), source=self.name),
                     inner_messages=[]
                 )
                 return


            # Use LLM for natural language analysis
            # Prepare data for LLM prompt
            results_summary = json.dumps(analyzed_results, indent=2)
            prompt = f"""
Analyze the following results from testing predicted sales segments.
Each item in the list represents a predicted segment filter, its initial prediction reasoning, and the actual measured metrics and calculated revenue velocity from the database.

Identify the segments with the highest *measured* Revenue Velocity.
Compare the measured results to the initial predictions and their reasoning.
Provide insights on which segments are the most promising based on the *measured* data.
Suggest potential actions based on the top-performing *measured* segments.

Tested Segment Results (JSON):
{results_summary}

Analysis:
"""
            print("Sending tested segment results to LLM for analysis...")

            try:
                # Langchain ChatOllama uses invoke
                llm_analysis_text = self._llm.invoke(prompt)


                final_result = {
                    "status": "success",
                    "tested_segments_count": len(analyzed_results),
                    "analysis": llm_analysis_text.content, # Access the string content
                    "results": analyzed_results # Include the structured results
                }

                yield Response(
                    chat_message=TextMessage(content=json.dumps(final_result), source=self.name),
                    inner_messages=[],
                )

            except Exception as llm_error:
                 yield Response(
                     chat_message=TextMessage(content=json.dumps({"status": "error", "message": f"Error during LLM analysis: {llm_error}\n\nStructured results:\n\n{results_summary}"}), source=self.name),
                     inner_messages=[]
                 )


        except json.JSONDecodeError:
            yield Response(
                 chat_message=TextMessage(content=json.dumps({"status": "error", "message": "Invalid JSON input from previous step"}), source=self.name),
                 inner_messages=[]
            )

    # Added missing methods required by BaseChatAgent
    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Resets the agent's state. No state to reset for this agent."""
        pass # No state to reset

    # Added missing method required by BaseChatAgent
    async def on_messages(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        """Handles a sequence of messages by processing the last message in the stream."""
        # This agent primarily works with the last message, so we can just pass it to the stream handler
        async for message in self.on_messages_stream(messages, cancellation_token):
            if isinstance(message, Response):
                return message
        raise RuntimeError("RevenueVelocityAnalysisAgent did not produce a response.")


# --- Main Workflow Orchestration ---
async def predict_measure_analyze_segments():
    print("Starting Predict, Measure, and Analyze Revenue Velocity workflow...")

    # The vector database population is now handled by populate_vector_db.py.
    # This script assumes the vector database is already populated.

    # Instantiate agents
    filter_agent = FilterPredictionAgent(name="FilterPredictionAgent", openai_model="gpt-4o", ollama_model="gemma3:1b")
    db_agent = DatabaseQueryAgent(name="DBQueryAgent", db_config=DB_CONFIG)
    analysis_agent = RevenueVelocityAnalysisAgent(name="AnalysisAgent", openai_model="gpt-4o", ollama_model="gemma3:1b")

    # Step 1: Use LLM to predict promising filter combinations
    print(f"\nRequesting filter predictions from {filter_agent.name}...")
    filter_response = await filter_agent.on_messages(
        [UserMessage(content="Predict segments with high revenue velocity.", source="user")],
        CancellationToken()
    )

    if not filter_response or not filter_response.chat_message:
        print(f"No valid response received from {filter_agent.name}.")
        return # Exit the async function

    try:
        filter_data = json.loads(filter_response.chat_message.content)
        if filter_data.get("status") != "success":
            print(f"Error from {filter_agent.name}: {filter_data.get('message')}")
            return # Exit the async function
        predicted_filters_with_reasoning = filter_data.get("predicted_filters", [])
        print(f"Received {len(predicted_filters_with_reasoning)} predicted filter combinations.")
        if not predicted_filters_with_reasoning:
            print("No filter combinations were predicted. Aborting workflow.")
            return # Exit the async function

    except json.JSONDecodeError:
        print("Error decoding predicted filters from LLM response.")
        return # Exit the async function

    # Step 2: For each predicted filter combination, query the database to get measured metrics
    tested_segments_results = []
    print("\n--- Measuring Performance of Predicted Segments ---")
    for idx, segment_prediction in enumerate(predicted_filters_with_reasoning, start=1):
        filter_combo = segment_prediction.get("filter", {})
        prediction_reasoning = segment_prediction.get("reasoning", "No reasoning provided.")

        if not filter_combo or not any(filter_combo.values()):
            print(f"\nSkipping predicted segment {idx}: Empty filter combination.")
            tested_segments_results.append({
                "filter": filter_combo,
                "reasoning": prediction_reasoning,
                "db_status": "skipped",
                "db_message": "Empty filter combination predicted.",
                "metrics": None
            })
            continue

        print(f"\nTesting predicted segment {idx}: Filter={filter_combo} (Reasoning: {prediction_reasoning[:100]}...)")

        db_request_message = UserMessage(content=json.dumps(filter_combo), source=filter_agent.name)
        db_response = await db_agent.on_messages([db_request_message], CancellationToken())

        if db_response and db_response.chat_message:
            try:
                db_result = json.loads(db_response.chat_message.content)
                tested_segments_results.append({
                    "filter": filter_combo,
                    "reasoning": prediction_reasoning,
                    "db_status": db_result.get("status"),
                    "db_message": db_result.get("message"),
                    "metrics": db_result.get("metrics")
                })
                print(f"  DB Query Status: {db_result.get('status')}")
                if db_result.get('status') == 'success' and db_result.get('metrics'):
                    metrics = db_result.get('metrics')
                    total = metrics.get('total_deals', 0)
                    won = metrics.get('won_deals', 0)
                    amount = metrics.get('avg_amount', 0)
                    days = metrics.get('avg_days_to_close', 0)
                    measured_rv = analysis_agent.calculate_revenue_velocity(total, won, amount, days)
                    print(f"  Measured Metrics: Total Deals={total}, Won Deals={won}, ACV=${amount:.2f}, Sales Cycle={days:.2f} days, Measured RV={measured_rv:.2f}")
                elif db_result.get('status') == 'success' and not db_result.get('metrics'):
                     print(f"  {db_result.get('message')}")

            except json.JSONDecodeError:
                print(f"  Error decoding DB response for segment {idx}.")
                tested_segments_results.append({
                    "filter": filter_combo,
                    "reasoning": prediction_reasoning,
                    "db_status": "error",
                    "db_message": "Invalid JSON response from DatabaseQueryAgent",
                    "metrics": None
                })
        else:
            print(f"No valid response received from {db_agent.name} for segment {idx}.")
            tested_segments_results.append({
                "filter": filter_combo,
                "reasoning": prediction_reasoning,
                "db_status": "error",
                "db_message": "No valid response from DatabaseQueryAgent",
                "metrics": None
            })

    # Step 3: Send tested segments results to Analysis Agent for final analysis and comparison
    if tested_segments_results:
        analyzable_results = [res for res in tested_segments_results if res.get("db_status") not in ["skipped", "error"]]
        if analyzable_results:
            print(f"\n--- Analyzing Measured Performance ---")
            print(f"Sending results of {len(analyzable_results)} tested segments to {analysis_agent.name} for final analysis...")
            analysis_request_message = UserMessage(content=json.dumps(analyzable_results), source=db_agent.name)
            analysis_response = await analysis_agent.on_messages([analysis_request_message], CancellationToken())

            if analysis_response and analysis_response.chat_message:
                final_result_json = analysis_response.chat_message.content
                print(f"Received final analysis from {analysis_agent.name}:")
                try:
                    final_result = json.loads(final_result_json)
                    # Print the full analysis result
                    print("--- Final Analysis Result ---")
                    print(json.dumps(final_result, indent=2))

                    # Extract the top predicted segments (which are sorted by RV in analyzed_results)
                    # The 'results' key in the final_result already contains the sorted analyzed_results
                    top_segments = final_result.get("results", [])

                    # Define output filename
                    output_filename = "top_predicted_segments.json"

                    # Write the top segments to a JSON file
                    if top_segments:
                        with open(output_filename, 'w') as f:
                            json.dump(top_segments, f, indent=2)
                        print(f"Successfully saved top predicted segments to {output_filename}")
                    else:
                        print("No analyzable segments with measured data to save.")

                except json.JSONDecodeError:
                    print("Error decoding final JSON response from Analysis agent.")
                    print(final_result_json)
            else:
                print(f"No valid response received from {analysis_agent.name}.")
        else:
            print("\nNo segments with valid database results to analyze.")
    else:
        print("\nNo segments were predicted or successfully tested for analysis.")

    print("\nWorkflow finished.")

# --- Run the workflow ---
if __name__ == "__main__":
    # Ensure you have psycopg2 and autogen dependencies installed
    # pip install psycopg2-binary autogen-agentchat autogen-core autogen-ext[ollama] chromadb langchain-community ollama
    # Make sure you have 'gemma3:1b' model pulled in Ollama

    # The vector database population is now handled by populate_vector_db.py.
    # Ensure you run `python populate_vector_db.py` at least once before running this script.

    # Run the main asynchronous workflow
    asyncio.run(predict_measure_analyze_segments())
