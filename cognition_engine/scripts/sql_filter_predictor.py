import psycopg2
import psycopg2.extras
import os
import json
import asyncio
from typing import Dict, List, Any, Sequence, AsyncGenerator
from datetime import datetime, timedelta

# Import Autogen components
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import TextMessage, BaseChatMessage, UserMessage, BaseAgentEvent
from autogen_core import CancellationToken

# Import for Vector DB and Embeddings
import chromadb
from langchain_community.embeddings import OllamaEmbeddings # Use langchain_community
from chromadb.api.types import Documents

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

MIN_DEAL_COUNT = 5 # Still useful for filtering out statistically insignificant segments

# --- Helper function to build dynamic WHERE clauses ---
def build_filter_conditions(filters: Dict[str, List[str]]) -> str:
    """Builds SQL WHERE clause fragments for IN conditions."""
    conditions = []
    if 'hs_analytics_source' in filters and filters['hs_analytics_source']:
        sources = ", ".join(f"'{s}'" for s in filters['hs_analytics_source'])
        conditions.append(f"d.hs_analytics_source IN ({sources})")
    if 'job_title' in filters and filters['job_title']:
        titles = ", ".join(f"'{t}'" for t in filters['job_title'])
        # Assuming contacts table is joined and job_title is on contacts (ct)
        conditions.append(f"ct.job_title IN ({titles})")
    if 'industry' in filters and filters['industry']:
        industries = ", ".join(f"'{i}'" for i in filters['industry'])
        # Assuming companies table is joined and industry is on companies (c)
        conditions.append(f"c.industry IN ({industries})")

    return " AND ".join(conditions) if conditions else "TRUE" # Return TRUE if no filters

# --- Vector Database Setup ---
CHROMA_DB_PATH = os.path.join(os.getcwd(), "chroma_db_segments")
COLLECTION_NAME = "high_win_rate_segments"

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Initialize Ollama embeddings (using langchain_community)
embedding_model = "qwen3:1.7b" # Or your preferred embedding model
ollama_embeddings = OllamaEmbeddings(model=embedding_model, base_url="http://localhost:11434")

# Define a custom embedding function for ChromaDB using Ollama
class ChromaDBEmbeddingFunction:
    """
    Custom embedding function for ChromaDB using embeddings from Ollama.
    Adapts the langchain_community OllamaEmbeddings for ChromaDB.
    """
    def __init__(self, langchain_embeddings: OllamaEmbeddings):
        self.langchain_embeddings = langchain_embeddings

    def __call__(self, input: Documents):
        # ChromaDB expects a list of embeddings corresponding to the list of documents
        return self.langchain_embeddings.embed_documents(input)

# Initialize the embedding function with Ollama embeddings
chroma_embedding_function = ChromaDBEmbeddingFunction(ollama_embeddings)

# Get or create the collection
# Note: When using a custom embedding function, you must specify it here.
try:
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Segments analyzed for high win rates"},
        embedding_function=chroma_embedding_function # Specify the custom embedding function
    )
    print(f"Initialized ChromaDB collection: {COLLECTION_NAME}")
except Exception as e:
    print(f"Error initializing ChromaDB collection: {e}")
    collection = None # Handle cases where ChromaDB initialization fails

# --- Component to process data and vectorize ---
def vectorize_segments_data(raw_data: List[Dict]):
    """
    Processes raw segment data, creates documents, and adds them to the vector DB.
    """
    if collection is None:
        print("ChromaDB collection not initialized. Skipping vectorization.")
        return

    documents_to_add = []
    ids_to_add = []
    metadatas_to_add = []

    for row in raw_data:
        industry = row.get('industry', 'Unknown Industry')
        job_title = row.get('job_title', 'Unknown Job Title')
        source = row.get('hs_analytics_source', 'Unknown Source')
        total_deals = row.get('total_deals', 0)
        won_deals = row.get('won_deals', 0)
        avg_amount = row.get('avg_amount', 0)
        avg_days_to_close = row.get('avg_days_to_close', 0)

        # Calculate metrics for the document content
        win_rate = (won_deals / total_deals * 100) if total_deals > 0 else 0
        acv = avg_amount or 0
        sales_cycle = avg_days_to_close or 0

        # Create a descriptive document string for the segment
        doc_content = (
            f"Segment: Industry='{industry}', Job Title='{job_title}', Source='{source}'. "
            f"Performance Metrics: Total Deals={total_deals}, Won Deals={won_deals}, Win Rate={win_rate:.2f}%, "
            f"Average Deal Value (ACV)=${acv:.2f}, Average Sales Cycle={sales_cycle:.2f} days."
        )

        # Create a unique ID for the document (combination of dimension values)
        doc_id = f"{industry}_{job_title}_{source}".replace(" ", "_").lower() # Simple ID generation

        # Store full data in metadata for retrieval
        metadata = {
            "industry": industry,
            "job_title": job_title,
            "hs_analytics_source": source,
            "total_deals": total_deals,
            "won_deals": won_deals,
            "win_rate": round(win_rate, 2),
            "acv": round(acv, 2),
            "sales_cycle_days": round(sales_cycle, 2)
        }

        documents_to_add.append(doc_content)
        ids_to_add.append(doc_id)
        metadatas_to_add.append(metadata)

    if documents_to_add:
        try:
            # Clear existing data in the collection (optional, for fresh runs)
            # collection.delete(ids=collection.get()['ids']) # Uncomment to clear

            collection.add(
                documents=documents_to_add,
                ids=ids_to_add,
                metadatas=metadatas_to_add
            )
            print(f"Added {len(documents_to_add)} segment documents to ChromaDB collection.")
        except Exception as e:
            print(f"Error adding documents to ChromaDB: {e}")


# --- Agent to query database and vectorize (modified DatabaseQueryAgent) ---
class DataVectorizerAgent(BaseChatAgent):
    def __init__(self, name: str, db_config: dict):
        super().__init__(name=name, description="Agent that queries the database for segment data and vectorizes it.")
        self.db_config = db_config

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,) # Will yield status message

    async def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        # This agent runs the query and vectorizes, it doesn't need complex input from messages
        print("Running database query and vectorization...")
        conn = None
        cursor = None
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            # Query to get metrics for all combinations of dimensions with minimum deals
            # This query is similar to the one in the previous script, grouping by all dimensions
            query = f"""
                SELECT
                    c.industry,
                    ct.job_title,
                    d.hs_analytics_source,
                    COUNT(d.id) AS total_deals,
                    COUNT(CASE WHEN d.dealstage = 'closedwon' THEN d.id END) AS won_deals,
                    AVG(d.amount) AS avg_amount,
                    AVG(d.days_to_close) AS avg_days_to_close
                FROM dim_deals d
                JOIN dim_companies c ON d.company_id = c.company_id
                LEFT JOIN dim_contacts ct ON d.company_id = ct.company_id
                GROUP BY
                    c.industry,
                    ct.job_title,
                    d.hs_analytics_source
                HAVING COUNT(d.id) >= {MIN_DEAL_COUNT}; -- Only include segments with enough data

            """
            cursor.execute(query)
            raw_data = cursor.fetchall()

            print(f"Fetched {len(raw_data)} segments from the database.")

            # Vectorize the fetched data and add to ChromaDB
            vectorize_segments_data(raw_data)

            result_message = {"status": "success", "message": f"Successfully vectorized {len(raw_data)} segments."}

        except Exception as e:
            result_message = {"status": "error", "message": str(e)}
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

        yield Response(
            chat_message=TextMessage(content=json.dumps(result_message), source=self.name),
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
        raise RuntimeError("DataVectorizerAgent did not produce a response.")


# --- Agent for RAG Analysis of High Win Rate Segments ---
class RAGAnalysisAgent(BaseChatAgent):
    def __init__(self, name: str, llm_model: str = "gemma3:1b"):
        super().__init__(name=name, description="Agent that uses RAG to analyze segments and find those with high win rates.")
        # Configure your LLM client here
        from langchain_community.chat_models import ChatOllama # Using ChatOllama for conversation capabilities
        self._llm = ChatOllama(model=llm_model, base_url="http://localhost:11434") # Using ChatOllama for conversation

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    async def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        latest_message = messages[-1]
        # The message content is the user's query (e.g., "Which segments have high win rates?")
        user_query = latest_message.content

        if collection is None:
             yield Response(
                 chat_message=TextMessage(content=json.dumps({"status": "error", "message": "Vector database not initialized."}), source=self.name),
                 inner_messages=[]
             )
             return

        try:
            # Step 1: Query the vector database for relevant segments
            # Use the user's query to search for similar segment descriptions
            print(f"Querying vector DB with user query: '{user_query}'...")
            results = collection.query(
                query_texts=[user_query],
                n_results=10, # Retrieve top 10 most relevant segments
                include=['documents', 'metadatas'] # Include document content and metadata
            )

            retrieved_segments = results.get("metadatas", [[]])[0] # Get metadata for the top query
            retrieved_documents = results.get("documents", [[]])[0] # Get documents for the top query

            if not retrieved_segments:
                 yield Response(
                     chat_message=TextMessage(content=json.dumps({"status": "success", "analysis": "No relevant segments found in the vector database.", "data": []}), source=self.name),
                     inner_messages=[]
                 )
                 return

            # Step 2: Format retrieved segments data as context for the LLM
            context = "Retrieved Segment Data:\n\n"
            for i, segment_metadata in enumerate(retrieved_segments):
                 context += f"Segment {i+1}:\n"
                 for key, value in segment_metadata.items():
                      context += f"  {key}: {value}\n"
                 context += "\n"

            # Step 3: Formulate prompt for the LLM with context and user query
            prompt = f"""
Analyze the following sales segment data provided as context.
Identify the segments within this data that have the highest Win Rate.
Explain *why* their win rates are high based on the other metrics (Total Deals, ACV, Sales Cycle) if possible from the provided data.
List the top 3 segments with the highest win rates from the context.

Context:
{context}

User Query: {user_query}

Analysis:
"""
            print("Sending augmented prompt to LLM...")
            # Use the LLM client to generate the analysis based on the context
            # Langchain ChatOllama uses invoke
            llm_analysis_text = self._llm.invoke(prompt)


            # You can structure the final output as needed, maybe including raw retrieved data
            final_result = {
                "status": "success",
                "query": user_query,
                "retrieved_segments_count": len(retrieved_segments),
                "retrieved_data": retrieved_segments, # Include raw retrieved data
                "analysis": llm_analysis_text
            }

            yield Response(
                chat_message=TextMessage(content=json.dumps(final_result), source=self.name),
                inner_messages=[],
            )

        except Exception as e:
            yield Response(
                chat_message=TextMessage(content=json.dumps({"status": "error", "message": str(e)}), source=self.name),
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
        raise RuntimeError("RAGAnalysisAgent did not produce a response.")


# --- Main RAG Workflow Orchestration ---
async def predict_high_win_rate_segments_rag():
    print("Starting RAG-based High Win Rate Prediction workflow...")

    # Instantiate agents
    data_vectorizer_agent = DataVectorizerAgent(name="DataVectorizerAgent", db_config=DB_CONFIG)
    rag_analysis_agent = RAGAnalysisAgent(name="RAGAnalysisAgent", llm_model="gemma3:1b") # Use your preferred LLM

    # Step 1: Extract data from DB and vectorize
    # Send an empty message to trigger the DataVectorizerAgent
    print(f"Starting data extraction and vectorization using {data_vectorizer_agent.name}...")
    vectorization_response = await data_vectorizer_agent.on_messages([UserMessage(content="", source="user")], CancellationToken())

    if not vectorization_response or not vectorization_response.chat_message:
        print(f"No valid response received from {data_vectorizer_agent.name}.")
        return

    try:
        vectorization_status = json.loads(vectorization_response.chat_message.content)
        if vectorization_status.get("status") != "success":
            print(f"Error during vectorization: {vectorization_status.get('message')}")
            return
        print(vectorization_status.get('message'))
    except json.JSONDecodeError:
        print("Error decoding vectorization response.")
        print(vectorization_response.chat_message.content)
        return

    # Step 2: Perform RAG analysis using the populated vector DB
    # Define the user query for the RAG agent
    rag_query = "Which sales segments have the highest win rate?" # The query for the RAG system

    print(f"\nStarting RAG analysis using {rag_analysis_agent.name} with query: '{rag_query}'...")
    rag_analysis_response = await rag_analysis_agent.on_messages([UserMessage(content=rag_query, source="user")], CancellationToken())

    if rag_analysis_response and rag_analysis_response.chat_message:
        final_result_json = rag_analysis_response.chat_message.content
        print(f"Received final RAG analysis from {rag_analysis_agent.name}:")
        # Parse and pretty print the final result
        try:
            final_result = json.loads(final_result_json)
            print(json.dumps(final_result, indent=2))
        except json.JSONDecodeError:
            print("Error decoding final JSON response from RAG agent.")
            print(final_result_json)
    else:
        print(f"No valid response received from {rag_analysis_agent.name}.")

    print("\nWorkflow finished.")


# --- Run the workflow ---
if __name__ == "__main__":
    # Ensure you have ChromaDB and Ollama embeddings dependencies installed
    # pip install chromadb langchain-community ollama
    asyncio.run(predict_high_win_rate_segments_rag())
