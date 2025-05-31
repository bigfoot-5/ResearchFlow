import psycopg2
import psycopg2.extras
import os
import json
import asyncio
from typing import Dict, List, Any, Sequence, Optional
from datetime import datetime, timedelta

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Import for Vector DB and Embeddings
import chromadb
# Import OllamaEmbeddings and ChatOllama from langchain_community or langchain_ollama
# Note: LangchainDeprecationWarning suggests using langchain-ollama.
# If you have `langchain-ollama` installed:
# from langchain_ollama import OllamaEmbeddings
# If you only have `langchain_community` installed:
from langchain_community.embeddings import OllamaEmbeddings # Use langchain_community

# Import OpenAI components
from langchain_openai import OpenAIEmbeddings # Use langchain_openai

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


# --- Vector Database Setup ---
# Using a different path or name to distinguish from segment summaries if needed
CHROMA_DB_PATH = os.path.join(os.getcwd(), "chroma_db_deals")
COLLECTION_NAME = "all_deals_context"

# Initialize ChromaDB client and embedding function
chroma_client: Optional[chromadb.PersistentClient] = None
all_deals_collection: Optional[chromadb.Collection] = None
embeddings_model: Optional[object] = None # Will hold either OllamaEmbeddings or OpenAIEmbeddings

print("Attempting to initialize ChromaDB client...")
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    print("ChromaDB client initialized successfully.")

    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Attempt to initialize OpenAI embeddings first
    if openai_api_key:
        openai_embedding_model_name = "text-embedding-ada-002" # Or another suitable OpenAI model
        print(f"Attempting to initialize OpenAI embeddings with model '{openai_embedding_model_name}'...")
        try:
            embeddings_model = OpenAIEmbeddings(api_key=openai_api_key, model=openai_embedding_model_name)
            # Perform a small test invocation to check if it works
            embeddings_model.embed_documents(["test"]) # Use embed_documents for batch embedding test
            print("Successfully initialized OpenAI embeddings.")
        except Exception as e:
            print(f"Failed to initialize OpenAI embeddings: {e}")
            embeddings_model = None # Ensure it's None if OpenAI fails
    else:
        print("OPENAI_API_KEY not found in environment variables. Skipping OpenAI embeddings initialization.")

    # If OpenAI initialization failed, attempt to initialize Ollama embeddings
    if embeddings_model is None:
        ollama_embedding_model_name = "qwen3:1.7b" # Ensure this matches sql_filter_predictor.py
        print(f"Attempting to initialize Ollama embeddings with model '{ollama_embedding_model_name}'...")
        try:
            embeddings_model = OllamaEmbeddings(model=ollama_embedding_model_name, base_url="http://localhost:11434")
             # Perform a small test invocation to check if it works
            embeddings_model.embed_documents(["test"]) # Use embed_documents for batch embedding test
            print("Successfully initialized Ollama embeddings.")
        except Exception as e:
            print(f"Failed to initialize Ollama embeddings: {e}")
            embeddings_model = None # Ensure it's None if Ollama fails

    # If neither LLM initialized, raise an error
    if embeddings_model is None:
        raise RuntimeError("Failed to initialize both OpenAI and Ollama embedding models.")

    # Define a simple embedding function wrapper for ChromaDB
    class ChromaDBEmbeddingFunction:
        def __init__(self, langchain_embeddings: object):
            self.langchain_embeddings = langchain_embeddings

        def __call__(self, input: Documents):
             # input is expected to be a list of strings (Documents type hint)
             # We assume langchain_embeddings object has an embed_documents method
             if not isinstance(input, list) or not all(isinstance(i, str) for i in input):
                  print("[EMBED] Warning: Unexpected input type for embedding function.")
                  if isinstance(input, str):
                      input = [input]
                  else:
                       raise TypeError("Input for embedding function must be a list of strings.")

             return self.langchain_embeddings.embed_documents(input)

    chroma_embedding_function = ChromaDBEmbeddingFunction(embeddings_model)
    print("Custom embedding function created.")

    # Get or create the collection for all deals
    print(f"Getting or creating ChromaDB collection: {COLLECTION_NAME}...")
    all_deals_collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Detailed context for all deals, companies, and contacts."},
        embedding_function=chroma_embedding_function # Specify the custom embedding function
    )
    print(f"Successfully accessed ChromaDB collection '{COLLECTION_NAME}'.")

except Exception as e:
    print(f"[INIT ERROR] Error initializing ChromaDB or embedding models: {e}")
    # Keep the global variables as None if initialization fails
    chroma_client = None
    all_deals_collection = None
    embeddings_model = None # Also set embeddings to None


# --- Component to extract and vectorize all deals ---
async def extract_and_vectorize_all_deals(db_config: Dict[str, Any]):
    """
    Extracts data for all deals (joined with company/contact) and vectorizes them.
    Returns a status message.
    """
    global chroma_client, embeddings_model, all_deals_collection # Declare globals to use the globally initialized variables

    # Check if both ChromaDB client and a valid embeddings model were initialized globally
    if not chroma_client or not embeddings_model or all_deals_collection is None:
        return {"status": "error", "message": "ChromaDB client, embedding model, or collection not initialized. Cannot vectorize deals. Check initialization logs."}

    print("Extracting and vectorizing all deals...")

    # --- Delete collection if it exists to ensure dimension consistency ---
    try:
        print(f"Attempting to delete existing collection '{COLLECTION_NAME}' if it exists...")
        chroma_client.delete_collection(name=COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' deleted (if it existed).")
        # After deletion, we need to get or create it again with the correct settings
        # This will happen below implicitly when we add documents, but re-fetching it here is safer.
        # However, the get_or_create call is already in the initialization block,
        # and add() will handle creation if it doesn't exist after delete.
        # So no need to explicitly re-get here.

    except Exception as e:
        # Catching potential errors if collection didn't exist, which is fine.
        print(f"No existing collection '{COLLECTION_NAME}' to delete or encountered error during deletion: {e}")

    # Re-get/create the collection after potential deletion to ensure it uses the current embeddings_model
    # This call needs to be here *after* potential deletion but *before* adding documents.
    try:
        # Use the embedding function created during global initialization
        global chroma_embedding_function # Access the globally defined embedding function
        if 'chroma_embedding_function' not in globals() or chroma_embedding_function is None:
             # This is a fallback if the global init failed but extract_and_vectorize was called
             # Re-create embedding function if it's somehow missing, using the globally initialized embeddings_model
             if embeddings_model is None:
                  raise ValueError("Embeddings model is not initialized.")
             class ChromaDBEmbeddingFunction:
                 def __init__(self, langchain_embeddings: object):
                     self.langchain_embeddings = langchain_embeddings

                 def __call__(self, input: Documents):
                      if not isinstance(input, list) or not all(isinstance(i, str) for i in input):
                           if isinstance(input, str):
                                input = [input]
                           else:
                                raise TypeError("Input for embedding function must be a list of strings.")

                      return self.langchain_embeddings.embed_documents(input)
             chroma_embedding_function = ChromaDBEmbeddingFunction(embeddings_model)
             print("Re-created custom embedding function within extract_and_vectorize.")

        # Get or create the collection again after potential deletion
        print(f"Getting or re-creating ChromaDB collection: {COLLECTION_NAME}...")
        all_deals_collection = chroma_client.get_or_create_collection(
             name=COLLECTION_NAME,
             metadata={"description": "Detailed context for all deals, companies, and contacts."},
             embedding_function=chroma_embedding_function # Specify the custom embedding function again
        )
        print(f"Successfully accessed or re-created ChromaDB collection '{COLLECTION_NAME}'.")

    except Exception as e:
        print(f"[RE-INIT ERROR] Error getting or re-creating collection after potential deletion: {e}")
        return {"status": "error", "message": f"Error getting or re-creating collection after deletion: {e}"}

    # --- Data Extraction ---
    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # SQL JOIN: deals + companies + contacts (one row per deal-contact-company combination)
        # Select columns useful for creating a detailed document
        query = """
            SELECT
                d.id AS deal_id,
                d.deal_name,
                d.amount,
                d.dealstage,
                d.closedate,
                d.days_to_close,
                d.hs_analytics_source,
                c.company_id,
                c.industry,
                c.numberofemployees,
                d.company_name,
                ct.id AS contact_id,
                ct.first_name,
                ct.last_name,
                ct.job_title
            FROM dim_deals d
            JOIN dim_companies c ON d.company_id = c.company_id
            LEFT JOIN dim_contacts ct ON d.company_id = ct.company_id
            -- Optionally add a WHERE clause here to filter deals before vectorizing if needed
            WHERE d.dealstage IN ('closedwon', 'closedlost') -- Example: only vectorize closed deals
        """ # Removed LIMIT 100 to get all relevant deals

        cursor.execute(query)
        rows = cursor.fetchall()

        documents_to_add = []
        ids_to_add = []
        metadatas_to_add = []

        for row in rows:
             # Use deal_id as a base for the doc_id, adding contact_id for uniqueness if a deal has multiple contacts
             doc_id = f"deal_{row['deal_id']}"
             if row['contact_id']:
                 doc_id += f"_contact_{row['contact_id']}"
             # Ensure uniqueness, though combining deal_id and contact_id should be sufficient per deal-contact pairing
             # If a deal could be associated with the same contact multiple times (unlikely but possible in some schemas)
             # you might need a more complex ID or handle duplicates. Assuming deal_id+contact_id is unique enough here.

             # Create a detailed document string for the deal context
             doc_content = (
                 f"Deal: '{row['deal_name']}' (ID: {row['deal_id']}). "
                 f"Status: {row['dealstage']}, Amount: ${row['amount'] or 0:.2f}, "
                 f"Sales Cycle: {row['days_to_close'] or 0:.2f} days. "
                 f"Source: {row['hs_analytics_source'] or 'N/A'}. "
                 f"Company: '{row['company_name']}' (ID: {row['company_id']}, Industry: {row['industry'] or 'N/A'}, Employees: {row['numberofemployees'] or 'N/A'}). "
                 f"Contact: {row['first_name'] or ''} {row['last_name'] or ''} (Job Title: {row['job_title'] or 'N/A'}, ID: {row['contact_id'] or 'N/A'})."
             )

             # Store key data in metadata for potential filtering or direct access after retrieval
             metadata = {
                 "deal_id": str(row['deal_id']),
                 "company_id": str(row['company_id']),
                 "contact_id": str(row['contact_id']) if row['contact_id'] else None,
                 "dealstage": row['dealstage'],
                 "amount": row['amount'],
                 "days_to_close": row['days_to_close'],
                 "hs_analytics_source": row['hs_analytics_source'],
                 "industry": row['industry'],
                 "numberofemployees": row['numberofemployees'],
                 "job_title": row['job_title']
             }

             documents_to_add.append(doc_content)
             ids_to_add.append(doc_id)
             metadatas_to_add.append(metadata)


        if documents_to_add:
            try:
                # Optional: Clear existing data in the collection before adding (for fresh data)
                # Check if collection is not empty before attempting to get ids
                # count = all_deals_collection.count()
                # if count > 0:
                #      print(f"Clearing {count} existing documents from collection '{COLLECTION_NAME}'")
                #      all_deals_collection.delete(ids=all_deals_collection.get(limit=count)['ids']) # Delete all

                # Add documents in batches to avoid potential issues with large numbers
                batch_size = 500
                for i in range(0, len(documents_to_add), batch_size):
                     batch_docs = documents_to_add[i:i+batch_size]
                     batch_ids = ids_to_add[i:i+batch_size]
                     batch_metadatas = metadatas_to_add[i:i+batch_size]
                     all_deals_collection.add(
                         documents=batch_docs,
                         ids=batch_ids,
                         metadatas=batch_metadatas
                     )
                     print(f"Added batch {i//batch_size + 1}/{len(documents_to_add)//batch_size + 1} ({len(batch_docs)} documents) to ChromaDB.")

                print(f"Finished adding {len(documents_to_add)} deal context documents to ChromaDB collection '{COLLECTION_NAME}'.")
                return {"status": "success", "message": f"Successfully extracted and vectorized {len(documents_to_add)} deals."}

            except Exception as e:
                print(f"Error adding deal documents to ChromaDB: {e}")
                return {"status": "error", "message": f"Error adding documents to ChromaDB: {e}"}

        else:
            print("No deals found to vectorize.")
            return {"status": "success", "message": "No deals found to vectorize."}

    except Exception as e:
        print(f"Error extracting deals from database: {e}")
        return {"status": "error", "message": f"Error extracting deals from database: {e}"}
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# --- Main execution block for population ---
if __name__ == "__main__":
    print("Starting Vector Database Population workflow...")
    asyncio.run(extract_and_vectorize_all_deals(DB_CONFIG))
    print("Vector Database Population workflow finished.") 