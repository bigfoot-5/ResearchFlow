import psycopg2
import psycopg2.extras
import os
import json
import asyncio
from typing import Dict, List, Any, Sequence, Optional
from datetime import datetime, timedelta

# Import for Vector DB and Embeddings
import chromadb
from langchain_community.embeddings import OllamaEmbeddings # Use langchain_community
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
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    # Initialize Ollama embeddings (using langchain_community)
    # Use a suitable embedding model
    embedding_model = "qwen3:1.7b" # Recommended for embeddings
    ollama_embeddings = OllamaEmbeddings(model=embedding_model, base_url="http://localhost:11434")

    class ChromaDBEmbeddingFunction:
        def __init__(self, langchain_embeddings: OllamaEmbeddings):
            self.langchain_embeddings = langchain_embeddings

        def __call__(self, input: Documents):
            return self.langchain_embeddings.embed_documents(input)

    chroma_embedding_function = ChromaDBEmbeddingFunction(ollama_embeddings)

    # Get or create the collection for all deals
    all_deals_collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Detailed context for all deals, companies, and contacts."},
        embedding_function=chroma_embedding_function # Specify the custom embedding function
    )
    print(f"Initialized ChromaDB collection for all deals: {COLLECTION_NAME}")
    chroma_initialized = True
except Exception as e:
    print(f"Error initializing ChromaDB for all deals: {e}")
    all_deals_collection = None
    chroma_initialized = False


# --- Component to extract and vectorize all deals ---
async def extract_and_vectorize_all_deals(db_config: Dict[str, Any]):
    """
    Extracts data for all deals (joined with company/contact) and vectorizes them.
    Returns a status message.
    """
    if not chroma_initialized or all_deals_collection is None:
        return {"status": "error", "message": "ChromaDB not initialized. Cannot vectorize deals."}

    print("Extracting and vectorizing all deals...")
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