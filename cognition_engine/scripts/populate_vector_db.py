import os
import json
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# Import for Vector DB and Embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

# --- Pinecone Configuration ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "deals-index")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# --- Embedding Model Setup (same as before) ---
openai_api_key = os.getenv("OPENAI_API_KEY")
embeddings_model: Optional[object] = None
if openai_api_key:
    try:
        embeddings_model = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-ada-002")
        embeddings_model.embed_documents(["test"])
    except Exception as e:
        print(f"Failed to initialize OpenAI embeddings: {e}")
        embeddings_model = None
if embeddings_model is None:
    try:
        embeddings_model = OllamaEmbeddings(model="qwen3:1.7b", base_url="http://localhost:11434")
        embeddings_model.embed_documents(["test"])
    except Exception as e:
        print(f"Failed to initialize Ollama embeddings: {e}")
        embeddings_model = None
if embeddings_model is None:
    raise RuntimeError("Failed to initialize embedding model.")

# --- Ensure Pinecone index exists ---
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,  # or your embedding dimension
        metric='cosine', # or 'euclidean', as needed
        spec=ServerlessSpec(
            cloud='aws',      # or 'gcp'
            region='us-east-1' # or your region
        )
    )
index = pc.Index(PINECONE_INDEX_NAME)

# --- Database Configuration (unchanged) ---
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://cognition_user:cognition_password@localhost:5432/cognition_db")
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", ""),
    "user": os.getenv("DB_USER", "cognition_user"),
    "password": os.getenv("DB_PASSWORD", "cognition_password"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
}
import psycopg2
import psycopg2.extras

# --- Main function to extract and vectorize all deals ---
async def extract_and_vectorize_all_deals(db_config: Dict[str, Any]):
    print("Extracting and vectorizing all deals for Pinecone...")
    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
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
                c.country,
                c.employee_bucket,
                c.numberofemployees,
                d.company_name,
                ct.id AS contact_id,
                ct.first_name,
                ct.last_name,
                ct.job_title
            FROM dim_deals d
            JOIN dim_companies c ON d.company_id = c.company_id
            LEFT JOIN dim_contacts ct ON d.company_id = ct.company_id
            WHERE d.dealstage IN ('closedwon', 'closedlost')
            AND d.amount IS NOT NULL
            AND d.days_to_close IS NOT NULL
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        print(f"\nDebug - Total deals fetched from database: {len(rows)}")
        
        pinecone_vectors = []
        for row in rows:
            doc_id = f"deal_{row['deal_id']}"
            if row['contact_id']:
                doc_id += f"_contact_{row['contact_id']}"
            
            # Debug print for first few deals
            if len(pinecone_vectors) < 5:
                print(f"\nDebug - Processing deal {row['deal_id']}:")
                print(f"Deal Name: {row['deal_name']}")
                print(f"Amount: ${row['amount']}")
                print(f"Stage: {row['dealstage']}")
                print(f"Days to Close: {row['days_to_close']}")
            
            doc_content = (
                f"Deal: '{row['deal_name']}' (ID: {row['deal_id']}). "
                f"Status: {row['dealstage']}, Amount: ${row['amount'] or 0:.2f}, "
                f"Sales Cycle: {row['days_to_close'] or 0:.2f} days. "
                f"Source: {row['hs_analytics_source'] or 'N/A'}. "
                f"Company: '{row['company_name']}' (ID: {row['company_id']}, Industry: {row['industry'] or 'N/A'}, Country: {row.get('country') or 'N/A'}, Employee Bucket: {row.get('employee_bucket') or 'N/A'}, Employees: {row['numberofemployees'] or 'N/A'}). "
                f"Contact: {row['first_name'] or ''} {row['last_name'] or ''} (Job Title: {row['job_title'] or 'N/A'}, ID: {row['contact_id'] or 'N/A'})."
            )
            metadata = {
                "deal_id": str(row['deal_id']),
                "company_id": str(row['company_id']),
                "contact_id": str(row['contact_id']) if row['contact_id'] is not None else "N/A",
                "dealstage": row['dealstage'] if row['dealstage'] is not None else "N/A",
                "amount": float(row['amount']) if row['amount'] is not None else 0.0,
                "days_to_close": float(row['days_to_close']) if row['days_to_close'] is not None else 0.0,
                "hs_analytics_source": row['hs_analytics_source'] if row['hs_analytics_source'] is not None else "N/A",
                "industry": row['industry'] if row['industry'] is not None else "N/A",
                "country": row['country'] if row['country'] is not None else "N/A",
                "employee_bucket": row['employee_bucket'] if row['employee_bucket'] is not None else "N/A",
                "numberofemployees": int(row['numberofemployees']) if row['numberofemployees'] is not None else 0,
                "job_title": row['job_title'] if row['job_title'] is not None else "N/A"
            }
            try:
                # Get embedding
                embedding = embeddings_model.embed_documents([doc_content])[0]
                pinecone_vectors.append((doc_id, embedding, metadata))
            except Exception as e:
                print(f"Error processing deal {row['deal_id']}: {e}")
                continue
        # Upsert to Pinecone
        if pinecone_vectors:
            print(f"Upserting {len(pinecone_vectors)} vectors to Pinecone index '{PINECONE_INDEX_NAME}'...")
            index.upsert(vectors=pinecone_vectors)
            print(f"Finished upserting to Pinecone.")
        else:
            print("No deals found to vectorize.")
        return {"status": "success", "message": f"Successfully extracted and vectorized {len(pinecone_vectors)} deals to Pinecone."}
    except Exception as e:
        print(f"Error extracting deals from database: {e}")
        return {"status": "error", "message": f"Error extracting deals from database: {e}"}
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

if __name__ == "__main__":
    print("Starting Pinecone Vector Database Population workflow...")
    asyncio.run(extract_and_vectorize_all_deals(DB_CONFIG))
    print("Pinecone Vector Database Population workflow finished.")