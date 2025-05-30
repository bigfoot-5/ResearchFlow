# Import required libraries
import psycopg2
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import chromadb
import os
from chromadb.api.types import Documents

# Database connection parameters
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://cognition_user:cognition_password@localhost:5432/cognition_db")

# Initialize PostgreSQL connection
conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()

# Example predefined filter: only closed-won deals with amount > 10000
PREDEFINED_FILTER = """
    d.dealstage = 'closedwon'
    AND d.amount > 10000
"""

# SQL JOIN: deals + companies + contacts (one row per deal-contact-company)
sql = f"""
SELECT
    d.id AS deal_id,
    d.deal_name,
    d.amount,
    d.dealstage,
    d.closedate,
    d.days_to_close,
    c.company_id,
    c.industry,
    c.numberofemployees,
    c.company_name,
    ct.id AS contact_id,
    ct.first_name,
    ct.last_name,
    ct.job_title
FROM dim_deals d
JOIN dim_companies c ON d.company_id = c.company_id
LEFT JOIN dim_contacts ct ON ct.company_id = c.company_id
WHERE {PREDEFINED_FILTER}
LIMIT 5
"""

# Fetch data from PostgreSQL by executing the joined query
cursor.execute(sql)
rows = cursor.fetchall()

# Initialize Ollama embeddings
embedding = OllamaEmbeddings(model="qwen3:1.7b", base_url="http://localhost:11434")

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), "chroma_db"))

# Define a custom embedding function for ChromaDB using Ollama
class ChromaDBEmbeddingFunction:
    def __init__(self, langchain_embeddings: OllamaEmbeddings):
        self.langchain_embeddings = langchain_embeddings

    def __call__(self, input: Documents):
        embeddings = self.langchain_embeddings.embed_documents(input)
        return embeddings

# Initialize the embedding function with Ollama embeddings
embedding_function = ChromaDBEmbeddingFunction(embedding)

# Define a collection for the RAG workflow
collection_name = "rag_collection_demo_1"
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    metadata={"description": "A collection for RAG with Ollama - Demo1"},
    embedding_function=embedding_function
)
documents = []
ids = []

# Iterate over each row fetched from the database
for row in rows:
    deal_id = str(row[0])  # Convert the deal_id to a string
    deal_name = row[1] if row[1] else "UNKNOWN"
    amount = row[2] if row[2] else "UNKNOWN"
    dealstage = row[3] if row[3] else "UNKNOWN"
    closedate = row[4] if row[4] else "UNKNOWN"
    days_to_close = row[5] if row[5] else "UNKNOWN"
    company_id = str(row[6])  # Convert the company_id to a string
    industry = row[7] if row[7] else "UNKNOWN"
    num_employees = row[8] if row[8] else "UNKNOWN"
    company_name = row[9] if row[9] else "UNKNOWN"
    contact_id = str(row[10]) if row[10] else "no_contact"
    first_name = row[11] if row[11] else "UNKNOWN"
    last_name = row[12] if row[12] else "UNKNOWN"
    job_title = row[13] if row[13] else "UNKNOWN"
    
    # Create a string representation of the row
    content = f"Deal with ID {deal_id} is in the {dealstage} stage, has an amount of {amount}, and is related to the company {company_name} in the {industry} industry. The company has {num_employees} employees."
    
    # Append the content and ID to the respective lists
    documents.append(content)
    ids.append(f"{deal_id}_{contact_id or 'no_contact'}")

# Add all documents to the ChromaDB collection in a single batch
collection.add(documents=documents, ids=ids)

# Close PostgreSQL connection
cursor.close()
conn.close()

print(f"Inserted {len(documents)} joined deal-company-contact records into the vector database.")

# Define the LLM model to be used
llm_model = "qwen3:1.7b"

# Configure ChromaDB
# Initialize the ChromaDB client with persistent storage in the current directory
chroma_client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), "chroma_db"))

# Define a custom embedding function for ChromaDB using Ollama
class ChromaDBEmbeddingFunction:
    """
    Custom embedding function for ChromaDB using embeddings from Ollama.
    """
    def __init__(self, langchain_embeddings):
        self.langchain_embeddings = langchain_embeddings

    def __call__(self, input):
        # Ensure the input is in a list format for processing
        if isinstance(input, str):
            input = [input]
        return self.langchain_embeddings.embed_documents(input)

# Initialize the embedding function with Ollama embeddings
embedding = ChromaDBEmbeddingFunction(
    OllamaEmbeddings(
        model=llm_model,
        base_url="http://localhost:11434"  # Adjust the base URL as per your Ollama server configuration
    )
)

# Define a collection for the RAG workflow
collection_name = "rag_collection_demo_1"
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    metadata={"description": "A collection for RAG with Ollama - Demo1"},
    embedding_function=embedding  # Use the custom embedding function
)

# Function to add documents to the ChromaDB collection
def add_documents_to_collection(documents, ids):
    """
    Add documents to the ChromaDB collection.
    
    Args:
        documents (list of str): The documents to add.
        ids (list of str): Unique IDs for the documents.
    """
    collection.add(
        documents=documents,
        ids=ids
    )

# Example: Add sample documents to the collection
documents = [
    "Artificial intelligence is the simulation of human intelligence processes by machines.",
    "Python is a programming language that lets you work quickly and integrate systems more effectively.",
    "ChromaDB is a vector database designed for AI applications."
]
doc_ids = ["doc1", "doc2", "doc3"]

# Documents only need to be added once or whenever an update is required. 
# This line of code is included for demonstration purposes:
# add_documents_to_collection(documents, doc_ids)

# Function to query the ChromaDB collection
def query_chromadb(query_text, n_results=1):
    """
    Query the ChromaDB collection for relevant documents.
    
    Args:
        query_text (str): The input query.
        n_results (int): The number of top results to return.
    
    Returns:
        list of dict: The top matching documents and their metadata.
    """
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    return results["documents"], results["metadatas"]

# Function to interact with the Ollama LLM
def query_ollama(prompt):
    """
    Send a query to Ollama and retrieve the response.
    
    Args:
        prompt (str): The input prompt for Ollama.
    
    Returns:
        str: The response from Ollama.
    """
    llm = OllamaLLM(model=llm_model)
    return llm.invoke(prompt)

# RAG pipeline: Combine ChromaDB and Ollama for Retrieval-Augmented Generation
def rag_pipeline(query_text, n_results=5):
    """
    Perform Retrieval-Augmented Generation (RAG) by combining ChromaDB and Ollama.
    
    Args:
        query_text (str): The input query.
        n_results (int): Number of top documents to retrieve.
    
    Returns:
        str: The generated response from Ollama augmented with retrieved context.
    """
    # Step 1: Retrieve relevant documents from ChromaDB
    retrieved_docs, metadata = query_chromadb(query_text, n_results=n_results)
    
    # Flatten the list of retrieved documents
    context = " ".join(doc for doc_list in retrieved_docs for doc in doc_list) if retrieved_docs else "No relevant documents found."

    # Step 2: Send the query along with the context to Ollama
    augmented_prompt = f"Context: {context}\n\nQuestion: {query_text}\nAnswer:"
    print("######## Augmented Prompt ########")
    print(augmented_prompt)

    response = query_ollama(augmented_prompt)
    return response

# Example usage
# Define a query to test the RAG pipeline
# query = "What can you say about company ID  145102662894. How many employees are there in the ID?"
query="Give me all company_ids which have industry as 'PHARMACEUTICALS. Also give me how many employees in this industry in total.'"   # Change the query as needed
response = rag_pipeline(query)
print("######## Response from LLM ########\n", response)