import sys
import os
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional

# Print ChromaDB version and path for debugging
print(f"ChromaDB library version: {chromadb.__version__}")
print(f"ChromaDB library path: {chromadb.__file__}")

import logging
import http.client as http_client

# Enable full HTTP request/response logging for debugging
http_client.HTTPConnection.debuglevel = 1 
logging.basicConfig(level=logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)
requests_log = logging.getLogger("urllib3.connectionpool") # More specific logger for urllib3
requests_log.setLevel(logging.DEBUG)
requests_log.propagate = True

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import settings

class ChromaVectorClient:
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None,
        embedding_model_name: Optional[str] = None
    ):
        self.host = host or settings.CHROMA_HOST
        self.port = port or settings.CHROMA_PORT
        self.collection_name = collection_name or settings.DEFAULT_CHROMA_COLLECTION
        self.embedding_model_name = embedding_model_name or settings.DEFAULT_EMBEDDING_MODEL
        
        try:
            # Connect to ChromaDB server
            print(f"Attempting to connect to ChromaDB at {self.host}:{str(self.port)}...")
            # Try a simpler way to initialize the client, with only supported settings
            self.client = chromadb.HttpClient(
                host=self.host, 
                port=str(self.port),
                headers={},  # Empty headers to avoid default auth headers
                settings=chromadb.config.Settings(
                    anonymized_telemetry=False,  # Disable telemetry which might trigger auth
                    allow_reset=True             # Allow reset operations (older API behavior)
                )
            )

            # Initialize the sentence transformer embedding function
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model_name
            )
            print(f"Initialized SentenceTransformer embedding model: {self.embedding_model_name}")

            # Get or create the collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function # Pass the function, not just the name
            )
            print(f"Using ChromaDB collection: '{self.collection_name}'")
            
        except Exception as e:
            print(f"Error initializing ChromaVectorClient: {e}")
            print("Please ensure ChromaDB is running and accessible, and the embedding model is valid.")
            self.client = None
            self.collection = None
            self.embedding_function = None

    def add_documents(
        self, 
        documents: List[str], 
        metadatas: Optional[List[Dict[str, Any]]] = None, 
        ids: Optional[List[str]] = None
    ) -> bool:
        """
        Adds documents to the ChromaDB collection.
        Embeddings are generated automatically by the collection's embedding function.

        Args:
            documents: A list of document texts to add.
            metadatas: An optional list of metadata dictionaries, one for each document.
            ids: An optional list of unique IDs, one for each document.

        Returns:
            True if documents were added successfully, False otherwise.
        """
        if not self.collection:
            print("ChromaDB collection not initialized. Cannot add documents.")
            return False
        if not documents:
            print("No documents provided to add.")
            return False
        if ids and len(ids) != len(documents):
            print("Number of IDs must match the number of documents.")
            return False
        if metadatas and len(metadatas) != len(documents):
            print("Number of metadatas must match the number of documents.")
            return False

        try:
            print(f"Adding {len(documents)} documents to collection '{self.collection_name}'...")
            self.collection.add(
                documents=documents,
                metadatas=metadatas, # type: ignore # Chroma client expects list[Document] which is fine
                ids=ids # type: ignore
            )
            print(f"Successfully added {len(documents)} documents.")
            return True
        except Exception as e:
            print(f"Error adding documents to ChromaDB: {e}")
            return False

    def query_collection(
        self, 
        query_texts: List[str], 
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None, # For metadata filtering
        include: Optional[List[str]] = ["metadatas", "documents", "distances"]
    ) -> Optional[Dict[str, Any]]:
        """
        Queries the collection for documents similar to the query_texts.

        Args:
            query_texts: A list of query texts.
            n_results: The number of results to return for each query.
            where: Optional metadata filter.
            include: Optional list of fields to include in the results.

        Returns:
            A dictionary containing the query results, or None if an error occurs.
        """
        if not self.collection:
            print("ChromaDB collection not initialized. Cannot query collection.")
            return None
        
        try:
            print(f"Querying collection '{self.collection_name}' with {len(query_texts)} query(s)...")
            results = self.collection.query(
                query_texts=query_texts,
                n_results=n_results,
                where=where,
                include=include # type: ignore # Chroma client expects list[QueryInclude] which is fine
            )
            print(f"Query successful. Retrieved results for {len(query_texts)} query(s).")
            return results
        except Exception as e:
            print(f"Error querying ChromaDB collection: {e}")
            return None

    def get_collection_count(self) -> Optional[int]:
        if not self.collection:
            print("ChromaDB collection not initialized.")
            return None
        return self.collection.count()

# Example Usage:
if __name__ == "__main__":
    print("Testing ChromaVectorClient...")
    # Ensure ChromaDB server is running (e.g., via docker compose up)
    vector_client = ChromaVectorClient()

    if vector_client.collection:
        print(f"Initial collection count: {vector_client.get_collection_count()}")

        # Test adding documents
        test_docs = [
            "This is document 1 about apples and oranges.",
            "Document 2 discusses bananas and grapes.",
            "The third document is a story about a clever fox.",
            "Another document about various fruits like apples, bananas, and cherries."
        ]
        test_metadatas = [
            {"source": "doc1", "category": "fruit"},
            {"source": "doc2", "category": "fruit"},
            {"source": "doc3", "category": "story"},
            {"source": "doc4", "category": "fruit"}
        ]
        test_ids = ["id1", "id2", "id3", "id4"]

        added = vector_client.add_documents(documents=test_docs, metadatas=test_metadatas, ids=test_ids)
        if added:
            print(f"Collection count after adding: {vector_client.get_collection_count()}")

            # Test querying
            query_results = vector_client.query_collection(query_texts=["information about apples", "tales"], n_results=2)
            if query_results:
                print("\nQuery Results for 'information about apples':")
                for i, doc in enumerate(query_results['documents'][0]): # Results for the first query
                    print(f"  - Doc: {doc}")
                    print(f"    Dist: {query_results['distances'][0][i]}")
                    print(f"    Meta: {query_results['metadatas'][0][i]}")
                
                print("\nQuery Results for 'tales':")
                for i, doc in enumerate(query_results['documents'][1]): # Results for the second query
                    print(f"  - Doc: {doc}")
                    print(f"    Dist: {query_results['distances'][1][i]}")
                    print(f"    Meta: {query_results['metadatas'][1][i]}")
            
            # Example: Get or create collection again (should get existing)
            print("\nAttempting to get_or_create_collection again...")
            collection_again = vector_client.client.get_or_create_collection(
                name=vector_client.collection_name,
                embedding_function=vector_client.embedding_function
            )
            print(f"Got collection: '{collection_again.name}' with count: {collection_again.count()}")
    else:
        print("ChromaVectorClient initialization failed. Skipping tests.") 