"""
Vector Store Manager for Cognition Engine.

This module provides abstractions for interacting with vector databases
to store and retrieve data using semantic similarity.
"""

import os
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreManager:
    """
    Manages interactions with the vector database.
    
    This class provides an abstraction over the vector database (ChromaDB),
    allowing for storage and retrieval of data using semantic similarity.
    """
    
    def __init__(
        self,
        collection_name: str = "default",
        embedding_function = None,
        persist_directory: str = "data/vector_db",
    ):
        """
        Initialize the vector store manager.
        
        Args:
            collection_name: Name of the collection to use
            embedding_function: Function to use for embedding documents (defaults to system default)
            persist_directory: Directory where vector DB data is stored
        """
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.persist_directory = persist_directory
        
        # In a complete implementation, this would initialize a ChromaDB client
        # For now, we'll use a simple in-memory dict to allow the agents to function
        self._data = {}
        
        # Ensure the persistence directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        logger.info(f"Initialized VectorStoreManager with collection: {collection_name}")
        
        # Try to load existing data if available
        self._load_from_disk()
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document texts to add
            metadatas: Optional metadata for each document
            ids: Optional IDs for each document (generated if not provided)
            
        Returns:
            List of document IDs that were added
        """
        # Generate IDs if not provided
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        
        # Use empty metadata if not provided
        if metadatas is None:
            metadatas = [{} for _ in range(len(documents))]
        
        # Add documents to our simple storage
        for i, doc_id in enumerate(ids):
            self._data[doc_id] = {
                "document": documents[i],
                "metadata": metadatas[i],
                "added_at": datetime.now().isoformat()
            }
        
        # In a real implementation, this would add to ChromaDB
        logger.info(f"Added {len(documents)} documents to collection {self.collection_name}")
        
        # Save to disk
        self._save_to_disk()
        
        return ids
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query.
        
        Args:
            query: The search query
            n_results: Number of results to return
            where: Optional filter criteria
            
        Returns:
            List of matching documents with their metadata and scores
        """
        # In a real implementation, this would use ChromaDB's similarity search
        # For our simplified version, we'll just return up to n_results documents
        # that optionally match the where filter
        
        results = []
        for doc_id, doc_data in self._data.items():
            # Apply filter if provided
            if where is not None:
                matches_filter = True
                for key, value in where.items():
                    if key not in doc_data["metadata"] or doc_data["metadata"][key] != value:
                        matches_filter = False
                        break
                
                if not matches_filter:
                    continue
            
            # Add to results
            results.append({
                "id": doc_id,
                "document": doc_data["document"],
                "metadata": doc_data["metadata"],
                "distance": 0.5  # Dummy score since we're not actually computing vector similarity
            })
            
            # Stop once we have enough results
            if len(results) >= n_results:
                break
        
        logger.info(f"Found {len(results)} results for query in collection {self.collection_name}")
        return results
    
    def get(self, ids: List[str]) -> Dict[str, Any]:
        """
        Get documents by their IDs.
        
        Args:
            ids: List of document IDs to retrieve
            
        Returns:
            Dictionary containing documents, metadatas, and IDs
        """
        documents = []
        metadatas = []
        found_ids = []
        
        for doc_id in ids:
            if doc_id in self._data:
                documents.append(self._data[doc_id]["document"])
                metadatas.append(self._data[doc_id]["metadata"])
                found_ids.append(doc_id)
        
        return {
            "documents": documents,
            "metadatas": metadatas,
            "ids": found_ids
        }
    
    def delete(self, ids: Optional[List[str]] = None, where: Optional[Dict[str, Any]] = None) -> None:
        """
        Delete documents from the vector store.
        
        Args:
            ids: Optional list of document IDs to delete
            where: Optional filter criteria for documents to delete
        """
        if ids is not None:
            # Delete by IDs
            for doc_id in ids:
                if doc_id in self._data:
                    del self._data[doc_id]
            
            logger.info(f"Deleted {len(ids)} documents by ID from collection {self.collection_name}")
        
        elif where is not None:
            # Delete by filter criteria
            docs_to_delete = []
            for doc_id, doc_data in self._data.items():
                matches_filter = True
                for key, value in where.items():
                    if key not in doc_data["metadata"] or doc_data["metadata"][key] != value:
                        matches_filter = False
                        break
                
                if matches_filter:
                    docs_to_delete.append(doc_id)
            
            for doc_id in docs_to_delete:
                del self._data[doc_id]
            
            logger.info(f"Deleted {len(docs_to_delete)} documents by filter from collection {self.collection_name}")
        
        # Save changes to disk
        self._save_to_disk()
    
    def _save_to_disk(self) -> None:
        """Save the current data to disk."""
        collection_file = os.path.join(self.persist_directory, f"{self.collection_name}.json")
        try:
            with open(collection_file, "w") as f:
                json.dump(self._data, f, indent=2)
            logger.info(f"Saved collection {self.collection_name} to disk")
        except Exception as e:
            logger.error(f"Error saving collection to disk: {e}")
    
    def _load_from_disk(self) -> None:
        """Load data from disk if available."""
        collection_file = os.path.join(self.persist_directory, f"{self.collection_name}.json")
        if os.path.exists(collection_file):
            try:
                with open(collection_file, "r") as f:
                    self._data = json.load(f)
                logger.info(f"Loaded collection {self.collection_name} from disk with {len(self._data)} documents")
            except Exception as e:
                logger.error(f"Error loading collection from disk: {e}")
                # Initialize with empty data
                self._data = {} 