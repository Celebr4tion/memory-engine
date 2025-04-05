"""
Vector storage implementation using Milvus.

This module provides an implementation of vector storage using Milvus,
allowing for efficient storage and retrieval of embeddings.
"""
import logging
from typing import List, Dict, Any, Optional
import time

try:
    from pymilvus import (
        connections,
        utility,
        FieldSchema,
        CollectionSchema,
        DataType,
        Collection,
    )
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False

class VectorStoreMilvus:
    """Vector storage implementation using Milvus."""

    def __init__(self, host: str = "localhost", port: int = 19530, 
                 collection_name: str = "knowledge_embeddings",
                 dimension: int = 256):
        """
        Initialize the Milvus vector store.

        Args:
            host: Milvus server host
            port: Milvus server port
            collection_name: Name of the collection to store embeddings
            dimension: Dimension of the embedding vectors
        """
        if not MILVUS_AVAILABLE:
            raise ImportError(
                "pymilvus is not installed. Please install it with 'pip install pymilvus'"
            )
        
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dimension = dimension
        self.collection = None
        self.logger = logging.getLogger(__name__)

    def connect(self, max_retries: int = 5, retry_interval: int = 10) -> bool:
        """
        Connect to Milvus server and ensure the collection exists.
        
        Includes retry logic to handle cases where Milvus is still initializing.

        Args:
            max_retries: Maximum number of connection attempts
            retry_interval: Time in seconds between retries

        Returns:
            bool: True if connection and setup were successful
        """
        for attempt in range(1, max_retries + 1):
            try:
                # Connect to Milvus server
                self.logger.info(f"Attempt {attempt}/{max_retries}: Connecting to Milvus at {self.host}:{self.port}")
                connections.connect("default", host=self.host, port=self.port, timeout=30)
                self.logger.info(f"Connected to Milvus server at {self.host}:{self.port}")
                
                # Create collection if it doesn't exist
                self._ensure_collection()
                return True
                
            except Exception as e:
                self.logger.warning(f"Connection attempt {attempt} failed: {str(e)}")
                
                # Special handling for "not ready yet" errors
                if "Milvus Proxy is not ready yet" in str(e) and attempt < max_retries:
                    self.logger.info(f"Milvus is still initializing. Waiting {retry_interval} seconds before retry...")
                elif attempt < max_retries:
                    self.logger.info(f"Retrying in {retry_interval} seconds...")
                else:
                    self.logger.error(f"Failed to connect to Milvus after {max_retries} attempts: {str(e)}")
                    return False
                
                # Ensure we're disconnected before retrying
                try:
                    connections.disconnect("default")
                except:
                    pass
                    
                time.sleep(retry_interval)
        
        return False

    def disconnect(self) -> None:
        """Disconnect from Milvus server."""
        try:
            # Release collection if loaded
            if self.collection:
                try:
                    self.collection.release()
                except Exception as e:
                    self.logger.warning(f"Error releasing collection: {str(e)}")
            
            # Disconnect from server
            connections.disconnect("default")
            self.logger.info("Disconnected from Milvus server")
        except Exception as e:
            self.logger.error(f"Error disconnecting from Milvus: {str(e)}")

    def _ensure_collection(self) -> None:
        """Ensure the collection exists and is loaded."""
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            self.logger.info(f"Using existing collection: {self.collection_name}")
        else:
            self._create_collection()
            self.logger.info(f"Created new collection: {self.collection_name}")
        
        # Load collection
        self.collection.load()

    def _create_collection(self) -> None:
        """Create a new collection for storing embeddings."""
        # Define fields for the collection
        fields = [
            FieldSchema(name="node_id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema(name="timestamp", dtype=DataType.DOUBLE)
        ]
        
        # Create schema and collection
        schema = CollectionSchema(fields=fields, description="Knowledge node embeddings")
        self.collection = Collection(name=self.collection_name, schema=schema)
        
        # Create index on vector field
        index_params = {
            "metric_type": "L2",
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 64}
        }
        self.collection.create_index(field_name="embedding", index_params=index_params)

    def add_embedding(self, node_id: str, embedding: List[float]) -> None:
        """
        Add an embedding vector for a node to the vector store.

        Args:
            node_id: ID of the knowledge node
            embedding: The embedding vector of the node
        """
        if len(embedding) != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch. Expected {self.dimension}, got {len(embedding)}"
            )
        
        if not self.collection:
            raise RuntimeError("Not connected to Milvus. Call connect() first.")
        
        # Check if node_id already exists
        query_result = self.collection.query(
            expr=f'node_id == "{node_id}"',
            output_fields=["node_id"]
        )
        
        # If node_id exists, delete it first
        if query_result:
            self.collection.delete(f'node_id == "{node_id}"')
        
        # Insert new embedding
        data = [
            [node_id],  # node_id
            [embedding],  # embedding vector
            [time.time()]  # timestamp
        ]
        
        self.collection.insert(data)
        self.collection.flush()
        self.logger.info(f"Added embedding for node {node_id}")

    def search_embedding(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings.

        Args:
            query_vector: The query embedding vector
            top_k: Number of top matches to return

        Returns:
            List of dictionaries containing node_id and similarity score
        """
        if len(query_vector) != self.dimension:
            raise ValueError(
                f"Query vector dimension mismatch. Expected {self.dimension}, got {len(query_vector)}"
            )
        
        if not self.collection:
            raise RuntimeError("Not connected to Milvus. Call connect() first.")
        
        search_params = {"metric_type": "L2", "params": {"ef": 64}}
        
        results = self.collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["node_id"]
        )
        
        matches = []
        if results and results[0]:
            for hit in results[0]:
                matches.append({
                    "node_id": hit.entity.get("node_id"),
                    "score": hit.score
                })
        
        return matches

    def get_node_ids(self, query_vector: List[float], top_k: int = 5) -> List[str]:
        """
        Search for similar embeddings and return only the node IDs.

        Args:
            query_vector: The query embedding vector
            top_k: Number of top matches to return

        Returns:
            List of node IDs of the most similar vectors
        """
        matches = self.search_embedding(query_vector, top_k)
        return [match["node_id"] for match in matches]

    def delete_embedding(self, node_id: str) -> bool:
        """
        Delete an embedding for a node from the vector store.

        Args:
            node_id: ID of the knowledge node

        Returns:
            bool: True if deletion was successful
        """
        if not self.collection:
            raise RuntimeError("Not connected to Milvus. Call connect() first.")
        
        result = self.collection.delete(f'node_id == "{node_id}"')
        self.collection.flush()
        return True 