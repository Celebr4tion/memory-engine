"""
Vector storage implementation using Milvus.

This module provides an implementation of vector storage using Milvus,
allowing for efficient storage and retrieval of embeddings.
"""
import logging
import time
from typing import List, Dict, Any, Optional, Union

from memory_core.config import get_config

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
    """
    Milvus implementation of the vector store.
    """

    def __init__(
        self, 
        host: Optional[str] = None, 
        port: Optional[Union[int, str]] = None,
        collection_name: Optional[str] = None, 
        dimension: Optional[int] = None
    ):
        """
        Initialize the Milvus vector store.

        Args:
            host: Milvus server host (optional, will use config if not provided)
            port: Milvus server port (optional, will use config if not provided)
            collection_name: Name of the collection to use (optional, will use config if not provided)
            dimension: Dimension of the embedding vectors (optional, will use config if not provided)
        """
        self.config = get_config()
        self.host = host or self.config.config.vector_store.milvus.host
        self.port = port or self.config.config.vector_store.milvus.port
        self.collection_name = collection_name or self.config.config.vector_store.milvus.collection_name
        self.dimension = dimension or self.config.config.vector_store.milvus.dimension
        self.logger = logging.getLogger(__name__)
        self.connected = False
        self.index_params = {
            "metric_type": self.config.config.vector_store.milvus.metric_type,
            "index_type": self.config.config.vector_store.milvus.index_type,
            "params": {"nlist": self.config.config.vector_store.milvus.nlist}
        }

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
                connect_args = {"host": self.host, "port": self.port, "timeout": 10}
                
                # Add authentication if configured
                if self.config.config.vector_store.milvus.user:
                    connect_args["user"] = self.config.config.vector_store.milvus.user
                if self.config.config.vector_store.milvus.password:
                    connect_args["password"] = self.config.config.vector_store.milvus.password
                
                connections.connect("default", **connect_args)
                self.logger.info(f"Connected to Milvus server at {self.host}:{self.port}")
                
                # Create collection if it doesn't exist
                self._ensure_collection()
                self.connected = True
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
            self.connected = False
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
        
        # Create index and flush collection
        self.logger.info(f"Creating index for collection {self.collection_name}...")
        self.collection.create_index(field_name="embedding", index_params=self.index_params)
        self.collection.flush()
        
        # Verify index exists
        index_info = self.collection.index()
        self.logger.info(f"Created index: {index_info}")

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
            expr = f'node_id in ["{node_id}"]'
            self.collection.delete(expr)
        
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
        
        results = self.collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=self.index_params,
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
        
        # Use Milvus compatible delete expression
        expr = f'node_id in ["{node_id}"]'
        result = self.collection.delete(expr)
        self.collection.flush()
        return True