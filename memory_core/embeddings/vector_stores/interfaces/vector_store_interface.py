"""
Abstract interface for vector stores.

This module defines the base interface that all vector store implementations must implement,
enabling the Memory Engine to work with different vector database backends.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
import numpy as np


class VectorStoreType(Enum):
    """Supported vector store types."""
    MILVUS = "milvus"
    CHROMA = "chroma"
    QDRANT = "qdrant"
    FAISS = "faiss"
    NUMPY = "numpy"


class MetricType(Enum):
    """Supported similarity metrics."""
    L2 = "L2"
    IP = "IP"  # Inner Product
    COSINE = "COSINE"
    HAMMING = "HAMMING"
    JACCARD = "JACCARD"


class IndexType(Enum):
    """Supported index types."""
    FLAT = "FLAT"
    IVF_FLAT = "IVF_FLAT"
    IVF_SQ8 = "IVF_SQ8"
    IVF_PQ = "IVF_PQ"
    HNSW = "HNSW"
    ANNOY = "ANNOY"
    AUTO = "AUTO"


class VectorStoreInterface(ABC):
    """
    Abstract base class for vector store implementations.
    
    All vector store backends must implement this interface to be compatible
    with the Memory Engine's modular embedding system.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the vector store.
        
        Args:
            config: Vector store specific configuration dictionary
        """
        self.config = config
        self._dimension = config.get('dimension', 768)
        self._collection_name = config.get('collection_name', 'default_collection')
        self._metric_type = MetricType(config.get('metric_type', 'L2'))
        self._is_connected = False

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if the vector store is connected and ready."""
        pass

    @property
    def dimension(self) -> int:
        """Return the vector dimension."""
        return self._dimension

    @property
    def collection_name(self) -> str:
        """Return the collection name."""
        return self._collection_name

    @property
    def metric_type(self) -> MetricType:
        """Return the similarity metric type."""
        return self._metric_type

    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to the vector store.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Disconnect from the vector store.
        
        Returns:
            True if disconnection successful, False otherwise
        """
        pass

    @abstractmethod
    async def create_collection(
        self, 
        collection_name: str, 
        dimension: int,
        index_type: IndexType = IndexType.AUTO,
        metric_type: MetricType = MetricType.L2,
        **kwargs
    ) -> bool:
        """
        Create a new collection in the vector store.
        
        Args:
            collection_name: Name of the collection
            dimension: Vector dimension
            index_type: Type of index to create
            metric_type: Similarity metric to use
            **kwargs: Additional provider-specific parameters
            
        Returns:
            True if creation successful, False otherwise
        """
        pass

    @abstractmethod
    async def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            True if collection exists, False otherwise
        """
        pass

    @abstractmethod
    async def drop_collection(self, collection_name: str) -> bool:
        """
        Drop a collection from the vector store.
        
        Args:
            collection_name: Name of the collection to drop
            
        Returns:
            True if drop successful, False otherwise
        """
        pass

    @abstractmethod
    async def add_vectors(
        self,
        vectors: List[np.ndarray],
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        collection_name: Optional[str] = None
    ) -> List[str]:
        """
        Add vectors to the store.
        
        Args:
            vectors: List of embedding vectors
            ids: Optional list of vector IDs
            metadata: Optional list of metadata dictionaries
            collection_name: Collection to add to (uses default if None)
            
        Returns:
            List of vector IDs that were added
        """
        pass

    @abstractmethod
    async def search_vectors(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        collection_name: Optional[str] = None,
        filter_expr: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            collection_name: Collection to search (uses default if None)
            filter_expr: Optional filter expression
            
        Returns:
            List of search results with id, score, and metadata
        """
        pass

    @abstractmethod
    async def get_vector(
        self,
        vector_id: str,
        collection_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get a specific vector by ID.
        
        Args:
            vector_id: ID of the vector to retrieve
            collection_name: Collection to search (uses default if None)
            
        Returns:
            Vector data with id, vector, and metadata, or None if not found
        """
        pass

    @abstractmethod
    async def delete_vectors(
        self,
        vector_ids: List[str],
        collection_name: Optional[str] = None
    ) -> int:
        """
        Delete vectors by IDs.
        
        Args:
            vector_ids: List of vector IDs to delete
            collection_name: Collection to delete from (uses default if None)
            
        Returns:
            Number of vectors actually deleted
        """
        pass

    @abstractmethod
    async def update_vector(
        self,
        vector_id: str,
        vector: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None
    ) -> bool:
        """
        Update a vector's data or metadata.
        
        Args:
            vector_id: ID of the vector to update
            vector: New vector data (optional)
            metadata: New metadata (optional)
            collection_name: Collection containing the vector (uses default if None)
            
        Returns:
            True if update successful, False otherwise
        """
        pass

    @abstractmethod
    async def count_vectors(self, collection_name: Optional[str] = None) -> int:
        """
        Count the number of vectors in a collection.
        
        Args:
            collection_name: Collection to count (uses default if None)
            
        Returns:
            Number of vectors in the collection
        """
        pass

    @abstractmethod
    async def get_collection_info(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a collection.
        
        Args:
            collection_name: Collection to get info for (uses default if None)
            
        Returns:
            Dictionary with collection information
        """
        pass

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the vector store.
        
        Returns:
            Dictionary with health status information
        """
        try:
            is_connected = await self.connect() if not self.is_connected else True
            if not is_connected:
                return {
                    'status': 'unhealthy',
                    'error': 'Unable to connect to vector store',
                    'details': {}
                }
            
            # Try to get collection info as a basic operation test
            info = await self.get_collection_info()
            return {
                'status': 'healthy',
                'connection': True,
                'collection_info': info
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'connection': self.is_connected
            }

    def __str__(self) -> str:
        """String representation of the vector store."""
        return f"{self.__class__.__name__}(collection={self.collection_name}, dim={self.dimension})"


class VectorStoreError(Exception):
    """Base exception for vector store related errors."""
    pass


class VectorStoreConnectionError(VectorStoreError):
    """Exception raised for connection issues."""
    pass


class VectorStoreOperationError(VectorStoreError):
    """Exception raised for operation failures."""
    pass


class VectorStoreDimensionError(VectorStoreError):
    """Exception raised for dimension mismatches."""
    
    def __init__(self, expected: int, actual: int):
        super().__init__(f"Vector dimension mismatch: expected {expected}, got {actual}")
        self.expected = expected
        self.actual = actual