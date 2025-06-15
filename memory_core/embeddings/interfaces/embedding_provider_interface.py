"""
Abstract interface for embedding providers.

This module defines the base interface that all embedding providers must implement,
enabling the Memory Engine to work with different embedding services and models.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from enum import Enum
import numpy as np


class TaskType(Enum):
    """Supported embedding task types."""
    SEMANTIC_SIMILARITY = "semantic_similarity"
    RETRIEVAL_DOCUMENT = "retrieval_document"  
    RETRIEVAL_QUERY = "retrieval_query"
    CLUSTERING = "clustering"
    CLASSIFICATION = "classification"


class EmbeddingProviderInterface(ABC):
    """
    Abstract base class for embedding providers.
    
    All embedding providers must implement this interface to be compatible
    with the Memory Engine's modular embedding system.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the embedding provider.
        
        Args:
            config: Provider-specific configuration dictionary
        """
        self.config = config
        self._dimension = None
        self._max_batch_size = config.get('max_batch_size', 32)
        self._model_name = config.get('model_name', 'default')

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension for this provider."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name being used."""
        pass

    @property
    def max_batch_size(self) -> int:
        """Return the maximum batch size supported."""
        return self._max_batch_size

    @abstractmethod
    async def generate_embedding(
        self, 
        text: str, 
        task_type: TaskType = TaskType.SEMANTIC_SIMILARITY
    ) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            task_type: Type of embedding task
            
        Returns:
            Embedding vector as numpy array
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        pass

    @abstractmethod
    async def generate_embeddings(
        self, 
        texts: List[str], 
        task_type: TaskType = TaskType.SEMANTIC_SIMILARITY
    ) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts to embed
            task_type: Type of embedding task
            
        Returns:
            List of embedding vectors as numpy arrays
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the embedding provider is available and configured correctly.
        
        Returns:
            True if provider is ready to use, False otherwise
        """
        pass

    @abstractmethod
    async def test_connection(self) -> bool:
        """
        Test the connection to the embedding service.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass

    def get_supported_task_types(self) -> List[TaskType]:
        """
        Get list of supported task types for this provider.
        
        Returns:
            List of supported TaskType enums
        """
        return [TaskType.SEMANTIC_SIMILARITY, TaskType.RETRIEVAL_DOCUMENT, TaskType.RETRIEVAL_QUERY]

    def normalize_embeddings(self, embeddings: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Normalize embeddings to unit length.
        
        Args:
            embeddings: Single embedding or list of embeddings
            
        Returns:
            Normalized embeddings
        """
        if isinstance(embeddings, list):
            return [self._normalize_single(emb) for emb in embeddings]
        return self._normalize_single(embeddings)

    def _normalize_single(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize a single embedding vector."""
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm

    def __str__(self) -> str:
        """String representation of the provider."""
        return f"{self.__class__.__name__}(model={self.model_name}, dim={self.dimension})"


class EmbeddingError(Exception):
    """Base exception for embedding-related errors."""
    pass


class EmbeddingProviderError(EmbeddingError):
    """Exception raised by embedding providers."""
    
    def __init__(self, message: str, provider: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.provider = provider
        self.details = details or {}


class EmbeddingConfigError(EmbeddingError):
    """Exception raised for embedding configuration errors."""
    pass


class EmbeddingDimensionMismatchError(EmbeddingError):
    """Exception raised when embedding dimensions don't match expectations."""
    
    def __init__(self, expected: int, actual: int):
        super().__init__(f"Embedding dimension mismatch: expected {expected}, got {actual}")
        self.expected = expected
        self.actual = actual