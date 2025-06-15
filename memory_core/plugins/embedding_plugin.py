"""
Embedding provider plugin interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import numpy as np
from .plugin_manager import PluginInterface


class EmbeddingPluginInterface(PluginInterface):
    """Interface for embedding provider plugins."""

    @property
    @abstractmethod
    def plugin_type(self) -> str:
        """Return 'embedding'."""
        return "embedding"

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this provider."""
        pass

    @abstractmethod
    async def generate_embedding(self, text: str, **kwargs) -> np.ndarray:
        """Generate embedding for a single text."""
        pass

    @abstractmethod
    async def generate_embeddings(self, texts: List[str], **kwargs) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        pass

    async def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        try:
            # Cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(dot_product / (norm1 * norm2))
        except Exception:
            return 0.0

    async def find_similar_embeddings(
        self, query_embedding: np.ndarray, candidate_embeddings: List[np.ndarray], top_k: int = 5
    ) -> List[tuple]:
        """Find top-k similar embeddings."""
        similarities = []

        for i, candidate in enumerate(candidate_embeddings):
            similarity = await self.compute_similarity(query_embedding, candidate)
            similarities.append((i, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    async def test_embedding(self, test_text: str = "Hello world") -> bool:
        """Test embedding generation."""
        try:
            embedding = await self.generate_embedding(test_text)
            return embedding is not None and len(embedding) == self.embedding_dimension
        except Exception:
            return False

    async def get_embedding_info(self) -> Dict[str, Any]:
        """Get information about the embedding provider."""
        return {
            "provider": self.name,
            "dimension": self.embedding_dimension,
            "version": self.version,
        }

    async def encode_batch(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Encode texts in batches for efficiency."""
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = await self.generate_embeddings(batch)
            embeddings.extend(batch_embeddings)

        return embeddings

    def normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding to unit length."""
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm

    async def compute_centroid(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """Compute centroid of multiple embeddings."""
        if not embeddings:
            return np.zeros(self.embedding_dimension)

        stacked = np.stack(embeddings)
        return np.mean(stacked, axis=0)


class EmbeddingPlugin(EmbeddingPluginInterface):
    """Base class for embedding plugins."""

    def __init__(self):
        self._config = {}
        self._model_name = None
        self._dimension = None
        self._initialized = False

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name or "unknown"

    @property
    def embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension or 768  # Default dimension

    @property
    def is_initialized(self) -> bool:
        """Check if plugin is initialized."""
        return self._initialized

    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the embedding plugin."""
        self._config = config
        self._model_name = config.get("model_name", "default")
        self._dimension = config.get("dimension", 768)

        # Test embedding generation
        try:
            test_result = await self.test_embedding()
            self._initialized = test_result
            return self._initialized
        except Exception:
            self._initialized = False
            return False

    async def shutdown(self) -> bool:
        """Shutdown the embedding plugin."""
        self._initialized = False
        return True

    async def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate embedding configuration."""
        # Basic validation - can be overridden
        if "model_name" not in config:
            return False

        dimension = config.get("dimension")
        if dimension is not None and (not isinstance(dimension, int) or dimension <= 0):
            return False

        return True
