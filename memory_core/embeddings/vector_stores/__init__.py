"""
Vector stores and factory system.
"""

from typing import Dict, Type, Optional, Any, List
from memory_core.embeddings.vector_stores.interfaces import VectorStoreInterface

# Import all vector stores
from .milvus import MilvusVectorStore
from .chroma import ChromaVectorStore
from .numpy import NumpyVectorStore

# Optional imports for additional vector stores
try:
    from .qdrant import QdrantVectorStore

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

try:
    from .faiss import FAISSVectorStore

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class VectorStoreFactory:
    """
    Factory for creating vector store instances.

    This factory provides a centralized way to instantiate vector stores
    based on configuration, enabling easy switching between different backends.
    """

    # Registry of available vector stores
    _stores: Dict[str, Type[VectorStoreInterface]] = {
        "milvus": MilvusVectorStore,
        "chroma": ChromaVectorStore,
        "numpy": NumpyVectorStore,
    }

    # Register optional stores if available
    if QDRANT_AVAILABLE:
        _stores["qdrant"] = QdrantVectorStore

    if FAISS_AVAILABLE:
        _stores["faiss"] = FAISSVectorStore

    @classmethod
    def create_vector_store(cls, store_type: str, config: Dict[str, Any]) -> VectorStoreInterface:
        """
        Create a vector store instance.

        Args:
            store_type: Type of vector store ('milvus', 'chroma', 'numpy', 'qdrant', 'faiss')
            config: Vector store specific configuration dictionary

        Returns:
            Configured vector store instance

        Raises:
            ValueError: If store type is not supported
            Exception: If store initialization fails
        """
        store_type = store_type.lower()

        if store_type not in cls._stores:
            available = ", ".join(cls._stores.keys())
            raise ValueError(
                f"Unsupported vector store: {store_type}. " f"Available stores: {available}"
            )

        store_class = cls._stores[store_type]

        try:
            return store_class(config)
        except Exception as e:
            raise Exception(f"Failed to create {store_type} vector store: {str(e)}") from e

    @classmethod
    def get_available_stores(cls) -> List[str]:
        """
        Get list of available vector stores.

        Returns:
            List of vector store names
        """
        return list(cls._stores.keys())

    @classmethod
    def register_store(cls, name: str, store_class: Type[VectorStoreInterface]) -> None:
        """
        Register a new vector store.

        Args:
            name: Name of the vector store
            store_class: Store class implementing VectorStoreInterface
        """
        if not issubclass(store_class, VectorStoreInterface):
            raise ValueError(f"Store class must implement VectorStoreInterface")

        cls._stores[name.lower()] = store_class

    @classmethod
    def get_store_class(cls, store_type: str) -> Type[VectorStoreInterface]:
        """
        Get the vector store class for a given type.

        Args:
            store_type: Type of vector store

        Returns:
            Vector store class

        Raises:
            ValueError: If store type is not supported
        """
        store_type = store_type.lower()

        if store_type not in cls._stores:
            available = ", ".join(cls._stores.keys())
            raise ValueError(
                f"Unsupported vector store: {store_type}. " f"Available stores: {available}"
            )

        return cls._stores[store_type]

    @classmethod
    def get_store_capabilities(cls, store_type: str) -> Dict[str, Any]:
        """
        Get capabilities and features of a vector store.

        Args:
            store_type: Type of vector store

        Returns:
            Dictionary describing store capabilities
        """
        capabilities = {
            "milvus": {
                "distributed": True,
                "persistent": True,
                "scalable": True,
                "gpu_support": True,
                "external_service": True,
                "production_ready": True,
            },
            "chroma": {
                "distributed": False,
                "persistent": True,
                "scalable": False,
                "gpu_support": False,
                "external_service": False,
                "production_ready": True,
            },
            "numpy": {
                "distributed": False,
                "persistent": True,
                "scalable": False,
                "gpu_support": False,
                "external_service": False,
                "production_ready": False,
            },
            "qdrant": {
                "distributed": True,
                "persistent": True,
                "scalable": True,
                "gpu_support": False,
                "external_service": True,
                "production_ready": True,
            },
            "faiss": {
                "distributed": False,
                "persistent": True,
                "scalable": True,
                "gpu_support": True,
                "external_service": False,
                "production_ready": True,
            },
        }

        store_type = store_type.lower()
        return capabilities.get(store_type, {})


# Convenience function for creating vector stores
def create_vector_store(store_type: str, config: Dict[str, Any]) -> VectorStoreInterface:
    """
    Create a vector store instance.

    Args:
        store_type: Type of vector store ('milvus', 'chroma', 'numpy', 'qdrant', 'faiss')
        config: Vector store specific configuration dictionary

    Returns:
        Configured vector store instance
    """
    return VectorStoreFactory.create_vector_store(store_type, config)


__all__ = [
    "VectorStoreFactory",
    "create_vector_store",
    "MilvusVectorStore",
    "ChromaVectorStore",
    "NumpyVectorStore",
]

# Add optional exports if available
if QDRANT_AVAILABLE:
    __all__.append("QdrantVectorStore")

if FAISS_AVAILABLE:
    __all__.append("FAISSVectorStore")
