"""
NumPy-based vector store implementation.

This package provides a lightweight, dependency-free vector store using only NumPy
for vector operations. Perfect for testing, development, and small-scale deployments
without external vector database dependencies.

Features:
- In-memory vector storage using NumPy arrays
- Efficient similarity search (L2, cosine, inner product)
- Metadata storage and filtering
- Optional persistence to disk
- Thread-safe operations
- Memory usage monitoring
- Full VectorStoreInterface compliance

Usage:
    config = {
        'collection_name': 'test_collection',
        'dimension': 768,
        'metric_type': 'COSINE',
        'persist_path': './data/vectors',
        'auto_save': True,
        'max_memory_usage': 1000  # MB
    }
    
    store = NumpyVectorStore(config)
    await store.connect()
    
    # Add vectors
    vectors = [np.random.random(768) for _ in range(100)]
    ids = await store.add_vectors(vectors)
    
    # Search
    query = np.random.random(768)
    results = await store.search_vectors(query, top_k=10)
"""

from .numpy_store import NumpyVectorStore

__all__ = ['NumpyVectorStore']