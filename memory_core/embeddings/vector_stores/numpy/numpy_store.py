"""
NumPy-based vector store implementation.

This module provides a lightweight, dependency-free vector store implementation
using only NumPy for vector operations. Perfect for testing, development, and
small-scale deployments without external vector database dependencies.
"""

import asyncio
import logging
import pickle
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from memory_core.embeddings.vector_stores.interfaces import (
    VectorStoreInterface,
    VectorStoreType,
    MetricType,
    IndexType,
    VectorStoreError,
    VectorStoreConnectionError,
    VectorStoreOperationError,
    VectorStoreDimensionError,
)


class NumpyVectorStore(VectorStoreInterface):
    """
    NumPy-based implementation of the vector store interface.

    Provides in-memory vector storage and similarity search using only NumPy.
    Supports optional persistence to disk using pickle/numpy serialization.
    Thread-safe operations with efficient similarity search algorithms.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the NumPy vector store.

        Args:
            config: Configuration dictionary with keys:
                - collection_name: Collection name (default: 'default_collection')
                - dimension: Vector dimension (default: 768)
                - metric_type: Similarity metric ('L2', 'COSINE', 'IP') (default: 'L2')
                - persist_path: Path to save data (None for memory-only)
                - auto_save: Auto-save on changes (default: True)
                - max_memory_usage: Max memory usage in MB (default: 1000)
        """
        super().__init__(config)

        self.logger = logging.getLogger(__name__)

        # Configuration
        self.persist_path = config.get("persist_path")
        self.auto_save = config.get("auto_save", True)
        self.max_memory_usage = config.get("max_memory_usage", 1000)  # MB

        # Thread safety
        self._lock = Lock()

        # Storage structures
        self._collections: Dict[str, Dict[str, Any]] = {}
        self._initialize_collection(self.collection_name)

        # Load from disk if path specified
        if self.persist_path:
            self._load_from_disk()

    @property
    def is_connected(self) -> bool:
        """Check if the vector store is ready (always True for in-memory store)."""
        return self._is_connected

    def _initialize_collection(self, collection_name: str):
        """Initialize a new collection structure."""
        self._collections[collection_name] = {
            "vectors": np.empty((0, self.dimension), dtype=np.float32),
            "ids": [],
            "metadata": [],
            "timestamps": [],
            "dimension": self.dimension,
            "metric_type": self.metric_type,
            "created_at": time.time(),
            "updated_at": time.time(),
        }

    async def connect(self) -> bool:
        """
        Connect to the vector store (no-op for in-memory store).

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.logger.info("Connecting to NumPy vector store")
            self._is_connected = True
            self.logger.info("Successfully connected to NumPy vector store")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to NumPy vector store: {str(e)}")
            self._is_connected = False
            raise VectorStoreConnectionError(f"Failed to connect: {str(e)}")

    async def disconnect(self) -> bool:
        """
        Disconnect from the vector store.

        Returns:
            True if disconnection successful, False otherwise
        """
        try:
            # Save data if persistence is enabled
            if self.persist_path and self.auto_save:
                await self._save_to_disk()

            self._is_connected = False
            self.logger.info("Disconnected from NumPy vector store")
            return True
        except Exception as e:
            self.logger.error(f"Error disconnecting from NumPy vector store: {str(e)}")
            return False

    async def create_collection(
        self,
        collection_name: str,
        dimension: int,
        index_type: IndexType = IndexType.AUTO,
        metric_type: MetricType = MetricType.L2,
        **kwargs,
    ) -> bool:
        """
        Create a new collection.

        Args:
            collection_name: Name of the collection
            dimension: Vector dimension
            index_type: Type of index (ignored for NumPy store)
            metric_type: Similarity metric to use
            **kwargs: Additional parameters

        Returns:
            True if creation successful, False otherwise
        """
        try:
            with self._lock:
                if collection_name in self._collections:
                    self.logger.info(f"Collection {collection_name} already exists")
                    return True

                # Create new collection with specified parameters
                self._collections[collection_name] = {
                    "vectors": np.empty((0, dimension), dtype=np.float32),
                    "ids": [],
                    "metadata": [],
                    "timestamps": [],
                    "dimension": dimension,
                    "metric_type": metric_type,
                    "created_at": time.time(),
                    "updated_at": time.time(),
                }

                self.logger.info(f"Created collection {collection_name} with dimension {dimension}")

                # Auto-save if enabled
                if self.persist_path and self.auto_save:
                    await self._save_to_disk()

                return True

        except Exception as e:
            self.logger.error(f"Failed to create collection {collection_name}: {str(e)}")
            raise VectorStoreOperationError(f"Failed to create collection: {str(e)}")

    async def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists.

        Args:
            collection_name: Name of the collection

        Returns:
            True if collection exists, False otherwise
        """
        with self._lock:
            return collection_name in self._collections

    async def drop_collection(self, collection_name: str) -> bool:
        """
        Drop a collection.

        Args:
            collection_name: Name of the collection to drop

        Returns:
            True if drop successful, False otherwise
        """
        try:
            with self._lock:
                if collection_name not in self._collections:
                    return False

                del self._collections[collection_name]
                self.logger.info(f"Dropped collection {collection_name}")

                # Auto-save if enabled
                if self.persist_path and self.auto_save:
                    await self._save_to_disk()

                return True

        except Exception as e:
            self.logger.error(f"Failed to drop collection {collection_name}: {str(e)}")
            raise VectorStoreOperationError(f"Failed to drop collection: {str(e)}")

    async def add_vectors(
        self,
        vectors: List[np.ndarray],
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        collection_name: Optional[str] = None,
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
        if not vectors:
            return []

        collection_name = collection_name or self.collection_name

        # Ensure collection exists
        if not await self.collection_exists(collection_name):
            await self.create_collection(collection_name, self.dimension)

        with self._lock:
            collection = self._collections[collection_name]

            # Validate dimensions
            for i, vector in enumerate(vectors):
                if len(vector) != collection["dimension"]:
                    raise VectorStoreDimensionError(collection["dimension"], len(vector))

            # Generate IDs if not provided
            if ids is None:
                ids = [f"vec_{uuid.uuid4().hex}" for _ in range(len(vectors))]

            if len(ids) != len(vectors):
                raise VectorStoreOperationError("Number of IDs must match number of vectors")

            # Check for duplicate IDs
            existing_ids = set(collection["ids"])
            duplicate_ids = [id_ for id_ in ids if id_ in existing_ids]
            if duplicate_ids:
                raise VectorStoreOperationError(f"Duplicate IDs found: {duplicate_ids}")

            try:
                # Convert vectors to numpy array
                vectors_array = np.array([vec.astype(np.float32) for vec in vectors])

                # Check memory usage
                self._check_memory_usage(vectors_array.nbytes)

                # Append to existing vectors
                if collection["vectors"].size == 0:
                    collection["vectors"] = vectors_array
                else:
                    collection["vectors"] = np.vstack([collection["vectors"], vectors_array])

                # Add IDs and metadata
                collection["ids"].extend(ids)

                if metadata:
                    if len(metadata) != len(vectors):
                        raise VectorStoreOperationError(
                            "Number of metadata entries must match number of vectors"
                        )
                    collection["metadata"].extend(metadata)
                else:
                    collection["metadata"].extend([{}] * len(vectors))

                # Add timestamps
                current_time = time.time()
                collection["timestamps"].extend([current_time] * len(vectors))
                collection["updated_at"] = current_time

                self.logger.info(f"Added {len(vectors)} vectors to collection {collection_name}")

                # Auto-save if enabled
                if self.persist_path and self.auto_save:
                    await self._save_to_disk()

                return ids

            except Exception as e:
                self.logger.error(f"Failed to add vectors: {str(e)}")
                raise VectorStoreOperationError(f"Failed to add vectors: {str(e)}")

    async def search_vectors(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        collection_name: Optional[str] = None,
        filter_expr: Optional[Dict[str, Any]] = None,
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
        collection_name = collection_name or self.collection_name

        if not await self.collection_exists(collection_name):
            return []

        with self._lock:
            collection = self._collections[collection_name]

            # Validate dimension
            if len(query_vector) != collection["dimension"]:
                raise VectorStoreDimensionError(collection["dimension"], len(query_vector))

            if collection["vectors"].size == 0:
                return []

            try:
                # Apply filters if provided
                valid_indices = self._apply_filters(collection, filter_expr)

                if not valid_indices:
                    return []

                # Get filtered vectors
                filtered_vectors = collection["vectors"][valid_indices]

                # Compute similarities
                similarities = self._compute_similarities(
                    query_vector.astype(np.float32), filtered_vectors, collection["metric_type"]
                )

                # Get top-k results
                if len(similarities) <= top_k:
                    top_indices = np.argsort(-similarities)  # Descending order
                else:
                    top_indices = np.argpartition(-similarities, top_k)[:top_k]
                    top_indices = top_indices[np.argsort(-similarities[top_indices])]

                # Build results
                results = []
                for idx in top_indices:
                    original_idx = valid_indices[idx]
                    result = {
                        "id": collection["ids"][original_idx],
                        "score": float(similarities[idx]),
                        "metadata": collection["metadata"][original_idx],
                    }
                    results.append(result)

                return results

            except Exception as e:
                self.logger.error(f"Failed to search vectors: {str(e)}")
                raise VectorStoreOperationError(f"Failed to search vectors: {str(e)}")

    async def get_vector(
        self, vector_id: str, collection_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get a specific vector by ID.

        Args:
            vector_id: ID of the vector to retrieve
            collection_name: Collection to search (uses default if None)

        Returns:
            Vector data with id, vector, and metadata, or None if not found
        """
        collection_name = collection_name or self.collection_name

        if not await self.collection_exists(collection_name):
            return None

        with self._lock:
            collection = self._collections[collection_name]

            try:
                if vector_id not in collection["ids"]:
                    return None

                idx = collection["ids"].index(vector_id)

                return {
                    "id": vector_id,
                    "vector": collection["vectors"][idx].copy(),
                    "metadata": collection["metadata"][idx].copy(),
                }

            except Exception as e:
                self.logger.error(f"Failed to get vector {vector_id}: {str(e)}")
                raise VectorStoreOperationError(f"Failed to get vector: {str(e)}")

    async def delete_vectors(
        self, vector_ids: List[str], collection_name: Optional[str] = None
    ) -> int:
        """
        Delete vectors by IDs.

        Args:
            vector_ids: List of vector IDs to delete
            collection_name: Collection to delete from (uses default if None)

        Returns:
            Number of vectors actually deleted
        """
        if not vector_ids:
            return 0

        collection_name = collection_name or self.collection_name

        if not await self.collection_exists(collection_name):
            return 0

        with self._lock:
            collection = self._collections[collection_name]

            try:
                # Find indices to delete
                indices_to_delete = []
                deleted_count = 0

                for vector_id in vector_ids:
                    if vector_id in collection["ids"]:
                        idx = collection["ids"].index(vector_id)
                        indices_to_delete.append(idx)
                        deleted_count += 1

                if not indices_to_delete:
                    return 0

                # Sort indices in descending order to delete from end
                indices_to_delete.sort(reverse=True)

                # Delete from all arrays
                for idx in indices_to_delete:
                    collection["vectors"] = np.delete(collection["vectors"], idx, axis=0)
                    del collection["ids"][idx]
                    del collection["metadata"][idx]
                    del collection["timestamps"][idx]

                collection["updated_at"] = time.time()

                self.logger.info(
                    f"Deleted {deleted_count} vectors from collection {collection_name}"
                )

                # Auto-save if enabled
                if self.persist_path and self.auto_save:
                    await self._save_to_disk()

                return deleted_count

            except Exception as e:
                self.logger.error(f"Failed to delete vectors: {str(e)}")
                raise VectorStoreOperationError(f"Failed to delete vectors: {str(e)}")

    async def update_vector(
        self,
        vector_id: str,
        vector: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None,
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
        collection_name = collection_name or self.collection_name

        if not await self.collection_exists(collection_name):
            return False

        with self._lock:
            collection = self._collections[collection_name]

            try:
                if vector_id not in collection["ids"]:
                    return False

                idx = collection["ids"].index(vector_id)

                # Update vector if provided
                if vector is not None:
                    if len(vector) != collection["dimension"]:
                        raise VectorStoreDimensionError(collection["dimension"], len(vector))
                    collection["vectors"][idx] = vector.astype(np.float32)

                # Update metadata if provided
                if metadata is not None:
                    collection["metadata"][idx] = metadata.copy()

                collection["updated_at"] = time.time()

                # Auto-save if enabled
                if self.persist_path and self.auto_save:
                    await self._save_to_disk()

                return True

            except Exception as e:
                self.logger.error(f"Failed to update vector {vector_id}: {str(e)}")
                raise VectorStoreOperationError(f"Failed to update vector: {str(e)}")

    async def count_vectors(self, collection_name: Optional[str] = None) -> int:
        """
        Count the number of vectors in a collection.

        Args:
            collection_name: Collection to count (uses default if None)

        Returns:
            Number of vectors in the collection
        """
        collection_name = collection_name or self.collection_name

        if not await self.collection_exists(collection_name):
            return 0

        with self._lock:
            return len(self._collections[collection_name]["ids"])

    async def get_collection_info(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a collection.

        Args:
            collection_name: Collection to get info for (uses default if None)

        Returns:
            Dictionary with collection information
        """
        collection_name = collection_name or self.collection_name

        if not await self.collection_exists(collection_name):
            raise VectorStoreOperationError(f"Collection {collection_name} does not exist")

        with self._lock:
            collection = self._collections[collection_name]

            return {
                "name": collection_name,
                "num_entities": len(collection["ids"]),
                "dimension": collection["dimension"],
                "metric_type": collection["metric_type"].value,
                "index_type": "FLAT",  # NumPy store uses flat indexing
                "created_at": collection["created_at"],
                "updated_at": collection["updated_at"],
                "memory_usage_mb": self._get_collection_memory_usage(collection) / (1024 * 1024),
            }

    def _compute_similarities(
        self, query_vector: np.ndarray, vectors: np.ndarray, metric_type: MetricType
    ) -> np.ndarray:
        """
        Compute similarities between query vector and stored vectors.

        Args:
            query_vector: Query vector
            vectors: Stored vectors
            metric_type: Similarity metric to use

        Returns:
            Array of similarity scores
        """
        if metric_type == MetricType.L2:
            # L2 distance (convert to similarity: higher is better)
            distances = np.linalg.norm(vectors - query_vector, axis=1)
            # Convert distance to similarity (inverse relationship)
            return 1.0 / (1.0 + distances)

        elif metric_type == MetricType.COSINE:
            # Cosine similarity
            query_norm = np.linalg.norm(query_vector)
            vectors_norm = np.linalg.norm(vectors, axis=1)

            # Handle zero vectors
            query_norm = max(query_norm, 1e-8)
            vectors_norm = np.maximum(vectors_norm, 1e-8)

            dot_products = np.dot(vectors, query_vector)
            similarities = dot_products / (vectors_norm * query_norm)

            # Ensure similarities are in [-1, 1] range
            return np.clip(similarities, -1.0, 1.0)

        elif metric_type == MetricType.IP:
            # Inner product (dot product)
            return np.dot(vectors, query_vector)

        else:
            raise VectorStoreOperationError(f"Unsupported metric type: {metric_type}")

    def _apply_filters(
        self, collection: Dict[str, Any], filter_expr: Optional[Dict[str, Any]]
    ) -> List[int]:
        """
        Apply filters to collection and return valid indices.

        Args:
            collection: Collection data
            filter_expr: Filter expression

        Returns:
            List of valid indices
        """
        if not filter_expr:
            return list(range(len(collection["ids"])))

        valid_indices = []

        for i, metadata in enumerate(collection["metadata"]):
            if self._matches_filter(metadata, filter_expr):
                valid_indices.append(i)

        return valid_indices

    def _matches_filter(self, metadata: Dict[str, Any], filter_expr: Dict[str, Any]) -> bool:
        """
        Check if metadata matches filter expression.

        Args:
            metadata: Metadata dictionary
            filter_expr: Filter expression

        Returns:
            True if metadata matches filter
        """
        for key, value in filter_expr.items():
            if key not in metadata:
                return False

            if isinstance(value, dict):
                # Handle complex filters (e.g., {"$gt": 5})
                for op, op_value in value.items():
                    if op == "$gt" and metadata[key] <= op_value:
                        return False
                    elif op == "$lt" and metadata[key] >= op_value:
                        return False
                    elif op == "$gte" and metadata[key] < op_value:
                        return False
                    elif op == "$lte" and metadata[key] > op_value:
                        return False
                    elif op == "$eq" and metadata[key] != op_value:
                        return False
                    elif op == "$ne" and metadata[key] == op_value:
                        return False
                    elif op == "$in" and metadata[key] not in op_value:
                        return False
                    elif op == "$nin" and metadata[key] in op_value:
                        return False
            else:
                # Simple equality check
                if metadata[key] != value:
                    return False

        return True

    def _check_memory_usage(self, additional_bytes: int):
        """
        Check if adding vectors would exceed memory limit.

        Args:
            additional_bytes: Number of additional bytes
        """
        current_usage = sum(
            self._get_collection_memory_usage(collection)
            for collection in self._collections.values()
        )

        total_usage_mb = (current_usage + additional_bytes) / (1024 * 1024)

        if total_usage_mb > self.max_memory_usage:
            raise VectorStoreOperationError(
                f"Memory limit exceeded: {total_usage_mb:.2f}MB > {self.max_memory_usage}MB"
            )

    def _get_collection_memory_usage(self, collection: Dict[str, Any]) -> int:
        """
        Get memory usage of a collection in bytes.

        Args:
            collection: Collection data

        Returns:
            Memory usage in bytes
        """
        vectors_size = collection["vectors"].nbytes if collection["vectors"].size > 0 else 0
        ids_size = sum(len(id_.encode("utf-8")) for id_ in collection["ids"])
        metadata_size = sum(len(str(meta).encode("utf-8")) for meta in collection["metadata"])
        timestamps_size = len(collection["timestamps"]) * 8  # 8 bytes per float

        return vectors_size + ids_size + metadata_size + timestamps_size

    async def _save_to_disk(self):
        """Save collections to disk."""
        if not self.persist_path:
            return

        try:
            persist_path = Path(self.persist_path)
            persist_path.mkdir(parents=True, exist_ok=True)

            # Save collections data
            collections_file = persist_path / "collections.pkl"
            with open(collections_file, "wb") as f:
                pickle.dump(self._collections, f, protocol=pickle.HIGHEST_PROTOCOL)

            self.logger.debug(f"Saved collections to {collections_file}")

        except Exception as e:
            self.logger.error(f"Failed to save to disk: {str(e)}")
            raise VectorStoreOperationError(f"Failed to save to disk: {str(e)}")

    def _load_from_disk(self):
        """Load collections from disk."""
        if not self.persist_path:
            return

        try:
            persist_path = Path(self.persist_path)
            collections_file = persist_path / "collections.pkl"

            if collections_file.exists():
                with open(collections_file, "rb") as f:
                    loaded_collections = pickle.load(f)

                # Merge with existing collections
                for name, data in loaded_collections.items():
                    self._collections[name] = data

                self.logger.info(f"Loaded {len(loaded_collections)} collections from disk")

        except Exception as e:
            self.logger.warning(f"Failed to load from disk: {str(e)}")
            # Don't raise exception - continue with empty collections

    async def save_collection(self, collection_name: Optional[str] = None):
        """
        Manually save a collection to disk.

        Args:
            collection_name: Collection to save (saves all if None)
        """
        if not self.persist_path:
            raise VectorStoreOperationError("Persistence not configured")

        await self._save_to_disk()
        self.logger.info(f"Manually saved collections to disk")

    async def load_collection(self, collection_name: Optional[str] = None):
        """
        Manually load a collection from disk.

        Args:
            collection_name: Collection to load (loads all if None)
        """
        if not self.persist_path:
            raise VectorStoreOperationError("Persistence not configured")

        self._load_from_disk()
        self.logger.info(f"Manually loaded collections from disk")

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.

        Returns:
            Dictionary with memory statistics
        """
        with self._lock:
            total_vectors = sum(len(col["ids"]) for col in self._collections.values())
            total_memory = sum(
                self._get_collection_memory_usage(col) for col in self._collections.values()
            )

            collection_stats = {}
            for name, collection in self._collections.items():
                collection_stats[name] = {
                    "num_vectors": len(collection["ids"]),
                    "memory_mb": self._get_collection_memory_usage(collection) / (1024 * 1024),
                    "dimension": collection["dimension"],
                }

            return {
                "total_collections": len(self._collections),
                "total_vectors": total_vectors,
                "total_memory_mb": total_memory / (1024 * 1024),
                "max_memory_mb": self.max_memory_usage,
                "memory_utilization": (total_memory / (1024 * 1024)) / self.max_memory_usage,
                "collections": collection_stats,
            }
