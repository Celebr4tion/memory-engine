"""
ChromaDB vector store implementation.

This module provides a ChromaDB implementation of the VectorStoreInterface,
allowing for efficient storage and retrieval of embeddings using ChromaDB.
"""

import logging
import time
import asyncio
from typing import List, Dict, Any, Optional, Union
import numpy as np
import uuid

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

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


class ChromaVectorStore(VectorStoreInterface):
    """
    ChromaDB implementation of the vector store interface.

    Provides high-performance vector storage and similarity search using ChromaDB.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ChromaDB vector store.

        Args:
            config: Configuration dictionary with keys:
                - path: Database path (None for in-memory, default: None)
                - collection_name: Collection name (default: 'default_collection')
                - dimension: Vector dimension (default: 768)
                - metric_type: Similarity metric (default: 'L2')
                - batch_size: Batch size for operations (default: 1000)
                - host: Optional host for server mode
                - port: Optional port for server mode
                - ssl: Use SSL for server mode (default: False)
                - headers: Optional headers for server mode
        """
        if not CHROMADB_AVAILABLE:
            raise VectorStoreError(
                "ChromaDB is not available. Install chromadb: pip install chromadb"
            )

        super().__init__(config)

        self.logger = logging.getLogger(__name__)

        # ChromaDB-specific configuration
        self.path = config.get("path")  # None for in-memory
        self.batch_size = config.get("batch_size", 1000)
        self.host = config.get("host")
        self.port = config.get("port", 8000)
        self.ssl = config.get("ssl", False)
        self.headers = config.get("headers", {})

        # Client and collection state
        self.client: Optional[chromadb.Client] = None
        self.collection: Optional[chromadb.Collection] = None

        # Convert metric type for ChromaDB
        self._chroma_distance = self._convert_metric_type()

    @property
    def is_connected(self) -> bool:
        """Check if connected to ChromaDB."""
        return self._is_connected and self.client is not None

    def _convert_metric_type(self) -> str:
        """Convert MetricType to ChromaDB distance function."""
        mapping = {MetricType.L2: "l2", MetricType.COSINE: "cosine", MetricType.IP: "ip"}
        return mapping.get(self.metric_type, "l2")

    async def connect(self) -> bool:
        """
        Connect to ChromaDB.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.logger.info("Connecting to ChromaDB")

            # Determine client type based on configuration
            if self.host:
                # HTTP client for server mode
                self.logger.info(f"Connecting to ChromaDB server at {self.host}:{self.port}")
                self.client = chromadb.HttpClient(
                    host=self.host, port=self.port, ssl=self.ssl, headers=self.headers
                )
            elif self.path:
                # Persistent client
                self.logger.info(f"Using persistent ChromaDB at {self.path}")
                self.client = chromadb.PersistentClient(path=self.path)
            else:
                # In-memory client
                self.logger.info("Using in-memory ChromaDB")
                self.client = chromadb.Client()

            # Test connection by listing collections
            await self._run_async(self.client.list_collections)

            # Ensure default collection exists
            await self._ensure_collection()

            self._is_connected = True
            self.logger.info("Successfully connected to ChromaDB")
            return True

        except Exception as e:
            self.logger.error(f"Failed to connect to ChromaDB: {str(e)}")
            self._is_connected = False
            raise VectorStoreConnectionError(f"Failed to connect to ChromaDB: {str(e)}")

    async def disconnect(self) -> bool:
        """
        Disconnect from ChromaDB.

        Returns:
            True if disconnection successful, False otherwise
        """
        try:
            # ChromaDB clients don't need explicit disconnection
            self.client = None
            self.collection = None
            self._is_connected = False
            self.logger.info("Disconnected from ChromaDB")
            return True

        except Exception as e:
            self.logger.error(f"Error disconnecting from ChromaDB: {str(e)}")
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
        Create a new collection in ChromaDB.

        Args:
            collection_name: Name of the collection
            dimension: Vector dimension (ChromaDB infers this automatically)
            index_type: Type of index to create (ChromaDB handles automatically)
            metric_type: Similarity metric to use
            **kwargs: Additional parameters

        Returns:
            True if creation successful, False otherwise
        """
        try:
            # Check if collection already exists
            existing_collections = await self._run_async(self.client.list_collections)
            for col in existing_collections:
                if col.name == collection_name:
                    self.logger.info(f"Collection {collection_name} already exists")
                    return True

            # Convert metric type
            distance_function = self._convert_metric_type()
            if metric_type != self.metric_type:
                mapping = {MetricType.L2: "l2", MetricType.COSINE: "cosine", MetricType.IP: "ip"}
                distance_function = mapping.get(metric_type, "l2")

            # Create collection
            metadata = kwargs.get("metadata", {})
            metadata.update({"dimension": dimension, "created_at": time.time()})

            collection = await self._run_async(
                self.client.create_collection,
                name=collection_name,
                metadata=metadata,
                embedding_function=None,  # We'll provide embeddings directly
                distance=distance_function,
            )

            self.logger.info(f"Created collection {collection_name} with dimension {dimension}")
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
        try:
            collections = await self._run_async(self.client.list_collections)
            return any(col.name == collection_name for col in collections)
        except Exception as e:
            self.logger.error(f"Error checking collection existence: {str(e)}")
            return False

    async def drop_collection(self, collection_name: str) -> bool:
        """
        Drop a collection from ChromaDB.

        Args:
            collection_name: Name of the collection to drop

        Returns:
            True if drop successful, False otherwise
        """
        try:
            if await self.collection_exists(collection_name):
                await self._run_async(self.client.delete_collection, name=collection_name)
                self.logger.info(f"Dropped collection {collection_name}")

                # Reset collection if it was the current one
                if self.collection and self.collection.name == collection_name:
                    self.collection = None

                return True
            return False

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
        Add vectors to ChromaDB.

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

        collection = await self._get_collection(collection_name)

        # Validate dimensions
        for i, vector in enumerate(vectors):
            if len(vector) != self.dimension:
                raise VectorStoreDimensionError(self.dimension, len(vector))

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]

        if len(ids) != len(vectors):
            raise VectorStoreOperationError("Number of IDs must match number of vectors")

        # Process in batches
        all_added_ids = []
        for i in range(0, len(vectors), self.batch_size):
            batch_end = min(i + self.batch_size, len(vectors))
            batch_vectors = vectors[i:batch_end]
            batch_ids = ids[i:batch_end]
            batch_metadata = metadata[i:batch_end] if metadata else None

            try:
                # Convert numpy arrays to lists
                embeddings = [
                    vector.tolist() if isinstance(vector, np.ndarray) else vector
                    for vector in batch_vectors
                ]

                # Add default metadata if none provided
                if batch_metadata is None:
                    batch_metadata = [{"timestamp": time.time()} for _ in batch_ids]
                else:
                    # Ensure metadata has required fields
                    for meta in batch_metadata:
                        if "timestamp" not in meta:
                            meta["timestamp"] = time.time()

                # Add to collection
                await self._run_async(
                    collection.add, embeddings=embeddings, metadatas=batch_metadata, ids=batch_ids
                )

                all_added_ids.extend(batch_ids)

            except Exception as e:
                self.logger.error(f"Failed to add batch {i//self.batch_size + 1}: {str(e)}")
                raise VectorStoreOperationError(f"Failed to add vectors: {str(e)}")

        self.logger.info(f"Added {len(all_added_ids)} vectors to collection {collection.name}")
        return all_added_ids

    async def search_vectors(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        collection_name: Optional[str] = None,
        filter_expr: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in ChromaDB.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            collection_name: Collection to search (uses default if None)
            filter_expr: Optional filter expression (ChromaDB where clause format)

        Returns:
            List of search results with id, score, and metadata
        """
        collection = await self._get_collection(collection_name)

        # Validate dimension
        if len(query_vector) != self.dimension:
            raise VectorStoreDimensionError(self.dimension, len(query_vector))

        try:
            # Convert query vector to list
            query_embedding = (
                query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector
            )

            # Build where clause from filter
            where_clause = self._build_where_clause(filter_expr) if filter_expr else None

            # Perform search
            results = await self._run_async(
                collection.query,
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause,
                include=["metadatas", "distances"],
            )

            # Parse results
            matches = []
            if results and results.get("ids") and results["ids"][0]:
                ids = results["ids"][0]
                distances = results.get("distances", [None])[0] or []
                metadatas = results.get("metadatas", [None])[0] or []

                for i, vector_id in enumerate(ids):
                    result = {
                        "id": vector_id,
                        "score": float(distances[i]) if i < len(distances) else 0.0,
                    }

                    # Add metadata if available
                    if i < len(metadatas) and metadatas[i]:
                        result["metadata"] = metadatas[i]

                    matches.append(result)

            return matches

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
        collection = await self._get_collection(collection_name)

        try:
            # Get vector by ID
            results = await self._run_async(
                collection.get, ids=[vector_id], include=["metadatas", "embeddings"]
            )

            if not results or not results.get("ids") or not results["ids"]:
                return None

            # Extract data
            vector_data = {"id": results["ids"][0]}

            # Add embedding if available
            if results.get("embeddings") and results["embeddings"]:
                vector_data["vector"] = np.array(results["embeddings"][0])

            # Add metadata if available
            if results.get("metadatas") and results["metadatas"] and results["metadatas"][0]:
                vector_data["metadata"] = results["metadatas"][0]

            return vector_data

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

        collection = await self._get_collection(collection_name)

        try:
            # ChromaDB delete by IDs
            await self._run_async(collection.delete, ids=vector_ids)

            self.logger.info(f"Deleted {len(vector_ids)} vectors from collection {collection.name}")
            return len(vector_ids)

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
        collection = await self._get_collection(collection_name)

        try:
            # Get current vector data
            current_data = await self.get_vector(vector_id, collection_name)
            if not current_data:
                return False

            # Prepare update data
            update_args = {"ids": [vector_id]}

            # Update vector if provided
            if vector is not None:
                if len(vector) != self.dimension:
                    raise VectorStoreDimensionError(self.dimension, len(vector))
                embedding = vector.tolist() if isinstance(vector, np.ndarray) else vector
                update_args["embeddings"] = [embedding]

            # Update metadata if provided
            if metadata is not None:
                # Merge with existing metadata
                current_metadata = current_data.get("metadata", {})
                updated_metadata = {**current_metadata, **metadata}
                updated_metadata["timestamp"] = time.time()  # Update timestamp
                update_args["metadatas"] = [updated_metadata]

            # Perform update
            await self._run_async(collection.update, **update_args)

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
        collection = await self._get_collection(collection_name)

        try:
            # ChromaDB count
            return await self._run_async(collection.count)
        except Exception as e:
            self.logger.error(f"Failed to count vectors: {str(e)}")
            raise VectorStoreOperationError(f"Failed to count vectors: {str(e)}")

    async def get_collection_info(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a collection.

        Args:
            collection_name: Collection to get info for (uses default if None)

        Returns:
            Dictionary with collection information
        """
        collection = await self._get_collection(collection_name)

        try:
            count = await self.count_vectors(collection_name)

            # Get collection metadata
            metadata = {}
            if hasattr(collection, "metadata") and collection.metadata:
                metadata = collection.metadata

            return {
                "name": collection.name,
                "num_entities": count,
                "dimension": self.dimension,
                "metric_type": self.metric_type.value,
                "distance_function": self._chroma_distance,
                "metadata": metadata,
            }
        except Exception as e:
            self.logger.error(f"Failed to get collection info: {str(e)}")
            raise VectorStoreOperationError(f"Failed to get collection info: {str(e)}")

    async def _ensure_collection(self):
        """Ensure the default collection exists."""
        if not await self.collection_exists(self.collection_name):
            await self.create_collection(self.collection_name, self.dimension)

        self.collection = await self._run_async(
            self.client.get_collection, name=self.collection_name
        )

    async def _get_collection(self, collection_name: Optional[str] = None):
        """Get collection instance, creating if necessary."""
        name = collection_name or self.collection_name

        if not self.collection or self.collection.name != name:
            if not await self.collection_exists(name):
                await self.create_collection(name, self.dimension)

            self.collection = await self._run_async(self.client.get_collection, name=name)

        return self.collection

    def _build_where_clause(self, filter_expr: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build ChromaDB where clause from filter expression.

        Args:
            filter_expr: Filter dictionary

        Returns:
            ChromaDB where clause
        """
        # Convert simple filters to ChromaDB format
        # This is a basic implementation - can be extended for complex queries
        where_clause = {}

        for key, value in filter_expr.items():
            if isinstance(value, dict):
                # Handle operators like {"$gt": 10}, {"$in": [1, 2, 3]}
                for op, op_value in value.items():
                    if op == "$gt":
                        where_clause[key] = {"$gt": op_value}
                    elif op == "$gte":
                        where_clause[key] = {"$gte": op_value}
                    elif op == "$lt":
                        where_clause[key] = {"$lt": op_value}
                    elif op == "$lte":
                        where_clause[key] = {"$lte": op_value}
                    elif op == "$ne":
                        where_clause[key] = {"$ne": op_value}
                    elif op == "$in":
                        where_clause[key] = {"$in": op_value}
                    elif op == "$nin":
                        where_clause[key] = {"$nin": op_value}
            else:
                # Direct equality
                where_clause[key] = {"$eq": value}

        return where_clause

    async def _run_async(self, func, *args, **kwargs):
        """Run a potentially blocking function asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
