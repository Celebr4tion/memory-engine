"""
Milvus vector store implementation.

This module provides a Milvus implementation of the VectorStoreInterface,
allowing for efficient storage and retrieval of embeddings using Milvus.
"""

import logging
import time
from typing import List, Dict, Any, Optional
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


class MilvusVectorStore(VectorStoreInterface):
    """
    Milvus implementation of the vector store interface.

    Provides high-performance vector storage and similarity search using Milvus.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Milvus vector store.

        Args:
            config: Configuration dictionary with keys:
                - host: Milvus server host (default: 'localhost')
                - port: Milvus server port (default: 19530)
                - collection_name: Collection name (default: 'default_collection')
                - dimension: Vector dimension (default: 768)
                - metric_type: Similarity metric (default: 'L2')
                - index_type: Index type (default: 'IVF_FLAT')
                - nlist: Index parameter (default: 1024)
                - nprobe: Search parameter (default: 10)
                - user: Optional username for authentication
                - password: Optional password for authentication
        """
        if not MILVUS_AVAILABLE:
            raise VectorStoreError(
                "Milvus is not available. Install pymilvus: pip install pymilvus"
            )

        super().__init__(config)

        self.logger = logging.getLogger(__name__)

        # Milvus-specific configuration
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 19530)
        self.user = config.get("user")
        self.password = config.get("password")

        # Index configuration
        self.index_type = IndexType(config.get("index_type", "IVF_FLAT"))
        self.nlist = config.get("nlist", 1024)
        self.nprobe = config.get("nprobe", 10)

        # Connection and collection state
        self.collection: Optional[Collection] = None
        self._connection_alias = f"milvus_{id(self)}"

        # Index parameters
        self._index_params = self._build_index_params()

    @property
    def is_connected(self) -> bool:
        """Check if connected to Milvus."""
        return self._is_connected

    def _build_index_params(self) -> Dict[str, Any]:
        """Build index parameters for Milvus."""
        metric_mapping = {MetricType.L2: "L2", MetricType.IP: "IP", MetricType.COSINE: "COSINE"}

        index_mapping = {
            IndexType.FLAT: "FLAT",
            IndexType.IVF_FLAT: "IVF_FLAT",
            IndexType.IVF_SQ8: "IVF_SQ8",
            IndexType.IVF_PQ: "IVF_PQ",
            IndexType.HNSW: "HNSW",
        }

        return {
            "metric_type": metric_mapping.get(self.metric_type, "L2"),
            "index_type": index_mapping.get(self.index_type, "IVF_FLAT"),
            "params": {"nlist": self.nlist},
        }

    async def connect(self) -> bool:
        """
        Connect to Milvus server.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.logger.info(f"Connecting to Milvus at {self.host}:{self.port}")

            # Build connection arguments
            connect_args = {"host": self.host, "port": self.port, "timeout": 10}

            # Add authentication if configured
            if self.user:
                connect_args["user"] = self.user
            if self.password:
                connect_args["password"] = self.password

            # Connect to Milvus
            connections.connect(self._connection_alias, **connect_args)

            # Ensure collection exists
            await self._ensure_collection()

            self._is_connected = True
            self.logger.info(f"Successfully connected to Milvus at {self.host}:{self.port}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to connect to Milvus: {str(e)}")
            self._is_connected = False
            raise VectorStoreConnectionError(f"Failed to connect to Milvus: {str(e)}")

    async def disconnect(self) -> bool:
        """
        Disconnect from Milvus server.

        Returns:
            True if disconnection successful, False otherwise
        """
        try:
            # Release collection if loaded
            if self.collection:
                try:
                    self.collection.release()
                except Exception as e:
                    self.logger.warning(f"Error releasing collection: {str(e)}")

            # Disconnect from server
            connections.disconnect(self._connection_alias)

            self._is_connected = False
            self.collection = None
            self.logger.info("Disconnected from Milvus")
            return True

        except Exception as e:
            self.logger.error(f"Error disconnecting from Milvus: {str(e)}")
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
        Create a new collection in Milvus.

        Args:
            collection_name: Name of the collection
            dimension: Vector dimension
            index_type: Type of index to create
            metric_type: Similarity metric to use
            **kwargs: Additional parameters

        Returns:
            True if creation successful, False otherwise
        """
        try:
            if utility.has_collection(collection_name, using=self._connection_alias):
                self.logger.info(f"Collection {collection_name} already exists")
                return True

            # Define fields
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=255),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
                FieldSchema(name="timestamp", dtype=DataType.DOUBLE),
            ]

            # Add metadata field if specified
            if kwargs.get("enable_metadata", True):
                fields.append(FieldSchema(name="metadata", dtype=DataType.JSON))

            # Create schema and collection
            schema = CollectionSchema(fields=fields, description="Vector embeddings collection")
            collection = Collection(
                name=collection_name, schema=schema, using=self._connection_alias
            )

            # Create index
            index_params = self._build_index_params()
            if index_type != IndexType.AUTO:
                index_params["index_type"] = index_type.value
            if metric_type != MetricType.L2:
                index_params["metric_type"] = metric_type.value

            collection.create_index(field_name="vector", index_params=index_params)
            collection.flush()

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
            return utility.has_collection(collection_name, using=self._connection_alias)
        except Exception as e:
            self.logger.error(f"Error checking collection existence: {str(e)}")
            return False

    async def drop_collection(self, collection_name: str) -> bool:
        """
        Drop a collection from Milvus.

        Args:
            collection_name: Name of the collection to drop

        Returns:
            True if drop successful, False otherwise
        """
        try:
            if await self.collection_exists(collection_name):
                utility.drop_collection(collection_name, using=self._connection_alias)
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
        Add vectors to Milvus.

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
            ids = [f"vec_{int(time.time() * 1000000)}_{i}" for i in range(len(vectors))]

        if len(ids) != len(vectors):
            raise VectorStoreOperationError("Number of IDs must match number of vectors")

        try:
            # Prepare data for insertion
            data = [
                ids,  # id field
                [
                    vector.tolist() if isinstance(vector, np.ndarray) else vector
                    for vector in vectors
                ],  # vector field
                [time.time()] * len(vectors),  # timestamp field
            ]

            # Add metadata if provided and collection supports it
            if metadata and await self._collection_has_metadata(collection):
                if len(metadata) != len(vectors):
                    raise VectorStoreOperationError(
                        "Number of metadata entries must match number of vectors"
                    )
                data.append(metadata)

            # Insert data
            collection.insert(data)
            collection.flush()

            self.logger.info(f"Added {len(vectors)} vectors to collection {collection.name}")
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
        Search for similar vectors in Milvus.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            collection_name: Collection to search (uses default if None)
            filter_expr: Optional filter expression

        Returns:
            List of search results with id, score, and metadata
        """
        collection = await self._get_collection(collection_name)

        # Validate dimension
        if len(query_vector) != self.dimension:
            raise VectorStoreDimensionError(self.dimension, len(query_vector))

        try:
            # Prepare search parameters
            search_params = {
                "metric_type": self._index_params["metric_type"],
                "params": {"nprobe": self.nprobe},
            }

            # Determine output fields
            output_fields = ["id"]
            if await self._collection_has_metadata(collection):
                output_fields.append("metadata")

            # Perform search
            results = collection.search(
                data=[
                    query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector
                ],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=output_fields,
                expr=self._build_filter_expr(filter_expr) if filter_expr else None,
            )

            # Parse results
            matches = []
            if results and results[0]:
                for hit in results[0]:
                    result = {"id": hit.entity.get("id"), "score": float(hit.score)}

                    # Add metadata if available
                    if "metadata" in output_fields and hit.entity.get("metadata"):
                        result["metadata"] = hit.entity.get("metadata")

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
            # Query for the specific vector
            results = collection.query(
                expr=f'id == "{vector_id}"',
                output_fields=(
                    ["id", "vector", "metadata"]
                    if await self._collection_has_metadata(collection)
                    else ["id", "vector"]
                ),
            )

            if not results:
                return None

            result = results[0]
            vector_data = {"id": result["id"], "vector": np.array(result["vector"])}

            if "metadata" in result:
                vector_data["metadata"] = result["metadata"]

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
            # Build delete expression
            ids_str = '", "'.join(vector_ids)
            expr = f'id in ["{ids_str}"]'

            # Delete vectors
            collection.delete(expr)
            collection.flush()

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

        Note: Milvus doesn't support direct updates, so this deletes and re-inserts.

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

            # Update fields
            new_vector = vector if vector is not None else current_data["vector"]
            new_metadata = metadata if metadata is not None else current_data.get("metadata")

            # Delete old vector
            await self.delete_vectors([vector_id], collection_name)

            # Insert new vector
            await self.add_vectors(
                vectors=[new_vector],
                ids=[vector_id],
                metadata=[new_metadata] if new_metadata else None,
                collection_name=collection_name,
            )

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
            collection.flush()  # Ensure all data is persisted
            return collection.num_entities
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
            collection.flush()

            return {
                "name": collection.name,
                "num_entities": collection.num_entities,
                "dimension": self.dimension,
                "metric_type": self.metric_type.value,
                "index_type": self.index_type.value,
                "schema": {
                    "fields": [
                        {"name": field.name, "type": str(field.dtype)}
                        for field in collection.schema.fields
                    ]
                },
            }
        except Exception as e:
            self.logger.error(f"Failed to get collection info: {str(e)}")
            raise VectorStoreOperationError(f"Failed to get collection info: {str(e)}")

    async def _ensure_collection(self):
        """Ensure the default collection exists."""
        if not await self.collection_exists(self.collection_name):
            await self.create_collection(self.collection_name, self.dimension)

        self.collection = Collection(self.collection_name, using=self._connection_alias)
        self.collection.load()

    async def _get_collection(self, collection_name: Optional[str] = None) -> Collection:
        """Get collection instance, creating if necessary."""
        name = collection_name or self.collection_name

        if not self.collection or self.collection.name != name:
            if not await self.collection_exists(name):
                await self.create_collection(name, self.dimension)

            self.collection = Collection(name, using=self._connection_alias)
            self.collection.load()

        return self.collection

    async def _collection_has_metadata(self, collection: Collection) -> bool:
        """Check if collection has metadata field."""
        try:
            field_names = [field.name for field in collection.schema.fields]
            return "metadata" in field_names
        except:
            return False

    def _build_filter_expr(self, filter_dict: Dict[str, Any]) -> str:
        """Build Milvus filter expression from dictionary."""
        # Simple implementation - can be extended
        conditions = []
        for key, value in filter_dict.items():
            if isinstance(value, str):
                conditions.append(f'{key} == "{value}"')
            else:
                conditions.append(f"{key} == {value}")

        return " and ".join(conditions)
