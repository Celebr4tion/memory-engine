"""
Modular embedding manager with pluggable providers and vector stores.

This module provides a unified interface for embedding operations while supporting
multiple embedding providers and vector store backends through the factory system.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from memory_core.embeddings.interfaces import (
    EmbeddingProviderInterface,
    TaskType,
    EmbeddingError,
    EmbeddingProviderError
)
from memory_core.embeddings.vector_stores.interfaces import (
    VectorStoreInterface,
    VectorStoreError
)
from memory_core.embeddings.providers import EmbeddingProviderFactory
from memory_core.embeddings.vector_stores import VectorStoreFactory


class ModularEmbeddingManager:
    """
    Modular embedding manager supporting multiple providers and vector stores.
    
    This manager provides a unified interface for embedding operations while
    allowing dynamic configuration of embedding providers and vector store backends.
    """
    
    def __init__(
        self,
        embedding_config: Dict[str, Any],
        vector_store_config: Dict[str, Any]
    ):
        """
        Initialize the modular embedding manager.
        
        Args:
            embedding_config: Configuration for embedding provider with keys:
                - provider: Provider type ('gemini', 'openai', 'sentence_transformers', 'ollama')
                - provider_config: Provider-specific configuration
            vector_store_config: Configuration for vector store with keys:
                - backend: Backend type ('milvus', 'chroma', 'numpy', 'qdrant', 'faiss')
                - backend_config: Backend-specific configuration
        """
        self.logger = logging.getLogger(__name__)
        
        # Extract configurations
        self.embedding_provider_type = embedding_config.get('provider', 'gemini')
        self.embedding_provider_config = embedding_config.get('provider_config', {})
        
        self.vector_store_type = vector_store_config.get('backend', 'numpy')
        self.vector_store_config = vector_store_config.get('backend_config', {})
        
        # Initialize components
        self.embedding_provider: Optional[EmbeddingProviderInterface] = None
        self.vector_store: Optional[VectorStoreInterface] = None
        
        # State
        self._initialized = False
        
    async def initialize(self) -> bool:
        """
        Initialize the embedding provider and vector store.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Initialize embedding provider
            self.logger.info(f"Initializing {self.embedding_provider_type} embedding provider")
            self.embedding_provider = EmbeddingProviderFactory.create_provider(
                self.embedding_provider_type,
                self.embedding_provider_config
            )
            
            # Test embedding provider connection
            if not await self.embedding_provider.test_connection():
                self.logger.warning(f"Embedding provider {self.embedding_provider_type} connection test failed")
                return False
            
            # Initialize vector store
            self.logger.info(f"Initializing {self.vector_store_type} vector store")
            self.vector_store = VectorStoreFactory.create_vector_store(
                self.vector_store_type,
                self.vector_store_config
            )
            
            # Connect to vector store
            if not await self.vector_store.connect():
                self.logger.warning(f"Vector store {self.vector_store_type} connection failed")
                return False
            
            self._initialized = True
            self.logger.info("Modular embedding manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding manager: {str(e)}")
            return False
    
    async def generate_embedding(
        self,
        text: str,
        task_type: TaskType = TaskType.SEMANTIC_SIMILARITY
    ) -> np.ndarray:
        """
        Generate an embedding for the given text.
        
        Args:
            text: Text to generate embedding for
            task_type: Type of embedding task
            
        Returns:
            Embedding vector as numpy array
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        self._check_initialized()
        
        try:
            return await self.embedding_provider.generate_embedding(text, task_type)
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {str(e)}")
            raise EmbeddingError(f"Embedding generation failed: {str(e)}") from e
    
    async def generate_embeddings(
        self,
        texts: List[str],
        task_type: TaskType = TaskType.SEMANTIC_SIMILARITY
    ) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to generate embeddings for
            task_type: Type of embedding task
            
        Returns:
            List of embedding vectors as numpy arrays
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        self._check_initialized()
        
        try:
            return await self.embedding_provider.generate_embeddings(texts, task_type)
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {str(e)}")
            raise EmbeddingError(f"Embeddings generation failed: {str(e)}") from e
    
    async def store_embedding(
        self,
        node_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        task_type: TaskType = TaskType.RETRIEVAL_DOCUMENT
    ) -> str:
        """
        Generate and store an embedding for a node.
        
        Args:
            node_id: Unique identifier for the node
            text: Text content to generate embedding for
            metadata: Optional metadata to store with the embedding
            task_type: Type of embedding task
            
        Returns:
            Vector ID that was stored
            
        Raises:
            EmbeddingError: If operation fails
        """
        self._check_initialized()
        
        try:
            # Generate embedding
            embedding = await self.generate_embedding(text, task_type)
            
            # Store in vector store
            vector_ids = await self.vector_store.add_vectors(
                vectors=[embedding],
                ids=[node_id],
                metadata=[metadata] if metadata else None
            )
            
            self.logger.info(f"Stored embedding for node {node_id}")
            return vector_ids[0]
            
        except Exception as e:
            self.logger.error(f"Failed to store embedding for node {node_id}: {str(e)}")
            raise EmbeddingError(f"Failed to store embedding: {str(e)}") from e
    
    async def search_similar(
        self,
        query_text: str,
        top_k: int = 5,
        task_type: TaskType = TaskType.RETRIEVAL_QUERY,
        filter_expr: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for nodes with embeddings similar to the query text.
        
        Args:
            query_text: Text to search for similar nodes
            top_k: Number of similar nodes to return
            task_type: Type of embedding task for the query
            filter_expr: Optional filter expression for the search
            
        Returns:
            List of search results with id, score, and metadata
            
        Raises:
            EmbeddingError: If search fails
        """
        self._check_initialized()
        
        try:
            # Generate query embedding
            query_embedding = await self.generate_embedding(query_text, task_type)
            
            # Search in vector store
            results = await self.vector_store.search_vectors(
                query_vector=query_embedding,
                top_k=top_k,
                filter_expr=filter_expr
            )
            
            self.logger.info(f"Found {len(results)} similar nodes for query")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to search for similar nodes: {str(e)}")
            raise EmbeddingError(f"Similarity search failed: {str(e)}") from e
    
    async def get_embedding(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get stored embedding for a node.
        
        Args:
            node_id: ID of the node
            
        Returns:
            Embedding data or None if not found
        """
        self._check_initialized()
        
        try:
            return await self.vector_store.get_vector(node_id)
        except Exception as e:
            self.logger.error(f"Failed to get embedding for node {node_id}: {str(e)}")
            return None
    
    async def delete_embedding(self, node_id: str) -> bool:
        """
        Delete stored embedding for a node.
        
        Args:
            node_id: ID of the node
            
        Returns:
            True if deletion successful, False otherwise
        """
        self._check_initialized()
        
        try:
            deleted_count = await self.vector_store.delete_vectors([node_id])
            return deleted_count > 0
        except Exception as e:
            self.logger.error(f"Failed to delete embedding for node {node_id}: {str(e)}")
            return False
    
    async def update_embedding(
        self,
        node_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        task_type: TaskType = TaskType.RETRIEVAL_DOCUMENT
    ) -> bool:
        """
        Update embedding for a node.
        
        Args:
            node_id: ID of the node
            text: New text content
            metadata: Optional new metadata
            task_type: Type of embedding task
            
        Returns:
            True if update successful, False otherwise
        """
        self._check_initialized()
        
        try:
            # Generate new embedding
            embedding = await self.generate_embedding(text, task_type)
            
            # Update in vector store
            return await self.vector_store.update_vector(
                vector_id=node_id,
                vector=embedding,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update embedding for node {node_id}: {str(e)}")
            return False
    
    async def count_embeddings(self) -> int:
        """
        Count total number of stored embeddings.
        
        Returns:
            Number of stored embeddings
        """
        self._check_initialized()
        
        try:
            return await self.vector_store.count_vectors()
        except Exception as e:
            self.logger.error(f"Failed to count embeddings: {str(e)}")
            return 0
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all components.
        
        Returns:
            Health status information
        """
        health_status = {
            'manager_initialized': self._initialized,
            'embedding_provider': None,
            'vector_store': None
        }
        
        if self._initialized:
            # Check embedding provider
            try:
                provider_healthy = await self.embedding_provider.test_connection()
                health_status['embedding_provider'] = {
                    'type': self.embedding_provider_type,
                    'healthy': provider_healthy,
                    'available': self.embedding_provider.is_available()
                }
            except Exception as e:
                health_status['embedding_provider'] = {
                    'type': self.embedding_provider_type,
                    'healthy': False,
                    'error': str(e)
                }
            
            # Check vector store
            try:
                store_health = await self.vector_store.health_check()
                health_status['vector_store'] = {
                    'type': self.vector_store_type,
                    **store_health
                }
            except Exception as e:
                health_status['vector_store'] = {
                    'type': self.vector_store_type,
                    'healthy': False,
                    'error': str(e)
                }
        
        return health_status
    
    async def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information about the embedding manager.
        
        Returns:
            System information dictionary
        """
        info = {
            'initialized': self._initialized,
            'embedding_provider_type': self.embedding_provider_type,
            'vector_store_type': self.vector_store_type,
            'available_providers': EmbeddingProviderFactory.get_available_providers(),
            'available_stores': VectorStoreFactory.get_available_stores()
        }
        
        if self._initialized:
            # Add provider info
            if self.embedding_provider:
                info['embedding_provider_info'] = {
                    'model_name': self.embedding_provider.model_name,
                    'dimension': self.embedding_provider.dimension,
                    'max_batch_size': self.embedding_provider.max_batch_size,
                    'supported_task_types': [t.value for t in self.embedding_provider.get_supported_task_types()]
                }
            
            # Add vector store info
            if self.vector_store:
                try:
                    store_info = await self.vector_store.get_collection_info()
                    info['vector_store_info'] = store_info
                except:
                    info['vector_store_info'] = {'error': 'Unable to get store info'}
        
        return info
    
    async def cleanup(self) -> None:
        """
        Clean up resources and disconnect from services.
        """
        if self.vector_store:
            try:
                await self.vector_store.disconnect()
                self.logger.info("Disconnected from vector store")
            except Exception as e:
                self.logger.warning(f"Error disconnecting from vector store: {str(e)}")
        
        self._initialized = False
    
    def _check_initialized(self) -> None:
        """Check if manager is initialized and raise error if not."""
        if not self._initialized:
            raise EmbeddingError("Embedding manager not initialized. Call initialize() first.")
    
    @property
    def is_initialized(self) -> bool:
        """Check if the manager is initialized."""
        return self._initialized
    
    @property
    def provider_type(self) -> str:
        """Get the embedding provider type."""
        return self.embedding_provider_type
    
    @property
    def store_type(self) -> str:
        """Get the vector store type."""
        return self.vector_store_type