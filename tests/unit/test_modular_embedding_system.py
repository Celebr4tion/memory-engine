"""
Basic tests for the modular embedding system.

This module contains tests to verify that the modular embedding system
components work correctly together.
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import os

from memory_core.embeddings.providers import EmbeddingProviderFactory
from memory_core.embeddings.vector_stores import VectorStoreFactory
from memory_core.embeddings.modular_embedding_manager import ModularEmbeddingManager
from memory_core.embeddings.interfaces import TaskType
from memory_core.config.config_manager import ConfigManager


class TestEmbeddingProviderFactory:
    """Test the embedding provider factory."""
    
    def test_get_available_providers(self):
        """Test getting available providers."""
        providers = EmbeddingProviderFactory.get_available_providers()
        expected_providers = ['gemini', 'openai', 'sentence_transformers', 'ollama']
        
        for provider in expected_providers:
            assert provider in providers
    
    def test_create_sentence_transformers_provider(self):
        """Test creating a sentence transformers provider."""
        config = {
            'model_name': 'all-MiniLM-L6-v2',
            'device': 'cpu',
            'max_batch_size': 32
        }
        
        provider = EmbeddingProviderFactory.create_provider('sentence_transformers', config)
        assert provider is not None
        assert provider.model_name == 'all-MiniLM-L6-v2'
        assert provider.max_batch_size == 32
    
    def test_unsupported_provider_raises_error(self):
        """Test that unsupported provider raises error."""
        with pytest.raises(ValueError, match="Unsupported embedding provider"):
            EmbeddingProviderFactory.create_provider('unsupported', {})


class TestVectorStoreFactory:
    """Test the vector store factory."""
    
    def test_get_available_stores(self):
        """Test getting available vector stores."""
        stores = VectorStoreFactory.get_available_stores()
        expected_stores = ['milvus', 'chroma', 'numpy']
        
        for store in expected_stores:
            assert store in stores
    
    def test_create_numpy_store(self):
        """Test creating a NumPy vector store."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                'collection_name': 'test_collection',
                'dimension': 128,
                'metric_type': 'L2',
                'persist_path': temp_dir,
                'auto_save': False
            }
            
            store = VectorStoreFactory.create_vector_store('numpy', config)
            assert store is not None
            assert store.collection_name == 'test_collection'
            assert store.dimension == 128
    
    def test_unsupported_store_raises_error(self):
        """Test that unsupported store raises error."""
        with pytest.raises(ValueError, match="Unsupported vector store"):
            VectorStoreFactory.create_vector_store('unsupported', {})
    
    def test_get_store_capabilities(self):
        """Test getting store capabilities."""
        capabilities = VectorStoreFactory.get_store_capabilities('numpy')
        assert 'persistent' in capabilities
        assert 'scalable' in capabilities
        assert capabilities['external_service'] == False


class TestModularEmbeddingManager:
    """Test the modular embedding manager."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def numpy_config(self, temp_dir):
        """Configuration for NumPy-based testing."""
        return {
            'embedding_config': {
                'provider': 'sentence_transformers',
                'provider_config': {
                    'model_name': 'all-MiniLM-L6-v2',
                    'device': 'cpu',
                    'max_batch_size': 32,
                    'normalize_embeddings': True
                }
            },
            'vector_store_config': {
                'backend': 'numpy',
                'backend_config': {
                    'collection_name': 'test_collection',
                    'dimension': 384,  # all-MiniLM-L6-v2 dimension
                    'metric_type': 'L2',
                    'persist_path': temp_dir,
                    'auto_save': False
                }
            }
        }
    
    @pytest.mark.asyncio
    async def test_manager_initialization(self, numpy_config):
        """Test that the manager initializes correctly."""
        manager = ModularEmbeddingManager(
            numpy_config['embedding_config'],
            numpy_config['vector_store_config']
        )
        
        assert not manager.is_initialized
        
        # Initialize should work
        success = await manager.initialize()
        assert success
        assert manager.is_initialized
        
        # Clean up
        await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_embedding_generation(self, numpy_config):
        """Test embedding generation."""
        manager = ModularEmbeddingManager(
            numpy_config['embedding_config'],
            numpy_config['vector_store_config']
        )
        
        await manager.initialize()
        
        try:
            # Generate single embedding
            text = "This is a test sentence for embedding."
            embedding = await manager.generate_embedding(text)
            
            assert isinstance(embedding, np.ndarray)
            assert len(embedding) == 384  # all-MiniLM-L6-v2 dimension
            
            # Generate multiple embeddings
            texts = ["First sentence.", "Second sentence.", "Third sentence."]
            embeddings = await manager.generate_embeddings(texts)
            
            assert len(embeddings) == 3
            for emb in embeddings:
                assert isinstance(emb, np.ndarray)
                assert len(emb) == 384
        
        finally:
            await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_store_and_search_embeddings(self, numpy_config):
        """Test storing and searching embeddings."""
        manager = ModularEmbeddingManager(
            numpy_config['embedding_config'],
            numpy_config['vector_store_config']
        )
        
        await manager.initialize()
        
        try:
            # Store some embeddings
            texts = [
                "The cat sat on the mat.",
                "Dogs are loyal animals.",
                "The weather is nice today.",
                "Programming is fun and challenging."
            ]
            
            node_ids = []
            for i, text in enumerate(texts):
                node_id = f"node_{i}"
                stored_id = await manager.store_embedding(
                    node_id, 
                    text, 
                    metadata={'text': text, 'index': i}
                )
                assert stored_id == node_id
                node_ids.append(node_id)
            
            # Search for similar embeddings
            query = "Cats and dogs are pets."
            results = await manager.search_similar(query, top_k=2)
            
            assert len(results) <= 2
            assert len(results) > 0
            
            # Results should have id and score
            for result in results:
                assert 'id' in result
                assert 'score' in result
                assert result['id'] in node_ids
        
        finally:
            await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_embedding_operations(self, numpy_config):
        """Test embedding CRUD operations."""
        manager = ModularEmbeddingManager(
            numpy_config['embedding_config'],
            numpy_config['vector_store_config']
        )
        
        await manager.initialize()
        
        try:
            node_id = "test_node"
            text = "This is a test document."
            metadata = {'type': 'test', 'version': 1}
            
            # Store embedding
            await manager.store_embedding(node_id, text, metadata)
            
            # Get embedding
            stored = await manager.get_embedding(node_id)
            assert stored is not None
            assert stored['id'] == node_id
            assert 'vector' in stored
            
            # Update embedding
            new_text = "This is an updated test document."
            new_metadata = {'type': 'test', 'version': 2}
            success = await manager.update_embedding(node_id, new_text, new_metadata)
            assert success
            
            # Delete embedding
            success = await manager.delete_embedding(node_id)
            assert success
            
            # Verify deletion
            deleted = await manager.get_embedding(node_id)
            assert deleted is None
        
        finally:
            await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_health_check(self, numpy_config):
        """Test health check functionality."""
        manager = ModularEmbeddingManager(
            numpy_config['embedding_config'],
            numpy_config['vector_store_config']
        )
        
        # Health check before initialization
        health = await manager.health_check()
        assert health['manager_initialized'] == False
        
        # Initialize and check again
        await manager.initialize()
        
        try:
            health = await manager.health_check()
            assert health['manager_initialized'] == True
            assert 'embedding_provider' in health
            assert 'vector_store' in health
            
            # Get system info
            info = await manager.get_system_info()
            assert info['initialized'] == True
            assert info['embedding_provider_type'] == 'sentence_transformers'
            assert info['vector_store_type'] == 'numpy'
            assert 'available_providers' in info
            assert 'available_stores' in info
        
        finally:
            await manager.cleanup()
    
    def test_uninitalized_operations_raise_error(self, numpy_config):
        """Test that operations on uninitialized manager raise errors."""
        manager = ModularEmbeddingManager(
            numpy_config['embedding_config'],
            numpy_config['vector_store_config']
        )
        
        # These should all raise errors
        with pytest.raises(Exception):
            asyncio.run(manager.generate_embedding("test"))
        
        with pytest.raises(Exception):
            asyncio.run(manager.store_embedding("id", "text"))
        
        with pytest.raises(Exception):
            asyncio.run(manager.search_similar("query"))


class TestConfigManagerIntegration:
    """Test integration with the config manager."""
    
    @patch.dict(os.environ, {'GOOGLE_API_KEY': 'test_key'})
    def test_config_manager_embedding_config(self):
        """Test that config manager provides correct embedding config."""
        # Mock the config loading to avoid file dependencies
        with patch('memory_core.config.config_manager.ConfigManager._load_configuration'):
            config_manager = ConfigManager()
            
            # Set up some test configuration
            config_manager.config.embeddings.provider = config_manager.config.embeddings.provider.__class__('gemini')
            config_manager.config.embeddings.gemini.api_key = None
            config_manager.config.api.google_api_key = 'test_key'
            
            embedding_config = config_manager.get_embedding_config()
            
            assert embedding_config['provider'] == 'gemini'
            assert embedding_config['provider_config']['api_key'] == 'test_key'
    
    @patch.dict(os.environ, {})
    def test_config_manager_vector_store_config(self):
        """Test that config manager provides correct vector store config."""
        with patch('memory_core.config.config_manager.ConfigManager._load_configuration'):
            config_manager = ConfigManager()
            
            # Set up test configuration
            config_manager.config.vector_stores.backend = config_manager.config.vector_stores.backend.__class__('numpy')
            
            vector_store_config = config_manager.get_vector_store_config()
            
            assert vector_store_config['backend'] == 'numpy'
            assert 'backend_config' in vector_store_config