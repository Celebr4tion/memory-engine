"""
Integration tests for Ollama embedding provider.

These tests require a running Ollama server with the 'nomic-embed-text' model.
They will be skipped if Ollama is not available.
"""

import pytest
import numpy as np
import asyncio

from memory_core.embeddings.providers.ollama import OllamaEmbeddingProvider
from memory_core.embeddings.interfaces import TaskType, EmbeddingProviderError


@pytest.fixture
def ollama_config():
    """Ollama configuration for testing."""
    return {
        'model_name': 'nomic-embed-text',
        'base_url': 'http://localhost:11434',
        'timeout': 30,
        'keep_alive': '5m'
    }


@pytest.fixture
async def ollama_provider(ollama_config):
    """Create Ollama provider and check availability."""
    provider = OllamaEmbeddingProvider(ollama_config)
    
    # Skip tests if Ollama is not available
    if not provider.is_available():
        pytest.skip("Ollama server is not available")
    
    # Skip tests if connection test fails
    if not await provider.test_connection():
        pytest.skip("Ollama connection test failed")
    
    return provider


class TestOllamaIntegration:
    """Integration tests for Ollama embedding provider."""

    @pytest.mark.asyncio
    async def test_single_embedding_generation(self, ollama_provider):
        """Test generating a single embedding."""
        text = "The quick brown fox jumps over the lazy dog."
        
        embedding = await ollama_provider.generate_embedding(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == np.float32
        assert len(embedding.shape) == 1
        assert embedding.shape[0] > 0
        
        # Check that embedding contains meaningful values
        assert not np.allclose(embedding, 0)
        assert np.isfinite(embedding).all()

    @pytest.mark.asyncio
    async def test_batch_embedding_generation(self, ollama_provider):
        """Test generating multiple embeddings."""
        texts = [
            "Machine learning is transforming technology.",
            "Natural language processing enables text understanding.",
            "Deep learning models require large datasets.",
            "Artificial intelligence is revolutionizing industries."
        ]
        
        embeddings = await ollama_provider.generate_embeddings(texts)
        
        assert len(embeddings) == len(texts)
        
        for i, embedding in enumerate(embeddings):
            assert isinstance(embedding, np.ndarray)
            assert embedding.dtype == np.float32
            assert len(embedding.shape) == 1
            assert embedding.shape[0] > 0
            assert not np.allclose(embedding, 0)
            assert np.isfinite(embedding).all()
        
        # Check that different texts produce different embeddings
        assert not np.allclose(embeddings[0], embeddings[1])

    @pytest.mark.asyncio
    async def test_embedding_consistency(self, ollama_provider):
        """Test that same text produces consistent embeddings."""
        text = "This is a test sentence for consistency checking."
        
        embedding1 = await ollama_provider.generate_embedding(text)
        embedding2 = await ollama_provider.generate_embedding(text)
        
        # Embeddings should be very similar (allowing for small numerical differences)
        np.testing.assert_allclose(embedding1, embedding2, rtol=1e-5, atol=1e-6)

    @pytest.mark.asyncio
    async def test_embedding_similarity(self, ollama_provider):
        """Test that similar texts produce similar embeddings."""
        text1 = "The cat is sleeping on the mat."
        text2 = "A cat is resting on the carpet."
        text3 = "The weather is very hot today."
        
        emb1 = await ollama_provider.generate_embedding(text1)
        emb2 = await ollama_provider.generate_embedding(text2)
        emb3 = await ollama_provider.generate_embedding(text3)
        
        # Normalize embeddings
        emb1_norm = emb1 / np.linalg.norm(emb1)
        emb2_norm = emb2 / np.linalg.norm(emb2)
        emb3_norm = emb3 / np.linalg.norm(emb3)
        
        # Similar texts should have higher similarity
        sim_12 = np.dot(emb1_norm, emb2_norm)  # cat/sleeping texts
        sim_13 = np.dot(emb1_norm, emb3_norm)  # cat vs weather
        
        assert sim_12 > sim_13, "Similar texts should have higher cosine similarity"

    @pytest.mark.asyncio
    async def test_different_task_types(self, ollama_provider):
        """Test embedding generation with different task types."""
        text = "This is a test sentence."
        
        # Test different task types (though Ollama might not differentiate)
        task_types = [
            TaskType.SEMANTIC_SIMILARITY,
            TaskType.RETRIEVAL_DOCUMENT,
            TaskType.RETRIEVAL_QUERY
        ]
        
        embeddings = {}
        for task_type in task_types:
            embedding = await ollama_provider.generate_embedding(text, task_type)
            embeddings[task_type] = embedding
            
            assert isinstance(embedding, np.ndarray)
            assert embedding.dtype == np.float32
            assert not np.allclose(embedding, 0)

    @pytest.mark.asyncio
    async def test_empty_text_handling(self, ollama_provider):
        """Test error handling for empty text."""
        with pytest.raises(EmbeddingProviderError) as exc_info:
            await ollama_provider.generate_embedding("")
        
        assert "Text cannot be empty" in str(exc_info.value)
        assert exc_info.value.provider == "ollama"

    @pytest.mark.asyncio
    async def test_model_info_retrieval(self, ollama_provider):
        """Test retrieving model information."""
        info = await ollama_provider.get_model_info()
        
        # Should return some information about the model
        assert isinstance(info, dict)
        # Check for common fields (may vary by model)
        expected_fields = ['modelfile', 'parameters', 'template', 'details']
        available_fields = [field for field in expected_fields if field in info]
        assert len(available_fields) > 0, f"Expected some model info fields, got: {list(info.keys())}"

    @pytest.mark.asyncio
    async def test_available_models_listing(self, ollama_provider):
        """Test listing available models."""
        models = await ollama_provider.get_available_models()
        
        assert isinstance(models, list)
        assert len(models) > 0, "Should have at least one model available"
        assert ollama_provider.model_name in models, f"Configured model should be in available models: {models}"

    @pytest.mark.asyncio
    async def test_dimension_detection(self, ollama_provider):
        """Test that dimension is correctly detected."""
        # Generate an embedding to trigger dimension detection
        embedding = await ollama_provider.generate_embedding("test")
        
        detected_dim = ollama_provider.dimension
        actual_dim = len(embedding)
        
        assert detected_dim == actual_dim, f"Detected dimension {detected_dim} should match actual {actual_dim}"

    @pytest.mark.asyncio
    async def test_large_text_handling(self, ollama_provider):
        """Test handling of large text inputs."""
        # Create a reasonably large text (but not too large to cause issues)
        large_text = "This is a test sentence. " * 100  # ~2500 characters
        
        embedding = await ollama_provider.generate_embedding(large_text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == np.float32
        assert not np.allclose(embedding, 0)
        assert np.isfinite(embedding).all()

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, ollama_provider):
        """Test handling of concurrent embedding requests."""
        texts = [f"Test sentence number {i}" for i in range(5)]
        
        # Generate embeddings concurrently
        tasks = [
            ollama_provider.generate_embedding(text) 
            for text in texts
        ]
        
        embeddings = await asyncio.gather(*tasks)
        
        assert len(embeddings) == len(texts)
        for embedding in embeddings:
            assert isinstance(embedding, np.ndarray)
            assert not np.allclose(embedding, 0)

    def test_provider_info(self, ollama_provider):
        """Test provider information methods."""
        # Test string representation
        str_repr = str(ollama_provider)
        assert "OllamaEmbeddingProvider" in str_repr
        assert ollama_provider.model_name in str_repr
        
        # Test properties
        assert isinstance(ollama_provider.model_name, str)
        assert isinstance(ollama_provider.dimension, int)
        assert ollama_provider.dimension > 0
        assert isinstance(ollama_provider.max_batch_size, int)
        assert ollama_provider.max_batch_size > 0
        
        # Test supported task types
        supported_types = ollama_provider.get_supported_task_types()
        assert isinstance(supported_types, list)
        assert len(supported_types) > 0
        assert TaskType.SEMANTIC_SIMILARITY in supported_types