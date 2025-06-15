"""
Unit tests for Ollama embedding provider.

Tests the OllamaEmbeddingProvider implementation including configuration,
error handling, and API interaction patterns.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
import aiohttp
from aioresponses import aioresponses

from memory_core.embeddings.providers.ollama import OllamaEmbeddingProvider
from memory_core.embeddings.interfaces import TaskType, EmbeddingProviderError


class TestOllamaEmbeddingProvider:
    """Test cases for OllamaEmbeddingProvider."""

    def test_init_default_config(self):
        """Test initialization with default configuration."""
        config = {}
        provider = OllamaEmbeddingProvider(config)

        assert provider.model_name == "nomic-embed-text"
        assert provider.base_url == "http://localhost:11434"
        assert provider.timeout == 60
        assert provider.keep_alive == "5m"
        assert provider.max_batch_size == 32

    def test_init_custom_config(self):
        """Test initialization with custom configuration."""
        config = {
            "model_name": "mxbai-embed-large",
            "base_url": "http://custom-server:8080",
            "timeout": 120,
            "keep_alive": "10m",
            "max_batch_size": 16,
        }
        provider = OllamaEmbeddingProvider(config)

        assert provider.model_name == "mxbai-embed-large"
        assert provider.base_url == "http://custom-server:8080"
        assert provider.timeout == 120
        assert provider.keep_alive == "10m"
        assert provider.max_batch_size == 16

    def test_default_dimensions(self):
        """Test default dimension detection for known models."""
        test_cases = [
            ("nomic-embed-text", 768),
            ("mxbai-embed-large", 1024),
            ("all-minilm", 384),
            ("snowflake-arctic-embed", 1024),
            ("unknown-model", 768),  # fallback
        ]

        for model_name, expected_dim in test_cases:
            config = {"model_name": model_name}
            provider = OllamaEmbeddingProvider(config)
            assert provider._get_default_dimension() == expected_dim

    @patch("requests.get")
    def test_is_available_success(self, mock_get):
        """Test is_available when server is responsive."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        config = {}
        provider = OllamaEmbeddingProvider(config)

        assert provider.is_available() is True
        mock_get.assert_called_once_with("http://localhost:11434/api/tags", timeout=5)

    @patch("requests.get")
    def test_is_available_failure(self, mock_get):
        """Test is_available when server is not responsive."""
        mock_get.side_effect = Exception("Connection failed")

        config = {}
        provider = OllamaEmbeddingProvider(config)

        assert provider.is_available() is False

    @pytest.mark.asyncio
    async def test_generate_embedding_success(self):
        """Test successful embedding generation."""
        config = {"model_name": "test-model"}
        provider = OllamaEmbeddingProvider(config)

        # Mock the dimension detection
        provider._dimension = 768
        provider._dimension_detected = True

        mock_embedding = [0.1] * 768

        with aioresponses() as m:
            m.post("http://localhost:11434/api/embeddings", payload={"embedding": mock_embedding})

            result = await provider.generate_embedding("test text")

            assert isinstance(result, np.ndarray)
            assert result.shape == (768,)
            assert result.dtype == np.float32
            np.testing.assert_array_almost_equal(result, mock_embedding)

    @pytest.mark.asyncio
    async def test_generate_embedding_empty_text(self):
        """Test embedding generation with empty text."""
        config = {}
        provider = OllamaEmbeddingProvider(config)

        with pytest.raises(EmbeddingProviderError) as exc_info:
            await provider.generate_embedding("")

        assert "Text cannot be empty" in str(exc_info.value)
        assert exc_info.value.provider == "ollama"

    @pytest.mark.asyncio
    async def test_generate_embedding_model_not_found(self):
        """Test embedding generation when model is not found."""
        config = {"model_name": "nonexistent-model"}
        provider = OllamaEmbeddingProvider(config)

        with aioresponses() as m:
            # Mock both dimension detection and actual embedding requests
            m.post(
                "http://localhost:11434/api/embeddings",
                status=404,
                payload={"error": "model not found"},
                repeat=True,
            )

            with pytest.raises(EmbeddingProviderError) as exc_info:
                await provider.generate_embedding("test text")

            assert "not found on Ollama server" in str(exc_info.value)
            assert exc_info.value.provider == "ollama"

    @pytest.mark.asyncio
    async def test_generate_embedding_invalid_response(self):
        """Test embedding generation with invalid response."""
        config = {}
        provider = OllamaEmbeddingProvider(config)
        provider._dimension_detected = True

        with aioresponses() as m:
            m.post("http://localhost:11434/api/embeddings", payload={"no_embedding": "invalid"})

            with pytest.raises(EmbeddingProviderError) as exc_info:
                await provider.generate_embedding("test text")

            assert "No embedding in Ollama response" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch(self):
        """Test batch embedding generation."""
        config = {}
        provider = OllamaEmbeddingProvider(config)
        provider._dimension = 768
        provider._dimension_detected = True

        texts = ["text1", "text2", "text3"]
        mock_embedding = [0.1] * 768

        with aioresponses() as m:
            # Mock all embedding requests (with repeat=True for safety)
            m.post(
                "http://localhost:11434/api/embeddings",
                payload={"embedding": mock_embedding},
                repeat=True,
            )

            results = await provider.generate_embeddings(texts)

            assert len(results) == 3
            for result in results:
                assert isinstance(result, np.ndarray)
                assert result.shape == (768,)
                assert result.dtype == np.float32

    @pytest.mark.asyncio
    async def test_generate_embeddings_empty_list(self):
        """Test batch embedding generation with empty list."""
        config = {}
        provider = OllamaEmbeddingProvider(config)

        results = await provider.generate_embeddings([])
        assert results == []

    @pytest.mark.asyncio
    async def test_test_connection_success(self):
        """Test successful connection test."""
        config = {"model_name": "test-model"}
        provider = OllamaEmbeddingProvider(config)

        mock_embedding = [0.1] * 768
        mock_tags = {"models": [{"name": "test-model"}]}

        with aioresponses() as m:
            m.get("http://localhost:11434/api/tags", payload=mock_tags)
            m.post(
                "http://localhost:11434/api/embeddings",
                payload={"embedding": mock_embedding},
                repeat=True,
            )

            result = await provider.test_connection()
            assert result is True

    @pytest.mark.asyncio
    async def test_test_connection_model_not_available(self):
        """Test connection test when model is not available."""
        config = {"model_name": "missing-model"}
        provider = OllamaEmbeddingProvider(config)

        mock_tags = {"models": [{"name": "other-model"}]}

        with aioresponses() as m:
            m.get("http://localhost:11434/api/tags", payload=mock_tags)

            result = await provider.test_connection()
            assert result is False

    @pytest.mark.asyncio
    async def test_get_available_models(self):
        """Test getting available models."""
        config = {}
        provider = OllamaEmbeddingProvider(config)

        mock_tags = {"models": [{"name": "model1"}, {"name": "model2"}, {"name": "model3"}]}

        with aioresponses() as m:
            m.get("http://localhost:11434/api/tags", payload=mock_tags)

            models = await provider.get_available_models()
            assert models == ["model1", "model2", "model3"]

    @pytest.mark.asyncio
    async def test_get_model_info(self):
        """Test getting model information."""
        config = {"model_name": "test-model"}
        provider = OllamaEmbeddingProvider(config)

        mock_info = {
            "license": "test",
            "modelfile": "test",
            "parameters": "test",
            "template": "test",
            "details": {"format": "gguf", "family": "test-family", "parameter_size": "7B"},
        }

        with aioresponses() as m:
            m.post("http://localhost:11434/api/show", payload=mock_info)

            info = await provider.get_model_info()
            assert info == mock_info

    def test_supported_task_types(self):
        """Test getting supported task types."""
        config = {}
        provider = OllamaEmbeddingProvider(config)

        supported = provider.get_supported_task_types()
        expected = [
            TaskType.SEMANTIC_SIMILARITY,
            TaskType.RETRIEVAL_DOCUMENT,
            TaskType.RETRIEVAL_QUERY,
            TaskType.CLUSTERING,
            TaskType.CLASSIFICATION,
        ]

        assert supported == expected

    def test_str_representation(self):
        """Test string representation of provider."""
        config = {"model_name": "test-model", "base_url": "http://test:1234"}
        provider = OllamaEmbeddingProvider(config)

        str_repr = str(provider)
        assert "OllamaEmbeddingProvider" in str_repr
        assert "test-model" in str_repr
        assert "http://test:1234" in str_repr

    @pytest.mark.asyncio
    async def test_dimension_detection(self):
        """Test automatic dimension detection."""
        config = {"model_name": "unknown-model"}
        provider = OllamaEmbeddingProvider(config)

        mock_embedding = [0.1] * 512  # Custom dimension

        with aioresponses() as m:
            m.post("http://localhost:11434/api/embeddings", payload={"embedding": mock_embedding})

            # Trigger dimension detection
            await provider._detect_dimension_async()

            assert provider._dimension == 512
            assert provider._dimension_detected is True
