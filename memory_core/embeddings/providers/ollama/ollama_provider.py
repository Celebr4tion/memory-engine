"""
Ollama embedding provider implementation.

This module implements the EmbeddingProviderInterface for local Ollama server,
supporting various Ollama embedding models for fully local embedding generation.
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import requests
import aiohttp
from urllib.parse import urljoin

from memory_core.embeddings.interfaces import (
    EmbeddingProviderInterface,
    TaskType,
    EmbeddingProviderError,
)


class OllamaEmbeddingProvider(EmbeddingProviderInterface):
    """
    Ollama embedding provider for local embedding generation.

    Supports Ollama embedding models including nomic-embed-text, mxbai-embed-large,
    and other models available on the local Ollama server.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Ollama embedding provider.

        Args:
            config: Configuration dictionary with keys:
                - model_name: Ollama model name (default: 'nomic-embed-text')
                - base_url: Ollama server URL (default: 'http://localhost:11434')
                - max_batch_size: Maximum batch size (default: 32)
                - timeout: Request timeout in seconds (default: 60)
                - keep_alive: Keep-alive duration (default: '5m')
                - max_retries: Maximum connection retries (default: 3)
                - retry_delay: Delay between retries in seconds (default: 1)
        """
        super().__init__(config)

        self.logger = logging.getLogger(__name__)

        # Extract configuration
        self._model_name = config.get("model_name", "nomic-embed-text")
        self.base_url = config.get("base_url", "http://localhost:11434").rstrip("/")
        self.timeout = config.get("timeout", 60)
        self.keep_alive = config.get("keep_alive", "5m")
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 1)

        # API endpoints
        self.embeddings_endpoint = urljoin(self.base_url + "/", "api/embeddings")
        self.tags_endpoint = urljoin(self.base_url + "/", "api/tags")
        self.show_endpoint = urljoin(self.base_url + "/", "api/show")

        # Initialize dimension (will be detected on first use)
        self._dimension = None
        self._dimension_detected = False

        self.logger.info(
            f"Initialized Ollama provider with model '{self._model_name}' at {self.base_url}"
        )

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        if self._dimension is None:
            if not self._dimension_detected:
                # Try to detect dimension synchronously if not already done
                try:
                    self._detect_dimension_sync()
                except Exception as e:
                    self.logger.warning(f"Could not detect dimension: {e}")
                    # Return default dimension for common models
                    return self._get_default_dimension()
            else:
                return self._get_default_dimension()
        return self._dimension

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    def _get_default_dimension(self) -> int:
        """Get default dimension based on model name."""
        model_dimensions = {
            "nomic-embed-text": 768,
            "mxbai-embed-large": 1024,
            "all-minilm": 384,
            "snowflake-arctic-embed": 1024,
        }

        for model_prefix, dimension in model_dimensions.items():
            if model_prefix in self._model_name.lower():
                return dimension

        # Default fallback
        return 768

    def _detect_dimension_sync(self):
        """Detect embedding dimension synchronously."""
        try:
            response = requests.post(
                self.embeddings_endpoint,
                json={"model": self._model_name, "prompt": "test", "keep_alive": self.keep_alive},
                timeout=self.timeout,
            )

            if response.status_code == 200:
                result = response.json()
                if "embedding" in result:
                    self._dimension = len(result["embedding"])
                    self._dimension_detected = True
                    self.logger.info(f"Detected embedding dimension: {self._dimension}")
                else:
                    raise ValueError("No embedding in response")
            else:
                raise ValueError(f"HTTP {response.status_code}: {response.text}")

        except Exception as e:
            self.logger.warning(f"Failed to detect dimension: {e}")
            self._dimension = self._get_default_dimension()

    async def _detect_dimension_async(self):
        """Detect embedding dimension asynchronously."""
        if self._dimension_detected:
            return

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                async with session.post(
                    self.embeddings_endpoint,
                    json={
                        "model": self._model_name,
                        "prompt": "test",
                        "keep_alive": self.keep_alive,
                    },
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "embedding" in result:
                            self._dimension = len(result["embedding"])
                            self._dimension_detected = True
                            self.logger.info(f"Detected embedding dimension: {self._dimension}")
                        else:
                            raise ValueError("No embedding in response")
                    else:
                        error_text = await response.text()
                        raise ValueError(f"HTTP {response.status}: {error_text}")

        except Exception as e:
            self.logger.warning(f"Failed to detect dimension: {e}")
            self._dimension = self._get_default_dimension()

    async def generate_embedding(
        self, text: str, task_type: TaskType = TaskType.SEMANTIC_SIMILARITY
    ) -> np.ndarray:
        """
        Generate embedding for a single text using Ollama API.

        Args:
            text: Input text to embed
            task_type: Type of embedding task (currently not used by Ollama)

        Returns:
            Embedding vector as numpy array

        Raises:
            EmbeddingProviderError: If embedding generation fails
        """
        if not text or not text.strip():
            raise EmbeddingProviderError(
                "Text cannot be empty or None",
                provider="ollama",
                details={"text_length": len(text) if text else 0},
            )

        # Detect dimension if not already done
        if not self._dimension_detected:
            await self._detect_dimension_async()

        try:
            self.logger.debug(f"Generating Ollama embedding for text: {text[:50]}...")

            payload = {"model": self._model_name, "prompt": text, "keep_alive": self.keep_alive}

            # Make request with retries
            for attempt in range(self.max_retries):
                try:
                    async with aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as session:
                        async with session.post(self.embeddings_endpoint, json=payload) as response:
                            if response.status == 200:
                                result = await response.json()

                                if "embedding" not in result:
                                    raise EmbeddingProviderError(
                                        "No embedding in Ollama response",
                                        provider="ollama",
                                        details={"response": result},
                                    )

                                embedding = result["embedding"]

                                # Validate embedding
                                if not isinstance(embedding, list) or not embedding:
                                    raise EmbeddingProviderError(
                                        "Invalid embedding format from Ollama",
                                        provider="ollama",
                                        details={"embedding_type": str(type(embedding))},
                                    )

                                self.logger.debug(
                                    f"Generated Ollama embedding of length {len(embedding)}"
                                )
                                return np.array(embedding, dtype=np.float32)

                            elif response.status == 404:
                                error_text = await response.text()
                                raise EmbeddingProviderError(
                                    f"Model '{self._model_name}' not found on Ollama server",
                                    provider="ollama",
                                    details={"model": self._model_name, "response": error_text},
                                )

                            else:
                                error_text = await response.text()
                                if attempt < self.max_retries - 1:
                                    self.logger.warning(
                                        f"Ollama request failed (attempt {attempt + 1}): {response.status} {error_text}"
                                    )
                                    await asyncio.sleep(self.retry_delay)
                                    continue
                                else:
                                    raise EmbeddingProviderError(
                                        f"Ollama API error: HTTP {response.status}",
                                        provider="ollama",
                                        details={"status": response.status, "response": error_text},
                                    )

                except aiohttp.ClientError as e:
                    if attempt < self.max_retries - 1:
                        self.logger.warning(
                            f"Ollama connection failed (attempt {attempt + 1}): {str(e)}"
                        )
                        await asyncio.sleep(self.retry_delay)
                        continue
                    else:
                        raise EmbeddingProviderError(
                            f"Failed to connect to Ollama server: {str(e)}",
                            provider="ollama",
                            details={"url": self.base_url, "error": str(e)},
                        )

        except EmbeddingProviderError:
            raise
        except Exception as e:
            self.logger.error(f"Error generating Ollama embedding: {str(e)}")
            raise EmbeddingProviderError(
                f"Failed to generate Ollama embedding: {str(e)}",
                provider="ollama",
                details={
                    "model": self._model_name,
                    "task_type": task_type.value,
                    "text_preview": text[:100] if text else "",
                    "error": str(e),
                },
            )

    async def generate_embeddings(
        self, texts: List[str], task_type: TaskType = TaskType.SEMANTIC_SIMILARITY
    ) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts to embed
            task_type: Type of embedding task

        Returns:
            List of embedding vectors as numpy arrays

        Raises:
            EmbeddingProviderError: If embedding generation fails
        """
        if not texts:
            return []

        # Detect dimension if not already done
        if not self._dimension_detected:
            await self._detect_dimension_async()

        embeddings = []
        failed_indices = []

        # Process in batches (Ollama typically processes one at a time)
        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i : i + self.max_batch_size]
            batch_start_idx = i

            # Process each text individually (Ollama API limitation)
            for j, text in enumerate(batch):
                try:
                    embedding = await self.generate_embedding(text, task_type)
                    embeddings.append(embedding)
                except Exception as text_error:
                    self.logger.error(
                        f"Failed to embed text at index {batch_start_idx + j}: {str(text_error)}"
                    )
                    failed_indices.append(batch_start_idx + j)
                    # Use zero vector as fallback
                    dimension = self._dimension or self._get_default_dimension()
                    embeddings.append(np.zeros(dimension, dtype=np.float32))

        if failed_indices:
            self.logger.warning(f"Failed to generate embeddings for {len(failed_indices)} texts")

        return embeddings

    def is_available(self) -> bool:
        """Check if the Ollama provider is available."""
        try:
            response = requests.get(self.tags_endpoint, timeout=5)
            return response.status_code == 200
        except Exception as e:
            self.logger.debug(f"Ollama availability check failed: {e}")
            return False

    async def test_connection(self) -> bool:
        """Test connection to Ollama server."""
        try:
            # First, check if server is running
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(self.tags_endpoint) as response:
                    if response.status != 200:
                        self.logger.error(f"Ollama server not responding: HTTP {response.status}")
                        return False

                    # Check if model is available
                    tags_data = await response.json()
                    available_models = [
                        model.get("name", "") for model in tags_data.get("models", [])
                    ]

                    if self._model_name not in available_models:
                        self.logger.error(
                            f"Model '{self._model_name}' not found. Available models: {available_models}"
                        )
                        return False

            # Test with a simple embedding request
            test_embedding = await self.generate_embedding(
                "Test connection", TaskType.SEMANTIC_SIMILARITY
            )
            expected_dim = self._dimension or self._get_default_dimension()
            return len(test_embedding) == expected_dim

        except Exception as e:
            self.logger.error(f"Ollama connection test failed: {str(e)}")
            return False

    async def get_available_models(self) -> List[str]:
        """Get list of available models from Ollama server."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(self.tags_endpoint) as response:
                    if response.status == 200:
                        tags_data = await response.json()
                        return [model.get("name", "") for model in tags_data.get("models", [])]
                    else:
                        return []
        except Exception as e:
            self.logger.error(f"Failed to get available models: {e}")
            return []

    async def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a specific model."""
        target_model = model_name or self._model_name

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.post(
                    self.show_endpoint, json={"name": target_model}
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {}
        except Exception as e:
            self.logger.error(f"Failed to get model info for {target_model}: {e}")
            return {}

    def get_supported_task_types(self) -> List[TaskType]:
        """Get supported task types for Ollama."""
        # Ollama doesn't differentiate between task types at the API level
        return [
            TaskType.SEMANTIC_SIMILARITY,
            TaskType.RETRIEVAL_DOCUMENT,
            TaskType.RETRIEVAL_QUERY,
            TaskType.CLUSTERING,
            TaskType.CLASSIFICATION,
        ]

    def __str__(self) -> str:
        """String representation of the provider."""
        return f"OllamaEmbeddingProvider(model={self.model_name}, url={self.base_url}, dim={self.dimension})"
