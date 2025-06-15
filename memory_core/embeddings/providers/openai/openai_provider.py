"""
OpenAI embedding provider implementation.

This module implements the EmbeddingProviderInterface for OpenAI's embedding API,
supporting text-embedding-3-small and text-embedding-3-large models with
configurable dimensions and batch processing.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
import asyncio

try:
    import openai
    from openai import AsyncOpenAI
except ImportError:
    raise ImportError(
        "OpenAI library is required for OpenAI embedding provider. "
        "Install it with: pip install openai"
    )

from memory_core.embeddings.interfaces import (
    EmbeddingProviderInterface,
    TaskType,
    EmbeddingProviderError,
)


class OpenAIEmbeddingProvider(EmbeddingProviderInterface):
    """
    OpenAI embedding provider.

    Supports OpenAI embedding models including text-embedding-3-small and
    text-embedding-3-large with configurable dimensions.
    """

    # Model specifications
    MODEL_SPECS = {
        "text-embedding-3-small": {
            "default_dimension": 1536,
            "max_dimension": 1536,
            "supports_variable_dim": True,
        },
        "text-embedding-3-large": {
            "default_dimension": 3072,
            "max_dimension": 3072,
            "supports_variable_dim": True,
        },
        "text-embedding-ada-002": {
            "default_dimension": 1536,
            "max_dimension": 1536,
            "supports_variable_dim": False,
        },
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the OpenAI embedding provider.

        Args:
            config: Configuration dictionary with keys:
                - api_key: OpenAI API key (required)
                - model_name: OpenAI model name (default: 'text-embedding-3-small')
                - dimension: Output dimension (default: model default)
                - max_batch_size: Maximum batch size (default: 100)
                - timeout: Request timeout in seconds (default: 30)
                - base_url: Custom base URL (optional)
                - organization: OpenAI organization ID (optional)
        """
        super().__init__(config)

        self.logger = logging.getLogger(__name__)

        # Extract configuration
        self.api_key = config.get("api_key")
        if not self.api_key:
            raise EmbeddingProviderError(
                "OpenAI API key is required for OpenAI provider",
                provider="openai",
                details={"config_key": "api_key"},
            )

        self._model_name = config.get("model_name", "text-embedding-3-small")
        self.timeout = config.get("timeout", 30)
        self.base_url = config.get("base_url")
        self.organization = config.get("organization")

        # Validate model and set dimension
        if self._model_name not in self.MODEL_SPECS:
            raise EmbeddingProviderError(
                f"Unsupported OpenAI model: {self._model_name}",
                provider="openai",
                details={
                    "model": self._model_name,
                    "supported_models": list(self.MODEL_SPECS.keys()),
                },
            )

        model_spec = self.MODEL_SPECS[self._model_name]
        requested_dimension = config.get("dimension")

        if requested_dimension:
            if not model_spec["supports_variable_dim"]:
                if requested_dimension != model_spec["default_dimension"]:
                    self.logger.warning(
                        f"Model {self._model_name} does not support variable dimensions. "
                        f"Using default dimension {model_spec['default_dimension']}"
                    )
                self._dimension = model_spec["default_dimension"]
            else:
                if requested_dimension > model_spec["max_dimension"]:
                    raise EmbeddingProviderError(
                        f"Requested dimension {requested_dimension} exceeds maximum "
                        f"{model_spec['max_dimension']} for model {self._model_name}",
                        provider="openai",
                        details={
                            "requested": requested_dimension,
                            "max_allowed": model_spec["max_dimension"],
                        },
                    )
                self._dimension = requested_dimension
        else:
            self._dimension = model_spec["default_dimension"]

        # Initialize OpenAI client
        try:
            client_kwargs = {"api_key": self.api_key, "timeout": self.timeout}

            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            if self.organization:
                client_kwargs["organization"] = self.organization

            self.client = AsyncOpenAI(**client_kwargs)

        except Exception as e:
            raise EmbeddingProviderError(
                f"Failed to initialize OpenAI client: {str(e)}",
                provider="openai",
                details={"error": str(e)},
            )

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    async def generate_embedding(
        self, text: str, task_type: TaskType = TaskType.SEMANTIC_SIMILARITY
    ) -> np.ndarray:
        """
        Generate embedding for a single text using OpenAI API.

        Args:
            text: Input text to embed
            task_type: Type of embedding task (for logging purposes)

        Returns:
            Embedding vector as numpy array

        Raises:
            EmbeddingProviderError: If embedding generation fails
        """
        if not text or not text.strip():
            raise EmbeddingProviderError(
                "Text cannot be empty or None",
                provider="openai",
                details={"text_length": len(text) if text else 0},
            )

        try:
            self.logger.debug(f"Generating OpenAI embedding for text: {text[:50]}...")

            # Prepare embedding request parameters
            embedding_params = {"model": self._model_name, "input": text}

            # Add dimension parameter if supported and different from default
            model_spec = self.MODEL_SPECS[self._model_name]
            if (
                model_spec["supports_variable_dim"]
                and self._dimension != model_spec["default_dimension"]
            ):
                embedding_params["dimensions"] = self._dimension

            # Generate embedding
            response = await self.client.embeddings.create(**embedding_params)

            # Extract embedding from response
            if not response.data or len(response.data) == 0:
                raise EmbeddingProviderError(
                    "Empty response from OpenAI API",
                    provider="openai",
                    details={"response": str(response)},
                )

            embedding = response.data[0].embedding

            # Validate embedding dimension
            if len(embedding) != self._dimension:
                raise EmbeddingProviderError(
                    f"Embedding dimension mismatch: expected {self._dimension}, got {len(embedding)}",
                    provider="openai",
                    details={
                        "expected": self._dimension,
                        "actual": len(embedding),
                        "model": self._model_name,
                    },
                )

            self.logger.debug(f"Generated OpenAI embedding of dimension {len(embedding)}")
            return np.array(embedding, dtype=np.float32)

        except openai.AuthenticationError as e:
            self.logger.error(f"OpenAI authentication error: {str(e)}")
            raise EmbeddingProviderError(
                f"OpenAI authentication failed: {str(e)}",
                provider="openai",
                details={"error_type": "authentication", "error": str(e)},
            )

        except openai.RateLimitError as e:
            self.logger.error(f"OpenAI rate limit error: {str(e)}")
            raise EmbeddingProviderError(
                f"OpenAI rate limit exceeded: {str(e)}",
                provider="openai",
                details={"error_type": "rate_limit", "error": str(e)},
            )

        except openai.APIError as e:
            self.logger.error(f"OpenAI API error: {str(e)}")
            raise EmbeddingProviderError(
                f"OpenAI API error: {str(e)}",
                provider="openai",
                details={"error_type": "api_error", "error": str(e)},
            )

        except Exception as e:
            self.logger.error(f"Error generating OpenAI embedding: {str(e)}")
            raise EmbeddingProviderError(
                f"Failed to generate OpenAI embedding: {str(e)}",
                provider="openai",
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
            task_type: Type of embedding task (for logging purposes)

        Returns:
            List of embedding vectors as numpy arrays

        Raises:
            EmbeddingProviderError: If embedding generation fails
        """
        if not texts:
            return []

        embeddings = []
        failed_indices = []

        # Process in batches
        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i : i + self.max_batch_size]
            batch_start_idx = i

            try:
                batch_embeddings = await self._generate_batch_embeddings(batch, task_type)
                embeddings.extend(batch_embeddings)

            except Exception as e:
                self.logger.warning(f"Batch embedding failed, falling back to individual: {str(e)}")

                # Fall back to individual processing for this batch
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
                        embeddings.append(np.zeros(self._dimension, dtype=np.float32))

        if failed_indices:
            self.logger.warning(f"Failed to generate embeddings for {len(failed_indices)} texts")

        return embeddings

    async def _generate_batch_embeddings(
        self, texts: List[str], task_type: TaskType
    ) -> List[np.ndarray]:
        """Generate embeddings for a batch of texts."""
        try:
            self.logger.debug(f"Generating batch embeddings for {len(texts)} texts")

            # Prepare embedding request parameters
            embedding_params = {"model": self._model_name, "input": texts}

            # Add dimension parameter if supported and different from default
            model_spec = self.MODEL_SPECS[self._model_name]
            if (
                model_spec["supports_variable_dim"]
                and self._dimension != model_spec["default_dimension"]
            ):
                embedding_params["dimensions"] = self._dimension

            # Generate embeddings for batch
            response = await self.client.embeddings.create(**embedding_params)

            # Validate response
            if not response.data or len(response.data) != len(texts):
                raise EmbeddingProviderError(
                    f"Batch response length mismatch: expected {len(texts)}, got {len(response.data) if response.data else 0}",
                    provider="openai",
                    details={
                        "expected": len(texts),
                        "actual": len(response.data) if response.data else 0,
                    },
                )

            # Extract embeddings from response
            embeddings = []
            for i, embedding_data in enumerate(response.data):
                embedding = embedding_data.embedding

                # Validate embedding dimension
                if len(embedding) != self._dimension:
                    raise EmbeddingProviderError(
                        f"Embedding dimension mismatch at index {i}: expected {self._dimension}, got {len(embedding)}",
                        provider="openai",
                        details={"index": i, "expected": self._dimension, "actual": len(embedding)},
                    )

                embeddings.append(np.array(embedding, dtype=np.float32))

            self.logger.debug(f"Generated {len(embeddings)} batch embeddings")
            return embeddings

        except Exception as e:
            # Re-raise with context
            raise EmbeddingProviderError(
                f"Batch embedding generation failed: {str(e)}",
                provider="openai",
                details={"batch_size": len(texts), "model": self._model_name, "error": str(e)},
            )

    def is_available(self) -> bool:
        """Check if the OpenAI provider is available."""
        return bool(self.api_key and self.client)

    async def test_connection(self) -> bool:
        """Test connection to OpenAI API."""
        try:
            # Test with a simple embedding request
            test_embedding = await self.generate_embedding(
                "Test connection", TaskType.SEMANTIC_SIMILARITY
            )
            return len(test_embedding) == self._dimension

        except Exception as e:
            self.logger.error(f"OpenAI connection test failed: {str(e)}")
            return False

    def get_supported_task_types(self) -> List[TaskType]:
        """
        Get supported task types for OpenAI.

        Note: OpenAI embeddings don't have explicit task type parameters,
        but the embeddings work well for all these use cases.
        """
        return [
            TaskType.SEMANTIC_SIMILARITY,
            TaskType.RETRIEVAL_DOCUMENT,
            TaskType.RETRIEVAL_QUERY,
            TaskType.CLUSTERING,
            TaskType.CLASSIFICATION,
        ]

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration."""
        model_spec = self.MODEL_SPECS[self._model_name]
        return {
            "model_name": self._model_name,
            "dimension": self._dimension,
            "default_dimension": model_spec["default_dimension"],
            "max_dimension": model_spec["max_dimension"],
            "supports_variable_dim": model_spec["supports_variable_dim"],
            "max_batch_size": self.max_batch_size,
        }
