"""
Google Gemini embedding provider implementation.

This module implements the EmbeddingProviderInterface for Google's Gemini API,
supporting various Gemini embedding models and task types.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

from google import genai
from google.genai import types

from memory_core.embeddings.interfaces import (
    EmbeddingProviderInterface,
    TaskType,
    EmbeddingProviderError,
)


class GeminiEmbeddingProvider(EmbeddingProviderInterface):
    """
    Google Gemini embedding provider.

    Supports Gemini embedding models including gemini-embedding-exp-03-07
    and other available Gemini embedding models.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Gemini embedding provider.

        Args:
            config: Configuration dictionary with keys:
                - api_key: Google API key (required)
                - model_name: Gemini model name (default: 'gemini-embedding-exp-03-07')
                - dimension: Output dimension for experimental models (default: 768)
                - max_batch_size: Maximum batch size (default: 32)
                - timeout: Request timeout in seconds (default: 30)
        """
        super().__init__(config)

        self.logger = logging.getLogger(__name__)

        # Extract configuration
        self.api_key = config.get("api_key")
        if not self.api_key:
            raise EmbeddingProviderError(
                "Google API key is required for Gemini provider",
                provider="gemini",
                details={"config_key": "api_key"},
            )

        self._model_name = config.get("model_name", "gemini-embedding-exp-03-07")
        self._dimension = config.get("dimension", 768)
        self.timeout = config.get("timeout", 30)

        # Initialize Gemini client
        try:
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            raise EmbeddingProviderError(
                f"Failed to initialize Gemini client: {str(e)}",
                provider="gemini",
                details={"error": str(e)},
            )

        # Task type mapping
        self._task_type_mapping = {
            TaskType.SEMANTIC_SIMILARITY: "SEMANTIC_SIMILARITY",
            TaskType.RETRIEVAL_DOCUMENT: "RETRIEVAL_DOCUMENT",
            TaskType.RETRIEVAL_QUERY: "RETRIEVAL_QUERY",
            TaskType.CLUSTERING: "CLUSTERING",
            TaskType.CLASSIFICATION: "CLASSIFICATION",
        }

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
        Generate embedding for a single text using Gemini API.

        Args:
            text: Input text to embed
            task_type: Type of embedding task

        Returns:
            Embedding vector as numpy array

        Raises:
            EmbeddingProviderError: If embedding generation fails
        """
        if not text or not text.strip():
            raise EmbeddingProviderError(
                "Text cannot be empty or None",
                provider="gemini",
                details={"text_length": len(text) if text else 0},
            )

        try:
            self.logger.debug(f"Generating Gemini embedding for text: {text[:50]}...")

            # Prepare embedding configuration
            gemini_task_type = self._task_type_mapping.get(task_type, "SEMANTIC_SIMILARITY")
            embedding_config = types.EmbedContentConfig(task_type=gemini_task_type)

            # Add dimension specification for experimental models
            if "gemini-embedding-exp" in self._model_name:
                embedding_config.output_dimensionality = self._dimension

            # Generate embedding
            result = self.client.models.embed_content(
                model=self._model_name, contents=text, config=embedding_config
            )

            # Parse response
            embedding = self._parse_embedding_response(result)

            self.logger.debug(f"Generated Gemini embedding of length {len(embedding)}")
            return np.array(embedding, dtype=np.float32)

        except Exception as e:
            self.logger.error(f"Error generating Gemini embedding: {str(e)}")
            raise EmbeddingProviderError(
                f"Failed to generate Gemini embedding: {str(e)}",
                provider="gemini",
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
        gemini_task_type = self._task_type_mapping.get(task_type, "SEMANTIC_SIMILARITY")
        embedding_config = types.EmbedContentConfig(task_type=gemini_task_type)

        if "gemini-embedding-exp" in self._model_name:
            embedding_config.output_dimensionality = self._dimension

        # Generate embeddings for batch
        result = self.client.models.embed_content(
            model=self._model_name, contents=texts, config=embedding_config
        )

        # Parse batch response
        embeddings = []
        if hasattr(result, "embeddings") and isinstance(result.embeddings, list):
            for embed_result in result.embeddings:
                embedding = self._parse_single_embedding(embed_result)
                embeddings.append(np.array(embedding, dtype=np.float32))
        else:
            raise EmbeddingProviderError(
                "Unexpected batch embedding response structure",
                provider="gemini",
                details={"response_type": str(type(result))},
            )

        return embeddings

    def _parse_embedding_response(self, result) -> List[float]:
        """Parse embedding response from Gemini API."""
        try:
            # Handle different response structures
            if hasattr(result, "embeddings"):
                if isinstance(result.embeddings, list) and result.embeddings:
                    first_item = result.embeddings[0]
                    return self._parse_single_embedding(first_item)
                else:
                    raise ValueError(f"Unexpected embeddings structure: {result.embeddings}")

            elif hasattr(result, "embedding"):
                return self._parse_single_embedding(result.embedding)

            else:
                raise ValueError(f"Unexpected embedding result structure: {result}")

        except Exception as e:
            raise EmbeddingProviderError(
                f"Failed to parse Gemini embedding response: {str(e)}",
                provider="gemini",
                details={"response": str(result), "error": str(e)},
            )

    def _parse_single_embedding(self, embed_item) -> List[float]:
        """Parse a single embedding from response."""
        if isinstance(embed_item, list):
            return embed_item
        elif hasattr(embed_item, "values"):
            return embed_item.values
        else:
            raise ValueError(f"Unexpected embedding item type: {type(embed_item)}")

    def is_available(self) -> bool:
        """Check if the Gemini provider is available."""
        return bool(self.api_key and self.client)

    async def test_connection(self) -> bool:
        """Test connection to Gemini API."""
        try:
            # Test with a simple embedding request
            test_embedding = await self.generate_embedding(
                "Test connection", TaskType.SEMANTIC_SIMILARITY
            )
            return len(test_embedding) == self._dimension

        except Exception as e:
            self.logger.error(f"Gemini connection test failed: {str(e)}")
            return False

    def get_supported_task_types(self) -> List[TaskType]:
        """Get supported task types for Gemini."""
        return [
            TaskType.SEMANTIC_SIMILARITY,
            TaskType.RETRIEVAL_DOCUMENT,
            TaskType.RETRIEVAL_QUERY,
            TaskType.CLUSTERING,
            TaskType.CLASSIFICATION,
        ]
