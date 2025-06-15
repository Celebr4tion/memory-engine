"""
Interfaces for the modular embedding system.
"""

from .embedding_provider_interface import (
    EmbeddingProviderInterface,
    TaskType,
    EmbeddingError,
    EmbeddingProviderError,
    EmbeddingConfigError,
    EmbeddingDimensionMismatchError,
)

__all__ = [
    "EmbeddingProviderInterface",
    "TaskType",
    "EmbeddingError",
    "EmbeddingProviderError",
    "EmbeddingConfigError",
    "EmbeddingDimensionMismatchError",
]
