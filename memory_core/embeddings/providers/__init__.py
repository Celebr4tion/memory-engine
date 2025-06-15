"""
Embedding providers and factory system.
"""

from typing import Dict, Type, Optional, Any, List
from memory_core.embeddings.interfaces import EmbeddingProviderInterface

# Import all providers
from .gemini import GeminiEmbeddingProvider
from .openai import OpenAIEmbeddingProvider
from .sentence_transformers import SentenceTransformersProvider
from .ollama import OllamaEmbeddingProvider


class EmbeddingProviderFactory:
    """
    Factory for creating embedding provider instances.

    This factory provides a centralized way to instantiate embedding providers
    based on configuration, enabling easy switching between different providers.
    """

    # Registry of available providers
    _providers: Dict[str, Type[EmbeddingProviderInterface]] = {
        "gemini": GeminiEmbeddingProvider,
        "openai": OpenAIEmbeddingProvider,
        "sentence_transformers": SentenceTransformersProvider,
        "ollama": OllamaEmbeddingProvider,
    }

    @classmethod
    def create_provider(
        cls, provider_type: str, config: Dict[str, Any]
    ) -> EmbeddingProviderInterface:
        """
        Create an embedding provider instance.

        Args:
            provider_type: Type of provider ('gemini', 'openai', 'sentence_transformers', 'ollama')
            config: Provider-specific configuration dictionary

        Returns:
            Configured embedding provider instance

        Raises:
            ValueError: If provider type is not supported
            Exception: If provider initialization fails
        """
        provider_type = provider_type.lower()

        if provider_type not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(
                f"Unsupported embedding provider: {provider_type}. "
                f"Available providers: {available}"
            )

        provider_class = cls._providers[provider_type]

        try:
            return provider_class(config)
        except Exception as e:
            raise Exception(f"Failed to create {provider_type} provider: {str(e)}") from e

    @classmethod
    def get_available_providers(cls) -> List[str]:
        """
        Get list of available embedding providers.

        Returns:
            List of provider names
        """
        return list(cls._providers.keys())

    @classmethod
    def register_provider(cls, name: str, provider_class: Type[EmbeddingProviderInterface]) -> None:
        """
        Register a new embedding provider.

        Args:
            name: Name of the provider
            provider_class: Provider class implementing EmbeddingProviderInterface
        """
        if not issubclass(provider_class, EmbeddingProviderInterface):
            raise ValueError(f"Provider class must implement EmbeddingProviderInterface")

        cls._providers[name.lower()] = provider_class

    @classmethod
    def get_provider_class(cls, provider_type: str) -> Type[EmbeddingProviderInterface]:
        """
        Get the provider class for a given type.

        Args:
            provider_type: Type of provider

        Returns:
            Provider class

        Raises:
            ValueError: If provider type is not supported
        """
        provider_type = provider_type.lower()

        if provider_type not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(
                f"Unsupported embedding provider: {provider_type}. "
                f"Available providers: {available}"
            )

        return cls._providers[provider_type]


# Convenience function for creating providers
def create_embedding_provider(
    provider_type: str, config: Dict[str, Any]
) -> EmbeddingProviderInterface:
    """
    Create an embedding provider instance.

    Args:
        provider_type: Type of provider ('gemini', 'openai', 'sentence_transformers', 'ollama')
        config: Provider-specific configuration dictionary

    Returns:
        Configured embedding provider instance
    """
    return EmbeddingProviderFactory.create_provider(provider_type, config)


__all__ = [
    "EmbeddingProviderFactory",
    "create_embedding_provider",
    "GeminiEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "SentenceTransformersProvider",
    "OllamaEmbeddingProvider",
]
