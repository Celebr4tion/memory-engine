"""
Embedding providers package.

This package contains implementations of various embedding providers
for the Memory Engine's modular embedding system.
"""

# Import all available providers
from .gemini import GeminiEmbeddingProvider
from .ollama import OllamaEmbeddingProvider

# Note: Other providers may require additional dependencies
try:
    from .openai import OpenAIEmbeddingProvider
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

try:
    from .sentence_transformers import SentenceTransformersEmbeddingProvider
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False

# Registry of available providers
PROVIDER_REGISTRY = {
    'gemini': GeminiEmbeddingProvider,
    'ollama': OllamaEmbeddingProvider,
}

if _OPENAI_AVAILABLE:
    PROVIDER_REGISTRY['openai'] = OpenAIEmbeddingProvider

if _SENTENCE_TRANSFORMERS_AVAILABLE:
    PROVIDER_REGISTRY['sentence_transformers'] = SentenceTransformersEmbeddingProvider

# Export available providers
__all__ = ['GeminiEmbeddingProvider', 'OllamaEmbeddingProvider', 'PROVIDER_REGISTRY']

if _OPENAI_AVAILABLE:
    __all__.append('OpenAIEmbeddingProvider')

if _SENTENCE_TRANSFORMERS_AVAILABLE:
    __all__.append('SentenceTransformersEmbeddingProvider')


def get_available_providers():
    """Get list of available provider names."""
    return list(PROVIDER_REGISTRY.keys())


def get_provider_class(provider_name: str):
    """Get provider class by name."""
    if provider_name not in PROVIDER_REGISTRY:
        available = ', '.join(get_available_providers())
        raise ValueError(f"Unknown provider '{provider_name}'. Available providers: {available}")
    
    return PROVIDER_REGISTRY[provider_name]