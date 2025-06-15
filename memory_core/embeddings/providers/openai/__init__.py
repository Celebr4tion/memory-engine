"""
OpenAI embedding provider package.

This package provides integration with OpenAI's embedding models including
text-embedding-3-small and text-embedding-3-large.
"""

from .openai_provider import OpenAIEmbeddingProvider

__all__ = ['OpenAIEmbeddingProvider']