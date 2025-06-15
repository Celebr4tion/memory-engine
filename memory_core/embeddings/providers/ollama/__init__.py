"""
Ollama embedding provider package.

This package provides embedding functionality using local Ollama server,
supporting various Ollama embedding models for fully local embedding generation.
"""

from .ollama_provider import OllamaEmbeddingProvider

__all__ = ['OllamaEmbeddingProvider']