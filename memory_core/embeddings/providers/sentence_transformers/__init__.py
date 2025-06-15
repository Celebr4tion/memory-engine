"""
Sentence Transformers embedding provider for Memory Engine.

This module provides local embedding generation using the sentence-transformers library,
supporting various pre-trained models for semantic text embeddings.
"""

from .sentence_transformers_provider import SentenceTransformersProvider

__all__ = ["SentenceTransformersProvider"]
