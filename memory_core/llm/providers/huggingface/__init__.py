"""
HuggingFace Transformers LLM provider for local and API-based language models.

This module implements the LLMProviderInterface for HuggingFace Transformers,
supporting both local model execution and HuggingFace Inference API.
"""

from .huggingface_provider import HuggingFaceLLMProvider

__all__ = ['HuggingFaceLLMProvider']