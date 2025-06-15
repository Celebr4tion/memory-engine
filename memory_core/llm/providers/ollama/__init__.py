"""
Ollama LLM provider for local model inference.

This module provides integration with Ollama for running local language models
without requiring external API services.
"""

from .ollama_provider import OllamaLLMProvider

__all__ = ['OllamaLLMProvider']