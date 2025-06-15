"""
Plugin system for Memory Engine custom backends and extensions.

This module provides a plugin architecture for:
- Custom storage backends
- Custom LLM providers  
- Custom embedding providers
- Custom processing components
- Plugin discovery and loading
"""

from .plugin_manager import PluginManager, PluginInfo
from .storage_plugin import StoragePlugin, StoragePluginInterface
from .llm_plugin import LLMPlugin, LLMPluginInterface
from .embedding_plugin import EmbeddingPlugin, EmbeddingPluginInterface
from .plugin_registry import PluginRegistry

__all__ = [
    'PluginManager',
    'PluginInfo',
    'StoragePlugin',
    'StoragePluginInterface',
    'LLMPlugin', 
    'LLMPluginInterface',
    'EmbeddingPlugin',
    'EmbeddingPluginInterface',
    'PluginRegistry'
]