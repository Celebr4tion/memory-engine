"""
Storage layer for the memory engine.

This module provides abstract interfaces and concrete implementations
for different storage backends including graph databases and vector stores.
"""

from .interfaces.graph_storage_interface import GraphStorageInterface
from .factory import create_storage, list_available_backends, is_backend_available

__all__ = [
    "GraphStorageInterface",
    "create_storage",
    "list_available_backends",
    "is_backend_available",
]
