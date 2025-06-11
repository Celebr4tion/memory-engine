"""
Storage interfaces for the memory engine.

This module defines abstract base classes that all storage implementations
must implement to ensure consistent behavior across different backends.
"""

from .graph_storage_interface import GraphStorageInterface

__all__ = ['GraphStorageInterface']