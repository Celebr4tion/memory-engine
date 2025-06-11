"""
JSON file storage backend implementation.

This module provides a lightweight JSON file-based implementation of the
graph storage interface for development, testing, and small deployments.
"""

from .json_file_storage import JsonFileStorage

__all__ = ['JsonFileStorage']