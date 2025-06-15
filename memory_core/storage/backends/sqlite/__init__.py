"""
SQLite storage backend implementation.

This module provides a SQLite-based implementation of the
graph storage interface for single-user deployments and applications
requiring a balance between simplicity and performance.
"""

from .sqlite_storage import SqliteStorage

__all__ = ["SqliteStorage"]
