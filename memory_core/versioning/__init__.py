"""
Versioning package for tracking changes to the knowledge graph.

This package provides functionality for recording changes to nodes and edges,
creating periodic snapshots, and reverting to previous states.
"""

from memory_core.versioning.revision_manager import RevisionManager

__all__ = ["RevisionManager"]
