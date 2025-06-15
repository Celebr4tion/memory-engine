"""
Knowledge Engine module that integrates all components.

This module provides a unified interface for working with the knowledge graph,
combining the storage, adapter, and versioning components.
"""

from typing import Dict, Any, Optional, List, Tuple, Union

from memory_core.db.janusgraph_storage import JanusGraphStorage
from memory_core.db.graph_storage_adapter import GraphStorageAdapter
from memory_core.db.versioned_graph_adapter import VersionedGraphAdapter
from memory_core.versioning.revision_manager import RevisionManager
from memory_core.model.knowledge_node import KnowledgeNode
from memory_core.model.relationship import Relationship
from memory_core.embeddings.embedding_manager import EmbeddingManager
from memory_core.embeddings.vector_store import VectorStoreMilvus


class KnowledgeEngine:
    """
    Main entry point for working with the knowledge graph.

    This class integrates all components (storage, adapter, versioning)
    into a complete solution for managing knowledge in a graph database.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8182,
        changes_threshold: int = 100,
        enable_versioning: bool = True,
        enable_snapshots: bool = True,
    ):
        """
        Initialize the KnowledgeEngine.

        Args:
            host: JanusGraph server host
            port: JanusGraph server port
            changes_threshold: Number of changes before creating a snapshot
            enable_versioning: Whether to enable versioning
            enable_snapshots: Whether to enable periodic snapshots
        """
        # Create the storage layer
        self.storage = JanusGraphStorage(host=host, port=port)

        # Create the graph adapter
        self.graph_adapter = GraphStorageAdapter(storage=self.storage)

        # Initialize embedding manager with vector store
        try:
            self.vector_store = VectorStoreMilvus()
            self.embedding_manager = EmbeddingManager(self.vector_store)
        except Exception as e:
            # If embedding manager fails to initialize, set to None
            # This allows the engine to work without embeddings
            self.vector_store = None
            self.embedding_manager = None

        # Create the revision manager if versioning is enabled
        if enable_versioning:
            self.revision_manager = RevisionManager(
                storage=self.storage,
                changes_threshold=changes_threshold,
                enable_snapshots=enable_snapshots,
            )

            # Create the versioned graph adapter
            self.versioned_adapter = VersionedGraphAdapter(
                graph_adapter=self.graph_adapter, revision_manager=self.revision_manager
            )

            # Use the versioned adapter as the primary interface
            self.graph = self.versioned_adapter
        else:
            # Use the basic adapter if versioning is disabled
            self.revision_manager = None
            self.versioned_adapter = None
            self.graph = self.graph_adapter

    def connect(self) -> bool:
        """
        Connect to the JanusGraph database.

        Returns:
            True if connection successful, False otherwise
        """
        return self.storage.connect()

    def disconnect(self) -> bool:
        """
        Disconnect from the JanusGraph database.

        Returns:
            True if disconnection successful, False otherwise
        """
        try:
            self.storage.close()
            return True
        except Exception:
            return False

    def save_node(self, node: KnowledgeNode) -> str:
        """
        Save a knowledge node to the graph.

        Args:
            node: The KnowledgeNode to save

        Returns:
            The ID of the saved node
        """
        return self.graph.save_knowledge_node(node)

    def get_node(self, node_id: str) -> KnowledgeNode:
        """
        Get a knowledge node by ID.

        Args:
            node_id: The ID of the node to retrieve

        Returns:
            The retrieved KnowledgeNode

        Raises:
            ValueError: If the node does not exist
        """
        return self.graph.get_knowledge_node(node_id)

    def delete_node(self, node_id: str) -> bool:
        """
        Delete a knowledge node.

        Args:
            node_id: The ID of the node to delete

        Returns:
            True if successful, False otherwise
        """
        return self.graph.delete_knowledge_node(node_id)

    def save_relationship(self, relationship: Relationship) -> str:
        """
        Save a relationship to the graph.

        Args:
            relationship: The Relationship to save

        Returns:
            The ID of the saved relationship
        """
        return self.graph.save_relationship(relationship)

    def get_relationship(self, edge_id: str) -> Relationship:
        """
        Get a relationship by ID.

        Args:
            edge_id: The ID of the relationship to retrieve

        Returns:
            The retrieved Relationship

        Raises:
            ValueError: If the relationship does not exist
        """
        return self.graph.get_relationship(edge_id)

    def delete_relationship(self, edge_id: str) -> bool:
        """
        Delete a relationship.

        Args:
            edge_id: The ID of the relationship to delete

        Returns:
            True if successful, False otherwise
        """
        return self.graph.delete_relationship(edge_id)

    def get_outgoing_relationships(self, node_id: str) -> List[Relationship]:
        """
        Get all outgoing relationships from a node.

        Args:
            node_id: The ID of the source node

        Returns:
            List of relationships where the node is the source
        """
        return self.graph.get_outgoing_relationships(node_id)

    def get_incoming_relationships(self, node_id: str) -> List[Relationship]:
        """
        Get all incoming relationships to a node.

        Args:
            node_id: The ID of the target node

        Returns:
            List of relationships where the node is the target
        """
        return self.graph.get_incoming_relationships(node_id)

    def revert_node(self, node_id: str) -> bool:
        """
        Revert a node to its previous state.

        Args:
            node_id: ID of the node to revert

        Returns:
            True if successful, False otherwise

        Raises:
            ValueError: If versioning is disabled
        """
        if self.versioned_adapter is None:
            raise ValueError("Versioning is disabled")

        return self.versioned_adapter.revert_node_to_previous_state(node_id)

    def revert_relationship(self, edge_id: str) -> bool:
        """
        Revert a relationship to its previous state.

        Args:
            edge_id: ID of the relationship to revert

        Returns:
            True if successful, False otherwise

        Raises:
            ValueError: If versioning is disabled
        """
        if self.versioned_adapter is None:
            raise ValueError("Versioning is disabled")

        return self.versioned_adapter.revert_relationship_to_previous_state(edge_id)

    def create_snapshot(self) -> str:
        """
        Create a snapshot of the current graph state.

        Returns:
            ID of the created snapshot

        Raises:
            ValueError: If versioning is disabled
        """
        if self.revision_manager is None:
            raise ValueError("Versioning is disabled")

        return self.revision_manager.create_snapshot()

    def get_revision_history(self, object_type: str, object_id: str) -> List[Dict[str, Any]]:
        """
        Get the revision history for a specific object.

        Args:
            object_type: Type of object ('node' or 'edge')
            object_id: ID of the object

        Returns:
            List of revision entries, sorted by timestamp (descending)

        Raises:
            ValueError: If versioning is disabled
        """
        if self.revision_manager is None:
            raise ValueError("Versioning is disabled")

        return self.revision_manager.get_revision_history(object_type, object_id)
