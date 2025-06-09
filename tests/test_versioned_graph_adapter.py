"""
Tests for the VersionedGraphAdapter class.

This module verifies that the VersionedGraphAdapter correctly integrates
the GraphStorageAdapter with the RevisionManager.
"""
import time
from unittest.mock import MagicMock, patch

import pytest

from memory_core.db.janusgraph_storage import JanusGraphStorage
from memory_core.db.graph_storage_adapter import GraphStorageAdapter
from memory_core.db.versioned_graph_adapter import VersionedGraphAdapter
from memory_core.versioning.revision_manager import RevisionManager
from memory_core.model.knowledge_node import KnowledgeNode
from memory_core.model.relationship import Relationship


class TestVersionedGraphAdapter:
    """Test cases for the VersionedGraphAdapter class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create mock JanusGraphStorage
        self.storage = MagicMock(spec=JanusGraphStorage)
        
        # Create mock GraphStorageAdapter
        self.graph_adapter = MagicMock(spec=GraphStorageAdapter)
        
        # Create mock RevisionManager
        self.revision_manager = MagicMock(spec=RevisionManager)
        
        # Create VersionedGraphAdapter with mocks
        self.versioned_adapter = VersionedGraphAdapter(
            graph_adapter=self.graph_adapter,
            revision_manager=self.revision_manager
        )
        
        # Sample test data
        self.node_id = "node123"
        self.edge_id = "edge456"
        
        # Create sample knowledge node
        self.node = KnowledgeNode(
            node_id=self.node_id,
            content="Test content",
            source="Test source",
            creation_timestamp=time.time(),
            rating_richness=0.8,
            rating_truthfulness=0.9,
            rating_stability=0.7
        )
        
        # Create sample relationship
        self.relationship = Relationship(
            edge_id=self.edge_id,
            from_id="node1",
            to_id="node2",
            relation_type="is_a",
            timestamp=time.time(),
            confidence_score=0.85,
            version=1
        )

    def test_save_new_knowledge_node(self):
        """Test saving a new knowledge node."""
        # Setup: Create a new node without ID
        new_node = KnowledgeNode(
            content="New content",
            source="New source"
        )
        
        # Mock return values
        self.graph_adapter.save_knowledge_node.return_value = self.node_id
        self.graph_adapter.get_knowledge_node.return_value = self.node
        
        # Act
        result = self.versioned_adapter.save_knowledge_node(new_node)
        
        # Assert
        assert result == self.node_id
        
        # Verify the adapter was called
        self.graph_adapter.save_knowledge_node.assert_called_once_with(new_node)
        self.graph_adapter.get_knowledge_node.assert_called_once_with(self.node_id)
        
        # Verify revision manager was called to log the creation
        self.revision_manager.log_node_creation.assert_called_once()
        args = self.revision_manager.log_node_creation.call_args[1]
        assert args['node_id'] == self.node_id
        assert args['node_data'] == self.node.to_dict()

    def test_save_existing_knowledge_node(self):
        """Test updating an existing knowledge node."""
        # Setup: Use existing node with ID
        
        # Mock return values
        self.graph_adapter.get_knowledge_node.return_value = self.node
        self.graph_adapter.save_knowledge_node.return_value = self.node_id
        
        # Act
        result = self.versioned_adapter.save_knowledge_node(self.node)
        
        # Assert
        assert result == self.node_id
        
        # Verify the adapter was called twice (once for get, once for save)
        self.graph_adapter.get_knowledge_node.assert_called_once_with(self.node_id)
        self.graph_adapter.save_knowledge_node.assert_called_once_with(self.node)
        
        # Verify revision manager was called to log the update
        self.revision_manager.log_node_update.assert_called_once()
        args = self.revision_manager.log_node_update.call_args[1]
        assert args['node_id'] == self.node_id
        assert args['old_data'] == self.node.to_dict()
        assert args['new_data'] == self.node.to_dict()

    def test_delete_knowledge_node(self):
        """Test deleting a knowledge node."""
        # Setup
        self.graph_adapter.get_knowledge_node.return_value = self.node
        self.graph_adapter.delete_knowledge_node.return_value = True
        
        # Act
        result = self.versioned_adapter.delete_knowledge_node(self.node_id)
        
        # Assert
        assert result is True
        
        # Verify the adapter was called
        self.graph_adapter.get_knowledge_node.assert_called_once_with(self.node_id)
        self.graph_adapter.delete_knowledge_node.assert_called_once_with(self.node_id)
        
        # Verify revision manager was called to log the deletion
        self.revision_manager.log_node_deletion.assert_called_once()
        args = self.revision_manager.log_node_deletion.call_args[1]
        assert args['node_id'] == self.node_id
        assert args['node_data'] == self.node.to_dict()

    def test_save_new_relationship(self):
        """Test saving a new relationship."""
        # Setup: Create a new relationship without ID
        new_relationship = Relationship(
            from_id="node1",
            to_id="node2",
            relation_type="is_a"
        )
        
        # Mock return values
        self.graph_adapter.save_relationship.return_value = self.edge_id
        self.graph_adapter.get_relationship.return_value = self.relationship
        
        # Act
        result = self.versioned_adapter.save_relationship(new_relationship)
        
        # Assert
        assert result == self.edge_id
        
        # Verify the adapter was called
        self.graph_adapter.save_relationship.assert_called_once_with(new_relationship)
        self.graph_adapter.get_relationship.assert_called_once_with(self.edge_id)
        
        # Verify revision manager was called to log the creation
        self.revision_manager.log_edge_creation.assert_called_once()
        args = self.revision_manager.log_edge_creation.call_args[1]
        assert args['edge_id'] == self.edge_id
        assert args['edge_data'] == self.relationship.to_dict()

    def test_save_existing_relationship(self):
        """Test updating an existing relationship."""
        # Setup: Use existing relationship with ID
        
        # Mock return values
        self.graph_adapter.get_relationship.return_value = self.relationship
        self.graph_adapter.save_relationship.return_value = self.edge_id
        
        # Act
        result = self.versioned_adapter.save_relationship(self.relationship)
        
        # Assert
        assert result == self.edge_id
        
        # Verify the adapter was called twice (once for get, once for save)
        self.graph_adapter.get_relationship.assert_called_once_with(self.edge_id)
        self.graph_adapter.save_relationship.assert_called_once_with(self.relationship)
        
        # Verify revision manager was called to log the update
        self.revision_manager.log_edge_update.assert_called_once()
        args = self.revision_manager.log_edge_update.call_args[1]
        assert args['edge_id'] == self.edge_id
        assert args['old_data'] == self.relationship.to_dict()
        assert args['new_data'] == self.relationship.to_dict()

    def test_delete_relationship(self):
        """Test deleting a relationship."""
        # Setup
        self.graph_adapter.get_relationship.return_value = self.relationship
        self.graph_adapter.delete_relationship.return_value = True
        
        # Act
        result = self.versioned_adapter.delete_relationship(self.edge_id)
        
        # Assert
        assert result is True
        
        # Verify the adapter was called
        self.graph_adapter.get_relationship.assert_called_once_with(self.edge_id)
        self.graph_adapter.delete_relationship.assert_called_once_with(self.edge_id)
        
        # Verify revision manager was called to log the deletion
        self.revision_manager.log_edge_deletion.assert_called_once()
        args = self.revision_manager.log_edge_deletion.call_args[1]
        assert args['edge_id'] == self.edge_id
        assert args['edge_data'] == self.relationship.to_dict()

    def test_get_knowledge_node(self):
        """Test retrieving a knowledge node."""
        # Setup
        self.graph_adapter.get_knowledge_node.return_value = self.node
        
        # Act
        result = self.versioned_adapter.get_knowledge_node(self.node_id)
        
        # Assert
        assert result == self.node
        
        # Verify the adapter was called
        self.graph_adapter.get_knowledge_node.assert_called_once_with(self.node_id)
        
        # Verify revision manager was not called
        self.revision_manager.log_node_creation.assert_not_called()
        self.revision_manager.log_node_update.assert_not_called()
        self.revision_manager.log_node_deletion.assert_not_called()

    def test_get_relationship(self):
        """Test retrieving a relationship."""
        # Setup
        self.graph_adapter.get_relationship.return_value = self.relationship
        
        # Act
        result = self.versioned_adapter.get_relationship(self.edge_id)
        
        # Assert
        assert result == self.relationship
        
        # Verify the adapter was called
        self.graph_adapter.get_relationship.assert_called_once_with(self.edge_id)
        
        # Verify revision manager was not called
        self.revision_manager.log_edge_creation.assert_not_called()
        self.revision_manager.log_edge_update.assert_not_called()
        self.revision_manager.log_edge_deletion.assert_not_called()

    def test_revert_node_to_previous_state(self):
        """Test reverting a node to its previous state."""
        # Setup
        self.revision_manager.revert_node_to_previous_state.return_value = True
        
        # Act
        result = self.versioned_adapter.revert_node_to_previous_state(self.node_id)
        
        # Assert
        assert result is True
        
        # Verify revision manager was called
        self.revision_manager.revert_node_to_previous_state.assert_called_once_with(self.node_id)

    def test_revert_relationship_to_previous_state(self):
        """Test reverting a relationship to its previous state."""
        # Setup
        self.revision_manager.revert_edge_to_previous_state.return_value = True
        
        # Act
        result = self.versioned_adapter.revert_relationship_to_previous_state(self.edge_id)
        
        # Assert
        assert result is True
        
        # Verify revision manager was called
        self.revision_manager.revert_edge_to_previous_state.assert_called_once_with(self.edge_id)

    def test_create_snapshot(self):
        """Test creating a snapshot."""
        # Setup
        snapshot_id = "snapshot123"
        self.revision_manager.create_snapshot.return_value = snapshot_id
        
        # Act
        result = self.versioned_adapter.create_snapshot()
        
        # Assert
        assert result == snapshot_id
        
        # Verify revision manager was called
        self.revision_manager.create_snapshot.assert_called_once()

    def test_get_revision_history(self):
        """Test getting revision history."""
        # Setup
        history = [{"id": "rev1"}, {"id": "rev2"}]
        self.revision_manager.get_revision_history.return_value = history
        
        # Act
        result = self.versioned_adapter.get_revision_history("node", self.node_id)
        
        # Assert
        assert result == history
        
        # Verify revision manager was called
        self.revision_manager.get_revision_history.assert_called_once_with("node", self.node_id) 