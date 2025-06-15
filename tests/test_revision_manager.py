"""
Tests for the RevisionManager class.

These tests ensure that the revision manager correctly logs changes to the
knowledge graph and creates snapshots.
"""

import json
import time
import uuid
from unittest.mock import MagicMock, patch, call

import pytest

from memory_core.db.janusgraph_storage import JanusGraphStorage
from memory_core.versioning.revision_manager import RevisionManager
from memory_core.model.knowledge_node import KnowledgeNode
from memory_core.model.relationship import Relationship


class TestRevisionManager:
    """Test cases for the RevisionManager class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create a mock JanusGraphStorage
        self.storage = MagicMock(spec=JanusGraphStorage)

        # Mock create_node to return a predictable ID
        self.storage.create_node.return_value = "mock_log_id"

        # Add missing methods to the mock
        self.storage.get_all_nodes = MagicMock()
        self.storage.get_all_edges = MagicMock()
        self.storage.query_vertices = MagicMock()

        # Create the revision manager with the mock storage
        self.revision_manager = RevisionManager(
            storage=self.storage,
            changes_threshold=3,  # Small threshold for testing
            enable_snapshots=True,
        )

        # Sample test data
        self.mock_node_id = "node123"
        self.mock_edge_id = "edge456"
        self.mock_snapshot_id = str(uuid.uuid4())

        self.node_data = {
            "content": "Test content",
            "source": "Test source",
            "creation_timestamp": time.time(),
            "rating_richness": 0.8,
            "rating_truthfulness": 0.9,
            "rating_stability": 0.7,
        }

        self.edge_data = {
            "from_id": "node1",
            "to_id": "node2",
            "relation_type": "is_a",
            "timestamp": time.time(),
            "confidence_score": 0.85,
            "version": 1,
        }

    def test_log_node_creation(self):
        """Test logging the creation of a node."""
        # Act
        log_id = self.revision_manager.log_node_creation(self.mock_node_id, self.node_data)

        # Assert
        assert log_id == "mock_log_id"

        # Verify the storage was called with the correct arguments
        self.storage.create_node.assert_called_once()
        args = self.storage.create_node.call_args[0][0]

        assert args["object_type"] == "node"
        assert args["object_id"] == self.mock_node_id
        assert args["change_type"] == "create"
        assert args["old_data"] is None
        assert json.loads(args["new_data"]) == self.node_data

        # Verify changes counter was incremented
        assert self.revision_manager.changes_since_snapshot == 1

    def test_log_node_update(self):
        """Test logging the update of a node."""
        # Prepare test data
        old_data = self.node_data.copy()
        new_data = self.node_data.copy()
        new_data["content"] = "Updated content"

        # Act
        log_id = self.revision_manager.log_node_update(self.mock_node_id, old_data, new_data)

        # Assert
        assert log_id == "mock_log_id"

        # Verify the storage was called with the correct arguments
        self.storage.create_node.assert_called_once()
        args = self.storage.create_node.call_args[0][0]

        assert args["object_type"] == "node"
        assert args["object_id"] == self.mock_node_id
        assert args["change_type"] == "update"
        assert json.loads(args["old_data"]) == old_data
        assert json.loads(args["new_data"]) == new_data

        # Verify changes counter was incremented
        assert self.revision_manager.changes_since_snapshot == 1

    def test_log_node_deletion(self):
        """Test logging the deletion of a node."""
        # Act
        log_id = self.revision_manager.log_node_deletion(self.mock_node_id, self.node_data)

        # Assert
        assert log_id == "mock_log_id"

        # Verify the storage was called with the correct arguments
        self.storage.create_node.assert_called_once()
        args = self.storage.create_node.call_args[0][0]

        assert args["object_type"] == "node"
        assert args["object_id"] == self.mock_node_id
        assert args["change_type"] == "delete"
        assert json.loads(args["old_data"]) == self.node_data
        assert args["new_data"] is None

        # Verify changes counter was incremented
        assert self.revision_manager.changes_since_snapshot == 1

    def test_log_edge_creation(self):
        """Test logging the creation of an edge."""
        # Act
        log_id = self.revision_manager.log_edge_creation(self.mock_edge_id, self.edge_data)

        # Assert
        assert log_id == "mock_log_id"

        # Verify the storage was called with the correct arguments
        self.storage.create_node.assert_called_once()
        args = self.storage.create_node.call_args[0][0]

        assert args["object_type"] == "edge"
        assert args["object_id"] == self.mock_edge_id
        assert args["change_type"] == "create"
        assert args["old_data"] is None
        assert json.loads(args["new_data"]) == self.edge_data

        # Verify changes counter was incremented
        assert self.revision_manager.changes_since_snapshot == 1

    def test_log_edge_update(self):
        """Test logging the update of an edge."""
        # Prepare test data
        old_data = self.edge_data.copy()
        new_data = self.edge_data.copy()
        new_data["confidence_score"] = 0.95

        # Act
        log_id = self.revision_manager.log_edge_update(self.mock_edge_id, old_data, new_data)

        # Assert
        assert log_id == "mock_log_id"

        # Verify the storage was called with the correct arguments
        self.storage.create_node.assert_called_once()
        args = self.storage.create_node.call_args[0][0]

        assert args["object_type"] == "edge"
        assert args["object_id"] == self.mock_edge_id
        assert args["change_type"] == "update"
        assert json.loads(args["old_data"]) == old_data
        assert json.loads(args["new_data"]) == new_data

        # Verify changes counter was incremented
        assert self.revision_manager.changes_since_snapshot == 1

    def test_log_edge_deletion(self):
        """Test logging the deletion of an edge."""
        # Act
        log_id = self.revision_manager.log_edge_deletion(self.mock_edge_id, self.edge_data)

        # Assert
        assert log_id == "mock_log_id"

        # Verify the storage was called with the correct arguments
        self.storage.create_node.assert_called_once()
        args = self.storage.create_node.call_args[0][0]

        assert args["object_type"] == "edge"
        assert args["object_id"] == self.mock_edge_id
        assert args["change_type"] == "delete"
        assert json.loads(args["old_data"]) == self.edge_data
        assert args["new_data"] is None

        # Verify changes counter was incremented
        assert self.revision_manager.changes_since_snapshot == 1

    def test_periodic_snapshot(self):
        """Test that snapshots are created automatically after reaching the threshold."""
        # Patch the create_snapshot method to avoid actual implementation
        with patch.object(self.revision_manager, "create_snapshot") as mock_create_snapshot:
            mock_create_snapshot.return_value = self.mock_snapshot_id

            # Make changes until threshold is reached
            self.revision_manager.log_node_creation(self.mock_node_id, self.node_data)
            assert self.revision_manager.changes_since_snapshot == 1
            assert mock_create_snapshot.call_count == 0

            self.revision_manager.log_node_update(self.mock_node_id, self.node_data, self.node_data)
            assert self.revision_manager.changes_since_snapshot == 2
            assert mock_create_snapshot.call_count == 0

            # This should trigger a snapshot
            self.revision_manager.log_node_deletion(self.mock_node_id, self.node_data)

            # Verify snapshot was created and counter reset
            mock_create_snapshot.assert_called_once()
            assert self.revision_manager.changes_since_snapshot == 0

    def test_create_snapshot(self):
        """Test creating a snapshot of the current graph state."""
        # Mock the get_all_nodes and get_all_edges methods
        self.storage.get_all_nodes.return_value = [
            {"id": "node1", "content": "Content 1"},
            {"id": "node2", "content": "Content 2"},
        ]

        self.storage.get_all_edges.return_value = [
            {"id": "edge1", "from_id": "node1", "to_id": "node2"}
        ]

        # Act
        snapshot_id = self.revision_manager.create_snapshot()

        # Assert
        assert snapshot_id is not None

        # Verify storage was called to create snapshot vertex
        self.storage.create_node.assert_called_once()
        args = self.storage.create_node.call_args[0][0]

        assert args["snapshot_id"] == snapshot_id
        assert "timestamp" in args
        assert "data" in args

        # Verify snapshot data
        snapshot_data = json.loads(args["data"])
        assert snapshot_data["snapshot_id"] == snapshot_id
        assert "timestamp" in snapshot_data
        assert "nodes" in snapshot_data
        assert "edges" in snapshot_data
        assert len(snapshot_data["nodes"]) == 2
        assert len(snapshot_data["edges"]) == 1

    def test_get_revision_history(self):
        """Test retrieving revision history for a node."""
        # Mock the query_vertices method
        mock_revisions = [
            {
                "id": "log1",
                "object_type": "node",
                "object_id": self.mock_node_id,
                "change_type": "create",
                "timestamp": time.time() - 300,
                "old_data": None,
                "new_data": json.dumps(self.node_data),
            },
            {
                "id": "log2",
                "object_type": "node",
                "object_id": self.mock_node_id,
                "change_type": "update",
                "timestamp": time.time() - 200,
                "old_data": json.dumps(self.node_data),
                "new_data": json.dumps({**self.node_data, "content": "Updated content"}),
            },
        ]

        self.storage.query_vertices.return_value = mock_revisions

        # Act
        history = self.revision_manager.get_revision_history("node", self.mock_node_id)

        # Assert
        assert len(history) == 2

        # Verify history is sorted by timestamp in descending order
        assert history[0]["id"] == "log2"  # Most recent first
        assert history[1]["id"] == "log1"

        # Verify JSON data was parsed
        assert history[0]["new_data"]["content"] == "Updated content"
        assert history[1]["new_data"]["content"] == self.node_data["content"]

    def test_revert_node_to_previous_state(self):
        """Test reverting a node to its previous state."""
        # Create mock revision history
        current_revision = {
            "id": "log2",
            "object_type": "node",
            "object_id": self.mock_node_id,
            "change_type": "update",
            "timestamp": time.time(),
            "old_data": self.node_data,
            "new_data": {**self.node_data, "content": "Updated content"},
        }

        previous_revision = {
            "id": "log1",
            "object_type": "node",
            "object_id": self.mock_node_id,
            "change_type": "create",
            "timestamp": time.time() - 100,
            "old_data": None,
            "new_data": self.node_data,
        }

        # Mock get_revision_history
        with patch.object(self.revision_manager, "get_revision_history") as mock_get_history:
            mock_get_history.return_value = [current_revision, previous_revision]

            # Mock get_node to return current node data
            self.storage.get_node.return_value = {**self.node_data, "content": "Updated content"}

            # Act
            result = self.revision_manager.revert_node_to_previous_state(self.mock_node_id)

            # Assert
            assert result is True

            # Verify node was updated with previous data
            self.storage.update_node.assert_called_once_with(
                self.mock_node_id, previous_revision["new_data"]
            )

            # Verify revert was logged
            assert self.storage.create_node.call_count == 1

    def test_revert_edge_to_previous_state(self):
        """Test reverting an edge to its previous state."""
        # Create mock revision history
        current_revision = {
            "id": "log2",
            "object_type": "edge",
            "object_id": self.mock_edge_id,
            "change_type": "update",
            "timestamp": time.time(),
            "old_data": self.edge_data,
            "new_data": {**self.edge_data, "confidence_score": 0.95},
        }

        previous_revision = {
            "id": "log1",
            "object_type": "edge",
            "object_id": self.mock_edge_id,
            "change_type": "create",
            "timestamp": time.time() - 100,
            "old_data": None,
            "new_data": self.edge_data,
        }

        # Mock get_revision_history
        with patch.object(self.revision_manager, "get_revision_history") as mock_get_history:
            mock_get_history.return_value = [current_revision, previous_revision]

            # Mock get_edge to return current edge data
            self.storage.get_edge.return_value = {**self.edge_data, "confidence_score": 0.95}

            # Act
            result = self.revision_manager.revert_edge_to_previous_state(self.mock_edge_id)

            # Assert
            assert result is True

            # Verify edge was updated with previous data
            # The update should exclude from_id, to_id, and relation_type
            expected_update_data = {
                "timestamp": self.edge_data["timestamp"],
                "confidence_score": self.edge_data["confidence_score"],
                "version": self.edge_data["version"],
            }

            self.storage.update_edge.assert_called_once()

            # Verify revert was logged
            assert self.storage.create_node.call_count == 1

    def test_not_enough_revisions_to_revert(self):
        """Test handling the case where there aren't enough revisions to revert."""
        # Mock get_revision_history to return only one revision
        with patch.object(self.revision_manager, "get_revision_history") as mock_get_history:
            mock_get_history.return_value = [
                {
                    "id": "log1",
                    "object_type": "node",
                    "object_id": self.mock_node_id,
                    "change_type": "create",
                    "timestamp": time.time(),
                    "old_data": None,
                    "new_data": self.node_data,
                }
            ]

            # Act
            result = self.revision_manager.revert_node_to_previous_state(self.mock_node_id)

            # Assert
            assert result is False

            # Verify no update was performed
            self.storage.update_node.assert_not_called()

            # Verify no log entry was created
            self.storage.create_node.assert_not_called()
