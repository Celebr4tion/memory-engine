"""
Tests for the KnowledgeEngine class.

This module tests the KnowledgeEngine class, which integrates all components
(storage, adapter, versioning) into a complete solution.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from memory_core.core.knowledge_engine import KnowledgeEngine
from memory_core.model.knowledge_node import KnowledgeNode
from memory_core.model.relationship import Relationship


class TestKnowledgeEngine:
    """Test cases for the KnowledgeEngine class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Sample test data
        self.node_id = "node123"
        self.edge_id = "edge456"

        # Create sample knowledge node
        self.node = KnowledgeNode(
            node_id=self.node_id,
            content="Test content",
            source="Test source",
            creation_timestamp=time.time(),
        )

        # Create sample relationship
        self.relationship = Relationship(
            edge_id=self.edge_id,
            from_id="node1",
            to_id="node2",
            relation_type="is_a",
            timestamp=time.time(),
        )

    @patch("memory_core.core.knowledge_engine.JanusGraphStorage")
    @patch("memory_core.core.knowledge_engine.GraphStorageAdapter")
    @patch("memory_core.core.knowledge_engine.RevisionManager")
    @patch("memory_core.core.knowledge_engine.VersionedGraphAdapter")
    def test_initialization_with_versioning(
        self, MockVersionedAdapter, MockRevisionManager, MockGraphAdapter, MockJanusStorage
    ):
        """Test initializing the KnowledgeEngine with versioning enabled."""
        # Create engine
        engine = KnowledgeEngine(
            host="localhost", port=8182, enable_versioning=True, enable_snapshots=True
        )

        # Verify constructor calls
        MockJanusStorage.assert_called_once_with(host="localhost", port=8182)
        MockGraphAdapter.assert_called_once()
        MockRevisionManager.assert_called_once()
        MockVersionedAdapter.assert_called_once()

        # Verify engine structure
        assert engine.graph is MockVersionedAdapter.return_value

    @patch("memory_core.core.knowledge_engine.JanusGraphStorage")
    @patch("memory_core.core.knowledge_engine.GraphStorageAdapter")
    @patch("memory_core.core.knowledge_engine.RevisionManager")
    @patch("memory_core.core.knowledge_engine.VersionedGraphAdapter")
    def test_initialization_without_versioning(
        self, MockVersionedAdapter, MockRevisionManager, MockGraphAdapter, MockJanusStorage
    ):
        """Test initializing the KnowledgeEngine with versioning disabled."""
        # Create engine
        engine = KnowledgeEngine(host="localhost", port=8182, enable_versioning=False)

        # Verify constructor calls
        MockJanusStorage.assert_called_with(host="localhost", port=8182)
        MockGraphAdapter.assert_called()
        MockRevisionManager.assert_not_called()
        MockVersionedAdapter.assert_not_called()

        # Verify engine structure
        assert engine.revision_manager is None
        assert engine.versioned_adapter is None
        assert engine.graph is MockGraphAdapter.return_value

    @patch("memory_core.core.knowledge_engine.JanusGraphStorage")
    @patch("memory_core.core.knowledge_engine.GraphStorageAdapter")
    @patch("memory_core.core.knowledge_engine.RevisionManager")
    @patch("memory_core.core.knowledge_engine.VersionedGraphAdapter")
    def test_connect_disconnect(
        self, MockVersionedAdapter, MockRevisionManager, MockGraphAdapter, MockJanusStorage
    ):
        """Test connecting and disconnecting from the database."""
        # Setup
        storage_mock = MockJanusStorage.return_value
        storage_mock.connect.return_value = True
        storage_mock.close.return_value = True

        # Create engine
        engine = KnowledgeEngine()

        # Act & Assert - Connect
        result = engine.connect()
        assert result is True
        storage_mock.connect.assert_called_once()

        # Act & Assert - Disconnect
        result = engine.disconnect()
        assert result is True
        storage_mock.close.assert_called_once()

    @patch("memory_core.core.knowledge_engine.JanusGraphStorage")
    @patch("memory_core.core.knowledge_engine.GraphStorageAdapter")
    @patch("memory_core.core.knowledge_engine.RevisionManager")
    @patch("memory_core.core.knowledge_engine.VersionedGraphAdapter")
    def test_save_node(
        self, MockVersionedAdapter, MockRevisionManager, MockGraphAdapter, MockJanusStorage
    ):
        """Test saving a node."""
        # Setup
        versioned_adapter_mock = MockVersionedAdapter.return_value
        versioned_adapter_mock.save_knowledge_node.return_value = self.node_id

        # Create engine
        engine = KnowledgeEngine(enable_versioning=True)

        # Act
        result = engine.save_node(self.node)

        # Assert
        assert result == self.node_id
        versioned_adapter_mock.save_knowledge_node.assert_called_once_with(self.node)

    @patch("memory_core.core.knowledge_engine.JanusGraphStorage")
    @patch("memory_core.core.knowledge_engine.GraphStorageAdapter")
    @patch("memory_core.core.knowledge_engine.RevisionManager")
    @patch("memory_core.core.knowledge_engine.VersionedGraphAdapter")
    def test_save_relationship(
        self, MockVersionedAdapter, MockRevisionManager, MockGraphAdapter, MockJanusStorage
    ):
        """Test saving a relationship."""
        # Setup
        versioned_adapter_mock = MockVersionedAdapter.return_value
        versioned_adapter_mock.save_relationship.return_value = self.edge_id

        # Create engine
        engine = KnowledgeEngine(enable_versioning=True)

        # Act
        result = engine.save_relationship(self.relationship)

        # Assert
        assert result == self.edge_id
        versioned_adapter_mock.save_relationship.assert_called_once_with(self.relationship)

    @patch("memory_core.core.knowledge_engine.JanusGraphStorage")
    @patch("memory_core.core.knowledge_engine.GraphStorageAdapter")
    @patch("memory_core.core.knowledge_engine.RevisionManager")
    @patch("memory_core.core.knowledge_engine.VersionedGraphAdapter")
    def test_versioning_methods_with_versioning_disabled(
        self, MockVersionedAdapter, MockRevisionManager, MockGraphAdapter, MockJanusStorage
    ):
        """Test versioning methods when versioning is disabled."""
        # Create engine without versioning
        engine = KnowledgeEngine(enable_versioning=False)

        # Verify that versioning methods raise ValueError when versioning is disabled
        with pytest.raises(ValueError, match="Versioning is disabled"):
            engine.revert_node(self.node_id)

        with pytest.raises(ValueError, match="Versioning is disabled"):
            engine.revert_relationship(self.edge_id)

        with pytest.raises(ValueError, match="Versioning is disabled"):
            engine.create_snapshot()

        with pytest.raises(ValueError, match="Versioning is disabled"):
            engine.get_revision_history("node", self.node_id)
