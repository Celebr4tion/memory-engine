"""
Tests for the KnowledgeNode class.

These tests ensure that the KnowledgeNode data model behaves correctly.
"""

import time
import pytest
from memory_core.model.knowledge_node import KnowledgeNode


class TestKnowledgeNode:
    """Test cases for the KnowledgeNode class."""

    def test_init_with_default_values(self):
        """Test initialization with default values."""
        # Arrange
        content = "This is test content"
        source = "Test Source"

        # Act
        node = KnowledgeNode(content=content, source=source)

        # Assert
        assert node.content == content
        assert node.source == source
        assert node.creation_timestamp is not None
        assert node.node_id is None
        assert node.rating_richness == 0.5
        assert node.rating_truthfulness == 0.5
        assert node.rating_stability == 0.5

    def test_init_with_all_values(self):
        """Test initialization with all values specified."""
        # Arrange
        node_id = "test123"
        content = "This is test content"
        source = "Test Source"
        timestamp = time.time()
        richness = 0.8
        truthfulness = 0.9
        stability = 0.7

        # Act
        node = KnowledgeNode(
            content=content,
            source=source,
            creation_timestamp=timestamp,
            rating_richness=richness,
            rating_truthfulness=truthfulness,
            rating_stability=stability,
            node_id=node_id,
        )

        # Assert
        assert node.node_id == node_id
        assert node.content == content
        assert node.source == source
        assert node.creation_timestamp == timestamp
        assert node.rating_richness == richness
        assert node.rating_truthfulness == truthfulness
        assert node.rating_stability == stability

    def test_to_dict(self):
        """Test conversion to dictionary."""
        # Arrange
        node_id = "test123"
        content = "This is test content"
        source = "Test Source"
        timestamp = time.time()
        richness = 0.8
        truthfulness = 0.9
        stability = 0.7

        node = KnowledgeNode(
            content=content,
            source=source,
            creation_timestamp=timestamp,
            rating_richness=richness,
            rating_truthfulness=truthfulness,
            rating_stability=stability,
            node_id=node_id,
        )

        # Act
        node_dict = node.to_dict()

        # Assert
        assert node_dict["node_id"] == node_id
        assert node_dict["content"] == content
        assert node_dict["source"] == source
        assert node_dict["creation_timestamp"] == timestamp
        assert node_dict["rating_richness"] == richness
        assert node_dict["rating_truthfulness"] == truthfulness
        assert node_dict["rating_stability"] == stability

    def test_from_dict(self):
        """Test creation from dictionary."""
        # Arrange
        node_dict = {
            "node_id": "test123",
            "content": "This is test content",
            "source": "Test Source",
            "creation_timestamp": time.time(),
            "rating_richness": 0.8,
            "rating_truthfulness": 0.9,
            "rating_stability": 0.7,
        }

        # Act
        node = KnowledgeNode.from_dict(node_dict)

        # Assert
        assert node.node_id == node_dict["node_id"]
        assert node.content == node_dict["content"]
        assert node.source == node_dict["source"]
        assert node.creation_timestamp == node_dict["creation_timestamp"]
        assert node.rating_richness == node_dict["rating_richness"]
        assert node.rating_truthfulness == node_dict["rating_truthfulness"]
        assert node.rating_stability == node_dict["rating_stability"]

    def test_round_trip_conversion(self):
        """Test converting to dict and back creates an equal object."""
        # Arrange
        original_node = KnowledgeNode(
            content="This is test content",
            source="Test Source",
            creation_timestamp=time.time(),
            rating_richness=0.8,
            rating_truthfulness=0.9,
            rating_stability=0.7,
            node_id="test123",
        )

        # Act
        node_dict = original_node.to_dict()
        reconstructed_node = KnowledgeNode.from_dict(node_dict)

        # Assert
        assert reconstructed_node == original_node

    def test_equality_comparison(self):
        """Test that equality comparison works correctly."""
        # Arrange
        timestamp = time.time()
        node1 = KnowledgeNode(
            content="This is test content",
            source="Test Source",
            creation_timestamp=timestamp,
            rating_richness=0.8,
            rating_truthfulness=0.9,
            rating_stability=0.7,
            node_id="test123",
        )

        node2 = KnowledgeNode(
            content="This is test content",
            source="Test Source",
            creation_timestamp=timestamp,
            rating_richness=0.8,
            rating_truthfulness=0.9,
            rating_stability=0.7,
            node_id="test123",
        )

        node3 = KnowledgeNode(
            content="This is different content",
            source="Test Source",
            creation_timestamp=timestamp,
            rating_richness=0.8,
            rating_truthfulness=0.9,
            rating_stability=0.7,
            node_id="test123",
        )

        # Assert
        assert node1 == node2
        assert node1 != node3
        assert node1 != "not a node"

    def test_long_content_representation(self):
        """Test the string representation with long content."""
        # Arrange
        long_content = "This is a very long piece of content that should be truncated in the string representation."
        node = KnowledgeNode(content=long_content, source="Test Source")

        # Act
        node_str = repr(node)

        # Assert
        assert "..." in node_str
        assert long_content[:30] in node_str
