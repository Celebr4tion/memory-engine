"""
Tests for the Relationship class.

These tests ensure that the Relationship data model behaves correctly.
"""

import time
import pytest
from memory_core.model.relationship import Relationship


class TestRelationship:
    """Test cases for the Relationship class."""

    def test_init_with_default_values(self):
        """Test initialization with default values."""
        # Arrange
        from_id = "node1"
        to_id = "node2"
        relation_type = "is_a"

        # Act
        relationship = Relationship(from_id=from_id, to_id=to_id, relation_type=relation_type)

        # Assert
        assert relationship.from_id == from_id
        assert relationship.to_id == to_id
        assert relationship.relation_type == relation_type
        assert relationship.timestamp is not None
        assert relationship.edge_id is None
        assert relationship.confidence_score == 0.5
        assert relationship.version == 1

    def test_init_with_all_values(self):
        """Test initialization with all values specified."""
        # Arrange
        edge_id = "edge123"
        from_id = "node1"
        to_id = "node2"
        relation_type = "part_of"
        timestamp = time.time()
        confidence_score = 0.8
        version = 2

        # Act
        relationship = Relationship(
            from_id=from_id,
            to_id=to_id,
            relation_type=relation_type,
            timestamp=timestamp,
            confidence_score=confidence_score,
            version=version,
            edge_id=edge_id,
        )

        # Assert
        assert relationship.edge_id == edge_id
        assert relationship.from_id == from_id
        assert relationship.to_id == to_id
        assert relationship.relation_type == relation_type
        assert relationship.timestamp == timestamp
        assert relationship.confidence_score == confidence_score
        assert relationship.version == version

    def test_to_dict(self):
        """Test conversion to dictionary."""
        # Arrange
        edge_id = "edge123"
        from_id = "node1"
        to_id = "node2"
        relation_type = "causes"
        timestamp = time.time()
        confidence_score = 0.8
        version = 2

        relationship = Relationship(
            from_id=from_id,
            to_id=to_id,
            relation_type=relation_type,
            timestamp=timestamp,
            confidence_score=confidence_score,
            version=version,
            edge_id=edge_id,
        )

        # Act
        relationship_dict = relationship.to_dict()

        # Assert
        assert relationship_dict["edge_id"] == edge_id
        assert relationship_dict["from_id"] == from_id
        assert relationship_dict["to_id"] == to_id
        assert relationship_dict["relation_type"] == relation_type
        assert relationship_dict["timestamp"] == timestamp
        assert relationship_dict["confidence_score"] == confidence_score
        assert relationship_dict["version"] == version

    def test_from_dict(self):
        """Test creation from dictionary."""
        # Arrange
        relationship_dict = {
            "edge_id": "edge123",
            "from_id": "node1",
            "to_id": "node2",
            "relation_type": "contradicts",
            "timestamp": time.time(),
            "confidence_score": 0.8,
            "version": 2,
        }

        # Act
        relationship = Relationship.from_dict(relationship_dict)

        # Assert
        assert relationship.edge_id == relationship_dict["edge_id"]
        assert relationship.from_id == relationship_dict["from_id"]
        assert relationship.to_id == relationship_dict["to_id"]
        assert relationship.relation_type == relationship_dict["relation_type"]
        assert relationship.timestamp == relationship_dict["timestamp"]
        assert relationship.confidence_score == relationship_dict["confidence_score"]
        assert relationship.version == relationship_dict["version"]

    def test_round_trip_conversion(self):
        """Test converting to dict and back creates an equal object."""
        # Arrange
        original_relationship = Relationship(
            from_id="node1",
            to_id="node2",
            relation_type="custom_user_defined",
            timestamp=time.time(),
            confidence_score=0.9,
            version=3,
            edge_id="edge123",
        )

        # Act
        relationship_dict = original_relationship.to_dict()
        reconstructed_relationship = Relationship.from_dict(relationship_dict)

        # Assert
        assert reconstructed_relationship == original_relationship

    def test_equality_comparison(self):
        """Test that equality comparison works correctly."""
        # Arrange
        timestamp = time.time()
        relationship1 = Relationship(
            from_id="node1",
            to_id="node2",
            relation_type="is_a",
            timestamp=timestamp,
            confidence_score=0.8,
            version=2,
            edge_id="edge123",
        )

        relationship2 = Relationship(
            from_id="node1",
            to_id="node2",
            relation_type="is_a",
            timestamp=timestamp,
            confidence_score=0.8,
            version=2,
            edge_id="edge123",
        )

        relationship3 = Relationship(
            from_id="node1",
            to_id="node3",  # Different to_id
            relation_type="is_a",
            timestamp=timestamp,
            confidence_score=0.8,
            version=2,
            edge_id="edge123",
        )

        # Assert
        assert relationship1 == relationship2
        assert relationship1 != relationship3
        assert relationship1 != "not a relationship"

    def test_various_relation_types(self):
        """Test creation with various relation types."""
        # Test different relation types
        relation_types = ["is_a", "part_of", "causes", "contradicts", "custom_user_defined"]

        for relation_type in relation_types:
            relationship = Relationship(from_id="node1", to_id="node2", relation_type=relation_type)
            assert relationship.relation_type == relation_type
