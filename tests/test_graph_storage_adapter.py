"""
Tests for the GraphStorageAdapter class.

These tests ensure that the adapter correctly handles the conversion between
domain models and JanusGraph storage.
"""
import time
import unittest
from unittest.mock import MagicMock, patch

import pytest

from memory_core.db.janusgraph_storage import JanusGraphStorage
from memory_core.db.graph_storage_adapter import GraphStorageAdapter
from memory_core.model.knowledge_node import KnowledgeNode
from memory_core.model.relationship import Relationship


class TestGraphStorageAdapter(unittest.TestCase):
    """Test cases for the GraphStorageAdapter class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock JanusGraphStorage
        self.storage = MagicMock(spec=JanusGraphStorage)
        
        # Create the adapter with the mock storage
        self.adapter = GraphStorageAdapter(self.storage)
        
        # Sample test data
        self.sample_node = KnowledgeNode(
            content="Test content",
            source="Test source",
            creation_timestamp=time.time(),
            rating_richness=0.8,
            rating_truthfulness=0.9,
            rating_stability=0.7,
        )
        
        self.sample_relationship = Relationship(
            from_id="node1",
            to_id="node2",
            relation_type="is_a",
            timestamp=time.time(),
            confidence_score=0.85,
            version=1,
        )
        
        # Mock IDs
        self.mock_node_id = "node123"
        self.mock_edge_id = "edge456"

    def test_save_new_knowledge_node(self):
        """Test saving a new KnowledgeNode."""
        # Setup the mock to return a node ID
        self.storage.create_node.return_value = self.mock_node_id
        
        # Act
        node_id = self.adapter.save_knowledge_node(self.sample_node)
        
        # Assert
        assert node_id == self.mock_node_id
        self.storage.create_node.assert_called_once()
        
        # Verify correct data was passed to create_node
        args = self.storage.create_node.call_args[0][0]
        assert args['content'] == self.sample_node.content
        assert args['source'] == self.sample_node.source
        assert args['creation_timestamp'] == self.sample_node.creation_timestamp
        assert args['rating_richness'] == self.sample_node.rating_richness
        assert args['rating_truthfulness'] == self.sample_node.rating_truthfulness
        assert args['rating_stability'] == self.sample_node.rating_stability

    def test_save_existing_knowledge_node(self):
        """Test updating an existing KnowledgeNode."""
        # Modify sample node to have an ID
        self.sample_node.node_id = self.mock_node_id
        
        # Act
        node_id = self.adapter.save_knowledge_node(self.sample_node)
        
        # Assert
        assert node_id == self.mock_node_id
        self.storage.update_node.assert_called_once()
        
        # Verify correct data was passed to update_node
        args_id = self.storage.update_node.call_args[0][0]
        args_data = self.storage.update_node.call_args[0][1]
        assert args_id == self.mock_node_id
        assert args_data['content'] == self.sample_node.content
        assert args_data['source'] == self.sample_node.source
        assert args_data['creation_timestamp'] == self.sample_node.creation_timestamp
        assert args_data['rating_richness'] == self.sample_node.rating_richness
        assert args_data['rating_truthfulness'] == self.sample_node.rating_truthfulness
        assert args_data['rating_stability'] == self.sample_node.rating_stability

    def test_get_knowledge_node(self):
        """Test retrieving a KnowledgeNode."""
        # Setup the mock to return node data
        self.storage.get_node.return_value = {
            'content': self.sample_node.content,
            'source': self.sample_node.source,
            'creation_timestamp': self.sample_node.creation_timestamp,
            'rating_richness': self.sample_node.rating_richness,
            'rating_truthfulness': self.sample_node.rating_truthfulness,
            'rating_stability': self.sample_node.rating_stability,
            'label': 'KnowledgeNode',
            'id': self.mock_node_id
        }
        
        # Act
        node = self.adapter.get_knowledge_node(self.mock_node_id)
        
        # Assert
        self.storage.get_node.assert_called_once_with(self.mock_node_id)
        assert node.node_id == self.mock_node_id
        assert node.content == self.sample_node.content
        assert node.source == self.sample_node.source
        assert node.creation_timestamp == self.sample_node.creation_timestamp
        assert node.rating_richness == self.sample_node.rating_richness
        assert node.rating_truthfulness == self.sample_node.rating_truthfulness
        assert node.rating_stability == self.sample_node.rating_stability

    def test_delete_knowledge_node(self):
        """Test deleting a KnowledgeNode."""
        # Act
        self.adapter.delete_knowledge_node(self.mock_node_id)
        
        # Assert
        self.storage.delete_node.assert_called_once_with(self.mock_node_id)

    def test_save_new_relationship(self):
        """Test saving a new Relationship."""
        # Setup the mock to return an edge ID
        self.storage.create_edge.return_value = self.mock_edge_id
        
        # Act
        edge_id = self.adapter.save_relationship(self.sample_relationship)
        
        # Assert
        assert edge_id == self.mock_edge_id
        self.storage.create_edge.assert_called_once()
        
        # Verify correct data was passed to create_edge
        args_from_id = self.storage.create_edge.call_args[0][0]
        args_to_id = self.storage.create_edge.call_args[0][1]
        args_type = self.storage.create_edge.call_args[0][2]
        args_metadata = self.storage.create_edge.call_args[0][3]
        
        assert args_from_id == self.sample_relationship.from_id
        assert args_to_id == self.sample_relationship.to_id
        assert args_type == self.sample_relationship.relation_type
        assert args_metadata['timestamp'] == self.sample_relationship.timestamp
        assert args_metadata['confidence_score'] == self.sample_relationship.confidence_score
        assert args_metadata['version'] == self.sample_relationship.version

    def test_save_existing_relationship(self):
        """Test updating an existing Relationship."""
        # Modify sample relationship to have an ID
        self.sample_relationship.edge_id = self.mock_edge_id
        
        # Act
        edge_id = self.adapter.save_relationship(self.sample_relationship)
        
        # Assert
        assert edge_id == self.mock_edge_id
        self.storage.update_edge.assert_called_once()
        
        # Verify correct data was passed to update_edge
        args_id = self.storage.update_edge.call_args[0][0]
        args_metadata = self.storage.update_edge.call_args[0][1]
        
        assert args_id == self.mock_edge_id
        assert args_metadata['timestamp'] == self.sample_relationship.timestamp
        assert args_metadata['confidence_score'] == self.sample_relationship.confidence_score
        assert args_metadata['version'] == self.sample_relationship.version

    def test_get_relationship(self):
        """Test retrieving a Relationship."""
        # Setup the mock to return edge data
        self.storage.get_edge.return_value = {
            'from_id': self.sample_relationship.from_id,
            'to_id': self.sample_relationship.to_id,
            'relation_type': self.sample_relationship.relation_type,
            'timestamp': self.sample_relationship.timestamp,
            'confidence_score': self.sample_relationship.confidence_score,
            'version': self.sample_relationship.version,
            'id': self.mock_edge_id
        }
        
        # Act
        relationship = self.adapter.get_relationship(self.mock_edge_id)
        
        # Assert
        self.storage.get_edge.assert_called_once_with(self.mock_edge_id)
        assert relationship.edge_id == self.mock_edge_id
        assert relationship.from_id == self.sample_relationship.from_id
        assert relationship.to_id == self.sample_relationship.to_id
        assert relationship.relation_type == self.sample_relationship.relation_type
        assert relationship.timestamp == self.sample_relationship.timestamp
        assert relationship.confidence_score == self.sample_relationship.confidence_score
        assert relationship.version == self.sample_relationship.version

    def test_delete_relationship(self):
        """Test deleting a Relationship."""
        # Act
        self.adapter.delete_relationship(self.mock_edge_id)
        
        # Assert
        self.storage.delete_edge.assert_called_once_with(self.mock_edge_id)


@pytest.mark.integration
class TestGraphStorageAdapterIntegration:
    """
    Integration tests for GraphStorageAdapter.
    
    These tests verify the interaction between the adapter and JanusGraph.
    They are marked as integration tests and will be skipped by default.
    """
    
    @pytest.fixture(scope="class")
    def storage(self):
        """Create and connect to a JanusGraph instance."""
        host = "localhost"
        port = 8182
        
        storage = JanusGraphStorage(host, port)
        try:
            storage.connect()
            yield storage
        except ConnectionError:
            pytest.skip("JanusGraph is not available")
        finally:
            if storage:
                storage.close()
    
    @pytest.fixture
    def adapter(self, storage):
        """Create a GraphStorageAdapter with a connected storage."""
        return GraphStorageAdapter(storage)
    
    @pytest.mark.skipif(
        True,  # Skip by default
        reason="Integration tests require a running JanusGraph instance"
    )
    def test_save_and_retrieve_node(self, adapter):
        """Test saving and retrieving a KnowledgeNode in JanusGraph."""
        # Create a test node
        test_node = KnowledgeNode(
            content="Integration test content",
            source="Integration test",
            creation_timestamp=time.time(),
            rating_richness=0.8,
            rating_truthfulness=0.9,
            rating_stability=0.7
        )
        
        # Save the node
        node_id = adapter.save_knowledge_node(test_node)
        assert node_id is not None
        
        # Retrieve the node
        retrieved_node = adapter.get_knowledge_node(node_id)
        assert retrieved_node is not None
        assert retrieved_node.content == test_node.content
        assert retrieved_node.source == test_node.source
        
        # Clean up
        adapter.delete_knowledge_node(node_id)
    
    @pytest.mark.skipif(
        True,  # Skip by default
        reason="Integration tests require a running JanusGraph instance"
    )
    def test_save_and_retrieve_relationship(self, adapter):
        """Test saving and retrieving a Relationship in JanusGraph."""
        # Create two test nodes
        node1 = KnowledgeNode(
            content="Node 1 content",
            source="Integration test",
            creation_timestamp=time.time()
        )
        
        node2 = KnowledgeNode(
            content="Node 2 content",
            source="Integration test",
            creation_timestamp=time.time()
        )
        
        # Save the nodes
        node1_id = adapter.save_knowledge_node(node1)
        node2_id = adapter.save_knowledge_node(node2)
        
        # Create a relationship
        test_relationship = Relationship(
            from_id=node1_id,
            to_id=node2_id,
            relation_type="is_a",
            timestamp=time.time(),
            confidence_score=0.85,
            version=1
        )
        
        # Save the relationship
        edge_id = adapter.save_relationship(test_relationship)
        assert edge_id is not None
        
        # Retrieve the relationship
        retrieved_relationship = adapter.get_relationship(edge_id)
        assert retrieved_relationship is not None
        assert retrieved_relationship.from_id == node1_id
        assert retrieved_relationship.to_id == node2_id
        assert retrieved_relationship.relation_type == "is_a"
        
        # Clean up
        adapter.delete_relationship(edge_id)
        adapter.delete_knowledge_node(node1_id)
        adapter.delete_knowledge_node(node2_id)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 