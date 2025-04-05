"""
Tests for the VectorStoreMilvus class.

This module tests the vector storage implementation using Milvus.
"""
import time
import random
import unittest
import uuid
from unittest.mock import patch, MagicMock

import pytest
import numpy as np

# We'll patch the MILVUS_AVAILABLE constant to True for testing
with patch('memory_core.embeddings.vector_store.MILVUS_AVAILABLE', True):
    from memory_core.embeddings.vector_store import VectorStoreMilvus


class TestVectorStoreMilvus(unittest.TestCase):
    """Test cases for the VectorStoreMilvus class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock the Milvus-related modules and classes
        self.connections_patcher = patch('memory_core.embeddings.vector_store.connections')
        self.mock_connections = self.connections_patcher.start()
        
        self.utility_patcher = patch('memory_core.embeddings.vector_store.utility')
        self.mock_utility = self.utility_patcher.start()
        self.mock_utility.has_collection.return_value = False
        
        self.collection_patcher = patch('memory_core.embeddings.vector_store.Collection')
        self.mock_collection_class = self.collection_patcher.start()
        self.mock_collection = MagicMock()
        self.mock_collection_class.return_value = self.mock_collection
        
        # Setup sample data
        self.dimension = 256
        self.host = "localhost"
        self.port = 19530
        self.collection_name = "test_collection"
        
        # Create sample embeddings (normalized vectors)
        self.node_ids = ["node1", "node2", "node3"]
        self.embeddings = [self._create_random_embedding() for _ in range(len(self.node_ids))]
        
        # Initialize vector store with patched MILVUS_AVAILABLE
        with patch('memory_core.embeddings.vector_store.MILVUS_AVAILABLE', True):
            self.vector_store = VectorStoreMilvus(
                host=self.host,
                port=self.port,
                collection_name=self.collection_name,
                dimension=self.dimension
            )
            self.vector_store.connect()
    
    def tearDown(self):
        """Clean up after each test method."""
        self.connections_patcher.stop()
        self.utility_patcher.stop()
        self.collection_patcher.stop()
    
    def _create_random_embedding(self):
        """Create a random embedding vector of the correct dimension."""
        # Create a random vector
        vector = np.random.rand(self.dimension)
        # Normalize it
        vector = vector / np.linalg.norm(vector)
        return vector.tolist()
    
    def _setup_search_results(self, query_results):
        """Setup mock search results."""
        # Create mock search results
        mock_hits = []
        for result in query_results:
            mock_hit = MagicMock()
            mock_hit.entity = {"node_id": result["node_id"]}
            mock_hit.score = result["score"]
            mock_hits.append(mock_hit)
        
        # Wrap in the expected structure (list of lists)
        mock_results = [mock_hits]
        self.mock_collection.search.return_value = mock_results
    
    def test_initialization(self):
        """Test initializing the VectorStoreMilvus class."""
        # Assert
        self.assertEqual(self.vector_store.host, self.host)
        self.assertEqual(self.vector_store.port, self.port)
        self.assertEqual(self.vector_store.collection_name, self.collection_name)
        self.assertEqual(self.vector_store.dimension, self.dimension)
        
        # Verify connect was called - check with any timeout parameter since it might vary
        connect_call = self.mock_connections.connect.call_args
        self.assertEqual(connect_call[0][0], "default")
        self.assertEqual(connect_call[1]["host"], self.host)
        self.assertEqual(connect_call[1]["port"], self.port)
        # We don't check for the exact timeout value as it could change
    
    def test_connect_creates_collection(self):
        """Test that connect creates a collection if it doesn't exist."""
        # Assert
        self.mock_utility.has_collection.assert_called_with(self.collection_name)
        self.mock_collection_class.assert_called_once()
        self.mock_collection.create_index.assert_called_once()
        self.mock_collection.load.assert_called_once()
    
    def test_add_embedding(self):
        """Test adding an embedding to the vector store."""
        # Setup
        node_id = "test_node"
        embedding = self._create_random_embedding()
        self.mock_collection.query.return_value = []  # Node doesn't exist
        
        # Act
        self.vector_store.add_embedding(node_id, embedding)
        
        # Assert
        self.mock_collection.query.assert_called_once()
        self.mock_collection.insert.assert_called_once()
        self.mock_collection.flush.assert_called_once()
    
    def test_add_embedding_replaces_existing(self):
        """Test that adding an embedding replaces an existing one with the same ID."""
        # Setup
        node_id = "test_node"
        embedding = self._create_random_embedding()
        self.mock_collection.query.return_value = [{"node_id": node_id}]  # Node exists
        
        # Act
        self.vector_store.add_embedding(node_id, embedding)
        
        # Assert
        self.mock_collection.query.assert_called_once()
        self.mock_collection.delete.assert_called_once()
        self.mock_collection.insert.assert_called_once()
    
    def test_add_embedding_wrong_dimension(self):
        """Test that adding an embedding with wrong dimension raises an error."""
        # Setup
        node_id = "test_node"
        embedding = self._create_random_embedding()[:-10]  # Wrong dimension
        
        # Act & Assert
        with self.assertRaises(ValueError):
            self.vector_store.add_embedding(node_id, embedding)
    
    def test_add_embedding_not_connected(self):
        """Test that adding an embedding when not connected raises an error."""
        # Setup
        with patch('memory_core.embeddings.vector_store.MILVUS_AVAILABLE', True):
            vector_store = VectorStoreMilvus(host=self.host, port=self.port)
        node_id = "test_node"
        embedding = self._create_random_embedding()
        
        # Act & Assert
        with self.assertRaises(RuntimeError):
            vector_store.add_embedding(node_id, embedding)
    
    def test_search_embedding(self):
        """Test searching for similar embeddings."""
        # Setup
        query_vector = self._create_random_embedding()
        expected_results = [
            {"node_id": "node1", "score": 0.95},
            {"node_id": "node2", "score": 0.85},
            {"node_id": "node3", "score": 0.75}
        ]
        self._setup_search_results(expected_results)
        
        # Act
        results = self.vector_store.search_embedding(query_vector, top_k=3)
        
        # Assert
        self.mock_collection.search.assert_called_once()
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["node_id"], "node1")
        self.assertEqual(results[0]["score"], 0.95)
    
    def test_get_node_ids(self):
        """Test getting only the node IDs from search results."""
        # Setup
        query_vector = self._create_random_embedding()
        expected_results = [
            {"node_id": "node1", "score": 0.95},
            {"node_id": "node2", "score": 0.85},
            {"node_id": "node3", "score": 0.75}
        ]
        self._setup_search_results(expected_results)
        
        # Act
        node_ids = self.vector_store.get_node_ids(query_vector, top_k=3)
        
        # Assert
        self.assertEqual(node_ids, ["node1", "node2", "node3"])
    
    def test_search_embedding_wrong_dimension(self):
        """Test that searching with wrong dimension raises an error."""
        # Setup
        query_vector = self._create_random_embedding()[:-10]  # Wrong dimension
        
        # Act & Assert
        with self.assertRaises(ValueError):
            self.vector_store.search_embedding(query_vector)
    
    def test_search_embedding_not_connected(self):
        """Test that searching when not connected raises an error."""
        # Setup
        with patch('memory_core.embeddings.vector_store.MILVUS_AVAILABLE', True):
            vector_store = VectorStoreMilvus(host=self.host, port=self.port)
        query_vector = self._create_random_embedding()
        
        # Act & Assert
        with self.assertRaises(RuntimeError):
            vector_store.search_embedding(query_vector)
    
    def test_search_empty_results(self):
        """Test searching with no results."""
        # Setup
        query_vector = self._create_random_embedding()
        self.mock_collection.search.return_value = [[]]  # Empty results
        
        # Act
        results = self.vector_store.search_embedding(query_vector)
        
        # Assert
        self.assertEqual(len(results), 0)
    
    def test_add_embedding_and_search(self):
        """Test adding and then searching for embeddings in an integrated manner."""
        # Setup
        node_id = "test_node"
        embedding = self._create_random_embedding()
        
        # Mock query to say node doesn't exist
        self.mock_collection.query.return_value = []
        
        # Add the embedding
        self.vector_store.add_embedding(node_id, embedding)
        
        # Setup search results
        expected_results = [
            {"node_id": node_id, "score": 0.99}
        ]
        self._setup_search_results(expected_results)
        
        # Act - Search for the embedding
        results = self.vector_store.search_embedding(embedding, top_k=1)
        
        # Assert
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["node_id"], node_id)
    
    def test_search_empty_db(self):
        """Test searching in an empty database."""
        # Setup
        query_vector = self._create_random_embedding()
        self.mock_collection.search.return_value = [[]]  # Empty results
        
        # Act
        results = self.vector_store.search_embedding(query_vector)
        node_ids = self.vector_store.get_node_ids(query_vector)
        
        # Assert
        self.assertEqual(len(results), 0)
        self.assertEqual(len(node_ids), 0)
    
    def test_delete_embedding(self):
        """Test deleting an embedding."""
        # Setup
        node_id = "test_node"
        
        # Act
        result = self.vector_store.delete_embedding(node_id)
        
        # Assert
        self.mock_collection.delete.assert_called_once_with(f'node_id == "{node_id}"')
        self.mock_collection.flush.assert_called_once()
        self.assertTrue(result)
    
    def test_disconnect(self):
        """Test disconnecting from Milvus."""
        # Act
        self.vector_store.disconnect()
        
        # Assert
        self.mock_connections.disconnect.assert_called_once_with("default")


@pytest.mark.integration
class TestVectorStoreMilvusIntegration:
    """Integration tests for VectorStoreMilvus with a real Milvus instance."""
    
    def setup_method(self):
        """Set up the test with actual connection to Milvus."""
        import os
        import uuid
        
        # Generate a unique collection name for this test run
        self.collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"
        self.dimension = 256
        
        # Create the vector store
        self.vector_store = VectorStoreMilvus(
            host="localhost",
            port=19530,
            collection_name=self.collection_name,
            dimension=self.dimension
        )
        
        # Check if we can connect to Milvus - skip test if not
        if not self.vector_store.connect():
            pytest.skip("Could not connect to Milvus. Make sure Docker container is running.")
    
    def teardown_method(self):
        """Clean up after test."""
        if hasattr(self, 'vector_store') and self.vector_store.collection:
            try:
                # Drop the collection
                from pymilvus import utility
                if utility.has_collection(self.collection_name):
                    utility.drop_collection(self.collection_name)
            except Exception as e:
                print(f"Error cleaning up: {e}")
            
            # Disconnect
            self.vector_store.disconnect()
    
    def _create_random_embedding(self):
        """Create a random embedding vector of the correct dimension."""
        # Create a random vector
        vector = np.random.rand(self.dimension)
        # Normalize it
        vector = vector / np.linalg.norm(vector)
        return vector.tolist()
    
    def test_live_add_and_search(self):
        """Test adding and searching with actual Milvus instance."""
        # Create a test node with embedding
        node_id = f"test_node_{uuid.uuid4().hex[:8]}"
        embedding = self._create_random_embedding()
        
        # Add the embedding
        self.vector_store.add_embedding(node_id, embedding)
        
        # Search for the embedding
        results = self.vector_store.search_embedding(embedding, top_k=1)
        
        # Verify results
        assert len(results) == 1
        assert results[0]["node_id"] == node_id
        
        # Add more nodes
        additional_nodes = 3
        for i in range(additional_nodes):
            additional_node_id = f"test_node_{uuid.uuid4().hex[:8]}"
            additional_embedding = self._create_random_embedding()
            self.vector_store.add_embedding(additional_node_id, additional_embedding)
        
        # Search with larger top_k
        all_results = self.vector_store.search_embedding(embedding, top_k=5)
        assert len(all_results) >= 1  # Should find at least our original node
    
    def test_live_empty_search(self):
        """Test searching when no embeddings exist."""
        # Create an empty collection (no embeddings added)
        # then search for a random vector
        query_vector = self._create_random_embedding()
        
        # Search
        results = self.vector_store.search_embedding(query_vector)
        
        # Verify empty results
        assert len(results) == 0 