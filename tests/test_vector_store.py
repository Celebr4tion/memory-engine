"""
Tests for the VectorStoreMilvus class.

This module tests the vector storage implementation using Milvus.
"""
import time
import random
import unittest
from unittest.mock import patch, MagicMock, call

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
        self.dimension = 3072  # Dimension for Gemini embeddings
        self.host = "localhost"
        self.port = 19530
        self.collection_name = "test_collection"
        
        # Create sample embeddings (normalized vectors)
        self.node_ids = ["node1", "node2", "node3"]
        self.embeddings = [self._create_random_embedding() for _ in range(len(self.node_ids))]
    
    def tearDown(self):
        """Clean up after each test method."""
        self.connections_patcher.stop()
        self.utility_patcher.stop()
        self.collection_patcher.stop()
    
    def _create_random_embedding(self):
        """Create a random embedding vector of the correct dimension."""
        # Create a random vector
        vector = np.random.rand(self.dimension)
        # Normalize it for cosine similarity
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
        # Initialize vector store with patched MILVUS_AVAILABLE
        with patch('memory_core.embeddings.vector_store.MILVUS_AVAILABLE', True):
            vector_store = VectorStoreMilvus(
                host=self.host,
                port=self.port,
                collection_name=self.collection_name,
                dimension=self.dimension
            )
            
        # Assert
        self.assertEqual(vector_store.host, self.host)
        self.assertEqual(vector_store.port, self.port)
        self.assertEqual(vector_store.collection_name, self.collection_name)
        self.assertEqual(vector_store.dimension, self.dimension)
    
    def test_connect_creates_collection(self):
        """Test that connect creates a collection if it doesn't exist."""
        # Initialize vector store and connect
        with patch('memory_core.embeddings.vector_store.MILVUS_AVAILABLE', True):
            vector_store = VectorStoreMilvus(
                host=self.host,
                port=self.port,
                collection_name=self.collection_name,
                dimension=self.dimension
            )
            vector_store.connect()
            
        # Assert
        self.mock_utility.has_collection.assert_called_with(self.collection_name)
        self.mock_collection_class.assert_called_once()
        self.mock_collection.create_index.assert_called_once()
        self.mock_collection.load.assert_called_once()
    
    def test_add_embedding(self):
        """Test adding an embedding to the vector store."""
        # Setup - create collection
        with patch('memory_core.embeddings.vector_store.MILVUS_AVAILABLE', True):
            vector_store = VectorStoreMilvus(
                host=self.host,
                port=self.port,
                collection_name=self.collection_name,
                dimension=self.dimension
            )
            vector_store.connect()
            
        # Reset mocks for clean test
        self.mock_collection.reset_mock()
        self.mock_collection.query.return_value = []  # Node doesn't exist
        
        # Setup test data
        node_id = "test_node"
        embedding = self._create_random_embedding()
        
        # Act
        vector_store.add_embedding(node_id, embedding)
        
        # Assert
        self.mock_collection.query.assert_called_once()
        self.mock_collection.insert.assert_called_once()
        self.mock_collection.flush.assert_called_once()
    
    def test_add_embedding_wrong_dimension(self):
        """Test that adding an embedding with wrong dimension raises an error."""
        # Setup - create collection
        with patch('memory_core.embeddings.vector_store.MILVUS_AVAILABLE', True):
            vector_store = VectorStoreMilvus(
                host=self.host,
                port=self.port,
                collection_name=self.collection_name,
                dimension=self.dimension
            )
            vector_store.connect()
            
        # Setup test data
        node_id = "test_node"
        embedding = self._create_random_embedding()[:-10]  # Wrong dimension
        
        # Act & Assert
        with self.assertRaises(ValueError):
            vector_store.add_embedding(node_id, embedding)
    
    def test_search_embedding(self):
        """Test searching for similar embeddings."""
        # Setup - create collection
        with patch('memory_core.embeddings.vector_store.MILVUS_AVAILABLE', True):
            vector_store = VectorStoreMilvus(
                host=self.host,
                port=self.port,
                collection_name=self.collection_name,
                dimension=self.dimension
            )
            vector_store.connect()
            
        # Reset mocks for clean test
        self.mock_collection.reset_mock()
        
        # Setup test data
        query_vector = self._create_random_embedding()
        expected_results = [
            {"node_id": "node1", "score": 0.95},
            {"node_id": "node2", "score": 0.85},
            {"node_id": "node3", "score": 0.75}
        ]
        self._setup_search_results(expected_results)
        
        # Act
        results = vector_store.search_embedding(query_vector, top_k=3)
        
        # Assert
        self.mock_collection.search.assert_called_once()
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["node_id"], "node1")
        self.assertEqual(results[0]["score"], 0.95)
    
    def test_get_node_ids(self):
        """Test getting only the node IDs from search results."""
        # Setup - create collection
        with patch('memory_core.embeddings.vector_store.MILVUS_AVAILABLE', True):
            vector_store = VectorStoreMilvus(
                host=self.host,
                port=self.port,
                collection_name=self.collection_name,
                dimension=self.dimension
            )
            vector_store.connect()
            
        # Reset mocks for clean test
        self.mock_collection.reset_mock()
        
        # Setup test data
        query_vector = self._create_random_embedding()
        expected_results = [
            {"node_id": "node1", "score": 0.95},
            {"node_id": "node2", "score": 0.85},
            {"node_id": "node3", "score": 0.75}
        ]
        self._setup_search_results(expected_results)
        
        # Act
        node_ids = vector_store.get_node_ids(query_vector, top_k=3)
        
        # Assert
        self.assertEqual(node_ids, ["node1", "node2", "node3"])
    
    def test_search_empty_results(self):
        """Test searching with no results."""
        # Setup - create collection
        with patch('memory_core.embeddings.vector_store.MILVUS_AVAILABLE', True):
            vector_store = VectorStoreMilvus(
                host=self.host,
                port=self.port,
                collection_name=self.collection_name,
                dimension=self.dimension
            )
            vector_store.connect()
            
        # Reset mocks for clean test
        self.mock_collection.reset_mock()
        
        # Setup test data
        query_vector = self._create_random_embedding()
        self.mock_collection.search.return_value = [[]]  # Empty results
        
        # Act
        results = vector_store.search_embedding(query_vector)
        
        # Assert
        self.assertEqual(len(results), 0)
    
    def test_add_embedding_and_search(self):
        """Test adding and then searching for embeddings in an integrated manner."""
        # Setup - create collection
        with patch('memory_core.embeddings.vector_store.MILVUS_AVAILABLE', True):
            vector_store = VectorStoreMilvus(
                host=self.host,
                port=self.port,
                collection_name=self.collection_name,
                dimension=self.dimension
            )
            vector_store.connect()
            
        # Reset mocks for clean test
        self.mock_collection.reset_mock()
        
        # Setup test data
        node_id = "test_node"
        embedding = self._create_random_embedding()
        self.mock_collection.query.return_value = []  # Node doesn't exist
        
        # Add the embedding
        vector_store.add_embedding(node_id, embedding)
        
        # Setup search results
        expected_results = [
            {"node_id": node_id, "score": 0.99}
        ]
        self._setup_search_results(expected_results)
        
        # Act - Search for the embedding
        results = vector_store.search_embedding(embedding, top_k=1)
        
        # Assert
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["node_id"], node_id)
    
    def test_search_empty_db(self):
        """Test searching in an empty database."""
        # Setup - create collection
        with patch('memory_core.embeddings.vector_store.MILVUS_AVAILABLE', True):
            vector_store = VectorStoreMilvus(
                host=self.host,
                port=self.port,
                collection_name=self.collection_name,
                dimension=self.dimension
            )
            vector_store.connect()
            
        # Reset mocks for clean test
        self.mock_collection.reset_mock()
        
        # Setup test data
        query_vector = self._create_random_embedding()
        self.mock_collection.search.return_value = [[]]  # Empty results
        
        # Act
        results = vector_store.search_embedding(query_vector)
        node_ids = vector_store.get_node_ids(query_vector)
        
        # Assert
        self.assertEqual(len(results), 0)
        self.assertEqual(len(node_ids), 0)
    
    def test_delete_embedding(self):
        """Test deleting an embedding."""
        # Setup - create collection
        with patch('memory_core.embeddings.vector_store.MILVUS_AVAILABLE', True):
            vector_store = VectorStoreMilvus(
                host=self.host,
                port=self.port,
                collection_name=self.collection_name,
                dimension=self.dimension
            )
            vector_store.connect()
            
        # Reset mocks for clean test
        self.mock_collection.reset_mock()
        
        # Setup test data
        node_id = "test_node"
        
        # Act
        result = vector_store.delete_embedding(node_id)
        
        # Assert
        self.mock_collection.delete.assert_called_once_with(f'node_id == "{node_id}"')
        self.mock_collection.flush.assert_called_once()
        self.assertTrue(result)


@pytest.mark.integration
class TestVectorStoreMilvusIntegration:
    """Integration tests for VectorStoreMilvus with a real Milvus instance."""
    
    def setup_method(self):
        """Set up the test with actual connection to Milvus."""
        import uuid
        
        # Generate a unique collection name for this test run
        self.collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"
        self.dimension = 384  # Use a smaller dimension for faster tests
        
        # Create the vector store
        self.vector_store = VectorStoreMilvus(
            host="localhost",
            port=19530,
            collection_name=self.collection_name,
            dimension=self.dimension
        )
        
        # Check if we can connect to Milvus - use increased retries and interval
        if not self.vector_store.connect(max_retries=3, retry_interval=5):
            pytest.skip("Could not connect to Milvus. Make sure Docker container is running.")
        
        # Add additional delay after connection to ensure everything is properly initialized
        print("Connection successful, adding stabilization delay...")
        time.sleep(2)
        print(f"Collection {self.collection_name} ready for testing")
    
    def teardown_method(self):
        """Clean up after test."""
        if hasattr(self, 'vector_store') and self.vector_store.collection:
            try:
                # Drop the collection
                from pymilvus import utility
                if utility.has_collection(self.collection_name):
                    utility.drop_collection(self.collection_name)
                    print(f"Collection {self.collection_name} dropped successfully")
            except Exception as e:
                print(f"Error cleaning up: {e}")
            
            # Disconnect
            self.vector_store.disconnect()
            print("Disconnected from Milvus")
    
    def _create_random_embedding(self):
        """Create a random embedding vector of the correct dimension."""
        # Create a random vector
        vector = np.random.rand(self.dimension)
        # Normalize it
        vector = vector / np.linalg.norm(vector)
        return vector.tolist()
    
    def test_live_add_and_search(self):
        """Test adding and searching with actual Milvus instance."""
        import uuid
        import threading
        
        # Create a test node with embedding
        node_id = f"test_node_{uuid.uuid4().hex[:8]}"
        embedding = self._create_random_embedding()
        
        print(f"Adding embedding for node {node_id}...")
        self.vector_store.add_embedding(node_id, embedding)
        print("Embedding added successfully")
        
        # Add a small delay to ensure data is searchable
        time.sleep(1)
        
        print("Searching for matching embedding...")
        
        # Use a flag to track if the test completes in time
        test_completed = False
        results = None
        
        def search_with_timeout():
            nonlocal test_completed, results
            # Search for the embedding
            try:
                results = self.vector_store.search_embedding(embedding, top_k=1)
                test_completed = True
            except Exception as e:
                print(f"Search failed with error: {e}")
        
        # Create a thread to run the search
        search_thread = threading.Thread(target=search_with_timeout)
        search_thread.daemon = True
        search_thread.start()
        
        # Wait for the thread to complete or timeout
        search_thread.join(30)  # 30 second timeout
        
        if not test_completed:
            pytest.fail("Search operation timed out after 30 seconds")
        
        print(f"Search returned {len(results)} results")
        
        # Verify results
        assert len(results) == 1, f"Expected 1 result, got {len(results)}"
        assert results[0]["node_id"] == node_id, f"Expected {node_id}, got {results[0]['node_id']}"
        # For cosine similarity, the score should be close to 1.0 for identical vectors
        assert results[0]["score"] > 0.99, f"Expected score > 0.99, got {results[0]['score']}"
        print("Test completed successfully")
    
    def test_live_empty_search(self):
        """Test searching when no embeddings exist."""
        import threading
        
        # Create an empty collection (no embeddings added)
        # then search for a random vector
        query_vector = self._create_random_embedding()
        
        print("Searching empty collection...")
        
        # Use a flag to track if the test completes in time
        test_completed = False
        results = None
        
        def search_with_timeout():
            nonlocal test_completed, results
            # Search in empty collection
            try:
                results = self.vector_store.search_embedding(query_vector)
                test_completed = True
            except Exception as e:
                print(f"Search failed with error: {e}")
        
        # Create a thread to run the search
        search_thread = threading.Thread(target=search_with_timeout)
        search_thread.daemon = True
        search_thread.start()
        
        # Wait for the thread to complete or timeout
        search_thread.join(10)  # 10 second timeout
        
        if not test_completed:
            pytest.fail("Empty search operation timed out after 10 seconds")
        
        print(f"Search returned {len(results)} results as expected")
        
        # Verify empty results
        assert len(results) == 0, f"Expected empty results, got {len(results)} results" 