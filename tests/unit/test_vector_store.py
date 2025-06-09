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
        self.mock_collection.delete.assert_called_once_with(f'node_id in ["{node_id}"]')
        self.mock_collection.flush.assert_called_once()
        self.assertTrue(result)


@pytest.mark.integration
class TestVectorStoreMilvusIntegration:
    """Integration tests with a real Milvus instance."""
    
    def setup_method(self, method, milvus_available=None):
        """Set up test fixtures."""
        # Skip if Milvus is not available (checked via fixture or directly)
        is_available = milvus_available
        if is_available is None:
            # If not provided via fixture, check directly
            try:
                from pymilvus import connections
                connections.connect("default", host="localhost", port="19530", timeout=5.0)
                connections.disconnect("default")
                is_available = True
            except Exception:
                is_available = False
        
        if not is_available:
            pytest.skip("Milvus server not available")
            
        # Create a test collection name with timestamp to avoid conflicts
        import time
        self.collection_name = f"test_collection_{int(time.time())}"
        
        # Create the vector store with 3072 dimensions to match gemini-embedding-exp-03-07
        from memory_core.embeddings.vector_store import VectorStoreMilvus
        self.vector_store = VectorStoreMilvus(
            host="localhost", 
            port=19530,
            collection_name=self.collection_name,
            dimension=3072  # Set dimension to 3072 to match gemini-embedding-exp-03-07
        )
        
        # Connect to Milvus
        self.vector_store.connect()
    
    def teardown_method(self, method):
        """Clean up test fixtures."""
        try:
            # Drop the test collection
            if hasattr(self, 'vector_store') and self.vector_store is not None:
                self.vector_store.drop_collection()
                self.vector_store.disconnect()
        except Exception:
            pass
    
    def test_live_add_and_search(self, milvus_available):
        """Test adding and searching embeddings with a real Milvus instance."""
        self.setup_method(None, milvus_available)
        
        # Create test embeddings with 3072 dimensions
        test_embedding1 = [0.1] * 3072  # 3072-dimensional vector
        test_embedding2 = [0.2] * 3072  # 3072-dimensional vector
        
        # Add embeddings to the vector store
        self.vector_store.add_embedding("node1", test_embedding1)
        self.vector_store.add_embedding("node2", test_embedding2)
        
        # Search for embeddings
        results = self.vector_store.search_embedding(test_embedding1, top_k=2)
        
        # Verify search results
        assert len(results) > 0
        assert results[0]["node_id"] == "node1"
    
    def test_live_empty_search(self, milvus_available):
        """Test searching an empty collection with a real Milvus instance."""
        self.setup_method(None, milvus_available)
        
        # Search for embeddings in an empty collection with 3072 dimensions
        test_embedding = [0.1] * 3072  # 3072-dimensional vector
        results = self.vector_store.search_embedding(test_embedding, top_k=2)
        
        # Verify search results
        assert results == []