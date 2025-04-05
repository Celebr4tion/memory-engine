"""
Tests for the embedding manager module.

This module tests the functions for generating and storing embeddings.
"""
import unittest
from unittest.mock import patch, MagicMock, Mock
import numpy as np

from memory_core.embeddings.embedding_manager import (
    get_model,
    generate_embedding,
    store_node_embedding,
    search_similar_nodes
)


class TestEmbeddingManager(unittest.TestCase):
    """Test cases for the embedding manager functions."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock the sentence transformer model
        self.model_patcher = patch('memory_core.embeddings.embedding_manager.SentenceTransformer')
        self.mock_model_class = self.model_patcher.start()
        self.mock_model = MagicMock()
        self.mock_model_class.return_value = self.mock_model
        
        # Set up mock embedding
        self.test_dimension = 384  # Common for MiniLM models
        self.mock_embedding = np.random.rand(self.test_dimension).astype(np.float32)
        self.mock_model.encode.return_value = self.mock_embedding
        
        # Mock the vector store
        self.vector_store_patcher = patch('memory_core.embeddings.embedding_manager.VectorStoreMilvus')
        self.mock_vector_store_class = self.vector_store_patcher.start()
        self.mock_vector_store = MagicMock()
        self.mock_vector_store_class.return_value = self.mock_vector_store
        self.mock_vector_store.connect.return_value = True
        self.mock_vector_store.collection = MagicMock()  # Set collection attribute
        
        # Sample test data
        self.test_node_id = "test_node_123"
        self.test_text = "This is a test text for embedding generation."
        self.similar_node_ids = ["node1", "node2", "node3"]
        self.mock_vector_store.get_node_ids.return_value = self.similar_node_ids

    def tearDown(self):
        """Clean up after each test method."""
        # Stop patchers
        self.model_patcher.stop()
        self.vector_store_patcher.stop()
        
        # Reset singleton instance
        import memory_core.embeddings.embedding_manager as em
        em._model_instance = None

    def test_get_model(self):
        """Test getting the embedding model."""
        # Call get_model
        model = get_model()
        
        # Verify model initialization
        self.mock_model_class.assert_called_once()
        self.assertEqual(model, self.mock_model)
        
        # Call get_model again, should reuse instance
        model2 = get_model()
        self.assertEqual(model, model2)
        self.mock_model_class.assert_called_once()  # Still only one call

    def test_generate_embedding(self):
        """Test generating an embedding from text."""
        # Call generate_embedding
        embedding = generate_embedding(self.test_text)
        
        # Verify the embedding
        self.mock_model.encode.assert_called_once_with(self.test_text)
        self.assertEqual(embedding, self.mock_embedding.tolist())
        self.assertEqual(len(embedding), self.test_dimension)

    def test_generate_embedding_error_handling(self):
        """Test error handling in generate_embedding."""
        # Set up the mock to raise an exception
        self.mock_model.encode.side_effect = Exception("Test error")
        
        # Call generate_embedding and check for exception
        with self.assertRaises(Exception):
            generate_embedding(self.test_text)

    def test_store_node_embedding(self):
        """Test storing a node embedding."""
        # Create a mock for graph storage
        mock_graph_storage = MagicMock()
        
        # Call store_node_embedding
        store_node_embedding(
            node_id=self.test_node_id,
            text=self.test_text,
            vector_store=self.mock_vector_store,
            graph_storage=mock_graph_storage
        )
        
        # Verify embedding generation and storage
        self.mock_model.encode.assert_called_once()
        self.mock_vector_store.add_embedding.assert_called_once_with(
            self.test_node_id, 
            self.mock_embedding.tolist()
        )
        
        # Verify graph storage update
        mock_graph_storage.update_node.assert_called_once_with(
            self.test_node_id, 
            {"embedding_exists": True}
        )

    def test_store_node_embedding_no_vector_store(self):
        """Test storing a node embedding with automatic vector store creation."""
        # Call store_node_embedding without providing a vector store
        with patch('memory_core.embeddings.embedding_manager.os.environ.get') as mock_env:
            mock_env.side_effect = lambda k, default: default  # Return the default value
            
            store_node_embedding(
                node_id=self.test_node_id,
                text=self.test_text
            )
        
        # Verify vector store creation and connection
        self.mock_vector_store_class.assert_called_once_with(host="localhost", port=19530)
        self.mock_vector_store.connect.assert_called_once()
        
        # Verify embedding generation and storage
        self.mock_model.encode.assert_called_once()
        self.mock_vector_store.add_embedding.assert_called_once_with(
            self.test_node_id, 
            self.mock_embedding.tolist()
        )

    def test_store_node_embedding_connection_error(self):
        """Test error handling when vector store connection fails."""
        # Use a mock vector store that needs connection but fails to connect
        mock_vector_store = MagicMock()
        mock_vector_store.connect.return_value = False
        mock_vector_store.collection = None  # This will trigger connection_required=True
        
        # Call store_node_embedding which should try to connect and fail
        with self.assertRaises(ConnectionError):
            store_node_embedding(
                node_id=self.test_node_id,
                text=self.test_text,
                vector_store=mock_vector_store
            )
        
        # Verify connect was called
        mock_vector_store.connect.assert_called_once()

    def test_search_similar_nodes(self):
        """Test searching for similar nodes."""
        # Call search_similar_nodes
        result = search_similar_nodes(
            query_text=self.test_text,
            vector_store=self.mock_vector_store
        )
        
        # Verify embedding generation and search
        self.mock_model.encode.assert_called_once()
        self.mock_vector_store.get_node_ids.assert_called_once_with(
            self.mock_embedding.tolist(), 
            5  # Default top_k
        )
        
        # Verify the result
        self.assertEqual(result, self.similar_node_ids)

    def test_search_similar_nodes_custom_top_k(self):
        """Test searching for similar nodes with custom top_k."""
        # Call search_similar_nodes with custom top_k
        result = search_similar_nodes(
            query_text=self.test_text,
            top_k=10,
            vector_store=self.mock_vector_store
        )
        
        # Verify search with custom top_k
        self.mock_vector_store.get_node_ids.assert_called_once_with(
            self.mock_embedding.tolist(), 
            10
        )


class TestEmbeddingManagerIntegration(unittest.TestCase):
    """Integration tests using mocked external dependencies but testing the full flow."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock model with real embedding behavior
        self.model_patcher = patch('memory_core.embeddings.embedding_manager.SentenceTransformer')
        self.mock_model_class = self.model_patcher.start()
        self.mock_model = MagicMock()
        self.mock_model_class.return_value = self.mock_model
        
        # Define feature embeddings for key concepts
        self.features = {
            "fox": np.array([1.0, 0.1, 0.0, 0.0, 0.0]),
            "dog": np.array([0.0, 0.1, 1.0, 0.0, 0.0]),
            "jump": np.array([0.1, 1.0, 0.1, 0.0, 0.0]),
            "lazy": np.array([0.0, 0.0, 0.8, 0.0, 0.0]),
            "python": np.array([0.0, 0.0, 0.0, 1.0, 0.2]),
            "java": np.array([0.0, 0.0, 0.0, 0.2, 1.0]),
            "programming": np.array([0.0, 0.0, 0.0, 0.8, 0.8]),
            "language": np.array([0.0, 0.0, 0.0, 0.6, 0.6])
        }
        
        # Define a more semantic-like embedding function
        def mock_encode(text):
            # Convert to lowercase
            text = text.lower()
            
            # Create a base embedding
            base_embedding = np.zeros(5)
            
            # Add feature embeddings for each word that matches a known concept
            for word, feature in self.features.items():
                if word in text:
                    base_embedding += feature
            
            # If empty, add a small random component
            if np.sum(base_embedding) == 0:
                base_embedding = np.random.rand(5) * 0.01
                
            # Normalize
            norm = np.linalg.norm(base_embedding)
            if norm > 0:
                base_embedding = base_embedding / norm
                
            return base_embedding
        
        self.mock_model.encode.side_effect = mock_encode
        
        # Create a more sophisticated mock for the vector store
        self.vector_store = MagicMock()
        self.vector_store.connect.return_value = True
        self.vector_store.collection = MagicMock()  # Simulate connected state
        
        # Storage for our simulated embeddings database
        self.stored_embeddings = {}
        
        # Mock add_embedding to store embeddings
        def mock_add_embedding(node_id, embedding):
            self.stored_embeddings[node_id] = embedding
        
        self.vector_store.add_embedding.side_effect = mock_add_embedding
        
        # Mock get_node_ids to return similar embeddings
        def mock_get_node_ids(query_embedding, top_k):
            # Calculate cosine similarity
            similarities = {}
            for node_id, embedding in self.stored_embeddings.items():
                # Ensure same length by padding if necessary
                min_length = min(len(query_embedding), len(embedding))
                query_vec = np.array(query_embedding[:min_length])
                embed_vec = np.array(embedding[:min_length])
                
                # Compute cosine similarity
                query_norm = np.linalg.norm(query_vec)
                embed_norm = np.linalg.norm(embed_vec)
                
                if query_norm > 0 and embed_norm > 0:
                    similarity = np.dot(query_vec, embed_vec) / (query_norm * embed_norm)
                else:
                    similarity = 0.0
                    
                similarities[node_id] = similarity
            
            # Sort by similarity and take top_k
            sorted_nodes = sorted(
                similarities.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:top_k]
            
            return [node_id for node_id, _ in sorted_nodes]
        
        self.vector_store.get_node_ids.side_effect = mock_get_node_ids
        
        # Test data
        self.test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "A fox quickly jumps over a lazy dog.",
            "The five boxing wizards jump quickly.",
            "Python is a programming language.",
            "Java is also a programming language."
        ]
        self.test_node_ids = [f"node_{i}" for i in range(len(self.test_texts))]

    def tearDown(self):
        """Clean up after each test method."""
        self.model_patcher.stop()
        
        # Reset singleton instance
        import memory_core.embeddings.embedding_manager as em
        em._model_instance = None

    def test_end_to_end_flow(self):
        """Test the full embedding flow from storage to retrieval."""
        # Store embeddings for all test texts
        for node_id, text in zip(self.test_node_ids, self.test_texts):
            store_node_embedding(
                node_id=node_id,
                text=text,
                vector_store=self.vector_store
            )
        
        # Verify all embeddings were stored
        self.assertEqual(len(self.stored_embeddings), len(self.test_texts))
        
        # Test searching with a similar query to the first text
        similar_query = "A quick brown fox jumps over a dog that is lazy."
        similar_nodes = search_similar_nodes(
            query_text=similar_query,
            vector_store=self.vector_store,
            top_k=2
        )
        
        # The first and second texts should be most similar to the query
        self.assertTrue(
            "node_0" in similar_nodes or "node_1" in similar_nodes,
            f"Expected similar texts to be found, but got {similar_nodes}"
        )
        
        # Test searching with a programming-related query
        programming_query = "Programming in Python language"
        programming_nodes = search_similar_nodes(
            query_text=programming_query,
            vector_store=self.vector_store,
            top_k=2
        )
        
        # The fourth and fifth texts should be most similar to this query
        self.assertTrue(
            "node_3" in programming_nodes or "node_4" in programming_nodes,
            f"Expected programming-related texts to be found, but got {programming_nodes}"
        ) 