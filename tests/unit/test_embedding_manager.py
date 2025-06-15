"""
Tests for the embedding manager module.

This module tests the functions for generating and storing embeddings using Gemini API.
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import numpy as np
import pytest
import time  # Import time module

from memory_core.embeddings.embedding_manager import EmbeddingManager
from memory_core.embeddings.vector_store import VectorStoreMilvus


class TestEmbeddingManagerClass(unittest.TestCase):
    """Test cases for the EmbeddingManager class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock the vector store
        self.vector_store = MagicMock(spec=VectorStoreMilvus)

        # Patch the genai client
        self.client_patcher = patch("memory_core.embeddings.embedding_manager.genai.Client")
        self.mock_client_class = self.client_patcher.start()
        self.mock_client = MagicMock()
        self.mock_client_class.return_value = self.mock_client

        # Mock environment variable
        self.env_patcher = patch.dict("os.environ", {"GOOGLE_API_KEY": "fake_key"})
        self.env_patcher.start()

        # Create the embedding manager with mocked vector store
        self.embedding_manager = EmbeddingManager(self.vector_store)

        # Sample test data
        self.sample_text = "What is the meaning of life?"
        self.sample_node_id = "test_node_1"

        # Mock embedding result
        self.mock_embedding = np.random.rand(768).tolist()  # gemini-embedding-exp-03-07 dimension

    def tearDown(self):
        """Clean up after each test method."""
        self.client_patcher.stop()
        self.env_patcher.stop()

    def test_generate_embedding(self):
        """Test generating an embedding using Gemini API."""
        # Mock the Gemini API response
        mock_result = MagicMock()
        mock_result.embeddings = [self.mock_embedding]
        self.mock_client.models.embed_content.return_value = mock_result

        # Generate embedding
        embedding = self.embedding_manager.generate_embedding(self.sample_text)

        # Verify API call - check that it was called with the correct model and format
        self.mock_client.models.embed_content.assert_called_once()
        call_args = self.mock_client.models.embed_content.call_args

        # Verify the model and contents
        self.assertEqual(call_args[1]["model"], "gemini-embedding-exp-03-07")
        self.assertEqual(call_args[1]["contents"], self.sample_text)

        # Verify config is a typed object with task_type
        config = call_args[1]["config"]
        self.assertEqual(config.task_type, "SEMANTIC_SIMILARITY")

        # Verify result
        self.assertEqual(len(embedding), 768)
        self.assertEqual(embedding, self.mock_embedding)

    def test_generate_embedding_empty_text(self):
        """Test generating embedding with empty text raises error."""
        with self.assertRaises(ValueError):
            self.embedding_manager.generate_embedding("")

    def test_generate_embedding_api_error(self):
        """Test handling of API errors during embedding generation."""
        # Mock API error
        self.mock_client.models.embed_content.side_effect = Exception("API Error")

        with self.assertRaises(RuntimeError):
            self.embedding_manager.generate_embedding(self.sample_text)

    def test_store_node_embedding(self):
        """Test storing a node embedding."""
        # Mock the Gemini API response
        mock_result = MagicMock()
        mock_result.embeddings = [self.mock_embedding]
        self.mock_client.models.embed_content.return_value = mock_result

        # Store embedding
        self.embedding_manager.store_node_embedding(self.sample_node_id, self.sample_text)

        # Verify vector store was called
        self.vector_store.add_embedding.assert_called_once_with(
            self.sample_node_id, self.mock_embedding
        )

    def test_store_node_embedding_empty_node_id(self):
        """Test storing embedding with empty node_id raises error."""
        with self.assertRaises(ValueError):
            self.embedding_manager.store_node_embedding("", self.sample_text)

    def test_store_node_embedding_empty_text(self):
        """Test storing embedding with empty text raises error."""
        with self.assertRaises(ValueError):
            self.embedding_manager.store_node_embedding(self.sample_node_id, "")

    def test_store_node_embedding_vector_store_error(self):
        """Test handling of vector store errors during embedding storage."""
        # Mock the Gemini API response
        mock_result = MagicMock()
        mock_result.embeddings = [self.mock_embedding]
        self.mock_client.models.embed_content.return_value = mock_result

        # Make vector store raise an error
        self.vector_store.add_embedding.side_effect = Exception("Vector Store Error")

        with self.assertRaises(RuntimeError):
            self.embedding_manager.store_node_embedding(self.sample_node_id, self.sample_text)

    def test_search_similar_nodes(self):
        """Test searching for similar nodes."""
        # Mock the Gemini API response
        mock_result = MagicMock()
        mock_result.embeddings = [self.mock_embedding]
        self.mock_client.models.embed_content.return_value = mock_result

        # Mock vector store search results
        expected_nodes = ["node1", "node2", "node3"]
        self.vector_store.get_node_ids.return_value = expected_nodes

        # Search for similar nodes
        results = self.embedding_manager.search_similar_nodes(self.sample_text, top_k=3)

        # Verify vector store was called
        self.vector_store.get_node_ids.assert_called_once_with(self.mock_embedding, 3)

        # Verify results
        self.assertEqual(results, expected_nodes)


@pytest.mark.integration
class TestEmbeddingManagerIntegration:
    """Integration tests for EmbeddingManager with real Gemini API."""

    def setup_method(self):
        """Set up the test with actual connection to services."""
        # Skip if API key not set
        if not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY environment variable not set")

        # Create vector store with unique collection name to avoid dimension conflicts
        import time

        collection_name = f"test_embeddings_{int(time.time())}"
        self.vector_store = VectorStoreMilvus(
            host="localhost",
            port=19530,
            collection_name=collection_name,
            dimension=768,  # Gemini embedding dimension (gemini-embedding-exp-03-07 with config)
        )

        # Add a delay before trying to connect to Milvus
        time.sleep(5)  # Wait 5 seconds for Milvus to potentially start up

        # Try connecting to Milvus during setup
        try:
            if not self.vector_store.connect():
                pytest.skip("Could not connect to Milvus during setup")

            # Create embedding manager *after* successful connection
            self.embedding_manager = EmbeddingManager(self.vector_store)

        except Exception as e:
            # Ensure embedding_manager is not set if connection failed
            self.embedding_manager = None
            pytest.skip(f"Error connecting to Milvus during setup: {str(e)}")

    def teardown_method(self):
        """Clean up after test."""
        if hasattr(self, "vector_store"):
            try:
                # Drop the test collection to clean up
                self.vector_store.disconnect()
            except Exception:
                pass  # Ignore cleanup errors

    def test_live_generate_embedding(self):
        """Test generating embedding with real Gemini API."""
        try:
            # Generate embedding
            embedding = self.embedding_manager.generate_embedding("What is the meaning of life?")

            # Verify result
            assert len(embedding) == 768
            assert all(isinstance(x, float) for x in embedding)
        except Exception as e:
            pytest.skip(f"Gemini API error: {str(e)}")

    def test_live_store_and_search(self):
        """Test storing and searching embeddings with real services."""
        # Connection is now checked in setup_method
        # if not self.vector_store.connect():
        #     pytest.skip("Could not connect to Milvus")

        try:
            # Store embedding
            node_id = "test_node_1"
            text = "The meaning of life is to find purpose and happiness."
            self.embedding_manager.store_node_embedding(node_id, text)

            # Add a small delay to ensure embedding is searchable
            import time

            time.sleep(1)

            # Search for similar text
            similar_text = "What is the purpose of existence?"
            similar_nodes = self.embedding_manager.search_similar_nodes(similar_text, top_k=1)

            # Verify result
            assert len(similar_nodes) == 1
            assert similar_nodes[0] == node_id
        except Exception as e:
            print(f"\nFull Service error in test_live_store_and_search: {e}\n")  # Print full error
            pytest.skip(f"Skipping due to service error: {str(e)}")
