"""
Tests for the node merging functionality.

These tests verify that the merging module correctly identifies similar nodes
and merges their data appropriately.
"""
import unittest
from unittest.mock import MagicMock, patch
import json
import time
import pytest
import os

from memory_core.ingestion.merging import merge_or_create_node, _merge_node_data, _deep_merge_dicts
from memory_core.db.janusgraph_storage import JanusGraphStorage
from memory_core.embeddings.embedding_manager import EmbeddingManager
from memory_core.embeddings.vector_store import VectorStoreMilvus


class TestMergingFunctions(unittest.TestCase):
    """Test cases for the merging utility functions."""
    
    def test_deep_merge_dicts(self):
        """Test deep merging of dictionaries."""
        # Simple case
        dict1 = {'a': 1, 'b': 2}
        dict2 = {'b': 3, 'c': 4}
        merged = _deep_merge_dicts(dict1, dict2)
        self.assertEqual(merged, {'a': 1, 'b': 3, 'c': 4})
        
        # Nested dictionaries
        dict1 = {'a': {'x': 1, 'y': 2}, 'b': 3}
        dict2 = {'a': {'y': 3, 'z': 4}, 'c': 5}
        merged = _deep_merge_dicts(dict1, dict2)
        self.assertEqual(merged, {'a': {'x': 1, 'y': 3, 'z': 4}, 'b': 3, 'c': 5})
        
        # Dict2 takes precedence for non-dict values
        dict1 = {'a': {'x': 1}, 'b': [1, 2, 3]}
        dict2 = {'a': {'x': 2}, 'b': 'overridden'}
        merged = _deep_merge_dicts(dict1, dict2)
        self.assertEqual(merged, {'a': {'x': 2}, 'b': 'overridden'})
    
    def test_merge_node_data(self):
        """Test merging of node data dictionaries."""
        # Test merging ratings (higher values should be kept)
        existing = {
            'content': 'Test content',
            'rating_truthfulness': 0.5,
            'rating_richness': 0.7,
            'rating_stability': 0.3
        }
        
        new = {
            'content': 'Updated content',  # This should not replace existing
            'rating_truthfulness': 0.8,    # Higher, should replace
            'rating_richness': 0.6,        # Lower, should not replace
            'rating_stability': 0.5        # Higher, should replace
        }
        
        merged = _merge_node_data(existing, new)
        
        # Content should not be replaced (not part of the merging logic)
        self.assertEqual(merged['content'], 'Test content')
        
        # Higher ratings should be kept
        self.assertEqual(merged['rating_truthfulness'], 0.8)
        self.assertEqual(merged['rating_richness'], 0.7)
        self.assertEqual(merged['rating_stability'], 0.5)
        
        # Should have a last_updated_timestamp
        self.assertIn('last_updated_timestamp', merged)
        
        # Test merging tags
        existing = {
            'content': 'Test content',
            'tags': 'ai,machine learning'
        }
        
        new = {
            'content': 'New content',
            'tags': 'ai,neural networks'
        }
        
        merged = _merge_node_data(existing, new)
        
        # Tags should be combined with duplicates removed
        expected_tags = set(['ai', 'machine learning', 'neural networks'])
        actual_tags = set(merged['tags'].split(','))
        self.assertEqual(actual_tags, expected_tags)
        
        # Test merging metadata
        existing = {
            'content': 'Test content',
            'extra_metadata': json.dumps({
                'domain': 'computer science',
                'importance': 0.7
            })
        }
        
        new = {
            'content': 'New content',
            'extra_metadata': json.dumps({
                'domain': 'artificial intelligence',
                'language': 'english'
            })
        }
        
        merged = _merge_node_data(existing, new)
        
        # Metadata should be deeply merged
        metadata = json.loads(merged['extra_metadata'])
        self.assertEqual(metadata['domain'], 'artificial intelligence')  # New takes precedence
        self.assertEqual(metadata['importance'], 0.7)  # Kept from existing
        self.assertEqual(metadata['language'], 'english')  # Added from new
        
        # Test merging source details
        existing = {
            'content': 'Test content',
            'source_details': 'Type: webpage; URL: https://example.com'
        }
        
        new = {
            'content': 'New content',
            'source_details': 'Type: article; Ref: Example et al.'
        }
        
        merged = _merge_node_data(existing, new)
        
        # Source details should be combined
        expected_details = set(['Type: webpage', 'URL: https://example.com', 
                              'Type: article', 'Ref: Example et al.'])
        actual_details = set(merged['source_details'].split('; '))
        self.assertEqual(actual_details, expected_details)


class TestMergeOrCreateNode(unittest.TestCase):
    """Test cases for the merge_or_create_node function."""
    
    def setUp(self):
        """Set up test dependencies."""
        # Mock storage
        self.mock_storage = MagicMock(spec=JanusGraphStorage)
        
        # Mock embedding manager
        self.mock_embedding_manager = MagicMock(spec=EmbeddingManager)
        
        # Mock vector store within embedding manager
        self.mock_vector_store = MagicMock(spec=VectorStoreMilvus)
        self.mock_embedding_manager.vector_store = self.mock_vector_store
        
        # Mock embedding generation
        self.mock_embedding_manager.generate_embedding.return_value = [0.1, 0.2, 0.3]
        
        # Sample node data
        self.node_data = {
            'content': 'Test content',
            'source': 'Test source',
            'creation_timestamp': time.time(),
            'rating_truthfulness': 0.8,
            'rating_richness': 0.7
        }
    
    def test_create_new_node_when_no_similar_exists(self):
        """Test creating a new node when no similar node exists."""
        # Mock search result (empty, no similar nodes)
        self.mock_vector_store.search_embedding.return_value = []
        
        # Mock node creation
        self.mock_storage.create_node.return_value = 'node123'
        
        # Call the function
        result = merge_or_create_node(
            content='Test content',
            node_data=self.node_data,
            storage=self.mock_storage,
            embedding_manager=self.mock_embedding_manager
        )
        
        # Verify embedding was generated
        self.mock_embedding_manager.generate_embedding.assert_called_once_with('Test content')
        
        # Verify search was performed
        self.mock_vector_store.search_embedding.assert_called_once()
        
        # Verify node was created
        self.mock_storage.create_node.assert_called_once_with(self.node_data)
        
        # Verify embedding was stored
        self.mock_vector_store.add_embedding.assert_called_once()
        
        # Verify result
        self.assertEqual(result, 'node123')
    
    def test_merge_when_similar_node_exists(self):
        """Test merging when a similar node exists."""
        # Mock high similarity search result
        self.mock_vector_store.search_embedding.return_value = [
            {'node_id': 'existing123', 'score': 0.95}  # High similarity
        ]
        
        # Mock existing node data
        existing_node = {
            'content': 'Existing content',
            'source': 'Original source',
            'rating_truthfulness': 0.6,
            'tags': 'ai,ml'
        }
        self.mock_storage.get_node.return_value = existing_node
        
        # Call the function
        result = merge_or_create_node(
            content='Test content',
            node_data=self.node_data,
            storage=self.mock_storage,
            embedding_manager=self.mock_embedding_manager,
            similarity_threshold=0.9  # Set threshold below search result score
        )
        
        # Verify embedding was generated and search was performed
        self.mock_embedding_manager.generate_embedding.assert_called_once()
        self.mock_vector_store.search_embedding.assert_called_once()
        
        # Verify existing node was retrieved
        self.mock_storage.get_node.assert_called_once_with('existing123')
        
        # Verify node was updated not created
        self.mock_storage.update_node.assert_called_once()
        self.mock_storage.create_node.assert_not_called()
        
        # Verify result is the existing node ID
        self.assertEqual(result, 'existing123')
    
    def test_create_new_when_similarity_below_threshold(self):
        """Test creating a new node when similarity is below threshold."""
        # Mock low similarity search result
        self.mock_vector_store.search_embedding.return_value = [
            {'node_id': 'existing123', 'score': 0.85}  # Below threshold
        ]
        
        # Mock node creation
        self.mock_storage.create_node.return_value = 'node123'
        
        # Call the function
        result = merge_or_create_node(
            content='Test content',
            node_data=self.node_data,
            storage=self.mock_storage,
            embedding_manager=self.mock_embedding_manager,
            similarity_threshold=0.9  # Set threshold above search result score
        )
        
        # Verify node was created (not merged)
        self.mock_storage.create_node.assert_called_once()
        self.mock_storage.update_node.assert_not_called()
        
        # Verify result is the new node ID
        self.assertEqual(result, 'node123')
    
    def test_fallback_to_creation_on_error(self):
        """Test falling back to node creation on error during similarity check."""
        # Mock embedding manager to raise error
        self.mock_embedding_manager.generate_embedding.side_effect = RuntimeError("API error")
        
        # Mock node creation
        self.mock_storage.create_node.return_value = 'fallback123'
        
        # Call the function
        result = merge_or_create_node(
            content='Test content',
            node_data=self.node_data,
            storage=self.mock_storage,
            embedding_manager=self.mock_embedding_manager
        )
        
        # Verify node was created as fallback
        self.mock_storage.create_node.assert_called_once()
        
        # Verify result is the fallback node ID
        self.assertEqual(result, 'fallback123')
    
    def test_no_embedding_manager(self):
        """Test creating node when no embedding manager is provided."""
        # Mock node creation
        self.mock_storage.create_node.return_value = 'node123'
        
        # Call the function without embedding_manager
        result = merge_or_create_node(
            content='Test content',
            node_data=self.node_data,
            storage=self.mock_storage,
            embedding_manager=None
        )
        
        # Verify node was created directly (no similarity check)
        self.mock_storage.create_node.assert_called_once()
        
        # Verify result
        self.assertEqual(result, 'node123')


@pytest.mark.integration
class TestMergingIntegration:
    """Integration tests for the merging functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Skip if not in integration test mode
        if os.environ.get("SKIP_INTEGRATION_TESTS", "true").lower() == "true":
            pytest.skip("Integration tests are disabled")
        
        try:
            # Create real dependencies
            from memory_core.core.knowledge_engine import KnowledgeEngine
            from memory_core.db.janusgraph_storage import JanusGraphStorage
            from memory_core.embeddings.embedding_manager import EmbeddingManager
            
            # Check if JanusGraph is available
            if not JanusGraphStorage.is_available_sync():
                pytest.skip("JanusGraph is not available")
            
            # Initialize KnowledgeEngine
            self.engine = KnowledgeEngine()
            self.engine.connect()
            
            # Store references to storage and embedding_manager
            self.storage = self.engine.storage
            self.embedding_manager = self.engine.embedding_manager
            
            # Check if vector store is available
            if self.embedding_manager is None or not hasattr(self.embedding_manager, 'vector_store'):
                pytest.skip("Embedding manager is not available")
                
            # Create test nodes
            self.node1_data = {
                'content': 'Artificial intelligence is intelligence demonstrated by machines.',
                'source': 'Integration Test',
                'creation_timestamp': time.time(),
                'tags': 'ai,machine learning,intelligence',
                'rating_truthfulness': 0.8,
                'rating_richness': 0.7,
                'rating_stability': 0.6
            }
            
            self.node2_data = {
                'content': 'Machine intelligence is the intelligence shown by machines.',
                'source': 'Another Source',
                'creation_timestamp': time.time(),
                'tags': 'ai,computer science',
                'rating_truthfulness': 0.7,
                'rating_richness': 0.8,
                'rating_stability': 0.5
            }
            
            self.node3_data = {
                'content': 'Quantum computing uses quantum mechanics to perform computation.',
                'source': 'Integration Test',
                'creation_timestamp': time.time(),
                'tags': 'quantum,computing',
                'rating_truthfulness': 0.9,
                'rating_richness': 0.8,
                'rating_stability': 0.7
            }
            
        except Exception as e:
            pytest.skip(f"Error setting up integration test: {str(e)}")
    
    def teardown_method(self):
        """Clean up after test."""
        # Add cleanup logic here if needed
        if hasattr(self, 'engine'):
            self.engine.disconnect()
        if hasattr(self, 'storage') and self.storage:
            self.storage.close()
        if hasattr(self, 'embedding_manager') and hasattr(self.embedding_manager, 'vector_store'):
            self.embedding_manager.vector_store.disconnect()
    
    @pytest.mark.skipif(
        os.environ.get("SKIP_INTEGRATION_TESTS", "true").lower() == "true",
        reason="Integration tests are disabled"
    )
    def test_merge_similar_nodes(self):
        """Test merging similar nodes with real backend."""
        # Ensure storage and its connection (g) are available
        if not hasattr(self, 'storage') or not self.storage.g:
            pytest.skip("JanusGraph storage not available in setup")
        # Ensure embedding manager and its connection are available
        if not hasattr(self, 'embedding_manager') or not self.embedding_manager.vector_store.connected:
            pytest.skip("Embedding manager / Milvus not available in setup")

        embedding_manager = self.embedding_manager # Use the one from setup

        try:
            # Create first node
            node1_id = merge_or_create_node(
                content=self.node1_data['content'],
                node_data=self.node1_data,
                storage=self.engine.storage,
                embedding_manager=self.engine.embedding_manager
            )
            
            # Create/merge second node (should be merged with first due to similarity)
            node2_id = merge_or_create_node(
                content=self.node2_data['content'],
                node_data=self.node2_data,
                storage=self.engine.storage,
                embedding_manager=self.engine.embedding_manager,
                similarity_threshold=0.85  # Threshold that allows merging
            )
            
            # Verify that IDs are the same (merged)
            assert node1_id == node2_id
            
            # Get the first node (or merged node if embeddings used)
            retrieved_node1 = self.engine.storage.get_node(node1_id)
            
            # Verify tags were handled correctly (either merged or kept separate)
            expected_tags_node1 = set(self.node1_data.get('tags', '').split(','))
            if embedding_manager: # Merged case
                # Note: graph db might store tags differently (e.g., sorted string)
                expected_tags_merged = sorted(list(expected_tags_node1.union(set(self.node2_data.get('tags', '').split(',')))))
                actual_tags = sorted(retrieved_node1.get('tags', '').split(',')) 
                assert actual_tags == expected_tags_merged, f"Expected {expected_tags_merged}, got {actual_tags}"
            else: # Separate case
                actual_tags_node1 = set(retrieved_node1.get('tags', '').split(','))
                assert actual_tags_node1 == expected_tags_node1
            
            # Create third node (different content, should not be merged)
            node3_id = merge_or_create_node(
                content=self.node3_data['content'],
                node_data=self.node3_data,
                storage=self.engine.storage,
                embedding_manager=self.engine.embedding_manager
            )
            
            # Verify that third node has a different ID (not merged)
            assert node1_id != node3_id
            
            # Clean up
            self.engine.storage.delete_node(node1_id)
            if not embedding_manager: # Delete node 2 if separate
                self.engine.storage.delete_node(node2_id)
            self.engine.storage.delete_node(node3_id)
            
        except Exception as e:
            # Print full exception for debugging
            import traceback
            print(f"\nIntegration test failed unexpectedly:\n{traceback.format_exc()}\n")
            pytest.fail(f"Integration test failed: {str(e)}")


if __name__ == "__main__":
    if os.environ.get("SKIP_INTEGRATION_TESTS", "true").lower() == "true":
        # Run only unit tests
        unittest.main()
    else:
        # Run all tests including integration
        pytest.main(["-xvs", __file__])