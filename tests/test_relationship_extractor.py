"""
Tests for the relationship extractor functionality.

This module tests the automatic detection and creation of relationships
between knowledge nodes based on various heuristics.
"""
import unittest
from unittest.mock import MagicMock, patch
import json
import time
import pytest

from memory_core.ingestion.relationship_extractor import (
    auto_relationship_by_tags,
    suggest_semantic_relationships,
    create_domain_relationships,
    analyze_and_create_relationships
)
from memory_core.db.janusgraph_storage import JanusGraphStorage
from memory_core.embeddings.embedding_manager import EmbeddingManager


class TestAutoRelationshipByTags(unittest.TestCase):
    """Test automatic relationship creation based on shared tags."""
    
    def setUp(self):
        """Set up test dependencies."""
        # Mock storage
        self.mock_storage = MagicMock(spec=JanusGraphStorage)
        
        # Create test node data
        self.node_data = {
            'node1': {
                'tags': 'ai,ml,neural networks'
            },
            'node2': {
                'tags': 'ai,deep learning,computer vision'
            },
            'node3': {
                'tags': 'blockchain,crypto,distributed systems'
            },
            'node4': {  # No tags
                'content': 'Node without tags'
            },
            'node5': {
                'tags': 'ai,ml'  # High overlap with node1
            }
        }
        
        # Configure mock get_node to return node data
        def mock_get_node(node_id):
            if node_id in self.node_data:
                return self.node_data[node_id]
            raise ValueError(f"Node {node_id} not found")
            
        self.mock_storage.get_node.side_effect = mock_get_node
        
        # Configure mock create_edge to return edge IDs
        self.edge_id_counter = 0
        
        def mock_create_edge(from_id, to_id, relation_type, edge_metadata):
            self.edge_id_counter += 1
            return f"edge{self.edge_id_counter}"
            
        self.mock_storage.create_edge.side_effect = mock_create_edge
    
    def test_detects_tag_based_relationships(self):
        """Test that relationships are created between nodes with shared tags."""
        # Call the function with all nodes
        node_ids = ['node1', 'node2', 'node3', 'node4', 'node5']
        edge_ids = auto_relationship_by_tags(node_ids, self.mock_storage)
        
        # Expected relationships: 
        # - node1 and node2 (share 'ai')
        # - node1 and node5 (share 'ai' and 'ml')
        # - node2 and node5 (share 'ai')
        
        # Verify number of edges created
        self.assertEqual(len(edge_ids), 3)
        
        # Verify edge creation calls
        self.assertEqual(self.mock_storage.create_edge.call_count, 3)
        
        # Get all call arguments
        call_args_list = self.mock_storage.create_edge.call_args_list
        
        # Extract node pairs from calls (ignoring order)
        node_pairs = []
        for call in call_args_list:
            args, kwargs = call
            from_id = kwargs.get('from_id') or args[0]
            to_id = kwargs.get('to_id') or args[1]
            node_pairs.append(tuple(sorted([from_id, to_id])))
        
        # Expected node pairs (sorted to ensure consistent comparison)
        expected_pairs = [
            ('node1', 'node2'),
            ('node1', 'node5'),
            ('node2', 'node5')
        ]
        
        # Convert to sets for comparison (order doesn't matter)
        self.assertEqual(set(node_pairs), set(tuple(sorted(p)) for p in expected_pairs))
        
    def test_confidence_increases_with_tag_similarity(self):
        """Test that confidence score increases with tag similarity."""
        # Call the function with nodes that have different levels of similarity
        node_ids = ['node1', 'node5']  # High overlap ('ai', 'ml')
        edge_ids = auto_relationship_by_tags(node_ids, self.mock_storage)
        
        # Verify edge was created
        self.assertEqual(len(edge_ids), 1)
        
        # Make sure create_edge was actually called
        self.mock_storage.create_edge.assert_called_once()
        
        # Get edge metadata from the call
        args, kwargs = self.mock_storage.create_edge.call_args
        
        # Extract confidence score
        edge_metadata = kwargs.get('edge_metadata', {})
        high_overlap_confidence = edge_metadata.get('confidence_score', 0)
        
        # Reset mock and counter
        self.mock_storage.create_edge.reset_mock()
        self.edge_id_counter = 0
        
        # Call the function with nodes that have lower similarity
        node_ids = ['node1', 'node2']  # Lower overlap (only 'ai')
        edge_ids = auto_relationship_by_tags(node_ids, self.mock_storage)
        
        # Make sure create_edge was actually called
        self.mock_storage.create_edge.assert_called_once()
        
        # Get edge metadata from the call
        args, kwargs = self.mock_storage.create_edge.call_args
        
        # Extract confidence score
        edge_metadata = kwargs.get('edge_metadata', {})
        low_overlap_confidence = edge_metadata.get('confidence_score', 0)
        
        # Verify higher similarity leads to higher confidence
        self.assertGreater(high_overlap_confidence, low_overlap_confidence)
        
    def test_no_relationships_below_threshold(self):
        """Test that no relationships are created when similarity is below threshold."""
        # Call the function with a high threshold
        node_ids = ['node1', 'node2']  # Share only 'ai'
        edge_ids = auto_relationship_by_tags(
            node_ids, 
            self.mock_storage,
            min_overlap_threshold=0.5  # Higher than the actual similarity
        )
        
        # Verify no edges were created
        self.assertEqual(len(edge_ids), 0)
        self.mock_storage.create_edge.assert_not_called()
        
    def test_nodes_without_tags_ignored(self):
        """Test that nodes without tags are ignored."""
        # Call the function with one node that has tags and one without
        node_ids = ['node1', 'node4']  # node4 has no tags
        edge_ids = auto_relationship_by_tags(node_ids, self.mock_storage)
        
        # Verify no edges were created
        self.assertEqual(len(edge_ids), 0)
        self.mock_storage.create_edge.assert_not_called()
        
    def test_proper_edge_metadata_created(self):
        """Test that edge metadata is properly created."""
        # Call the function
        node_ids = ['node1', 'node5']  # High tag overlap
        edge_ids = auto_relationship_by_tags(node_ids, self.mock_storage)
        
        # Get edge metadata from the call
        _, kwargs = self.mock_storage.create_edge.call_args
        
        # Extract edge metadata
        edge_metadata = kwargs.get('edge_metadata', {})
        
        # Verify metadata structure
        self.assertIn('confidence_score', edge_metadata)
        self.assertIn('timestamp', edge_metadata)
        self.assertIn('shared_tags', edge_metadata)
        self.assertIn('tag_similarity', edge_metadata)
        
        # Verify correct relation type
        relation_type = kwargs.get('relation_type')
        self.assertEqual(relation_type, "RELATED")


class TestCreateDomainRelationships(unittest.TestCase):
    """Test automatic relationship creation based on shared domain."""
    
    def setUp(self):
        """Set up test dependencies."""
        # Mock storage
        self.mock_storage = MagicMock(spec=JanusGraphStorage)
        
        # Create test node data
        self.node_data = {
            'node1': {
                'extra_metadata': json.dumps({'domain': 'computer science'})
            },
            'node2': {
                'extra_metadata': json.dumps({'domain': 'computer science'})
            },
            'node3': {
                'extra_metadata': json.dumps({'domain': 'physics'})
            },
            'node4': {  # No domain
                'extra_metadata': json.dumps({'language': 'english'})
            },
            'node5': {
                'extra_metadata': json.dumps({'domain': 'physics'})
            }
        }
        
        # Configure mock get_node
        def mock_get_node(node_id):
            if node_id in self.node_data:
                return self.node_data[node_id]
            raise ValueError(f"Node {node_id} not found")
            
        self.mock_storage.get_node.side_effect = mock_get_node
        
        # Configure mock create_edge
        self.edge_id_counter = 0
        
        def mock_create_edge(from_id, to_id, relation_type, edge_metadata):
            self.edge_id_counter += 1
            return f"edge{self.edge_id_counter}"
            
        self.mock_storage.create_edge.side_effect = mock_create_edge
    
    def test_detects_domain_based_relationships(self):
        """Test that relationships are created between nodes with the same domain."""
        # Call the function with all nodes
        node_ids = ['node1', 'node2', 'node3', 'node4', 'node5']
        edge_ids = create_domain_relationships(node_ids, self.mock_storage)
        
        # Expected relationships:
        # - node1 and node2 (both 'computer science')
        # - node3 and node5 (both 'physics')
        
        # Verify number of edges created
        self.assertEqual(len(edge_ids), 2)
        
        # Verify edge creation calls
        self.assertEqual(self.mock_storage.create_edge.call_count, 2)
        
        # Get all call arguments
        call_args_list = self.mock_storage.create_edge.call_args_list
        
        # Extract node pairs and relation types from calls
        edge_details = []
        for call in call_args_list:
            args, kwargs = call
            from_id = kwargs.get('from_id') or args[0]
            to_id = kwargs.get('to_id') or args[1]
            relation_type = kwargs.get('relation_type') or args[2]
            edge_details.append((
                tuple(sorted([from_id, to_id])),
                relation_type
            ))
        
        # Expected edge details
        expected_edges = [
            (('node1', 'node2'), 'SAME_DOMAIN'),
            (('node3', 'node5'), 'SAME_DOMAIN')
        ]
        
        # Convert to sets for comparison
        self.assertEqual(
            set((tuple(sorted(pair)), rel_type) for pair, rel_type in edge_details),
            set((tuple(sorted(pair)), rel_type) for pair, rel_type in expected_edges)
        )
        
    def test_nodes_without_domain_ignored(self):
        """Test that nodes without domain metadata are ignored."""
        # Call the function with one node with domain and one without
        node_ids = ['node1', 'node4']  # node4 has no domain
        edge_ids = create_domain_relationships(node_ids, self.mock_storage)
        
        # Verify no edges were created
        self.assertEqual(len(edge_ids), 0)
        self.mock_storage.create_edge.assert_not_called()
        
    def test_proper_edge_metadata_created(self):
        """Test that edge metadata is properly created."""
        # Call the function
        node_ids = ['node1', 'node2']  # Same domain
        edge_ids = create_domain_relationships(node_ids, self.mock_storage)
        
        # Get edge metadata from the call
        _, kwargs = self.mock_storage.create_edge.call_args
        
        # Extract edge metadata
        edge_metadata = kwargs.get('edge_metadata', {})
        
        # Verify metadata structure
        self.assertIn('confidence_score', edge_metadata)
        self.assertIn('timestamp', edge_metadata)
        self.assertIn('shared_domain', edge_metadata)
        
        # Verify correct relation type
        relation_type = kwargs.get('relation_type')
        self.assertEqual(relation_type, "SAME_DOMAIN")
        
        # Verify shared domain value
        self.assertEqual(edge_metadata['shared_domain'], 'computer science')


class TestSemanticRelationships(unittest.TestCase):
    """Test suggestion of semantic relationships based on embedding similarity."""
    
    def setUp(self):
        """Set up test dependencies."""
        # Mock storage
        self.mock_storage = MagicMock(spec=JanusGraphStorage)
        
        # Mock embedding manager
        self.mock_embedding_manager = MagicMock(spec=EmbeddingManager)
        
        # Create test node data
        self.node_data = {
            'node1': {'content': 'AI is transforming healthcare.'},
            'node2': {'content': 'Machine learning is changing medicine.'},
            'node3': {'content': 'Blockchain is a distributed ledger technology.'}
        }
        
        # Configure mock get_node
        def mock_get_node(node_id):
            if node_id in self.node_data:
                return self.node_data[node_id]
            raise ValueError(f"Node {node_id} not found")
            
        self.mock_storage.get_node.side_effect = mock_get_node
        
        # Configure mock embedding generation
        self.embeddings = {
            'node1': [0.1, 0.2, 0.3],
            'node2': [0.15, 0.25, 0.35],  # Similar to node1
            'node3': [0.7, 0.8, 0.9]      # Different from others
        }
        
        def mock_generate_embedding(content):
            for node_id, node_data in self.node_data.items():
                if node_data['content'] == content:
                    return self.embeddings[node_id]
            return [0, 0, 0]  # Default for unknown content
            
        self.mock_embedding_manager.generate_embedding.side_effect = mock_generate_embedding
        
        # Configure mock vector store in embedding manager
        self.mock_vector_store = MagicMock()
        self.mock_embedding_manager.vector_store = self.mock_vector_store
        
        # Configure mock search behavior
        def mock_search_embedding(embedding, top_k=5):
            # For testing, return nodes similar to node1 or node2
            if embedding == self.embeddings['node1'] or embedding == self.embeddings['node2']:
                # These are semantically similar
                return [{'node_id': 'node2', 'score': 0.9}]
            # For node3, return low similarity to show contrast
            return [{'node_id': 'node1', 'score': 0.3}]
            
        self.mock_vector_store.search_embedding.side_effect = mock_search_embedding
    
    def test_suggest_semantic_relationships(self):
        """Test that semantic relationships are suggested based on embedding similarity."""
        # Define a fixed mock search behavior for the vector store
        def mock_search_embedding(embedding, top_k=5):
            # Always return a result with a high similarity score
            return [{'node_id': 'node2', 'score': 0.9}]
            
        self.mock_vector_store.search_embedding = mock_search_embedding
        
        # Add a calculate_similarity method to the mock vector store
        def mock_calculate_similarity(embedding1, embedding2):
            # Always return high similarity for our test case
            return 0.9
            
        self.mock_vector_store.calculate_similarity = mock_calculate_similarity
        
        # Call the function with nodes that are known to be similar
        node_ids = ['node1', 'node2']
        suggestions = suggest_semantic_relationships(
            node_ids,
            self.mock_embedding_manager,
            self.mock_storage
        )
        
        # Verify suggestions were made
        self.assertGreater(len(suggestions), 0)
        
        # Check suggestion format
        suggestion = suggestions[0]
        self.assertIn('from_id', suggestion)
        self.assertIn('to_id', suggestion)
        self.assertIn('similarity', suggestion)
        self.assertIn('relation_type', suggestion)
        
        # Verify correct relation type
        self.assertEqual(suggestion['relation_type'], 'SEMANTICALLY_SIMILAR')
    
    def test_similarity_threshold_filtering(self):
        """Test that suggested relationships respect the similarity threshold."""
        # Create a properly mocked calculate_similarity method
        def mock_calculate_similarity(embedding1, embedding2):
            # Return low similarity for node3 compared to node1
            if embedding1 == self.embeddings['node1'] and embedding2 == self.embeddings['node3']:
                return 0.3
            # Return high similarity otherwise
            return 0.9
            
        self.mock_vector_store.calculate_similarity = mock_calculate_similarity
        
        # Call with a high threshold
        node_ids = ['node1', 'node3']  # Low similarity between these
        suggestions = suggest_semantic_relationships(
            node_ids,
            self.mock_embedding_manager,
            self.mock_storage,
            similarity_threshold=0.8  # Higher than the similarity between node1 and node3
        )
        
        # Verify no suggestions were made
        self.assertEqual(len(suggestions), 0)
        
        # Now call with a lower threshold
        suggestions = suggest_semantic_relationships(
            node_ids,
            self.mock_embedding_manager,
            self.mock_storage,
            similarity_threshold=0.2  # Lower than the similarity between node1 and node3
        )
        
        # Verify suggestions were made
        self.assertGreater(len(suggestions), 0)
    
    def test_returns_empty_list_when_no_embedding_manager(self):
        """Test that function returns empty list when no embedding manager is provided."""
        # Call without embedding manager
        node_ids = ['node1', 'node2']
        suggestions = suggest_semantic_relationships(
            node_ids,
            embedding_manager=None,
            storage=self.mock_storage
        )
        
        # Verify empty list returned
        self.assertEqual(suggestions, [])


class TestAnalyzeAndCreateRelationships(unittest.TestCase):
    """Test the comprehensive relationship analysis function."""
    
    def setUp(self):
        """Set up test dependencies."""
        # Set up patches for component functions
        self.auto_tag_patcher = patch('memory_core.ingestion.relationship_extractor.auto_relationship_by_tags')
        self.mock_auto_tag = self.auto_tag_patcher.start()
        self.mock_auto_tag.return_value = ['tag_edge1', 'tag_edge2']
        
        self.domain_patcher = patch('memory_core.ingestion.relationship_extractor.create_domain_relationships')
        self.mock_domain = self.domain_patcher.start()
        self.mock_domain.return_value = ['domain_edge1']
        
        self.semantic_patcher = patch('memory_core.ingestion.relationship_extractor.suggest_semantic_relationships')
        self.mock_semantic = self.semantic_patcher.start()
        self.mock_semantic.return_value = [
            {'from_id': 'node1', 'to_id': 'node2', 'similarity': 0.9, 'relation_type': 'SEMANTICALLY_SIMILAR'}
        ]
        
        # Mock storage and embedding manager
        self.mock_storage = MagicMock(spec=JanusGraphStorage)
        self.mock_embedding_manager = MagicMock(spec=EmbeddingManager)
        
        # Configure create_edge mock
        self.edge_id_counter = 0
        
        def mock_create_edge(from_id, to_id, relation_type, edge_metadata):
            self.edge_id_counter += 1
            return f"semantic_edge{self.edge_id_counter}"
            
        self.mock_storage.create_edge.side_effect = mock_create_edge
    
    def tearDown(self):
        """Clean up patches."""
        self.auto_tag_patcher.stop()
        self.domain_patcher.stop()
        self.semantic_patcher.stop()
    
    def test_calls_all_relationship_methods(self):
        """Test that analyze_and_create_relationships calls all component methods."""
        # Call the function
        node_ids = ['node1', 'node2', 'node3']
        results = analyze_and_create_relationships(
            node_ids,
            self.mock_storage,
            self.mock_embedding_manager
        )
        
        # Verify all component methods were called
        self.mock_auto_tag.assert_called_once_with(node_ids, self.mock_storage)
        self.mock_domain.assert_called_once_with(node_ids, self.mock_storage)
        self.mock_semantic.assert_called_once()
        
        # Verify semantic relationships were created
        self.assertEqual(self.mock_storage.create_edge.call_count, 1)
        
        # Verify results contain all relationship types
        self.assertIn('tag_relationships', results)
        self.assertIn('domain_relationships', results)
        self.assertIn('semantic_relationships', results)
        
        # Verify correct number of relationships
        self.assertEqual(len(results['tag_relationships']), 2)
        self.assertEqual(len(results['domain_relationships']), 1)
        self.assertEqual(len(results['semantic_relationships']), 1)
    
    def test_works_without_embedding_manager(self):
        """Test that analyze_and_create_relationships works without embedding manager."""
        # Call the function without embedding manager
        node_ids = ['node1', 'node2']
        results = analyze_and_create_relationships(
            node_ids,
            self.mock_storage,
            embedding_manager=None
        )
        
        # Verify tag and domain methods were called
        self.mock_auto_tag.assert_called_once()
        self.mock_domain.assert_called_once()
        
        # Verify semantic method was not called
        self.mock_semantic.assert_not_called()
        
        # Verify no semantic relationships were created
        self.mock_storage.create_edge.assert_not_called()
        
        # Verify results don't contain semantic relationships
        self.assertEqual(results['semantic_relationships'], [])


if __name__ == "__main__":
    unittest.main()