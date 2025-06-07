"""
Tests for the rating system module.

This module tests the functionality for updating and managing ratings
of knowledge nodes based on evidence.
"""
import unittest
from unittest.mock import MagicMock, patch
import json
import time

from memory_core.rating.rating_system import update_rating, RatingUpdater
from memory_core.db.janusgraph_storage import JanusGraphStorage


class TestUpdateRating(unittest.TestCase):
    """Test cases for the update_rating function."""
    
    def setUp(self):
        """Set up test dependencies."""
        # Mock storage
        self.mock_storage = MagicMock(spec=JanusGraphStorage)
        
        # Sample node data
        self.node_data = {
            'node_id': 'test_node_123',
            'content': 'Test content',
            'rating_truthfulness': 0.5,
            'rating_richness': 0.5,
            'rating_stability': 0.5
        }
        
        # Configure mock get_node to return our sample node
        self.mock_storage.get_node.return_value = self.node_data.copy()
    
    def test_update_truthfulness_with_confirmation(self):
        """Test that truthfulness increases with confirmation."""
        # Call the function with confirmation evidence
        evidence = {'confirmation': 1.0}
        result = update_rating('test_node_123', evidence, self.mock_storage)
        
        # Verify node was retrieved
        self.mock_storage.get_node.assert_called_once_with('test_node_123')
        
        # Verify node was updated
        self.mock_storage.update_node.assert_called_once()
        
        # Extract the updated data
        _, kwargs = self.mock_storage.update_node.call_args
        updated_data = kwargs
        
        # Verify truthfulness increased
        self.assertIn('rating_truthfulness', updated_data)
        self.assertGreater(updated_data['rating_truthfulness'], 0.5)
        self.assertLessEqual(updated_data['rating_truthfulness'], 1.0)
        
        # Verify other ratings were not updated
        self.assertNotIn('rating_richness', updated_data)
        self.assertNotIn('rating_stability', updated_data)
        
        # Verify result format
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['node_id'], 'test_node_123')
        self.assertIn('updates', result)
        self.assertIn('previous', result)
    
    def test_update_truthfulness_with_contradiction(self):
        """Test that truthfulness decreases with contradiction."""
        # Call the function with contradiction evidence
        evidence = {'contradiction': 1.0}
        result = update_rating('test_node_123', evidence, self.mock_storage)
        
        # Extract the updated data
        _, kwargs = self.mock_storage.update_node.call_args
        updated_data = kwargs
        
        # Verify truthfulness decreased
        self.assertIn('rating_truthfulness', updated_data)
        self.assertLess(updated_data['rating_truthfulness'], 0.5)
        self.assertGreaterEqual(updated_data['rating_truthfulness'], 0.0)
    
    def test_update_all_ratings(self):
        """Test updating all ratings simultaneously."""
        # Call the function with evidence for all ratings
        evidence = {
            'confirmation': 0.5,
            'richness': 1.0,
            'stability': -0.5
        }
        result = update_rating('test_node_123', evidence, self.mock_storage)
        
        # Extract the updated data
        _, kwargs = self.mock_storage.update_node.call_args
        updated_data = kwargs
        
        # Verify all ratings were updated
        self.assertIn('rating_truthfulness', updated_data)
        self.assertIn('rating_richness', updated_data)
        self.assertIn('rating_stability', updated_data)
        
        # Verify directions of changes
        self.assertGreater(updated_data['rating_truthfulness'], 0.5)  # Increased with confirmation
        self.assertGreater(updated_data['rating_richness'], 0.5)      # Increased with positive richness
        self.assertLess(updated_data['rating_stability'], 0.5)        # Decreased with negative stability
    
    def test_no_changes_for_empty_evidence(self):
        """Test that no changes occur with empty evidence."""
        # Call the function with empty evidence
        evidence = {}
        result = update_rating('test_node_123', evidence, self.mock_storage)
        
        # Verify node was retrieved but not updated
        self.mock_storage.get_node.assert_called_once()
        self.mock_storage.update_node.assert_not_called()
        
        # Verify result indicates no changes
        self.assertEqual(result['status'], 'no_changes')
    
    def test_boundary_values(self):
        """Test that ratings stay within 0.0 to 1.0 bounds."""
        # Set up extreme initial values
        high_node = {
            'node_id': 'high_node',
            'rating_truthfulness': 0.9,
            'rating_richness': 0.9,
            'rating_stability': 0.9
        }
        
        low_node = {
            'node_id': 'low_node',
            'rating_truthfulness': 0.1,
            'rating_richness': 0.1,
            'rating_stability': 0.1
        }
        
        # Configure mock to return different nodes
        def mock_get_node(node_id):
            if node_id == 'high_node':
                return high_node
            elif node_id == 'low_node':
                return low_node
            return self.node_data.copy()
            
        self.mock_storage.get_node.side_effect = mock_get_node
        
        # Test that high values don't exceed 1.0
        evidence = {
            'confirmation': 1.0,
            'richness': 1.0,
            'stability': 1.0
        }
        result = update_rating('high_node', evidence, self.mock_storage)
        
        # Extract updated data
        _, kwargs = self.mock_storage.update_node.call_args
        updated_data = kwargs
        
        # Verify ratings capped at 1.0
        self.assertLessEqual(updated_data['rating_truthfulness'], 1.0)
        self.assertLessEqual(updated_data['rating_richness'], 1.0)
        self.assertLessEqual(updated_data['rating_stability'], 1.0)
        
        # Reset mock
        self.mock_storage.update_node.reset_mock()
        
        # Test that low values don't go below 0.0
        evidence = {
            'contradiction': 1.0,
            'richness': -1.0,
            'stability': -1.0
        }
        result = update_rating('low_node', evidence, self.mock_storage)
        
        # Extract updated data
        _, kwargs = self.mock_storage.update_node.call_args
        updated_data = kwargs
        
        # Verify ratings floored at 0.0
        self.assertGreaterEqual(updated_data['rating_truthfulness'], 0.0)
        self.assertGreaterEqual(updated_data['rating_richness'], 0.0)
        self.assertGreaterEqual(updated_data['rating_stability'], 0.0)
    
    def test_handles_missing_ratings_in_node(self):
        """Test handling of nodes that are missing rating fields."""
        # Node with missing ratings
        incomplete_node = {
            'node_id': 'incomplete_node',
            'content': 'Content without ratings'
            # No rating fields
        }
        
        # Configure mock to return incomplete node
        self.mock_storage.get_node.return_value = incomplete_node
        
        # Test updating ratings
        evidence = {
            'confirmation': 0.5,
            'richness': 0.5,
            'stability': 0.5
        }
        result = update_rating('incomplete_node', evidence, self.mock_storage)
        
        # Extract updated data
        _, kwargs = self.mock_storage.update_node.call_args
        updated_data = kwargs
        
        # Verify default values were used (0.5 + positive adjustment)
        self.assertGreater(updated_data['rating_truthfulness'], 0.5)
        self.assertGreater(updated_data['rating_richness'], 0.5)
        self.assertGreater(updated_data['rating_stability'], 0.5)


class TestRatingUpdater(unittest.TestCase):
    """Test cases for the RatingUpdater class."""
    
    def setUp(self):
        """Set up test dependencies."""
        # Mock storage
        self.mock_storage = MagicMock(spec=JanusGraphStorage)
        
        # Mock update_rating function
        self.update_rating_patcher = patch('memory_core.rating.rating_system.update_rating')
        self.mock_update_rating = self.update_rating_patcher.start()
        
        # Configure mock update_rating to return success result
        self.mock_update_rating.return_value = {
            'status': 'success',
            'node_id': 'test_node',
            'updates': {'rating_truthfulness': 0.7},
            'previous': {'rating_truthfulness': 0.5}
        }
        
        # Create RatingUpdater instance
        self.rating_updater = RatingUpdater(self.mock_storage)
    
    def tearDown(self):
        """Clean up patches."""
        self.update_rating_patcher.stop()
    
    def test_update_ratings_delegates_to_update_rating(self):
        """Test that update_ratings method delegates to update_rating function."""
        # Call the method
        evidence = {'confirmation': 0.5}
        result = self.rating_updater.update_ratings('test_node', evidence)
        
        # Verify update_rating was called with correct arguments
        self.mock_update_rating.assert_called_once_with('test_node', evidence, self.mock_storage)
        
        # Verify result is passed through
        self.assertEqual(result, self.mock_update_rating.return_value)
    
    def test_record_confirmation(self):
        """Test record_confirmation convenience method."""
        # Call the method
        result = self.rating_updater.record_confirmation('test_node', 0.8)
        
        # Verify update_rating was called with confirmation evidence
        self.mock_update_rating.assert_called_once()
        _, kwargs = self.mock_update_rating.call_args
        evidence = kwargs.get('evidence')
        
        # Verify evidence contains only confirmation
        self.assertEqual(len(evidence), 1)
        self.assertIn('confirmation', evidence)
        self.assertEqual(evidence['confirmation'], 0.8)
    
    def test_record_contradiction(self):
        """Test record_contradiction convenience method."""
        # Call the method
        result = self.rating_updater.record_contradiction('test_node', 0.6)
        
        # Verify update_rating was called with contradiction evidence
        self.mock_update_rating.assert_called_once()
        _, kwargs = self.mock_update_rating.call_args
        evidence = kwargs.get('evidence')
        
        # Verify evidence contains only contradiction
        self.assertEqual(len(evidence), 1)
        self.assertIn('contradiction', evidence)
        self.assertEqual(evidence['contradiction'], 0.6)
    
    def test_update_all_ratings(self):
        """Test update_all_ratings convenience method."""
        # Call the method with mixed changes
        result = self.rating_updater.update_all_ratings(
            'test_node',
            truthfulness_change=0.5,    # Positive: use confirmation
            richness_change=-0.3,       # Negative: should still work
            stability_change=0.0        # Zero: should be skipped
        )
        
        # Verify update_rating was called
        self.mock_update_rating.assert_called_once()
        _, kwargs = self.mock_update_rating.call_args
        evidence = kwargs.get('evidence')
        
        # Verify evidence structure
        self.assertIn('confirmation', evidence)     # Positive truthfulness uses confirmation
        self.assertEqual(evidence['confirmation'], 0.5)
        
        self.assertIn('richness', evidence)        # Richness included as-is
        self.assertEqual(evidence['richness'], -0.3)
        
        # Stability should be skipped since change is 0
        self.assertNotIn('stability', evidence)    
    
    def test_update_all_ratings_with_negative_truthfulness(self):
        """Test update_all_ratings with negative truthfulness."""
        # Call the method with negative truthfulness
        result = self.rating_updater.update_all_ratings(
            'test_node',
            truthfulness_change=-0.4,   # Negative: should use contradiction
            richness_change=0.2,
            stability_change=0.3
        )
        
        # Verify update_rating was called
        self.mock_update_rating.assert_called_once()
        _, kwargs = self.mock_update_rating.call_args
        evidence = kwargs.get('evidence')
        
        # Verify evidence structure
        self.assertIn('contradiction', evidence)   # Negative truthfulness uses contradiction
        self.assertEqual(evidence['contradiction'], 0.4)  # Absolute value used
        
        self.assertNotIn('confirmation', evidence)  # No confirmation used
        
        # Other ratings included as normal
        self.assertIn('richness', evidence)
        self.assertIn('stability', evidence)


if __name__ == "__main__":
    unittest.main()