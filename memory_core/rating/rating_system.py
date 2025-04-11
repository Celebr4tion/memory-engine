"""
Rating system for evaluating and updating knowledge node quality.

This module provides functionality to systematically update knowledge node ratings
based on new evidence or information.
"""

import logging
from typing import Dict, Any, Optional, Union

from memory_core.db.janusgraph_storage import JanusGraphStorage
from memory_core.model.knowledge_node import KnowledgeNode


def update_rating(node_id: str, evidence: Dict[str, float], storage=None) -> Dict[str, Any]:
    """
    Update a node's ratings based on new evidence.
    
    This function retrieves a knowledge node, updates its ratings using the provided
    evidence, and saves the updated node back to the graph database.
    
    Args:
        node_id: ID of the node to update
        evidence: Dictionary containing evidence for rating updates
            - 'confirmation': Positive evidence of truthfulness (0.0-1.0)
            - 'contradiction': Negative evidence of truthfulness (0.0-1.0)
            - 'richness': Evidence of information richness (-1.0 to 1.0)
            - 'stability': Evidence of information stability (-1.0 to 1.0)
        storage: Optional JanusGraphStorage instance (if None, one will be created)
    
    Returns:
        Dictionary with the updated node's data
    
    Raises:
        ValueError: If node_id doesn't exist or evidence is invalid
        RuntimeError: If update operation fails
    """
    logger = logging.getLogger(__name__)
    
    # Create storage if not provided
    if storage is None:
        from memory_core.core.knowledge_engine import KnowledgeEngine
        engine = KnowledgeEngine(enable_versioning=True)
        engine.connect()
        storage = engine.storage
    
    try:
        # Get the node from JanusGraph
        node_data = storage.get_node(node_id)
        
        # Extract current ratings
        old_truthfulness = node_data.get('rating_truthfulness', 0.5)
        old_richness = node_data.get('rating_richness', 0.5)
        old_stability = node_data.get('rating_stability', 0.5)
        
        # Calculate new ratings
        updated_data = {}
        
        # Update truthfulness based on confirmation/contradiction evidence
        if 'confirmation' in evidence or 'contradiction' in evidence:
            confirmation = evidence.get('confirmation', 0.0)
            contradiction = evidence.get('contradiction', 0.0)
            
            # Apply the formula: new = min(1.0, max(0.0, old + 0.2*confirmation - 0.2*contradiction))
            new_truthfulness = min(1.0, max(0.0, 
                old_truthfulness + 0.2 * confirmation - 0.2 * contradiction
            ))
            
            updated_data['rating_truthfulness'] = new_truthfulness
            logger.debug(f"Updated truthfulness: {old_truthfulness:.2f} -> {new_truthfulness:.2f}")
        
        # Update richness rating
        if 'richness' in evidence:
            richness_factor = evidence.get('richness', 0.0)
            
            # Apply the formula: new = min(1.0, max(0.0, old + 0.2*richness_factor))
            new_richness = min(1.0, max(0.0, 
                old_richness + 0.2 * richness_factor
            ))
            
            updated_data['rating_richness'] = new_richness
            logger.debug(f"Updated richness: {old_richness:.2f} -> {new_richness:.2f}")
        
        # Update stability rating
        if 'stability' in evidence:
            stability_factor = evidence.get('stability', 0.0)
            
            # Apply the formula: new = min(1.0, max(0.0, old + 0.2*stability_factor))
            new_stability = min(1.0, max(0.0, 
                old_stability + 0.2 * stability_factor
            ))
            
            updated_data['rating_stability'] = new_stability
            logger.debug(f"Updated stability: {old_stability:.2f} -> {new_stability:.2f}")
        
        # If we have changes, update the node
        if updated_data:
            # Keep track of the old data for versioning
            old_data = {
                k: node_data[k] for k in updated_data.keys() if k in node_data
            }
            
            # Update the node - pass updated data as keyword arguments
            storage.update_node(node_id, **updated_data)
            logger.info(f"Updated ratings for node {node_id}")
            
            # Record the change in the revision log if available
            try:
                # Check if revision manager is available through adapter
                from memory_core.db.versioned_graph_adapter import VersionedGraphAdapter
                if hasattr(storage, 'revision_manager') and storage.revision_manager:
                    storage.revision_manager.log_node_update(node_id, old_data, updated_data)
                    logger.debug("Recorded rating change in revision log")
            except ImportError:
                # Revision manager not available, skip logging
                pass
            
            # Return the updated node data
            return {
                'status': 'success',
                'node_id': node_id,
                'updates': updated_data,
                'previous': old_data
            }
        else:
            logger.info(f"No rating updates required for node {node_id}")
            return {
                'status': 'no_changes',
                'node_id': node_id
            }
    
    except Exception as e:
        logger.error(f"Error updating ratings for node {node_id}: {str(e)}")
        raise RuntimeError(f"Failed to update node ratings: {str(e)}")


class RatingUpdater:
    """Class-based interface for updating node ratings."""
    
    def __init__(self, storage=None):
        """
        Initialize the rating updater.
        
        Args:
            storage: Optional JanusGraphStorage instance
        """
        self.logger = logging.getLogger(__name__)
        
        # Create storage if not provided
        if storage is None:
            from memory_core.core.knowledge_engine import KnowledgeEngine
            engine = KnowledgeEngine(enable_versioning=True)
            engine.connect()
            self.storage = engine.storage
        else:
            self.storage = storage
    
    def update_ratings(self, node_id: str, evidence: Dict[str, float]) -> Dict[str, Any]:
        """
        Update a node's ratings using the provided evidence.
        
        Args:
            node_id: ID of the node to update
            evidence: Dictionary with rating evidence
            
        Returns:
            Dictionary with the update results
        """
        return update_rating(node_id, evidence, self.storage)
    
    def record_confirmation(self, node_id: str, strength: float = 0.5) -> Dict[str, Any]:
        """
        Record evidence of truthfulness confirmation
        
        Args:
            node_id: ID of the node
            strength: Strength of the confirmation evidence (0.0-1.0)
            
        Returns:
            Dictionary with the update results
        """
        evidence = {"confirmation": strength}
        return update_rating(node_id, evidence=evidence, storage=self.storage)
    
    def record_contradiction(self, node_id: str, strength: float = 0.5) -> Dict[str, Any]:
        """
        Record evidence of truthfulness contradiction
        
        Args:
            node_id: ID of the node
            strength: Strength of the contradiction evidence (0.0-1.0)
            
        Returns:
            Dictionary with the update results
        """
        evidence = {"contradiction": strength}
        return update_rating(node_id, evidence=evidence, storage=self.storage)
    
    def update_all_ratings(self, node_id: str, evidence_dict: Dict[str, float] = None, truthfulness_change: float = None, richness_change: float = None, stability_change: float = None) -> Dict[str, Any]:
        """
        Update multiple ratings at once with different evidence types.
        
        Args:
            node_id: ID of the node
            evidence_dict: Dictionary mapping evidence types to strength values
            truthfulness_change: Direct change to truthfulness rating (-1.0 to 1.0)
            richness_change: Direct change to richness rating (-1.0 to 1.0)
            stability_change: Direct change to stability rating (-1.0 to 1.0)
            
        Returns:
            Dictionary with the update results
        """
        # If no evidence_dict was provided, create one
        if evidence_dict is None:
            evidence_dict = {}
            
        # Handle direct rating changes by converting them to appropriate evidence
        if truthfulness_change is not None and truthfulness_change != 0.0:
            if truthfulness_change >= 0:
                evidence_dict['confirmation'] = abs(truthfulness_change)
            else:
                evidence_dict['contradiction'] = abs(truthfulness_change)
                
        if richness_change is not None and richness_change != 0.0:
            evidence_dict['richness'] = richness_change
            
        if stability_change is not None and stability_change != 0.0:
            evidence_dict['stability'] = stability_change
            
        return update_rating(node_id, evidence=evidence_dict, storage=self.storage)