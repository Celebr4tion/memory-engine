"""
Versioned graph adapter that integrates revision tracking with graph operations.

This module provides a wrapper around GraphStorageAdapter that automatically
records changes to the graph in the RevisionManager.
"""
from typing import Dict, Any, Optional, List, Tuple, Union

from memory_core.db.graph_storage_adapter import GraphStorageAdapter
from memory_core.versioning.revision_manager import RevisionManager
from memory_core.model.knowledge_node import KnowledgeNode
from memory_core.model.relationship import Relationship


class VersionedGraphAdapter:
    """
    A versioned graph adapter that automatically tracks changes.
    
    This class wraps the GraphStorageAdapter and automatically records
    all changes to nodes and edges in the RevisionManager.
    """
    
    def __init__(
        self, 
        graph_adapter: GraphStorageAdapter, 
        revision_manager: RevisionManager
    ):
        """
        Initialize the VersionedGraphAdapter.
        
        Args:
            graph_adapter: The underlying GraphStorageAdapter
            revision_manager: The RevisionManager for tracking changes
        """
        self.graph_adapter = graph_adapter
        self.revision_manager = revision_manager
    
    def save_knowledge_node(self, node: KnowledgeNode) -> str:
        """
        Save a knowledge node and record the change.
        
        Args:
            node: The KnowledgeNode to save
            
        Returns:
            The ID of the saved node
        """
        # Check if the node already exists
        is_update = node.node_id is not None
        
        if is_update:
            try:
                # Get the existing node for before/after comparison
                existing_node = self.graph_adapter.get_knowledge_node(node.node_id)
                existing_data = existing_node.to_dict()
                
                # Save the node
                result = self.graph_adapter.save_knowledge_node(node)
                
                # Log the update
                self.revision_manager.log_node_update(
                    node_id=node.node_id,
                    old_data=existing_data,
                    new_data=node.to_dict()
                )
                
                return result
            except ValueError:
                # Node doesn't exist, treat as creation
                is_update = False
        
        if not is_update:
            # Save the node (creates new)
            result = self.graph_adapter.save_knowledge_node(node)
            
            # Get the updated node with ID
            node_with_id = self.graph_adapter.get_knowledge_node(result)
            
            # Log the creation
            self.revision_manager.log_node_creation(
                node_id=result,
                node_data=node_with_id.to_dict()
            )
            
            return result
    
    def get_knowledge_node(self, node_id: str) -> KnowledgeNode:
        """
        Get a knowledge node by ID.
        
        Args:
            node_id: The ID of the node to retrieve
            
        Returns:
            The retrieved KnowledgeNode
            
        Raises:
            ValueError: If the node does not exist
        """
        return self.graph_adapter.get_knowledge_node(node_id)
    
    def delete_knowledge_node(self, node_id: str) -> bool:
        """
        Delete a knowledge node and record the change.
        
        Args:
            node_id: The ID of the node to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get the node before deletion
            node = self.graph_adapter.get_knowledge_node(node_id)
            node_data = node.to_dict()
            
            # Delete the node
            result = self.graph_adapter.delete_knowledge_node(node_id)
            
            if result:
                # Log the deletion
                self.revision_manager.log_node_deletion(
                    node_id=node_id,
                    node_data=node_data
                )
            
            return result
        except ValueError:
            # Node doesn't exist
            return False
    
    def save_relationship(self, relationship: Relationship) -> str:
        """
        Save a relationship and record the change.
        
        Args:
            relationship: The Relationship to save
            
        Returns:
            The ID of the saved relationship
        """
        # Check if the relationship already exists
        is_update = relationship.edge_id is not None
        
        if is_update:
            try:
                # Get the existing relationship for before/after comparison
                existing_rel = self.graph_adapter.get_relationship(relationship.edge_id)
                existing_data = existing_rel.to_dict()
                
                # Save the relationship
                result = self.graph_adapter.save_relationship(relationship)
                
                # Log the update
                self.revision_manager.log_edge_update(
                    edge_id=relationship.edge_id,
                    old_data=existing_data,
                    new_data=relationship.to_dict()
                )
                
                return result
            except ValueError:
                # Relationship doesn't exist, treat as creation
                is_update = False
        
        if not is_update:
            # Save the relationship (creates new)
            result = self.graph_adapter.save_relationship(relationship)
            
            # Get the updated relationship with ID
            rel_with_id = self.graph_adapter.get_relationship(result)
            
            # Log the creation
            self.revision_manager.log_edge_creation(
                edge_id=result,
                edge_data=rel_with_id.to_dict()
            )
            
            return result
    
    def get_relationship(self, edge_id: str) -> Relationship:
        """
        Get a relationship by ID.
        
        Args:
            edge_id: The ID of the relationship to retrieve
            
        Returns:
            The retrieved Relationship
            
        Raises:
            ValueError: If the relationship does not exist
        """
        return self.graph_adapter.get_relationship(edge_id)
    
    def delete_relationship(self, edge_id: str) -> bool:
        """
        Delete a relationship and record the change.
        
        Args:
            edge_id: The ID of the relationship to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get the relationship before deletion
            rel = self.graph_adapter.get_relationship(edge_id)
            rel_data = rel.to_dict()
            
            # Delete the relationship
            result = self.graph_adapter.delete_relationship(edge_id)
            
            if result:
                # Log the deletion
                self.revision_manager.log_edge_deletion(
                    edge_id=edge_id,
                    edge_data=rel_data
                )
            
            return result
        except ValueError:
            # Relationship doesn't exist
            return False
    
    def get_outgoing_relationships(self, node_id: str) -> List[Relationship]:
        """
        Get all outgoing relationships from a node.
        
        Args:
            node_id: The ID of the source node
            
        Returns:
            List of relationships where the node is the source
        """
        return self.graph_adapter.get_outgoing_relationships(node_id)
    
    def get_incoming_relationships(self, node_id: str) -> List[Relationship]:
        """
        Get all incoming relationships to a node.
        
        Args:
            node_id: The ID of the target node
            
        Returns:
            List of relationships where the node is the target
        """
        return self.graph_adapter.get_incoming_relationships(node_id)
    
    def revert_node_to_previous_state(self, node_id: str) -> bool:
        """
        Revert a node to its previous state.
        
        Args:
            node_id: ID of the node to revert
            
        Returns:
            True if successful, False otherwise
        """
        return self.revision_manager.revert_node_to_previous_state(node_id)
    
    def revert_relationship_to_previous_state(self, edge_id: str) -> bool:
        """
        Revert a relationship to its previous state.
        
        Args:
            edge_id: ID of the relationship to revert
            
        Returns:
            True if successful, False otherwise
        """
        return self.revision_manager.revert_edge_to_previous_state(edge_id)
    
    def create_snapshot(self) -> str:
        """
        Create a snapshot of the current graph state.
        
        Returns:
            ID of the created snapshot
        """
        return self.revision_manager.create_snapshot()
    
    def get_revision_history(self, object_type: str, object_id: str) -> List[Dict[str, Any]]:
        """
        Get the revision history for a specific object.
        
        Args:
            object_type: Type of object ('node' or 'edge')
            object_id: ID of the object
            
        Returns:
            List of revision entries, sorted by timestamp (descending)
        """
        return self.revision_manager.get_revision_history(object_type, object_id) 