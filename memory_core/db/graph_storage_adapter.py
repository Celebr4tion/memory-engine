"""
Graph storage adapter module for converting between domain models and database storage.

This module provides adapters for converting KnowledgeNode and Relationship
objects to and from JanusGraph storage formats.
"""
from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING

# Remove direct import to break cycle
# from memory_core.db.janusgraph_storage import JanusGraphStorage
from memory_core.model.knowledge_node import KnowledgeNode
from memory_core.model.relationship import Relationship

# Use TYPE_CHECKING to allow type hint without runtime import
if TYPE_CHECKING:
    from memory_core.db.janusgraph_storage import JanusGraphStorage

class GraphStorageAdapter:
    """
    Adapter for storing and retrieving domain models from JanusGraph.
    
    This class provides methods to convert KnowledgeNode and Relationship objects
    to a format suitable for JanusGraph storage, and vice versa.
    """
    
    def __init__(self, storage: 'JanusGraphStorage'):
        """
        Initialize the adapter with a JanusGraphStorage instance.
        
        Args:
            storage: A connected JanusGraphStorage instance
        """
        self.storage = storage
    
    def save_knowledge_node(self, node: KnowledgeNode) -> str:
        """
        Save a KnowledgeNode to the database.
        
        Args:
            node: The KnowledgeNode to save
            
        Returns:
            The node ID (newly created if it didn't exist)
        """
        node_data = {
            'content': node.content,
            'source': node.source,
            'creation_timestamp': node.creation_timestamp,
            'rating_richness': node.rating_richness,
            'rating_truthfulness': node.rating_truthfulness,
            'rating_stability': node.rating_stability
        }
        
        if node.node_id is None:
            # Create new node
            node_id = self.storage.create_node(node_data)
        else:
            # Update existing node
            node_id = node.node_id
            self.storage.update_node(node_id, node_data)
        
        return node_id
    
    def get_knowledge_node(self, node_id: str) -> KnowledgeNode:
        """
        Retrieve a KnowledgeNode from the database.
        
        Args:
            node_id: The ID of the node to retrieve
            
        Returns:
            The retrieved KnowledgeNode
            
        Raises:
            ValueError: If the node doesn't exist
        """
        node_data = self.storage.get_node(node_id)
        
        return KnowledgeNode(
            content=node_data['content'],
            source=node_data['source'],
            creation_timestamp=node_data['creation_timestamp'],
            rating_richness=node_data['rating_richness'],
            rating_truthfulness=node_data['rating_truthfulness'],
            rating_stability=node_data['rating_stability'],
            node_id=node_id
        )
    
    def delete_knowledge_node(self, node_id: str) -> None:
        """
        Delete a KnowledgeNode from the database.
        
        Args:
            node_id: The ID of the node to delete
            
        Raises:
            ValueError: If the node doesn't exist
        """
        self.storage.delete_node(node_id)
    
    def save_relationship(self, relationship: Relationship) -> str:
        """
        Save a Relationship to the database.
        
        Args:
            relationship: The Relationship to save
            
        Returns:
            The edge ID (newly created if it didn't exist)
            
        Raises:
            ValueError: If either the source or target node doesn't exist
        """
        edge_metadata = {
            'timestamp': relationship.timestamp,
            'confidence_score': relationship.confidence_score,
            'version': relationship.version
        }
        
        if relationship.edge_id is None:
            # Create new relationship
            edge_id = self.storage.create_edge(
                relationship.from_id,
                relationship.to_id,
                relationship.relation_type,
                edge_metadata
            )
        else:
            # Update existing relationship
            edge_id = relationship.edge_id
            self.storage.update_edge(edge_id, edge_metadata)
        
        return edge_id
    
    def get_relationship(self, edge_id: str) -> Relationship:
        """
        Retrieve a Relationship from the database.
        
        Args:
            edge_id: The ID of the edge to retrieve
            
        Returns:
            The retrieved Relationship
            
        Raises:
            ValueError: If the edge doesn't exist
        """
        edge_data = self.storage.get_edge(edge_id)
        
        return Relationship(
            from_id=edge_data['from_id'],
            to_id=edge_data['to_id'],
            relation_type=edge_data['relation_type'],
            timestamp=edge_data['timestamp'],
            confidence_score=edge_data['confidence_score'],
            version=edge_data['version'],
            edge_id=edge_id
        )
    
    def delete_relationship(self, edge_id: str) -> None:
        """
        Delete a Relationship from the database.
        
        Args:
            edge_id: The ID of the edge to delete
            
        Raises:
            ValueError: If the edge doesn't exist
        """
        self.storage.delete_edge(edge_id) 