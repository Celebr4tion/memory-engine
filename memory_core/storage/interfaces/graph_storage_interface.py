"""
Abstract interface for graph storage backends.

This module defines the contract that all graph storage implementations
must follow to ensure consistent behavior across different backends.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from memory_core.model.knowledge_node import KnowledgeNode
from memory_core.model.relationship import Relationship


class GraphStorageInterface(ABC):
    """
    Abstract base class for graph storage backends.
    
    This interface defines all the operations that a graph storage backend
    must implement to be compatible with the memory engine.
    """
    
    # Connection Management
    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the storage backend."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close connection to the storage backend."""
        pass
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """Test if the connection to storage backend is working."""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the storage backend is available."""
        pass
    
    # Synchronous versions for backwards compatibility
    @abstractmethod
    def connect_sync(self) -> None:
        """Establish connection to the storage backend (synchronous)."""
        pass
    
    @abstractmethod
    def close_sync(self) -> None:
        """Close connection to the storage backend (synchronous)."""
        pass
    
    @abstractmethod
    def test_connection_sync(self) -> bool:
        """Test if the connection to storage backend is working (synchronous)."""
        pass
    
    @abstractmethod
    def is_available_sync(self) -> bool:
        """Check if the storage backend is available (synchronous)."""
        pass
    
    # Raw Node Operations
    @abstractmethod
    async def create_node(self, node_data: Dict[str, Any]) -> str:
        """
        Create a new node in the graph.
        
        Args:
            node_data: Dictionary containing node properties
            
        Returns:
            The ID of the created node
        """
        pass
    
    @abstractmethod
    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a node by its ID.
        
        Args:
            node_id: The ID of the node to retrieve
            
        Returns:
            Dictionary containing node data, or None if not found
        """
        pass
    
    @abstractmethod
    async def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """
        Update properties of an existing node.
        
        Args:
            node_id: The ID of the node to update
            properties: Dictionary of properties to update
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def delete_node(self, node_id: str) -> bool:
        """
        Delete a node from the graph.
        
        Args:
            node_id: The ID of the node to delete
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    # Raw Edge Operations
    @abstractmethod
    async def create_edge(self, from_node_id: str, to_node_id: str, 
                         relation_type: str, properties: Dict[str, Any]) -> str:
        """
        Create a new edge between two nodes.
        
        Args:
            from_node_id: ID of the source node
            to_node_id: ID of the target node
            relation_type: Type of the relationship
            properties: Dictionary containing edge properties
            
        Returns:
            The ID of the created edge
        """
        pass
    
    @abstractmethod
    async def get_edge(self, edge_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve an edge by its ID.
        
        Args:
            edge_id: The ID of the edge to retrieve
            
        Returns:
            Dictionary containing edge data, or None if not found
        """
        pass
    
    @abstractmethod
    async def update_edge(self, edge_id: str, properties: Dict[str, Any]) -> bool:
        """
        Update properties of an existing edge.
        
        Args:
            edge_id: The ID of the edge to update
            properties: Dictionary of properties to update
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def delete_edge(self, edge_id: str) -> bool:
        """
        Delete an edge from the graph.
        
        Args:
            edge_id: The ID of the edge to delete
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    # Graph Traversal Operations
    @abstractmethod
    async def find_neighbors(self, node_id: str, 
                           relation_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Find neighboring nodes of a given node.
        
        Args:
            node_id: The ID of the node to find neighbors for
            relation_type: Optional filter by relationship type
            
        Returns:
            List of neighbor node data dictionaries
        """
        pass
    
    @abstractmethod
    async def get_outgoing_relationships(self, node_id: str) -> List[Dict[str, Any]]:
        """
        Get all outgoing relationships from a node.
        
        Args:
            node_id: The ID of the source node
            
        Returns:
            List of relationship data dictionaries
        """
        pass
    
    @abstractmethod
    async def get_incoming_relationships(self, node_id: str) -> List[Dict[str, Any]]:
        """
        Get all incoming relationships to a node.
        
        Args:
            node_id: The ID of the target node
            
        Returns:
            List of relationship data dictionaries
        """
        pass
    
    @abstractmethod
    async def find_shortest_path(self, from_node_id: str, to_node_id: str,
                               max_hops: int = 5) -> List[str]:
        """
        Find the shortest path between two nodes.
        
        Args:
            from_node_id: ID of the starting node
            to_node_id: ID of the ending node
            max_hops: Maximum number of hops to consider
            
        Returns:
            List of node IDs representing the shortest path
        """
        pass
    
    # Query Operations
    @abstractmethod
    async def find_nodes_by_content(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Find nodes by content search.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of matching node data dictionaries
        """
        pass
    
    @abstractmethod
    async def get_relationships_for_node(self, node_id: str, 
                                       max_depth: int = 1) -> List[Dict[str, Any]]:
        """
        Get relationships for a node up to a certain depth.
        
        Args:
            node_id: The ID of the node
            max_depth: Maximum depth to traverse
            
        Returns:
            List of relationship data dictionaries
        """
        pass
    
    # Domain Model Operations
    @abstractmethod
    async def save_knowledge_node(self, node: KnowledgeNode) -> str:
        """
        Save a KnowledgeNode to storage.
        
        Args:
            node: The KnowledgeNode to save
            
        Returns:
            The ID of the saved node
        """
        pass
    
    @abstractmethod
    async def get_knowledge_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """
        Retrieve a KnowledgeNode by ID.
        
        Args:
            node_id: The ID of the node to retrieve
            
        Returns:
            KnowledgeNode instance or None if not found
        """
        pass
    
    @abstractmethod
    async def delete_knowledge_node(self, node_id: str) -> bool:
        """
        Delete a KnowledgeNode from storage.
        
        Args:
            node_id: The ID of the node to delete
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def save_relationship(self, relationship: Relationship) -> str:
        """
        Save a Relationship to storage.
        
        Args:
            relationship: The Relationship to save
            
        Returns:
            The ID of the saved relationship
        """
        pass
    
    @abstractmethod
    async def get_relationship(self, edge_id: str) -> Optional[Relationship]:
        """
        Retrieve a Relationship by ID.
        
        Args:
            edge_id: The ID of the relationship to retrieve
            
        Returns:
            Relationship instance or None if not found
        """
        pass
    
    @abstractmethod
    async def delete_relationship(self, edge_id: str) -> bool:
        """
        Delete a Relationship from storage.
        
        Args:
            edge_id: The ID of the relationship to delete
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    # Utility Operations
    @abstractmethod
    async def clear_all_data(self) -> bool:
        """
        Clear all data from the storage backend.
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def merge_nodes(self, node_id1: str, node_id2: str) -> bool:
        """
        Merge two nodes into one.
        
        Args:
            node_id1: ID of the first node
            node_id2: ID of the second node (will be merged into first)
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    # Performance and Caching
    @abstractmethod
    def clear_caches(self) -> None:
        """Clear any internal caches."""
        pass
    
    @abstractmethod
    def get_traversal_statistics(self) -> Dict[str, int]:
        """
        Get traversal statistics for performance monitoring.
        
        Returns:
            Dictionary containing performance statistics
        """
        pass
    
    # Synchronous versions of key operations (for backwards compatibility)
    @abstractmethod
    def create_node_sync(self, node_data: Dict[str, Any]) -> str:
        """Create a new node in the graph (synchronous)."""
        pass
    
    @abstractmethod
    def get_node_sync(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a node by its ID (synchronous)."""
        pass
    
    @abstractmethod
    def save_knowledge_node_sync(self, node: KnowledgeNode) -> str:
        """Save a KnowledgeNode to storage (synchronous)."""
        pass
    
    @abstractmethod
    def get_knowledge_node_sync(self, node_id: str) -> Optional[KnowledgeNode]:
        """Retrieve a KnowledgeNode by ID (synchronous)."""
        pass