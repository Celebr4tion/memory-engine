"""
JanusGraph adapter module for converting between domain models and database storage.

This module provides an adapter for JanusGraph that converts KnowledgeNode and Relationship
objects to and from JanusGraph storage formats, with optimized graph traversal operations.
"""
import logging
from collections import deque
from typing import Dict, Any, Optional, List
import asyncio

from memory_core.storage.interfaces.graph_storage_interface import GraphStorageInterface
from memory_core.model.knowledge_node import KnowledgeNode
from memory_core.model.relationship import Relationship
from .janusgraph_storage import JanusGraphStorage


class JanusGraphAdapter:
    """
    Adapter for storing and retrieving domain models from JanusGraph.
    
    This class provides methods to convert KnowledgeNode and Relationship objects
    to a format suitable for JanusGraph storage, and vice versa, with optimized
    graph traversal operations for performance.
    """
    
    def __init__(self, storage: JanusGraphStorage):
        """
        Initialize the adapter with a JanusGraphStorage instance.
        
        Args:
            storage: A JanusGraphStorage instance
        """
        self.storage = storage
        self.logger = logging.getLogger(__name__)
        
        # Traversal optimization caches
        self._relationship_cache = {}
        self._content_index_cache = {}
    
    def save_knowledge_node(self, node: KnowledgeNode) -> str:
        """
        Save a KnowledgeNode to the database.
        
        Args:
            node: The KnowledgeNode to save
            
        Returns:
            The node ID (newly created if it didn't exist)
        """
        return self.storage.save_knowledge_node_sync(node)
    
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
        node = self.storage.get_knowledge_node_sync(node_id)
        if node is None:
            raise ValueError(f"Node {node_id} not found")
        return node
    
    def delete_knowledge_node(self, node_id: str) -> None:
        """
        Delete a KnowledgeNode from the database.
        
        Args:
            node_id: The ID of the node to delete
            
        Raises:
            ValueError: If the node doesn't exist
        """
        success = self._run_async_in_sync_context(self.storage.delete_knowledge_node(node_id))
        if not success:
            raise ValueError(f"Failed to delete node {node_id}")
    
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
        return self._run_async_in_sync_context(self.storage.save_relationship(relationship))
    
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
        relationship = self._run_async_in_sync_context(self.storage.get_relationship(edge_id))
        if relationship is None:
            raise ValueError(f"Relationship {edge_id} not found")
        return relationship
    
    def delete_relationship(self, edge_id: str) -> None:
        """
        Delete a Relationship from the database.
        
        Args:
            edge_id: The ID of the edge to delete
            
        Raises:
            ValueError: If the edge doesn't exist
        """
        success = self._run_async_in_sync_context(self.storage.delete_relationship(edge_id))
        if not success:
            raise ValueError(f"Failed to delete relationship {edge_id}")
    
    def get_node_by_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get node data by ID with caching optimization.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Node data dictionary or None if not found
        """
        return self.storage.get_node_sync(node_id)
    
    def find_nodes_by_content(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Find nodes containing specific content with optimized text search.
        
        Args:
            query: Content query string
            limit: Maximum number of nodes to return
            
        Returns:
            List of node data dictionaries
        """
        cache_key = f"{query.lower().strip()}:{limit}"
        
        # Check cache first
        if cache_key in self._content_index_cache:
            self.logger.debug(f"Using cached content search for: {query[:30]}...")
            return self._content_index_cache[cache_key]
        
        try:
            results = self._run_async_in_sync_context(
                self.storage.find_nodes_by_content(query, limit)
            )
            
            # Cache results
            self._content_index_cache[cache_key] = results
            
            # Limit cache size
            if len(self._content_index_cache) > 100:
                # Remove oldest entry
                oldest_key = next(iter(self._content_index_cache))
                del self._content_index_cache[oldest_key]
            
            return results
            
        except Exception as e:
            self.logger.error(f"Content search failed: {e}")
            return []
    
    def get_relationships_for_node(self, node_id: str, max_depth: int = 1) -> List[Dict[str, Any]]:
        """
        Get relationships for a node with optimized multi-hop traversal.
        
        Args:
            node_id: Node identifier
            max_depth: Maximum traversal depth
            
        Returns:
            List of relationship data
        """
        cache_key = f"{node_id}:{max_depth}"
        
        # Check cache first
        if cache_key in self._relationship_cache:
            self.logger.debug(f"Using cached relationships for node: {node_id}")
            return self._relationship_cache[cache_key]
        
        try:
            relationships_data = self._run_async_in_sync_context(
                self.storage.get_relationships_for_node(node_id, max_depth)
            )
            
            # Cache results
            self._relationship_cache[cache_key] = relationships_data
            
            # Limit cache size
            if len(self._relationship_cache) > 50:
                # Remove oldest entry
                oldest_key = next(iter(self._relationship_cache))
                del self._relationship_cache[oldest_key]
            
            return relationships_data
            
        except Exception as e:
            self.logger.error(f"Failed to get relationships for node {node_id}: {e}")
            return []
    
    def get_outgoing_relationships(self, node_id: str) -> List[Relationship]:
        """
        Get outgoing relationships for a node.
        
        Args:
            node_id: Node identifier
            
        Returns:
            List of Relationship objects
        """
        try:
            relationships_data = self._run_async_in_sync_context(
                self.storage.get_outgoing_relationships(node_id)
            )
            
            relationships = []
            for rel_data in relationships_data:
                relationship = Relationship(
                    from_id=rel_data['from_id'],
                    to_id=rel_data['to_id'],
                    relation_type=rel_data['relation_type'],
                    timestamp=rel_data['timestamp'],
                    confidence_score=rel_data['confidence_score'],
                    version=rel_data['version'],
                    edge_id=rel_data['edge_id']
                )
                relationships.append(relationship)
            
            return relationships
            
        except Exception as e:
            self.logger.error(f"Failed to get outgoing relationships for {node_id}: {e}")
            return []
    
    def get_incoming_relationships(self, node_id: str) -> List[Relationship]:
        """
        Get incoming relationships for a node.
        
        Args:
            node_id: Node identifier
            
        Returns:
            List of Relationship objects
        """
        try:
            relationships_data = self._run_async_in_sync_context(
                self.storage.get_incoming_relationships(node_id)
            )
            
            relationships = []
            for rel_data in relationships_data:
                relationship = Relationship(
                    from_id=rel_data['from_id'],
                    to_id=rel_data['to_id'],
                    relation_type=rel_data['relation_type'],
                    timestamp=rel_data['timestamp'],
                    confidence_score=rel_data['confidence_score'],
                    version=rel_data['version'],
                    edge_id=rel_data['edge_id']
                )
                relationships.append(relationship)
            
            return relationships
            
        except Exception as e:
            self.logger.error(f"Failed to get incoming relationships for {node_id}: {e}")
            return []
    
    def find_shortest_path(self, from_node_id: str, to_node_id: str, max_hops: int = 5) -> List[str]:
        """
        Find shortest path between two nodes using optimized graph traversal.
        
        Args:
            from_node_id: Starting node ID
            to_node_id: Target node ID
            max_hops: Maximum number of hops to search
            
        Returns:
            List of node IDs representing the path, empty if no path found
        """
        return self._run_async_in_sync_context(
            self.storage.find_shortest_path(from_node_id, to_node_id, max_hops)
        )
    
    def get_node_neighbors(self, node_id: str, relation_types: Optional[List[str]] = None) -> List[str]:
        """
        Get immediate neighbors of a node with relation type filtering.
        
        Args:
            node_id: Node identifier
            relation_types: Optional list of relation types to filter by
            
        Returns:
            List of neighbor node IDs
        """
        try:
            neighbors_data = self._run_async_in_sync_context(
                self.storage.find_neighbors(
                    node_id, 
                    relation_types[0] if relation_types else None
                )
            )
            
            # Extract neighbor node IDs
            neighbor_ids = []
            for neighbor in neighbors_data:
                if neighbor['from_id'] != node_id:
                    neighbor_ids.append(neighbor['from_id'])
                if neighbor['to_id'] != node_id:
                    neighbor_ids.append(neighbor['to_id'])
            
            return list(set(neighbor_ids))  # Remove duplicates
            
        except Exception as e:
            self.logger.error(f"Failed to get neighbors for {node_id}: {e}")
            return []
    
    def clear_caches(self):
        """Clear traversal caches."""
        self._relationship_cache.clear()
        self._content_index_cache.clear()
        self.storage.clear_caches()
        self.logger.info("JanusGraph adapter caches cleared")
    
    def get_traversal_statistics(self) -> Dict[str, int]:
        """Get traversal performance statistics."""
        storage_stats = self.storage.get_traversal_statistics()
        adapter_stats = {
            'adapter_relationship_cache_entries': len(self._relationship_cache),
            'adapter_content_cache_entries': len(self._content_index_cache)
        }
        return {**storage_stats, **adapter_stats}
    
    def _run_async_in_sync_context(self, coroutine, timeout=10):
        """Helper function to run an async coroutine in a sync context."""
        def run_in_thread():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(coroutine)
                finally:
                    loop.close()
            except Exception as e:
                self.logger.error(f"Error in thread: {e}")
                raise
        
        try:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=timeout)
        except Exception as e:
            self.logger.error(f"Error running async in sync context: {e}")
            raise