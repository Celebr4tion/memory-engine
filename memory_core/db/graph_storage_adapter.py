"""
Graph storage adapter module for converting between domain models and database storage.

This module provides adapters for converting KnowledgeNode and Relationship
objects to and from JanusGraph storage formats, with optimized graph traversal operations.
"""
import logging
from collections import deque
from typing import Dict, Any, Optional, Tuple, List, Set, TYPE_CHECKING

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
    to a format suitable for JanusGraph storage, and vice versa, with optimized
    graph traversal operations for performance.
    """
    
    def __init__(self, storage: 'JanusGraphStorage'):
        """
        Initialize the adapter with a JanusGraphStorage instance.
        
        Args:
            storage: A connected JanusGraphStorage instance
        """
        self.storage = storage
        self.logger = logging.getLogger(__name__)
        
        # Traversal optimization caches
        self._relationship_cache = {}  # Cache for recent relationship queries
        self._content_index_cache = {}  # Cache for content search results
    
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
    
    def get_node_by_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get node data by ID with caching optimization.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Node data dictionary or None if not found
        """
        try:
            return self.storage.get_node(node_id)
        except ValueError:
            return None
    
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
            # Use storage's content search capability
            if hasattr(self.storage, 'find_nodes_by_content'):
                results = self.storage.find_nodes_by_content(query, limit)
            else:
                # Fallback implementation
                results = self._fallback_content_search(query, limit)
            
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
            if max_depth == 1:
                # Simple case - direct relationships only
                relationships = self._get_direct_relationships(node_id)
            else:
                # Multi-hop traversal
                relationships = self._get_multi_hop_relationships(node_id, max_depth)
            
            # Cache results
            self._relationship_cache[cache_key] = relationships
            
            # Limit cache size
            if len(self._relationship_cache) > 50:
                # Remove oldest entry
                oldest_key = next(iter(self._relationship_cache))
                del self._relationship_cache[oldest_key]
            
            return relationships
            
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
            # Use storage method if available
            if hasattr(self.storage, 'get_outgoing_relationships'):
                return self.storage.get_outgoing_relationships(node_id)
            else:
                # Fallback implementation
                return self._get_relationships_by_direction(node_id, 'outgoing')
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
            # Use storage method if available
            if hasattr(self.storage, 'get_incoming_relationships'):
                return self.storage.get_incoming_relationships(node_id)
            else:
                # Fallback implementation
                return self._get_relationships_by_direction(node_id, 'incoming')
        except Exception as e:
            self.logger.error(f"Failed to get incoming relationships for {node_id}: {e}")
            return []
    
    def find_shortest_path(self, from_node_id: str, to_node_id: str, max_hops: int = 5) -> List[str]:
        """
        Find shortest path between two nodes using BFS optimization.
        
        Args:
            from_node_id: Starting node ID
            to_node_id: Target node ID
            max_hops: Maximum number of hops to search
            
        Returns:
            List of node IDs representing the path, empty if no path found
        """
        if from_node_id == to_node_id:
            return [from_node_id]
        
        visited = set()
        queue = deque([(from_node_id, [from_node_id])])
        
        for hop in range(max_hops):
            if not queue:
                break
                
            level_size = len(queue)
            for _ in range(level_size):
                current_node, path = queue.popleft()
                
                if current_node in visited:
                    continue
                visited.add(current_node)
                
                # Get neighbors efficiently
                neighbors = self._get_neighbor_node_ids(current_node)
                
                for neighbor_id in neighbors:
                    if neighbor_id == to_node_id:
                        return path + [neighbor_id]
                    
                    if neighbor_id not in visited:
                        queue.append((neighbor_id, path + [neighbor_id]))
        
        return []  # No path found
    
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
            if hasattr(self.storage, 'find_neighbors'):
                # Use async storage method (simplified for sync interface)
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                neighbors = loop.run_until_complete(
                    self.storage.find_neighbors(node_id, relation_types[0] if relation_types else None)
                )
                return [n['id'] for n in neighbors if 'id' in n]
            else:
                return self._get_neighbor_node_ids(node_id)
        except Exception as e:
            self.logger.error(f"Failed to get neighbors for {node_id}: {e}")
            return []
    
    def _get_direct_relationships(self, node_id: str) -> List[Dict[str, Any]]:
        """Get direct relationships for a node."""
        relationships = []
        
        # Get outgoing relationships
        outgoing = self.get_outgoing_relationships(node_id)
        for rel in outgoing:
            relationships.append({
                'id': rel.edge_id,
                'from_id': rel.from_id,
                'to_id': rel.to_id,
                'relation_type': rel.relation_type,
                'direction': 'outgoing',
                'confidence_score': rel.confidence_score,
                'timestamp': rel.timestamp
            })
        
        # Get incoming relationships
        incoming = self.get_incoming_relationships(node_id)
        for rel in incoming:
            relationships.append({
                'id': rel.edge_id,
                'from_id': rel.from_id,
                'to_id': rel.to_id,
                'relation_type': rel.relation_type,
                'direction': 'incoming',
                'confidence_score': rel.confidence_score,
                'timestamp': rel.timestamp
            })
        
        return relationships
    
    def _get_multi_hop_relationships(self, node_id: str, max_depth: int) -> List[Dict[str, Any]]:
        """Get multi-hop relationships using BFS traversal."""
        all_relationships = []
        visited_nodes = set()
        current_level = {node_id}
        
        for depth in range(max_depth):
            if not current_level:
                break
                
            next_level = set()
            
            for current_node in current_level:
                if current_node in visited_nodes:
                    continue
                
                visited_nodes.add(current_node)
                node_relationships = self._get_direct_relationships(current_node)
                
                for rel in node_relationships:
                    rel['hop_distance'] = depth + 1
                    all_relationships.append(rel)
                    
                    # Add connected nodes to next level
                    if rel['direction'] == 'outgoing':
                        next_level.add(rel['to_id'])
                    else:
                        next_level.add(rel['from_id'])
            
            current_level = next_level
        
        return all_relationships
    
    def _get_relationships_by_direction(self, node_id: str, direction: str) -> List[Relationship]:
        """Fallback method to get relationships by direction."""
        # This would need to be implemented based on the specific storage backend
        # For now, return empty list
        self.logger.warning(f"Fallback relationship query for {node_id} ({direction}) - not implemented")
        return []
    
    def _get_neighbor_node_ids(self, node_id: str) -> List[str]:
        """Get neighbor node IDs efficiently."""
        neighbors = []
        
        try:
            # Get both directions
            outgoing_rels = self.get_outgoing_relationships(node_id)
            incoming_rels = self.get_incoming_relationships(node_id)
            
            for rel in outgoing_rels:
                neighbors.append(rel.to_id)
            
            for rel in incoming_rels:
                neighbors.append(rel.from_id)
                
        except Exception as e:
            self.logger.error(f"Failed to get neighbors for {node_id}: {e}")
        
        return list(set(neighbors))  # Remove duplicates
    
    def _fallback_content_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Fallback content search implementation."""
        # This would need a proper implementation based on storage backend
        self.logger.warning(f"Fallback content search for '{query}' - not implemented")
        return []
    
    def clear_caches(self):
        """Clear traversal caches."""
        self._relationship_cache.clear()
        self._content_index_cache.clear()
        self.logger.info("Graph traversal caches cleared")
    
    def get_traversal_statistics(self) -> Dict[str, int]:
        """Get traversal performance statistics."""
        return {
            'relationship_cache_entries': len(self._relationship_cache),
            'content_cache_entries': len(self._content_index_cache)
        } 