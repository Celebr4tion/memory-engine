"""
JSON file storage implementation for lightweight graph storage.

This module provides a simple file-based implementation of the GraphStorageInterface
using JSON files for storage. Ideal for development, testing, and small deployments.
"""
import json
import os
import uuid
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import asyncio

from memory_core.storage.interfaces.graph_storage_interface import GraphStorageInterface
from memory_core.model.knowledge_node import KnowledgeNode
from memory_core.model.relationship import Relationship


class JsonFileStorage(GraphStorageInterface):
    """
    JSON file-based implementation of the GraphStorageInterface.
    
    This implementation stores graph data in JSON files on the local filesystem.
    It's designed for lightweight deployments, development, and testing scenarios.
    """

    def __init__(self, directory: str = "./data/graph", pretty_print: bool = True):
        """
        Initialize JsonFileStorage with directory and formatting options.
        
        Args:
            directory: Directory path to store JSON files
            pretty_print: Whether to format JSON files for readability
        """
        self.directory = Path(directory)
        self.pretty_print = pretty_print
        self.logger = logging.getLogger(__name__)
        
        # File paths
        self.nodes_file = self.directory / "nodes.json"
        self.edges_file = self.directory / "edges.json"
        self.index_file = self.directory / "indexes.json"
        
        # In-memory data structures for performance
        self._nodes = {}
        self._edges = {}
        self._content_index = {}  # content keyword -> set of node_ids
        self._node_edges = {}  # node_id -> {'outgoing': set, 'incoming': set}
        
        # Performance caches
        self._relationship_cache = {}
        self._traversal_cache = {}
        
        # Connection state
        self._connected = False

    # Connection Management
    async def connect(self) -> None:
        """Establish connection to the JSON file storage."""
        if self._connected:
            return
        
        try:
            # Create directory if it doesn't exist
            self.directory.mkdir(parents=True, exist_ok=True)
            
            # Load existing data
            await self._load_data()
            
            self._connected = True
            self.logger.info(f"Connected to JSON file storage at {self.directory}")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to JSON file storage: {e}")
            raise

    async def close(self) -> None:
        """Close connection to the JSON file storage."""
        if not self._connected:
            return
        
        try:
            # Save data before closing
            await self._save_data()
            self._connected = False
            self.logger.info("Disconnected from JSON file storage")
            
        except Exception as e:
            self.logger.error(f"Error closing JSON file storage: {e}")

    async def test_connection(self) -> bool:
        """Test if the connection to storage backend is working."""
        try:
            # Test directory access
            test_file = self.directory / ".connection_test"
            self.directory.mkdir(parents=True, exist_ok=True)
            test_file.write_text("test")
            test_file.unlink()
            return True
        except Exception as e:
            self.logger.error(f"JSON file storage connection test failed: {e}")
            return False

    async def is_available(self) -> bool:
        """Check if the storage backend is available."""
        return await self.test_connection()

    def connect_sync(self) -> None:
        """Establish connection to the storage backend (synchronous)."""
        return self._run_async_in_sync_context(self.connect())

    def close_sync(self) -> None:
        """Close connection to the storage backend (synchronous)."""
        return self._run_async_in_sync_context(self.close())

    def test_connection_sync(self) -> bool:
        """Test if the connection to storage backend is working (synchronous)."""
        return self._run_async_in_sync_context(self.test_connection())

    def is_available_sync(self) -> bool:
        """Check if the storage backend is available (synchronous)."""
        return self.test_connection_sync()

    # Raw Node Operations
    async def create_node(self, node_data: Dict[str, Any]) -> str:
        """Create a new node in the graph."""
        await self.connect()
        
        # Validate required fields
        required_fields = ['content', 'source', 'creation_timestamp', 'rating_richness', 
                          'rating_truthfulness', 'rating_stability']
        missing_fields = [field for field in required_fields if field not in node_data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        node_id = str(uuid.uuid4())
        node_data_with_id = {**node_data, 'node_id': node_id}
        
        # Store in memory
        self._nodes[node_id] = node_data_with_id
        
        # Update content index
        self._update_content_index(node_id, node_data['content'])
        
        # Initialize node edges
        self._node_edges[node_id] = {'outgoing': set(), 'incoming': set()}
        
        # Save to disk
        await self._save_data()
        
        self.logger.debug(f"Created node {node_id}")
        return node_id

    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a node by its ID."""
        await self.connect()
        
        node_data = self._nodes.get(node_id)
        if node_data:
            return dict(node_data)  # Return a copy
        return None

    async def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """Update properties of an existing node."""
        await self.connect()
        
        if node_id not in self._nodes:
            return False
        
        # Update content index if content changed
        old_content = self._nodes[node_id].get('content', '')
        new_content = properties.get('content', old_content)
        if old_content != new_content:
            self._remove_from_content_index(node_id, old_content)
            self._update_content_index(node_id, new_content)
        
        # Update node data
        self._nodes[node_id].update(properties)
        
        # Save to disk
        await self._save_data()
        
        self.logger.debug(f"Updated node {node_id}")
        return True

    async def delete_node(self, node_id: str) -> bool:
        """Delete a node from the graph."""
        await self.connect()
        
        if node_id not in self._nodes:
            return False
        
        # Remove from content index
        content = self._nodes[node_id].get('content', '')
        self._remove_from_content_index(node_id, content)
        
        # Delete all connected edges
        if node_id in self._node_edges:
            edges_to_delete = list(self._node_edges[node_id]['outgoing']) + list(self._node_edges[node_id]['incoming'])
            for edge_id in edges_to_delete:
                await self.delete_edge(edge_id)
            del self._node_edges[node_id]
        
        # Delete node
        del self._nodes[node_id]
        
        # Save to disk
        await self._save_data()
        
        self.logger.debug(f"Deleted node {node_id}")
        return True

    # Raw Edge Operations
    async def create_edge(self, from_node_id: str, to_node_id: str, 
                         relation_type: str, properties: Dict[str, Any]) -> str:
        """Create a new edge between two nodes."""
        await self.connect()
        
        # Validate that both nodes exist
        if from_node_id not in self._nodes or to_node_id not in self._nodes:
            raise ValueError("Source or target node does not exist")
        
        # Validate required fields
        required_fields = ['timestamp', 'confidence_score', 'version']
        missing_fields = [field for field in required_fields if field not in properties]
        if missing_fields:
            raise ValueError(f"Missing required edge fields: {missing_fields}")
        
        edge_id = str(uuid.uuid4())
        edge_data = {
            'edge_id': edge_id,
            'from_id': from_node_id,
            'to_id': to_node_id,
            'relation_type': relation_type,
            **properties
        }
        
        # Store edge
        self._edges[edge_id] = edge_data
        
        # Update node edge tracking
        if from_node_id not in self._node_edges:
            self._node_edges[from_node_id] = {'outgoing': set(), 'incoming': set()}
        if to_node_id not in self._node_edges:
            self._node_edges[to_node_id] = {'outgoing': set(), 'incoming': set()}
        
        self._node_edges[from_node_id]['outgoing'].add(edge_id)
        self._node_edges[to_node_id]['incoming'].add(edge_id)
        
        # Save to disk
        await self._save_data()
        
        self.logger.debug(f"Created edge {edge_id} from {from_node_id} to {to_node_id}")
        return edge_id

    async def get_edge(self, edge_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an edge by its ID."""
        await self.connect()
        
        edge_data = self._edges.get(edge_id)
        if edge_data:
            return dict(edge_data)  # Return a copy
        return None

    async def update_edge(self, edge_id: str, properties: Dict[str, Any]) -> bool:
        """Update properties of an existing edge."""
        await self.connect()
        
        if edge_id not in self._edges:
            return False
        
        # Update edge data
        self._edges[edge_id].update(properties)
        
        # Save to disk
        await self._save_data()
        
        self.logger.debug(f"Updated edge {edge_id}")
        return True

    async def delete_edge(self, edge_id: str) -> bool:
        """Delete an edge from the graph."""
        await self.connect()
        
        if edge_id not in self._edges:
            return False
        
        edge_data = self._edges[edge_id]
        from_node_id = edge_data['from_id']
        to_node_id = edge_data['to_id']
        
        # Update node edge tracking
        if from_node_id in self._node_edges:
            self._node_edges[from_node_id]['outgoing'].discard(edge_id)
        if to_node_id in self._node_edges:
            self._node_edges[to_node_id]['incoming'].discard(edge_id)
        
        # Delete edge
        del self._edges[edge_id]
        
        # Save to disk
        await self._save_data()
        
        self.logger.debug(f"Deleted edge {edge_id}")
        return True

    # Graph Traversal Operations
    async def find_neighbors(self, node_id: str, 
                           relation_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Find neighboring nodes of a given node."""
        await self.connect()
        
        if node_id not in self._node_edges:
            return []
        
        neighbor_edges = []
        all_edge_ids = (self._node_edges[node_id]['outgoing'] | 
                       self._node_edges[node_id]['incoming'])
        
        for edge_id in all_edge_ids:
            edge_data = self._edges.get(edge_id)
            if edge_data and (relation_type is None or edge_data['relation_type'] == relation_type):
                neighbor_edges.append(dict(edge_data))
        
        return neighbor_edges

    async def get_outgoing_relationships(self, node_id: str) -> List[Dict[str, Any]]:
        """Get all outgoing relationships from a node."""
        await self.connect()
        
        if node_id not in self._node_edges:
            return []
        
        relationships = []
        for edge_id in self._node_edges[node_id]['outgoing']:
            edge_data = self._edges.get(edge_id)
            if edge_data:
                relationships.append(dict(edge_data))
        
        return relationships

    async def get_incoming_relationships(self, node_id: str) -> List[Dict[str, Any]]:
        """Get all incoming relationships to a node."""
        await self.connect()
        
        if node_id not in self._node_edges:
            return []
        
        relationships = []
        for edge_id in self._node_edges[node_id]['incoming']:
            edge_data = self._edges.get(edge_id)
            if edge_data:
                relationships.append(dict(edge_data))
        
        return relationships

    async def find_shortest_path(self, from_node_id: str, to_node_id: str,
                               max_hops: int = 5) -> List[str]:
        """Find the shortest path between two nodes using BFS."""
        if from_node_id == to_node_id:
            return [from_node_id]
        
        await self.connect()
        
        if from_node_id not in self._nodes or to_node_id not in self._nodes:
            return []
        
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
                
                # Get neighbors
                neighbors = await self._get_neighbor_node_ids(current_node)
                
                for neighbor_id in neighbors:
                    if neighbor_id == to_node_id:
                        return path + [neighbor_id]
                    
                    if neighbor_id not in visited:
                        queue.append((neighbor_id, path + [neighbor_id]))
        
        return []  # No path found

    # Query Operations
    async def find_nodes_by_content(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Find nodes by content search."""
        await self.connect()
        
        query_lower = query.lower()
        matching_nodes = []
        
        # Simple text search implementation
        for node_id, node_data in self._nodes.items():
            content = node_data.get('content', '').lower()
            if query_lower in content:
                matching_nodes.append(dict(node_data))
                if len(matching_nodes) >= limit:
                    break
        
        return matching_nodes

    async def get_relationships_for_node(self, node_id: str, 
                                       max_depth: int = 1) -> List[Dict[str, Any]]:
        """Get relationships for a node up to a certain depth."""
        if max_depth == 1:
            outgoing = await self.get_outgoing_relationships(node_id)
            incoming = await self.get_incoming_relationships(node_id)
            return outgoing + incoming
        else:
            # Multi-hop traversal - simplified implementation
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
                    node_relationships = await self.get_relationships_for_node(current_node, 1)
                    
                    for rel in node_relationships:
                        rel['hop_distance'] = depth + 1
                        all_relationships.append(rel)
                        
                        # Add connected nodes to next level
                        if rel['from_id'] == current_node:
                            next_level.add(rel['to_id'])
                        else:
                            next_level.add(rel['from_id'])
                
                current_level = next_level
            
            return all_relationships

    # Domain Model Operations
    async def save_knowledge_node(self, node: KnowledgeNode) -> str:
        """Save a KnowledgeNode to storage."""
        node_data = {
            'content': node.content,
            'source': node.source,
            'creation_timestamp': node.creation_timestamp,
            'rating_richness': node.rating_richness,
            'rating_truthfulness': node.rating_truthfulness,
            'rating_stability': node.rating_stability
        }
        
        if node.node_id is None:
            node_id = await self.create_node(node_data)
            node.node_id = node_id
            return node_id
        else:
            await self.update_node(node.node_id, node_data)
            return node.node_id

    async def get_knowledge_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """Retrieve a KnowledgeNode by ID."""
        node_data = await self.get_node(node_id)
        if not node_data:
            return None
        
        return KnowledgeNode(
            content=node_data['content'],
            source=node_data['source'],
            creation_timestamp=node_data['creation_timestamp'],
            rating_richness=node_data['rating_richness'],
            rating_truthfulness=node_data['rating_truthfulness'],
            rating_stability=node_data['rating_stability'],
            node_id=node_id
        )

    async def delete_knowledge_node(self, node_id: str) -> bool:
        """Delete a KnowledgeNode from storage."""
        return await self.delete_node(node_id)

    async def save_relationship(self, relationship: Relationship) -> str:
        """Save a Relationship to storage."""
        edge_metadata = {
            'timestamp': relationship.timestamp,
            'confidence_score': relationship.confidence_score,
            'version': relationship.version
        }
        
        if relationship.edge_id is None:
            edge_id = await self.create_edge(
                relationship.from_id,
                relationship.to_id,
                relationship.relation_type,
                edge_metadata
            )
            relationship.edge_id = edge_id
            return edge_id
        else:
            await self.update_edge(relationship.edge_id, edge_metadata)
            return relationship.edge_id

    async def get_relationship(self, edge_id: str) -> Optional[Relationship]:
        """Retrieve a Relationship by ID."""
        edge_data = await self.get_edge(edge_id)
        if not edge_data:
            return None
        
        return Relationship(
            from_id=edge_data['from_id'],
            to_id=edge_data['to_id'],
            relation_type=edge_data['relation_type'],
            timestamp=edge_data['timestamp'],
            confidence_score=edge_data['confidence_score'],
            version=edge_data['version'],
            edge_id=edge_id
        )

    async def delete_relationship(self, edge_id: str) -> bool:
        """Delete a Relationship from storage."""
        return await self.delete_edge(edge_id)

    # Utility Operations
    async def clear_all_data(self) -> bool:
        """Clear all data from the storage backend."""
        await self.connect()
        
        try:
            self._nodes.clear()
            self._edges.clear()
            self._content_index.clear()
            self._node_edges.clear()
            
            # Save empty data
            await self._save_data()
            
            self.logger.info("All data cleared from JSON file storage")
            return True
        except Exception as e:
            self.logger.error(f"Error clearing data: {e}")
            return False

    async def merge_nodes(self, node_id1: str, node_id2: str) -> bool:
        """Merge two nodes into one."""
        await self.connect()
        
        if node_id1 not in self._nodes or node_id2 not in self._nodes:
            return False
        
        try:
            # Get all edges connected to node2
            node2_outgoing = list(self._node_edges.get(node_id2, {}).get('outgoing', set()))
            node2_incoming = list(self._node_edges.get(node_id2, {}).get('incoming', set()))
            
            # Recreate edges to point to node1
            for edge_id in node2_outgoing:
                edge_data = self._edges.get(edge_id)
                if edge_data and edge_data['to_id'] != node_id1:  # Avoid self-loops
                    # Create new edge from node1
                    properties = {k: v for k, v in edge_data.items() 
                                 if k not in ['edge_id', 'from_id', 'to_id', 'relation_type']}
                    await self.create_edge(node_id1, edge_data['to_id'], 
                                         edge_data['relation_type'], properties)
                # Delete old edge
                await self.delete_edge(edge_id)
            
            for edge_id in node2_incoming:
                edge_data = self._edges.get(edge_id)
                if edge_data and edge_data['from_id'] != node_id1:  # Avoid self-loops
                    # Create new edge to node1
                    properties = {k: v for k, v in edge_data.items() 
                                 if k not in ['edge_id', 'from_id', 'to_id', 'relation_type']}
                    await self.create_edge(edge_data['from_id'], node_id1, 
                                         edge_data['relation_type'], properties)
                # Delete old edge
                await self.delete_edge(edge_id)
            
            # Delete node2
            await self.delete_node(node_id2)
            
            self.logger.info(f"Successfully merged nodes {node_id1} and {node_id2}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error merging nodes: {e}")
            return False

    # Performance and Caching
    def clear_caches(self) -> None:
        """Clear any internal caches."""
        self._relationship_cache.clear()
        self._traversal_cache.clear()
        self.logger.debug("JSON file storage caches cleared")

    def get_traversal_statistics(self) -> Dict[str, int]:
        """Get traversal statistics for performance monitoring."""
        return {
            'total_nodes': len(self._nodes),
            'total_edges': len(self._edges),
            'content_index_entries': len(self._content_index),
            'relationship_cache_entries': len(self._relationship_cache),
            'traversal_cache_entries': len(self._traversal_cache)
        }

    # Synchronous versions for backwards compatibility
    def create_node_sync(self, node_data: Dict[str, Any]) -> str:
        """Create a new node in the graph (synchronous)."""
        return self._run_async_in_sync_context(self.create_node(node_data))

    def get_node_sync(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a node by its ID (synchronous)."""
        return self._run_async_in_sync_context(self.get_node(node_id))

    def save_knowledge_node_sync(self, node: KnowledgeNode) -> str:
        """Save a KnowledgeNode to storage (synchronous)."""
        return self._run_async_in_sync_context(self.save_knowledge_node(node))

    def get_knowledge_node_sync(self, node_id: str) -> Optional[KnowledgeNode]:
        """Retrieve a KnowledgeNode by ID (synchronous)."""
        return self._run_async_in_sync_context(self.get_knowledge_node(node_id))

    # Private helper methods
    async def _load_data(self):
        """Load data from JSON files."""
        try:
            # Load nodes
            if self.nodes_file.exists():
                with open(self.nodes_file, 'r') as f:
                    self._nodes = json.load(f)
            
            # Load edges
            if self.edges_file.exists():
                with open(self.edges_file, 'r') as f:
                    self._edges = json.load(f)
            
            # Load indexes
            if self.index_file.exists():
                with open(self.index_file, 'r') as f:
                    index_data = json.load(f)
                    # Convert sets back from lists
                    self._content_index = {k: set(v) for k, v in index_data.get('content_index', {}).items()}
                    node_edges_data = index_data.get('node_edges', {})
                    self._node_edges = {
                        k: {'outgoing': set(v['outgoing']), 'incoming': set(v['incoming'])}
                        for k, v in node_edges_data.items()
                    }
            
            # Rebuild indexes if missing
            if not self._content_index:
                self._rebuild_content_index()
            if not self._node_edges:
                self._rebuild_node_edges_index()
            
            self.logger.info(f"Loaded {len(self._nodes)} nodes and {len(self._edges)} edges")
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            # Initialize empty structures
            self._nodes = {}
            self._edges = {}
            self._content_index = {}
            self._node_edges = {}

    async def _save_data(self):
        """Save data to JSON files."""
        try:
            # Save nodes
            with open(self.nodes_file, 'w') as f:
                if self.pretty_print:
                    json.dump(self._nodes, f, indent=2, default=str)
                else:
                    json.dump(self._nodes, f, default=str)
            
            # Save edges
            with open(self.edges_file, 'w') as f:
                if self.pretty_print:
                    json.dump(self._edges, f, indent=2, default=str)
                else:
                    json.dump(self._edges, f, default=str)
            
            # Save indexes (convert sets to lists for JSON serialization)
            index_data = {
                'content_index': {k: list(v) for k, v in self._content_index.items()},
                'node_edges': {
                    k: {'outgoing': list(v['outgoing']), 'incoming': list(v['incoming'])}
                    for k, v in self._node_edges.items()
                }
            }
            with open(self.index_file, 'w') as f:
                if self.pretty_print:
                    json.dump(index_data, f, indent=2)
                else:
                    json.dump(index_data, f)
            
        except Exception as e:
            self.logger.error(f"Error saving data: {e}")
            raise

    def _update_content_index(self, node_id: str, content: str):
        """Update the content search index."""
        # Simple word-based indexing
        words = content.lower().split()
        for word in words:
            if len(word) > 2:  # Skip very short words
                if word not in self._content_index:
                    self._content_index[word] = set()
                self._content_index[word].add(node_id)

    def _remove_from_content_index(self, node_id: str, content: str):
        """Remove node from content search index."""
        words = content.lower().split()
        for word in words:
            if word in self._content_index:
                self._content_index[word].discard(node_id)
                if not self._content_index[word]:
                    del self._content_index[word]

    def _rebuild_content_index(self):
        """Rebuild the content search index."""
        self._content_index.clear()
        for node_id, node_data in self._nodes.items():
            content = node_data.get('content', '')
            self._update_content_index(node_id, content)

    def _rebuild_node_edges_index(self):
        """Rebuild the node edges index."""
        self._node_edges.clear()
        
        # Initialize all nodes
        for node_id in self._nodes:
            self._node_edges[node_id] = {'outgoing': set(), 'incoming': set()}
        
        # Add edges
        for edge_id, edge_data in self._edges.items():
            from_id = edge_data['from_id']
            to_id = edge_data['to_id']
            
            if from_id not in self._node_edges:
                self._node_edges[from_id] = {'outgoing': set(), 'incoming': set()}
            if to_id not in self._node_edges:
                self._node_edges[to_id] = {'outgoing': set(), 'incoming': set()}
            
            self._node_edges[from_id]['outgoing'].add(edge_id)
            self._node_edges[to_id]['incoming'].add(edge_id)

    async def _get_neighbor_node_ids(self, node_id: str) -> List[str]:
        """Get neighbor node IDs efficiently."""
        if node_id not in self._node_edges:
            return []
        
        neighbors = set()
        
        # Add outgoing neighbors
        for edge_id in self._node_edges[node_id]['outgoing']:
            edge_data = self._edges.get(edge_id)
            if edge_data:
                neighbors.add(edge_data['to_id'])
        
        # Add incoming neighbors
        for edge_id in self._node_edges[node_id]['incoming']:
            edge_data = self._edges.get(edge_id)
            if edge_data:
                neighbors.add(edge_data['from_id'])
        
        return list(neighbors)

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
            with ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=timeout)
        except Exception as e:
            self.logger.error(f"Error running async in sync context: {e}")
            raise