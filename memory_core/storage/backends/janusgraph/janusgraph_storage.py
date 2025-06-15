"""
JanusGraph storage implementation for managing knowledge graph operations.

This module provides a JanusGraph-specific implementation of the GraphStorageInterface,
serving as the production-grade graph storage backend.
"""

import logging
import uuid
import asyncio
import socket
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor

from gremlin_python.driver import client, protocol, serializer
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
from gremlin_python.process.anonymous_traversal import traversal
from gremlin_python.process.graph_traversal import __
from gremlin_python.structure.graph import Graph
from gremlin_python.process.strategies import *
from gremlin_python.process.traversal import T

from memory_core.storage.interfaces.graph_storage_interface import GraphStorageInterface
from memory_core.model.knowledge_node import KnowledgeNode
from memory_core.model.relationship import Relationship
from memory_core.config import get_config


class JanusGraphStorage(GraphStorageInterface):
    """
    JanusGraph implementation of the GraphStorageInterface.

    This class provides methods to manage nodes and edges in a JanusGraph database,
    implementing the full GraphStorageInterface for consistent behavior across backends.
    """

    def __init__(self, host=None, port=None, traversal_source="g"):
        """
        Initialize JanusGraphStorage with connection details.

        Args:
            host: The hostname or IP address of the JanusGraph server (optional, uses config if not provided)
            port: The port number of the JanusGraph server (optional, uses config if not provided)
            traversal_source: The traversal source to use for connecting to JanusGraph
        """
        self.config = get_config()
        self.host = host or self.config.config.janusgraph.host
        self.port = port or self.config.config.janusgraph.port
        self.traversal_source = traversal_source
        self._client = None
        self.g = None
        self._remote_connection = None
        self.logger = logging.getLogger(__name__)

        # Performance caches
        self._relationship_cache = {}
        self._content_index_cache = {}

    # Connection Management
    async def connect(self) -> None:
        """Establish connection to the JanusGraph database."""
        if self.g is not None:
            self.logger.info("Already connected to JanusGraph.")
            return

        try:
            connection_url = self.config.config.janusgraph.connection_url
            self.logger.info(f"Connecting to JanusGraph at {connection_url}...")

            self._remote_connection = DriverRemoteConnection(connection_url, self.traversal_source)
            self.g = traversal().withRemote(self._remote_connection)

            self.logger.info(f"Connected to JanusGraph at {connection_url}")

        except Exception as e:
            self.logger.error(f"Failed to connect to JanusGraph: {e}")
            self.g = None
            self._remote_connection = None
            raise

    async def close(self) -> None:
        """Close connection to the JanusGraph database."""
        if self._remote_connection:
            try:
                if hasattr(self._remote_connection, "close"):
                    if asyncio.iscoroutinefunction(self._remote_connection.close):
                        await self._remote_connection.close()
                    else:
                        self._remote_connection.close()

                self._remote_connection = None
                self.g = None
                self.logger.info("Disconnected from JanusGraph.")
            except Exception as e:
                self.logger.error(f"Error closing JanusGraph connection: {e}")
                self._remote_connection = None
                self.g = None

    async def test_connection(self) -> bool:
        """Test if the connection to storage backend is working."""
        client_obj = None
        try:
            host = self.host
            port = self.port

            # First try a simple socket connection
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((host, port))
            sock.close()

            # Initialize and test the gremlin client
            client_obj = client.Client(
                f"ws://{host}:{port}/gremlin",
                "g",
                connection_timeout=5,
                message_serializer=serializer.GraphSONMessageSerializer(),
            )

            result = await client_obj.submit("g.V().count()")
            vertices_count = result.all().result()
            self.logger.info(
                f"JanusGraph connection test successful. Found {vertices_count} vertices."
            )
            return True

        except Exception as e:
            self.logger.error(f"JanusGraph connection test failed: {e}")
            return False
        finally:
            if client_obj:
                try:
                    await client_obj.close()
                except Exception:
                    pass

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
        try:
            host = self.host
            port = self.port

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((host, port))

            # Try WebSocket handshake
            handshake = (
                f"GET /gremlin HTTP/1.1\\r\\n"
                f"Host: {host}:{port}\\r\\n"
                f"Upgrade: websocket\\r\\n"
                f"Connection: Upgrade\\r\\n"
                f"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\\r\\n"
                f"Sec-WebSocket-Version: 13\\r\\n\\r\\n"
            )
            sock.sendall(handshake.encode())
            response = sock.recv(1024)
            sock.close()

            success = b"HTTP/1.1 101" in response or b"Upgrade: websocket" in response
            if success:
                self.logger.info("JanusGraph WebSocket connection test successful")
            return success

        except Exception as e:
            self.logger.error(f"JanusGraph connection test failed: {e}")
            return False

    def is_available_sync(self) -> bool:
        """Check if the storage backend is available (synchronous)."""
        return self.test_connection_sync()

    # Raw Node Operations
    async def create_node(self, node_data: Dict[str, Any]) -> str:
        """Create a new node in the graph."""
        await self.connect()
        if not self.g:
            raise ConnectionError("Not connected to JanusGraph")

        # Validate required fields
        required_fields = [
            "content",
            "source",
            "creation_timestamp",
            "rating_richness",
            "rating_truthfulness",
            "rating_stability",
        ]
        missing_fields = [field for field in required_fields if field not in node_data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        try:
            node_id = str(uuid.uuid4())
            vertex = self.g.add_v("KnowledgeNode")
            vertex = vertex.property("node_id", node_id)
            for key, value in node_data.items():
                vertex = vertex.property(key, value)
            new_vertex = vertex.next()
            self.logger.info(f"Node created with graph ID: {new_vertex.id}, node_id: {node_id}")
            return node_id
        except Exception as e:
            self.logger.error(f"Error creating node: {e}")
            raise

    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a node by its ID."""
        await self.connect()
        if not self.g:
            raise ConnectionError("Not connected to JanusGraph")

        try:
            node_data_list = self.g.V().has("node_id", node_id).value_map(True).to_list()
            if not node_data_list:
                return None
            node_data = node_data_list[0]

            # Process node data
            formatted_data = {
                k: v[0] if isinstance(v, list) and len(v) == 1 else v
                for k, v in node_data.items()
                if k not in ["id", "label", T.id, T.label]
            }
            formatted_data["id"] = str(node_data.get(T.id))
            formatted_data["node_id"] = node_data.get("node_id", [node_id])[0]
            formatted_data["label"] = str(node_data.get(T.label))
            return formatted_data

        except Exception as e:
            self.logger.error(f"Error getting node {node_id}: {e}")
            raise

    async def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """Update properties of an existing node."""
        await self.connect()
        if not self.g:
            raise ConnectionError("Not connected to JanusGraph")

        try:
            vertex = self.g.V().has("node_id", node_id)
            for key, value in properties.items():
                vertex = vertex.property(key, value)
            vertex.iterate()
            self.logger.info(f"Node {node_id} updated.")
            return True
        except Exception as e:
            self.logger.error(f"Error updating node {node_id}: {e}")
            return False

    async def delete_node(self, node_id: str) -> bool:
        """Delete a node from the graph."""
        await self.connect()
        if not self.g:
            raise ConnectionError("Not connected to JanusGraph")

        try:
            self.g.V().has("node_id", node_id).drop().iterate()
            self.logger.info(f"Node {node_id} deleted.")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting node {node_id}: {e}")
            return False

    # Raw Edge Operations
    async def create_edge(
        self, from_node_id: str, to_node_id: str, relation_type: str, properties: Dict[str, Any]
    ) -> str:
        """Create a new edge between two nodes."""
        await self.connect()
        if not self.g:
            raise ConnectionError("Not connected to JanusGraph")

        # Validate required fields
        required_fields = ["timestamp", "confidence_score", "version"]
        missing_fields = [field for field in required_fields if field not in properties]
        if missing_fields:
            raise ValueError(f"Missing required edge fields: {missing_fields}")

        try:
            edge_id = str(uuid.uuid4())

            from_vertex = self.g.V().has("node_id", from_node_id)
            to_vertex = self.g.V().has("node_id", to_node_id)

            edge_traversal = self.g.add_e(relation_type).from_(from_vertex).to(to_vertex)
            edge_traversal = edge_traversal.property("edge_id", edge_id)

            for key, value in properties.items():
                edge_traversal = edge_traversal.property(key, value)

            new_edge = edge_traversal.next()
            self.logger.info(f"Edge created with graph ID: {new_edge.id}, edge_id: {edge_id}")
            return edge_id
        except Exception as e:
            self.logger.error(f"Error creating edge from {from_node_id} to {to_node_id}: {e}")
            raise

    async def get_edge(self, edge_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an edge by its ID."""
        await self.connect()
        if not self.g:
            raise ConnectionError("Not connected to JanusGraph")

        try:
            edge_data_list = self.g.E().has("edge_id", edge_id).value_map(True).to_list()
            if not edge_data_list:
                return None
            edge_data = edge_data_list[0]

            # Get the source and target node IDs
            out_node_id_list = (
                self.g.E().has("edge_id", edge_id).out_v().values("node_id").to_list()
            )
            in_node_id_list = self.g.E().has("edge_id", edge_id).in_v().values("node_id").to_list()

            if not out_node_id_list or not in_node_id_list:
                self.logger.warning(f"Could not find source or target node for edge {edge_id}")
                return None

            # Process edge data
            formatted_data = {
                k: v[0] if isinstance(v, list) and len(v) == 1 else v
                for k, v in edge_data.items()
                if k not in ["id", "label", T.id, T.label]
            }
            formatted_data["id"] = str(edge_data.get(T.id))
            formatted_data["edge_id"] = edge_data.get("edge_id", [edge_id])[0]
            formatted_data["relation_type"] = str(edge_data.get(T.label))
            formatted_data["from_id"] = str(out_node_id_list[0])
            formatted_data["to_id"] = str(in_node_id_list[0])
            return formatted_data

        except Exception as e:
            self.logger.error(f"Error getting edge {edge_id}: {e}")
            raise

    async def update_edge(self, edge_id: str, properties: Dict[str, Any]) -> bool:
        """Update properties of an existing edge."""
        await self.connect()
        if not self.g:
            raise ConnectionError("Not connected to JanusGraph")

        try:
            edge_traversal = self.g.E().has("edge_id", edge_id)
            for key, value in properties.items():
                edge_traversal = edge_traversal.property(key, value)
            edge_traversal.iterate()
            self.logger.info(f"Edge {edge_id} updated.")
            return True
        except Exception as e:
            self.logger.error(f"Error updating edge {edge_id}: {e}")
            return False

    async def delete_edge(self, edge_id: str) -> bool:
        """Delete an edge from the graph."""
        await self.connect()
        if not self.g:
            raise ConnectionError("Not connected to JanusGraph")

        try:
            self.g.E().has("edge_id", edge_id).drop().iterate()
            self.logger.info(f"Edge {edge_id} deleted.")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting edge {edge_id}: {e}")
            return False

    # Graph Traversal Operations
    async def find_neighbors(
        self, node_id: str, relation_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Find neighboring nodes of a given node."""
        await self.connect()
        if not self.g:
            raise ConnectionError("Not connected to JanusGraph")

        try:
            traversal = self.g.V().has("node_id", node_id).both_e()
            if relation_type:
                traversal = traversal.has_label(relation_type)
            neighbor_edges = traversal.value_map(True).to_list()

            neighbors = []
            for edge_data in neighbor_edges:
                edge_id = edge_data.get("edge_id", [None])[0]
                if edge_id:
                    edge = await self.get_edge(edge_id)
                    if edge:
                        neighbors.append(edge)
            return neighbors
        except Exception as e:
            self.logger.error(f"Error finding neighbors for node {node_id}: {e}")
            raise

    async def get_outgoing_relationships(self, node_id: str) -> List[Dict[str, Any]]:
        """Get all outgoing relationships from a node."""
        await self.connect()
        if not self.g:
            raise ConnectionError("Not connected to JanusGraph")

        try:
            edge_data_list = self.g.V().has("node_id", node_id).out_e().value_map(True).to_list()
            relationships = []

            for edge_data in edge_data_list:
                edge_id = edge_data.get("edge_id", [None])[0]
                if edge_id:
                    edge = await self.get_edge(edge_id)
                    if edge:
                        relationships.append(edge)

            return relationships
        except Exception as e:
            self.logger.error(f"Error getting outgoing relationships for {node_id}: {e}")
            return []

    async def get_incoming_relationships(self, node_id: str) -> List[Dict[str, Any]]:
        """Get all incoming relationships to a node."""
        await self.connect()
        if not self.g:
            raise ConnectionError("Not connected to JanusGraph")

        try:
            edge_data_list = self.g.V().has("node_id", node_id).in_e().value_map(True).to_list()
            relationships = []

            for edge_data in edge_data_list:
                edge_id = edge_data.get("edge_id", [None])[0]
                if edge_id:
                    edge = await self.get_edge(edge_id)
                    if edge:
                        relationships.append(edge)

            return relationships
        except Exception as e:
            self.logger.error(f"Error getting incoming relationships for {node_id}: {e}")
            return []

    async def find_shortest_path(
        self, from_node_id: str, to_node_id: str, max_hops: int = 5
    ) -> List[str]:
        """Find the shortest path between two nodes."""
        if from_node_id == to_node_id:
            return [from_node_id]

        await self.connect()
        if not self.g:
            raise ConnectionError("Not connected to JanusGraph")

        try:
            # Use JanusGraph's path finding capabilities
            path_result = (
                self.g.V()
                .has("node_id", from_node_id)
                .repeat(__.both().simplePath())
                .until(__.has("node_id", to_node_id))
                .limit(1)
                .path()
                .by("node_id")
                .to_list()
            )

            if path_result:
                return path_result[0]
            return []

        except Exception as e:
            self.logger.error(
                f"Error finding shortest path from {from_node_id} to {to_node_id}: {e}"
            )
            return []

    # Query Operations
    async def find_nodes_by_content(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Find nodes by content search."""
        await self.connect()
        if not self.g:
            raise ConnectionError("Not connected to JanusGraph")

        try:
            # Simple text search implementation - could be enhanced with full-text search
            nodes = (
                self.g.V()
                .has("content", __.containing(query))
                .limit(limit)
                .value_map(True)
                .to_list()
            )

            results = []
            for node_data in nodes:
                formatted_data = {
                    k: v[0] if isinstance(v, list) and len(v) == 1 else v
                    for k, v in node_data.items()
                    if k not in ["id", "label", T.id, T.label]
                }
                formatted_data["id"] = str(node_data.get(T.id))
                formatted_data["label"] = str(node_data.get(T.label))
                results.append(formatted_data)

            return results
        except Exception as e:
            self.logger.error(f"Error searching nodes by content '{query}': {e}")
            return []

    async def get_relationships_for_node(
        self, node_id: str, max_depth: int = 1
    ) -> List[Dict[str, Any]]:
        """Get relationships for a node up to a certain depth."""
        if max_depth == 1:
            outgoing = await self.get_outgoing_relationships(node_id)
            incoming = await self.get_incoming_relationships(node_id)
            return outgoing + incoming
        else:
            # Multi-hop traversal would need more complex implementation
            # For now, just return direct relationships
            return await self.get_relationships_for_node(node_id, 1)

    # Domain Model Operations
    async def save_knowledge_node(self, node: KnowledgeNode) -> str:
        """Save a KnowledgeNode to storage."""
        node_data = {
            "content": node.content,
            "source": node.source,
            "creation_timestamp": node.creation_timestamp,
            "rating_richness": node.rating_richness,
            "rating_truthfulness": node.rating_truthfulness,
            "rating_stability": node.rating_stability,
        }

        if node.node_id is None:
            node_id = await self.create_node(node_data)
            node.node_id = node_id  # Update the node object
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
            content=node_data["content"],
            source=node_data["source"],
            creation_timestamp=node_data["creation_timestamp"],
            rating_richness=node_data["rating_richness"],
            rating_truthfulness=node_data["rating_truthfulness"],
            rating_stability=node_data["rating_stability"],
            node_id=node_id,
        )

    async def delete_knowledge_node(self, node_id: str) -> bool:
        """Delete a KnowledgeNode from storage."""
        return await self.delete_node(node_id)

    async def save_relationship(self, relationship: Relationship) -> str:
        """Save a Relationship to storage."""
        edge_metadata = {
            "timestamp": relationship.timestamp,
            "confidence_score": relationship.confidence_score,
            "version": relationship.version,
        }

        if relationship.edge_id is None:
            edge_id = await self.create_edge(
                relationship.from_id, relationship.to_id, relationship.relation_type, edge_metadata
            )
            relationship.edge_id = edge_id  # Update the relationship object
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
            from_id=edge_data["from_id"],
            to_id=edge_data["to_id"],
            relation_type=edge_data["relation_type"],
            timestamp=edge_data["timestamp"],
            confidence_score=edge_data["confidence_score"],
            version=edge_data["version"],
            edge_id=edge_id,
        )

    async def delete_relationship(self, edge_id: str) -> bool:
        """Delete a Relationship from storage."""
        return await self.delete_edge(edge_id)

    # Utility Operations
    async def clear_all_data(self) -> bool:
        """Clear all data from the storage backend."""
        await self.connect()
        if not self.g:
            raise ConnectionError("Not connected to JanusGraph")

        try:
            self.g.V().drop().iterate()
            self.logger.info("All data cleared from JanusGraph.")
            return True
        except Exception as e:
            self.logger.error(f"Error clearing data: {e}")
            raise

    async def merge_nodes(self, node_id1: str, node_id2: str) -> bool:
        """Merge two nodes into one."""
        await self.connect()
        if not self.g:
            raise ConnectionError("Not connected to JanusGraph")

        try:
            # Get data for both nodes
            node1_data = await self.get_node(node_id1)
            node2_data = await self.get_node(node_id2)

            if not node1_data or not node2_data:
                self.logger.error("One or both nodes not found")
                return False

            # Store all edges connected to node2
            neighbors = await self.find_neighbors(node_id2)

            # Create new edges to node1 for all of node2's connections
            for edge in neighbors:
                if edge["from_id"] == node_id1 or edge["to_id"] == node_id1:
                    continue

                properties = {
                    k: v
                    for k, v in edge.items()
                    if k not in ["id", "edge_id", "from_id", "to_id", "relation_type"]
                }

                # Determine direction
                if edge["from_id"] == node_id2:
                    await self.create_edge(
                        node_id1, edge["to_id"], edge["relation_type"], properties
                    )
                else:
                    await self.create_edge(
                        edge["from_id"], node_id1, edge["relation_type"], properties
                    )

            # Delete all edges connected to node2
            for edge in neighbors:
                await self.delete_edge(edge["edge_id"])

            # Delete node2
            await self.delete_node(node_id2)

            self.logger.info(f"Nodes {node_id1} and {node_id2} merged successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error merging nodes: {e}")
            return False

    # Performance and Caching
    def clear_caches(self) -> None:
        """Clear any internal caches."""
        self._relationship_cache.clear()
        self._content_index_cache.clear()
        self.logger.info("JanusGraph caches cleared")

    def get_traversal_statistics(self) -> Dict[str, int]:
        """Get traversal statistics for performance monitoring."""
        return {
            "relationship_cache_entries": len(self._relationship_cache),
            "content_cache_entries": len(self._content_index_cache),
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
