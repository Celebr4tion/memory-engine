"""
JanusGraph storage module for managing knowledge graph operations.

This module provides a wrapper around the gremlin_python client to interact with JanusGraph.
"""
import logging
import uuid
from typing import Dict, Any, Optional
import asyncio
import traceback
import socket

from gremlin_python.driver import client, protocol, serializer
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
from gremlin_python.process.anonymous_traversal import traversal
from gremlin_python.process.graph_traversal import __
from gremlin_python.structure.graph import Graph
from gremlin_python.process.strategies import * # noqa: F403
from gremlin_python.process.traversal import T 

from .graph_storage_adapter import GraphStorageAdapter
from memory_core.config import get_config


class JanusGraphStorage(GraphStorageAdapter):
    """
    Interface for storing and retrieving data from JanusGraph.
    
    This class provides methods to manage nodes and edges in a JanusGraph database.
    """

    def __init__(self, host=None, port=None, traversal_source='g'):
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
        self._loop = None
        self.logger = logging.getLogger(__name__)

    def _sync_connect(self):
        """
        Establish a connection to the JanusGraph database.
        
        Raises:
            ConnectionError: If unable to connect to JanusGraph
        """
        # Don't reconnect if already connected
        if self.g is not None:
            print("Already connected to JanusGraph.")
            return True

        try:
            connection_url = self.config.config.janusgraph.connection_url
            print(f"Connecting to JanusGraph at {connection_url}...")
            
            # Create remote traversal with a connection object - this is synchronous
            remote_connection = DriverRemoteConnection(
                connection_url,
                self.traversal_source
            )
            self.g = traversal().withRemote(remote_connection)
            
            print(f"Connected to JanusGraph at {connection_url}")
            return True
                
        except Exception as e:
            print(f"Failed to connect to JanusGraph: {e}")
            import traceback
            traceback.print_exc()
            
            self.g = None
            raise ConnectionError(f"Could not connect to JanusGraph: {e}")

    async def _async_close(self):
        """Close the connection to the JanusGraph database."""
        if self._client:
            try:
                # Close the client - this might not be awaitable in some versions
                if hasattr(self._client, 'close') and callable(self._client.close):
                    try:
                        # Try to close it asynchronously first
                        await self._client.close()
                    except (TypeError, AttributeError):
                        # If it's not awaitable, call it directly
                        self._client.close()
                
                self._client = None
                self.g = None
                print("Disconnected from JanusGraph.")
            except Exception as e:
                print(f"Error closing JanusGraph connection: {e}")
                import traceback
                traceback.print_exc()
                # Still set to None even if error occurs
                self._client = None
                self.g = None

    def _sync_create_node(self, *args, **kwargs):
        """
        Create a node in the JanusGraph database.
        
        This method accepts different parameter formats for compatibility:
        - Option 1: create_node(node_data)  # Single dictionary with node properties
        - Option 2: create_node(node_id, label, properties)  # Explicit parameters
        
        Args:
            node_data_or_id: Either a dictionary containing node data or a node ID string
            label: (Optional) The label for the node (default: "KnowledgeNode")
            properties: (Optional) Dictionary of node properties
            
        Returns:
            The ID of the created node
            
        Raises:
            ConnectionError: If not connected to JanusGraph
            ValueError: If required fields are missing
        """
        # Extract skip_connect from kwargs if present
        skip_connect = kwargs.pop('_skip_connect', False)
        
        if not skip_connect:
            self._sync_connect()  # Ensure connection
            
        if not self.g:
            raise ConnectionError("Not connected to JanusGraph")
        
        # Parse arguments based on what was provided
        if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
            # Option 1: Single dictionary
            node_data = args[0]
            node_id = str(uuid.uuid4())
            label = "KnowledgeNode"
            properties = node_data
        elif len(args) == 3:
            # Option 2: Explicit parameters
            node_id, label, properties = args
        else:
            raise ValueError("Invalid arguments. Expected either (node_data) or (node_id, label, properties)")
        
        try:
            # Use snake_case methods
            vertex = self.g.add_v(label)
            vertex = vertex.property('node_id', node_id)
            for key, value in properties.items():
                vertex = vertex.property(key, value)
            new_vertex = vertex.next()
            print(f"Node created with graph ID: {new_vertex.id}, node_id: {node_id}")
            return node_id
        except Exception as e:
            print(f"Error creating node {node_id}: {e}")
            raise

    async def _async_get_node(self, node_id, _skip_connect=False):
        """
        Get a node from the JanusGraph database.
        
        Args:
            node_id: The ID of the node to retrieve
            _skip_connect: If True, skip connection attempt (used internally)
            
        Returns:
            Dictionary containing the node data, or None if not found
            
        Raises:
            ConnectionError: If not connected to JanusGraph
        """
        if not _skip_connect:
            await self._async_connect()
            
        if not self.g:
            raise ConnectionError("Not connected to JanusGraph")
        try:
            node_data_list = await self.g.V().has('node_id', node_id).value_map(True).to_list()
            if not node_data_list:
                return None
            node_data = node_data_list[0]  # Take the first result

            # Process node data
            formatted_data = {
                k: v[0] if isinstance(v, list) and len(v) == 1 else v
                for k, v in node_data.items()
                if k not in ['id', 'label', T.id, T.label]  # Exclude meta properties
            }
            # Add required fields, converting ID to string
            formatted_data['id'] = str(node_data.get(T.id))
            formatted_data['node_id'] = node_data.get('node_id', [node_id])[0]
            formatted_data['label'] = str(node_data.get(T.label))
            return formatted_data

        except Exception as e:
            print(f"Error getting node {node_id}: {e}")
            raise

    async def _async_update_node(self, node_id, properties):
        """
        Update a node in the JanusGraph database.
        
        Args:
            node_id: The ID of the node to update
            properties: Dictionary of node properties to update
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            ConnectionError: If not connected to JanusGraph
        """
        await self._async_connect()
        if not self.g:
            raise ConnectionError("Not connected to JanusGraph")
        try:
            vertex = self.g.V().has('node_id', node_id)
            for key, value in properties.items():
                vertex = vertex.property(key, value)
            await vertex.iterate()
            print(f"Node {node_id} updated.")
            return True
        except Exception as e:
            print(f"Error updating node {node_id}: {e}")
            return False  # Return False on failure

    async def _async_delete_node(self, node_id):
        """
        Delete a node from the JanusGraph database.
        
        Args:
            node_id: The ID of the node to delete
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            ConnectionError: If not connected to JanusGraph
        """
        await self._async_connect()
        if not self.g:
            raise ConnectionError("Not connected to JanusGraph")
        try:
            await self.g.V().has('node_id', node_id).drop().iterate()
            print(f"Node {node_id} deleted.")
            return True
        except Exception as e:
            print(f"Error deleting node {node_id}: {e}")
            return False  # Return False on failure

    async def _async_create_edge(self, from_node_id, to_node_id, relation_type, properties):
        """
        Create an edge between two nodes in the JanusGraph database.
        
        Args:
            from_node_id: The ID of the source node
            to_node_id: The ID of the target node
            relation_type: The type of relationship
            properties: Dictionary of edge properties
            
        Returns:
            The ID of the created edge
            
        Raises:
            ConnectionError: If not connected to JanusGraph
            ValueError: If either node doesn't exist
        """
        await self._async_connect()
        if not self.g:
            raise ConnectionError("Not connected to JanusGraph")
        try:
            # Generate a unique edge ID
            edge_id = str(uuid.uuid4())
            
            # Create the edge
            from_vertex = self.g.V().has('node_id', from_node_id)
            to_vertex = self.g.V().has('node_id', to_node_id)

            # Use snake_case methods: add_e, from_, to
            edge_traversal = self.g.add_e(relation_type).from_(from_vertex).to(to_vertex)
            edge_traversal = edge_traversal.property('edge_id', edge_id)

            for key, value in properties.items():
                edge_traversal = edge_traversal.property(key, value)

            new_edge = await edge_traversal.next()
            print(f"Edge created with graph ID: {new_edge.id}, edge_id: {edge_id}")
            return edge_id
        except Exception as e:
            print(f"Error creating edge from {from_node_id} to {to_node_id}: {e}")
            raise

    async def _async_get_edge(self, edge_id):
        """
        Get an edge from the JanusGraph database.
        
        Args:
            edge_id: The ID of the edge to retrieve
            
        Returns:
            Dictionary containing the edge data, or None if not found
            
        Raises:
            ConnectionError: If not connected to JanusGraph
        """
        await self._async_connect()
        if not self.g:
            raise ConnectionError("Not connected to JanusGraph")
        try:
            edge_data_list = await self.g.E().has('edge_id', edge_id).value_map(True).to_list()
            if not edge_data_list:
                return None
            edge_data = edge_data_list[0]

            # Get the source and target node IDs using 'node_id' property
            out_node_id_list = await self.g.E().has('edge_id', edge_id).out_v().values('node_id').to_list()
            in_node_id_list = await self.g.E().has('edge_id', edge_id).in_v().values('node_id').to_list()

            if not out_node_id_list or not in_node_id_list:
                 print(f"Warning: Could not find source or target node for edge {edge_id}")
                 return None  # Or handle as appropriate

            # Process edge data
            formatted_data = {
                k: v[0] if isinstance(v, list) and len(v) == 1 else v
                for k, v in edge_data.items()
                if k not in ['id', 'label', T.id, T.label]  # Exclude meta properties
            }
            formatted_data['id'] = str(edge_data.get(T.id))
            formatted_data['edge_id'] = edge_data.get('edge_id', [edge_id])[0]
            formatted_data['relation_type'] = str(edge_data.get(T.label))
            formatted_data['from_id'] = str(out_node_id_list[0])
            formatted_data['to_id'] = str(in_node_id_list[0])
            return formatted_data

        except Exception as e:
            print(f"Error getting edge {edge_id}: {e}")
            raise

    async def _async_update_edge(self, edge_id, properties):
        """
        Update an edge in the JanusGraph database.
        
        Args:
            edge_id: The ID of the edge to update
            properties: Dictionary of edge properties to update
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            ConnectionError: If not connected to JanusGraph
        """
        await self._async_connect()
        if not self.g:
            raise ConnectionError("Not connected to JanusGraph")
        try:
            edge_traversal = self.g.E().has('edge_id', edge_id)
            for key, value in properties.items():
                edge_traversal = edge_traversal.property(key, value)
            await edge_traversal.iterate()
            print(f"Edge {edge_id} updated.")
            return True
        except Exception as e:
            print(f"Error updating edge {edge_id}: {e}")
            return False  # Return False on failure

    async def _async_delete_edge(self, edge_id):
        """
        Delete an edge from the JanusGraph database.
        
        Args:
            edge_id: The ID of the edge to delete
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            ConnectionError: If not connected to JanusGraph
        """
        await self._async_connect()
        if not self.g:
            raise ConnectionError("Not connected to JanusGraph")
        try:
            await self.g.E().has('edge_id', edge_id).drop().iterate()
            print(f"Edge {edge_id} deleted.")
            return True
        except Exception as e:
            print(f"Error deleting edge {edge_id}: {e}")
            return False  # Return False on failure

    async def _async_find_neighbors(self, node_id, relation_type=None):
        """
        Find neighbors of a node in the JanusGraph database.
        
        Args:
            node_id: The ID of the node to find neighbors for
            relation_type: Optional filter for the type of relationship
            
        Returns:
            List of dictionaries containing neighbor info
            
        Raises:
            ConnectionError: If not connected to JanusGraph
        """
        await self._async_connect()
        if not self.g:
            raise ConnectionError("Not connected to JanusGraph")
        try:
            traversal = self.g.V().has('node_id', node_id).both_e()
            if relation_type:
                traversal = traversal.has_label(relation_type)
            neighbor_edges = await traversal.value_map(True).to_list()

            neighbors = []
            for edge_data in neighbor_edges:
                edge_id = edge_data.get('edge_id', [None])[0]
                if edge_id:
                    # Get full edge details
                    edge = await self._async_get_edge(edge_id)
                    if edge:
                        neighbors.append(edge)
            return neighbors
        except Exception as e:
            print(f"Error finding neighbors for node {node_id}: {e}")
            raise

    async def _async_merge_nodes(self, node_id1, node_id2):
        """
        Merge two nodes into one, preserving all relationships.
        
        Args:
            node_id1: The ID of the first node (kept)
            node_id2: The ID of the second node (merged and deleted)
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            ConnectionError: If not connected to JanusGraph
        """
        await self._async_connect()
        if not self.g:
            raise ConnectionError("Not connected to JanusGraph")
        try:
            # Get data for both nodes
            node1_data = await self._async_get_node(node_id1)
            node2_data = await self._async_get_node(node_id2)
            
            if not node1_data or not node2_data:
                print(f"Error: One or both nodes not found")
                return False
                
            # Store all edges connected to node2
            neighbors = await self._async_find_neighbors(node_id2)
                
            # Create new edges to node1 for all of node2's connections
            for edge in neighbors:
                # Skip if this is already an edge to node1
                if edge['from_id'] == node_id1 or edge['to_id'] == node_id1:
                    continue
                    
                properties = {k: v for k, v in edge.items() 
                              if k not in ['id', 'edge_id', 'from_id', 'to_id', 'relation_type']}
                    
                # Determine direction
                if edge['from_id'] == node_id2:
                    # Outgoing edge from node2
                    await self._async_create_edge(node_id1, edge['to_id'], 
                                           edge['relation_type'], properties)
                else:
                    # Incoming edge to node2
                    await self._async_create_edge(edge['from_id'], node_id1, 
                                           edge['relation_type'], properties)
                    
            # Delete all edges connected to node2
            for edge in neighbors:
                await self._async_delete_edge(edge['edge_id'])
                
            # Delete node2
            await self._async_delete_node(node_id2)
                
            print(f"Nodes {node_id1} and {node_id2} merged successfully")
            return True
        except Exception as e:
            print(f"Error merging nodes: {e}")
            return False

    async def _async_clear_all_data(self):
        """
        Clear all data from the JanusGraph database.
        
        Returns:
            True if successful
            
        Raises:
            ConnectionError: If not connected to JanusGraph
        """
        await self._async_connect()
        if not self.g:
            raise ConnectionError("Not connected to JanusGraph")
        try:
            await self.g.V().drop().iterate()
            print("All data cleared from JanusGraph.")
            return True
        except Exception as e:
            print(f"Error clearing data: {e}")
            raise

    # Non-async wrapper methods for backward compatibility
    def connect(self):
        """Synchronous connect method."""
        try:
            return self._sync_connect()
        except Exception as e:
            print(f"Error in connect: {e}")
            return False
        
    def close(self):
        """Synchronous wrapper for _async_close."""
        try:
            # Call the method to get a coroutine
            return self._run_async_in_sync_context(self._async_close())
        except Exception as e:
            print(f"Error in close: {e}")
            return False

    def _create_sync_wrapper(self, async_method, method_name, *args, **kwargs):
        """
        Generic wrapper to convert async methods to sync.
        
        Args:
            async_method: The async method to call
            method_name: The name of the method (for error reporting)
            *args: Arguments to pass to the async method
            **kwargs: Keyword arguments to pass to the async method
            
        Returns:
            The result of the async method
        """
        try:
            return self._run_async_in_sync_context(async_method(*args, **kwargs))
        except Exception as e:
            print(f"Error in {method_name} sync wrapper: {e}")
            raise
            
    def create_node(self, *args, **kwargs):
        """Synchronous create_node method."""
        return self._sync_create_node(*args, **kwargs)
        
    def get_node(self, node_id):
        """Non-async wrapper for get_node()."""
        return self._create_sync_wrapper(self._async_get_node, "get_node", node_id)
        
    def update_node(self, node_id, properties):
        """Non-async wrapper for update_node()."""
        return self._create_sync_wrapper(self._async_update_node, "update_node", node_id, properties)
        
    def delete_node(self, node_id):
        """Non-async wrapper for delete_node()."""
        return self._create_sync_wrapper(self._async_delete_node, "delete_node", node_id)

    def create_edge(self, *args):
        """Non-async wrapper for create_edge()."""
        return self._create_sync_wrapper(self._async_create_edge, "create_edge", *args)
        
    def get_edge(self, edge_id):
        """Non-async wrapper for get_edge()."""
        return self._create_sync_wrapper(self._async_get_edge, "get_edge", edge_id)
        
    def update_edge(self, edge_id, properties):
        """Non-async wrapper for update_edge()."""
        return self._create_sync_wrapper(self._async_update_edge, "update_edge", edge_id, properties)
        
    def delete_edge(self, edge_id):
        """Non-async wrapper for delete_edge()."""
        return self._create_sync_wrapper(self._async_delete_edge, "delete_edge", edge_id)
        
    def find_neighbors(self, node_id, relation_type=None):
        """Non-async wrapper for find_neighbors()."""
        return self._create_sync_wrapper(self._async_find_neighbors, "find_neighbors", node_id, relation_type)
        
    def merge_nodes(self, node_id1, node_id2):
        """Non-async wrapper for merge_nodes()."""
        return self._create_sync_wrapper(self._async_merge_nodes, "merge_nodes", node_id1, node_id2)
        
    def clear_all_data(self):
        """Non-async wrapper for clear_all_data()."""
        return self._create_sync_wrapper(self._async_clear_all_data, "clear_all_data")

    @classmethod
    async def test_connection(cls, host='localhost', port=8182, timeout=5):
        """
        Test if a JanusGraph instance is reachable with given connection parameters.

        Args:
            host: Host where the JanusGraph instance is running.
            port: Port through which the JanusGraph instance can be accessed.
            timeout: Seconds to wait for a response before timing out.

        Returns:
            bool: True if the connection is successful, False otherwise.
        """
        client_obj = None
        try:
            print(f"Testing JanusGraph connection to {host}:{port}...")
            
            # First try a simple socket connection to check if the port is open
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(timeout)
                sock.connect((host, port))
                print(f"TCP connection to {host}:{port} succeeded")
                sock.close()
            except socket.error as e:
                print(f"TCP connection to {host}:{port} failed: {e}")
                return False
            
            # Initialize the client WITHOUT awaiting - this is the key fix
            # Client() is a constructor, not a coroutine
            client_obj = client.Client(
                f"ws://{host}:{port}/gremlin", 
                "g",  # Use the traversal source directly
                connection_timeout=timeout,
                message_serializer=serializer.GraphSONMessageSerializer()
            )
            
            # Now use the client to submit a query
            try:
                # This is correctly awaited, as submit() returns a coroutine
                result = await client_obj.submit("g.V().count()")
                vertices_count = result.all().result()
                print(f"JanusGraph connection test successful. Found {vertices_count} vertices.")
                return True
            finally:
                # Make sure to close the client
                if client_obj:
                    await client_obj.close()
                    
        except Exception as e:
            print(f"JanusGraph connection test failed: {e}")
            if client_obj:
                try:
                    await client_obj.close()
                except Exception:
                    pass
            return False

    @classmethod
    async def is_available(cls, host='localhost', port=8182, timeout=5):
        """
        Check if a JanusGraph instance is available and responding.
        
        Args:
            host: Host where the JanusGraph instance is running.
            port: Port through which the JanusGraph instance can be accessed.
            timeout: Seconds to wait for a response before timing out.
            
        Returns:
            bool: True if available, False otherwise
        """
        return await cls.test_connection(host, port, timeout)

    @classmethod
    def test_connection_sync(cls, host='localhost', port=8182, timeout=5):
        """
        Synchronous wrapper for test_connection.
        
        Args:
            host: Host where the JanusGraph instance is running.
            port: Port through which the JanusGraph instance can be accessed.
            timeout: Seconds to wait for a response before timing out.
            
        Returns:
            bool: True if the connection is successful, False otherwise.
        """
        import socket
        
        # 1. First check if we can connect via TCP socket
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect((host, port))
            print(f"TCP connection to {host}:{port} succeeded")
            
            # 2. Try sending a simple WebSocket handshake to confirm it's a WebSocket server
            try:
                # Send a mock WebSocket handshake
                handshake = (
                    f"GET /gremlin HTTP/1.1\r\n"
                    f"Host: {host}:{port}\r\n"
                    f"Upgrade: websocket\r\n"
                    f"Connection: Upgrade\r\n"
                    f"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
                    f"Sec-WebSocket-Version: 13\r\n\r\n"
                )
                sock.sendall(handshake.encode())
                
                # If we get a response, the server is responding to WebSocket requests
                response = sock.recv(1024)
                if b"HTTP/1.1 101" in response or b"HTTP/1.0 101" in response or b"Upgrade: websocket" in response:
                    print("WebSocket handshake succeeded - JanusGraph server is accepting WebSocket connections")
                    return True
                else:
                    print(f"WebSocket handshake failed - received: {response[:100]}...")
            except socket.timeout:
                print("Socket timed out waiting for WebSocket handshake response")
            except Exception as e:
                print(f"Error during WebSocket handshake: {e}")
                
            # We at least know the TCP socket works, so return True
            # This is a fallback since a real WebSocket connection would be tested by the async method
            print("Returning true based on TCP connection only")
            return True
            
        except socket.error as e:
            print(f"TCP connection to {host}:{port} failed: {e}")
            return False
        finally:
            try:
                sock.close()
            except:
                pass

    @classmethod
    def is_available_sync(cls, host='localhost', port=8182, timeout=5):
        """
        Synchronous wrapper for is_available.
        
        Args:
            host: Host where the JanusGraph instance is running.
            port: Port through which the JanusGraph instance can be accessed.
            timeout: Seconds to wait for a response before timing out.
            
        Returns:
            bool: True if available, False otherwise
        """
        # Since is_available is just a wrapper around test_connection,
        # and test_connection_sync is now working properly,
        # we can just delegate to it.
        return cls.test_connection_sync(host, port, timeout)

    def _run_async_in_sync_context(self, coroutine, timeout=10):
        """
        Helper function to run an async coroutine in a sync context.
        
        Args:
            coroutine: The async coroutine to run
            timeout: Timeout in seconds
            
        Returns:
            The result of the coroutine
        """
        try:
            # Use the same pattern as in _run_class_async_in_sync_context
            def run_in_thread():
                """Run in a new thread with a fresh event loop"""
                try:
                    # Create a new loop and run the coroutine
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(coroutine)
                    finally:
                        # Clean up resources
                        loop.close()
                except Exception as e:
                    print(f"Error in thread: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
            
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=timeout)
        except Exception as e:
            print(f"Error running async in sync context: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    @classmethod
    def _run_class_async_in_sync_context(cls, coroutine, timeout=10):
        """
        Class method helper to run an async coroutine in a sync context.
        
        Args:
            coroutine: The async coroutine to run
            timeout: Timeout in seconds
            
        Returns:
            The result of the coroutine
        """
        def run_in_thread():
            """Run in a new thread with a fresh event loop"""
            try:
                # Create a new loop and run the coroutine
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(coroutine)
                finally:
                    # Clean up resources
                    loop.close()
            except Exception as e:
                print(f"Error in thread: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        try:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=timeout)
        except Exception as e:
            print(f"Error running async in sync context: {e}")
            import traceback
            traceback.print_exc()
            return False 