"""
JanusGraph storage module for managing knowledge graph operations.

This module provides a wrapper around the gremlin_python client to interact with JanusGraph.
"""
import logging
from typing import Dict, Any, Optional

from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
from gremlin_python.process.anonymous_traversal import traversal
from gremlin_python.process.graph_traversal import __
from gremlin_python.structure.graph import Graph


class JanusGraphStorage:
    """
    Interface for storing and retrieving data from JanusGraph.
    
    This class provides methods to manage nodes and edges in a JanusGraph database.
    """

    def __init__(self, host: str, port: int):
        """
        Initialize JanusGraphStorage with connection details.
        
        Args:
            host: The hostname or IP address of the JanusGraph server
            port: The port number of the JanusGraph server
        """
        self.host = host
        self.port = port
        self.connection_url = f"ws://{host}:{port}/gremlin"
        self.graph = None
        self.g = None
        self.conn = None
        self.logger = logging.getLogger(__name__)

    def connect(self) -> None:
        """
        Establish a connection to the JanusGraph database.
        
        Raises:
            ConnectionError: If unable to connect to JanusGraph
        """
        try:
            self.graph = Graph()
            self.conn = DriverRemoteConnection(self.connection_url, 'g')
            self.g = traversal().withRemote(self.conn)
            self.logger.info(f"Connected to JanusGraph at {self.connection_url}")
        except Exception as e:
            self.logger.error(f"Failed to connect to JanusGraph: {str(e)}")
            raise ConnectionError(f"Failed to connect to JanusGraph: {str(e)}")

    def close(self) -> None:
        """Close the connection to the JanusGraph database."""
        if self.conn:
            self.conn.close()
            self.logger.info("Connection to JanusGraph closed")

    def create_node(self, node_data: dict) -> str:
        """
        Create a new node in the knowledge graph.
        
        Args:
            node_data: Dictionary containing node properties:
                - content: str - The content of the node
                - source: str - The source of the information
                - creation_timestamp: float - When the node was created
                - rating_richness: float - Rating for information richness
                - rating_truthfulness: float - Rating for truthfulness
                - rating_stability: float - Rating for stability
                
        Returns:
            The ID of the newly created node as a string
            
        Raises:
            ValueError: If required fields are missing
            ConnectionError: If not connected to JanusGraph
        """
        required_fields = [
            'content', 'source', 'creation_timestamp',
            'rating_richness', 'rating_truthfulness', 'rating_stability'
        ]
        
        for field in required_fields:
            if field not in node_data:
                raise ValueError(f"Missing required field: {field}")
        
        if not self.g:
            raise ConnectionError("Not connected to JanusGraph. Call connect() first.")
        
        try:
            vertex = self.g.addV('KnowledgeNode')
            
            # Add properties
            for key, value in node_data.items():
                vertex = vertex.property(key, value)
                
            # Execute and get the ID
            result = vertex.next()
            node_id = str(result.id)
            self.logger.info(f"Created node with ID: {node_id}")
            return node_id
        except Exception as e:
            self.logger.error(f"Failed to create node: {str(e)}")
            raise

    def get_node(self, node_id: str) -> dict:
        """
        Retrieve a node and its properties from the knowledge graph.
        
        Args:
            node_id: The ID of the node to retrieve
            
        Returns:
            Dictionary containing the node's properties
            
        Raises:
            ValueError: If the node doesn't exist
            ConnectionError: If not connected to JanusGraph
        """
        if not self.g:
            raise ConnectionError("Not connected to JanusGraph. Call connect() first.")
        
        try:
            result = self.g.V(node_id).valueMap(True).next()
            
            # Convert from JanusGraph format to dictionary
            node_data = {}
            for key, value in result.items():
                if key != 'T.id' and key != 'T.label':
                    # JanusGraph returns property values as lists
                    node_data[key] = value[0] if isinstance(value, list) and value else value
                elif key == 'T.id':
                    node_data['id'] = value
                elif key == 'T.label':
                    node_data['label'] = value
                    
            return node_data
        except StopIteration:
            raise ValueError(f"Node with ID {node_id} not found")
        except Exception as e:
            self.logger.error(f"Error retrieving node {node_id}: {str(e)}")
            raise

    def update_node(self, node_id: str, updated_data: dict) -> None:
        """
        Update a node's properties in the knowledge graph.
        
        Args:
            node_id: The ID of the node to update
            updated_data: Dictionary containing the properties to update
            
        Raises:
            ValueError: If the node doesn't exist
            ConnectionError: If not connected to JanusGraph
        """
        if not self.g:
            raise ConnectionError("Not connected to JanusGraph. Call connect() first.")
        
        try:
            # Check if node exists
            self.g.V(node_id).next()
            
            # Update properties
            vertex = self.g.V(node_id)
            for key, value in updated_data.items():
                vertex = vertex.property(key, value)
                
            # Execute the update
            vertex.next()
            self.logger.info(f"Updated node with ID: {node_id}")
        except StopIteration:
            raise ValueError(f"Node with ID {node_id} not found")
        except Exception as e:
            self.logger.error(f"Error updating node {node_id}: {str(e)}")
            raise

    def delete_node(self, node_id: str) -> None:
        """
        Delete a node and its connected edges from the knowledge graph.
        
        Args:
            node_id: The ID of the node to delete
            
        Raises:
            ValueError: If the node doesn't exist
            ConnectionError: If not connected to JanusGraph
        """
        if not self.g:
            raise ConnectionError("Not connected to JanusGraph. Call connect() first.")
        
        try:
            # Check if node exists
            self.g.V(node_id).next()
            
            # Delete the node (this will also delete connected edges in JanusGraph)
            self.g.V(node_id).drop().iterate()
            self.logger.info(f"Deleted node with ID: {node_id}")
        except StopIteration:
            raise ValueError(f"Node with ID {node_id} not found")
        except Exception as e:
            self.logger.error(f"Error deleting node {node_id}: {str(e)}")
            raise

    def create_edge(self, from_id: str, to_id: str, relation_type: str, edge_metadata: dict) -> str:
        """
        Create an edge between two nodes in the knowledge graph.
        
        Args:
            from_id: The ID of the source node
            to_id: The ID of the target node
            relation_type: The type of relationship between the nodes
            edge_metadata: Dictionary containing edge properties:
                - timestamp: float - When the edge was created
                - confidence_score: float - Confidence in the relationship
                - version: int - Version of the edge
                
        Returns:
            The ID of the newly created edge as a string
            
        Raises:
            ValueError: If the nodes don't exist or required metadata is missing
            ConnectionError: If not connected to JanusGraph
        """
        required_fields = ['timestamp', 'confidence_score', 'version']
        
        for field in required_fields:
            if field not in edge_metadata:
                raise ValueError(f"Missing required field in edge_metadata: {field}")
        
        if not self.g:
            raise ConnectionError("Not connected to JanusGraph. Call connect() first.")
        
        try:
            # Check if both nodes exist
            self.g.V(from_id).next()
            self.g.V(to_id).next()
            
            # Create the edge
            edge = self.g.V(from_id).addE(relation_type).to(__.V(to_id))
            
            # Add properties
            for key, value in edge_metadata.items():
                edge = edge.property(key, value)
                
            # Execute and get the ID
            result = edge.next()
            edge_id = str(result.id)
            self.logger.info(f"Created edge with ID: {edge_id} from {from_id} to {to_id}")
            return edge_id
        except StopIteration:
            raise ValueError(f"One or both nodes ({from_id}, {to_id}) not found")
        except Exception as e:
            self.logger.error(f"Error creating edge: {str(e)}")
            raise

    def get_edge(self, edge_id: str) -> dict:
        """
        Retrieve an edge and its properties from the knowledge graph.
        
        Args:
            edge_id: The ID of the edge to retrieve
            
        Returns:
            Dictionary containing the edge's properties
            
        Raises:
            ValueError: If the edge doesn't exist
            ConnectionError: If not connected to JanusGraph
        """
        if not self.g:
            raise ConnectionError("Not connected to JanusGraph. Call connect() first.")
        
        try:
            result = self.g.E(edge_id).valueMap(True).next()
            
            # Convert from JanusGraph format to dictionary
            edge_data = {}
            for key, value in result.items():
                if key != 'T.id' and key != 'T.label':
                    # JanusGraph returns property values as lists
                    edge_data[key] = value[0] if isinstance(value, list) and value else value
                elif key == 'T.id':
                    edge_data['id'] = value
                elif key == 'T.label':
                    edge_data['relation_type'] = value
                    
            # Get the source and target node IDs
            out_v = self.g.E(edge_id).outV().id().next()
            in_v = self.g.E(edge_id).inV().id().next()
            edge_data['from_id'] = str(out_v)
            edge_data['to_id'] = str(in_v)
            
            return edge_data
        except StopIteration:
            raise ValueError(f"Edge with ID {edge_id} not found")
        except Exception as e:
            self.logger.error(f"Error retrieving edge {edge_id}: {str(e)}")
            raise

    def update_edge(self, edge_id: str, updated_metadata: dict) -> None:
        """
        Update an edge's properties in the knowledge graph.
        
        Args:
            edge_id: The ID of the edge to update
            updated_metadata: Dictionary containing the properties to update
            
        Raises:
            ValueError: If the edge doesn't exist
            ConnectionError: If not connected to JanusGraph
        """
        if not self.g:
            raise ConnectionError("Not connected to JanusGraph. Call connect() first.")
        
        try:
            # Check if edge exists
            self.g.E(edge_id).next()
            
            # Update properties
            edge = self.g.E(edge_id)
            for key, value in updated_metadata.items():
                edge = edge.property(key, value)
                
            # Execute the update
            edge.next()
            self.logger.info(f"Updated edge with ID: {edge_id}")
        except StopIteration:
            raise ValueError(f"Edge with ID {edge_id} not found")
        except Exception as e:
            self.logger.error(f"Error updating edge {edge_id}: {str(e)}")
            raise

    def delete_edge(self, edge_id: str) -> None:
        """
        Delete an edge from the knowledge graph.
        
        Args:
            edge_id: The ID of the edge to delete
            
        Raises:
            ValueError: If the edge doesn't exist
            ConnectionError: If not connected to JanusGraph
        """
        if not self.g:
            raise ConnectionError("Not connected to JanusGraph. Call connect() first.")
        
        try:
            # Check if edge exists
            self.g.E(edge_id).next()
            
            # Delete the edge
            self.g.E(edge_id).drop().iterate()
            self.logger.info(f"Deleted edge with ID: {edge_id}")
        except StopIteration:
            raise ValueError(f"Edge with ID {edge_id} not found")
        except Exception as e:
            self.logger.error(f"Error deleting edge {edge_id}: {str(e)}")
            raise 