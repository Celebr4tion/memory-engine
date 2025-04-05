"""
RevisionManager for tracking changes to the knowledge graph.

This module provides functionality for:
1. Recording changes to nodes and edges in a 'RevisionLog'
2. Taking periodic snapshots of the graph state
3. Retrieving revision history and reverting to previous states
"""
import json
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple, Union

from memory_core.db.janusgraph_storage import JanusGraphStorage


class RevisionManager:
    """
    Manages revisions and snapshots of the knowledge graph.
    
    Records changes to nodes and edges in a 'RevisionLog' and periodically
    creates snapshots of the current graph state.
    """
    
    # Revision log vertex label
    REVISION_LOG_LABEL = 'RevisionLog'
    
    # Snapshot vertex label
    SNAPSHOT_LABEL = 'GraphSnapshot'
    
    # Change types
    CHANGE_CREATE = 'create'
    CHANGE_UPDATE = 'update'
    CHANGE_DELETE = 'delete'
    
    # Object types
    OBJECT_NODE = 'node'
    OBJECT_EDGE = 'edge'
    
    def __init__(self, storage, changes_threshold=100, enable_snapshots=True):
        """
        Initialize the RevisionManager.
        
        Args:
            storage: The storage backend (JanusGraphStorage instance)
            changes_threshold: Number of changes before creating a snapshot
            enable_snapshots: Whether to automatically create snapshots
        """
        self.storage = storage
        self.changes_threshold = changes_threshold
        self.enable_snapshots = enable_snapshots
        self.changes_since_snapshot = 0

    def _log_change(self, object_type: str, object_id: str, 
                   change_type: str, old_data: Optional[Dict] = None, 
                   new_data: Optional[Dict] = None) -> str:
        """
        Log a change to the RevisionLog.
        
        Args:
            object_type: Type of object ('node' or 'edge')
            object_id: ID of the object
            change_type: Type of change ('create', 'update', or 'delete')
            old_data: Previous state of the object (None for creation)
            new_data: New state of the object (None for deletion)
            
        Returns:
            ID of the created log entry
        """
        try:
            # Prepare log data
            log_data = {
                'label': 'RevisionLog',
                'object_type': object_type,
                'object_id': object_id,
                'change_type': change_type,
                'timestamp': time.time()
            }
            
            # Convert data to JSON strings for storage
            if old_data is not None:
                log_data['old_data'] = json.dumps(old_data)
            else:
                log_data['old_data'] = None
                
            if new_data is not None:
                log_data['new_data'] = json.dumps(new_data)
            else:
                log_data['new_data'] = None
            
            # Create the log entry in the database
            log_id = self.storage.create_node(log_data)
            
            # Increment changes counter and check if we need to create a snapshot
            self.changes_since_snapshot += 1
            if self.enable_snapshots and self.changes_since_snapshot >= self.changes_threshold:
                snapshot_id = self.create_snapshot()
                # Reset counter after snapshot
                self.changes_since_snapshot = 0
            
            return log_id
        except Exception as e:
            # Handle errors appropriately
            print(f"Error logging change: {e}")
            raise

    def log_node_creation(self, node_id: str, node_data: Dict[str, Any]) -> str:
        """
        Log the creation of a node.
        
        Args:
            node_id: ID of the created node
            node_data: Data of the created node
            
        Returns:
            ID of the created log entry
        """
        return self._log_change('node', node_id, 'create', None, node_data)

    def log_node_update(self, node_id: str, old_data: Dict[str, Any], 
                       new_data: Dict[str, Any]) -> str:
        """
        Log the update of a node.
        
        Args:
            node_id: ID of the updated node
            old_data: Previous state of the node
            new_data: New state of the node
            
        Returns:
            ID of the created log entry
        """
        return self._log_change('node', node_id, 'update', old_data, new_data)

    def log_node_deletion(self, node_id: str, node_data: Dict[str, Any]) -> str:
        """
        Log the deletion of a node.
        
        Args:
            node_id: ID of the deleted node
            node_data: Data of the deleted node
            
        Returns:
            ID of the created log entry
        """
        return self._log_change('node', node_id, 'delete', node_data, None)

    def log_edge_creation(self, edge_id: str, edge_data: Dict[str, Any]) -> str:
        """
        Log the creation of an edge.
        
        Args:
            edge_id: ID of the created edge
            edge_data: Data of the created edge
            
        Returns:
            ID of the created log entry
        """
        return self._log_change('edge', edge_id, 'create', None, edge_data)

    def log_edge_update(self, edge_id: str, old_data: Dict[str, Any], 
                       new_data: Dict[str, Any]) -> str:
        """
        Log the update of an edge.
        
        Args:
            edge_id: ID of the updated edge
            old_data: Previous state of the edge
            new_data: New state of the edge
            
        Returns:
            ID of the created log entry
        """
        return self._log_change('edge', edge_id, 'update', old_data, new_data)

    def log_edge_deletion(self, edge_id: str, edge_data: Dict[str, Any]) -> str:
        """
        Log the deletion of an edge.
        
        Args:
            edge_id: ID of the deleted edge
            edge_data: Data of the deleted edge
            
        Returns:
            ID of the created log entry
        """
        return self._log_change('edge', edge_id, 'delete', edge_data, None)

    def create_snapshot(self) -> str:
        """
        Create a snapshot of the current graph state.
        
        Returns:
            ID of the created snapshot
        """
        try:
            # Generate a unique ID for the snapshot
            snapshot_id = str(uuid.uuid4())
            timestamp = time.time()
            
            # Get all nodes and edges
            nodes = self.storage.get_all_nodes()
            edges = self.storage.get_all_edges()
            
            # Create snapshot data
            snapshot_data = {
                'snapshot_id': snapshot_id,
                'timestamp': timestamp,
                'nodes': nodes,
                'edges': edges
            }
            
            # Store the snapshot as a vertex in the graph
            snapshot_vertex = {
                'label': 'GraphSnapshot',
                'snapshot_id': snapshot_id,
                'timestamp': timestamp,
                'data': json.dumps(snapshot_data)
            }
            
            # Create the snapshot vertex
            self.storage.create_node(snapshot_vertex)
            
            return snapshot_id
        except Exception as e:
            print(f"Error creating snapshot: {e}")
            raise

    def get_all_snapshots(self) -> List[Dict[str, Any]]:
        """
        Get all snapshots ordered by timestamp.
        
        Returns:
            List of snapshot metadata (without full data)
        """
        try:
            # Query for all snapshots
            snapshots = self.storage.query_vertices({
                'label': 'GraphSnapshot'
            })
            
            # Sort by timestamp (descending)
            snapshots.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            
            # Return without the full data field to save bandwidth
            return [{
                'snapshot_id': s.get('snapshot_id'),
                'timestamp': s.get('timestamp')
            } for s in snapshots]
        except Exception as e:
            print(f"Error getting snapshots: {e}")
            raise

    def get_snapshot(self, snapshot_id: str) -> Dict[str, Any]:
        """
        Get a specific snapshot by ID.
        
        Args:
            snapshot_id: ID of the snapshot to retrieve
            
        Returns:
            Snapshot data including nodes and edges
            
        Raises:
            ValueError: If the snapshot does not exist
        """
        try:
            # Query for the specific snapshot
            snapshots = self.storage.query_vertices({
                'label': 'GraphSnapshot',
                'snapshot_id': snapshot_id
            })
            
            if not snapshots:
                raise ValueError(f"Snapshot with ID {snapshot_id} not found")
            
            snapshot = snapshots[0]
            
            # Parse the JSON data
            return json.loads(snapshot.get('data', '{}'))
        except Exception as e:
            print(f"Error getting snapshot: {e}")
            raise

    def get_revision_history(self, object_type: str, object_id: str) -> List[Dict[str, Any]]:
        """
        Get the revision history for a specific object.
        
        Args:
            object_type: Type of object ('node' or 'edge')
            object_id: ID of the object
            
        Returns:
            List of revision entries, sorted by timestamp (descending)
        """
        try:
            # Query for all revisions for this object
            revisions = self.storage.query_vertices({
                'label': 'RevisionLog',
                'object_type': object_type,
                'object_id': object_id
            })
            
            # Sort by timestamp (descending)
            revisions.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            
            # Parse JSON data
            for revision in revisions:
                if revision.get('old_data'):
                    revision['old_data'] = json.loads(revision['old_data'])
                if revision.get('new_data'):
                    revision['new_data'] = json.loads(revision['new_data'])
            
            return revisions
        except Exception as e:
            print(f"Error getting revision history: {e}")
            raise

    def revert_node_to_previous_state(self, node_id: str) -> bool:
        """
        Revert a node to its previous state.
        
        Args:
            node_id: ID of the node to revert
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get revision history
            revisions = self.get_revision_history('node', node_id)
            
            if len(revisions) < 2:
                print(f"Not enough revisions to revert node {node_id}")
                return False
            
            # Get the current and previous revision
            current_revision = revisions[0]
            previous_revision = revisions[1]
            
            # Get current node data to verify
            current_node = self.storage.get_node(node_id)
            
            # Update the node with the previous state
            self.storage.update_node(node_id, previous_revision['new_data'])
            
            # Log the revert operation
            self._log_change('node', node_id, 'revert', 
                           current_revision['new_data'], 
                           previous_revision['new_data'])
            
            return True
        except Exception as e:
            print(f"Error reverting node: {e}")
            return False

    def revert_edge_to_previous_state(self, edge_id: str) -> bool:
        """
        Revert an edge to its previous state.
        
        Args:
            edge_id: ID of the edge to revert
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get revision history
            revisions = self.get_revision_history('edge', edge_id)
            
            if len(revisions) < 2:
                print(f"Not enough revisions to revert edge {edge_id}")
                return False
            
            # Get the current and previous revision
            current_revision = revisions[0]
            previous_revision = revisions[1]
            
            # Get current edge data to verify
            current_edge = self.storage.get_edge(edge_id)
            
            # For edges, we need to be careful about structure vs properties
            # Structure (from_id, to_id, relation_type) should typically not be changed
            # So we only update properties like confidence_score, timestamp, version
            edge_data = previous_revision['new_data']
            update_data = {
                key: value for key, value in edge_data.items() 
                if key not in ['from_id', 'to_id', 'relation_type']
            }
            
            # Update the edge with the previous state properties
            self.storage.update_edge(edge_id, update_data)
            
            # Log the revert operation
            self._log_change('edge', edge_id, 'revert', 
                           current_revision['new_data'], 
                           previous_revision['new_data'])
            
            return True
        except Exception as e:
            print(f"Error reverting edge: {e}")
            return False 