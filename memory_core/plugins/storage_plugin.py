"""
Storage backend plugin interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, AsyncGenerator
from .plugin_manager import PluginInterface


class StoragePluginInterface(PluginInterface):
    """Interface for storage backend plugins."""
    
    @property
    @abstractmethod
    def plugin_type(self) -> str:
        """Return 'storage'."""
        return "storage"
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the storage backend."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the storage backend."""
        pass
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """Test the connection to the storage backend."""
        pass
    
    @abstractmethod
    async def create_knowledge_node(self, content: str, node_type: str = "knowledge", 
                                   **kwargs) -> str:
        """Create a new knowledge node."""
        pass
    
    @abstractmethod
    async def get_knowledge_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a knowledge node by ID."""
        pass
    
    @abstractmethod
    async def update_knowledge_node(self, node_id: str, **updates) -> bool:
        """Update a knowledge node."""
        pass
    
    @abstractmethod
    async def delete_knowledge_node(self, node_id: str) -> bool:
        """Delete a knowledge node."""
        pass
    
    @abstractmethod
    async def search_nodes(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for nodes matching the query."""
        pass
    
    @abstractmethod
    async def get_all_knowledge_nodes(self, offset: int = 0, 
                                     limit: int = None) -> List[Dict[str, Any]]:
        """Get all knowledge nodes."""
        pass
    
    @abstractmethod
    async def create_relationship(self, source_id: str, target_id: str, 
                                 relationship_type: str, **properties) -> str:
        """Create a relationship between nodes."""
        pass
    
    @abstractmethod
    async def get_relationships(self, node_id: str, 
                               direction: str = "both") -> List[Dict[str, Any]]:
        """Get relationships for a node."""
        pass
    
    @abstractmethod
    async def delete_relationship(self, relationship_id: str) -> bool:
        """Delete a relationship."""
        pass
    
    async def get_node_count(self) -> int:
        """Get total number of nodes."""
        nodes = await self.get_all_knowledge_nodes()
        return len(nodes)
    
    async def get_relationship_count(self) -> int:
        """Get total number of relationships."""
        # Default implementation - can be overridden for efficiency
        count = 0
        nodes = await self.get_all_knowledge_nodes()
        for node in nodes:
            node_id = node.get('id') or node.get('node_id')
            if node_id:
                relationships = await self.get_relationships(node_id)
                count += len(relationships)
        return count // 2  # Divide by 2 to avoid double counting
    
    async def clear_all_data(self) -> bool:
        """Clear all data from storage."""
        # Default implementation - delete all nodes
        try:
            nodes = await self.get_all_knowledge_nodes()
            for node in nodes:
                node_id = node.get('id') or node.get('node_id')
                if node_id:
                    await self.delete_knowledge_node(node_id)
            return True
        except Exception:
            return False
    
    async def export_data(self) -> Dict[str, Any]:
        """Export all data from storage."""
        nodes = await self.get_all_knowledge_nodes()
        
        all_relationships = []
        for node in nodes:
            node_id = node.get('id') or node.get('node_id')
            if node_id:
                relationships = await self.get_relationships(node_id, "out")
                all_relationships.extend(relationships)
        
        return {
            'nodes': nodes,
            'relationships': all_relationships
        }
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            'node_count': await self.get_node_count(),
            'relationship_count': await self.get_relationship_count(),
            'storage_type': self.name
        }


class StoragePlugin(StoragePluginInterface):
    """Base class for storage plugins."""
    
    def __init__(self):
        self._connected = False
        self._config = {}
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to storage."""
        return self._connected
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the storage plugin."""
        self._config = config
        return await self.connect()
    
    async def shutdown(self) -> bool:
        """Shutdown the storage plugin."""
        return await self.disconnect()
    
    async def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate storage configuration."""
        # Basic validation - override in subclasses for specific requirements
        return True