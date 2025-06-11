"""
Storage factory for creating graph storage backend instances.

This module provides a factory function to instantiate the appropriate
graph storage backend based on configuration settings.
"""
import logging
from typing import Dict, Any, Optional, List

from memory_core.storage.interfaces.graph_storage_interface import GraphStorageInterface
from memory_core.config import get_config


class StorageFactory:
    """
    Factory class for creating graph storage backend instances.
    
    This factory creates and configures storage backends based on the
    application configuration, providing a unified way to instantiate
    different storage implementations.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._backends = {}
        self._register_backends()
    
    def _register_backends(self):
        """Register available storage backends."""
        try:
            from memory_core.storage.backends.janusgraph import JanusGraphStorage
            self._backends['janusgraph'] = JanusGraphStorage
        except ImportError:
            self.logger.warning("JanusGraph backend not available (missing dependencies)")
        
        try:
            from memory_core.storage.backends.json_file import JsonFileStorage
            self._backends['json_file'] = JsonFileStorage
        except ImportError:
            self.logger.warning("JSON file backend not available")
        
        try:
            from memory_core.storage.backends.sqlite import SqliteStorage
            self._backends['sqlite'] = SqliteStorage
        except ImportError:
            self.logger.warning("SQLite backend not available (missing aiosqlite)")
    
    def create_storage(self, backend_type: Optional[str] = None, 
                      config_override: Optional[Dict[str, Any]] = None) -> GraphStorageInterface:
        """
        Create a graph storage backend instance.
        
        Args:
            backend_type: Type of backend to create ('janusgraph', 'json_file', 'sqlite').
                         If None, uses configuration setting.
            config_override: Optional configuration override for the backend.
            
        Returns:
            Configured storage backend instance
            
        Raises:
            ValueError: If the backend type is not supported
            ImportError: If the backend dependencies are not available
        """
        config = get_config()
        
        # Determine backend type
        if backend_type is None:
            backend_type = getattr(config.config.storage.graph, 'backend', 'janusgraph')
        
        if backend_type not in self._backends:
            available_backends = list(self._backends.keys())
            raise ValueError(f"Unsupported backend type '{backend_type}'. "
                           f"Available backends: {available_backends}")
        
        backend_class = self._backends[backend_type]
        
        # Get backend-specific configuration
        backend_config = self._get_backend_config(backend_type, config, config_override)
        
        # Create backend instance
        try:
            if backend_type == 'janusgraph':
                return self._create_janusgraph_storage(backend_class, backend_config)
            elif backend_type == 'json_file':
                return self._create_json_file_storage(backend_class, backend_config)
            elif backend_type == 'sqlite':
                return self._create_sqlite_storage(backend_class, backend_config)
            else:
                # Generic instantiation
                return backend_class(**backend_config)
                
        except Exception as e:
            self.logger.error(f"Failed to create {backend_type} storage backend: {e}")
            raise
    
    def _get_backend_config(self, backend_type: str, config: Any, 
                           config_override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Get configuration for a specific backend."""
        backend_config = {}
        
        # Get base configuration from config object
        storage_config = getattr(config.config, 'storage', None)
        if storage_config:
            graph_config = getattr(storage_config, 'graph', None)
            if graph_config:
                backend_specific_config = getattr(graph_config, backend_type, None)
                if backend_specific_config:
                    backend_config = dict(backend_specific_config)
        
        # Apply override configuration
        if config_override:
            backend_config.update(config_override)
        
        return backend_config
    
    def _create_janusgraph_storage(self, backend_class, config: Dict[str, Any]):
        """Create JanusGraph storage instance with proper configuration."""
        return backend_class(
            host=config.get('host'),
            port=config.get('port'),
            traversal_source=config.get('traversal_source', 'g')
        )
    
    def _create_json_file_storage(self, backend_class, config: Dict[str, Any]):
        """Create JSON file storage instance with proper configuration."""
        return backend_class(
            directory=config.get('directory', './data/graph'),
            pretty_print=config.get('pretty_print', True)
        )
    
    def _create_sqlite_storage(self, backend_class, config: Dict[str, Any]):
        """Create SQLite storage instance with proper configuration."""
        return backend_class(
            database_path=config.get('database_path', './data/knowledge.db')
        )
    
    def list_available_backends(self) -> List[str]:
        """
        List all available storage backends.
        
        Returns:
            List of backend type names
        """
        return list(self._backends.keys())
    
    def is_backend_available(self, backend_type: str) -> bool:
        """
        Check if a specific backend is available.
        
        Args:
            backend_type: Type of backend to check
            
        Returns:
            True if backend is available, False otherwise
        """
        return backend_type in self._backends


# Global factory instance
_storage_factory = StorageFactory()


def create_storage(backend_type: Optional[str] = None, 
                  config_override: Optional[Dict[str, Any]] = None) -> GraphStorageInterface:
    """
    Create a graph storage backend instance using the global factory.
    
    Args:
        backend_type: Type of backend to create ('janusgraph', 'json_file', 'sqlite').
                     If None, uses configuration setting.
        config_override: Optional configuration override for the backend.
        
    Returns:
        Configured storage backend instance
        
    Raises:
        ValueError: If the backend type is not supported
        ImportError: If the backend dependencies are not available
    """
    return _storage_factory.create_storage(backend_type, config_override)


def list_available_backends() -> List[str]:
    """
    List all available storage backends.
    
    Returns:
        List of backend type names
    """
    return _storage_factory.list_available_backends()


def is_backend_available(backend_type: str) -> bool:
    """
    Check if a specific backend is available.
    
    Args:
        backend_type: Type of backend to check
        
    Returns:
        True if backend is available, False otherwise
    """
    return _storage_factory.is_backend_available(backend_type)