"""
Plugin registry for managing plugin metadata and discovery.
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    description: str
    author: str
    plugin_type: str
    tags: List[str]
    homepage: str
    repository: str
    license: str
    min_memory_engine_version: str
    max_memory_engine_version: str
    platform_support: List[str]
    python_requires: str
    dependencies: List[str]
    optional_dependencies: Dict[str, List[str]]
    config_schema: Dict[str, Any]
    examples: List[Dict[str, Any]]
    changelog: List[Dict[str, str]]


class PluginRegistry:
    """Registry for plugin metadata and discovery."""
    
    def __init__(self, registry_file: str = None):
        self.registry_file = registry_file or self._get_default_registry_file()
        self._plugins: Dict[str, PluginMetadata] = {}
        self._categories: Dict[str, Set[str]] = {}
        self._tags: Dict[str, Set[str]] = {}
    
    def _get_default_registry_file(self) -> str:
        """Get default registry file path."""
        user_dir = os.path.expanduser('~/.memory-engine')
        os.makedirs(user_dir, exist_ok=True)
        return os.path.join(user_dir, 'plugin_registry.json')
    
    async def load_registry(self) -> bool:
        """Load plugin registry from file."""
        if not os.path.exists(self.registry_file):
            logger.info("Plugin registry file not found, starting with empty registry")
            return True
        
        try:
            with open(self.registry_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self._plugins = {}
            for plugin_name, plugin_data in data.get('plugins', {}).items():
                metadata = PluginMetadata(**plugin_data)
                self._plugins[plugin_name] = metadata
                
                # Update indices
                self._update_indices(metadata)
            
            logger.info(f"Loaded {len(self._plugins)} plugins from registry")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load plugin registry: {e}")
            return False
    
    async def save_registry(self) -> bool:
        """Save plugin registry to file."""
        try:
            data = {
                'version': '1.0',
                'plugins': {
                    name: asdict(metadata) 
                    for name, metadata in self._plugins.items()
                }
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.registry_file), exist_ok=True)
            
            # Write to temporary file first
            temp_file = self.registry_file + '.tmp'
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Atomic move
            os.replace(temp_file, self.registry_file)
            
            logger.info(f"Saved plugin registry with {len(self._plugins)} plugins")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save plugin registry: {e}")
            return False
    
    def register_plugin(self, metadata: PluginMetadata) -> bool:
        """Register a plugin in the registry."""
        try:
            self._plugins[metadata.name] = metadata
            self._update_indices(metadata)
            
            logger.info(f"Registered plugin: {metadata.name} v{metadata.version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register plugin {metadata.name}: {e}")
            return False
    
    def unregister_plugin(self, plugin_name: str) -> bool:
        """Unregister a plugin from the registry."""
        if plugin_name not in self._plugins:
            return False
        
        try:
            metadata = self._plugins[plugin_name]
            del self._plugins[plugin_name]
            
            # Update indices
            self._remove_from_indices(metadata)
            
            logger.info(f"Unregistered plugin: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister plugin {plugin_name}: {e}")
            return False
    
    def get_plugin(self, plugin_name: str) -> Optional[PluginMetadata]:
        """Get plugin metadata by name."""
        return self._plugins.get(plugin_name)
    
    def list_plugins(self, plugin_type: str = None, 
                    tags: List[str] = None) -> List[PluginMetadata]:
        """List plugins with optional filtering."""
        plugins = list(self._plugins.values())
        
        if plugin_type:
            plugins = [p for p in plugins if p.plugin_type == plugin_type]
        
        if tags:
            plugins = [
                p for p in plugins 
                if any(tag in p.tags for tag in tags)
            ]
        
        return sorted(plugins, key=lambda p: p.name)
    
    def search_plugins(self, query: str) -> List[PluginMetadata]:
        """Search plugins by name, description, or tags."""
        query_lower = query.lower()
        results = []
        
        for metadata in self._plugins.values():
            score = 0
            
            # Name match (highest score)
            if query_lower in metadata.name.lower():
                score += 10
            
            # Description match
            if query_lower in metadata.description.lower():
                score += 5
            
            # Tags match
            for tag in metadata.tags:
                if query_lower in tag.lower():
                    score += 3
            
            # Author match
            if query_lower in metadata.author.lower():
                score += 2
            
            if score > 0:
                results.append((metadata, score))
        
        # Sort by score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return [metadata for metadata, score in results]
    
    def get_plugins_by_category(self, category: str) -> List[PluginMetadata]:
        """Get plugins by category (plugin type)."""
        return [
            metadata for metadata in self._plugins.values()
            if metadata.plugin_type == category
        ]
    
    def get_plugins_by_tag(self, tag: str) -> List[PluginMetadata]:
        """Get plugins by tag."""
        return [
            metadata for metadata in self._plugins.values()
            if tag in metadata.tags
        ]
    
    def get_categories(self) -> List[str]:
        """Get list of available plugin categories."""
        return sorted(self._categories.keys())
    
    def get_tags(self) -> List[str]:
        """Get list of available tags."""
        return sorted(self._tags.keys())
    
    def get_plugin_count(self) -> int:
        """Get total number of registered plugins."""
        return len(self._plugins)
    
    def get_category_counts(self) -> Dict[str, int]:
        """Get plugin count by category."""
        return {
            category: len(plugins)
            for category, plugins in self._categories.items()
        }
    
    def validate_plugin_metadata(self, metadata: PluginMetadata) -> List[str]:
        """Validate plugin metadata."""
        errors = []
        
        if not metadata.name:
            errors.append("Plugin name is required")
        
        if not metadata.version:
            errors.append("Plugin version is required")
        
        if not metadata.plugin_type:
            errors.append("Plugin type is required")
        
        if not metadata.description:
            errors.append("Plugin description is required")
        
        # Check version format (basic)
        if metadata.version and not self._is_valid_version(metadata.version):
            errors.append("Invalid version format")
        
        # Check if name conflicts with existing plugin
        if (metadata.name in self._plugins and 
            self._plugins[metadata.name].version != metadata.version):
            errors.append(f"Plugin {metadata.name} already exists with different version")
        
        return errors
    
    def _is_valid_version(self, version: str) -> bool:
        """Check if version string is valid (basic check)."""
        try:
            parts = version.split('.')
            return len(parts) >= 2 and all(part.isdigit() for part in parts[:2])
        except:
            return False
    
    def _update_indices(self, metadata: PluginMetadata):
        """Update internal indices."""
        # Category index
        if metadata.plugin_type not in self._categories:
            self._categories[metadata.plugin_type] = set()
        self._categories[metadata.plugin_type].add(metadata.name)
        
        # Tags index
        for tag in metadata.tags:
            if tag not in self._tags:
                self._tags[tag] = set()
            self._tags[tag].add(metadata.name)
    
    def _remove_from_indices(self, metadata: PluginMetadata):
        """Remove from internal indices."""
        # Category index
        if metadata.plugin_type in self._categories:
            self._categories[metadata.plugin_type].discard(metadata.name)
            if not self._categories[metadata.plugin_type]:
                del self._categories[metadata.plugin_type]
        
        # Tags index
        for tag in metadata.tags:
            if tag in self._tags:
                self._tags[tag].discard(metadata.name)
                if not self._tags[tag]:
                    del self._tags[tag]
    
    async def update_from_remote(self, registry_url: str) -> bool:
        """Update registry from remote source."""
        # This would fetch plugin metadata from a remote registry
        # Implementation depends on the registry format and protocol
        logger.warning("Remote registry update not implemented yet")
        return False
    
    async def publish_plugin(self, metadata: PluginMetadata, 
                           plugin_package_path: str) -> bool:
        """Publish plugin to remote registry."""
        # This would upload plugin to a remote registry
        # Implementation depends on the registry service
        logger.warning("Plugin publishing not implemented yet")
        return False
    
    def export_metadata(self, plugin_name: str, output_file: str) -> bool:
        """Export plugin metadata to file."""
        if plugin_name not in self._plugins:
            return False
        
        try:
            metadata = self._plugins[plugin_name]
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(metadata), f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported metadata for {plugin_name} to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export metadata for {plugin_name}: {e}")
            return False
    
    def import_metadata(self, metadata_file: str) -> bool:
        """Import plugin metadata from file."""
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            metadata = PluginMetadata(**data)
            return self.register_plugin(metadata)
            
        except Exception as e:
            logger.error(f"Failed to import metadata from {metadata_file}: {e}")
            return False