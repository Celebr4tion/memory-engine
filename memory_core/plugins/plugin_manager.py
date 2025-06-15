"""
Plugin manager for Memory Engine.
"""

import os
import sys
import importlib
import importlib.util
import inspect
import pkgutil
from typing import Dict, List, Any, Optional, Type, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class PluginInfo:
    """Information about a plugin."""

    name: str
    version: str
    description: str
    author: str
    plugin_type: str
    module_path: str
    class_name: str
    dependencies: List[str]
    config_schema: Dict[str, Any]
    is_loaded: bool = False
    instance: Optional[Any] = None


class PluginInterface(ABC):
    """Base interface for all plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Plugin description."""
        pass

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin."""
        pass

    @abstractmethod
    async def shutdown(self) -> bool:
        """Shutdown the plugin."""
        pass

    async def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate plugin configuration."""
        return True


class PluginManager:
    """Manages plugin discovery, loading, and lifecycle."""

    def __init__(self, plugin_directories: List[str] = None):
        self.plugin_directories = plugin_directories or []
        self._plugins: Dict[str, PluginInfo] = {}
        self._loaded_plugins: Dict[str, Any] = {}
        self._plugin_types: Dict[str, Type[PluginInterface]] = {}

        # Default plugin directories
        self._add_default_directories()

    def _add_default_directories(self):
        """Add default plugin directories."""
        # Current package plugins
        current_dir = os.path.dirname(__file__)
        self.plugin_directories.append(os.path.join(current_dir, "builtin"))

        # User plugins directory
        user_dir = os.path.expanduser("~/.memory-engine/plugins")
        if os.path.exists(user_dir):
            self.plugin_directories.append(user_dir)

        # System plugins directory
        system_dir = "/usr/local/share/memory-engine/plugins"
        if os.path.exists(system_dir):
            self.plugin_directories.append(system_dir)

    def register_plugin_type(self, plugin_type: str, interface_class: Type[PluginInterface]):
        """Register a plugin type with its interface."""
        self._plugin_types[plugin_type] = interface_class

    async def discover_plugins(self) -> List[PluginInfo]:
        """Discover all available plugins."""
        discovered = []

        for directory in self.plugin_directories:
            if not os.path.exists(directory):
                continue

            logger.info(f"Scanning plugin directory: {directory}")

            try:
                # Scan Python packages
                discovered.extend(await self._discover_python_plugins(directory))

                # Scan standalone plugin files
                discovered.extend(await self._discover_standalone_plugins(directory))

            except Exception as e:
                logger.error(f"Error scanning plugin directory {directory}: {e}")

        # Update plugin registry
        for plugin_info in discovered:
            self._plugins[plugin_info.name] = plugin_info

        logger.info(f"Discovered {len(discovered)} plugins")
        return discovered

    async def _discover_python_plugins(self, directory: str) -> List[PluginInfo]:
        """Discover Python package plugins."""
        plugins = []

        # Add directory to Python path temporarily
        if directory not in sys.path:
            sys.path.insert(0, directory)

        try:
            # Look for packages with plugin.json
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)

                if os.path.isdir(item_path):
                    plugin_json = os.path.join(item_path, "plugin.json")

                    if os.path.exists(plugin_json):
                        try:
                            plugin_info = await self._load_plugin_info(plugin_json, item_path)
                            if plugin_info:
                                plugins.append(plugin_info)
                        except Exception as e:
                            logger.error(f"Error loading plugin info from {plugin_json}: {e}")

        finally:
            # Remove from path
            if directory in sys.path:
                sys.path.remove(directory)

        return plugins

    async def _discover_standalone_plugins(self, directory: str) -> List[PluginInfo]:
        """Discover standalone Python file plugins."""
        plugins = []

        for filename in os.listdir(directory):
            if filename.endswith(".py") and not filename.startswith("_"):
                file_path = os.path.join(directory, filename)

                try:
                    plugin_info = await self._inspect_python_file(file_path)
                    if plugin_info:
                        plugins.append(plugin_info)
                except Exception as e:
                    logger.error(f"Error inspecting plugin file {file_path}: {e}")

        return plugins

    async def _load_plugin_info(
        self, plugin_json_path: str, plugin_dir: str
    ) -> Optional[PluginInfo]:
        """Load plugin info from plugin.json."""
        try:
            with open(plugin_json_path, "r") as f:
                data = json.load(f)

            return PluginInfo(
                name=data["name"],
                version=data["version"],
                description=data.get("description", ""),
                author=data.get("author", ""),
                plugin_type=data["type"],
                module_path=os.path.join(plugin_dir, data["module"]),
                class_name=data["class"],
                dependencies=data.get("dependencies", []),
                config_schema=data.get("config_schema", {}),
            )

        except Exception as e:
            logger.error(f"Error parsing plugin.json {plugin_json_path}: {e}")
            return None

    async def _inspect_python_file(self, file_path: str) -> Optional[PluginInfo]:
        """Inspect Python file for plugin classes."""
        try:
            module_name = os.path.basename(file_path)[:-3]  # Remove .py
            spec = importlib.util.spec_from_file_location(module_name, file_path)

            if not spec or not spec.loader:
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Look for plugin classes
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    hasattr(obj, "__plugin_info__")
                    and issubclass(obj, PluginInterface)
                    and obj != PluginInterface
                ):

                    info = obj.__plugin_info__

                    return PluginInfo(
                        name=info.get("name", name),
                        version=info.get("version", "1.0.0"),
                        description=info.get("description", ""),
                        author=info.get("author", ""),
                        plugin_type=info.get("type", "unknown"),
                        module_path=file_path,
                        class_name=name,
                        dependencies=info.get("dependencies", []),
                        config_schema=info.get("config_schema", {}),
                    )

            return None

        except Exception as e:
            logger.error(f"Error inspecting plugin file {file_path}: {e}")
            return None

    async def load_plugin(self, plugin_name: str, config: Dict[str, Any] = None) -> Optional[Any]:
        """Load and initialize a plugin."""
        if plugin_name not in self._plugins:
            logger.error(f"Plugin {plugin_name} not found")
            return None

        plugin_info = self._plugins[plugin_name]

        if plugin_info.is_loaded:
            logger.warning(f"Plugin {plugin_name} is already loaded")
            return plugin_info.instance

        try:
            # Check dependencies
            if not await self._check_dependencies(plugin_info):
                logger.error(f"Plugin {plugin_name} dependencies not satisfied")
                return None

            # Load the plugin module
            instance = await self._load_plugin_instance(plugin_info)
            if not instance:
                return None

            # Validate configuration
            if config and hasattr(instance, "validate_config"):
                if not await instance.validate_config(config):
                    logger.error(f"Plugin {plugin_name} configuration validation failed")
                    return None

            # Initialize plugin
            if await instance.initialize(config or {}):
                plugin_info.instance = instance
                plugin_info.is_loaded = True
                self._loaded_plugins[plugin_name] = instance

                logger.info(f"Plugin {plugin_name} loaded successfully")
                return instance
            else:
                logger.error(f"Plugin {plugin_name} initialization failed")
                return None

        except Exception as e:
            logger.error(f"Error loading plugin {plugin_name}: {e}")
            return None

    async def _check_dependencies(self, plugin_info: PluginInfo) -> bool:
        """Check if plugin dependencies are satisfied."""
        for dependency in plugin_info.dependencies:
            try:
                importlib.import_module(dependency)
            except ImportError:
                logger.error(f"Plugin {plugin_info.name} missing dependency: {dependency}")
                return False

        return True

    async def _load_plugin_instance(self, plugin_info: PluginInfo) -> Optional[Any]:
        """Load plugin instance from module."""
        try:
            if os.path.isfile(plugin_info.module_path):
                # Load from file
                module_name = os.path.basename(plugin_info.module_path)[:-3]
                spec = importlib.util.spec_from_file_location(module_name, plugin_info.module_path)

                if not spec or not spec.loader:
                    return None

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            else:
                # Load from package
                module_name = os.path.basename(plugin_info.module_path)
                sys.path.insert(0, os.path.dirname(plugin_info.module_path))
                try:
                    module = importlib.import_module(module_name)
                finally:
                    sys.path.pop(0)

            # Get plugin class
            plugin_class = getattr(module, plugin_info.class_name)

            # Verify it's a valid plugin
            if not issubclass(plugin_class, PluginInterface):
                logger.error(
                    f"Plugin class {plugin_info.class_name} does not implement PluginInterface"
                )
                return None

            # Create instance
            return plugin_class()

        except Exception as e:
            logger.error(f"Error loading plugin instance {plugin_info.name}: {e}")
            return None

    async def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin."""
        if plugin_name not in self._loaded_plugins:
            return False

        try:
            plugin = self._loaded_plugins[plugin_name]

            # Shutdown plugin
            if hasattr(plugin, "shutdown"):
                await plugin.shutdown()

            # Remove from loaded plugins
            del self._loaded_plugins[plugin_name]

            # Update plugin info
            if plugin_name in self._plugins:
                self._plugins[plugin_name].is_loaded = False
                self._plugins[plugin_name].instance = None

            logger.info(f"Plugin {plugin_name} unloaded successfully")
            return True

        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_name}: {e}")
            return False

    async def reload_plugin(self, plugin_name: str, config: Dict[str, Any] = None) -> Optional[Any]:
        """Reload a plugin."""
        await self.unload_plugin(plugin_name)
        return await self.load_plugin(plugin_name, config)

    def get_plugin(self, plugin_name: str) -> Optional[Any]:
        """Get a loaded plugin instance."""
        return self._loaded_plugins.get(plugin_name)

    def get_plugins_by_type(self, plugin_type: str) -> List[Any]:
        """Get all loaded plugins of a specific type."""
        return [
            plugin
            for plugin_name, plugin in self._loaded_plugins.items()
            if self._plugins[plugin_name].plugin_type == plugin_type
        ]

    def list_available_plugins(self) -> List[PluginInfo]:
        """List all available plugins."""
        return list(self._plugins.values())

    def list_loaded_plugins(self) -> List[PluginInfo]:
        """List all loaded plugins."""
        return [info for info in self._plugins.values() if info.is_loaded]

    async def shutdown_all(self):
        """Shutdown all loaded plugins."""
        for plugin_name in list(self._loaded_plugins.keys()):
            await self.unload_plugin(plugin_name)


# Decorator for marking plugin classes
def plugin(
    name: str = None,
    version: str = "1.0.0",
    description: str = "",
    author: str = "",
    plugin_type: str = "unknown",
    dependencies: List[str] = None,
    config_schema: Dict[str, Any] = None,
):
    """Decorator to mark a class as a plugin."""

    def decorator(cls):
        cls.__plugin_info__ = {
            "name": name or cls.__name__,
            "version": version,
            "description": description,
            "author": author,
            "type": plugin_type,
            "dependencies": dependencies or [],
            "config_schema": config_schema or {},
        }
        return cls

    return decorator


# Plugin validation utilities
def validate_plugin_config(config: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    """Validate plugin configuration against schema."""
    errors = []

    # Basic validation - could be extended with jsonschema
    for key, requirements in schema.items():
        if requirements.get("required", False) and key not in config:
            errors.append(f"Required configuration key '{key}' is missing")

        if key in config:
            expected_type = requirements.get("type")
            if expected_type and not isinstance(config[key], expected_type):
                errors.append(f"Configuration key '{key}' should be of type {expected_type}")

    return errors
