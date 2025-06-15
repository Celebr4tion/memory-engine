"""
Module Registry for Memory Engine Orchestrator Integration

This module provides a registry system for inter-module communication and orchestration:
- Module registration and discovery
- Capability advertisement and negotiation
- Version compatibility checking
- Health monitoring and status tracking
- Dependency management
"""

import asyncio
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Callable, Tuple, Union
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
import weakref

from ..monitoring.structured_logger import get_logger

logger = get_logger(__name__, "ModuleRegistry")


class ModuleStatus(Enum):
    """Status of registered modules."""

    INITIALIZING = "initializing"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    DEPRECATED = "deprecated"


class CapabilityType(Enum):
    """Types of capabilities modules can provide."""

    KNOWLEDGE_STORAGE = "knowledge_storage"
    VECTOR_SEARCH = "vector_search"
    LLM_PROVIDER = "llm_provider"
    EMBEDDING_PROVIDER = "embedding_provider"
    GRAPH_ANALYSIS = "graph_analysis"
    DATA_EXPORT = "data_export"
    DATA_IMPORT = "data_import"
    QUERY_PROCESSING = "query_processing"
    EVENT_HANDLING = "event_handling"
    CUSTOM = "custom"


class VersionRequirement(Enum):
    """Version requirement types."""

    EXACT = "exact"
    MIN = "min"
    MAX = "max"
    RANGE = "range"
    COMPATIBLE = "compatible"


@dataclass
class Version:
    """Version information."""

    major: int
    minor: int
    patch: int
    pre_release: Optional[str] = None
    build: Optional[str] = None

    def __str__(self) -> str:
        version_str = f"{self.major}.{self.minor}.{self.patch}"
        if self.pre_release:
            version_str += f"-{self.pre_release}"
        if self.build:
            version_str += f"+{self.build}"
        return version_str

    def __lt__(self, other: "Version") -> bool:
        """Compare versions for ordering."""
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __eq__(self, other: "Version") -> bool:
        """Check version equality."""
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)

    def is_compatible(self, other: "Version") -> bool:
        """Check if versions are compatible (same major version)."""
        return self.major == other.major

    @classmethod
    def parse(cls, version_str: str) -> "Version":
        """Parse version string."""
        # Handle pre-release and build metadata
        parts = version_str.split("+")
        build = parts[1] if len(parts) > 1 else None

        parts = parts[0].split("-")
        pre_release = parts[1] if len(parts) > 1 else None

        # Parse main version
        version_parts = parts[0].split(".")
        major = int(version_parts[0])
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0
        patch = int(version_parts[2]) if len(version_parts) > 2 else 0

        return cls(major, minor, patch, pre_release, build)


@dataclass
class ModuleCapability:
    """Capability provided by a module."""

    capability_type: CapabilityType
    name: str
    description: str
    version: Version
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["capability_type"] = self.capability_type.value
        result["version"] = str(self.version)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModuleCapability":
        """Create from dictionary."""
        data["capability_type"] = CapabilityType(data["capability_type"])
        data["version"] = Version.parse(data["version"])
        return cls(**data)


@dataclass
class ModuleMetadata:
    """Metadata for registered modules."""

    module_id: str
    name: str
    description: str
    version: Version
    author: str
    capabilities: List[ModuleCapability] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)
    health_check_endpoint: Optional[str] = None
    documentation_url: Optional[str] = None
    source_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["version"] = str(self.version)
        result["capabilities"] = [cap.to_dict() for cap in self.capabilities]
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModuleMetadata":
        """Create from dictionary."""
        data["version"] = Version.parse(data["version"])
        if "capabilities" in data:
            data["capabilities"] = [ModuleCapability.from_dict(cap) for cap in data["capabilities"]]
        return cls(**data)


class ModuleInterface(ABC):
    """Interface that registered modules must implement."""

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the module."""
        pass

    @abstractmethod
    async def shutdown(self) -> bool:
        """Shutdown the module."""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        pass

    @abstractmethod
    def get_metadata(self) -> ModuleMetadata:
        """Get module metadata."""
        pass

    @abstractmethod
    def get_capabilities(self) -> List[ModuleCapability]:
        """Get module capabilities."""
        pass


@dataclass
class RegisteredModule:
    """Information about a registered module."""

    metadata: ModuleMetadata
    interface: Optional[ModuleInterface] = None
    status: ModuleStatus = ModuleStatus.INITIALIZING
    registration_time: float = field(default_factory=time.time)
    last_health_check: Optional[float] = None
    health_status: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    statistics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding interface)."""
        result = asdict(self)
        result.pop("interface", None)  # Remove interface from serialization
        result["metadata"] = self.metadata.to_dict()
        result["status"] = self.status.value
        return result


class DependencyResolver:
    """Resolves module dependencies."""

    def __init__(self):
        self.dependency_graph: Dict[str, Set[str]] = {}
        self.reverse_dependencies: Dict[str, Set[str]] = {}

    def add_module(self, module_id: str, dependencies: List[str]):
        """Add module dependencies."""
        self.dependency_graph[module_id] = set(dependencies)

        # Update reverse dependencies
        for dep in dependencies:
            if dep not in self.reverse_dependencies:
                self.reverse_dependencies[dep] = set()
            self.reverse_dependencies[dep].add(module_id)

    def remove_module(self, module_id: str):
        """Remove module and clean up dependencies."""
        # Remove from dependency graph
        dependencies = self.dependency_graph.pop(module_id, set())

        # Clean up reverse dependencies
        for dep in dependencies:
            if dep in self.reverse_dependencies:
                self.reverse_dependencies[dep].discard(module_id)
                if not self.reverse_dependencies[dep]:
                    del self.reverse_dependencies[dep]

        # Remove from reverse dependencies
        self.reverse_dependencies.pop(module_id, None)

    def get_initialization_order(self) -> List[str]:
        """Get modules in dependency order for initialization."""
        visited = set()
        temp_visited = set()
        result = []

        def visit(module_id: str):
            if module_id in temp_visited:
                raise ValueError(f"Circular dependency detected involving {module_id}")
            if module_id in visited:
                return

            temp_visited.add(module_id)

            # Visit dependencies first
            for dep in self.dependency_graph.get(module_id, set()):
                visit(dep)

            temp_visited.remove(module_id)
            visited.add(module_id)
            result.append(module_id)

        # Visit all modules
        for module_id in self.dependency_graph.keys():
            if module_id not in visited:
                visit(module_id)

        return result

    def get_shutdown_order(self) -> List[str]:
        """Get modules in reverse dependency order for shutdown."""
        return list(reversed(self.get_initialization_order()))

    def get_dependents(self, module_id: str) -> Set[str]:
        """Get modules that depend on the given module."""
        return self.reverse_dependencies.get(module_id, set()).copy()

    def validate_dependencies(self, available_modules: Set[str]) -> Dict[str, List[str]]:
        """Validate that all dependencies are available."""
        missing_deps = {}

        for module_id, dependencies in self.dependency_graph.items():
            missing = [dep for dep in dependencies if dep not in available_modules]
            if missing:
                missing_deps[module_id] = missing

        return missing_deps


class CapabilityMatcher:
    """Matches capability requirements with available capabilities."""

    def __init__(self):
        self.capabilities: Dict[str, List[Tuple[str, ModuleCapability]]] = {}

    def register_capability(self, module_id: str, capability: ModuleCapability):
        """Register a capability."""
        capability_key = f"{capability.capability_type.value}:{capability.name}"

        if capability_key not in self.capabilities:
            self.capabilities[capability_key] = []

        self.capabilities[capability_key].append((module_id, capability))

    def unregister_module_capabilities(self, module_id: str):
        """Unregister all capabilities for a module."""
        for capability_key in list(self.capabilities.keys()):
            self.capabilities[capability_key] = [
                (mid, cap) for mid, cap in self.capabilities[capability_key] if mid != module_id
            ]

            # Remove empty capability lists
            if not self.capabilities[capability_key]:
                del self.capabilities[capability_key]

    def find_capabilities(
        self,
        capability_type: CapabilityType,
        name: Optional[str] = None,
        version_requirement: Optional[Tuple[VersionRequirement, str]] = None,
    ) -> List[Tuple[str, ModuleCapability]]:
        """Find capabilities matching criteria."""
        results = []

        # Search pattern
        if name:
            search_key = f"{capability_type.value}:{name}"
            candidates = self.capabilities.get(search_key, [])
        else:
            # Search all capabilities of this type
            candidates = []
            prefix = f"{capability_type.value}:"
            for key, caps in self.capabilities.items():
                if key.startswith(prefix):
                    candidates.extend(caps)

        # Apply version filtering if specified
        if version_requirement:
            req_type, req_version = version_requirement
            required_version = Version.parse(req_version)

            for module_id, capability in candidates:
                if self._version_matches(capability.version, req_type, required_version):
                    results.append((module_id, capability))
        else:
            results = candidates

        # Sort by version (newest first)
        results.sort(key=lambda x: x[1].version, reverse=True)
        return results

    def _version_matches(
        self, available: Version, req_type: VersionRequirement, required: Version
    ) -> bool:
        """Check if version matches requirement."""
        if req_type == VersionRequirement.EXACT:
            return available == required
        elif req_type == VersionRequirement.MIN:
            return available >= required
        elif req_type == VersionRequirement.MAX:
            return available <= required
        elif req_type == VersionRequirement.COMPATIBLE:
            return available.is_compatible(required)
        else:
            return True

    def get_capability_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all available capabilities."""
        summary = {}

        for capability_key, providers in self.capabilities.items():
            capability_type, name = capability_key.split(":", 1)

            if capability_type not in summary:
                summary[capability_type] = {}

            summary[capability_type][name] = {
                "providers": len(providers),
                "versions": [str(cap.version) for _, cap in providers],
            }

        return summary


class ModuleRegistry:
    """
    Registry for managing modules in the Memory Engine orchestrator.

    Provides centralized registration, discovery, health monitoring,
    and dependency management for all system modules.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.modules: Dict[str, RegisteredModule] = {}
        self.dependency_resolver = DependencyResolver()
        self.capability_matcher = CapabilityMatcher()
        self.health_check_interval = self.config.get("health_check_interval", 30.0)
        self.storage_path = Path(self.config.get("storage_path", "./data/registry"))
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Threading and async control
        self._lock = threading.RLock()
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        self._running = False

        # Event callbacks
        self.on_module_registered: List[Callable[[str, ModuleMetadata], None]] = []
        self.on_module_unregistered: List[Callable[[str], None]] = []
        self.on_module_status_changed: List[Callable[[str, ModuleStatus], None]] = []

    async def initialize(self):
        """Initialize the registry."""
        await self._load_persistent_data()
        await self._start_health_monitoring()
        self._running = True
        logger.info("Module registry initialized")

    async def shutdown(self):
        """Shutdown the registry."""
        self._running = False
        self._shutdown_event.set()

        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        await self._shutdown_all_modules()
        await self._save_persistent_data()
        logger.info("Module registry shutdown")

    def register_module(
        self, metadata: ModuleMetadata, interface: Optional[ModuleInterface] = None
    ) -> bool:
        """Register a new module."""
        with self._lock:
            if metadata.module_id in self.modules:
                logger.warning(f"Module {metadata.module_id} already registered")
                return False

            # Validate dependencies
            available_modules = set(self.modules.keys())
            missing_deps = []
            for dep in metadata.dependencies:
                if dep not in available_modules:
                    missing_deps.append(dep)

            if missing_deps:
                logger.error(
                    f"Cannot register {metadata.module_id}: missing dependencies {missing_deps}"
                )
                return False

            # Create registered module
            registered_module = RegisteredModule(
                metadata=metadata, interface=interface, status=ModuleStatus.INITIALIZING
            )

            self.modules[metadata.module_id] = registered_module

            # Update dependency resolver
            self.dependency_resolver.add_module(metadata.module_id, metadata.dependencies)

            # Register capabilities
            for capability in metadata.capabilities:
                self.capability_matcher.register_capability(metadata.module_id, capability)

            # Trigger callbacks
            for callback in self.on_module_registered:
                try:
                    callback(metadata.module_id, metadata)
                except Exception as e:
                    logger.error(f"Module registration callback failed: {e}")

            logger.info(f"Registered module: {metadata.module_id} v{metadata.version}")
            return True

    def unregister_module(self, module_id: str) -> bool:
        """Unregister a module."""
        with self._lock:
            if module_id not in self.modules:
                logger.warning(f"Module {module_id} not found for unregistration")
                return False

            # Check for dependents
            dependents = self.dependency_resolver.get_dependents(module_id)
            if dependents:
                logger.error(f"Cannot unregister {module_id}: has dependents {list(dependents)}")
                return False

            # Remove from registry
            registered_module = self.modules.pop(module_id)

            # Clean up dependencies and capabilities
            self.dependency_resolver.remove_module(module_id)
            self.capability_matcher.unregister_module_capabilities(module_id)

            # Trigger callbacks
            for callback in self.on_module_unregistered:
                try:
                    callback(module_id)
                except Exception as e:
                    logger.error(f"Module unregistration callback failed: {e}")

            logger.info(f"Unregistered module: {module_id}")
            return True

    def get_module(self, module_id: str) -> Optional[RegisteredModule]:
        """Get registered module information."""
        with self._lock:
            return self.modules.get(module_id)

    def get_all_modules(self) -> Dict[str, RegisteredModule]:
        """Get all registered modules."""
        with self._lock:
            return self.modules.copy()

    def find_modules_by_capability(
        self,
        capability_type: CapabilityType,
        name: Optional[str] = None,
        version_requirement: Optional[Tuple[VersionRequirement, str]] = None,
    ) -> List[Tuple[str, RegisteredModule]]:
        """Find modules that provide a specific capability."""
        matching_capabilities = self.capability_matcher.find_capabilities(
            capability_type, name, version_requirement
        )

        result = []
        for module_id, capability in matching_capabilities:
            if module_id in self.modules:
                result.append((module_id, self.modules[module_id]))

        return result

    def find_modules_by_status(self, status: ModuleStatus) -> List[Tuple[str, RegisteredModule]]:
        """Find modules with specific status."""
        with self._lock:
            return [
                (module_id, module)
                for module_id, module in self.modules.items()
                if module.status == status
            ]

    def update_module_status(
        self, module_id: str, status: ModuleStatus, error_message: Optional[str] = None
    ):
        """Update module status."""
        with self._lock:
            if module_id not in self.modules:
                return False

            old_status = self.modules[module_id].status
            self.modules[module_id].status = status

            if error_message:
                self.modules[module_id].error_message = error_message

            # Trigger callbacks if status changed
            if old_status != status:
                for callback in self.on_module_status_changed:
                    try:
                        callback(module_id, status)
                    except Exception as e:
                        logger.error(f"Status change callback failed: {e}")

            return True

    async def initialize_modules(self) -> Dict[str, bool]:
        """Initialize all registered modules in dependency order."""
        initialization_results = {}

        try:
            init_order = self.dependency_resolver.get_initialization_order()
            logger.info(f"Initializing modules in order: {init_order}")

            for module_id in init_order:
                if module_id not in self.modules:
                    continue

                registered_module = self.modules[module_id]

                if registered_module.interface is None:
                    logger.warning(f"Module {module_id} has no interface, skipping initialization")
                    initialization_results[module_id] = False
                    continue

                try:
                    self.update_module_status(module_id, ModuleStatus.INITIALIZING)

                    config = registered_module.metadata.configuration
                    success = await registered_module.interface.initialize(config)

                    if success:
                        self.update_module_status(module_id, ModuleStatus.ACTIVE)
                        logger.info(f"Successfully initialized module: {module_id}")
                    else:
                        self.update_module_status(
                            module_id, ModuleStatus.ERROR, "Initialization returned False"
                        )
                        logger.error(f"Failed to initialize module: {module_id}")

                    initialization_results[module_id] = success

                except Exception as e:
                    error_msg = f"Initialization error: {e}"
                    self.update_module_status(module_id, ModuleStatus.ERROR, error_msg)
                    logger.error(f"Exception initializing module {module_id}: {e}")
                    initialization_results[module_id] = False

        except Exception as e:
            logger.error(f"Error during module initialization: {e}")

        return initialization_results

    async def shutdown_modules(self) -> Dict[str, bool]:
        """Shutdown all modules in reverse dependency order."""
        shutdown_results = {}

        try:
            shutdown_order = self.dependency_resolver.get_shutdown_order()
            logger.info(f"Shutting down modules in order: {shutdown_order}")

            for module_id in shutdown_order:
                if module_id not in self.modules:
                    continue

                registered_module = self.modules[module_id]

                if registered_module.interface is None:
                    shutdown_results[module_id] = True
                    continue

                try:
                    success = await registered_module.interface.shutdown()

                    if success:
                        self.update_module_status(module_id, ModuleStatus.INACTIVE)
                        logger.info(f"Successfully shutdown module: {module_id}")
                    else:
                        self.update_module_status(
                            module_id, ModuleStatus.ERROR, "Shutdown returned False"
                        )
                        logger.error(f"Failed to shutdown module: {module_id}")

                    shutdown_results[module_id] = success

                except Exception as e:
                    error_msg = f"Shutdown error: {e}"
                    self.update_module_status(module_id, ModuleStatus.ERROR, error_msg)
                    logger.error(f"Exception shutting down module {module_id}: {e}")
                    shutdown_results[module_id] = False

        except Exception as e:
            logger.error(f"Error during module shutdown: {e}")

        return shutdown_results

    async def perform_health_checks(self) -> Dict[str, Dict[str, Any]]:
        """Perform health checks on all active modules."""
        health_results = {}

        active_modules = [
            (module_id, module)
            for module_id, module in self.modules.items()
            if module.status == ModuleStatus.ACTIVE and module.interface is not None
        ]

        for module_id, registered_module in active_modules:
            try:
                health_status = await registered_module.interface.health_check()
                registered_module.last_health_check = time.time()
                registered_module.health_status = health_status

                health_results[module_id] = health_status

                # Update status based on health check
                if health_status.get("status") == "error":
                    self.update_module_status(
                        module_id,
                        ModuleStatus.ERROR,
                        health_status.get("error", "Health check failed"),
                    )

            except Exception as e:
                error_msg = f"Health check error: {e}"
                self.update_module_status(module_id, ModuleStatus.ERROR, error_msg)
                health_results[module_id] = {"status": "error", "error": error_msg}
                logger.error(f"Health check failed for module {module_id}: {e}")

        return health_results

    def get_registry_summary(self) -> Dict[str, Any]:
        """Get comprehensive registry summary."""
        with self._lock:
            status_counts = {}
            for module in self.modules.values():
                status = module.status.value
                status_counts[status] = status_counts.get(status, 0) + 1

            return {
                "total_modules": len(self.modules),
                "status_distribution": status_counts,
                "capabilities": self.capability_matcher.get_capability_summary(),
                "dependency_chains": len(self.dependency_resolver.dependency_graph),
                "health_monitoring": self._running,
            }

    def get_dependency_graph(self) -> Dict[str, Any]:
        """Get dependency graph information."""
        return {
            "dependencies": dict(self.dependency_resolver.dependency_graph),
            "reverse_dependencies": dict(self.dependency_resolver.reverse_dependencies),
            "initialization_order": self.dependency_resolver.get_initialization_order(),
            "shutdown_order": self.dependency_resolver.get_shutdown_order(),
        }

    async def _start_health_monitoring(self):
        """Start periodic health monitoring."""
        if self.health_check_interval <= 0:
            return

        self._health_check_task = asyncio.create_task(self._health_monitor_loop())

    async def _health_monitor_loop(self):
        """Health monitoring loop."""
        while self._running:
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(), timeout=self.health_check_interval
                )
                break  # Shutdown requested
            except asyncio.TimeoutError:
                # Perform health checks
                await self.perform_health_checks()

    async def _shutdown_all_modules(self):
        """Shutdown all modules during registry shutdown."""
        await self.shutdown_modules()

    async def _save_persistent_data(self):
        """Save registry data to persistent storage."""
        try:
            registry_data = {
                "modules": {
                    module_id: module.to_dict() for module_id, module in self.modules.items()
                },
                "timestamp": time.time(),
            }

            with open(self.storage_path / "registry.json", "w") as f:
                json.dump(registry_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save registry data: {e}")

    async def _load_persistent_data(self):
        """Load registry data from persistent storage."""
        registry_file = self.storage_path / "registry.json"

        if not registry_file.exists():
            return

        try:
            with open(registry_file, "r") as f:
                registry_data = json.load(f)

            # Load modules (without interfaces - those need to be re-registered)
            for module_id, module_data in registry_data.get("modules", {}).items():
                metadata = ModuleMetadata.from_dict(module_data["metadata"])
                registered_module = RegisteredModule(
                    metadata=metadata,
                    interface=None,  # Interfaces must be re-registered
                    status=ModuleStatus.INACTIVE,  # Start as inactive
                )

                self.modules[module_id] = registered_module

                # Update dependency resolver
                self.dependency_resolver.add_module(module_id, metadata.dependencies)

                # Register capabilities
                for capability in metadata.capabilities:
                    self.capability_matcher.register_capability(module_id, capability)

            logger.info(f"Loaded {len(self.modules)} modules from persistent storage")

        except Exception as e:
            logger.error(f"Failed to load registry data: {e}")


# Convenience functions for common operations
def create_module_metadata(
    module_id: str, name: str, version_str: str, description: str = "", author: str = ""
) -> ModuleMetadata:
    """Create module metadata with basic information."""
    return ModuleMetadata(
        module_id=module_id,
        name=name,
        description=description,
        version=Version.parse(version_str),
        author=author,
    )


def create_capability(
    capability_type: CapabilityType, name: str, version_str: str, description: str = ""
) -> ModuleCapability:
    """Create a module capability."""
    return ModuleCapability(
        capability_type=capability_type,
        name=name,
        description=description,
        version=Version.parse(version_str),
    )
