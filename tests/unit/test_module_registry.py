"""Unit tests for module registry system."""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any

from memory_core.orchestrator.module_registry import (
    ModuleRegistry,
    ModuleInterface,
    ModuleMetadata,
    ModuleCapability,
    CapabilityType,
    Version,
    ModuleStatus,
    RegistryError,
    ModuleNotFoundError,
    CapabilityMismatchError
)


class MockModule(ModuleInterface):
    """Mock module for testing."""
    
    def __init__(self, module_id: str):
        self.module_id = module_id
        self.initialized = False
        self.config = {}
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        self.config = config
        self.initialized = True
        return True
    
    async def shutdown(self) -> bool:
        self.initialized = False
        return True
    
    async def health_check(self) -> Dict[str, Any]:
        return {
            'status': 'healthy' if self.initialized else 'unhealthy',
            'initialized': self.initialized
        }
    
    async def process(self, data: Any) -> Any:
        if not self.initialized:
            raise RuntimeError("Module not initialized")
        return {'processed': data}


class TestModuleRegistry:
    """Test the module registry functionality."""
    
    @pytest.fixture
    def registry(self):
        """Create a fresh registry instance."""
        return ModuleRegistry()
    
    @pytest.fixture
    def sample_metadata(self):
        """Create sample module metadata."""
        return ModuleMetadata(
            module_id='test-module-1',
            name='Test Module',
            version=Version(major=1, minor=0, patch=0),
            description='A test module for unit testing',
            capabilities=[
                ModuleCapability(
                    capability_type=CapabilityType.STORAGE,
                    description='Can store data',
                    parameters={'max_size': '1GB'}
                ),
                ModuleCapability(
                    capability_type=CapabilityType.PROCESSING,
                    description='Can process text',
                    parameters={'formats': ['text', 'json']}
                )
            ],
            author='Test Author',
            license='MIT'
        )
    
    @pytest.mark.asyncio
    async def test_register_module(self, registry, sample_metadata):
        """Test registering a module."""
        module = MockModule('test-module-1')
        
        # Register module
        success = await registry.register_module(sample_metadata, module)
        assert success is True
        
        # Check module is registered
        modules = await registry.list_modules()
        assert len(modules) == 1
        assert modules[0].module_id == 'test-module-1'
        assert modules[0].status == ModuleStatus.INACTIVE
    
    @pytest.mark.asyncio
    async def test_register_duplicate_module(self, registry, sample_metadata):
        """Test registering duplicate module fails."""
        module1 = MockModule('test-module-1')
        module2 = MockModule('test-module-1')
        
        # First registration should succeed
        await registry.register_module(sample_metadata, module1)
        
        # Second registration should fail
        with pytest.raises(RegistryError):
            await registry.register_module(sample_metadata, module2)
    
    @pytest.mark.asyncio
    async def test_unregister_module(self, registry, sample_metadata):
        """Test unregistering a module."""
        module = MockModule('test-module-1')
        
        # Register and then unregister
        await registry.register_module(sample_metadata, module)
        success = await registry.unregister_module('test-module-1')
        assert success is True
        
        # Module should be gone
        modules = await registry.list_modules()
        assert len(modules) == 0
    
    @pytest.mark.asyncio
    async def test_unregister_nonexistent_module(self, registry):
        """Test unregistering non-existent module fails."""
        with pytest.raises(ModuleNotFoundError):
            await registry.unregister_module('non-existent')
    
    @pytest.mark.asyncio
    async def test_get_module(self, registry, sample_metadata):
        """Test getting a specific module."""
        module = MockModule('test-module-1')
        await registry.register_module(sample_metadata, module)
        
        # Get module
        retrieved = await registry.get_module('test-module-1')
        assert retrieved is not None
        assert retrieved.module_id == 'test-module-1'
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_module(self, registry):
        """Test getting non-existent module returns None."""
        module = await registry.get_module('non-existent')
        assert module is None
    
    @pytest.mark.asyncio
    async def test_find_modules_by_capability(self, registry):
        """Test finding modules by capability."""
        # Create modules with different capabilities
        storage_module = MockModule('storage-1')
        storage_metadata = ModuleMetadata(
            module_id='storage-1',
            name='Storage Module',
            version=Version(1, 0, 0),
            capabilities=[
                ModuleCapability(
                    capability_type=CapabilityType.STORAGE,
                    description='Storage capability'
                )
            ]
        )
        
        processing_module = MockModule('processing-1')
        processing_metadata = ModuleMetadata(
            module_id='processing-1',
            name='Processing Module',
            version=Version(1, 0, 0),
            capabilities=[
                ModuleCapability(
                    capability_type=CapabilityType.PROCESSING,
                    description='Processing capability'
                )
            ]
        )
        
        # Register modules
        await registry.register_module(storage_metadata, storage_module)
        await registry.register_module(processing_metadata, processing_module)
        
        # Find by capability
        storage_modules = await registry.find_by_capability(CapabilityType.STORAGE)
        assert len(storage_modules) == 1
        assert storage_modules[0].module_id == 'storage-1'
        
        processing_modules = await registry.find_by_capability(CapabilityType.PROCESSING)
        assert len(processing_modules) == 1
        assert processing_modules[0].module_id == 'processing-1'
    
    @pytest.mark.asyncio
    async def test_activate_module(self, registry, sample_metadata):
        """Test activating a module."""
        module = MockModule('test-module-1')
        await registry.register_module(sample_metadata, module)
        
        # Activate with config
        config = {'test_param': 'value'}
        success = await registry.activate_module('test-module-1', config)
        assert success is True
        
        # Check module status
        modules = await registry.list_modules()
        assert modules[0].status == ModuleStatus.ACTIVE
        
        # Check module is initialized
        assert module.initialized is True
        assert module.config == config
    
    @pytest.mark.asyncio
    async def test_deactivate_module(self, registry, sample_metadata):
        """Test deactivating a module."""
        module = MockModule('test-module-1')
        await registry.register_module(sample_metadata, module)
        await registry.activate_module('test-module-1', {})
        
        # Deactivate
        success = await registry.deactivate_module('test-module-1')
        assert success is True
        
        # Check status
        modules = await registry.list_modules()
        assert modules[0].status == ModuleStatus.INACTIVE
        assert module.initialized is False
    
    @pytest.mark.asyncio
    async def test_module_health_check(self, registry, sample_metadata):
        """Test module health checking."""
        module = MockModule('test-module-1')
        await registry.register_module(sample_metadata, module)
        
        # Check health before activation
        health = await registry.check_module_health('test-module-1')
        assert health['status'] == 'unhealthy'
        
        # Activate and check again
        await registry.activate_module('test-module-1', {})
        health = await registry.check_module_health('test-module-1')
        assert health['status'] == 'healthy'
    
    @pytest.mark.asyncio
    async def test_broadcast_event(self, registry):
        """Test broadcasting events to modules."""
        # Create modules that track events
        class EventTrackingModule(MockModule):
            def __init__(self, module_id):
                super().__init__(module_id)
                self.events_received = []
            
            async def handle_event(self, event: Dict[str, Any]) -> None:
                self.events_received.append(event)
        
        # Register multiple modules
        module1 = EventTrackingModule('module-1')
        module2 = EventTrackingModule('module-2')
        
        metadata1 = ModuleMetadata(
            module_id='module-1',
            name='Module 1',
            version=Version(1, 0, 0),
            capabilities=[
                ModuleCapability(CapabilityType.EVENT_HANDLING)
            ]
        )
        
        metadata2 = ModuleMetadata(
            module_id='module-2',
            name='Module 2',
            version=Version(1, 0, 0),
            capabilities=[
                ModuleCapability(CapabilityType.EVENT_HANDLING)
            ]
        )
        
        await registry.register_module(metadata1, module1)
        await registry.register_module(metadata2, module2)
        await registry.activate_module('module-1', {})
        await registry.activate_module('module-2', {})
        
        # Broadcast event
        event = {'type': 'test_event', 'data': 'test_data'}
        await registry.broadcast_event(event)
        
        # Both modules should receive the event
        assert len(module1.events_received) == 1
        assert module1.events_received[0] == event
        assert len(module2.events_received) == 1
        assert module2.events_received[0] == event
    
    @pytest.mark.asyncio
    async def test_version_compatibility(self, registry):
        """Test version compatibility checking."""
        module = MockModule('test-module')
        metadata = ModuleMetadata(
            module_id='test-module',
            name='Test Module',
            version=Version(2, 3, 1),
            capabilities=[]
        )
        
        await registry.register_module(metadata, module)
        
        # Test version comparisons
        module_info = await registry.get_module('test-module')
        assert module_info.version.major == 2
        assert module_info.version.minor == 3
        assert module_info.version.patch == 1
        
        # Version string
        assert str(module_info.version) == '2.3.1'
    
    @pytest.mark.asyncio
    async def test_capability_matching(self, registry):
        """Test capability matching logic."""
        # Create a module with specific capabilities
        module = MockModule('multi-cap')
        metadata = ModuleMetadata(
            module_id='multi-cap',
            name='Multi Capability Module',
            version=Version(1, 0, 0),
            capabilities=[
                ModuleCapability(
                    capability_type=CapabilityType.STORAGE,
                    parameters={'format': 'json'}
                ),
                ModuleCapability(
                    capability_type=CapabilityType.PROCESSING,
                    parameters={'types': ['text', 'binary']}
                ),
                ModuleCapability(
                    capability_type=CapabilityType.VECTOR_OPERATIONS,
                    parameters={'dimensions': 768}
                )
            ]
        )
        
        await registry.register_module(metadata, module)
        
        # Test finding by multiple capabilities
        modules = await registry.find_by_capability(CapabilityType.STORAGE)
        assert len(modules) == 1
        
        modules = await registry.find_by_capability(CapabilityType.VECTOR_OPERATIONS)
        assert len(modules) == 1
        
        # Module should appear in both searches
        assert modules[0].module_id == 'multi-cap'
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, registry):
        """Test concurrent module operations."""
        # Register multiple modules concurrently
        modules = []
        metadatas = []
        
        for i in range(10):
            module = MockModule(f'concurrent-{i}')
            metadata = ModuleMetadata(
                module_id=f'concurrent-{i}',
                name=f'Concurrent Module {i}',
                version=Version(1, 0, 0),
                capabilities=[]
            )
            modules.append(module)
            metadatas.append(metadata)
        
        # Register all concurrently
        tasks = [
            registry.register_module(metadata, module)
            for metadata, module in zip(metadatas, modules)
        ]
        results = await asyncio.gather(*tasks)
        
        assert all(results)
        
        # List all modules
        all_modules = await registry.list_modules()
        assert len(all_modules) == 10
    
    @pytest.mark.asyncio
    async def test_module_lifecycle(self, registry, sample_metadata):
        """Test complete module lifecycle."""
        module = MockModule('lifecycle-test')
        
        # 1. Register
        await registry.register_module(sample_metadata, module)
        modules = await registry.list_modules()
        assert modules[0].status == ModuleStatus.INACTIVE
        
        # 2. Activate
        await registry.activate_module('lifecycle-test', {'param': 'value'})
        modules = await registry.list_modules()
        assert modules[0].status == ModuleStatus.ACTIVE
        assert module.initialized is True
        
        # 3. Health check
        health = await registry.check_module_health('lifecycle-test')
        assert health['status'] == 'healthy'
        
        # 4. Deactivate
        await registry.deactivate_module('lifecycle-test')
        modules = await registry.list_modules()
        assert modules[0].status == ModuleStatus.INACTIVE
        assert module.initialized is False
        
        # 5. Unregister
        await registry.unregister_module('lifecycle-test')
        modules = await registry.list_modules()
        assert len(modules) == 0