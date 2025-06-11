"""
Tests for the new modular storage backends.

This module tests the JsonFileStorage, SqliteStorage, and the StorageFactory
to ensure they implement the GraphStorageInterface correctly.
"""
import unittest
import tempfile
import shutil
import asyncio
import time
from pathlib import Path
from unittest.mock import patch

from memory_core.storage.interfaces.graph_storage_interface import GraphStorageInterface
from memory_core.storage.backends.json_file import JsonFileStorage
from memory_core.storage.factory import create_storage, list_available_backends, is_backend_available
from memory_core.model.knowledge_node import KnowledgeNode
from memory_core.model.relationship import Relationship

# Try to import SQLite backend, skip tests if not available
try:
    from memory_core.storage.backends.sqlite import SqliteStorage
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False
    SqliteStorage = None


class TestStorageBackends(unittest.TestCase):
    """Test all storage backends for interface compliance."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.test_db_path = Path(self.test_dir) / "test.db"
        
        # Test data
        self.test_node_data = {
            'content': 'Test content for storage backends',
            'source': 'test_source',
            'creation_timestamp': time.time(),
            'rating_richness': 0.8,
            'rating_truthfulness': 0.9,
            'rating_stability': 0.7
        }
        
        self.test_edge_data = {
            'timestamp': time.time(),
            'confidence_score': 0.85,
            'version': 1
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_json_file_storage_interface_compliance(self):
        """Test that JsonFileStorage implements GraphStorageInterface correctly."""
        storage = JsonFileStorage(directory=str(Path(self.test_dir) / "json_test"))
        self.assertIsInstance(storage, GraphStorageInterface)
        
        # Test basic operations
        self._test_storage_basic_operations(storage)
    
    @unittest.skipUnless(SQLITE_AVAILABLE, "SQLite backend not available (missing aiosqlite)")
    def test_sqlite_storage_interface_compliance(self):
        """Test that SqliteStorage implements GraphStorageInterface correctly."""
        storage = SqliteStorage(database_path=str(self.test_db_path))
        self.assertIsInstance(storage, GraphStorageInterface)
        
        # Test basic operations
        self._test_storage_basic_operations(storage)
    
    def test_storage_factory(self):
        """Test the storage factory creates backends correctly."""
        # Test available backends
        backends = list_available_backends()
        self.assertIn('json_file', backends)
        if SQLITE_AVAILABLE:
            self.assertIn('sqlite', backends)
        
        # Test backend availability check
        self.assertTrue(is_backend_available('json_file'))
        if SQLITE_AVAILABLE:
            self.assertTrue(is_backend_available('sqlite'))
        self.assertFalse(is_backend_available('nonexistent_backend'))
        
        # Test factory creation with overrides
        json_storage = create_storage(
            backend_type='json_file', 
            config_override={'directory': str(Path(self.test_dir) / "factory_test")}
        )
        self.assertIsInstance(json_storage, JsonFileStorage)
        
        if SQLITE_AVAILABLE:
            sqlite_storage = create_storage(
                backend_type='sqlite',
                config_override={'database_path': str(self.test_db_path)}
            )
            self.assertIsInstance(sqlite_storage, SqliteStorage)
    
    def test_json_file_storage_specific_features(self):
        """Test JSON file storage specific features."""
        storage = JsonFileStorage(
            directory=str(Path(self.test_dir) / "json_specific"),
            pretty_print=True
        )
        
        # Test connection
        self.assertTrue(storage.test_connection_sync())
        storage.connect_sync()
        
        # Test node operations
        node_id = storage.create_node_sync(self.test_node_data)
        self.assertIsInstance(node_id, str)
        
        # Verify files are created
        storage_dir = Path(self.test_dir) / "json_specific"
        self.assertTrue((storage_dir / "nodes.json").exists())
        self.assertTrue((storage_dir / "indexes.json").exists())
        
        storage.close_sync()
    
    @unittest.skipUnless(SQLITE_AVAILABLE, "SQLite backend not available (missing aiosqlite)")
    def test_sqlite_storage_specific_features(self):
        """Test SQLite storage specific features."""
        storage = SqliteStorage(database_path=str(self.test_db_path))
        
        # Test connection
        self.assertTrue(storage.test_connection_sync())
        storage.connect_sync()
        
        # Test node operations
        node_id = storage.create_node_sync(self.test_node_data)
        self.assertIsInstance(node_id, str)
        
        # Verify database file is created
        self.assertTrue(self.test_db_path.exists())
        
        storage.close_sync()
    
    def test_knowledge_node_operations_across_backends(self):
        """Test KnowledgeNode operations work consistently across backends."""
        backends = [
            JsonFileStorage(directory=str(Path(self.test_dir) / "json_kn"))
        ]
        if SQLITE_AVAILABLE:
            backends.append(SqliteStorage(database_path=str(Path(self.test_dir) / "sqlite_kn.db")))
        
        for storage in backends:
            with self.subTest(storage=storage.__class__.__name__):
                storage.connect_sync()
                
                # Create a knowledge node
                node = KnowledgeNode(
                    content="Test knowledge node",
                    source="test_source",
                    rating_richness=0.8,
                    rating_truthfulness=0.9,
                    rating_stability=0.7
                )
                
                # Save and retrieve
                node_id = storage.save_knowledge_node_sync(node)
                self.assertIsNotNone(node_id)
                self.assertEqual(node.node_id, node_id)
                
                retrieved_node = storage.get_knowledge_node_sync(node_id)
                self.assertIsNotNone(retrieved_node)
                self.assertEqual(retrieved_node.content, node.content)
                self.assertEqual(retrieved_node.source, node.source)
                
                storage.close_sync()
    
    def test_relationship_operations_across_backends(self):
        """Test Relationship operations work consistently across backends."""
        backends = [
            JsonFileStorage(directory=str(Path(self.test_dir) / "json_rel"))
        ]
        if SQLITE_AVAILABLE:
            backends.append(SqliteStorage(database_path=str(Path(self.test_dir) / "sqlite_rel.db")))
        
        for storage in backends:
            with self.subTest(storage=storage.__class__.__name__):
                storage.connect_sync()
                
                # Create nodes first
                node1_id = storage.create_node_sync(self.test_node_data)
                node2_data = dict(self.test_node_data)
                node2_data['content'] = 'Second test node'
                node2_id = storage.create_node_sync(node2_data)
                
                # Create a relationship
                relationship = Relationship(
                    from_id=node1_id,
                    to_id=node2_id,
                    relation_type="test_relation",
                    confidence_score=0.85,
                    version=1
                )
                
                # Save and retrieve
                edge_id = self._run_async_sync(storage.save_relationship(relationship))
                self.assertIsNotNone(edge_id)
                self.assertEqual(relationship.edge_id, edge_id)
                
                retrieved_rel = self._run_async_sync(storage.get_relationship(edge_id))
                self.assertIsNotNone(retrieved_rel)
                self.assertEqual(retrieved_rel.from_id, relationship.from_id)
                self.assertEqual(retrieved_rel.to_id, relationship.to_id)
                self.assertEqual(retrieved_rel.relation_type, relationship.relation_type)
                
                storage.close_sync()
    
    def _test_storage_basic_operations(self, storage):
        """Test basic storage operations for interface compliance."""
        # Test connection
        self.assertTrue(storage.test_connection_sync())
        storage.connect_sync()
        
        # Test node creation
        node_id = storage.create_node_sync(self.test_node_data)
        self.assertIsInstance(node_id, str)
        
        # Test node retrieval
        node = storage.get_node_sync(node_id)
        self.assertIsNotNone(node)
        self.assertEqual(node['content'], self.test_node_data['content'])
        
        # Test node update
        update_data = {'content': 'Updated content'}
        success = self._run_async_sync(storage.update_node(node_id, update_data))
        self.assertTrue(success)
        
        # Verify update
        updated_node = storage.get_node_sync(node_id)
        self.assertEqual(updated_node['content'], 'Updated content')
        
        # Test edge creation (need a second node)
        node2_data = dict(self.test_node_data)
        node2_data['content'] = 'Second node'
        node2_id = storage.create_node_sync(node2_data)
        
        edge_id = self._run_async_sync(storage.create_edge(
            node_id, node2_id, 'test_relation', self.test_edge_data
        ))
        self.assertIsInstance(edge_id, str)
        
        # Test edge retrieval
        edge = self._run_async_sync(storage.get_edge(edge_id))
        self.assertIsNotNone(edge)
        self.assertEqual(edge['from_id'], node_id)
        self.assertEqual(edge['to_id'], node2_id)
        
        # Test content search
        results = self._run_async_sync(storage.find_nodes_by_content('Updated'))
        self.assertGreater(len(results), 0)
        
        # Test cleanup
        success = self._run_async_sync(storage.clear_all_data())
        self.assertTrue(success)
        
        storage.close_sync()
    
    def _run_async_sync(self, coroutine):
        """Helper to run async operations in sync context."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coroutine)
        finally:
            loop.close()


class TestStorageConfigurationIntegration(unittest.TestCase):
    """Test storage configuration integration."""
    
    def test_factory_with_configuration(self):
        """Test factory uses configuration correctly."""
        # Test with mock configuration
        with patch('memory_core.storage.factory.get_config') as mock_config:
            # Mock config structure
            mock_config.return_value.config.storage.graph.backend = 'json_file'
            mock_config.return_value.config.storage.graph.json_file.directory = './test_data'
            mock_config.return_value.config.storage.graph.json_file.pretty_print = False
            
            storage = create_storage()
            self.assertIsInstance(storage, JsonFileStorage)
    
    def test_invalid_backend_type(self):
        """Test factory raises error for invalid backend type."""
        with self.assertRaises(ValueError):
            create_storage(backend_type='invalid_backend')


if __name__ == '__main__':
    unittest.main()