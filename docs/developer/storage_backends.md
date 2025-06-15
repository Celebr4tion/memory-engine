# Storage Backends

This document describes the modular storage backend system introduced in version 0.2.0, which allows the Memory Engine to work with different graph storage implementations. The system has been enhanced in subsequent versions with improved configuration and compatibility.

## Overview

The Memory Engine now supports multiple graph storage backends through a unified interface system. This provides flexibility to choose the right storage solution based on your deployment requirements, from lightweight development setups to production-grade distributed systems.

## Available Backends

### 1. JanusGraph Backend (Production)

**Use Case:** Production deployments requiring high performance, scalability, and advanced graph features.

**Configuration:**
```yaml
storage:
  graph:
    backend: "janusgraph"
    janusgraph:
      host: "localhost"
      port: 8182
      use_ssl: false
      connection_timeout: 30
      max_retry_attempts: 3
      retry_delay: 1.0
```

**Features:**
- Distributed graph database
- ACID transactions
- High availability
- Advanced graph traversal capabilities
- Production-ready performance

**Dependencies:**
- JanusGraph server
- `gremlin_python` library

### 2. JSON File Backend (Development)

**Use Case:** Development, testing, small datasets, and scenarios where simplicity is preferred over performance.

**Configuration:**
```yaml
storage:
  graph:
    backend: "json_file"
    json_file:
      directory: "./data/graph"
      pretty_print: true
```

**Features:**
- File-based storage using JSON
- Human-readable data format
- No external dependencies
- Simple backup and data inspection
- Built-in content indexing

**Files Created:**
- `nodes.json` - All graph nodes
- `edges.json` - All graph edges  
- `indexes.json` - Search and traversal indexes

### 3. SQLite Backend (Single-User)

**Use Case:** Single-user deployments, embedded applications, and scenarios requiring a balance between simplicity and performance.

**Configuration:**
```yaml
storage:
  graph:
    backend: "sqlite"
    sqlite:
      database_path: "./data/knowledge.db"
```

**Features:**
- Relational database with graph operations
- ACID transactions
- Single-file database
- SQL queries for advanced analytics
- Good performance for medium datasets

**Dependencies:**
- `aiosqlite` library

## Storage Interface

All storage backends implement the `GraphStorageInterface`, ensuring consistent behavior across different implementations.

### Core Operations

```python
from memory_core.storage import create_storage, GraphStorageInterface

# Create storage instance
storage: GraphStorageInterface = create_storage(backend_type="json_file")

# Connection management
await storage.connect()
await storage.test_connection()
await storage.close()

# Node operations
node_id = await storage.create_node(node_data)
node = await storage.get_node(node_id)
success = await storage.update_node(node_id, properties)
success = await storage.delete_node(node_id)

# Edge operations
edge_id = await storage.create_edge(from_id, to_id, relation_type, properties)
edge = await storage.get_edge(edge_id)
success = await storage.update_edge(edge_id, properties)
success = await storage.delete_edge(edge_id)

# Graph traversal
neighbors = await storage.find_neighbors(node_id)
path = await storage.find_shortest_path(from_id, to_id)
relationships = await storage.get_relationships_for_node(node_id)

# Search operations
results = await storage.find_nodes_by_content(query)

# Domain model operations
node_id = await storage.save_knowledge_node(knowledge_node)
knowledge_node = await storage.get_knowledge_node(node_id)
relationship_id = await storage.save_relationship(relationship)
```

## Factory Pattern

Use the storage factory to create backend instances:

```python
from memory_core.storage import create_storage, list_available_backends

# List available backends
backends = list_available_backends()
print(backends)  # ['janusgraph', 'json_file', 'sqlite']

# Create with configuration override
storage = create_storage(
    backend_type='json_file',
    config_override={'directory': '/custom/path'}
)

# Create using configuration file settings
storage = create_storage()  # Uses config.yaml settings
```

## Migration Between Backends

To migrate data between storage backends:

```python
import asyncio
from memory_core.storage import create_storage

async def migrate_data(source_type: str, target_type: str):
    # Create source and target storage
    source = create_storage(backend_type=source_type)
    target = create_storage(backend_type=target_type)
    
    await source.connect()
    await target.connect()
    
    # Get all nodes from source
    # Note: This is a simplified example
    # You would need to implement proper pagination for large datasets
    
    await source.close()
    await target.close()

# Example: Migrate from JSON to SQLite
asyncio.run(migrate_data('json_file', 'sqlite'))
```

## Performance Considerations

### JanusGraph
- **Best for:** Large datasets (>1M nodes), distributed deployments, complex queries
- **Throughput:** Very high
- **Latency:** Low for simple operations, optimized for complex traversals
- **Scalability:** Horizontal scaling supported

### SQLite
- **Best for:** Medium datasets (<1M nodes), single-user applications
- **Throughput:** High for single-threaded operations
- **Latency:** Very low for simple operations
- **Scalability:** Vertical scaling only

### JSON File
- **Best for:** Small datasets (<10K nodes), development, testing
- **Throughput:** Low to medium
- **Latency:** Medium (full file I/O on writes)
- **Scalability:** Not suitable for large datasets

## Development Guidelines

### Adding a New Backend

1. **Create Backend Module:**
   ```
   memory_core/storage/backends/mybackend/
   ├── __init__.py
   └── mybackend_storage.py
   ```

2. **Implement Interface:**
   ```python
   from memory_core.storage.interfaces import GraphStorageInterface
   
   class MyBackendStorage(GraphStorageInterface):
       # Implement all abstract methods
       async def connect(self) -> None:
           # Implementation
           pass
       
       # ... other methods
   ```

3. **Register in Factory:**
   ```python
   # In storage/factory.py
   def _register_backends(self):
       try:
           from memory_core.storage.backends.mybackend import MyBackendStorage
           self._backends['mybackend'] = MyBackendStorage
       except ImportError:
           self.logger.warning("MyBackend not available")
   ```

4. **Add Configuration:**
   ```python
   # In config/config_manager.py
   @dataclass
   class MyBackendConfig:
       setting1: str = "default_value"
       setting2: int = 42
   
   @dataclass
   class GraphStorageConfig:
       mybackend: MyBackendConfig = field(default_factory=MyBackendConfig)
   ```

5. **Write Tests:**
   ```python
   # Test interface compliance
   def test_mybackend_interface_compliance(self):
       storage = MyBackendStorage()
       self.assertIsInstance(storage, GraphStorageInterface)
       self._test_storage_basic_operations(storage)
   ```

### Testing

Run the storage backend tests:

```bash
# Test all available backends
python -m pytest tests/test_storage_backends.py -v

# Test specific backend
python -m pytest tests/test_storage_backends.py::TestStorageBackends::test_json_file_storage_interface_compliance -v
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies:**
   ```
   ModuleNotFoundError: No module named 'aiosqlite'
   ```
   **Solution:** Install optional dependencies: `pip install aiosqlite`

2. **Configuration Errors:**
   ```
   ValueError: Unsupported backend type 'invalid_backend'
   ```
   **Solution:** Check available backends with `list_available_backends()`

3. **Permission Errors:**
   ```
   PermissionError: [Errno 13] Permission denied: './data/graph'
   ```
   **Solution:** Ensure write permissions to storage directory

4. **Connection Failures:**
   ```
   ConnectionError: Not connected to storage backend
   ```
   **Solution:** Call `await storage.connect()` before operations

### Debug Mode

Enable debug logging for storage operations:

```python
import logging
logging.getLogger('memory_core.storage').setLevel(logging.DEBUG)
```

### Data Recovery

For JSON File Backend:
- Files are human-readable JSON
- Manual editing possible for corruption recovery
- Backup by copying directory

For SQLite Backend:
- Use SQLite CLI tools for inspection: `sqlite3 knowledge.db`
- Backup with `.backup` command
- Recovery tools available for corruption

For JanusGraph Backend:
- Follow JanusGraph backup procedures
- Use Gremlin console for manual operations
- Check JanusGraph logs for detailed errors

## Best Practices

1. **Development:** Use JSON File backend for rapid prototyping
2. **Testing:** Use SQLite backend for integration tests
3. **Production:** Use JanusGraph backend for scalable deployments
4. **Configuration:** Use environment-specific config files
5. **Migration:** Test migrations with small datasets first
6. **Monitoring:** Implement health checks using `test_connection()`
7. **Backup:** Regular backups appropriate for your backend choice

## Future Enhancements

Planned improvements include:

- **Neo4j Backend:** Native Cypher query support
- **Amazon Neptune:** Cloud-native graph database
- **Redis Graph:** In-memory graph operations
- **Async Migration Tools:** Automated data migration utilities
- **Performance Benchmarks:** Comparative performance testing
- **Connection Pooling:** Advanced connection management