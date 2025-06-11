"""
Storage Backends Example

This example demonstrates how to use different storage backends
with the Memory Engine system, showing the modularity and
flexibility of the new storage architecture.
"""

import asyncio
import time
from pathlib import Path
from memory_core.storage import create_storage, list_available_backends, is_backend_available
from memory_core.model.knowledge_node import KnowledgeNode
from memory_core.model.relationship import Relationship


async def demonstrate_backend(backend_type: str, config_override=None):
    """Demonstrate operations with a specific storage backend."""
    print(f"\n{'='*60}")
    print(f"🗄️  Testing {backend_type.upper()} Storage Backend")
    print(f"{'='*60}")
    
    try:
        # Create storage backend
        storage = create_storage(backend_type=backend_type, config_override=config_override)
        print(f"✅ Created {backend_type} storage instance")
        
        # Test connection
        await storage.connect()
        is_connected = await storage.test_connection()
        print(f"🔗 Connection test: {'✅ Success' if is_connected else '❌ Failed'}")
        
        if not is_connected:
            print(f"⚠️  Skipping {backend_type} tests due to connection failure")
            return
        
        # Create sample knowledge nodes
        print("\n📝 Creating knowledge nodes...")
        
        node1 = KnowledgeNode(
            content=f"Python is a versatile programming language - {backend_type} test",
            source=f"example_{backend_type}",
            rating_richness=0.8,
            rating_truthfulness=0.9,
            rating_stability=0.7
        )
        
        node2 = KnowledgeNode(
            content=f"Machine learning enables computers to learn - {backend_type} test",
            source=f"example_{backend_type}",
            rating_richness=0.9,
            rating_truthfulness=0.8,
            rating_stability=0.8
        )
        
        # Save nodes
        node1_id = await storage.save_knowledge_node(node1)
        node2_id = await storage.save_knowledge_node(node2)
        print(f"✅ Created nodes: {node1_id[:8]}..., {node2_id[:8]}...")
        
        # Create relationship
        print("\n🔗 Creating relationship...")
        relationship = Relationship(
            from_id=node1_id,
            to_id=node2_id,
            relation_type="enables",
            confidence_score=0.85,
            version=1
        )
        
        rel_id = await storage.save_relationship(relationship)
        print(f"✅ Created relationship: {rel_id[:8]}...")
        
        # Retrieve and verify
        print("\n🔍 Retrieving data...")
        retrieved_node1 = await storage.get_knowledge_node(node1_id)
        retrieved_rel = await storage.get_relationship(rel_id)
        
        print(f"📄 Node content: {retrieved_node1.content[:50]}...")
        print(f"🔗 Relationship type: {retrieved_rel.relation_type}")
        
        # Test search
        print("\n🔎 Testing content search...")
        search_results = await storage.find_nodes_by_content("Python")
        print(f"📊 Found {len(search_results)} nodes matching 'Python'")
        
        # Test graph traversal
        print("\n🗺️  Testing graph traversal...")
        neighbors = await storage.find_neighbors(node1_id)
        print(f"👥 Found {len(neighbors)} neighbors for node1")
        
        # Performance statistics
        stats = storage.get_traversal_statistics()
        print(f"\n📈 Performance stats: {stats}")
        
        # Cleanup
        print("\n🧹 Cleaning up...")
        await storage.clear_all_data()
        print("✅ Data cleared")
        
        await storage.close()
        print("✅ Connection closed")
        
    except Exception as e:
        print(f"❌ Error with {backend_type}: {e}")
        import traceback
        traceback.print_exc()


async def compare_backends():
    """Compare performance and features across backends."""
    print(f"\n{'='*60}")
    print("⚡ Performance Comparison")
    print(f"{'='*60}")
    
    backends_to_test = []
    
    # JSON File backend (always available)
    backends_to_test.append(('json_file', {'directory': './temp_test_json'}))
    
    # SQLite backend (if available)
    if is_backend_available('sqlite'):
        backends_to_test.append(('sqlite', {'database_path': './temp_test.db'}))
    
    # JanusGraph backend (if available)
    if is_backend_available('janusgraph'):
        backends_to_test.append(('janusgraph', None))
    
    performance_results = {}
    
    for backend_type, config in backends_to_test:
        print(f"\n🔍 Testing {backend_type} performance...")
        
        try:
            storage = create_storage(backend_type=backend_type, config_override=config)
            await storage.connect()
            
            # Time node creation
            start_time = time.time()
            
            nodes = []
            for i in range(10):
                node = KnowledgeNode(
                    content=f"Performance test node {i}",
                    source="performance_test",
                    rating_richness=0.8,
                    rating_truthfulness=0.9,
                    rating_stability=0.7
                )
                node_id = await storage.save_knowledge_node(node)
                nodes.append(node_id)
            
            creation_time = time.time() - start_time
            
            # Time search operation
            start_time = time.time()
            results = await storage.find_nodes_by_content("Performance")
            search_time = time.time() - start_time
            
            performance_results[backend_type] = {
                'creation_time': creation_time,
                'search_time': search_time,
                'results_count': len(results)
            }
            
            await storage.clear_all_data()
            await storage.close()
            
        except Exception as e:
            print(f"❌ Error testing {backend_type}: {e}")
            performance_results[backend_type] = {'error': str(e)}
    
    # Display results
    print(f"\n📊 Performance Results:")
    print("-" * 60)
    for backend, results in performance_results.items():
        if 'error' in results:
            print(f"{backend:12} | Error: {results['error']}")
        else:
            print(f"{backend:12} | Creation: {results['creation_time']:.3f}s | "
                  f"Search: {results['search_time']:.3f}s | "
                  f"Results: {results['results_count']}")


async def migration_example():
    """Demonstrate migrating data between storage backends."""
    print(f"\n{'='*60}")
    print("🔄 Storage Migration Example")
    print(f"{'='*60}")
    
    # Create data in JSON backend
    json_storage = create_storage(
        backend_type='json_file', 
        config_override={'directory': './migration_source'}
    )
    
    await json_storage.connect()
    print("📝 Creating sample data in JSON storage...")
    
    # Create sample nodes
    nodes = []
    for i in range(3):
        node = KnowledgeNode(
            content=f"Migration test content {i+1}: Knowledge management systems",
            source="migration_example",
            rating_richness=0.8 + i * 0.05,
            rating_truthfulness=0.9,
            rating_stability=0.7 + i * 0.1
        )
        node_id = await json_storage.save_knowledge_node(node)
        nodes.append(node_id)
    
    print(f"✅ Created {len(nodes)} nodes in JSON storage")
    
    # Create relationships
    for i in range(len(nodes) - 1):
        rel = Relationship(
            from_id=nodes[i],
            to_id=nodes[i + 1],
            relation_type="relates_to",
            confidence_score=0.8,
            version=1
        )
        await json_storage.save_relationship(rel)
    
    print("✅ Created relationships between nodes")
    
    # Search in source
    search_results = await json_storage.find_nodes_by_content("Knowledge management")
    print(f"🔍 Found {len(search_results)} nodes in source storage")
    
    # If SQLite is available, demonstrate migration
    if is_backend_available('sqlite'):
        print("\n➡️  Migrating to SQLite storage...")
        
        sqlite_storage = create_storage(
            backend_type='sqlite',
            config_override={'database_path': './migration_target.db'}
        )
        
        await sqlite_storage.connect()
        
        # Migrate nodes (simplified example)
        migrated_nodes = {}
        for node_id in nodes:
            source_node = await json_storage.get_knowledge_node(node_id)
            # Create new node in target (will get new ID)
            new_node = KnowledgeNode(
                content=source_node.content,
                source=source_node.source,
                creation_timestamp=source_node.creation_timestamp,
                rating_richness=source_node.rating_richness,
                rating_truthfulness=source_node.rating_truthfulness,
                rating_stability=source_node.rating_stability
            )
            new_id = await sqlite_storage.save_knowledge_node(new_node)
            migrated_nodes[node_id] = new_id
        
        print(f"✅ Migrated {len(migrated_nodes)} nodes to SQLite")
        
        # Verify migration
        search_results_target = await sqlite_storage.find_nodes_by_content("Knowledge management")
        print(f"🔍 Found {len(search_results_target)} nodes in target storage")
        
        await sqlite_storage.close()
        print("✅ Migration completed successfully")
    
    await json_storage.close()
    
    # Cleanup
    import shutil
    shutil.rmtree('./migration_source', ignore_errors=True)
    Path('./migration_target.db').unlink(missing_ok=True)


async def main():
    """Main function demonstrating storage backend capabilities."""
    print("🌟 Memory Engine - Storage Backends Example")
    print("=" * 60)
    
    # List available backends
    backends = list_available_backends()
    print(f"📋 Available storage backends: {', '.join(backends)}")
    
    for backend in backends:
        available = is_backend_available(backend)
        print(f"   {backend}: {'✅ Available' if available else '❌ Not available'}")
    
    # Test each available backend
    
    # JSON File backend (always available)
    await demonstrate_backend(
        'json_file',
        config_override={'directory': './demo_json_storage'}
    )
    
    # SQLite backend (if aiosqlite is installed)
    if is_backend_available('sqlite'):
        await demonstrate_backend(
            'sqlite',
            config_override={'database_path': './demo_storage.db'}
        )
    else:
        print("\n⚠️  SQLite backend not available (install aiosqlite: pip install aiosqlite)")
    
    # JanusGraph backend (if server is running)
    if is_backend_available('janusgraph'):
        await demonstrate_backend('janusgraph')
    else:
        print("\n⚠️  JanusGraph backend not available (server not running or dependencies missing)")
    
    # Performance comparison
    await compare_backends()
    
    # Migration example
    await migration_example()
    
    # Cleanup
    print(f"\n🧹 Cleaning up temporary files...")
    import shutil
    shutil.rmtree('./demo_json_storage', ignore_errors=True)
    Path('./demo_storage.db').unlink(missing_ok=True)
    
    print(f"\n✅ Storage backends example completed!")
    print(f"\n📚 Next steps:")
    print("   - Read docs/developer/storage_backends.md for detailed guide")
    print("   - Configure your preferred backend in config.yaml") 
    print("   - Try different backends for your use case")


if __name__ == "__main__":
    asyncio.run(main())