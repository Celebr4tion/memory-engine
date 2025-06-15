#!/usr/bin/env python3
"""
Example demonstrating the Memory Engine Orchestrator Integration (v0.5.0)

This example showcases:
1. Enhanced MCP interface with streaming support
2. GraphQL-like query language
3. Event system for inter-module communication
4. Module registry with capability advertisement
5. Standardized data formats and interfaces
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, Any

# Import orchestrator components
from memory_core.orchestrator import (
    # Enhanced MCP
    EnhancedMCPServer,
    MCPStreaming,
    ProgressCallback,
    # Query Language
    GraphQLQueryProcessor,
    QueryBuilder,
    QueryValidator,
    QueryType,
    FilterOperator,
    # Event System
    EventSystem,
    EventBus,
    EventSubscriber,
    EventPublisher,
    EventType,
    EventPriority,
    KnowledgeChangeEvent,
    # Module Registry
    ModuleRegistry,
    ModuleInterface,
    ModuleMetadata,
    ModuleCapability,
    CapabilityType,
    Version,
    ModuleStatus,
    # Data Formats
    StandardizedKnowledge,
    StandardizedIdentifier,
    UnifiedError,
    EntityType,
    ErrorCode,
    OperationResult,
    create_knowledge_entity,
)

# Import core Memory Engine components
from memory_core.memory_engine import MemoryEngine


class ExampleModule(ModuleInterface):
    """Example module implementation."""

    def __init__(self, module_id: str):
        self.module_id = module_id
        self.initialized = False
        self.config = {}

    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the module."""
        self.config = config
        self.initialized = True
        print(f"Module {self.module_id} initialized with config: {config}")
        return True

    async def shutdown(self) -> bool:
        """Shutdown the module."""
        self.initialized = False
        print(f"Module {self.module_id} shutdown")
        return True

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            "status": "healthy" if self.initialized else "unhealthy",
            "module_id": self.module_id,
            "initialized": self.initialized,
        }

    def get_metadata(self) -> ModuleMetadata:
        """Get module metadata."""
        capability = ModuleCapability(
            capability_type=CapabilityType.CUSTOM,
            name="example_capability",
            description="Example capability for demonstration",
            version=Version(1, 0, 0),
        )

        return ModuleMetadata(
            module_id=self.module_id,
            name=f"Example Module {self.module_id}",
            description="Example module for orchestrator integration demo",
            version=Version(1, 0, 0),
            author="Memory Engine",
            capabilities=[capability],
        )

    def get_capabilities(self) -> list:
        """Get module capabilities."""
        return self.get_metadata().capabilities


async def demonstrate_streaming_mcp():
    """Demonstrate enhanced MCP with streaming support."""
    print("\n=== Enhanced MCP Streaming Demo ===")

    # Mock knowledge engine for demo
    class MockKnowledgeEngine:
        async def query(self, query, limit=100):
            # Simulate query results
            return [
                {"id": f"node_{i}", "content": f"Content {i}", "score": 0.9 - i * 0.1}
                for i in range(min(limit, 25))
            ]

        async def save_node(self, item):
            return f"saved_{item.get('id', 'unknown')}"

    knowledge_engine = MockKnowledgeEngine()
    mcp_server = EnhancedMCPServer(knowledge_engine)

    def progress_callback(progress):
        print(f"Progress: {progress.message} ({progress.current}/{progress.total})")

    # Start streaming query
    response = await mcp_server._handle_stream_query(
        {"query": "find important concepts", "limit": 20, "batch_size": 5}
    )

    print(f"Streaming query started: {response}")

    if response["success"]:
        operation_id = response["operation_id"]

        # Stream results
        print("Streaming results:")
        async for result in mcp_server.stream_results(operation_id):
            print(f"  Batch result: {result.get('data', {}).get('batch_index', 'final')}")
            if result.get("state") == "completed":
                break


async def demonstrate_graphql_queries():
    """Demonstrate GraphQL-like query language."""
    print("\n=== GraphQL-like Query Language Demo ===")

    # Mock knowledge engine with sample data
    class MockKnowledgeEngine:
        def __init__(self):
            self.nodes = [
                {
                    "id": "1",
                    "content": "Machine Learning Concepts",
                    "rating_importance": 0.9,
                    "tags": ["AI", "ML"],
                },
                {
                    "id": "2",
                    "content": "Neural Networks",
                    "rating_importance": 0.8,
                    "tags": ["AI", "NN"],
                },
                {
                    "id": "3",
                    "content": "Data Science",
                    "rating_importance": 0.7,
                    "tags": ["Data", "Science"],
                },
            ]
            self.relationships = [
                {"id": "r1", "source_id": "1", "target_id": "2", "relationship_type": "contains"},
                {"id": "r2", "source_id": "2", "target_id": "3", "relationship_type": "related_to"},
            ]

        async def get_all_nodes(self):
            return self.nodes

        async def get_all_relationships(self):
            return self.relationships

        async def semantic_search(self, query, limit=10):
            # Simple text matching for demo
            results = [node for node in self.nodes if query.lower() in node["content"].lower()]
            return results[:limit]

    knowledge_engine = MockKnowledgeEngine()
    processor = GraphQLQueryProcessor(knowledge_engine)

    # Build complex query using QueryBuilder
    query = (
        QueryBuilder()
        .query_type(QueryType.NODES)
        .where_greater_than("rating_importance", 0.75)
        .where_contains("tags", "AI")
        .select("id", "content", "rating_importance")
        .order_by("rating_importance", ascending=False)
        .limit(10)
        .build()
    )

    print(f"Query specification: {json.dumps(query.to_dict(), indent=2)}")

    # Execute query
    result = await processor.execute_query(query)
    print(f"Query result: {json.dumps(result, indent=2)}")

    # Build semantic search query
    search_query = (
        QueryBuilder()
        .query_type(QueryType.SEARCH)
        .where_contains("content", "Neural")
        .select("id", "content")
        .limit(5)
        .build()
    )

    search_result = await processor.execute_query(search_query)
    print(f"Search result: {json.dumps(search_result, indent=2)}")


async def demonstrate_event_system():
    """Demonstrate event system for inter-module communication."""
    print("\n=== Event System Demo ===")

    # Initialize event system
    event_system = EventSystem(
        {"event_bus": {"enable_persistence": False, "enable_batching": True, "batch_size": 5}}
    )

    await event_system.initialize()

    # Create publisher and subscriber
    publisher = event_system.create_publisher("demo_module")
    subscriber = event_system.create_subscriber("demo_listener")

    # Event handler
    async def handle_knowledge_event(event):
        print(f"Received knowledge event: {event.data['action']} for node {event.data['node_id']}")

    # Subscribe to knowledge change events
    subscriber.subscribe(EventType.KNOWLEDGE_NODE_CREATED, handle_knowledge_event)
    subscriber.subscribe(EventType.KNOWLEDGE_NODE_UPDATED, handle_knowledge_event)

    # Publish some events
    await publisher.publish_knowledge_change(
        "created", "node_123", {"content": "New knowledge about AI", "importance": 0.8}
    )

    await publisher.publish_knowledge_change(
        "updated", "node_123", {"content": "Updated knowledge about AI", "importance": 0.9}
    )

    # Wait for events to be processed
    await asyncio.sleep(0.5)

    # Get system metrics
    metrics = event_system.get_system_metrics()
    print(f"Event system metrics: {json.dumps(metrics, indent=2)}")

    await event_system.shutdown()


async def demonstrate_module_registry():
    """Demonstrate module registry with capability advertisement."""
    print("\n=== Module Registry Demo ===")

    # Initialize registry
    registry = ModuleRegistry(
        {"health_check_interval": 5.0, "storage_path": "./data/registry_demo"}
    )

    await registry.initialize()

    # Create and register example modules
    module1 = ExampleModule("ai_module")
    module2 = ExampleModule("storage_module")

    # Register modules
    success1 = registry.register_module(module1.get_metadata(), module1)
    success2 = registry.register_module(module2.get_metadata(), module2)

    print(f"Module 1 registration: {success1}")
    print(f"Module 2 registration: {success2}")

    # Initialize modules
    init_results = await registry.initialize_modules()
    print(f"Module initialization results: {init_results}")

    # Find modules by capability
    custom_modules = registry.find_modules_by_capability(CapabilityType.CUSTOM)
    print(f"Found {len(custom_modules)} modules with custom capabilities")

    # Get registry summary
    summary = registry.get_registry_summary()
    print(f"Registry summary: {json.dumps(summary, indent=2)}")

    # Perform health checks
    health_results = await registry.perform_health_checks()
    print(f"Health check results: {json.dumps(health_results, indent=2)}")

    await registry.shutdown()


async def demonstrate_standardized_data_formats():
    """Demonstrate standardized data formats."""
    print("\n=== Standardized Data Formats Demo ===")

    # Create standardized knowledge entity
    knowledge = create_knowledge_entity(
        module_id="demo_module",
        content="This is standardized knowledge content",
        entity_id="demo_knowledge_1",
    )

    # Add metadata and embeddings
    knowledge.metadata = {"category": "demo", "priority": "high"}
    knowledge.tags = ["demo", "standardized", "knowledge"]
    knowledge.set_embedding("openai", [0.1, 0.2, 0.3, 0.4, 0.5])

    print(f"Created knowledge entity: {knowledge.identifier}")
    print(f"Knowledge data: {json.dumps(knowledge.to_dict(), indent=2)}")

    # Create operation result
    success_result = OperationResult.success_result(
        data={"processed": True, "entity_count": 1},
        metadata={"operation": "create_knowledge", "timestamp": knowledge.created_at},
    )

    print(f"Success result: {json.dumps(success_result.to_dict(), indent=2)}")

    # Create error result
    error = UnifiedError(
        code=ErrorCode.VALIDATION_ERROR,
        message="Invalid content format",
        details={"field": "content", "expected": "string"},
    )

    error_result = OperationResult.error_result(error)
    print(f"Error result: {json.dumps(error_result.to_dict(), indent=2)}")


async def comprehensive_orchestrator_demo():
    """Run comprehensive demonstration of all orchestrator features."""
    print("🚀 Memory Engine Orchestrator Integration Demo (v0.5.0)")
    print("=" * 60)

    try:
        await demonstrate_streaming_mcp()
        await demonstrate_graphql_queries()
        await demonstrate_event_system()
        await demonstrate_module_registry()
        await demonstrate_standardized_data_formats()

        print("\n✅ All orchestrator integration features demonstrated successfully!")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Ensure data directory exists
    Path("./data").mkdir(exist_ok=True)

    # Run the comprehensive demo
    asyncio.run(comprehensive_orchestrator_demo())
