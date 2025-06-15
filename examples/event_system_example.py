#!/usr/bin/env python3
"""
Event System Example for Memory Engine

This example demonstrates the comprehensive event system capabilities including:
- Event publishing and subscription
- Event filtering and batching
- Async and sync event handlers
- Event persistence and replay
- Dead letter queue handling
- System metrics and monitoring
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any, List

# Import Memory Engine event system
from memory_core.orchestrator.event_system import (
    EventSystem,
    EventType,
    EventPriority,
    Event,
    KnowledgeChangeEvent,
    RelationshipChangeEvent,
    QueryEvent,
    SystemEvent,
    CustomEvent,
)


class KnowledgeServiceSimulator:
    """Simulates a knowledge service that publishes events."""

    def __init__(self, event_system: EventSystem):
        self.event_system = event_system
        self.publisher = event_system.create_publisher("knowledge_service")

    async def create_knowledge_node(self, node_id: str, content: str, source: str):
        """Simulate creating a knowledge node."""
        print(f"🔍 Creating knowledge node: {node_id}")

        # Simulate some processing time
        await asyncio.sleep(0.1)

        # Publish knowledge change event
        await self.publisher.publish_knowledge_change(
            action="created",
            node_id=node_id,
            node_data={"content": content, "source": source, "timestamp": time.time()},
            priority=EventPriority.HIGH,
        )

        print(f"✅ Knowledge node {node_id} created and event published")

    async def update_knowledge_node(self, node_id: str, new_content: str):
        """Simulate updating a knowledge node."""
        print(f"📝 Updating knowledge node: {node_id}")

        await asyncio.sleep(0.1)

        await self.publisher.publish_knowledge_change(
            action="updated",
            node_id=node_id,
            node_data={"content": new_content, "timestamp": time.time()},
            priority=EventPriority.NORMAL,
        )

        print(f"✅ Knowledge node {node_id} updated and event published")

    async def create_relationship(self, rel_id: str, from_node: str, to_node: str, rel_type: str):
        """Simulate creating a relationship."""
        print(f"🔗 Creating relationship: {from_node} -> {to_node} ({rel_type})")

        await asyncio.sleep(0.1)

        await self.publisher.publish_relationship_change(
            action="created",
            relationship_id=rel_id,
            relationship_data={
                "from_node": from_node,
                "to_node": to_node,
                "relationship_type": rel_type,
                "timestamp": time.time(),
            },
        )

        print(f"✅ Relationship {rel_id} created and event published")


class QueryServiceSimulator:
    """Simulates a query service that publishes query events."""

    def __init__(self, event_system: EventSystem):
        self.event_system = event_system
        self.publisher = event_system.create_publisher("query_service")

    async def execute_query(self, query: str, query_type: str = "semantic_search"):
        """Simulate executing a query."""
        print(f"🔍 Executing query: {query}")

        # Simulate query processing
        await asyncio.sleep(0.2)

        # Simulate random success/failure
        import random

        if random.random() > 0.8:  # 20% failure rate
            await self.publisher.publish_query_event(
                query_type=query_type,
                query=query,
                error="Simulated query timeout",
                priority=EventPriority.HIGH,
            )
            print(f"❌ Query failed: {query}")
        else:
            results = [f"result_{i}" for i in range(random.randint(1, 5))]
            await self.publisher.publish_query_event(
                query_type=query_type, query=query, results=results
            )
            print(f"✅ Query executed successfully: {len(results)} results")


class EventMonitor:
    """Monitors and logs events for demonstration."""

    def __init__(self, event_system: EventSystem):
        self.event_system = event_system
        self.subscriber = event_system.create_subscriber("event_monitor")
        self.event_counts: Dict[str, int] = {}

        # Subscribe to all event types
        for event_type in EventType:
            self.subscriber.subscribe(event_type, self.handle_event)

    async def handle_event(self, event: Event):
        """Handle all events for monitoring."""
        event_type_name = event.event_type.value
        self.event_counts[event_type_name] = self.event_counts.get(event_type_name, 0) + 1

        print(
            f"📊 [MONITOR] {event_type_name}: {event.id[:8]}... "
            f"(Priority: {event.priority.value}, Total: {self.event_counts[event_type_name]})"
        )

    def get_statistics(self) -> Dict[str, int]:
        """Get monitoring statistics."""
        return self.event_counts.copy()


class KnowledgeIndexer:
    """Simulates a knowledge indexer that reacts to knowledge changes."""

    def __init__(self, event_system: EventSystem):
        self.event_system = event_system
        self.subscriber = event_system.create_subscriber("knowledge_indexer")
        self.indexed_nodes: List[str] = []

        # Subscribe to knowledge change events only
        self.subscriber.subscribe(EventType.KNOWLEDGE_NODE_CREATED, self.index_new_node)
        self.subscriber.subscribe(EventType.KNOWLEDGE_NODE_UPDATED, self.reindex_node)
        self.subscriber.subscribe(EventType.KNOWLEDGE_NODE_DELETED, self.remove_from_index)

    async def index_new_node(self, event: Event):
        """Handle new knowledge node creation."""
        node_id = event.data.get("node_id")
        content = event.data.get("node_data", {}).get("content", "")

        # Simulate indexing
        await asyncio.sleep(0.05)
        self.indexed_nodes.append(node_id)

        print(f"📚 [INDEXER] Indexed new node: {node_id} (Content: {content[:30]}...)")

    async def reindex_node(self, event: Event):
        """Handle knowledge node updates."""
        node_id = event.data.get("node_id")

        # Simulate reindexing
        await asyncio.sleep(0.05)

        print(f"🔄 [INDEXER] Reindexed node: {node_id}")

    async def remove_from_index(self, event: Event):
        """Handle knowledge node deletion."""
        node_id = event.data.get("node_id")

        if node_id in self.indexed_nodes:
            self.indexed_nodes.remove(node_id)

        print(f"🗑️ [INDEXER] Removed from index: {node_id}")

    def get_indexed_count(self) -> int:
        """Get number of indexed nodes."""
        return len(self.indexed_nodes)


class CacheInvalidator:
    """Simulates a cache invalidator that reacts to data changes."""

    def __init__(self, event_system: EventSystem):
        self.event_system = event_system
        self.subscriber = event_system.create_subscriber("cache_invalidator")
        self.cache_keys_invalidated: List[str] = []

        # Subscribe to change events with high priority filter
        def high_priority_filter(event: Event) -> bool:
            return event.priority in [EventPriority.HIGH, EventPriority.CRITICAL]

        self.subscriber.subscribe(
            EventType.KNOWLEDGE_NODE_CREATED, self.invalidate_cache, high_priority_filter
        )
        self.subscriber.subscribe(
            EventType.KNOWLEDGE_NODE_UPDATED, self.invalidate_cache, high_priority_filter
        )
        self.subscriber.subscribe(EventType.RELATIONSHIP_CREATED, self.invalidate_cache)

    def invalidate_cache(self, event: Event):  # Sync handler example
        """Invalidate cache entries (sync handler)."""
        cache_key = f"cache_{event.data.get('node_id', 'unknown')}"
        self.cache_keys_invalidated.append(cache_key)

        print(f"🗄️ [CACHE] Invalidated cache key: {cache_key}")

    def get_invalidation_count(self) -> int:
        """Get number of cache invalidations."""
        return len(self.cache_keys_invalidated)


class SystemHealthMonitor:
    """Monitors system health events."""

    def __init__(self, event_system: EventSystem):
        self.event_system = event_system
        self.subscriber = event_system.create_subscriber("health_monitor")
        self.alerts: List[Dict[str, Any]] = []

        # Subscribe to system events
        self.subscriber.subscribe(EventType.SYSTEM_ERROR, self.handle_system_error)
        self.subscriber.subscribe(EventType.SYSTEM_WARNING, self.handle_system_warning)
        self.subscriber.subscribe(EventType.QUERY_FAILED, self.handle_query_failure)

    async def handle_system_error(self, event: Event):
        """Handle system errors."""
        alert = {
            "type": "system_error",
            "component": event.data.get("component"),
            "details": event.data.get("details"),
            "timestamp": event.timestamp,
        }
        self.alerts.append(alert)

        print(f"🚨 [HEALTH] System error in {alert['component']}: {alert['details']}")

    async def handle_system_warning(self, event: Event):
        """Handle system warnings."""
        alert = {
            "type": "system_warning",
            "component": event.data.get("component"),
            "details": event.data.get("details"),
            "timestamp": event.timestamp,
        }
        self.alerts.append(alert)

        print(f"⚠️ [HEALTH] System warning in {alert['component']}: {alert['details']}")

    async def handle_query_failure(self, event: Event):
        """Handle query failures."""
        alert = {
            "type": "query_failure",
            "query": event.data.get("query"),
            "error": event.data.get("error"),
            "timestamp": event.timestamp,
        }
        self.alerts.append(alert)

        print(f"⚠️ [HEALTH] Query failed: {alert['query']} - {alert['error']}")

    def get_alert_count(self) -> int:
        """Get number of alerts."""
        return len(self.alerts)


async def demonstrate_basic_event_flow():
    """Demonstrate basic event publishing and subscription."""
    print("\n" + "=" * 60)
    print("BASIC EVENT FLOW DEMONSTRATION")
    print("=" * 60)

    # Create event system with persistence and batching
    config = {
        "event_bus": {
            "enable_persistence": True,
            "enable_batching": True,
            "batch_size": 5,
            "flush_interval": 2.0,
            "storage_path": "./data/events_demo",
        }
    }

    event_system = EventSystem(config)
    await event_system.initialize()

    # Create services and subscribers
    knowledge_service = KnowledgeServiceSimulator(event_system)
    query_service = QueryServiceSimulator(event_system)
    monitor = EventMonitor(event_system)
    indexer = KnowledgeIndexer(event_system)
    cache_invalidator = CacheInvalidator(event_system)

    # Wait a moment for initialization
    await asyncio.sleep(0.1)

    # Demonstrate knowledge operations
    print("\n🚀 Starting knowledge operations...")
    await knowledge_service.create_knowledge_node(
        "node_001", "Python is a programming language", "wikipedia"
    )

    await knowledge_service.create_knowledge_node(
        "node_002", "Machine learning uses algorithms to learn patterns", "textbook"
    )

    await knowledge_service.create_relationship("rel_001", "node_001", "node_002", "related_to")

    await knowledge_service.update_knowledge_node(
        "node_001", "Python is a high-level programming language"
    )

    # Demonstrate query operations
    print("\n🔍 Starting query operations...")
    await query_service.execute_query("What is Python?", "semantic_search")
    await query_service.execute_query("Machine learning examples", "keyword_search")
    await query_service.execute_query("Complex query that might fail", "complex_analysis")

    # Wait for event processing
    await asyncio.sleep(3)

    # Show statistics
    print(f"\n📊 Event Statistics:")
    print(f"   Monitor Events: {monitor.get_statistics()}")
    print(f"   Indexed Nodes: {indexer.get_indexed_count()}")
    print(f"   Cache Invalidations: {cache_invalidator.get_invalidation_count()}")

    # Show system metrics
    metrics = event_system.get_system_metrics()
    print(f"\n📈 System Metrics:")
    print(f"   Events Published: {metrics['event_bus']['events_published']}")
    print(f"   Events Processed: {metrics['event_bus']['events_processed']}")
    print(f"   Success Rate: {metrics['event_bus']['success_rate']:.2%}")
    print(f"   Queue Size: {metrics['event_bus']['queue_size']}")

    await event_system.shutdown()


async def demonstrate_custom_events_and_filtering():
    """Demonstrate custom events and advanced filtering."""
    print("\n" + "=" * 60)
    print("CUSTOM EVENTS AND FILTERING DEMONSTRATION")
    print("=" * 60)

    event_system = EventSystem()
    await event_system.initialize()

    # Create publisher for custom events
    custom_publisher = event_system.create_publisher("custom_app")

    # Create subscriber with filtering
    filtered_subscriber = event_system.create_subscriber("filtered_processor")

    # Filter for high-priority custom events only
    def high_priority_custom_filter(event: Event) -> bool:
        return (
            event.event_type == EventType.CUSTOM
            and event.priority == EventPriority.HIGH
            and event.data.get("custom_data", {}).get("category") == "important"
        )

    async def handle_important_custom_event(event: Event):
        custom_data = event.data.get("custom_data", {})
        print(f"🎯 [FILTERED] Important custom event: {custom_data.get('message')}")

    filtered_subscriber.subscribe(
        EventType.CUSTOM, handle_important_custom_event, high_priority_custom_filter
    )

    # Publish various custom events
    print("\n🎨 Publishing custom events...")

    await custom_publisher.publish_custom_event(
        "user_action",
        {"message": "User logged in", "category": "normal"},
        priority=EventPriority.NORMAL,
    )

    await custom_publisher.publish_custom_event(
        "security_alert",
        {"message": "Suspicious activity detected", "category": "important"},
        priority=EventPriority.HIGH,
    )

    await custom_publisher.publish_custom_event(
        "maintenance",
        {"message": "System maintenance scheduled", "category": "important"},
        priority=EventPriority.LOW,  # Won't be processed due to filter
    )

    await custom_publisher.publish_custom_event(
        "critical_error",
        {"message": "Database connection lost", "category": "important"},
        priority=EventPriority.HIGH,
    )

    await asyncio.sleep(1)
    await event_system.shutdown()


async def demonstrate_system_monitoring():
    """Demonstrate system monitoring and health events."""
    print("\n" + "=" * 60)
    print("SYSTEM MONITORING DEMONSTRATION")
    print("=" * 60)

    event_system = EventSystem()
    await event_system.initialize()

    # Set up monitoring
    health_monitor = SystemHealthMonitor(event_system)
    query_service = QueryServiceSimulator(event_system)

    # Create system publisher
    system_publisher = event_system.create_publisher("system_monitor")

    print("\n🔧 Simulating system events...")

    # Simulate system events
    await system_publisher.publish_system_event(
        "warning", "database", {"message": "High connection pool usage", "usage": 0.85}
    )

    await system_publisher.publish_system_event(
        "error", "llm_provider", {"message": "API rate limit exceeded", "provider": "openai"}
    )

    # Execute some queries (some will fail)
    for i in range(5):
        await query_service.execute_query(f"test query {i}")
        await asyncio.sleep(0.1)

    await asyncio.sleep(1)

    print(f"\n🚨 Health Monitor Summary:")
    print(f"   Total Alerts: {health_monitor.get_alert_count()}")

    await event_system.shutdown()


async def demonstrate_event_persistence_and_replay():
    """Demonstrate event persistence and replay capabilities."""
    print("\n" + "=" * 60)
    print("EVENT PERSISTENCE AND REPLAY DEMONSTRATION")
    print("=" * 60)

    # Ensure data directory exists
    data_dir = Path("./data/events_replay_demo")
    data_dir.mkdir(parents=True, exist_ok=True)

    config = {"event_bus": {"enable_persistence": True, "storage_path": str(data_dir)}}

    # Phase 1: Generate and persist events
    print("\n📝 Phase 1: Generating events...")
    event_system = EventSystem(config)
    await event_system.initialize()

    knowledge_service = KnowledgeServiceSimulator(event_system)

    # Generate some events
    for i in range(3):
        await knowledge_service.create_knowledge_node(
            f"replay_node_{i}", f"Content for node {i}", "replay_demo"
        )
        await asyncio.sleep(0.1)

    await asyncio.sleep(1)  # Let events process
    await event_system.shutdown()

    # Phase 2: Create new system and replay events
    print("\n🔄 Phase 2: Replaying events...")
    new_event_system = EventSystem(config)
    await new_event_system.initialize()

    # Set up new subscriber
    replay_subscriber = new_event_system.create_subscriber("replay_processor")

    async def handle_replayed_event(event: Event):
        print(f"🔄 [REPLAY] Processed event: {event.event_type.value} - {event.id[:8]}...")

    replay_subscriber.subscribe(EventType.KNOWLEDGE_NODE_CREATED, handle_replayed_event)

    # Replay events
    replay_count = await new_event_system.replay_events()
    print(f"✅ Replayed {replay_count} events")

    await asyncio.sleep(1)
    await new_event_system.shutdown()


async def main():
    """Run all demonstrations."""
    print("🚀 Memory Engine Event System Demonstration")
    print("This example showcases comprehensive event-driven architecture")

    try:
        await demonstrate_basic_event_flow()
        await demonstrate_custom_events_and_filtering()
        await demonstrate_system_monitoring()
        await demonstrate_event_persistence_and_replay()

        print("\n" + "=" * 60)
        print("✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
