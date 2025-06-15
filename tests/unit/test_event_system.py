"""
Unit tests for the Memory Engine Event System.

Tests cover all major functionality including event creation, publishing,
subscription, filtering, persistence, and system management.
"""

import asyncio
import json
import pytest
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock

from memory_core.orchestrator.event_system import (
    Event,
    EventType,
    EventPriority,
    EventStatus,
    EventMetadata,
    KnowledgeChangeEvent,
    RelationshipChangeEvent,
    QueryEvent,
    SystemEvent,
    CustomEvent,
    EventBus,
    EventSubscriber,
    EventPublisher,
    EventSystem,
    EventPersistence,
    DeadLetterQueue,
    EventMetrics,
    EventBatchProcessor,
    EventThrottler,
)


class TestEvent:
    """Test Event class functionality."""

    def test_event_creation(self):
        """Test basic event creation."""
        event = Event(
            event_type=EventType.KNOWLEDGE_NODE_CREATED,
            data={"test": "data"},
            priority=EventPriority.HIGH,
        )

        assert event.event_type == EventType.KNOWLEDGE_NODE_CREATED
        assert event.data == {"test": "data"}
        assert event.priority == EventPriority.HIGH
        assert event.status == EventStatus.PENDING
        assert event.id is not None
        assert event.timestamp > 0

    def test_event_serialization(self):
        """Test event serialization and deserialization."""
        original_event = Event(
            event_type=EventType.QUERY_EXECUTED,
            data={"query": "test query", "results": ["result1", "result2"]},
            priority=EventPriority.NORMAL,
        )

        # Serialize to dict
        event_dict = original_event.to_dict()
        assert isinstance(event_dict, dict)
        assert event_dict["event_type"] == EventType.QUERY_EXECUTED.value
        assert event_dict["data"]["query"] == "test query"

        # Deserialize back to event
        restored_event = Event.from_dict(event_dict)
        assert restored_event.event_type == original_event.event_type
        assert restored_event.data == original_event.data
        assert restored_event.priority == original_event.priority
        assert restored_event.id == original_event.id

    def test_event_processing_lifecycle(self):
        """Test event processing status changes."""
        event = Event(event_type=EventType.CUSTOM)

        # Initial state
        assert event.status == EventStatus.PENDING
        assert event.processing_started is None
        assert event.processing_completed is None

        # Mark processing started
        event.mark_processing_started()
        assert event.status == EventStatus.IN_PROGRESS
        assert event.processing_started is not None

        # Mark processing completed
        event.mark_processing_completed()
        assert event.status == EventStatus.PROCESSED
        assert event.processing_completed is not None

        # Check duration calculation
        duration = event.get_processing_duration()
        assert duration is not None
        assert duration >= 0

    def test_event_failure_and_retry(self):
        """Test event failure and retry logic."""
        event = Event(event_type=EventType.CUSTOM)
        event.metadata.max_retries = 2

        # Mark as failed
        event.mark_processing_failed("Test error")
        assert event.status == EventStatus.FAILED
        assert event.error_message == "Test error"

        # Should be retryable
        assert event.should_retry() == True

        # Increment retry
        event.increment_retry()
        assert event.metadata.retry_count == 1
        assert event.status == EventStatus.RETRYING

        # Another failure and retry
        event.mark_processing_failed("Another error")
        event.increment_retry()
        assert event.metadata.retry_count == 2

        # Should not retry anymore
        event.mark_processing_failed("Final error")
        assert event.should_retry() == False


class TestSpecificEventTypes:
    """Test specific event type implementations."""

    def test_knowledge_change_event(self):
        """Test KnowledgeChangeEvent creation."""
        event = KnowledgeChangeEvent(
            action="created",
            node_id="test_node_123",
            node_data={"content": "test content", "source": "test"},
        )

        assert event.event_type == EventType.KNOWLEDGE_NODE_CREATED
        assert event.data["action"] == "created"
        assert event.data["node_id"] == "test_node_123"
        assert event.data["node_data"]["content"] == "test content"

    def test_relationship_change_event(self):
        """Test RelationshipChangeEvent creation."""
        event = RelationshipChangeEvent(
            action="updated",
            relationship_id="rel_456",
            relationship_data={"type": "related_to", "strength": 0.8},
        )

        assert event.event_type == EventType.RELATIONSHIP_UPDATED
        assert event.data["relationship_id"] == "rel_456"
        assert event.data["relationship_data"]["type"] == "related_to"

    def test_query_event(self):
        """Test QueryEvent creation."""
        # Successful query
        success_event = QueryEvent(
            query_type="semantic_search", query="test query", results=["result1", "result2"]
        )

        assert success_event.event_type == EventType.QUERY_EXECUTED
        assert success_event.data["query"] == "test query"
        assert success_event.data["results"] == ["result1", "result2"]

        # Failed query
        failed_event = QueryEvent(
            query_type="semantic_search", query="failing query", error="Connection timeout"
        )

        assert failed_event.event_type == EventType.QUERY_FAILED
        assert failed_event.data["error"] == "Connection timeout"

    def test_system_event(self):
        """Test SystemEvent creation."""
        event = SystemEvent(
            system_event_type="error",
            component="database",
            details={"error_code": 500, "message": "Connection lost"},
        )

        assert event.event_type == EventType.SYSTEM_ERROR
        assert event.data["component"] == "database"
        assert event.data["details"]["error_code"] == 500

    def test_custom_event(self):
        """Test CustomEvent creation."""
        event = CustomEvent(
            custom_type="user_action", custom_data={"action": "login", "user_id": "user123"}
        )

        assert event.event_type == EventType.CUSTOM
        assert event.data["custom_type"] == "user_action"
        assert event.data["custom_data"]["action"] == "login"


class TestEventSubscriber:
    """Test EventSubscriber functionality."""

    def test_subscriber_creation(self):
        """Test subscriber creation and basic properties."""
        subscriber = EventSubscriber("test_subscriber")

        assert subscriber.subscriber_id == "test_subscriber"
        assert subscriber.active == True
        assert len(subscriber.handlers) == 0

    def test_subscription_management(self):
        """Test subscribing and unsubscribing."""
        subscriber = EventSubscriber("test_subscriber")

        def test_handler(event: Event):
            pass

        # Subscribe
        subscriber.subscribe(EventType.KNOWLEDGE_NODE_CREATED, test_handler)
        assert EventType.KNOWLEDGE_NODE_CREATED in subscriber.handlers
        assert len(subscriber.handlers[EventType.KNOWLEDGE_NODE_CREATED]) == 1

        # Unsubscribe
        subscriber.unsubscribe(EventType.KNOWLEDGE_NODE_CREATED, test_handler)
        assert EventType.KNOWLEDGE_NODE_CREATED not in subscriber.handlers

    @pytest.mark.asyncio
    async def test_sync_handler_execution(self):
        """Test execution of synchronous event handlers."""
        subscriber = EventSubscriber("test_subscriber")
        handled_events = []

        def sync_handler(event: Event):
            handled_events.append(event.id)

        subscriber.subscribe(EventType.CUSTOM, sync_handler)

        event = Event(event_type=EventType.CUSTOM, data={"test": "data"})
        result = await subscriber.handle_event(event)

        assert result == True
        assert len(handled_events) == 1
        assert handled_events[0] == event.id

    @pytest.mark.asyncio
    async def test_async_handler_execution(self):
        """Test execution of asynchronous event handlers."""
        subscriber = EventSubscriber("test_subscriber")
        handled_events = []

        async def async_handler(event: Event):
            await asyncio.sleep(0.01)  # Simulate async work
            handled_events.append(event.id)

        subscriber.subscribe(EventType.CUSTOM, async_handler)

        event = Event(event_type=EventType.CUSTOM, data={"test": "data"})
        result = await subscriber.handle_event(event)

        assert result == True
        assert len(handled_events) == 1
        assert handled_events[0] == event.id

    @pytest.mark.asyncio
    async def test_event_filtering(self):
        """Test event filtering functionality."""
        subscriber = EventSubscriber("test_subscriber")
        handled_events = []

        def handler(event: Event):
            handled_events.append(event.id)

        def priority_filter(event: Event) -> bool:
            return event.priority == EventPriority.HIGH

        subscriber.subscribe(EventType.CUSTOM, handler, priority_filter)

        # High priority event (should be handled)
        high_event = Event(event_type=EventType.CUSTOM, priority=EventPriority.HIGH)
        result1 = await subscriber.handle_event(high_event)

        # Low priority event (should be filtered out)
        low_event = Event(event_type=EventType.CUSTOM, priority=EventPriority.LOW)
        result2 = await subscriber.handle_event(low_event)

        assert result1 == True
        assert result2 == False
        assert len(handled_events) == 1
        assert handled_events[0] == high_event.id


class TestEventPublisher:
    """Test EventPublisher functionality."""

    @pytest.mark.asyncio
    async def test_publisher_creation_and_publishing(self):
        """Test publisher creation and basic publishing."""
        # Create mock event bus
        mock_event_bus = AsyncMock()
        mock_event_bus.publish_event = AsyncMock()
        mock_event_bus.publish_events = AsyncMock()

        publisher = EventPublisher("test_publisher", mock_event_bus)

        # Test single event publishing
        event = Event(event_type=EventType.CUSTOM)
        await publisher.publish(event)

        assert event.metadata.source_component == "test_publisher"
        mock_event_bus.publish_event.assert_called_once_with(event)
        assert publisher.published_count == 1

        # Test batch publishing
        events = [Event(event_type=EventType.CUSTOM) for _ in range(3)]
        await publisher.publish_batch(events)

        mock_event_bus.publish_events.assert_called_once_with(events)
        assert publisher.published_count == 4

        # Verify all events have source component set
        for event in events:
            assert event.metadata.source_component == "test_publisher"

    @pytest.mark.asyncio
    async def test_convenience_publishing_methods(self):
        """Test convenience methods for publishing specific event types."""
        mock_event_bus = AsyncMock()
        mock_event_bus.publish_event = AsyncMock()

        publisher = EventPublisher("test_publisher", mock_event_bus)

        # Test knowledge change publishing
        await publisher.publish_knowledge_change("created", "node_123", {"content": "test"})

        # Verify the call
        mock_event_bus.publish_event.assert_called()
        published_event = mock_event_bus.publish_event.call_args[0][0]
        assert isinstance(published_event, KnowledgeChangeEvent)
        assert published_event.data["node_id"] == "node_123"


class TestEventBus:
    """Test EventBus functionality."""

    def test_event_bus_creation(self):
        """Test event bus creation with configuration."""
        config = {"max_queue_size": 5000, "enable_persistence": False, "enable_batching": False}

        event_bus = EventBus(config)

        assert event_bus.config == config
        assert event_bus.persistence is None
        assert event_bus.batch_processor is None
        assert event_bus.running == False

    def test_subscriber_registration(self):
        """Test subscriber registration and unregistration."""
        event_bus = EventBus()
        subscriber = EventSubscriber("test_subscriber")

        # Register
        event_bus.register_subscriber(subscriber)
        assert "test_subscriber" in event_bus.subscribers
        assert event_bus.subscribers["test_subscriber"] is subscriber

        # Unregister
        event_bus.unregister_subscriber("test_subscriber")
        assert "test_subscriber" not in event_bus.subscribers

    def test_publisher_creation(self):
        """Test publisher creation."""
        event_bus = EventBus()
        publisher = event_bus.create_publisher("test_publisher")

        assert isinstance(publisher, EventPublisher)
        assert publisher.publisher_id == "test_publisher"
        assert publisher.event_bus is event_bus

    @pytest.mark.asyncio
    async def test_event_publishing_and_processing(self):
        """Test end-to-end event publishing and processing."""
        event_bus = EventBus({"enable_persistence": False, "enable_batching": False})

        # Set up subscriber
        subscriber = EventSubscriber("test_subscriber")
        handled_events = []

        def handler(event: Event):
            handled_events.append(event.id)

        subscriber.subscribe(EventType.CUSTOM, handler)
        event_bus.register_subscriber(subscriber)

        # Start processing
        await event_bus.start_processing()

        # Publish event
        event = Event(event_type=EventType.CUSTOM, data={"test": "data"})
        await event_bus.publish_event(event)

        # Wait for processing
        await asyncio.sleep(0.1)

        # Stop processing
        await event_bus.stop_processing()

        # Verify event was handled
        assert len(handled_events) == 1
        assert handled_events[0] == event.id
        assert event.status == EventStatus.PROCESSED

    @pytest.mark.asyncio
    async def test_event_retry_mechanism(self):
        """Test event retry mechanism for failed events."""
        event_bus = EventBus({"enable_persistence": False, "enable_batching": False})

        # Set up subscriber with failing handler
        subscriber = EventSubscriber("failing_subscriber")
        attempt_count = 0

        def failing_handler(event: Event):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:  # Fail first 2 attempts
                raise Exception("Simulated failure")

        subscriber.subscribe(EventType.CUSTOM, failing_handler)
        event_bus.register_subscriber(subscriber)

        await event_bus.start_processing()

        # Publish event with max retries
        event = Event(event_type=EventType.CUSTOM)
        event.metadata.max_retries = 2
        await event_bus.publish_event(event)

        # Wait for processing and retries
        await asyncio.sleep(0.2)
        await event_bus.stop_processing()

        # Should have attempted 3 times (1 initial + 2 retries)
        assert attempt_count == 3
        assert event.status == EventStatus.PROCESSED


class TestEventPersistence:
    """Test event persistence functionality."""

    def test_persistence_creation(self):
        """Test persistence component creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)
            persistence = EventPersistence(storage_path)

            assert persistence.storage_path == storage_path
            assert persistence.event_log_file.exists() == False  # Not created until first write

    def test_event_persistence_and_loading(self):
        """Test persisting and loading events."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)
            persistence = EventPersistence(storage_path)

            # Create test events
            events = [
                Event(event_type=EventType.KNOWLEDGE_NODE_CREATED, data={"node_id": "1"}),
                Event(event_type=EventType.QUERY_EXECUTED, data={"query": "test"}),
                Event(event_type=EventType.CUSTOM, data={"custom": "data"}),
            ]

            # Persist events
            persistence.persist_events(events)

            # Load events back
            loaded_events = persistence.load_events()

            assert len(loaded_events) == 3
            assert loaded_events[0].event_type == EventType.KNOWLEDGE_NODE_CREATED
            assert loaded_events[1].event_type == EventType.QUERY_EXECUTED
            assert loaded_events[2].event_type == EventType.CUSTOM

    def test_checkpoint_functionality(self):
        """Test checkpoint save/load functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)
            persistence = EventPersistence(storage_path)

            # Save checkpoint
            checkpoint_data = {
                "last_processed_event": "event_123",
                "processing_timestamp": time.time(),
                "subscriber_states": {"sub1": "active", "sub2": "paused"},
            }

            persistence.save_checkpoint(checkpoint_data)

            # Load checkpoint
            loaded_checkpoint = persistence.load_checkpoint()

            assert loaded_checkpoint["last_processed_event"] == "event_123"
            assert "processing_timestamp" in loaded_checkpoint
            assert loaded_checkpoint["subscriber_states"]["sub1"] == "active"


class TestEventMetrics:
    """Test event metrics functionality."""

    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = EventMetrics()

        assert metrics.events_published == 0
        assert metrics.events_processed == 0
        assert metrics.events_failed == 0
        assert len(metrics.processing_times) == 0
        assert len(metrics.event_type_counts) == 0

    def test_metrics_recording(self):
        """Test metrics recording functionality."""
        metrics = EventMetrics()

        # Create test event
        event = Event(event_type=EventType.KNOWLEDGE_NODE_CREATED)
        event.mark_processing_started()
        time.sleep(0.01)  # Small delay
        event.mark_processing_completed()

        # Record metrics
        metrics.record_event_published(event)
        metrics.record_event_processed(event, "test_handler")

        # Verify metrics
        assert metrics.events_published == 1
        assert metrics.events_processed == 1
        assert metrics.event_type_counts[EventType.KNOWLEDGE_NODE_CREATED] == 1
        assert len(metrics.processing_times) == 1
        assert "test_handler" in metrics.handler_performance

    def test_metrics_summary(self):
        """Test metrics summary generation."""
        metrics = EventMetrics()

        # Record some metrics
        event = Event(event_type=EventType.CUSTOM)
        event.mark_processing_started()
        event.mark_processing_completed()

        metrics.record_event_published(event)
        metrics.record_event_processed(event, "handler1")

        # Get summary
        summary = metrics.get_metrics()

        assert summary["events_published"] == 1
        assert summary["events_processed"] == 1
        assert summary["events_failed"] == 0
        assert summary["success_rate"] == 1.0
        assert "event_type_distribution" in summary
        assert "handler_performance" in summary


class TestDeadLetterQueue:
    """Test dead letter queue functionality."""

    def test_dead_letter_queue_creation(self):
        """Test dead letter queue creation."""
        dlq = DeadLetterQueue(max_size=100)

        assert dlq.max_size == 100
        assert len(dlq.queue) == 0

    def test_failed_event_handling(self):
        """Test adding and retrieving failed events."""
        dlq = DeadLetterQueue(max_size=10)

        # Create failed event
        event = Event(event_type=EventType.CUSTOM)
        event.mark_processing_failed("Test failure")

        # Add to dead letter queue
        dlq.add_failed_event(event)

        assert len(dlq.queue) == 1
        assert event.status == EventStatus.DEAD_LETTER

        # Retrieve failed events
        failed_events = dlq.get_failed_events()
        assert len(failed_events) == 1
        assert failed_events[0].id == event.id

    def test_retry_failed_event(self):
        """Test retrying failed events."""
        dlq = DeadLetterQueue()

        event = Event(event_type=EventType.CUSTOM)
        event.mark_processing_failed("Test failure")
        dlq.add_failed_event(event)

        # Retry the event
        success = dlq.retry_failed_event(event.id)

        assert success == True
        assert len(dlq.queue) == 0
        assert event.status == EventStatus.PENDING
        assert event.error_message is None


@pytest.mark.asyncio
async def test_event_throttling():
    """Test event throttling functionality."""
    throttler = EventThrottler(max_events_per_second=10.0)  # 10 events per second

    start_time = time.time()

    # Process 3 events
    for _ in range(3):
        await throttler.throttle()

    elapsed_time = time.time() - start_time

    # Should take at least 0.2 seconds (3 events at 10/sec = 0.1s intervals)
    assert elapsed_time >= 0.2


@pytest.mark.asyncio
async def test_event_batch_processor():
    """Test event batch processing functionality."""
    batch_processor = EventBatchProcessor(batch_size=3, flush_interval=1.0)

    processed_batches = []

    async def batch_handler(events: List[Event]):
        processed_batches.append(len(events))

    batch_processor.add_handler(batch_handler)

    # Add events one by one
    for i in range(5):
        event = Event(event_type=EventType.CUSTOM, data={"index": i})
        await batch_processor.add_event(event)

    # Force flush
    await batch_processor.flush()

    # Should have processed one batch of 3 and one batch of 2
    assert len(processed_batches) == 2
    assert 3 in processed_batches
    assert 2 in processed_batches


class TestEventSystem:
    """Test complete EventSystem functionality."""

    @pytest.mark.asyncio
    async def test_event_system_initialization(self):
        """Test event system initialization and shutdown."""
        event_system = EventSystem()

        # Initialize
        await event_system.initialize()
        assert event_system.event_bus.running == True

        # Shutdown
        await event_system.shutdown()
        assert event_system.event_bus.running == False

    @pytest.mark.asyncio
    async def test_publisher_and_subscriber_creation(self):
        """Test creating publishers and subscribers."""
        event_system = EventSystem()
        await event_system.initialize()

        # Create publisher and subscriber
        publisher = event_system.create_publisher("test_pub")
        subscriber = event_system.create_subscriber("test_sub")

        assert isinstance(publisher, EventPublisher)
        assert isinstance(subscriber, EventSubscriber)
        assert "test_pub" in event_system.publishers
        assert "test_sub" in event_system.subscribers

        await event_system.shutdown()

    @pytest.mark.asyncio
    async def test_end_to_end_event_flow(self):
        """Test complete end-to-end event flow."""
        event_system = EventSystem()
        await event_system.initialize()

        # Set up publisher and subscriber
        publisher = event_system.create_publisher("test_publisher")
        subscriber = event_system.create_subscriber("test_subscriber")

        received_events = []

        async def event_handler(event: Event):
            received_events.append(event.data["message"])

        subscriber.subscribe(EventType.CUSTOM, event_handler)

        # Publish custom event
        await publisher.publish_custom_event("test_event", {"message": "Hello, Event System!"})

        # Wait for processing
        await asyncio.sleep(0.1)

        # Verify event was received
        assert len(received_events) == 1
        assert received_events[0] == "Hello, Event System!"

        await event_system.shutdown()

    @pytest.mark.asyncio
    async def test_system_metrics(self):
        """Test system metrics collection."""
        event_system = EventSystem()
        await event_system.initialize()

        publisher = event_system.create_publisher("metrics_test")

        # Publish some events
        for i in range(3):
            await publisher.publish_custom_event("test", {"index": i})

        await asyncio.sleep(0.1)

        # Get metrics
        metrics = event_system.get_system_metrics()

        assert "event_bus" in metrics
        assert "publishers" in metrics
        assert "subscribers" in metrics
        assert metrics["event_bus"]["events_published"] >= 3

        await event_system.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
