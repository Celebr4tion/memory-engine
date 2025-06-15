"""
Comprehensive Event System for Memory Engine Inter-Module Communication

This module provides a robust event-driven architecture for inter-module communication
in the Memory Engine. It supports async/sync event handling, event persistence,
replay capabilities, and comprehensive monitoring.
"""

import asyncio
import json
import time
import uuid
import pickle
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from typing import (
    Dict,
    List,
    Any,
    Optional,
    Union,
    Callable,
    Awaitable,
    AsyncGenerator,
    Set,
    Tuple,
    Type,
    Protocol,
)
import logging
from concurrent.futures import ThreadPoolExecutor
import threading

from ..monitoring.structured_logger import get_logger, CorrelationIdManager


logger = get_logger(__name__, "EventSystem")


class EventType(Enum):
    """Types of events in the Memory Engine."""

    # Knowledge Events
    KNOWLEDGE_NODE_CREATED = "knowledge.node.created"
    KNOWLEDGE_NODE_UPDATED = "knowledge.node.updated"
    KNOWLEDGE_NODE_DELETED = "knowledge.node.deleted"

    # Relationship Events
    RELATIONSHIP_CREATED = "relationship.created"
    RELATIONSHIP_UPDATED = "relationship.updated"
    RELATIONSHIP_DELETED = "relationship.deleted"

    # Query Events
    QUERY_EXECUTED = "query.executed"
    QUERY_FAILED = "query.failed"
    QUERY_CACHED = "query.cached"

    # System Events
    SYSTEM_HEALTH_CHANGED = "system.health.changed"
    SYSTEM_ERROR = "system.error"
    SYSTEM_WARNING = "system.warning"
    SYSTEM_MAINTENANCE = "system.maintenance"

    # Provider Events
    LLM_PROVIDER_CONNECTED = "llm.provider.connected"
    LLM_PROVIDER_DISCONNECTED = "llm.provider.disconnected"
    EMBEDDING_PROVIDER_CHANGED = "embedding.provider.changed"
    STORAGE_BACKEND_CHANGED = "storage.backend.changed"

    # Custom Events
    CUSTOM = "custom"


class EventPriority(Enum):
    """Priority levels for events."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class EventStatus(Enum):
    """Status of event processing."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PROCESSED = "processed"
    FAILED = "failed"
    RETRYING = "retrying"
    DEAD_LETTER = "dead_letter"


@dataclass
class EventMetadata:
    """Metadata for events."""

    correlation_id: Optional[str] = None
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    source_component: Optional[str] = None
    target_component: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class Event:
    """
    Core event class for the Memory Engine event system.

    All events in the system inherit from this base class, providing
    consistent structure and metadata tracking.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.CUSTOM
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: EventMetadata = field(default_factory=EventMetadata)
    priority: EventPriority = EventPriority.NORMAL
    status: EventStatus = EventStatus.PENDING
    error_message: Optional[str] = None
    processing_started: Optional[float] = None
    processing_completed: Optional[float] = None

    def __post_init__(self):
        """Initialize correlation ID if not provided."""
        if not self.metadata.correlation_id:
            self.metadata.correlation_id = CorrelationIdManager.get_correlation_id()
        if not self.metadata.request_id:
            self.metadata.request_id = CorrelationIdManager.get_request_id()
        if not self.metadata.user_id:
            self.metadata.user_id = CorrelationIdManager.get_user_id()

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        result = asdict(self)
        result["event_type"] = self.event_type.value
        result["priority"] = self.priority.value
        result["status"] = self.status.value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create event from dictionary."""
        # Handle enum conversions
        data["event_type"] = EventType(data["event_type"])
        data["priority"] = EventPriority(data["priority"])
        data["status"] = EventStatus(data["status"])

        # Handle metadata
        if "metadata" in data and isinstance(data["metadata"], dict):
            data["metadata"] = EventMetadata(**data["metadata"])

        return cls(**data)

    def mark_processing_started(self):
        """Mark event as processing started."""
        self.status = EventStatus.IN_PROGRESS
        self.processing_started = time.time()

    def mark_processing_completed(self):
        """Mark event as successfully processed."""
        self.status = EventStatus.PROCESSED
        self.processing_completed = time.time()

    def mark_processing_failed(self, error_message: str):
        """Mark event as failed with error message."""
        self.status = EventStatus.FAILED
        self.error_message = error_message
        self.processing_completed = time.time()

    def should_retry(self) -> bool:
        """Check if event should be retried."""
        return (
            self.status == EventStatus.FAILED
            and self.metadata.retry_count < self.metadata.max_retries
        )

    def increment_retry(self):
        """Increment retry count and update status."""
        self.metadata.retry_count += 1
        self.status = EventStatus.RETRYING

    def get_processing_duration(self) -> Optional[float]:
        """Get processing duration in seconds."""
        if self.processing_started and self.processing_completed:
            return self.processing_completed - self.processing_started
        return None


# Specific event types
@dataclass
class KnowledgeChangeEvent(Event):
    """Event for knowledge node changes."""

    def __init__(
        self,
        action: str,
        node_id: str,
        node_data: Dict[str, Any],
        source: str = "unknown",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.event_type = getattr(EventType, f"KNOWLEDGE_NODE_{action.upper()}")
        self.data = {"action": action, "node_id": node_id, "node_data": node_data, "source": source}


@dataclass
class RelationshipChangeEvent(Event):
    """Event for relationship changes."""

    def __init__(
        self, action: str, relationship_id: str, relationship_data: Dict[str, Any], **kwargs
    ):
        super().__init__(**kwargs)
        self.event_type = getattr(EventType, f"RELATIONSHIP_{action.upper()}")
        self.data = {
            "action": action,
            "relationship_id": relationship_id,
            "relationship_data": relationship_data,
        }


@dataclass
class QueryEvent(Event):
    """Event for query operations."""

    def __init__(
        self,
        query_type: str,
        query: str,
        results: Optional[Any] = None,
        error: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if error:
            self.event_type = EventType.QUERY_FAILED
        else:
            self.event_type = EventType.QUERY_EXECUTED

        self.data = {"query_type": query_type, "query": query, "results": results, "error": error}


@dataclass
class SystemEvent(Event):
    """Event for system-level changes."""

    def __init__(self, system_event_type: str, component: str, details: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.event_type = getattr(EventType, f"SYSTEM_{system_event_type.upper()}")
        self.data = {"component": component, "details": details}


@dataclass
class CustomEvent(Event):
    """Event for custom user-defined events."""

    def __init__(self, custom_type: str, custom_data: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.event_type = EventType.CUSTOM
        self.data = {"custom_type": custom_type, "custom_data": custom_data}


# Event handler protocols
class EventHandler(Protocol):
    """Protocol for event handlers."""

    def __call__(self, event: Event) -> Union[None, Awaitable[None]]:
        """Handle an event."""
        ...


class AsyncEventHandler(Protocol):
    """Protocol for async event handlers."""

    async def __call__(self, event: Event) -> None:
        """Handle an event asynchronously."""
        ...


class EventFilter(Protocol):
    """Protocol for event filters."""

    def __call__(self, event: Event) -> bool:
        """Return True if event should be processed."""
        ...


class EventBatchProcessor:
    """Processes events in batches for efficiency."""

    def __init__(self, batch_size: int = 10, flush_interval: float = 1.0):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.batch: List[Event] = []
        self.handlers: List[Callable[[List[Event]], Awaitable[None]]] = []
        self._last_flush = time.time()
        self._lock = asyncio.Lock()

    def add_handler(self, handler: Callable[[List[Event]], Awaitable[None]]):
        """Add batch handler."""
        self.handlers.append(handler)

    async def add_event(self, event: Event):
        """Add event to batch."""
        async with self._lock:
            self.batch.append(event)

            # Check if we should flush
            should_flush = (
                len(self.batch) >= self.batch_size
                or time.time() - self._last_flush >= self.flush_interval
            )

            if should_flush:
                await self._flush_batch()

    async def _flush_batch(self):
        """Flush current batch to handlers."""
        if not self.batch:
            return

        batch_to_process = self.batch.copy()
        self.batch.clear()
        self._last_flush = time.time()

        # Process batch with all handlers
        for handler in self.handlers:
            try:
                await handler(batch_to_process)
            except Exception as e:
                logger.error(f"Batch handler failed: {e}")

    async def flush(self):
        """Force flush current batch."""
        async with self._lock:
            await self._flush_batch()


class EventThrottler:
    """Throttles event processing to prevent overwhelming consumers."""

    def __init__(self, max_events_per_second: float = 100.0):
        self.max_events_per_second = max_events_per_second
        self.min_interval = 1.0 / max_events_per_second
        self.last_event_time = 0.0
        self._lock = asyncio.Lock()

    async def throttle(self):
        """Apply throttling delay if needed."""
        async with self._lock:
            current_time = time.time()
            time_since_last = current_time - self.last_event_time

            if time_since_last < self.min_interval:
                delay = self.min_interval - time_since_last
                await asyncio.sleep(delay)

            self.last_event_time = time.time()


class EventPersistence:
    """Handles event persistence and recovery."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("./data/events")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.event_log_file = self.storage_path / "event_log.jsonl"
        self.checkpoint_file = self.storage_path / "checkpoint.json"
        self._lock = threading.Lock()

    def persist_event(self, event: Event):
        """Persist event to storage."""
        try:
            with self._lock:
                with open(self.event_log_file, "a") as f:
                    json.dump(event.to_dict(), f)
                    f.write("\n")
        except Exception as e:
            logger.error(f"Failed to persist event {event.id}: {e}")

    def persist_events(self, events: List[Event]):
        """Persist multiple events."""
        try:
            with self._lock:
                with open(self.event_log_file, "a") as f:
                    for event in events:
                        json.dump(event.to_dict(), f)
                        f.write("\n")
        except Exception as e:
            logger.error(f"Failed to persist events: {e}")

    def load_events(self, from_timestamp: Optional[float] = None) -> List[Event]:
        """Load events from storage."""
        events = []

        if not self.event_log_file.exists():
            return events

        try:
            with open(self.event_log_file, "r") as f:
                for line in f:
                    try:
                        event_data = json.loads(line.strip())
                        event = Event.from_dict(event_data)

                        if from_timestamp is None or event.timestamp >= from_timestamp:
                            events.append(event)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Failed to load events: {e}")

        return events

    def save_checkpoint(self, checkpoint_data: Dict[str, Any]):
        """Save checkpoint data."""
        try:
            with self._lock:
                with open(self.checkpoint_file, "w") as f:
                    json.dump(checkpoint_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self) -> Dict[str, Any]:
        """Load checkpoint data."""
        if not self.checkpoint_file.exists():
            return {}

        try:
            with open(self.checkpoint_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return {}


class DeadLetterQueue:
    """Handles events that failed processing."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.queue: deque = deque(maxlen=max_size)
        self._lock = threading.Lock()

    def add_failed_event(self, event: Event):
        """Add failed event to dead letter queue."""
        with self._lock:
            event.status = EventStatus.DEAD_LETTER
            self.queue.append(event)

        logger.warning(f"Event {event.id} added to dead letter queue: {event.error_message}")

    def get_failed_events(self) -> List[Event]:
        """Get all failed events."""
        with self._lock:
            return list(self.queue)

    def retry_failed_event(self, event_id: str) -> bool:
        """Retry a specific failed event."""
        with self._lock:
            for i, event in enumerate(self.queue):
                if event.id == event_id:
                    event.status = EventStatus.PENDING
                    event.error_message = None
                    del self.queue[i]
                    return True
        return False

    def clear(self):
        """Clear all failed events."""
        with self._lock:
            self.queue.clear()


class EventMetrics:
    """Tracks event system metrics."""

    def __init__(self):
        self.events_published = 0
        self.events_processed = 0
        self.events_failed = 0
        self.processing_times: deque = deque(maxlen=1000)
        self.event_type_counts: Dict[EventType, int] = defaultdict(int)
        self.handler_performance: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def record_event_published(self, event: Event):
        """Record event published."""
        with self._lock:
            self.events_published += 1
            self.event_type_counts[event.event_type] += 1

    def record_event_processed(self, event: Event, handler_name: str):
        """Record event processed."""
        with self._lock:
            self.events_processed += 1

            duration = event.get_processing_duration()
            if duration is not None:
                self.processing_times.append(duration)
                self.handler_performance[handler_name].append(duration)

    def record_event_failed(self, event: Event):
        """Record event failed."""
        with self._lock:
            self.events_failed += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        with self._lock:
            avg_processing_time = (
                sum(self.processing_times) / len(self.processing_times)
                if self.processing_times
                else 0
            )

            return {
                "events_published": self.events_published,
                "events_processed": self.events_processed,
                "events_failed": self.events_failed,
                "success_rate": (self.events_processed / max(self.events_published, 1)),
                "average_processing_time": avg_processing_time,
                "event_type_distribution": dict(self.event_type_counts),
                "handler_performance": {
                    name: {
                        "count": len(times),
                        "avg_time": sum(times) / len(times) if times else 0,
                        "max_time": max(times) if times else 0,
                    }
                    for name, times in self.handler_performance.items()
                },
            }


class EventSubscriber:
    """
    Event subscriber for handling specific event types.

    Subscribers can register handlers for specific event types and filters,
    supporting both sync and async handlers.
    """

    def __init__(self, subscriber_id: str):
        self.subscriber_id = subscriber_id
        self.handlers: Dict[EventType, List[Tuple[EventHandler, Optional[EventFilter]]]] = (
            defaultdict(list)
        )
        self.active = True
        self._executor = ThreadPoolExecutor(max_workers=4)

    def subscribe(
        self,
        event_type: EventType,
        handler: EventHandler,
        event_filter: Optional[EventFilter] = None,
    ):
        """Subscribe to an event type with optional filter."""
        self.handlers[event_type].append((handler, event_filter))
        logger.info(f"Subscriber {self.subscriber_id} subscribed to {event_type.value}")

    def unsubscribe(self, event_type: EventType, handler: EventHandler):
        """Unsubscribe from an event type."""
        self.handlers[event_type] = [(h, f) for h, f in self.handlers[event_type] if h != handler]

        if not self.handlers[event_type]:
            del self.handlers[event_type]

        logger.info(f"Subscriber {self.subscriber_id} unsubscribed from {event_type.value}")

    async def handle_event(self, event: Event) -> bool:
        """Handle an event if subscribed."""
        if not self.active or event.event_type not in self.handlers:
            return False

        handlers_called = 0

        for handler, event_filter in self.handlers[event.event_type]:
            # Apply filter if provided
            if event_filter and not event_filter(event):
                continue

            try:
                # Check if handler is async
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    # Run sync handler in thread pool
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(self._executor, handler, event)

                handlers_called += 1

            except Exception as e:
                logger.error(f"Handler failed for event {event.id}: {e}")

        return handlers_called > 0

    def get_subscriptions(self) -> Dict[str, int]:
        """Get current subscriptions."""
        return {event_type.value: len(handlers) for event_type, handlers in self.handlers.items()}

    def deactivate(self):
        """Deactivate subscriber."""
        self.active = False
        self._executor.shutdown(wait=True)


class EventPublisher:
    """
    Event publisher for publishing events to the event bus.

    Provides methods for publishing individual events or batches,
    with support for priority queuing and throttling.
    """

    def __init__(self, publisher_id: str, event_bus: "EventBus"):
        self.publisher_id = publisher_id
        self.event_bus = event_bus
        self.published_count = 0

    async def publish(self, event: Event):
        """Publish a single event."""
        event.metadata.source_component = self.publisher_id
        await self.event_bus.publish_event(event)
        self.published_count += 1

    async def publish_batch(self, events: List[Event]):
        """Publish multiple events."""
        for event in events:
            event.metadata.source_component = self.publisher_id

        await self.event_bus.publish_events(events)
        self.published_count += len(events)

    async def publish_knowledge_change(
        self, action: str, node_id: str, node_data: Dict[str, Any], **kwargs
    ):
        """Publish knowledge change event."""
        event = KnowledgeChangeEvent(action, node_id, node_data, **kwargs)
        await self.publish(event)

    async def publish_relationship_change(
        self, action: str, relationship_id: str, relationship_data: Dict[str, Any], **kwargs
    ):
        """Publish relationship change event."""
        event = RelationshipChangeEvent(action, relationship_id, relationship_data, **kwargs)
        await self.publish(event)

    async def publish_query_event(
        self,
        query_type: str,
        query: str,
        results: Optional[Any] = None,
        error: Optional[str] = None,
        **kwargs,
    ):
        """Publish query event."""
        event = QueryEvent(query_type, query, results, error, **kwargs)
        await self.publish(event)

    async def publish_system_event(
        self, system_event_type: str, component: str, details: Dict[str, Any], **kwargs
    ):
        """Publish system event."""
        event = SystemEvent(system_event_type, component, details, **kwargs)
        await self.publish(event)

    async def publish_custom_event(self, custom_type: str, custom_data: Dict[str, Any], **kwargs):
        """Publish custom event."""
        event = CustomEvent(custom_type, custom_data, **kwargs)
        await self.publish(event)

    def get_stats(self) -> Dict[str, Any]:
        """Get publisher statistics."""
        return {"publisher_id": self.publisher_id, "published_count": self.published_count}


class EventBus:
    """
    Central event bus for routing and managing events.

    The EventBus is the core component that manages event distribution,
    subscriber registration, persistence, and monitoring.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.subscribers: Dict[str, EventSubscriber] = {}
        self.event_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.config.get("max_queue_size", 10000)
        )
        self.dead_letter_queue = DeadLetterQueue(
            max_size=self.config.get("dead_letter_max_size", 1000)
        )
        self.metrics = EventMetrics()
        self.persistence = None
        self.batch_processor = None
        self.throttler = None

        # Initialize optional components
        if self.config.get("enable_persistence", True):
            storage_path = self.config.get("storage_path")
            self.persistence = EventPersistence(Path(storage_path) if storage_path else None)

        if self.config.get("enable_batching", True):
            self.batch_processor = EventBatchProcessor(
                batch_size=self.config.get("batch_size", 10),
                flush_interval=self.config.get("flush_interval", 1.0),
            )

            if self.persistence:
                self.batch_processor.add_handler(self._persist_event_batch)

        if self.config.get("enable_throttling", False):
            self.throttler = EventThrottler(
                max_events_per_second=self.config.get("max_events_per_second", 100.0)
            )

        # Processing control
        self.running = False
        self._processing_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

    def register_subscriber(self, subscriber: EventSubscriber):
        """Register an event subscriber."""
        self.subscribers[subscriber.subscriber_id] = subscriber
        logger.info(f"Registered subscriber: {subscriber.subscriber_id}")

    def unregister_subscriber(self, subscriber_id: str):
        """Unregister an event subscriber."""
        if subscriber_id in self.subscribers:
            self.subscribers[subscriber_id].deactivate()
            del self.subscribers[subscriber_id]
            logger.info(f"Unregistered subscriber: {subscriber_id}")

    def create_publisher(self, publisher_id: str) -> EventPublisher:
        """Create a new event publisher."""
        return EventPublisher(publisher_id, self)

    async def publish_event(self, event: Event):
        """Publish a single event."""
        self.metrics.record_event_published(event)

        if self.throttler:
            await self.throttler.throttle()

        try:
            await self.event_queue.put(event)
        except asyncio.QueueFull:
            logger.error(f"Event queue full, dropping event {event.id}")
            self.metrics.record_event_failed(event)

    async def publish_events(self, events: List[Event]):
        """Publish multiple events."""
        for event in events:
            await self.publish_event(event)

    async def start_processing(self):
        """Start event processing."""
        if self.running:
            return

        self.running = True
        self._shutdown_event.clear()
        self._processing_task = asyncio.create_task(self._process_events())
        logger.info("Event bus started")

    async def stop_processing(self):
        """Stop event processing."""
        if not self.running:
            return

        self.running = False
        self._shutdown_event.set()

        if self._processing_task:
            await self._processing_task
            self._processing_task = None

        # Flush any remaining batches
        if self.batch_processor:
            await self.batch_processor.flush()

        logger.info("Event bus stopped")

    async def _process_events(self):
        """Main event processing loop."""
        while self.running:
            try:
                # Wait for event or shutdown
                event_task = asyncio.create_task(self.event_queue.get())
                shutdown_task = asyncio.create_task(self._shutdown_event.wait())

                done, pending = await asyncio.wait(
                    [event_task, shutdown_task], return_when=asyncio.FIRST_COMPLETED
                )

                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

                # Check if shutdown was requested
                if shutdown_task in done:
                    break

                # Process the event
                if event_task in done:
                    event = event_task.result()
                    await self._handle_event(event)

            except Exception as e:
                logger.error(f"Error in event processing loop: {e}")
                await asyncio.sleep(0.1)  # Brief pause before continuing

    async def _handle_event(self, event: Event):
        """Handle a single event."""
        event.mark_processing_started()

        try:
            # Send to batch processor if enabled
            if self.batch_processor:
                await self.batch_processor.add_event(event)

            # Send to persistence if enabled and not using batching
            elif self.persistence:
                self.persistence.persist_event(event)

            # Route to subscribers
            handlers_called = await self._route_event_to_subscribers(event)

            if handlers_called > 0:
                event.mark_processing_completed()

                # Record metrics for each handler (simplified)
                for subscriber in self.subscribers.values():
                    if event.event_type in subscriber.handlers:
                        self.metrics.record_event_processed(event, subscriber.subscriber_id)
            else:
                logger.debug(f"No handlers for event {event.id} of type {event.event_type.value}")
                event.mark_processing_completed()

        except Exception as e:
            event.mark_processing_failed(str(e))
            self.metrics.record_event_failed(event)

            # Retry logic
            if event.should_retry():
                event.increment_retry()
                await self.event_queue.put(event)
                logger.info(f"Retrying event {event.id} (attempt {event.metadata.retry_count})")
            else:
                self.dead_letter_queue.add_failed_event(event)

    async def _route_event_to_subscribers(self, event: Event) -> int:
        """Route event to appropriate subscribers."""
        handlers_called = 0

        for subscriber in self.subscribers.values():
            try:
                if await subscriber.handle_event(event):
                    handlers_called += 1
            except Exception as e:
                logger.error(f"Subscriber {subscriber.subscriber_id} failed: {e}")

        return handlers_called

    async def _persist_event_batch(self, events: List[Event]):
        """Persist batch of events."""
        if self.persistence:
            self.persistence.persist_events(events)

    async def replay_events(
        self, from_timestamp: Optional[float] = None, event_filter: Optional[EventFilter] = None
    ) -> int:
        """Replay events from persistence."""
        if not self.persistence:
            logger.warning("Persistence not enabled, cannot replay events")
            return 0

        events = self.persistence.load_events(from_timestamp)

        if event_filter:
            events = [e for e in events if event_filter(e)]

        replayed_count = 0
        for event in events:
            # Reset event status for replay
            event.status = EventStatus.PENDING
            event.error_message = None
            event.processing_started = None
            event.processing_completed = None

            await self.publish_event(event)
            replayed_count += 1

        logger.info(f"Replayed {replayed_count} events")
        return replayed_count

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics."""
        metrics = self.metrics.get_metrics()
        metrics.update(
            {
                "queue_size": self.event_queue.qsize(),
                "dead_letter_queue_size": len(self.dead_letter_queue.queue),
                "active_subscribers": len(self.subscribers),
                "running": self.running,
            }
        )
        return metrics

    def get_subscriber_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all subscribers."""
        return {
            subscriber_id: {
                "active": subscriber.active,
                "subscriptions": subscriber.get_subscriptions(),
            }
            for subscriber_id, subscriber in self.subscribers.items()
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on event system."""
        return {
            "status": "healthy" if self.running else "stopped",
            "queue_size": self.event_queue.qsize(),
            "subscribers": len(self.subscribers),
            "metrics": self.get_metrics(),
            "persistence_enabled": self.persistence is not None,
            "batching_enabled": self.batch_processor is not None,
            "throttling_enabled": self.throttler is not None,
        }


class EventSystem:
    """
    Main event system orchestrator.

    This is the primary interface for the Memory Engine event system,
    providing high-level methods for event management and system control.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.event_bus = EventBus(self.config.get("event_bus", {}))
        self.publishers: Dict[str, EventPublisher] = {}
        self.subscribers: Dict[str, EventSubscriber] = {}
        self._system_publisher: Optional[EventPublisher] = None

    async def initialize(self):
        """Initialize the event system."""
        await self.event_bus.start_processing()

        # Create system publisher for internal events
        self._system_publisher = self.create_publisher("event_system")

        logger.info("Event system initialized")

    async def shutdown(self):
        """Shutdown the event system."""
        await self.event_bus.stop_processing()

        # Deactivate all subscribers
        for subscriber in self.subscribers.values():
            subscriber.deactivate()

        logger.info("Event system shutdown")

    def create_publisher(self, publisher_id: str) -> EventPublisher:
        """Create and register a new publisher."""
        publisher = self.event_bus.create_publisher(publisher_id)
        self.publishers[publisher_id] = publisher
        return publisher

    def create_subscriber(self, subscriber_id: str) -> EventSubscriber:
        """Create and register a new subscriber."""
        subscriber = EventSubscriber(subscriber_id)
        self.event_bus.register_subscriber(subscriber)
        self.subscribers[subscriber_id] = subscriber
        return subscriber

    def remove_publisher(self, publisher_id: str):
        """Remove a publisher."""
        self.publishers.pop(publisher_id, None)

    def remove_subscriber(self, subscriber_id: str):
        """Remove a subscriber."""
        self.event_bus.unregister_subscriber(subscriber_id)
        self.subscribers.pop(subscriber_id, None)

    async def publish_system_event(self, event_type: str, component: str, details: Dict[str, Any]):
        """Publish system event."""
        if self._system_publisher:
            await self._system_publisher.publish_system_event(event_type, component, details)

    async def replay_events(self, from_timestamp: Optional[float] = None) -> int:
        """Replay events from the specified timestamp."""
        return await self.event_bus.replay_events(from_timestamp)

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        return {
            "event_bus": self.event_bus.get_metrics(),
            "publishers": {pub_id: pub.get_stats() for pub_id, pub in self.publishers.items()},
            "subscribers": self.event_bus.get_subscriber_info(),
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        return await self.event_bus.health_check()

    @asynccontextmanager
    async def event_context(self, correlation_id: Optional[str] = None):
        """Context manager for event operations with correlation ID."""
        if not correlation_id:
            correlation_id = str(uuid.uuid4())

        # Set correlation ID for this context
        original_correlation_id = CorrelationIdManager.get_correlation_id()
        CorrelationIdManager.set_correlation_id(correlation_id)

        try:
            yield correlation_id
        finally:
            # Restore original correlation ID
            if original_correlation_id:
                CorrelationIdManager.set_correlation_id(original_correlation_id)
            else:
                CorrelationIdManager.clear_context()
