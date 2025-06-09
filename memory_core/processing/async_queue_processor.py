"""
Asynchronous processing queue system for high-throughput operations.

This module provides queue-based processing for CPU-intensive and I/O-bound
operations, enabling scalable background processing for knowledge ingestion,
relationship extraction, and other performance-critical tasks.
"""

import asyncio
import logging
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Type, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from threading import Lock
import json
import uuid


class TaskStatus(Enum):
    """Task processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class QueueTask:
    """Represents a task in the processing queue."""
    id: str
    task_type: str
    data: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: Optional[float] = None
    
    @property
    def processing_time_seconds(self) -> float:
        """Get task processing time in seconds."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        elif self.started_at:
            return time.time() - self.started_at
        return 0.0
    
    @property
    def is_expired(self) -> bool:
        """Check if task has exceeded timeout."""
        if not self.timeout_seconds or not self.started_at:
            return False
        return time.time() - self.started_at > self.timeout_seconds


@dataclass
class QueueMetrics:
    """Metrics for queue performance monitoring."""
    total_tasks: int = 0
    pending_tasks: int = 0
    processing_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    average_processing_time: float = 0.0
    throughput_tasks_per_second: float = 0.0
    queue_start_time: float = field(default_factory=time.time)
    
    def update_throughput(self):
        """Update throughput calculation."""
        elapsed = time.time() - self.queue_start_time
        if elapsed > 0:
            self.throughput_tasks_per_second = self.completed_tasks / elapsed


class TaskProcessor(ABC):
    """Abstract base class for task processors."""
    
    @abstractmethod
    async def process(self, task: QueueTask) -> Any:
        """
        Process a task and return the result.
        
        Args:
            task: Task to process
            
        Returns:
            Processing result
            
        Raises:
            Exception: If processing fails
        """
        pass
    
    @property
    @abstractmethod
    def supported_task_types(self) -> List[str]:
        """Return list of supported task types."""
        pass


class EmbeddingGenerationProcessor(TaskProcessor):
    """Processor for embedding generation tasks."""
    
    def __init__(self, embedding_manager):
        self.embedding_manager = embedding_manager
        self.logger = logging.getLogger(__name__)
    
    async def process(self, task: QueueTask) -> Any:
        """Generate embeddings for text content."""
        try:
            texts = task.data.get('texts', [])
            task_type = task.data.get('task_type', 'SEMANTIC_SIMILARITY')
            
            if not texts:
                raise ValueError("No texts provided for embedding generation")
            
            # Run embedding generation in thread pool
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, 
                self.embedding_manager.generate_embeddings,
                texts, 
                task_type
            )
            
            return {
                'embeddings': embeddings,
                'count': len(embeddings)
            }
            
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            raise
    
    @property
    def supported_task_types(self) -> List[str]:
        return ['embedding_generation', 'batch_embedding']


class KnowledgeExtractionProcessor(TaskProcessor):
    """Processor for knowledge unit extraction tasks."""
    
    def __init__(self, extractor):
        self.extractor = extractor
        self.logger = logging.getLogger(__name__)
    
    async def process(self, task: QueueTask) -> Any:
        """Extract knowledge units from text."""
        try:
            content = task.data.get('content', '')
            source_label = task.data.get('source_label', 'unknown')
            
            if not content:
                raise ValueError("No content provided for knowledge extraction")
            
            # Run extraction in thread pool
            loop = asyncio.get_event_loop()
            knowledge_units = await loop.run_in_executor(
                None,
                self.extractor.extract_knowledge_units,
                content
            )
            
            return {
                'knowledge_units': knowledge_units,
                'count': len(knowledge_units),
                'source_label': source_label
            }
            
        except Exception as e:
            self.logger.error(f"Knowledge extraction failed: {e}")
            raise
    
    @property
    def supported_task_types(self) -> List[str]:
        return ['knowledge_extraction', 'document_processing']


class RelationshipExtractionProcessor(TaskProcessor):
    """Processor for relationship extraction tasks."""
    
    def __init__(self, relationship_extractor):
        self.relationship_extractor = relationship_extractor
        self.logger = logging.getLogger(__name__)
    
    async def process(self, task: QueueTask) -> Any:
        """Extract relationships between nodes."""
        try:
            node_ids = task.data.get('node_ids', [])
            strategies = task.data.get('strategies', ['tags', 'content_similarity'])
            min_confidence = task.data.get('min_confidence', 0.5)
            
            if not node_ids:
                raise ValueError("No node IDs provided for relationship extraction")
            
            # Run relationship extraction
            from memory_core.ingestion.relationship_extractor import extract_relationships_async
            metrics = await extract_relationships_async(
                node_ids=node_ids,
                storage=self.relationship_extractor.storage,
                strategies=strategies,
                max_concurrent=10
            )
            
            return {
                'metrics': metrics,
                'relationships_created': metrics.relationships_created
            }
            
        except Exception as e:
            self.logger.error(f"Relationship extraction failed: {e}")
            raise
    
    @property
    def supported_task_types(self) -> List[str]:
        return ['relationship_extraction', 'batch_relationship_analysis']


class AsyncProcessingQueue:
    """
    Asynchronous processing queue for scalable background operations.
    
    Features:
    - Priority-based task scheduling
    - Configurable worker pools
    - Task retry mechanisms
    - Performance monitoring
    - Error handling and recovery
    - Graceful shutdown
    """
    
    def __init__(self, 
                 max_workers: int = 4,
                 max_queue_size: int = 1000,
                 task_timeout: float = 300.0):
        """
        Initialize the async processing queue.
        
        Args:
            max_workers: Maximum number of concurrent workers
            max_queue_size: Maximum queue size
            task_timeout: Default task timeout in seconds
        """
        self.logger = logging.getLogger(__name__)
        
        # Queue configuration
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.task_timeout = task_timeout
        
        # Task storage
        self.pending_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self.active_tasks: Dict[str, QueueTask] = {}
        self.completed_tasks: Dict[str, QueueTask] = {}
        
        # Task processors
        self.processors: Dict[str, TaskProcessor] = {}
        
        # Worker management
        self.workers: List[asyncio.Task] = []
        self.running = False
        self.shutdown_event = asyncio.Event()
        
        # Metrics and monitoring
        self.metrics = QueueMetrics()
        self._metrics_lock = Lock()
        
        # Thread and process pools
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=max(1, max_workers // 2))
        
        self.logger.info(f"Async processing queue initialized: workers={max_workers}")
    
    def register_processor(self, processor: TaskProcessor):
        """Register a task processor for specific task types."""
        for task_type in processor.supported_task_types:
            self.processors[task_type] = processor
            self.logger.info(f"Registered processor for task type: {task_type}")
    
    async def start(self):
        """Start the processing queue and worker tasks."""
        if self.running:
            return
        
        self.running = True
        self.shutdown_event.clear()
        
        # Start worker tasks
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        # Start metrics updater
        metrics_task = asyncio.create_task(self._update_metrics())
        self.workers.append(metrics_task)
        
        self.logger.info(f"Started {len(self.workers)} workers")
    
    async def shutdown(self, timeout: float = 30.0):
        """Gracefully shutdown the processing queue."""
        if not self.running:
            return
        
        self.logger.info("Shutting down async processing queue...")
        
        self.running = False
        self.shutdown_event.set()
        
        # Wait for workers to complete with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*self.workers, return_exceptions=True),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            self.logger.warning("Worker shutdown timeout, cancelling remaining tasks")
            for worker in self.workers:
                worker.cancel()
        
        # Shutdown thread and process pools
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        
        self.logger.info("Async processing queue shutdown complete")
    
    async def submit_task(self, 
                         task_type: str, 
                         data: Dict[str, Any],
                         priority: TaskPriority = TaskPriority.NORMAL,
                         timeout: Optional[float] = None) -> str:
        """
        Submit a task to the processing queue.
        
        Args:
            task_type: Type of task to process
            data: Task data
            priority: Task priority
            timeout: Task timeout in seconds
            
        Returns:
            Task ID
            
        Raises:
            ValueError: If task type is not supported
            asyncio.QueueFull: If queue is full
        """
        if task_type not in self.processors:
            raise ValueError(f"Unsupported task type: {task_type}")
        
        task = QueueTask(
            id=str(uuid.uuid4()),
            task_type=task_type,
            data=data,
            priority=priority,
            timeout_seconds=timeout or self.task_timeout
        )
        
        # Add to queue with priority (lower number = higher priority)
        priority_value = 5 - priority.value  # Invert for priority queue
        await self.pending_queue.put((priority_value, time.time(), task))
        
        with self._metrics_lock:
            self.metrics.total_tasks += 1
            self.metrics.pending_tasks += 1
        
        self.logger.debug(f"Submitted task {task.id} of type {task_type}")
        return task.id
    
    async def get_task_status(self, task_id: str) -> Optional[QueueTask]:
        """Get the current status of a task."""
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]
        elif task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        return None
    
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> QueueTask:
        """
        Wait for a task to complete.
        
        Args:
            task_id: Task ID to wait for
            timeout: Wait timeout in seconds
            
        Returns:
            Completed task
            
        Raises:
            asyncio.TimeoutError: If timeout is exceeded
            ValueError: If task not found
        """
        start_time = time.time()
        
        while True:
            task = await self.get_task_status(task_id)
            
            if not task:
                raise ValueError(f"Task {task_id} not found")
            
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                return task
            
            if timeout and time.time() - start_time > timeout:
                raise asyncio.TimeoutError(f"Task {task_id} wait timeout")
            
            await asyncio.sleep(0.1)
    
    async def _worker(self, worker_name: str):
        """Worker coroutine for processing tasks."""
        self.logger.info(f"Worker {worker_name} started")
        
        while self.running:
            try:
                # Get task from queue with timeout
                try:
                    priority, queued_time, task = await asyncio.wait_for(
                        self.pending_queue.get(), 
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Move task to active
                self.active_tasks[task.id] = task
                task.status = TaskStatus.PROCESSING
                task.started_at = time.time()
                
                with self._metrics_lock:
                    self.metrics.pending_tasks -= 1
                    self.metrics.processing_tasks += 1
                
                self.logger.debug(f"Worker {worker_name} processing task {task.id}")
                
                try:
                    # Check if task is expired
                    if task.is_expired:
                        raise asyncio.TimeoutError("Task expired before processing")
                    
                    # Get processor for task type
                    processor = self.processors.get(task.task_type)
                    if not processor:
                        raise ValueError(f"No processor for task type: {task.task_type}")
                    
                    # Process task with timeout
                    task.result = await asyncio.wait_for(
                        processor.process(task),
                        timeout=task.timeout_seconds
                    )
                    
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = time.time()
                    
                    self.logger.debug(
                        f"Task {task.id} completed in {task.processing_time_seconds:.2f}s"
                    )
                    
                except Exception as e:
                    task.error = str(e)
                    task.retry_count += 1
                    
                    if task.retry_count <= task.max_retries:
                        # Retry task
                        task.status = TaskStatus.PENDING
                        task.started_at = None
                        
                        # Re-queue with lower priority
                        retry_priority = min(priority + 1, 4)
                        await self.pending_queue.put((retry_priority, time.time(), task))
                        
                        self.logger.warning(
                            f"Task {task.id} failed, retry {task.retry_count}/{task.max_retries}: {e}"
                        )
                        continue
                    else:
                        # Max retries exceeded
                        task.status = TaskStatus.FAILED
                        task.completed_at = time.time()
                        
                        self.logger.error(f"Task {task.id} failed permanently: {e}")
                
                # Move to completed
                del self.active_tasks[task.id]
                self.completed_tasks[task.id] = task
                
                # Update metrics
                with self._metrics_lock:
                    self.metrics.processing_tasks -= 1
                    if task.status == TaskStatus.COMPLETED:
                        self.metrics.completed_tasks += 1
                    else:
                        self.metrics.failed_tasks += 1
                
                # Clean up old completed tasks
                await self._cleanup_completed_tasks()
                
            except Exception as e:
                self.logger.error(f"Worker {worker_name} error: {e}")
                await asyncio.sleep(1.0)
        
        self.logger.info(f"Worker {worker_name} stopped")
    
    async def _update_metrics(self):
        """Periodically update queue metrics."""
        while self.running:
            try:
                with self._metrics_lock:
                    # Update throughput
                    self.metrics.update_throughput()
                    
                    # Update average processing time
                    if self.completed_tasks:
                        total_time = sum(
                            task.processing_time_seconds 
                            for task in self.completed_tasks.values()
                            if task.status == TaskStatus.COMPLETED
                        )
                        self.metrics.average_processing_time = (
                            total_time / self.metrics.completed_tasks 
                            if self.metrics.completed_tasks > 0 else 0.0
                        )
                
                await asyncio.sleep(5.0)  # Update every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Metrics update error: {e}")
                await asyncio.sleep(1.0)
    
    async def _cleanup_completed_tasks(self, max_completed: int = 1000):
        """Clean up old completed tasks to prevent memory buildup."""
        if len(self.completed_tasks) > max_completed:
            # Remove oldest completed tasks
            sorted_tasks = sorted(
                self.completed_tasks.items(),
                key=lambda x: x[1].completed_at or 0
            )
            
            tasks_to_remove = len(self.completed_tasks) - max_completed
            for task_id, _ in sorted_tasks[:tasks_to_remove]:
                del self.completed_tasks[task_id]
    
    def get_queue_metrics(self) -> QueueMetrics:
        """Get current queue performance metrics."""
        with self._metrics_lock:
            return QueueMetrics(
                total_tasks=self.metrics.total_tasks,
                pending_tasks=len(self.pending_queue._queue) if hasattr(self.pending_queue, '_queue') else self.metrics.pending_tasks,
                processing_tasks=len(self.active_tasks),
                completed_tasks=self.metrics.completed_tasks,
                failed_tasks=self.metrics.failed_tasks,
                average_processing_time=self.metrics.average_processing_time,
                throughput_tasks_per_second=self.metrics.throughput_tasks_per_second,
                queue_start_time=self.metrics.queue_start_time
            )
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get detailed queue status information."""
        metrics = self.get_queue_metrics()
        
        return {
            'running': self.running,
            'workers': len(self.workers),
            'registered_processors': list(self.processors.keys()),
            'metrics': {
                'total_tasks': metrics.total_tasks,
                'pending_tasks': metrics.pending_tasks,
                'processing_tasks': metrics.processing_tasks,
                'completed_tasks': metrics.completed_tasks,
                'failed_tasks': metrics.failed_tasks,
                'average_processing_time': metrics.average_processing_time,
                'throughput_tasks_per_second': metrics.throughput_tasks_per_second
            },
            'active_task_ids': list(self.active_tasks.keys()),
            'queue_capacity': self.max_queue_size
        }


# Factory function for creating pre-configured queues
async def create_knowledge_processing_queue(
    storage=None, 
    embedding_manager=None, 
    extractor=None,
    relationship_extractor=None,
    **kwargs
) -> AsyncProcessingQueue:
    """
    Create a pre-configured processing queue for knowledge operations.
    
    Args:
        storage: Storage backend
        embedding_manager: Embedding manager
        extractor: Knowledge extractor
        relationship_extractor: Relationship extractor
        **kwargs: Additional queue configuration
        
    Returns:
        Configured AsyncProcessingQueue
    """
    queue = AsyncProcessingQueue(**kwargs)
    
    # Register processors
    if embedding_manager:
        queue.register_processor(EmbeddingGenerationProcessor(embedding_manager))
    
    if extractor:
        queue.register_processor(KnowledgeExtractionProcessor(extractor))
    
    if relationship_extractor:
        queue.register_processor(RelationshipExtractionProcessor(relationship_extractor))
    
    await queue.start()
    return queue