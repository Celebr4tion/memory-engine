"""
Batch operation optimizer for high-performance bulk processing.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

logger = logging.getLogger(__name__)


class BatchStrategy(Enum):
    """Batch processing strategies."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"
    MEMORY_OPTIMIZED = "memory_optimized"


@dataclass
class BatchConfig:
    """Configuration for batch processing."""

    batch_size: int = 1000
    max_batch_size: int = 10000
    min_batch_size: int = 100
    max_concurrent_batches: int = 4
    strategy: BatchStrategy = BatchStrategy.ADAPTIVE
    memory_limit_mb: int = 512
    timeout_seconds: float = 300.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    enable_progress_tracking: bool = True


@dataclass
class BatchMetrics:
    """Metrics for batch processing."""

    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0
    total_batches: int = 0
    completed_batches: int = 0
    failed_batches: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    avg_batch_time: float = 0.0
    throughput_items_per_second: float = 0.0
    memory_usage_mb: float = 0.0

    @property
    def elapsed_time(self) -> float:
        """Get elapsed processing time."""
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def progress_percentage(self) -> float:
        """Get progress percentage."""
        return (self.processed_items / self.total_items * 100) if self.total_items > 0 else 0.0


class BatchProcessor:
    """Base class for batch processors."""

    def __init__(self, config: BatchConfig):
        self.config = config
        self.metrics = BatchMetrics()
        self._executor = ThreadPoolExecutor(max_workers=config.max_concurrent_batches)
        self._stop_requested = False

    async def process_batch(self, items: List[Any], batch_id: int) -> List[Any]:
        """Process a single batch of items."""
        raise NotImplementedError("Subclasses must implement process_batch")

    async def process_all(
        self, items: List[Any], progress_callback: Optional[Callable[[BatchMetrics], None]] = None
    ) -> List[Any]:
        """Process all items in optimized batches."""
        self.metrics = BatchMetrics()
        self.metrics.total_items = len(items)
        self.metrics.start_time = time.time()

        if not items:
            return []

        # Determine optimal batch size
        batch_size = self._calculate_optimal_batch_size(items)
        batches = self._create_batches(items, batch_size)
        self.metrics.total_batches = len(batches)

        results = []

        try:
            if self.config.strategy == BatchStrategy.SEQUENTIAL:
                results = await self._process_sequential(batches, progress_callback)
            elif self.config.strategy == BatchStrategy.PARALLEL:
                results = await self._process_parallel(batches, progress_callback)
            elif self.config.strategy == BatchStrategy.ADAPTIVE:
                results = await self._process_adaptive(batches, progress_callback)
            elif self.config.strategy == BatchStrategy.MEMORY_OPTIMIZED:
                results = await self._process_memory_optimized(batches, progress_callback)
            else:
                raise ValueError(f"Unknown batch strategy: {self.config.strategy}")

        finally:
            self.metrics.end_time = time.time()
            self._calculate_final_metrics()

        return results

    def _calculate_optimal_batch_size(self, items: List[Any]) -> int:
        """Calculate optimal batch size based on items and system resources."""
        # Base batch size from config
        base_size = self.config.batch_size

        # Adjust based on total items
        if len(items) < 1000:
            # Small datasets - use smaller batches
            base_size = min(base_size, len(items) // 2 + 1)
        elif len(items) > 100000:
            # Large datasets - use larger batches
            base_size = min(self.config.max_batch_size, base_size * 2)

        # Adjust based on memory constraints
        if self.config.memory_limit_mb > 0:
            estimated_item_size = self._estimate_item_memory_size(items[:10] if items else [])
            if estimated_item_size > 0:
                max_items_in_memory = (
                    self.config.memory_limit_mb * 1024 * 1024
                ) // estimated_item_size
                base_size = min(
                    base_size, max_items_in_memory // self.config.max_concurrent_batches
                )

        return max(self.config.min_batch_size, min(base_size, self.config.max_batch_size))

    def _estimate_item_memory_size(self, sample_items: List[Any]) -> int:
        """Estimate memory size per item."""
        if not sample_items:
            return 1024  # Default estimate

        try:
            import sys

            total_size = sum(sys.getsizeof(item) for item in sample_items)
            return total_size // len(sample_items)
        except:
            return 1024  # Fallback estimate

    def _create_batches(self, items: List[Any], batch_size: int) -> List[List[Any]]:
        """Create batches from items."""
        batches = []
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            batches.append(batch)
        return batches

    async def _process_sequential(
        self, batches: List[List[Any]], progress_callback: Optional[Callable[[BatchMetrics], None]]
    ) -> List[Any]:
        """Process batches sequentially."""
        results = []

        for batch_id, batch in enumerate(batches):
            if self._stop_requested:
                break

            try:
                start_time = time.time()
                batch_results = await self.process_batch(batch, batch_id)
                batch_time = time.time() - start_time

                results.extend(batch_results)
                self._update_metrics_after_batch(batch, batch_time, True)

                if progress_callback:
                    progress_callback(self.metrics)

            except Exception as e:
                logger.error(f"Batch {batch_id} failed: {e}")
                self._update_metrics_after_batch(batch, 0, False)

        return results

    async def _process_parallel(
        self, batches: List[List[Any]], progress_callback: Optional[Callable[[BatchMetrics], None]]
    ) -> List[Any]:
        """Process batches in parallel."""
        results = []
        semaphore = asyncio.Semaphore(self.config.max_concurrent_batches)

        async def process_batch_with_semaphore(
            batch: List[Any], batch_id: int
        ) -> Tuple[int, List[Any]]:
            async with semaphore:
                try:
                    start_time = time.time()
                    batch_results = await self.process_batch(batch, batch_id)
                    batch_time = time.time() - start_time

                    self._update_metrics_after_batch(batch, batch_time, True)
                    return batch_id, batch_results
                except Exception as e:
                    logger.error(f"Batch {batch_id} failed: {e}")
                    self._update_metrics_after_batch(batch, 0, False)
                    return batch_id, []

        # Create tasks for all batches
        tasks = [
            process_batch_with_semaphore(batch, batch_id) for batch_id, batch in enumerate(batches)
        ]

        # Process with progress updates
        completed_tasks = []
        for coro in asyncio.as_completed(tasks):
            if self._stop_requested:
                break

            batch_id, batch_results = await coro
            completed_tasks.append((batch_id, batch_results))

            if progress_callback:
                progress_callback(self.metrics)

        # Sort results by batch_id to maintain order
        completed_tasks.sort(key=lambda x: x[0])
        for _, batch_results in completed_tasks:
            results.extend(batch_results)

        return results

    async def _process_adaptive(
        self, batches: List[List[Any]], progress_callback: Optional[Callable[[BatchMetrics], None]]
    ) -> List[Any]:
        """Process batches with adaptive strategy."""
        # Start with small number of parallel batches
        current_parallelism = 2
        batch_times = []
        results = []

        i = 0
        while i < len(batches):
            if self._stop_requested:
                break

            # Process current batch set
            current_batches = batches[i : i + current_parallelism]

            start_time = time.time()
            batch_results = await self._process_parallel_subset(current_batches, i)
            batch_time = time.time() - start_time

            results.extend(batch_results)
            batch_times.append(batch_time)

            # Adapt parallelism based on performance
            if len(batch_times) >= 2:
                recent_times = batch_times[-2:]
                if recent_times[-1] < recent_times[-2] * 0.8:  # Significant improvement
                    current_parallelism = min(
                        current_parallelism + 1, self.config.max_concurrent_batches
                    )
                elif recent_times[-1] > recent_times[-2] * 1.2:  # Performance degraded
                    current_parallelism = max(current_parallelism - 1, 1)

            i += len(current_batches)

            if progress_callback:
                progress_callback(self.metrics)

        return results

    async def _process_memory_optimized(
        self, batches: List[List[Any]], progress_callback: Optional[Callable[[BatchMetrics], None]]
    ) -> List[Any]:
        """Process batches with memory optimization."""
        results = []

        # Process in smaller chunks to manage memory
        chunk_size = max(1, self.config.max_concurrent_batches // 2)

        for i in range(0, len(batches), chunk_size):
            if self._stop_requested:
                break

            chunk_batches = batches[i : i + chunk_size]
            chunk_results = await self._process_parallel_subset(chunk_batches, i)
            results.extend(chunk_results)

            # Force garbage collection after each chunk
            import gc

            gc.collect()

            if progress_callback:
                progress_callback(self.metrics)

        return results

    async def _process_parallel_subset(
        self, batches: List[List[Any]], start_batch_id: int
    ) -> List[Any]:
        """Process a subset of batches in parallel."""
        semaphore = asyncio.Semaphore(len(batches))

        async def process_single_batch(batch: List[Any], batch_id: int) -> Tuple[int, List[Any]]:
            async with semaphore:
                try:
                    start_time = time.time()
                    batch_results = await self.process_batch(batch, batch_id)
                    batch_time = time.time() - start_time

                    self._update_metrics_after_batch(batch, batch_time, True)
                    return batch_id, batch_results
                except Exception as e:
                    logger.error(f"Batch {batch_id} failed: {e}")
                    self._update_metrics_after_batch(batch, 0, False)
                    return batch_id, []

        tasks = [process_single_batch(batch, start_batch_id + i) for i, batch in enumerate(batches)]

        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)

        # Sort and extract results
        valid_results = [task for task in completed_tasks if not isinstance(task, Exception)]
        valid_results.sort(key=lambda x: x[0])

        results = []
        for _, batch_results in valid_results:
            results.extend(batch_results)

        return results

    def _update_metrics_after_batch(self, batch: List[Any], batch_time: float, success: bool):
        """Update metrics after processing a batch."""
        if success:
            self.metrics.processed_items += len(batch)
            self.metrics.completed_batches += 1
        else:
            self.metrics.failed_items += len(batch)
            self.metrics.failed_batches += 1

        if batch_time > 0:
            if self.metrics.avg_batch_time == 0:
                self.metrics.avg_batch_time = batch_time
            else:
                total_completed = self.metrics.completed_batches
                self.metrics.avg_batch_time = (
                    self.metrics.avg_batch_time * (total_completed - 1) + batch_time
                ) / total_completed

    def _calculate_final_metrics(self):
        """Calculate final processing metrics."""
        elapsed_time = self.metrics.elapsed_time
        if elapsed_time > 0:
            self.metrics.throughput_items_per_second = self.metrics.processed_items / elapsed_time

    def stop(self):
        """Request to stop processing."""
        self._stop_requested = True

    def get_metrics(self) -> BatchMetrics:
        """Get current processing metrics."""
        return self.metrics


class KnowledgeIngestionBatchProcessor(BatchProcessor):
    """Specialized batch processor for knowledge ingestion."""

    def __init__(self, config: BatchConfig, knowledge_engine: Any):
        super().__init__(config)
        self.knowledge_engine = knowledge_engine

    async def process_batch(self, items: List[Any], batch_id: int) -> List[Any]:
        """Process a batch of knowledge items."""
        results = []

        try:
            # Group items by type for optimized processing
            text_items = [item for item in items if isinstance(item, str)]
            document_items = [item for item in items if hasattr(item, "content")]

            # Process text items
            if text_items:
                text_results = await self._process_text_batch(text_items)
                results.extend(text_results)

            # Process document items
            if document_items:
                doc_results = await self._process_document_batch(document_items)
                results.extend(doc_results)

            logger.info(f"Batch {batch_id} processed {len(items)} items successfully")

        except Exception as e:
            logger.error(f"Batch {batch_id} processing failed: {e}")
            raise

        return results

    async def _process_text_batch(self, text_items: List[str]) -> List[Any]:
        """Process batch of text items."""
        results = []

        # Batch embedding generation
        if hasattr(self.knowledge_engine, "embedding_manager"):
            embeddings = await self.knowledge_engine.embedding_manager.generate_embeddings(
                text_items
            )
        else:
            embeddings = [None] * len(text_items)

        # Batch knowledge extraction
        for text, embedding in zip(text_items, embeddings):
            try:
                knowledge_units = await self.knowledge_engine.extract_knowledge_units(text)
                results.append(
                    {"text": text, "embedding": embedding, "knowledge_units": knowledge_units}
                )
            except Exception as e:
                logger.error(f"Failed to process text item: {e}")
                results.append({"text": text, "error": str(e)})

        return results

    async def _process_document_batch(self, document_items: List[Any]) -> List[Any]:
        """Process batch of document items."""
        results = []

        for doc in document_items:
            try:
                # Extract text from document
                text = getattr(doc, "content", str(doc))

                # Process as text
                text_results = await self._process_text_batch([text])
                results.extend(text_results)

            except Exception as e:
                logger.error(f"Failed to process document: {e}")
                results.append({"document": doc, "error": str(e)})

        return results


class QueryBatchProcessor(BatchProcessor):
    """Specialized batch processor for query operations."""

    def __init__(self, config: BatchConfig, storage_backend: Any):
        super().__init__(config)
        self.storage_backend = storage_backend

    async def process_batch(
        self, items: List[Tuple[str, Dict[str, Any]]], batch_id: int
    ) -> List[Any]:
        """Process a batch of queries."""
        results = []

        try:
            # Group queries by similarity for optimization
            query_groups = self._group_similar_queries(items)

            for group in query_groups:
                group_results = await self._process_query_group(group)
                results.extend(group_results)

            logger.info(f"Query batch {batch_id} processed {len(items)} queries")

        except Exception as e:
            logger.error(f"Query batch {batch_id} failed: {e}")
            raise

        return results

    def _group_similar_queries(
        self, queries: List[Tuple[str, Dict[str, Any]]]
    ) -> List[List[Tuple[str, Dict[str, Any]]]]:
        """Group similar queries for batch optimization."""
        # Simple grouping by query template
        groups = {}

        for query, params in queries:
            # Extract query template (remove specific values)
            template = self._extract_query_template(query)

            if template not in groups:
                groups[template] = []
            groups[template].append((query, params))

        return list(groups.values())

    def _extract_query_template(self, query: str) -> str:
        """Extract query template for grouping."""
        # Simple template extraction - replace specific values with placeholders
        import re

        # Replace string literals
        template = re.sub(r"'[^']*'", "'?'", query)

        # Replace numbers
        template = re.sub(r"\b\d+\b", "?", template)

        return template

    async def _process_query_group(self, group: List[Tuple[str, Dict[str, Any]]]) -> List[Any]:
        """Process a group of similar queries."""
        results = []

        # For groups with the same template, we could use prepared statements
        for query, params in group:
            try:
                result = await self.storage_backend.execute_query(query, params)
                results.append(result)
            except Exception as e:
                logger.error(f"Query failed: {e}")
                results.append({"error": str(e)})

        return results
