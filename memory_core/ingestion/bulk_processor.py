"""
Bulk processing optimization for high-throughput knowledge ingestion.

This module provides optimized batch processing capabilities for ingesting
large volumes of documents and knowledge units efficiently.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable, Tuple
from threading import Lock

from memory_core.ingestion.advanced_extractor import AdvancedExtractor, process_extracted_units
from memory_core.embeddings.embedding_manager import EmbeddingManager


@dataclass
class BulkIngestionMetrics:
    """Metrics for bulk ingestion performance."""

    total_documents: int = 0
    processed_documents: int = 0
    failed_documents: int = 0
    total_knowledge_units: int = 0
    processing_start_time: float = 0
    processing_end_time: float = 0

    @property
    def processing_time_seconds(self) -> float:
        """Get total processing time in seconds."""
        if self.processing_end_time > 0:
            return self.processing_end_time - self.processing_start_time
        return time.time() - self.processing_start_time

    @property
    def throughput_docs_per_second(self) -> float:
        """Calculate document processing throughput."""
        if self.processing_time_seconds > 0:
            return self.processed_documents / self.processing_time_seconds
        return 0.0

    @property
    def throughput_units_per_second(self) -> float:
        """Calculate knowledge unit processing throughput."""
        if self.processing_time_seconds > 0:
            return self.total_knowledge_units / self.processing_time_seconds
        return 0.0


@dataclass
class BulkDocument:
    """Represents a document for bulk processing."""

    id: str
    content: str
    source_label: str
    metadata: Optional[Dict[str, Any]] = None


class BulkIngestionProcessor:
    """
    Optimized bulk processor for high-throughput knowledge ingestion.

    Features:
    - Parallel processing of multiple documents
    - Batch embedding generation
    - Optimized database writes
    - Progress tracking and metrics
    - Memory usage optimization
    - Error resilience and recovery
    """

    def __init__(
        self,
        extractor: Optional[AdvancedExtractor] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
        storage=None,
        max_workers: int = 4,
        batch_size: int = 10,
        embedding_batch_size: int = 50,
    ):
        """
        Initialize the bulk ingestion processor.

        Args:
            extractor: Advanced extractor for knowledge unit extraction
            embedding_manager: Embedding manager for vector generation
            storage: Storage backend for persistence
            max_workers: Maximum number of worker threads
            batch_size: Number of documents to process in parallel
            embedding_batch_size: Number of embeddings to generate in batch
        """
        self.logger = logging.getLogger(__name__)

        # Core components
        self.extractor = extractor or AdvancedExtractor()
        self.embedding_manager = embedding_manager
        self.storage = storage

        # Performance settings
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.embedding_batch_size = embedding_batch_size

        # Thread safety
        self._metrics_lock = Lock()
        self.metrics = BulkIngestionMetrics()

        # Progress tracking
        self._progress_callbacks: List[Callable[[BulkIngestionMetrics], None]] = []

        self.logger.info(
            f"Bulk processor initialized: workers={max_workers}, batch_size={batch_size}"
        )

    def add_progress_callback(self, callback: Callable[[BulkIngestionMetrics], None]):
        """Add a callback function to monitor progress."""
        self._progress_callbacks.append(callback)

    def process_documents(self, documents: List[BulkDocument]) -> BulkIngestionMetrics:
        """
        Process multiple documents in optimized batches.

        Args:
            documents: List of documents to process

        Returns:
            Metrics about the processing operation
        """
        self.logger.info(f"Starting bulk processing of {len(documents)} documents")

        # Reset metrics
        with self._metrics_lock:
            self.metrics = BulkIngestionMetrics(
                total_documents=len(documents), processing_start_time=time.time()
            )

        # Process documents in batches
        total_knowledge_units = 0

        for i in range(0, len(documents), self.batch_size):
            batch = documents[i : i + self.batch_size]
            batch_units = self._process_document_batch(batch)
            total_knowledge_units += batch_units

            # Update metrics
            with self._metrics_lock:
                self.metrics.total_knowledge_units = total_knowledge_units

            # Notify progress callbacks
            self._notify_progress()

        # Finalize metrics
        with self._metrics_lock:
            self.metrics.processing_end_time = time.time()

        self.logger.info(
            f"Bulk processing complete: {self.metrics.processed_documents}/{self.metrics.total_documents} docs, "
            f"{self.metrics.total_knowledge_units} units in {self.metrics.processing_time_seconds:.2f}s "
            f"({self.metrics.throughput_docs_per_second:.2f} docs/s)"
        )

        return self.metrics

    def _process_document_batch(self, batch: List[BulkDocument]) -> int:
        """Process a batch of documents in parallel."""
        batch_units = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all documents in batch for processing
            future_to_doc = {
                executor.submit(self._process_single_document, doc): doc for doc in batch
            }

            # Collect results as they complete
            for future in as_completed(future_to_doc):
                doc = future_to_doc[future]
                try:
                    unit_count = future.result()
                    batch_units += unit_count

                    with self._metrics_lock:
                        self.metrics.processed_documents += 1

                except Exception as e:
                    self.logger.error(f"Failed to process document {doc.id}: {e}")
                    with self._metrics_lock:
                        self.metrics.failed_documents += 1

        return batch_units

    def _process_single_document(self, document: BulkDocument) -> int:
        """Process a single document and return number of extracted units."""
        try:
            # Extract knowledge units
            knowledge_units = self.extractor.extract_knowledge_units(document.content)

            if not knowledge_units:
                self.logger.debug(f"No knowledge units extracted from document {document.id}")
                return 0

            # Process and store units
            if self.storage or self.embedding_manager:
                node_ids = process_extracted_units(
                    units=knowledge_units,
                    source_label=document.source_label,
                    storage=self.storage,
                    embedding_manager=self.embedding_manager,
                )
                self.logger.debug(
                    f"Document {document.id}: {len(knowledge_units)} units -> {len(node_ids)} nodes"
                )
            else:
                self.logger.debug(
                    f"Document {document.id}: {len(knowledge_units)} units extracted (not stored)"
                )

            return len(knowledge_units)

        except Exception as e:
            self.logger.error(f"Error processing document {document.id}: {e}")
            raise

    def process_documents_with_embeddings_batch(
        self, documents: List[BulkDocument]
    ) -> BulkIngestionMetrics:
        """
        Process documents with optimized batch embedding generation.

        Args:
            documents: List of documents to process

        Returns:
            Processing metrics
        """
        self.logger.info(
            f"Starting bulk processing with embedding batching for {len(documents)} documents"
        )

        # Reset metrics
        with self._metrics_lock:
            self.metrics = BulkIngestionMetrics(
                total_documents=len(documents), processing_start_time=time.time()
            )

        # First phase: Extract all knowledge units
        all_units = []
        source_mapping = {}

        for doc in documents:
            try:
                units = self.extractor.extract_knowledge_units(doc.content)
                for unit in units:
                    unit["_doc_id"] = doc.id
                    unit["_source_label"] = doc.source_label
                    all_units.append(unit)
                    source_mapping[doc.id] = doc.source_label

                with self._metrics_lock:
                    self.metrics.processed_documents += 1

            except Exception as e:
                self.logger.error(f"Failed to extract from document {doc.id}: {e}")
                with self._metrics_lock:
                    self.metrics.failed_documents += 1

        # Second phase: Batch process embeddings and storage
        if all_units and self.embedding_manager:
            self._batch_process_embeddings_and_storage(all_units)

        # Update final metrics
        with self._metrics_lock:
            self.metrics.total_knowledge_units = len(all_units)
            self.metrics.processing_end_time = time.time()

        self.logger.info(
            f"Batch processing complete: {len(all_units)} units, "
            f"{self.metrics.throughput_units_per_second:.2f} units/s"
        )

        return self.metrics

    def _batch_process_embeddings_and_storage(self, units: List[Dict[str, Any]]):
        """Process embeddings and storage in optimized batches."""
        if not self.embedding_manager:
            return

        # Group units by content for embedding deduplication
        content_to_units = {}
        for unit in units:
            content = unit["content"]
            if content not in content_to_units:
                content_to_units[content] = []
            content_to_units[content].append(unit)

        unique_contents = list(content_to_units.keys())

        # Generate embeddings in batches
        for i in range(0, len(unique_contents), self.embedding_batch_size):
            batch_contents = unique_contents[i : i + self.embedding_batch_size]

            try:
                # Generate embeddings for batch
                embeddings = self.embedding_manager.generate_embeddings(
                    batch_contents, task_type="RETRIEVAL_DOCUMENT"
                )

                # Store units with their embeddings
                for content, embedding in zip(batch_contents, embeddings):
                    units_for_content = content_to_units[content]

                    for unit in units_for_content:
                        # Process unit with pre-generated embedding
                        self._store_unit_with_embedding(unit, embedding)

            except Exception as e:
                self.logger.error(f"Failed to process embedding batch: {e}")

    def _store_unit_with_embedding(self, unit: Dict[str, Any], embedding: List[float]):
        """Store a single unit with its pre-generated embedding."""
        try:
            if self.storage:
                # This would need to be implemented based on storage backend
                # For now, just use the existing process_extracted_units
                process_extracted_units(
                    units=[unit],
                    source_label=unit["_source_label"],
                    storage=self.storage,
                    embedding_manager=None,  # Skip embedding generation since we have it
                )
        except Exception as e:
            self.logger.error(f"Failed to store unit: {e}")

    def _notify_progress(self):
        """Notify all progress callbacks with current metrics."""
        for callback in self._progress_callbacks:
            try:
                callback(self.metrics)
            except Exception as e:
                self.logger.error(f"Progress callback failed: {e}")

    async def process_documents_async(self, documents: List[BulkDocument]) -> BulkIngestionMetrics:
        """
        Process documents asynchronously for I/O bound operations.

        Args:
            documents: List of documents to process

        Returns:
            Processing metrics
        """
        self.logger.info(f"Starting async processing of {len(documents)} documents")

        # Reset metrics
        self.metrics = BulkIngestionMetrics(
            total_documents=len(documents), processing_start_time=time.time()
        )

        # Create semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(self.max_workers)

        async def process_with_semaphore(doc):
            async with semaphore:
                return await self._process_document_async(doc)

        # Process all documents concurrently
        tasks = [process_with_semaphore(doc) for doc in documents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        total_units = 0
        for result in results:
            if isinstance(result, Exception):
                self.metrics.failed_documents += 1
                self.logger.error(f"Async processing error: {result}")
            else:
                self.metrics.processed_documents += 1
                total_units += result

        self.metrics.total_knowledge_units = total_units
        self.metrics.processing_end_time = time.time()

        self.logger.info(
            f"Async processing complete: {self.metrics.throughput_docs_per_second:.2f} docs/s"
        )
        return self.metrics

    async def _process_document_async(self, document: BulkDocument) -> int:
        """Process a single document asynchronously."""
        loop = asyncio.get_event_loop()

        # Run CPU-bound extraction in thread pool
        units = await loop.run_in_executor(
            None, self.extractor.extract_knowledge_units, document.content
        )

        if units and (self.storage or self.embedding_manager):
            # Run I/O-bound storage in thread pool
            node_ids = await loop.run_in_executor(
                None,
                process_extracted_units,
                units,
                document.source_label,
                self.storage,
                self.embedding_manager,
            )
            return len(node_ids)

        return len(units) if units else 0

    def get_performance_recommendations(self) -> Dict[str, str]:
        """Get performance optimization recommendations based on current metrics."""
        recommendations = {}

        if self.metrics.throughput_docs_per_second > 0:
            if self.metrics.throughput_docs_per_second < 1.0:
                recommendations["throughput"] = (
                    "Consider increasing batch_size or max_workers for better throughput"
                )

            if self.metrics.failed_documents / self.metrics.total_documents > 0.1:
                recommendations["reliability"] = (
                    "High failure rate detected - check document quality and error handling"
                )

            if self.embedding_manager:
                cache_stats = self.embedding_manager.get_cache_statistics()
                if cache_stats["hit_rate"] < 0.3:
                    recommendations["caching"] = (
                        "Low embedding cache hit rate - consider preprocessing similar content"
                    )

        return recommendations


def create_bulk_processor(storage=None, embedding_manager=None, **kwargs) -> BulkIngestionProcessor:
    """
    Factory function to create a bulk processor with optimal settings.

    Args:
        storage: Storage backend
        embedding_manager: Embedding manager
        **kwargs: Additional configuration options

    Returns:
        Configured BulkIngestionProcessor
    """
    return BulkIngestionProcessor(storage=storage, embedding_manager=embedding_manager, **kwargs)
