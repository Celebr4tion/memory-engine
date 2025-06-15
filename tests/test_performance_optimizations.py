"""
Tests for performance optimization features.

This module tests the caching, bulk processing, parallel relationship extraction,
async queues, and performance monitoring systems.
"""

import asyncio
import pytest
import time
import threading
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any

from memory_core.embeddings.embedding_manager import EmbeddingManager, EmbeddingCache
from memory_core.ingestion.bulk_processor import BulkIngestionProcessor, BulkDocument
from memory_core.ingestion.relationship_extractor import ParallelRelationshipExtractor
from memory_core.processing.async_queue_processor import (
    AsyncProcessingQueue,
    QueueTask,
    TaskPriority,
)
from memory_core.monitoring.performance_monitor import PerformanceMonitor
from memory_core.testing.performance_regression_tests import PerformanceRegressionTester
from memory_core.db.graph_storage_adapter import GraphStorageAdapter


class TestEmbeddingCache:
    """Test cases for embedding cache functionality."""

    def test_cache_initialization(self):
        """Test cache initialization with default parameters."""
        cache = EmbeddingCache()
        assert cache.max_entries == 1000
        assert cache.ttl_seconds == 3600
        assert cache.hits == 0
        assert cache.misses == 0

    def test_cache_put_and_get(self):
        """Test basic cache put and get operations."""
        cache = EmbeddingCache(max_entries=10, ttl_seconds=60)

        # Test cache miss
        result = cache.get("test text", "SEMANTIC_SIMILARITY")
        assert result is None
        assert cache.misses == 1

        # Add to cache
        embedding = [0.1, 0.2, 0.3]
        cache.put("test text", "SEMANTIC_SIMILARITY", embedding)

        # Test cache hit
        result = cache.get("test text", "SEMANTIC_SIMILARITY")
        assert result == embedding
        assert cache.hits == 1

    def test_cache_expiration(self):
        """Test cache TTL expiration."""
        cache = EmbeddingCache(max_entries=10, ttl_seconds=0.1)  # 100ms TTL

        embedding = [0.1, 0.2, 0.3]
        cache.put("test text", "SEMANTIC_SIMILARITY", embedding)

        # Should hit immediately
        result = cache.get("test text", "SEMANTIC_SIMILARITY")
        assert result == embedding

        # Wait for expiration
        time.sleep(0.2)

        # Should miss after expiration
        result = cache.get("test text", "SEMANTIC_SIMILARITY")
        assert result is None
        assert cache.misses == 1

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = EmbeddingCache(max_entries=2, ttl_seconds=60)

        # Fill cache
        cache.put("text1", "SEMANTIC_SIMILARITY", [0.1])
        cache.put("text2", "SEMANTIC_SIMILARITY", [0.2])

        # Add third item (should evict first)
        cache.put("text3", "SEMANTIC_SIMILARITY", [0.3])

        # First item should be evicted
        assert cache.get("text1", "SEMANTIC_SIMILARITY") is None
        assert cache.get("text2", "SEMANTIC_SIMILARITY") == [0.2]
        assert cache.get("text3", "SEMANTIC_SIMILARITY") == [0.3]


class TestBulkIngestionProcessor:
    """Test cases for bulk ingestion processor."""

    def setup_method(self):
        """Setup test fixtures."""
        self.mock_extractor = Mock()
        self.mock_embedding_manager = Mock()
        self.mock_storage = Mock()

        self.processor = BulkIngestionProcessor(
            extractor=self.mock_extractor,
            embedding_manager=self.mock_embedding_manager,
            storage=self.mock_storage,
            max_workers=2,
            batch_size=5,
        )

    def test_process_single_document(self):
        """Test processing a single document."""
        # Mock extractor response
        self.mock_extractor.extract_knowledge_units.return_value = [
            {"content": "test knowledge", "tags": ["test"]},
            {"content": "more knowledge", "tags": ["more"]},
        ]

        document = BulkDocument(
            id="test_doc", content="Test document content", source_label="test_source"
        )

        with patch(
            "memory_core.ingestion.bulk_processor.process_extracted_units",
            return_value=["node1", "node2"],
        ):
            unit_count = self.processor._process_single_document(document)

        assert unit_count == 2
        self.mock_extractor.extract_knowledge_units.assert_called_once_with("Test document content")

    def test_process_documents_batch(self):
        """Test batch processing of documents."""
        documents = [
            BulkDocument(id=f"doc_{i}", content=f"Content {i}", source_label="test")
            for i in range(3)
        ]

        # Mock successful processing
        self.mock_extractor.extract_knowledge_units.return_value = [{"content": "test"}]

        with patch(
            "memory_core.ingestion.bulk_processor.process_extracted_units", return_value=["node1"]
        ):
            metrics = self.processor.process_documents(documents)

        assert metrics.total_documents == 3
        assert metrics.processed_documents == 3
        assert metrics.failed_documents == 0
        assert metrics.total_knowledge_units == 3

    def test_process_documents_with_failures(self):
        """Test handling of document processing failures."""
        documents = [
            BulkDocument(id="good_doc", content="Good content", source_label="test"),
            BulkDocument(
                id="bad_doc", content="", source_label="test"
            ),  # Empty content should fail
        ]

        def mock_extract(content):
            if not content:
                raise ValueError("Empty content")
            return [{"content": "test"}]

        self.mock_extractor.extract_knowledge_units.side_effect = mock_extract

        with patch(
            "memory_core.ingestion.bulk_processor.process_extracted_units", return_value=["node1"]
        ):
            metrics = self.processor.process_documents(documents)

        assert metrics.total_documents == 2
        assert metrics.processed_documents == 1
        assert metrics.failed_documents == 1


class TestParallelRelationshipExtractor:
    """Test cases for parallel relationship extractor."""

    def setup_method(self):
        """Setup test fixtures."""
        self.mock_storage = Mock()
        self.extractor = ParallelRelationshipExtractor(
            storage=self.mock_storage, max_workers=2, batch_size=5, chunk_size=10
        )

    def test_tag_relationship_detection(self):
        """Test tag-based relationship detection."""
        node1_data = {"tags": "machine learning,AI,algorithms,data science"}
        node2_data = {"tags": "AI,neural networks,deep learning,data science"}

        relationship = self.extractor._detect_tag_relationship(
            "node1", node1_data, "node2", node2_data
        )

        assert relationship is not None
        assert relationship["relation_type"] == "SIMILAR_TAGS"
        assert relationship["from_id"] == "node1"
        assert relationship["to_id"] == "node2"
        assert "AI" in relationship["metadata"]["shared_tags"]
        assert "data science" in relationship["metadata"]["shared_tags"]

    def test_content_similarity_detection(self):
        """Test content-based relationship detection."""
        node1_data = {"content": "Machine learning algorithms are used in artificial intelligence"}
        node2_data = {"content": "Artificial intelligence uses machine learning techniques"}

        relationship = self.extractor._detect_content_similarity_relationship(
            "node1", node1_data, "node2", node2_data
        )

        assert relationship is not None
        assert relationship["relation_type"] == "SIMILAR_CONTENT"
        assert relationship["confidence_score"] > 0.4

    def test_metadata_relationship_detection(self):
        """Test metadata-based relationship detection."""
        node1_data = {"source": "wikipedia", "creation_timestamp": 1000}
        node2_data = {"source": "wikipedia", "creation_timestamp": 1001}

        relationship = self.extractor._detect_metadata_relationship(
            "node1", node1_data, "node2", node2_data
        )

        assert relationship is not None
        assert relationship["relation_type"] == "SAME_SOURCE"
        assert relationship["metadata"]["shared_source"] == "wikipedia"


@pytest.mark.asyncio
class TestAsyncProcessingQueue:
    """Test cases for async processing queue."""

    async def test_queue_initialization_and_startup(self):
        """Test queue initialization and startup."""
        queue = AsyncProcessingQueue(max_workers=2, max_queue_size=10)

        assert queue.max_workers == 2
        assert queue.max_queue_size == 10
        assert not queue.running

        await queue.start()
        assert queue.running
        assert len(queue.workers) == 3  # 2 workers + 1 metrics updater

        await queue.shutdown()

    async def test_task_submission_and_processing(self):
        """Test task submission and processing."""
        queue = AsyncProcessingQueue(max_workers=1, max_queue_size=10)

        # Mock processor with async process method
        mock_processor = Mock()
        mock_processor.supported_task_types = ["test_task"]

        async def mock_process(task):
            return {"result": "success"}

        mock_processor.process = mock_process

        queue.register_processor(mock_processor)
        await queue.start()

        try:
            # Submit task
            task_id = await queue.submit_task(
                task_type="test_task", data={"input": "test"}, priority=TaskPriority.HIGH
            )

            # Give some time for processing
            await asyncio.sleep(0.5)

            # Wait for completion
            task = await queue.wait_for_task(task_id, timeout=5.0)

            assert task.status.value == "completed"
            assert task.result == {"result": "success"}

        finally:
            await queue.shutdown()

    async def test_task_retry_mechanism(self):
        """Test task retry on failure."""
        queue = AsyncProcessingQueue(max_workers=1, max_queue_size=10)

        # Mock processor that fails twice then succeeds
        mock_processor = Mock()
        mock_processor.supported_task_types = ["test_task"]

        call_count = 0

        async def failing_process(task):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Temporary failure")
            return {"result": "success"}

        mock_processor.process = failing_process

        queue.register_processor(mock_processor)
        await queue.start()

        try:
            task_id = await queue.submit_task(task_type="test_task", data={"input": "test"})

            # Give more time for retries
            await asyncio.sleep(2.0)

            task = await queue.wait_for_task(task_id, timeout=10.0)

            assert task.status.value == "completed"
            assert task.retry_count == 2
            assert call_count == 3

        finally:
            await queue.shutdown()


class TestPerformanceMonitor:
    """Test cases for performance monitor."""

    def setup_method(self):
        """Setup test fixtures."""
        self.monitor = PerformanceMonitor(
            alert_thresholds={"cpu_percent": 80.0, "query_avg_time_ms": 1000.0},
            monitoring_interval=0.1,  # Fast interval for testing
        )

    def test_query_performance_recording(self):
        """Test recording query performance metrics."""
        self.monitor.record_query_performance(
            query_id="test_query",
            query_type="semantic_search",
            execution_time_ms=500.0,
            result_count=10,
            cache_hit=True,
        )

        stats = self.monitor.aggregator.get_query_statistics()
        assert stats["total_queries"] == 1
        assert stats["average_execution_time_ms"] == 500.0
        assert stats["cache_hit_rate"] == 1.0

    def test_ingestion_performance_recording(self):
        """Test recording ingestion performance metrics."""
        self.monitor.record_ingestion_performance(
            operation_id="test_op",
            operation_type="document",
            items_processed=100,
            processing_time_ms=2000.0,
            error_count=5,
        )

        stats = self.monitor.aggregator.get_ingestion_statistics()
        assert stats["total_operations"] == 1
        assert stats["total_items_processed"] == 100
        assert stats["total_errors"] == 5

    def test_alert_generation(self):
        """Test performance alert generation."""
        alerts_triggered = []

        def alert_callback(alert):
            alerts_triggered.append(alert)

        self.monitor.add_alert_callback(alert_callback)

        # Record slow query that should trigger alert
        self.monitor.record_query_performance(
            query_id="slow_query",
            query_type="semantic_search",
            execution_time_ms=1500.0,  # Above threshold
            result_count=10,
        )

        # Should have triggered an alert
        assert len(alerts_triggered) == 1
        assert alerts_triggered[0].metric_type == "query_performance"
        assert alerts_triggered[0].severity == "warning"

    def test_resource_monitoring(self):
        """Test resource utilization monitoring."""
        # Start monitoring briefly
        self.monitor.start_monitoring()
        time.sleep(0.2)  # Let it collect some samples
        self.monitor.stop_monitoring()

        stats = self.monitor.aggregator.get_resource_statistics()
        assert stats["sample_count"] > 0
        assert "cpu_utilization" in stats
        assert "memory_utilization" in stats


class TestPerformanceRegressionTester:
    """Test cases for performance regression testing."""

    def setup_method(self):
        """Setup test fixtures."""
        self.tester = PerformanceRegressionTester(
            baseline_storage_path="test_baselines.json", regression_threshold=0.10
        )

    def test_benchmark_registration(self):
        """Test benchmark registration."""

        def test_function():
            time.sleep(0.01)
            return {"result": "test"}

        benchmark = self.tester.create_query_benchmark(
            name="test_benchmark", query_function=test_function, expected_max_time_ms=100.0
        )

        assert "test_benchmark" in self.tester.benchmarks
        assert self.tester.benchmarks["test_benchmark"].category == "query"

    @pytest.mark.asyncio
    async def test_benchmark_execution(self):
        """Test benchmark execution."""

        def fast_function():
            return {"result": "fast"}

        benchmark = self.tester.create_query_benchmark(
            name="fast_test", query_function=fast_function, expected_max_time_ms=100.0
        )
        benchmark.test_runs = 3  # Reduce for faster testing
        benchmark.warmup_runs = 1

        results = await self.tester.run_benchmark_suite(benchmark_names=["fast_test"])

        assert "fast_test" in results
        result = results["fast_test"]
        assert result.success_count == 3
        assert result.error_count == 0
        assert len(result.execution_times_ms) == 3

    def test_regression_detection(self):
        """Test performance regression detection."""
        from memory_core.testing.performance_regression_tests import BenchmarkResult

        # Create baseline result
        baseline = BenchmarkResult(
            benchmark_id="test",
            benchmark_name="test_benchmark",
            category="query",
            timestamp=time.time() - 3600,
            execution_times_ms=[100.0, 105.0, 95.0],
            success_count=3,
        )

        # Create current result (slower)
        current = BenchmarkResult(
            benchmark_id="test",
            benchmark_name="test_benchmark",
            category="query",
            timestamp=time.time(),
            execution_times_ms=[130.0, 125.0, 135.0],  # 30% slower
            success_count=3,
        )

        self.tester.baseline_results["test_benchmark"] = baseline
        self.tester.current_results["test_benchmark"] = current

        regressions = self.tester.detect_regressions()

        assert len(regressions) == 1
        regression = regressions[0]
        assert regression.is_regression
        assert regression.time_regression_percent > 20  # Should detect significant regression


class TestGraphStorageAdapterOptimizations:
    """Test cases for graph storage adapter optimizations."""

    def setup_method(self):
        """Setup test fixtures."""
        self.mock_storage = Mock()
        self.adapter = GraphStorageAdapter(self.mock_storage)

    def test_content_search_caching(self):
        """Test content search result caching."""
        # Mock storage response
        mock_results = [{"id": "node1", "content": "test content"}]
        self.mock_storage.find_nodes_by_content.return_value = mock_results

        # First call should hit storage
        results1 = self.adapter.find_nodes_by_content("test query", limit=10)
        assert results1 == mock_results
        assert self.mock_storage.find_nodes_by_content.call_count == 1

        # Second call should hit cache
        results2 = self.adapter.find_nodes_by_content("test query", limit=10)
        assert results2 == mock_results
        assert self.mock_storage.find_nodes_by_content.call_count == 1  # No additional calls

    def test_relationship_caching(self):
        """Test relationship query result caching."""
        # Mock outgoing relationships
        self.mock_storage.get_outgoing_relationships.return_value = []
        self.mock_storage.get_incoming_relationships.return_value = []

        # First call should hit storage
        relationships1 = self.adapter.get_relationships_for_node("node1", max_depth=1)
        assert isinstance(relationships1, list)

        # Second call should hit cache
        relationships2 = self.adapter.get_relationships_for_node("node1", max_depth=1)
        assert relationships2 == relationships1

    def test_shortest_path_algorithm(self):
        """Test shortest path finding algorithm."""

        # Mock neighbor relationships
        def mock_get_neighbors(node_id):
            if node_id == "node1":
                return ["node2", "node3"]
            elif node_id == "node2":
                return ["node4"]
            elif node_id == "node3":
                return ["node4"]
            return []

        self.adapter._get_neighbor_node_ids = mock_get_neighbors

        # Test direct path
        path = self.adapter.find_shortest_path("node1", "node4", max_hops=3)
        assert len(path) == 3  # node1 -> node2 -> node4 or node1 -> node3 -> node4
        assert path[0] == "node1"
        assert path[-1] == "node4"

    def test_cache_statistics(self):
        """Test cache statistics reporting."""
        # Add some data to caches
        self.adapter._relationship_cache["test"] = []
        self.adapter._content_index_cache["test"] = []

        stats = self.adapter.get_traversal_statistics()
        assert stats["relationship_cache_entries"] == 1
        assert stats["content_cache_entries"] == 1

    def test_cache_clearing(self):
        """Test cache clearing functionality."""
        # Add data to caches
        self.adapter._relationship_cache["test"] = []
        self.adapter._content_index_cache["test"] = []

        # Clear caches
        self.adapter.clear_caches()

        assert len(self.adapter._relationship_cache) == 0
        assert len(self.adapter._content_index_cache) == 0


if __name__ == "__main__":
    pytest.main([__file__])
