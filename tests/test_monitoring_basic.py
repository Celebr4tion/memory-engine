"""
Basic tests for monitoring components.
"""

import pytest
import time
from unittest.mock import Mock, patch

from memory_core.monitoring.health_checks import HealthChecker, HealthStatus
from memory_core.monitoring.performance_monitor import (
    PerformanceMonitor,
    QueryMetrics,
    IngestionMetrics,
    MetricsAggregator,
)


class TestHealthChecker:
    """Test health check system."""

    def test_initialization(self):
        """Test health checker initialization."""
        health_checker = HealthChecker()
        assert health_checker is not None
        assert hasattr(health_checker, "check_system_health")

    @pytest.mark.asyncio
    async def test_system_resource_checks(self):
        """Test system resource health checks."""
        health_checker = HealthChecker()
        system_health = await health_checker._check_system_resources()
        memory_health = await health_checker._check_memory_usage()
        disk_health = await health_checker._check_disk_space()

        assert system_health.status in [
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
            HealthStatus.UNHEALTHY,
        ]
        assert memory_health.status in [
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
            HealthStatus.UNHEALTHY,
        ]
        assert disk_health.status in [
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
            HealthStatus.UNHEALTHY,
        ]


class TestPerformanceMonitor:
    """Test performance monitoring system."""

    def test_initialization(self):
        """Test performance monitor initialization."""
        performance_monitor = PerformanceMonitor(monitoring_interval=0.1)
        assert performance_monitor is not None
        assert not performance_monitor.running
        assert performance_monitor.alert_thresholds is not None

    def test_query_performance_recording(self):
        """Test recording query performance metrics."""
        performance_monitor = PerformanceMonitor(monitoring_interval=0.1)
        performance_monitor.record_query_performance(
            query_id="test_query_1",
            query_type="semantic_search",
            execution_time_ms=150.5,
            result_count=25,
            cache_hit=True,
        )

        # Check that metric was recorded
        stats = performance_monitor.aggregator.get_query_statistics()
        assert stats["total_queries"] == 1
        assert stats["average_execution_time_ms"] == 150.5
        assert stats["cache_hit_rate"] == 1.0

    def test_ingestion_performance_recording(self):
        """Test recording ingestion performance metrics."""
        performance_monitor = PerformanceMonitor(monitoring_interval=0.1)
        performance_monitor.record_ingestion_performance(
            operation_id="ingest_1",
            operation_type="document",
            items_processed=100,
            processing_time_ms=2000.0,
            error_count=2,
        )

        stats = performance_monitor.aggregator.get_ingestion_statistics()
        assert stats["total_operations"] == 1
        assert stats["total_items_processed"] == 100
        assert stats["total_errors"] == 2

    def test_alert_triggering(self):
        """Test performance alert triggering."""
        performance_monitor = PerformanceMonitor(monitoring_interval=0.1)
        alerts_triggered = []

        def alert_callback(alert):
            alerts_triggered.append(alert)

        performance_monitor.add_alert_callback(alert_callback)

        # Record a slow query that should trigger an alert
        performance_monitor.record_query_performance(
            query_id="slow_query",
            query_type="complex_search",
            execution_time_ms=10000.0,  # 10 seconds - should trigger alert
            result_count=5,
        )

        # Give some time for alert processing
        time.sleep(0.1)

        assert len(alerts_triggered) > 0
        assert alerts_triggered[0].metric_type == "query_performance"


class TestMetricsAggregator:
    """Test metrics aggregation functionality."""

    def test_query_metrics_aggregation(self):
        """Test aggregation of query metrics."""
        aggregator = MetricsAggregator(window_size_minutes=1)

        # Add multiple query metrics
        for i in range(5):
            metric = QueryMetrics(
                query_id=f"query_{i}",
                query_type="search",
                execution_time_ms=100 + i * 10,
                result_count=10 + i,
                cache_hit=i % 2 == 0,
                timestamp=time.time(),
            )
            aggregator.add_query_metric(metric)

        stats = aggregator.get_query_statistics()
        assert stats["total_queries"] == 5
        assert stats["cache_hit_rate"] == 0.6  # 3 out of 5
        assert "average_execution_time_ms" in stats
        assert "by_query_type" in stats

    def test_metrics_cleanup(self):
        """Test automatic cleanup of old metrics."""
        aggregator = MetricsAggregator(window_size_minutes=1)

        # Add old metric
        old_metric = QueryMetrics(
            query_id="old_query",
            query_type="search",
            execution_time_ms=100,
            result_count=10,
            cache_hit=True,
            timestamp=time.time() - 3600,  # 1 hour ago
        )
        aggregator.add_query_metric(old_metric)

        # Add recent metric
        recent_metric = QueryMetrics(
            query_id="recent_query",
            query_type="search",
            execution_time_ms=200,
            result_count=20,
            cache_hit=False,
            timestamp=time.time(),
        )
        aggregator.add_query_metric(recent_metric)

        stats = aggregator.get_query_statistics()
        # Should only contain recent metric after cleanup
        assert stats["total_queries"] == 1
