"""
Comprehensive tests for the monitoring and observability system.

Tests all monitoring components including health checks, performance monitoring,
Prometheus metrics collection, structured logging, and distributed tracing.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from memory_core.monitoring.health_checks import HealthChecker, HealthStatus
from memory_core.monitoring.performance_monitor import (
    PerformanceMonitor, QueryMetrics, IngestionMetrics, MetricsAggregator
)
from memory_core.monitoring.metrics_collector import (
    PrometheusMetricsCollector, MetricsIntegration
)
from memory_core.monitoring.structured_logger import (
    StructuredLogger, CorrelationIdManager, LoggingContext, OperationLogger
)
from memory_core.monitoring.distributed_tracing import (
    MemoryEngineTracer, TracingIntegration, initialize_tracing
)


class TestHealthChecker:
    """Test health check system."""
    
    @pytest.fixture
    def health_checker(self):
        return HealthChecker()
    
    def test_initialization(self, health_checker):
        """Test health checker initialization."""
        assert health_checker is not None
        assert hasattr(health_checker, 'check_system_health')
    
    @pytest.mark.asyncio
    async def test_system_health_check(self, health_checker):
        """Test overall system health check."""
        with patch.object(health_checker, '_check_janusgraph_health') as mock_janus, \
             patch.object(health_checker, '_check_milvus_health') as mock_milvus, \
             patch.object(health_checker, '_check_gemini_api_health') as mock_gemini:
            
            # Mock successful health checks
            mock_janus.return_value = Mock(status=HealthStatus.HEALTHY, message="OK")
            mock_milvus.return_value = Mock(status=HealthStatus.HEALTHY, message="OK") 
            mock_gemini.return_value = Mock(status=HealthStatus.HEALTHY, message="OK")
            
            health_report = await health_checker.check_system_health()
            
            assert health_report is not None
            assert health_report.overall_status == HealthStatus.HEALTHY
            assert len(health_report.component_health) >= 3
    
    def test_system_resource_checks(self, health_checker):
        """Test system resource health checks."""
        cpu_health = health_checker._check_cpu_health()
        memory_health = health_checker._check_memory_health()
        disk_health = health_checker._check_disk_health()
        
        assert cpu_health.status in [HealthStatus.HEALTHY, HealthStatus.WARNING, HealthStatus.UNHEALTHY]
        assert memory_health.status in [HealthStatus.HEALTHY, HealthStatus.WARNING, HealthStatus.UNHEALTHY]
        assert disk_health.status in [HealthStatus.HEALTHY, HealthStatus.WARNING, HealthStatus.UNHEALTHY]


class TestPerformanceMonitor:
    """Test performance monitoring system."""
    
    @pytest.fixture
    def performance_monitor(self):
        return PerformanceMonitor(monitoring_interval=0.1)  # Fast interval for testing
    
    def test_initialization(self, performance_monitor):
        """Test performance monitor initialization."""
        assert performance_monitor is not None
        assert not performance_monitor.running
        assert performance_monitor.alert_thresholds is not None
    
    def test_query_performance_recording(self, performance_monitor):
        """Test recording query performance metrics."""
        performance_monitor.record_query_performance(
            query_id="test_query_1",
            query_type="semantic_search",
            execution_time_ms=150.5,
            result_count=25,
            cache_hit=True
        )
        
        # Check that metric was recorded
        stats = performance_monitor.aggregator.get_query_statistics()
        assert stats['total_queries'] == 1
        assert stats['average_execution_time_ms'] == 150.5
        assert stats['cache_hit_rate'] == 1.0
    
    def test_ingestion_performance_recording(self, performance_monitor):
        """Test recording ingestion performance metrics."""
        performance_monitor.record_ingestion_performance(
            operation_id="ingest_1",
            operation_type="document",
            items_processed=100,
            processing_time_ms=2000.0,
            error_count=2
        )
        
        stats = performance_monitor.aggregator.get_ingestion_statistics()
        assert stats['total_operations'] == 1
        assert stats['total_items_processed'] == 100
        assert stats['total_errors'] == 2
    
    def test_alert_triggering(self, performance_monitor):
        """Test performance alert triggering."""
        alerts_triggered = []
        
        def alert_callback(alert):
            alerts_triggered.append(alert)
        
        performance_monitor.add_alert_callback(alert_callback)
        
        # Record a slow query that should trigger an alert
        performance_monitor.record_query_performance(
            query_id="slow_query",
            query_type="complex_search",
            execution_time_ms=10000.0,  # 10 seconds - should trigger alert
            result_count=5
        )
        
        # Give some time for alert processing
        time.sleep(0.1)
        
        assert len(alerts_triggered) > 0
        assert alerts_triggered[0].metric_type == "query_performance"
    
    def test_monitoring_lifecycle(self, performance_monitor):
        """Test starting and stopping monitoring."""
        assert not performance_monitor.running
        
        performance_monitor.start_monitoring()
        assert performance_monitor.running
        assert performance_monitor.monitoring_thread is not None
        
        time.sleep(0.2)  # Let it run briefly
        
        performance_monitor.stop_monitoring()
        assert not performance_monitor.running


class TestMetricsAggregator:
    """Test metrics aggregation functionality."""
    
    @pytest.fixture
    def aggregator(self):
        return MetricsAggregator(window_size_minutes=1)
    
    def test_query_metrics_aggregation(self, aggregator):
        """Test aggregation of query metrics."""
        # Add multiple query metrics
        for i in range(5):
            metric = QueryMetrics(
                query_id=f"query_{i}",
                query_type="search",
                execution_time_ms=100 + i * 10,
                result_count=10 + i,
                cache_hit=i % 2 == 0,
                timestamp=time.time()
            )
            aggregator.add_query_metric(metric)
        
        stats = aggregator.get_query_statistics()
        assert stats['total_queries'] == 5
        assert stats['cache_hit_rate'] == 0.6  # 3 out of 5
        assert 'average_execution_time_ms' in stats
        assert 'by_query_type' in stats
    
    def test_metrics_cleanup(self, aggregator):
        """Test automatic cleanup of old metrics."""
        # Add old metric
        old_metric = QueryMetrics(
            query_id="old_query",
            query_type="search",
            execution_time_ms=100,
            result_count=10,
            cache_hit=True,
            timestamp=time.time() - 3600  # 1 hour ago
        )
        aggregator.add_query_metric(old_metric)
        
        # Add recent metric
        recent_metric = QueryMetrics(
            query_id="recent_query",
            query_type="search",
            execution_time_ms=200,
            result_count=20,
            cache_hit=False,
            timestamp=time.time()
        )
        aggregator.add_query_metric(recent_metric)
        
        stats = aggregator.get_query_statistics()
        # Should only contain recent metric after cleanup
        assert stats['total_queries'] == 1


class TestPrometheusMetricsCollector:
    """Test Prometheus metrics collection."""
    
    @pytest.fixture
    def performance_monitor(self):
        return PerformanceMonitor()
    
    @pytest.fixture
    def metrics_collector(self, performance_monitor):
        return PrometheusMetricsCollector(performance_monitor)
    
    def test_initialization(self, metrics_collector):
        """Test Prometheus collector initialization."""
        assert metrics_collector is not None
        assert metrics_collector.registry is not None
        assert hasattr(metrics_collector, 'query_duration_histogram')
    
    def test_query_metric_recording(self, metrics_collector):
        """Test direct query metric recording."""
        metrics_collector.record_query_metric(
            query_type="search",
            execution_time_ms=150.0,
            result_count=25,
            cache_hit=True
        )
        
        # Verify metrics were recorded
        summary = metrics_collector.get_metrics_summary()
        assert summary is not None
        assert 'timestamp' in summary
    
    def test_prometheus_text_output(self, metrics_collector):
        """Test Prometheus text format generation."""
        # Record some metrics
        metrics_collector.record_query_metric("search", 100.0, 10, True)
        metrics_collector.record_ingestion_metric("document", 500.0, 50, 1)
        
        prometheus_text = metrics_collector.get_metrics_text()
        assert isinstance(prometheus_text, str)
        assert 'memory_engine_' in prometheus_text
        assert 'HELP' in prometheus_text
        assert 'TYPE' in prometheus_text


class TestMetricsIntegration:
    """Test integrated metrics system."""
    
    @pytest.fixture
    def metrics_integration(self):
        return MetricsIntegration(monitoring_interval=0.1, metrics_update_interval=0.1)
    
    def test_initialization(self, metrics_integration):
        """Test metrics integration initialization."""
        assert metrics_integration.performance_monitor is not None
        assert metrics_integration.prometheus_collector is not None
    
    def test_unified_query_recording(self, metrics_integration):
        """Test recording metrics in both systems."""
        metrics_integration.record_query_performance(
            query_id="unified_test",
            query_type="semantic",
            execution_time_ms=200.0,
            result_count=15,
            cache_hit=False
        )
        
        # Check both systems recorded the metric
        perf_summary = metrics_integration.get_performance_summary()
        metrics_summary = metrics_integration.get_metrics_summary()
        
        assert perf_summary['query_performance']['total_queries'] == 1
        assert metrics_summary is not None


class TestStructuredLogger:
    """Test structured logging system."""
    
    @pytest.fixture
    def structured_logger(self):
        return StructuredLogger("test_logger", "test_component")
    
    def test_initialization(self, structured_logger):
        """Test structured logger initialization."""
        assert structured_logger.name == "test_logger"
        assert structured_logger.component == "test_component"
        assert structured_logger.logger is not None
    
    def test_correlation_id_context(self):
        """Test correlation ID context management."""
        test_correlation_id = "test-correlation-123"
        
        with LoggingContext(correlation_id=test_correlation_id):
            current_id = CorrelationIdManager.get_correlation_id()
            assert current_id == test_correlation_id
        
        # Should be cleared after context
        assert CorrelationIdManager.get_correlation_id() != test_correlation_id
    
    def test_operation_logging(self, structured_logger):
        """Test operation logging with timing."""
        with OperationLogger(structured_logger, "test_operation") as op_logger:
            time.sleep(0.01)  # Small delay
            assert op_logger.start_time is not None
    
    def test_logging_with_context(self, structured_logger):
        """Test logging with additional context."""
        contextual_logger = structured_logger.with_context(
            user_id="user123",
            session_id="session456"
        )
        
        # Should be able to log with context
        contextual_logger.info("Test message with context")


class TestCorrelationIdManager:
    """Test correlation ID management."""
    
    def test_id_generation(self):
        """Test ID generation."""
        correlation_id = CorrelationIdManager.generate_correlation_id()
        request_id = CorrelationIdManager.generate_request_id()
        
        assert correlation_id is not None
        assert request_id is not None
        assert correlation_id != request_id
    
    def test_context_management(self):
        """Test context variable management."""
        test_correlation_id = "test-123"
        test_request_id = "request-456"
        test_user_id = "user-789"
        
        CorrelationIdManager.set_correlation_id(test_correlation_id)
        CorrelationIdManager.set_request_id(test_request_id)
        CorrelationIdManager.set_user_id(test_user_id)
        
        assert CorrelationIdManager.get_correlation_id() == test_correlation_id
        assert CorrelationIdManager.get_request_id() == test_request_id
        assert CorrelationIdManager.get_user_id() == test_user_id
        
        CorrelationIdManager.clear_context()
        
        assert CorrelationIdManager.get_correlation_id() is None
        assert CorrelationIdManager.get_request_id() is None
        assert CorrelationIdManager.get_user_id() is None


class TestDistributedTracing:
    """Test distributed tracing system."""
    
    @pytest.fixture
    def tracer(self):
        return MemoryEngineTracer("test-service", enable_console_export=True)
    
    def test_tracer_initialization(self, tracer):
        """Test tracer initialization."""
        assert tracer.service_name == "test-service"
        assert tracer.tracer is not None
    
    def test_span_creation(self, tracer):
        """Test creating and managing spans."""
        with tracer.trace_operation("test_operation", component="test") as span:
            assert span is not None
            span.set_attribute("test_attribute", "test_value")
    
    def test_query_operation_tracing(self, tracer):
        """Test query operation tracing."""
        with tracer.trace_query_operation("semantic_search", "query_123") as span:
            assert span is not None
            # Span should have proper attributes
    
    def test_storage_operation_tracing(self, tracer):
        """Test storage operation tracing."""
        with tracer.trace_storage_operation("read", "janusgraph") as span:
            assert span is not None
    
    def test_api_call_tracing(self, tracer):
        """Test API call tracing."""
        with tracer.trace_api_call("gemini", "https://api.example.com/endpoint") as span:
            assert span is not None


class TestTracingIntegration:
    """Test tracing integration with components."""
    
    @pytest.fixture
    def tracing_integration(self):
        return TracingIntegration("test-service", enable_console_export=True)
    
    def test_initialization(self, tracing_integration):
        """Test tracing integration initialization."""
        assert tracing_integration.tracer is not None
        assert tracing_integration.decorators is not None
    
    def test_component_instrumentation(self, tracing_integration):
        """Test component instrumentation."""
        # Mock components
        mock_query_engine = Mock()
        mock_query_engine.execute_query = Mock(return_value=[1, 2, 3])
        
        mock_storage = Mock()
        mock_storage.store = Mock(return_value=True)
        
        mock_embedding_manager = Mock()
        mock_embedding_manager.generate_embedding = Mock(return_value=[0.1, 0.2, 0.3])
        
        # Instrument components
        tracing_integration.instrument_query_engine(mock_query_engine)
        tracing_integration.instrument_storage_layer(mock_storage)
        tracing_integration.instrument_embedding_manager(mock_embedding_manager)
        
        # Test that methods are wrapped
        assert mock_query_engine.execute_query != Mock.execute_query
        assert mock_storage.store != Mock.store
        assert mock_embedding_manager.generate_embedding != Mock.generate_embedding


@pytest.mark.integration
class TestMonitoringSystemIntegration:
    """Integration tests for the complete monitoring system."""
    
    def test_end_to_end_monitoring(self):
        """Test complete monitoring workflow."""
        # Initialize all monitoring components
        metrics_integration = MetricsIntegration(monitoring_interval=0.1)
        tracer = MemoryEngineTracer("integration-test", enable_console_export=True)
        logger = StructuredLogger("integration-test", "test")
        
        # Start monitoring
        metrics_integration.start_monitoring()
        
        try:
            # Simulate operations with tracing and logging
            with LoggingContext() as log_ctx:
                correlation_id = log_ctx.correlation_id
                
                with tracer.trace_operation("integration_test", 
                                           correlation_id=correlation_id) as span:
                    
                    # Record various metrics
                    metrics_integration.record_query_performance(
                        query_id="integration_query",
                        query_type="test_search",
                        execution_time_ms=150.0,
                        result_count=10,
                        cache_hit=True
                    )
                    
                    metrics_integration.record_ingestion_performance(
                        operation_id="integration_ingest",
                        operation_type="test_document",
                        items_processed=50,
                        processing_time_ms=500.0
                    )
                    
                    logger.info("Integration test completed successfully",
                              correlation_id=correlation_id,
                              test_type="end_to_end")
            
            # Wait for metrics to be processed
            time.sleep(0.2)
            
            # Verify metrics were recorded
            perf_summary = metrics_integration.get_performance_summary()
            assert perf_summary['query_performance']['total_queries'] == 1
            assert perf_summary['ingestion_performance']['total_operations'] == 1
            
            # Verify Prometheus metrics
            prometheus_text = metrics_integration.get_prometheus_metrics()
            assert 'memory_engine_' in prometheus_text
            
        finally:
            metrics_integration.stop_monitoring()
    
    def test_health_and_performance_correlation(self):
        """Test correlation between health checks and performance metrics."""
        health_checker = HealthChecker()
        performance_monitor = PerformanceMonitor()
        
        # Simulate some load
        for i in range(10):
            performance_monitor.record_query_performance(
                query_id=f"load_test_{i}",
                query_type="load_test",
                execution_time_ms=100 + i * 10,
                result_count=10
            )
        
        # Get performance summary
        perf_summary = performance_monitor.get_performance_summary()
        assert perf_summary['query_performance']['total_queries'] == 10
        
        # Performance metrics should influence health assessment
        # (This would be more comprehensive in real implementation)
        assert perf_summary is not None