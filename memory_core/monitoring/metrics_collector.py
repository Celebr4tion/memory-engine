"""
Prometheus-compatible metrics collection for Memory Engine.

This module provides Prometheus metrics collection that integrates with the existing
performance monitoring system to expose metrics for external monitoring and alerting.
"""

import logging
import time
import threading
from typing import Dict, Any, Optional, List
from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, generate_latest
from prometheus_client.core import REGISTRY

from .performance_monitor import PerformanceMonitor, PerformanceAlert


class PrometheusMetricsCollector:
    """
    Prometheus-compatible metrics collector for Memory Engine.
    
    Integrates with the existing PerformanceMonitor to expose metrics
    in Prometheus format for external monitoring systems.
    """
    
    def __init__(self, performance_monitor: PerformanceMonitor, 
                 registry: Optional[CollectorRegistry] = None):
        """
        Initialize the Prometheus metrics collector.
        
        Args:
            performance_monitor: Existing performance monitor instance
            registry: Prometheus registry (defaults to global registry)
        """
        self.logger = logging.getLogger(__name__)
        self.performance_monitor = performance_monitor
        self.registry = registry or REGISTRY
        
        # Initialize Prometheus metrics
        self._init_metrics()
        
        # Set up performance monitor integration
        self._setup_performance_integration()
        
        # Metrics update state
        self._last_update = 0
        self._update_interval = 5.0  # seconds
        self._lock = threading.RLock()
        
        self.logger.info("Prometheus metrics collector initialized")
    
    def _init_metrics(self):
        """Initialize Prometheus metrics definitions."""
        
        # Query Performance Metrics
        self.query_duration_histogram = Histogram(
            'memory_engine_query_duration_seconds',
            'Query execution time in seconds',
            labelnames=['query_type', 'cache_hit'],
            registry=self.registry
        )
        
        self.query_total_counter = Counter(
            'memory_engine_queries_total',
            'Total number of queries executed',
            labelnames=['query_type', 'status'],
            registry=self.registry
        )
        
        self.query_result_count_histogram = Histogram(
            'memory_engine_query_results',
            'Number of results returned by queries',
            labelnames=['query_type'],
            registry=self.registry
        )
        
        # Ingestion Performance Metrics
        self.ingestion_duration_histogram = Histogram(
            'memory_engine_ingestion_duration_seconds',
            'Ingestion operation time in seconds',
            labelnames=['operation_type'],
            registry=self.registry
        )
        
        self.ingestion_throughput_gauge = Gauge(
            'memory_engine_ingestion_throughput_per_second',
            'Ingestion throughput in items per second',
            labelnames=['operation_type'],
            registry=self.registry
        )
        
        self.ingestion_items_counter = Counter(
            'memory_engine_ingestion_items_total',
            'Total number of items processed during ingestion',
            labelnames=['operation_type', 'status'],
            registry=self.registry
        )
        
        # System Resource Metrics
        self.system_cpu_percent = Gauge(
            'memory_engine_cpu_utilization_percent',
            'CPU utilization percentage',
            registry=self.registry
        )
        
        self.system_memory_percent = Gauge(
            'memory_engine_memory_utilization_percent',
            'Memory utilization percentage',
            registry=self.registry
        )
        
        self.system_memory_used_bytes = Gauge(
            'memory_engine_memory_used_bytes',
            'Memory used in bytes',
            registry=self.registry
        )
        
        self.system_disk_io_bytes = Counter(
            'memory_engine_disk_io_bytes_total',
            'Total disk I/O in bytes',
            labelnames=['direction'],  # 'read' or 'write'
            registry=self.registry
        )
        
        self.system_network_io_bytes = Counter(
            'memory_engine_network_io_bytes_total',
            'Total network I/O in bytes',
            labelnames=['direction'],  # 'sent' or 'received'
            registry=self.registry
        )
        
        # Alert Metrics
        self.alerts_total_counter = Counter(
            'memory_engine_alerts_total',
            'Total number of performance alerts triggered',
            labelnames=['metric_type', 'severity'],
            registry=self.registry
        )
        
        self.active_alerts_gauge = Gauge(
            'memory_engine_active_alerts',
            'Number of currently active alerts',
            labelnames=['severity'],
            registry=self.registry
        )
        
        # Cache Performance Metrics
        self.cache_hit_rate_gauge = Gauge(
            'memory_engine_cache_hit_rate',
            'Query cache hit rate (0-1)',
            registry=self.registry
        )
        
        # Health Status Metrics
        self.system_health_info = Info(
            'memory_engine_system_health',
            'System health status information',
            registry=self.registry
        )
        
        self.monitoring_active_gauge = Gauge(
            'memory_engine_monitoring_active',
            'Whether performance monitoring is active (1=active, 0=inactive)',
            registry=self.registry
        )
    
    def _setup_performance_integration(self):
        """Set up integration with the existing performance monitor."""
        # Register as alert callback to capture alerts
        self.performance_monitor.add_alert_callback(self._handle_alert)
    
    def _handle_alert(self, alert: PerformanceAlert):
        """Handle performance alerts by updating Prometheus metrics."""
        with self._lock:
            # Increment alert counter
            self.alerts_total_counter.labels(
                metric_type=alert.metric_type,
                severity=alert.severity
            ).inc()
            
            self.logger.debug(f"Recorded Prometheus alert metric: {alert.metric_type} ({alert.severity})")
    
    def update_metrics(self, force: bool = False):
        """
        Update Prometheus metrics from performance monitor data.
        
        Args:
            force: Force update regardless of interval
        """
        current_time = time.time()
        
        if not force and (current_time - self._last_update) < self._update_interval:
            return
        
        with self._lock:
            try:
                # Get performance summary
                summary = self.performance_monitor.get_performance_summary()
                
                # Update system-level metrics
                self.monitoring_active_gauge.set(1 if summary.get('monitoring_active', False) else 0)
                
                # Update query metrics
                query_stats = summary.get('query_performance', {})
                if query_stats:
                    self._update_query_metrics(query_stats)
                
                # Update ingestion metrics
                ingestion_stats = summary.get('ingestion_performance', {})
                if ingestion_stats:
                    self._update_ingestion_metrics(ingestion_stats)
                
                # Update resource metrics
                resource_stats = summary.get('resource_utilization', {})
                if resource_stats:
                    self._update_resource_metrics(resource_stats)
                
                # Update alert metrics
                self._update_alert_metrics(summary.get('recent_alerts', []))
                
                self._last_update = current_time
                self.logger.debug("Prometheus metrics updated successfully")
                
            except Exception as e:
                self.logger.error(f"Failed to update Prometheus metrics: {e}")
    
    def _update_query_metrics(self, query_stats: Dict[str, Any]):
        """Update query-related Prometheus metrics."""
        # Cache hit rate
        cache_hit_rate = query_stats.get('cache_hit_rate', 0)
        self.cache_hit_rate_gauge.set(cache_hit_rate)
        
        # Query statistics by type
        by_type = query_stats.get('by_query_type', {})
        for query_type, type_stats in by_type.items():
            # Average execution time (convert ms to seconds)
            avg_time_seconds = type_stats.get('avg_time_ms', 0) / 1000.0
            
            # Update histogram with current average (approximation)
            # In real implementation, we'd track individual query metrics
            self.query_duration_histogram.labels(
                query_type=query_type,
                cache_hit='unknown'
            ).observe(avg_time_seconds)
            
            # Query count
            count = type_stats.get('count', 0)
            self.query_total_counter.labels(
                query_type=query_type,
                status='success'
            )._value._value = count  # Set absolute value
    
    def _update_ingestion_metrics(self, ingestion_stats: Dict[str, Any]):
        """Update ingestion-related Prometheus metrics."""
        # Overall throughput
        avg_throughput = ingestion_stats.get('average_throughput_per_second', 0)
        
        # Statistics by operation type
        by_type = ingestion_stats.get('by_operation_type', {})
        for op_type, type_stats in by_type.items():
            throughput = type_stats.get('avg_throughput', 0)
            self.ingestion_throughput_gauge.labels(operation_type=op_type).set(throughput)
            
            count = type_stats.get('count', 0)
            self.ingestion_items_counter.labels(
                operation_type=op_type,
                status='success'
            )._value._value = count  # Set absolute value
    
    def _update_resource_metrics(self, resource_stats: Dict[str, Any]):
        """Update resource utilization Prometheus metrics."""
        # CPU metrics
        cpu_util = resource_stats.get('cpu_utilization', {})
        if cpu_util:
            avg_cpu = cpu_util.get('average', 0)
            self.system_cpu_percent.set(avg_cpu)
        
        # Memory metrics
        memory_util = resource_stats.get('memory_utilization', {})
        if memory_util:
            avg_memory_percent = memory_util.get('average_percent', 0)
            avg_memory_mb = memory_util.get('average_used_mb', 0)
            
            self.system_memory_percent.set(avg_memory_percent)
            self.system_memory_used_bytes.set(avg_memory_mb * 1024 * 1024)  # Convert MB to bytes
    
    def _update_alert_metrics(self, recent_alerts: List[Dict[str, Any]]):
        """Update alert-related Prometheus metrics."""
        # Count active alerts by severity
        severity_counts = {'warning': 0, 'critical': 0}
        cutoff_time = time.time() - 300  # 5 minutes
        
        for alert in recent_alerts:
            if alert.get('timestamp', 0) > cutoff_time:
                severity = alert.get('severity', 'unknown')
                if severity in severity_counts:
                    severity_counts[severity] += 1
        
        # Update gauges
        for severity, count in severity_counts.items():
            self.active_alerts_gauge.labels(severity=severity).set(count)
    
    def record_query_metric(self, query_type: str, execution_time_ms: float, 
                           result_count: int, cache_hit: bool, error: Optional[str] = None):
        """
        Record a single query metric directly.
        
        Args:
            query_type: Type of query executed
            execution_time_ms: Execution time in milliseconds
            result_count: Number of results returned
            cache_hit: Whether the query hit cache
            error: Error message if query failed
        """
        with self._lock:
            # Execution time
            self.query_duration_histogram.labels(
                query_type=query_type,
                cache_hit=str(cache_hit).lower()
            ).observe(execution_time_ms / 1000.0)
            
            # Query count
            status = 'error' if error else 'success'
            self.query_total_counter.labels(
                query_type=query_type,
                status=status
            ).inc()
            
            # Result count
            if not error:
                self.query_result_count_histogram.labels(
                    query_type=query_type
                ).observe(result_count)
    
    def record_ingestion_metric(self, operation_type: str, processing_time_ms: float,
                               items_processed: int, error_count: int = 0):
        """
        Record a single ingestion metric directly.
        
        Args:
            operation_type: Type of ingestion operation
            processing_time_ms: Processing time in milliseconds
            items_processed: Number of items processed
            error_count: Number of errors encountered
        """
        with self._lock:
            # Processing time
            self.ingestion_duration_histogram.labels(
                operation_type=operation_type
            ).observe(processing_time_ms / 1000.0)
            
            # Throughput
            if processing_time_ms > 0:
                throughput = items_processed / (processing_time_ms / 1000.0)
                self.ingestion_throughput_gauge.labels(
                    operation_type=operation_type
                ).set(throughput)
            
            # Item counts
            success_count = items_processed - error_count
            if success_count > 0:
                self.ingestion_items_counter.labels(
                    operation_type=operation_type,
                    status='success'
                ).inc(success_count)
            
            if error_count > 0:
                self.ingestion_items_counter.labels(
                    operation_type=operation_type,
                    status='error'
                ).inc(error_count)
    
    def get_metrics_text(self) -> str:
        """
        Get metrics in Prometheus text format.
        
        Returns:
            Metrics formatted for Prometheus scraping
        """
        # Update metrics before returning
        self.update_metrics()
        
        # Generate Prometheus format
        return generate_latest(self.registry).decode('utf-8')
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current metrics values.
        
        Returns:
            Dictionary containing current metrics values
        """
        self.update_metrics()
        
        with self._lock:
            return {
                'timestamp': time.time(),
                'monitoring_active': bool(self.monitoring_active_gauge._value._value),
                'cache_hit_rate': self.cache_hit_rate_gauge._value._value,
                'cpu_percent': self.system_cpu_percent._value._value,
                'memory_percent': self.system_memory_percent._value._value,
                'memory_used_bytes': self.system_memory_used_bytes._value._value,
                'active_alerts': {
                    'warning': self.active_alerts_gauge.labels(severity='warning')._value._value,
                    'critical': self.active_alerts_gauge.labels(severity='critical')._value._value
                }
            }


class MetricsIntegration:
    """
    Integration class that connects PerformanceMonitor with PrometheusMetricsCollector.
    
    Provides a unified interface for metrics collection with both internal monitoring
    and Prometheus export capabilities.
    """
    
    def __init__(self, alert_thresholds: Optional[Dict[str, float]] = None,
                 monitoring_interval: float = 30.0,
                 metrics_update_interval: float = 5.0):
        """
        Initialize the integrated metrics system.
        
        Args:
            alert_thresholds: Thresholds for performance alerts
            monitoring_interval: Interval for resource monitoring
            metrics_update_interval: Interval for Prometheus metrics updates
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize performance monitor
        self.performance_monitor = PerformanceMonitor(
            alert_thresholds=alert_thresholds,
            monitoring_interval=monitoring_interval
        )
        
        # Initialize Prometheus collector
        self.prometheus_collector = PrometheusMetricsCollector(
            performance_monitor=self.performance_monitor
        )
        self.prometheus_collector._update_interval = metrics_update_interval
        
        self.logger.info("Integrated metrics system initialized")
    
    def start_monitoring(self):
        """Start both performance monitoring and metrics collection."""
        self.performance_monitor.start_monitoring()
        self.logger.info("Integrated metrics monitoring started")
    
    def stop_monitoring(self):
        """Stop both performance monitoring and metrics collection."""
        self.performance_monitor.stop_monitoring()
        self.logger.info("Integrated metrics monitoring stopped")
    
    def record_query_performance(self, query_id: str, query_type: str,
                                execution_time_ms: float, result_count: int,
                                cache_hit: bool = False, error: Optional[str] = None):
        """Record query performance in both systems."""
        # Record in performance monitor
        self.performance_monitor.record_query_performance(
            query_id=query_id,
            query_type=query_type,
            execution_time_ms=execution_time_ms,
            result_count=result_count,
            cache_hit=cache_hit,
            error=error
        )
        
        # Record in Prometheus collector
        self.prometheus_collector.record_query_metric(
            query_type=query_type,
            execution_time_ms=execution_time_ms,
            result_count=result_count,
            cache_hit=cache_hit,
            error=error
        )
    
    def record_ingestion_performance(self, operation_id: str, operation_type: str,
                                   items_processed: int, processing_time_ms: float,
                                   error_count: int = 0):
        """Record ingestion performance in both systems."""
        # Record in performance monitor
        self.performance_monitor.record_ingestion_performance(
            operation_id=operation_id,
            operation_type=operation_type,
            items_processed=items_processed,
            processing_time_ms=processing_time_ms,
            error_count=error_count
        )
        
        # Record in Prometheus collector
        self.prometheus_collector.record_ingestion_metric(
            operation_type=operation_type,
            processing_time_ms=processing_time_ms,
            items_processed=items_processed,
            error_count=error_count
        )
    
    def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        return self.prometheus_collector.get_metrics_text()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return self.performance_monitor.get_performance_summary()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get Prometheus metrics summary."""
        return self.prometheus_collector.get_metrics_summary()