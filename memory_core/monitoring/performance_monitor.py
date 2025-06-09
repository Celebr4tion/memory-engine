"""
Performance monitoring system for Memory Engine.

This module provides comprehensive performance monitoring, metrics collection,
and alerting capabilities for query performance, ingestion throughput,
and resource utilization tracking.
"""

import logging
import psutil
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
import json
import statistics


@dataclass
class QueryMetrics:
    """Metrics for individual query performance."""
    query_id: str
    query_type: str
    execution_time_ms: float
    result_count: int
    cache_hit: bool
    timestamp: float
    error: Optional[str] = None


@dataclass
class IngestionMetrics:
    """Metrics for ingestion operations."""
    operation_id: str
    operation_type: str  # 'document', 'knowledge_unit', 'relationship'
    items_processed: int
    processing_time_ms: float
    throughput_per_second: float
    timestamp: float
    error_count: int = 0


@dataclass
class ResourceMetrics:
    """System resource utilization metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_io_sent_mb: float
    network_io_recv_mb: float


@dataclass
class PerformanceAlert:
    """Performance alert for threshold violations."""
    alert_id: str
    metric_type: str
    severity: str  # 'warning', 'critical'
    message: str
    value: float
    threshold: float
    timestamp: float


class MetricsAggregator:
    """Aggregates and analyzes performance metrics over time windows."""
    
    def __init__(self, window_size_minutes: int = 5):
        self.window_size = timedelta(minutes=window_size_minutes)
        self.query_metrics: deque = deque()
        self.ingestion_metrics: deque = deque()
        self.resource_metrics: deque = deque()
        self._lock = threading.RLock()
    
    def add_query_metric(self, metric: QueryMetrics):
        """Add a query performance metric."""
        with self._lock:
            self.query_metrics.append(metric)
            self._cleanup_old_metrics()
    
    def add_ingestion_metric(self, metric: IngestionMetrics):
        """Add an ingestion performance metric."""
        with self._lock:
            self.ingestion_metrics.append(metric)
            self._cleanup_old_metrics()
    
    def add_resource_metric(self, metric: ResourceMetrics):
        """Add a resource utilization metric."""
        with self._lock:
            self.resource_metrics.append(metric)
            self._cleanup_old_metrics()
    
    def _cleanup_old_metrics(self):
        """Remove metrics older than the window size."""
        cutoff_time = time.time() - self.window_size.total_seconds()
        
        # Clean query metrics
        while self.query_metrics and self.query_metrics[0].timestamp < cutoff_time:
            self.query_metrics.popleft()
        
        # Clean ingestion metrics
        while self.ingestion_metrics and self.ingestion_metrics[0].timestamp < cutoff_time:
            self.ingestion_metrics.popleft()
        
        # Clean resource metrics
        while self.resource_metrics and self.resource_metrics[0].timestamp < cutoff_time:
            self.resource_metrics.popleft()
    
    def get_query_statistics(self) -> Dict[str, Any]:
        """Get aggregated query performance statistics."""
        with self._lock:
            if not self.query_metrics:
                return {}
            
            execution_times = [m.execution_time_ms for m in self.query_metrics]
            result_counts = [m.result_count for m in self.query_metrics]
            cache_hits = sum(1 for m in self.query_metrics if m.cache_hit)
            error_count = sum(1 for m in self.query_metrics if m.error)
            
            # Group by query type
            by_type = defaultdict(list)
            for metric in self.query_metrics:
                by_type[metric.query_type].append(metric.execution_time_ms)
            
            return {
                'total_queries': len(self.query_metrics),
                'average_execution_time_ms': statistics.mean(execution_times),
                'median_execution_time_ms': statistics.median(execution_times),
                'p95_execution_time_ms': statistics.quantiles(execution_times, n=20)[18] if len(execution_times) > 10 else max(execution_times),
                'max_execution_time_ms': max(execution_times),
                'average_result_count': statistics.mean(result_counts),
                'cache_hit_rate': cache_hits / len(self.query_metrics),
                'error_rate': error_count / len(self.query_metrics),
                'queries_per_second': len(self.query_metrics) / self.window_size.total_seconds(),
                'by_query_type': {
                    qtype: {
                        'count': len(times),
                        'avg_time_ms': statistics.mean(times),
                        'max_time_ms': max(times)
                    }
                    for qtype, times in by_type.items()
                }
            }
    
    def get_ingestion_statistics(self) -> Dict[str, Any]:
        """Get aggregated ingestion performance statistics."""
        with self._lock:
            if not self.ingestion_metrics:
                return {}
            
            throughputs = [m.throughput_per_second for m in self.ingestion_metrics]
            processing_times = [m.processing_time_ms for m in self.ingestion_metrics]
            total_items = sum(m.items_processed for m in self.ingestion_metrics)
            total_errors = sum(m.error_count for m in self.ingestion_metrics)
            
            # Group by operation type
            by_type = defaultdict(list)
            for metric in self.ingestion_metrics:
                by_type[metric.operation_type].append(metric.throughput_per_second)
            
            return {
                'total_operations': len(self.ingestion_metrics),
                'total_items_processed': total_items,
                'average_throughput_per_second': statistics.mean(throughputs),
                'peak_throughput_per_second': max(throughputs),
                'average_processing_time_ms': statistics.mean(processing_times),
                'total_errors': total_errors,
                'error_rate': total_errors / total_items if total_items > 0 else 0,
                'by_operation_type': {
                    op_type: {
                        'count': len(throughputs_list),
                        'avg_throughput': statistics.mean(throughputs_list),
                        'peak_throughput': max(throughputs_list)
                    }
                    for op_type, throughputs_list in by_type.items()
                }
            }
    
    def get_resource_statistics(self) -> Dict[str, Any]:
        """Get aggregated resource utilization statistics."""
        with self._lock:
            if not self.resource_metrics:
                return {}
            
            cpu_values = [m.cpu_percent for m in self.resource_metrics]
            memory_values = [m.memory_percent for m in self.resource_metrics]
            memory_used = [m.memory_used_mb for m in self.resource_metrics]
            
            return {
                'sample_count': len(self.resource_metrics),
                'cpu_utilization': {
                    'average': statistics.mean(cpu_values),
                    'max': max(cpu_values),
                    'p95': statistics.quantiles(cpu_values, n=20)[18] if len(cpu_values) > 10 else max(cpu_values)
                },
                'memory_utilization': {
                    'average_percent': statistics.mean(memory_values),
                    'max_percent': max(memory_values),
                    'average_used_mb': statistics.mean(memory_used),
                    'max_used_mb': max(memory_used)
                }
            }


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system.
    
    Features:
    - Real-time metrics collection
    - Performance threshold monitoring
    - Automatic alerting
    - Historical data tracking
    - Resource utilization monitoring
    - Query and ingestion performance tracking
    """
    
    def __init__(self, 
                 alert_thresholds: Optional[Dict[str, float]] = None,
                 monitoring_interval: float = 30.0):
        """
        Initialize the performance monitor.
        
        Args:
            alert_thresholds: Dictionary of metric thresholds for alerting
            monitoring_interval: Interval between resource monitoring samples (seconds)
        """
        self.logger = logging.getLogger(__name__)
        
        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'query_avg_time_ms': 5000.0,
            'query_error_rate': 0.05,
            'ingestion_error_rate': 0.02
        }
        
        self.monitoring_interval = monitoring_interval
        
        # Metrics storage
        self.aggregator = MetricsAggregator()
        self.alerts: List[PerformanceAlert] = []
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[PerformanceAlert], None]] = []
        
        # Monitoring state
        self.running = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Resource monitoring state
        self._last_disk_io = psutil.disk_io_counters()
        self._last_network_io = psutil.net_io_counters()
        self._last_sample_time = time.time()
        
        self.logger.info("Performance monitor initialized")
    
    def start_monitoring(self):
        """Start the performance monitoring system."""
        if self.running:
            return
        
        self.running = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="performance-monitor",
            daemon=True
        )
        self.monitoring_thread.start()
        
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop the performance monitoring system."""
        if not self.running:
            return
        
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.info("Performance monitoring stopped")
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Add a callback function to be called when alerts are triggered."""
        self.alert_callbacks.append(callback)
    
    def record_query_performance(self, 
                                query_id: str,
                                query_type: str,
                                execution_time_ms: float,
                                result_count: int,
                                cache_hit: bool = False,
                                error: Optional[str] = None):
        """Record query performance metrics."""
        metric = QueryMetrics(
            query_id=query_id,
            query_type=query_type,
            execution_time_ms=execution_time_ms,
            result_count=result_count,
            cache_hit=cache_hit,
            timestamp=time.time(),
            error=error
        )
        
        self.aggregator.add_query_metric(metric)
        self._check_query_alerts(metric)
    
    def record_ingestion_performance(self,
                                   operation_id: str,
                                   operation_type: str,
                                   items_processed: int,
                                   processing_time_ms: float,
                                   error_count: int = 0):
        """Record ingestion performance metrics."""
        throughput = (items_processed / (processing_time_ms / 1000.0)) if processing_time_ms > 0 else 0
        
        metric = IngestionMetrics(
            operation_id=operation_id,
            operation_type=operation_type,
            items_processed=items_processed,
            processing_time_ms=processing_time_ms,
            throughput_per_second=throughput,
            timestamp=time.time(),
            error_count=error_count
        )
        
        self.aggregator.add_ingestion_metric(metric)
        self._check_ingestion_alerts(metric)
    
    def _monitoring_loop(self):
        """Main monitoring loop for resource collection."""
        while self.running:
            try:
                # Collect resource metrics
                resource_metric = self._collect_resource_metrics()
                self.aggregator.add_resource_metric(resource_metric)
                
                # Check resource alerts
                self._check_resource_alerts(resource_metric)
                
                # Check aggregated metrics alerts
                self._check_aggregated_alerts()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_resource_metrics(self) -> ResourceMetrics:
        """Collect current system resource metrics."""
        current_time = time.time()
        
        # CPU and memory
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_read_delta = (disk_io.read_bytes - self._last_disk_io.read_bytes) / (1024 * 1024)
        disk_write_delta = (disk_io.write_bytes - self._last_disk_io.write_bytes) / (1024 * 1024)
        
        # Network I/O
        network_io = psutil.net_io_counters()
        network_sent_delta = (network_io.bytes_sent - self._last_network_io.bytes_sent) / (1024 * 1024)
        network_recv_delta = (network_io.bytes_recv - self._last_network_io.bytes_recv) / (1024 * 1024)
        
        # Update last values
        self._last_disk_io = disk_io
        self._last_network_io = network_io
        self._last_sample_time = current_time
        
        return ResourceMetrics(
            timestamp=current_time,
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            disk_io_read_mb=disk_read_delta,
            disk_io_write_mb=disk_write_delta,
            network_io_sent_mb=network_sent_delta,
            network_io_recv_mb=network_recv_delta
        )
    
    def _check_query_alerts(self, metric: QueryMetrics):
        """Check for query performance alerts."""
        # Check execution time threshold
        if (metric.execution_time_ms > self.alert_thresholds.get('query_avg_time_ms', float('inf')) and
            not metric.error):
            self._trigger_alert(
                'query_performance',
                'warning',
                f"Slow query detected: {metric.execution_time_ms:.2f}ms (ID: {metric.query_id})",
                metric.execution_time_ms,
                self.alert_thresholds['query_avg_time_ms']
            )
    
    def _check_ingestion_alerts(self, metric: IngestionMetrics):
        """Check for ingestion performance alerts."""
        if metric.error_count > 0:
            error_rate = metric.error_count / metric.items_processed
            threshold = self.alert_thresholds.get('ingestion_error_rate', 1.0)
            
            if error_rate > threshold:
                self._trigger_alert(
                    'ingestion_errors',
                    'warning',
                    f"High ingestion error rate: {error_rate:.2%} in operation {metric.operation_id}",
                    error_rate,
                    threshold
                )
    
    def _check_resource_alerts(self, metric: ResourceMetrics):
        """Check for resource utilization alerts."""
        # CPU threshold
        cpu_threshold = self.alert_thresholds.get('cpu_percent', 100.0)
        if metric.cpu_percent > cpu_threshold:
            self._trigger_alert(
                'cpu_utilization',
                'critical' if metric.cpu_percent > 90 else 'warning',
                f"High CPU utilization: {metric.cpu_percent:.1f}%",
                metric.cpu_percent,
                cpu_threshold
            )
        
        # Memory threshold
        memory_threshold = self.alert_thresholds.get('memory_percent', 100.0)
        if metric.memory_percent > memory_threshold:
            self._trigger_alert(
                'memory_utilization',
                'critical' if metric.memory_percent > 95 else 'warning',
                f"High memory utilization: {metric.memory_percent:.1f}%",
                metric.memory_percent,
                memory_threshold
            )
    
    def _check_aggregated_alerts(self):
        """Check alerts based on aggregated metrics over time windows."""
        # Check query error rate
        query_stats = self.aggregator.get_query_statistics()
        if query_stats:
            error_rate = query_stats.get('error_rate', 0)
            threshold = self.alert_thresholds.get('query_error_rate', 1.0)
            
            if error_rate > threshold:
                self._trigger_alert(
                    'query_error_rate',
                    'critical' if error_rate > 0.1 else 'warning',
                    f"High query error rate over time window: {error_rate:.2%}",
                    error_rate,
                    threshold
                )
    
    def _trigger_alert(self, 
                      metric_type: str, 
                      severity: str, 
                      message: str,
                      value: float,
                      threshold: float):
        """Trigger a performance alert."""
        alert = PerformanceAlert(
            alert_id=f"{metric_type}_{int(time.time())}",
            metric_type=metric_type,
            severity=severity,
            message=message,
            value=value,
            threshold=threshold,
            timestamp=time.time()
        )
        
        self.alerts.append(alert)
        
        # Keep only recent alerts
        cutoff_time = time.time() - 3600  # 1 hour
        self.alerts = [a for a in self.alerts if a.timestamp > cutoff_time]
        
        # Log alert
        log_level = logging.CRITICAL if severity == 'critical' else logging.WARNING
        self.logger.log(log_level, f"Performance Alert [{severity.upper()}]: {message}")
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            'timestamp': time.time(),
            'monitoring_active': self.running,
            'query_performance': self.aggregator.get_query_statistics(),
            'ingestion_performance': self.aggregator.get_ingestion_statistics(),
            'resource_utilization': self.aggregator.get_resource_statistics(),
            'recent_alerts': [
                {
                    'metric_type': alert.metric_type,
                    'severity': alert.severity,
                    'message': alert.message,
                    'timestamp': alert.timestamp
                }
                for alert in self.alerts[-10:]  # Last 10 alerts
            ],
            'alert_thresholds': self.alert_thresholds
        }
    
    def export_metrics(self, format: str = 'json') -> str:
        """Export metrics in specified format."""
        summary = self.get_performance_summary()
        
        if format.lower() == 'json':
            return json.dumps(summary, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_performance_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        query_stats = self.aggregator.get_query_statistics()
        if query_stats:
            # Query performance recommendations
            if query_stats.get('cache_hit_rate', 0) < 0.3:
                recommendations.append("Consider optimizing query caching - low cache hit rate detected")
            
            if query_stats.get('average_execution_time_ms', 0) > 1000:
                recommendations.append("Query execution times are high - consider optimizing query patterns or adding indexes")
            
            if query_stats.get('error_rate', 0) > 0.01:
                recommendations.append("Query error rate is elevated - review query validation and error handling")
        
        resource_stats = self.aggregator.get_resource_statistics()
        if resource_stats:
            # Resource utilization recommendations
            cpu_avg = resource_stats.get('cpu_utilization', {}).get('average', 0)
            if cpu_avg > 70:
                recommendations.append("High CPU utilization - consider scaling up or optimizing algorithms")
            
            memory_avg = resource_stats.get('memory_utilization', {}).get('average_percent', 0)
            if memory_avg > 80:
                recommendations.append("High memory utilization - consider increasing memory or optimizing data structures")
        
        return recommendations


# Integration decorator for automatic performance tracking
def track_performance(monitor: PerformanceMonitor, operation_type: str = 'operation'):
    """Decorator to automatically track function performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            operation_id = f"{func.__name__}_{int(start_time)}"
            error_count = 0
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                error_count = 1
                raise
            finally:
                execution_time = (time.time() - start_time) * 1000
                
                # Record performance metric
                if operation_type.startswith('query'):
                    monitor.record_query_performance(
                        query_id=operation_id,
                        query_type=operation_type,
                        execution_time_ms=execution_time,
                        result_count=len(result) if hasattr(result, '__len__') else 1,
                        error='error' if error_count > 0 else None
                    )
                else:
                    monitor.record_ingestion_performance(
                        operation_id=operation_id,
                        operation_type=operation_type,
                        items_processed=1,
                        processing_time_ms=execution_time,
                        error_count=error_count
                    )
        
        return wrapper
    return decorator