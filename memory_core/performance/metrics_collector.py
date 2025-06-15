"""
Comprehensive metrics collection and monitoring system.
"""

import time
import threading
import asyncio
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


@dataclass
class MetricValue:
    """A single metric value with timestamp."""
    value: Union[int, float]
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSummary:
    """Summary statistics for a metric."""
    name: str
    metric_type: MetricType
    count: int
    current_value: Union[int, float]
    min_value: Union[int, float]
    max_value: Union[int, float]
    avg_value: float
    sum_value: Union[int, float]
    last_updated: float
    tags: Dict[str, str] = field(default_factory=dict)


class Metric:
    """Base metric class."""
    
    def __init__(self, name: str, metric_type: MetricType, 
                 tags: Dict[str, str] = None, max_history: int = 1000):
        self.name = name
        self.metric_type = metric_type
        self.tags = tags or {}
        self.max_history = max_history
        self._values: deque = deque(maxlen=max_history)
        self._lock = threading.RLock()
    
    def record(self, value: Union[int, float], tags: Dict[str, str] = None):
        """Record a metric value."""
        with self._lock:
            metric_value = MetricValue(
                value=value,
                timestamp=time.time(),
                tags={**self.tags, **(tags or {})}
            )
            self._values.append(metric_value)
    
    def get_current_value(self) -> Optional[Union[int, float]]:
        """Get the most recent value."""
        with self._lock:
            if self._values:
                return self._values[-1].value
            return None
    
    def get_summary(self) -> MetricSummary:
        """Get summary statistics."""
        with self._lock:
            if not self._values:
                return MetricSummary(
                    name=self.name,
                    metric_type=self.metric_type,
                    count=0,
                    current_value=0,
                    min_value=0,
                    max_value=0,
                    avg_value=0,
                    sum_value=0,
                    last_updated=0,
                    tags=self.tags
                )
            
            values = [v.value for v in self._values]
            
            return MetricSummary(
                name=self.name,
                metric_type=self.metric_type,
                count=len(values),
                current_value=values[-1],
                min_value=min(values),
                max_value=max(values),
                avg_value=statistics.mean(values),
                sum_value=sum(values),
                last_updated=self._values[-1].timestamp,
                tags=self.tags
            )
    
    def get_values(self, since: Optional[float] = None) -> List[MetricValue]:
        """Get metric values since timestamp."""
        with self._lock:
            if since is None:
                return list(self._values)
            
            return [v for v in self._values if v.timestamp >= since]


class Counter(Metric):
    """Counter metric - monotonically increasing value."""
    
    def __init__(self, name: str, tags: Dict[str, str] = None):
        super().__init__(name, MetricType.COUNTER, tags)
        self._count = 0
    
    def increment(self, amount: Union[int, float] = 1, tags: Dict[str, str] = None):
        """Increment counter."""
        with self._lock:
            self._count += amount
            self.record(self._count, tags)
    
    def get_count(self) -> Union[int, float]:
        """Get current count."""
        return self._count


class Gauge(Metric):
    """Gauge metric - current value that can go up or down."""
    
    def __init__(self, name: str, tags: Dict[str, str] = None):
        super().__init__(name, MetricType.GAUGE, tags)
    
    def set(self, value: Union[int, float], tags: Dict[str, str] = None):
        """Set gauge value."""
        self.record(value, tags)
    
    def increment(self, amount: Union[int, float] = 1, tags: Dict[str, str] = None):
        """Increment gauge value."""
        current = self.get_current_value() or 0
        self.set(current + amount, tags)
    
    def decrement(self, amount: Union[int, float] = 1, tags: Dict[str, str] = None):
        """Decrement gauge value."""
        current = self.get_current_value() or 0
        self.set(current - amount, tags)


class Histogram(Metric):
    """Histogram metric - distribution of values."""
    
    def __init__(self, name: str, buckets: List[float] = None, tags: Dict[str, str] = None):
        super().__init__(name, MetricType.HISTOGRAM, tags)
        self.buckets = buckets or [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, float('inf')]
        self._bucket_counts = defaultdict(int)
    
    def observe(self, value: Union[int, float], tags: Dict[str, str] = None):
        """Observe a value."""
        self.record(value, tags)
        
        # Update bucket counts
        for bucket in self.buckets:
            if value <= bucket:
                self._bucket_counts[bucket] += 1
    
    def get_percentile(self, percentile: float) -> Optional[float]:
        """Get percentile value."""
        with self._lock:
            if not self._values:
                return None
            
            values = sorted([v.value for v in self._values])
            index = int(len(values) * percentile / 100)
            return values[min(index, len(values) - 1)]
    
    def get_bucket_counts(self) -> Dict[float, int]:
        """Get bucket counts."""
        return dict(self._bucket_counts)


class Timer(Metric):
    """Timer metric - measures duration."""
    
    def __init__(self, name: str, tags: Dict[str, str] = None):
        super().__init__(name, MetricType.TIMER, tags)
    
    def time(self, tags: Dict[str, str] = None):
        """Context manager for timing operations."""
        return TimerContext(self, tags)
    
    def record_duration(self, duration: float, tags: Dict[str, str] = None):
        """Record a duration in seconds."""
        self.record(duration, tags)


class TimerContext:
    """Context manager for timer metric."""
    
    def __init__(self, timer: Timer, tags: Dict[str, str] = None):
        self.timer = timer
        self.tags = tags
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.timer.record_duration(duration, self.tags)


class Rate(Metric):
    """Rate metric - rate of events over time."""
    
    def __init__(self, name: str, window_size: int = 60, tags: Dict[str, str] = None):
        super().__init__(name, MetricType.RATE, tags)
        self.window_size = window_size  # seconds
        self._events: deque = deque()
    
    def mark(self, count: int = 1, tags: Dict[str, str] = None):
        """Mark events."""
        timestamp = time.time()
        self._events.append((timestamp, count))
        
        # Clean old events
        cutoff = timestamp - self.window_size
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()
        
        # Calculate current rate
        total_events = sum(count for _, count in self._events)
        rate = total_events / self.window_size
        self.record(rate, tags)
    
    def get_rate(self) -> float:
        """Get current rate (events per second)."""
        current_value = self.get_current_value()
        return current_value or 0.0


class MetricsCollector:
    """Central metrics collection system."""
    
    def __init__(self, enable_export: bool = True, export_interval: int = 60):
        self.enable_export = enable_export
        self.export_interval = export_interval
        
        self._metrics: Dict[str, Metric] = {}
        self._lock = threading.RLock()
        self._exporters: List[Callable[[Dict[str, MetricSummary]], None]] = []
        self._export_thread: Optional[threading.Thread] = None
        self._stop_export = threading.Event()
        
        # Built-in system metrics
        self._setup_system_metrics()
        
        # Start export thread
        if enable_export:
            self._start_export_thread()
    
    def _setup_system_metrics(self):
        """Setup built-in system metrics."""
        self.counter("system.requests.total", {"component": "memory_engine"})
        self.gauge("system.memory.usage_mb", {"component": "memory_engine"})
        self.timer("system.requests.duration", {"component": "memory_engine"})
        self.rate("system.requests.rate", {"component": "memory_engine"})
    
    def _start_export_thread(self):
        """Start metrics export thread."""
        def export_loop():
            while not self._stop_export.wait(self.export_interval):
                try:
                    self._export_metrics()
                except Exception as e:
                    logger.error(f"Metrics export failed: {e}")
        
        self._export_thread = threading.Thread(target=export_loop, daemon=True)
        self._export_thread.start()
    
    def counter(self, name: str, tags: Dict[str, str] = None) -> Counter:
        """Get or create counter metric."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Counter(name, tags)
            return self._metrics[name]
    
    def gauge(self, name: str, tags: Dict[str, str] = None) -> Gauge:
        """Get or create gauge metric."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Gauge(name, tags)
            return self._metrics[name]
    
    def histogram(self, name: str, buckets: List[float] = None, 
                  tags: Dict[str, str] = None) -> Histogram:
        """Get or create histogram metric."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Histogram(name, buckets, tags)
            return self._metrics[name]
    
    def timer(self, name: str, tags: Dict[str, str] = None) -> Timer:
        """Get or create timer metric."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Timer(name, tags)
            return self._metrics[name]
    
    def rate(self, name: str, window_size: int = 60, 
             tags: Dict[str, str] = None) -> Rate:
        """Get or create rate metric."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Rate(name, window_size, tags)
            return self._metrics[name]
    
    def get_metric(self, name: str) -> Optional[Metric]:
        """Get metric by name."""
        return self._metrics.get(name)
    
    def get_all_metrics(self) -> Dict[str, Metric]:
        """Get all metrics."""
        with self._lock:
            return dict(self._metrics)
    
    def get_summary(self) -> Dict[str, MetricSummary]:
        """Get summary of all metrics."""
        with self._lock:
            return {name: metric.get_summary() for name, metric in self._metrics.items()}
    
    def register_exporter(self, exporter: Callable[[Dict[str, MetricSummary]], None]):
        """Register metrics exporter."""
        self._exporters.append(exporter)
    
    def _export_metrics(self):
        """Export metrics to registered exporters."""
        summary = self.get_summary()
        for exporter in self._exporters:
            try:
                exporter(summary)
            except Exception as e:
                logger.error(f"Metrics exporter failed: {e}")
    
    def export_to_json(self, file_path: str):
        """Export metrics to JSON file."""
        summary = self.get_summary()
        
        # Convert to serializable format
        json_data = {}
        for name, metric_summary in summary.items():
            json_data[name] = {
                'type': metric_summary.metric_type.value,
                'count': metric_summary.count,
                'current_value': metric_summary.current_value,
                'min_value': metric_summary.min_value,
                'max_value': metric_summary.max_value,
                'avg_value': metric_summary.avg_value,
                'sum_value': metric_summary.sum_value,
                'last_updated': metric_summary.last_updated,
                'tags': metric_summary.tags
            }
        
        with open(file_path, 'w') as f:
            json.dump(json_data, f, indent=2)
    
    def clear(self):
        """Clear all metrics."""
        with self._lock:
            self._metrics.clear()
    
    def shutdown(self):
        """Shutdown metrics collector."""
        self._stop_export.set()
        if self._export_thread:
            self._export_thread.join(timeout=1)


class PerformanceMonitor:
    """High-level performance monitoring system."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self._component_metrics: Dict[str, Dict[str, Metric]] = defaultdict(dict)
    
    def track_request(self, component: str, operation: str):
        """Track a request operation."""
        # Increment request counter
        self.metrics.counter(
            f"{component}.requests.total",
            {"operation": operation}
        ).increment()
        
        # Mark request rate
        self.metrics.rate(
            f"{component}.requests.rate",
            tags={"operation": operation}
        ).mark()
        
        # Return timer context for duration tracking
        return self.metrics.timer(
            f"{component}.requests.duration",
            {"operation": operation}
        ).time()
    
    def track_error(self, component: str, operation: str, error_type: str):
        """Track an error."""
        self.metrics.counter(
            f"{component}.errors.total",
            {"operation": operation, "error_type": error_type}
        ).increment()
    
    def track_resource_usage(self, component: str, resource: str, value: Union[int, float]):
        """Track resource usage."""
        self.metrics.gauge(
            f"{component}.resources.{resource}",
            {"component": component}
        ).set(value)
    
    def track_cache_hit(self, cache_name: str):
        """Track cache hit."""
        self.metrics.counter(
            f"cache.hits.total",
            {"cache": cache_name}
        ).increment()
    
    def track_cache_miss(self, cache_name: str):
        """Track cache miss."""
        self.metrics.counter(
            f"cache.misses.total",
            {"cache": cache_name}
        ).increment()
    
    def track_database_query(self, backend: str, operation: str, duration: float):
        """Track database query."""
        self.metrics.timer(
            f"database.query.duration",
            {"backend": backend, "operation": operation}
        ).record_duration(duration)
        
        self.metrics.counter(
            f"database.queries.total",
            {"backend": backend, "operation": operation}
        ).increment()
    
    def track_embedding_generation(self, provider: str, count: int, duration: float):
        """Track embedding generation."""
        self.metrics.timer(
            f"embeddings.generation.duration",
            {"provider": provider}
        ).record_duration(duration)
        
        self.metrics.histogram(
            f"embeddings.batch_size",
            tags={"provider": provider}
        ).observe(count)
    
    def track_llm_request(self, provider: str, model: str, tokens: int, duration: float):
        """Track LLM request."""
        self.metrics.timer(
            f"llm.requests.duration",
            {"provider": provider, "model": model}
        ).record_duration(duration)
        
        self.metrics.histogram(
            f"llm.tokens.total",
            tags={"provider": provider, "model": model}
        ).observe(tokens)
    
    def get_component_health(self, component: str) -> Dict[str, Any]:
        """Get health metrics for a component."""
        summary = self.metrics.get_summary()
        
        component_metrics = {
            name: metric for name, metric in summary.items()
            if name.startswith(f"{component}.")
        }
        
        # Calculate health score based on error rate
        total_requests = 0
        total_errors = 0
        
        for name, metric in component_metrics.items():
            if "requests.total" in name:
                total_requests += metric.current_value
            elif "errors.total" in name:
                total_errors += metric.current_value
        
        error_rate = (total_errors / total_requests) if total_requests > 0 else 0
        health_score = max(0, 100 - (error_rate * 100))
        
        return {
            'component': component,
            'health_score': health_score,
            'total_requests': total_requests,
            'total_errors': total_errors,
            'error_rate': error_rate,
            'metrics': component_metrics
        }
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get system-wide performance overview."""
        summary = self.metrics.get_summary()
        
        # Group metrics by component
        components = defaultdict(list)
        for name, metric in summary.items():
            if '.' in name:
                component = name.split('.')[0]
                components[component].append((name, metric))
        
        overview = {
            'total_metrics': len(summary),
            'components': {},
            'system_health': 'unknown'
        }
        
        total_health_score = 0
        component_count = 0
        
        for component, metrics in components.items():
            health = self.get_component_health(component)
            overview['components'][component] = health
            total_health_score += health['health_score']
            component_count += 1
        
        if component_count > 0:
            avg_health = total_health_score / component_count
            if avg_health >= 90:
                overview['system_health'] = 'excellent'
            elif avg_health >= 75:
                overview['system_health'] = 'good'
            elif avg_health >= 50:
                overview['system_health'] = 'fair'
            else:
                overview['system_health'] = 'poor'
        
        return overview