"""
Performance optimization module for Memory Engine.

This module provides comprehensive performance optimization features including:
- Advanced query result caching with TTL
- Connection pooling management
- Batch operation optimizations
- Memory management and garbage collection
- Performance monitoring and metrics
"""

from .cache_manager import CacheManager, CacheConfig
from .connection_pool import ConnectionPoolManager
from .batch_optimizer import BatchOptimizer
from .memory_manager import MemoryManager
from .metrics_collector import MetricsCollector, PerformanceMonitor
from .prepared_statements import PreparedStatementManager

__all__ = [
    'CacheManager',
    'CacheConfig', 
    'ConnectionPoolManager',
    'BatchOptimizer',
    'MemoryManager',
    'MetricsCollector',
    'PerformanceMonitor',
    'PreparedStatementManager'
]