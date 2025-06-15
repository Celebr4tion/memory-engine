"""
Advanced memory management and garbage collection optimization.
"""

import gc
import psutil
import threading
import time
import weakref
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass
from enum import Enum
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class MemoryPressureLevel(Enum):
    """Memory pressure levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MemoryConfig:
    """Memory management configuration."""

    max_memory_mb: int = 1024
    warning_threshold: float = 0.8  # 80%
    critical_threshold: float = 0.95  # 95%
    gc_interval: int = 60  # seconds
    aggressive_gc_threshold: float = 0.9
    enable_auto_cleanup: bool = True
    enable_memory_monitoring: bool = True
    cleanup_interval: int = 300  # 5 minutes
    large_object_threshold_mb: int = 10


@dataclass
class MemoryStats:
    """Memory usage statistics."""

    total_memory_mb: float = 0.0
    used_memory_mb: float = 0.0
    available_memory_mb: float = 0.0
    memory_percent: float = 0.0
    process_memory_mb: float = 0.0
    process_memory_percent: float = 0.0
    gc_collections: int = 0
    objects_tracked: int = 0
    large_objects_count: int = 0
    cache_memory_mb: float = 0.0
    pressure_level: MemoryPressureLevel = MemoryPressureLevel.LOW

    def update_pressure_level(self, warning_threshold: float, critical_threshold: float):
        """Update memory pressure level."""
        if self.memory_percent >= critical_threshold:
            self.pressure_level = MemoryPressureLevel.CRITICAL
        elif self.memory_percent >= warning_threshold:
            self.pressure_level = MemoryPressureLevel.HIGH
        elif self.memory_percent >= warning_threshold * 0.7:
            self.pressure_level = MemoryPressureLevel.MEDIUM
        else:
            self.pressure_level = MemoryPressureLevel.LOW


class MemoryTracker:
    """Track memory usage of objects and caches."""

    def __init__(self):
        self._tracked_objects: Dict[str, weakref.ref] = {}
        self._object_sizes: Dict[str, int] = {}
        self._large_objects: Set[str] = set()
        self._lock = threading.RLock()

    def track_object(self, obj: Any, name: str, size_bytes: int = 0):
        """Track an object's memory usage."""
        with self._lock:
            if size_bytes == 0:
                size_bytes = self._estimate_object_size(obj)

            obj_id = f"{name}_{id(obj)}"
            self._tracked_objects[obj_id] = weakref.ref(obj, self._cleanup_callback(obj_id))
            self._object_sizes[obj_id] = size_bytes

            # Track large objects separately
            if size_bytes > 10 * 1024 * 1024:  # 10MB
                self._large_objects.add(obj_id)

    def untrack_object(self, obj: Any, name: str):
        """Stop tracking an object."""
        with self._lock:
            obj_id = f"{name}_{id(obj)}"
            self._tracked_objects.pop(obj_id, None)
            self._object_sizes.pop(obj_id, None)
            self._large_objects.discard(obj_id)

    def get_tracked_memory(self) -> int:
        """Get total memory of tracked objects."""
        with self._lock:
            # Clean up dead references first
            self._cleanup_dead_references()
            return sum(self._object_sizes.values())

    def get_large_objects_count(self) -> int:
        """Get count of large objects."""
        with self._lock:
            return len(self._large_objects)

    def _cleanup_callback(self, obj_id: str) -> Callable:
        """Create cleanup callback for weak reference."""

        def cleanup():
            with self._lock:
                self._object_sizes.pop(obj_id, None)
                self._large_objects.discard(obj_id)

        return cleanup

    def _cleanup_dead_references(self):
        """Clean up dead weak references."""
        dead_refs = []
        for obj_id, ref in self._tracked_objects.items():
            if ref() is None:
                dead_refs.append(obj_id)

        for obj_id in dead_refs:
            self._tracked_objects.pop(obj_id, None)
            self._object_sizes.pop(obj_id, None)
            self._large_objects.discard(obj_id)

    def _estimate_object_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            import sys

            return sys.getsizeof(obj)
        except:
            return 1024  # Fallback estimate


class GarbageCollectionOptimizer:
    """Optimize garbage collection based on memory pressure."""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self._last_gc_time = time.time()
        self._gc_stats = {"collections": 0, "collected_objects": 0}
        self._lock = threading.Lock()

    def should_run_gc(self, memory_stats: MemoryStats) -> bool:
        """Determine if garbage collection should run."""
        current_time = time.time()
        time_since_last_gc = current_time - self._last_gc_time

        # Always run if enough time has passed
        if time_since_last_gc >= self.config.gc_interval:
            return True

        # Run more frequently under memory pressure
        if memory_stats.pressure_level == MemoryPressureLevel.CRITICAL:
            return time_since_last_gc >= 5  # Every 5 seconds
        elif memory_stats.pressure_level == MemoryPressureLevel.HIGH:
            return time_since_last_gc >= 15  # Every 15 seconds
        elif memory_stats.pressure_level == MemoryPressureLevel.MEDIUM:
            return time_since_last_gc >= 30  # Every 30 seconds

        return False

    def run_gc(self, aggressive: bool = False) -> Dict[str, int]:
        """Run garbage collection with optional aggressive mode."""
        with self._lock:
            start_time = time.time()
            collected_objects = {"gen0": 0, "gen1": 0, "gen2": 0}

            try:
                if aggressive:
                    # Aggressive GC - collect all generations
                    collected_objects["gen0"] = gc.collect(0)
                    collected_objects["gen1"] = gc.collect(1)
                    collected_objects["gen2"] = gc.collect(2)

                    # Force collection of unreachable objects
                    gc.collect()
                else:
                    # Normal GC - collect generation 0 and 1
                    collected_objects["gen0"] = gc.collect(0)
                    collected_objects["gen1"] = gc.collect(1)

                total_collected = sum(collected_objects.values())
                gc_time = time.time() - start_time

                self._last_gc_time = time.time()
                self._gc_stats["collections"] += 1
                self._gc_stats["collected_objects"] += total_collected

                logger.debug(f"GC completed: collected {total_collected} objects in {gc_time:.3f}s")

                return {"total_collected": total_collected, "gc_time": gc_time, **collected_objects}

            except Exception as e:
                logger.error(f"Garbage collection failed: {e}")
                return {"total_collected": 0, "gc_time": 0, "error": str(e)}

    def get_gc_stats(self) -> Dict[str, Any]:
        """Get garbage collection statistics."""
        with self._lock:
            return {
                **self._gc_stats,
                "gc_counts": gc.get_count(),
                "gc_thresholds": gc.get_threshold(),
                "last_gc_time": self._last_gc_time,
            }

    def optimize_gc_thresholds(self, memory_stats: MemoryStats):
        """Optimize GC thresholds based on memory pressure."""
        current_thresholds = gc.get_threshold()

        if memory_stats.pressure_level == MemoryPressureLevel.CRITICAL:
            # More aggressive thresholds
            new_thresholds = (400, 5, 5)
        elif memory_stats.pressure_level == MemoryPressureLevel.HIGH:
            new_thresholds = (500, 8, 8)
        elif memory_stats.pressure_level == MemoryPressureLevel.MEDIUM:
            new_thresholds = (600, 10, 10)
        else:
            # Default/relaxed thresholds
            new_thresholds = (700, 10, 10)

        if new_thresholds != current_thresholds:
            gc.set_threshold(*new_thresholds)
            logger.debug(f"Updated GC thresholds: {new_thresholds}")


class MemoryManager:
    """Comprehensive memory management system."""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.stats = MemoryStats()
        self.tracker = MemoryTracker()
        self.gc_optimizer = GarbageCollectionOptimizer(config)

        self._cleanup_callbacks: List[Callable[[], None]] = []
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="memory-manager")

        # Start monitoring if enabled
        if config.enable_memory_monitoring:
            self._start_monitoring()

    def _start_monitoring(self):
        """Start memory monitoring thread."""

        def monitoring_loop():
            while not self._stop_monitoring.wait(10):  # Check every 10 seconds
                try:
                    self._update_memory_stats()
                    self._handle_memory_pressure()

                    # Run periodic cleanup
                    if time.time() % self.config.cleanup_interval < 10:
                        self._run_cleanup()

                except Exception as e:
                    logger.error(f"Memory monitoring error: {e}")

        self._monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self._monitoring_thread.start()

    def _update_memory_stats(self):
        """Update current memory statistics."""
        try:
            # System memory
            memory = psutil.virtual_memory()
            self.stats.total_memory_mb = memory.total / (1024 * 1024)
            self.stats.used_memory_mb = memory.used / (1024 * 1024)
            self.stats.available_memory_mb = memory.available / (1024 * 1024)
            self.stats.memory_percent = memory.percent

            # Process memory
            process = psutil.Process()
            process_memory = process.memory_info()
            self.stats.process_memory_mb = process_memory.rss / (1024 * 1024)
            self.stats.process_memory_percent = (
                self.stats.process_memory_mb / self.stats.total_memory_mb * 100
            )

            # Tracked objects
            self.stats.objects_tracked = len(self.tracker._tracked_objects)
            self.stats.large_objects_count = self.tracker.get_large_objects_count()
            self.stats.cache_memory_mb = self.tracker.get_tracked_memory() / (1024 * 1024)

            # GC stats
            self.stats.gc_collections = self.gc_optimizer._gc_stats["collections"]

            # Update pressure level
            self.stats.update_pressure_level(
                self.config.warning_threshold * 100, self.config.critical_threshold * 100
            )

        except Exception as e:
            logger.error(f"Failed to update memory stats: {e}")

    def _handle_memory_pressure(self):
        """Handle memory pressure based on current stats."""
        if self.stats.pressure_level == MemoryPressureLevel.CRITICAL:
            logger.warning("CRITICAL memory pressure detected!")
            self._emergency_cleanup()
        elif self.stats.pressure_level == MemoryPressureLevel.HIGH:
            logger.warning("HIGH memory pressure detected")
            self._aggressive_cleanup()
        elif self.stats.pressure_level == MemoryPressureLevel.MEDIUM:
            self._moderate_cleanup()

        # Run GC if needed
        if self.gc_optimizer.should_run_gc(self.stats):
            aggressive = self.stats.pressure_level in (
                MemoryPressureLevel.HIGH,
                MemoryPressureLevel.CRITICAL,
            )
            self.gc_optimizer.run_gc(aggressive)

    def _run_cleanup(self):
        """Run registered cleanup callbacks."""
        for callback in self._cleanup_callbacks[:]:  # Copy to avoid modification during iteration
            try:
                callback()
            except Exception as e:
                logger.error(f"Cleanup callback failed: {e}")

    def _moderate_cleanup(self):
        """Moderate cleanup actions."""
        self._run_cleanup()

        # Optimize GC thresholds
        self.gc_optimizer.optimize_gc_thresholds(self.stats)

    def _aggressive_cleanup(self):
        """Aggressive cleanup actions."""
        self._moderate_cleanup()

        # Force GC
        self.gc_optimizer.run_gc(aggressive=True)

    def _emergency_cleanup(self):
        """Emergency cleanup actions."""
        self._aggressive_cleanup()

        # Additional emergency measures could be added here
        # e.g., clearing caches, releasing non-essential resources
        logger.critical("Emergency memory cleanup executed")

    def register_cleanup_callback(self, callback: Callable[[], None]):
        """Register cleanup callback for memory pressure situations."""
        self._cleanup_callbacks.append(callback)

    def unregister_cleanup_callback(self, callback: Callable[[], None]):
        """Unregister cleanup callback."""
        if callback in self._cleanup_callbacks:
            self._cleanup_callbacks.remove(callback)

    def track_object(self, obj: Any, name: str, size_bytes: int = 0):
        """Track an object's memory usage."""
        self.tracker.track_object(obj, name, size_bytes)

    def untrack_object(self, obj: Any, name: str):
        """Stop tracking an object."""
        self.tracker.untrack_object(obj, name)

    def force_gc(self, aggressive: bool = False) -> Dict[str, int]:
        """Force garbage collection."""
        return self.gc_optimizer.run_gc(aggressive)

    def get_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        self._update_memory_stats()
        return self.stats

    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed memory and GC statistics."""
        self._update_memory_stats()
        return {
            "memory": {
                "total_mb": self.stats.total_memory_mb,
                "used_mb": self.stats.used_memory_mb,
                "available_mb": self.stats.available_memory_mb,
                "percent": self.stats.memory_percent,
                "pressure_level": self.stats.pressure_level.value,
            },
            "process": {
                "memory_mb": self.stats.process_memory_mb,
                "memory_percent": self.stats.process_memory_percent,
            },
            "tracking": {
                "objects_tracked": self.stats.objects_tracked,
                "large_objects": self.stats.large_objects_count,
                "cache_memory_mb": self.stats.cache_memory_mb,
            },
            "gc": self.gc_optimizer.get_gc_stats(),
        }

    def is_memory_available(self, required_mb: float) -> bool:
        """Check if enough memory is available for an operation."""
        self._update_memory_stats()
        return self.stats.available_memory_mb >= required_mb

    def wait_for_memory(self, required_mb: float, timeout: float = 30.0) -> bool:
        """Wait for memory to become available."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.is_memory_available(required_mb):
                return True

            # Try cleanup and GC
            self._run_cleanup()
            self.force_gc(aggressive=True)

            time.sleep(1)

        return False

    def shutdown(self):
        """Shutdown memory manager."""
        self._stop_monitoring.set()

        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=1)

        self._executor.shutdown(wait=False)


class MemoryOptimizedCache:
    """Cache with integrated memory management."""

    def __init__(self, memory_manager: MemoryManager, max_size_mb: float = 100):
        self.memory_manager = memory_manager
        self.max_size_mb = max_size_mb
        self._cache: Dict[str, Any] = {}
        self._access_times: Dict[str, float] = {}
        self._sizes: Dict[str, int] = {}
        self._lock = threading.RLock()
        self._total_size = 0

        # Register cleanup callback
        self.memory_manager.register_cleanup_callback(self._cleanup_on_pressure)

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key in self._cache:
                self._access_times[key] = time.time()
                return self._cache[key]
            return None

    def put(self, key: str, value: Any) -> bool:
        """Put value in cache."""
        size = self._estimate_size(value)

        # Check if we have enough memory
        if not self.memory_manager.is_memory_available(size / (1024 * 1024)):
            return False

        with self._lock:
            # Remove old value if exists
            if key in self._cache:
                self._total_size -= self._sizes[key]

            # Check size limits
            if self._total_size + size > self.max_size_mb * 1024 * 1024:
                self._evict_lru(size)

            # Add new value
            self._cache[key] = value
            self._access_times[key] = time.time()
            self._sizes[key] = size
            self._total_size += size

            return True

    def _evict_lru(self, needed_size: int):
        """Evict least recently used items."""
        # Sort by access time
        items_by_access = sorted(self._access_times.items(), key=lambda x: x[1])

        for key, _ in items_by_access:
            if self._total_size + needed_size <= self.max_size_mb * 1024 * 1024:
                break

            # Remove item
            self._cache.pop(key, None)
            self._access_times.pop(key, None)
            item_size = self._sizes.pop(key, 0)
            self._total_size -= item_size

    def _cleanup_on_pressure(self):
        """Cleanup cache during memory pressure."""
        with self._lock:
            if self.memory_manager.stats.pressure_level == MemoryPressureLevel.CRITICAL:
                # Remove 50% of cache
                items_to_remove = len(self._cache) // 2
            elif self.memory_manager.stats.pressure_level == MemoryPressureLevel.HIGH:
                # Remove 30% of cache
                items_to_remove = len(self._cache) // 3
            else:
                # Remove 10% of cache
                items_to_remove = len(self._cache) // 10

            if items_to_remove > 0:
                # Remove LRU items
                items_by_access = sorted(self._access_times.items(), key=lambda x: x[1])

                for key, _ in items_by_access[:items_to_remove]:
                    self._cache.pop(key, None)
                    self._access_times.pop(key, None)
                    item_size = self._sizes.pop(key, 0)
                    self._total_size -= item_size

    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size."""
        try:
            import sys

            return sys.getsizeof(obj)
        except:
            return 1024

    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._sizes.clear()
            self._total_size = 0
