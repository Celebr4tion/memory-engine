"""
Advanced cache manager with TTL, memory limits, and multi-level caching.
"""

import time
import hashlib
import threading
import gc
from typing import Any, Dict, Optional, Tuple, List, Callable
from dataclasses import dataclass
from collections import OrderedDict
import weakref
import asyncio
from concurrent.futures import ThreadPoolExecutor


@dataclass
class CacheConfig:
    """Configuration for cache manager."""
    max_memory_mb: int = 256
    default_ttl: int = 3600  # 1 hour
    max_items: int = 10000
    cleanup_interval: int = 300  # 5 minutes
    enable_compression: bool = True
    enable_stats: bool = True
    enable_async_cleanup: bool = True


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_usage: int = 0
    item_count: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class CacheEntry:
    """Cache entry with TTL and metadata."""
    
    def __init__(self, value: Any, ttl: int, size: int = 0):
        self.value = value
        self.created_at = time.time()
        self.ttl = ttl
        self.size = size
        self.access_count = 0
        self.last_accessed = self.created_at
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        return time.time() > (self.created_at + self.ttl)
    
    def access(self) -> Any:
        """Access the cached value and update stats."""
        self.access_count += 1
        self.last_accessed = time.time()
        return self.value


class CacheManager:
    """Advanced cache manager with multi-level caching and optimization."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.stats = CacheStats()
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=2)
        
        # Start cleanup thread
        if config.enable_async_cleanup:
            self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        def cleanup_worker():
            while not self._stop_cleanup.wait(self.config.cleanup_interval):
                try:
                    self._cleanup_expired()
                    if self._is_memory_exceeded():
                        self._evict_lru()
                except Exception as e:
                    # Log error but continue
                    pass
        
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self.stats.misses += 1
                return None
            
            if entry.is_expired():
                del self._cache[key]
                self.stats.misses += 1
                self.stats.evictions += 1
                return None
            
            # Move to end (LRU)
            self._cache.move_to_end(key)
            self.stats.hits += 1
            
            return entry.access()
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Put value in cache."""
        ttl = ttl or self.config.default_ttl
        size = self._estimate_size(value)
        
        with self._lock:
            # Check if we need to evict
            if self._should_evict(size):
                if not self._make_space(size):
                    return False
            
            entry = CacheEntry(value, ttl, size)
            self._cache[key] = entry
            self.stats.memory_usage += size
            self.stats.item_count += 1
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            entry = self._cache.pop(key, None)
            if entry:
                self.stats.memory_usage -= entry.size
                self.stats.item_count -= 1
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self.stats = CacheStats()
    
    def _cleanup_expired(self):
        """Remove expired entries."""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                entry = self._cache.pop(key)
                self.stats.memory_usage -= entry.size
                self.stats.item_count -= 1
                self.stats.evictions += 1
    
    def _evict_lru(self):
        """Evict least recently used entries."""
        target_memory = int(self.config.max_memory_mb * 1024 * 1024 * 0.8)
        
        with self._lock:
            while (self.stats.memory_usage > target_memory and 
                   len(self._cache) > 0):
                # Remove oldest item
                key, entry = self._cache.popitem(last=False)
                self.stats.memory_usage -= entry.size
                self.stats.item_count -= 1
                self.stats.evictions += 1
    
    def _should_evict(self, new_size: int) -> bool:
        """Check if eviction is needed."""
        max_memory = self.config.max_memory_mb * 1024 * 1024
        return (self.stats.memory_usage + new_size > max_memory or
                self.stats.item_count >= self.config.max_items)
    
    def _make_space(self, needed_size: int) -> bool:
        """Make space for new entry."""
        max_memory = self.config.max_memory_mb * 1024 * 1024
        
        # Try cleaning expired first
        self._cleanup_expired()
        
        # If still not enough space, evict LRU
        while (self.stats.memory_usage + needed_size > max_memory and
               len(self._cache) > 0):
            key, entry = self._cache.popitem(last=False)
            self.stats.memory_usage -= entry.size
            self.stats.item_count -= 1
            self.stats.evictions += 1
        
        return self.stats.memory_usage + needed_size <= max_memory
    
    def _is_memory_exceeded(self) -> bool:
        """Check if memory limit is exceeded."""
        max_memory = self.config.max_memory_mb * 1024 * 1024
        return self.stats.memory_usage > max_memory
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            import sys
            return sys.getsizeof(obj)
        except:
            # Fallback estimation
            if isinstance(obj, str):
                return len(obj.encode('utf-8'))
            elif isinstance(obj, (list, tuple)):
                return sum(self._estimate_size(item) for item in obj)
            elif isinstance(obj, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) 
                          for k, v in obj.items())
            else:
                return 1024  # Default estimate
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self.stats
    
    def optimize(self):
        """Run optimization routines."""
        with self._lock:
            # Clean expired entries
            self._cleanup_expired()
            
            # Trigger garbage collection
            gc.collect()
            
            # Compress cache if enabled
            if self.config.enable_compression:
                self._compress_cache()
    
    def _compress_cache(self):
        """Compress cache entries (placeholder for compression logic)."""
        # Could implement compression for large string values
        pass
    
    def shutdown(self):
        """Shutdown cache manager."""
        self._stop_cleanup.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=1)
        self._executor.shutdown(wait=False)


class MultiLevelCache:
    """Multi-level cache with L1 (memory) and L2 (disk) tiers."""
    
    def __init__(self, l1_config: CacheConfig, l2_config: Optional[CacheConfig] = None):
        self.l1_cache = CacheManager(l1_config)
        self.l2_cache = CacheManager(l2_config) if l2_config else None
        self._lock = threading.RLock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from multi-level cache."""
        # Try L1 first
        value = self.l1_cache.get(key)
        if value is not None:
            return value
        
        # Try L2 if available
        if self.l2_cache:
            value = self.l2_cache.get(key)
            if value is not None:
                # Promote to L1
                self.l1_cache.put(key, value)
                return value
        
        return None
    
    async def put(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Put value in multi-level cache."""
        # Store in L1
        l1_success = self.l1_cache.put(key, value, ttl)
        
        # Store in L2 if available
        l2_success = True
        if self.l2_cache:
            l2_success = self.l2_cache.put(key, value, ttl)
        
        return l1_success or l2_success
    
    def get_combined_stats(self) -> Dict[str, CacheStats]:
        """Get combined statistics."""
        stats = {'l1': self.l1_cache.get_stats()}
        if self.l2_cache:
            stats['l2'] = self.l2_cache.get_stats()
        return stats
    
    def shutdown(self):
        """Shutdown multi-level cache."""
        self.l1_cache.shutdown()
        if self.l2_cache:
            self.l2_cache.shutdown()


class QueryResultCache:
    """Specialized cache for query results with intelligent invalidation."""
    
    def __init__(self, config: CacheConfig):
        self.cache_manager = CacheManager(config)
        self._invalidation_patterns: Dict[str, List[str]] = {}
        self._lock = threading.RLock()
    
    def get_query_result(self, query: str, params: Dict[str, Any]) -> Optional[Any]:
        """Get cached query result."""
        cache_key = self._generate_query_key(query, params)
        return self.cache_manager.get(cache_key)
    
    def put_query_result(self, query: str, params: Dict[str, Any], 
                        result: Any, ttl: Optional[int] = None) -> bool:
        """Cache query result."""
        cache_key = self._generate_query_key(query, params)
        return self.cache_manager.put(cache_key, result, ttl)
    
    def invalidate_pattern(self, pattern: str):
        """Invalidate all cached results matching pattern."""
        with self._lock:
            keys_to_invalidate = []
            for key in self.cache_manager._cache.keys():
                if pattern in key:
                    keys_to_invalidate.append(key)
            
            for key in keys_to_invalidate:
                self.cache_manager.delete(key)
    
    def _generate_query_key(self, query: str, params: Dict[str, Any]) -> str:
        """Generate cache key for query."""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        params_hash = hashlib.md5(str(sorted(params.items())).encode()).hexdigest()
        return f"query:{query_hash}:{params_hash}"
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self.cache_manager.get_stats()
    
    def shutdown(self):
        """Shutdown query result cache."""
        self.cache_manager.shutdown()