"""
Query Result Caching System for Advanced Query Engine

Provides intelligent caching of query results with TTL, LRU eviction, and cache invalidation.
"""

import hashlib
import json
import logging
import pickle
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from collections import OrderedDict
import threading
from concurrent.futures import ThreadPoolExecutor

from .query_types import QueryRequest, QueryResponse, QueryStatistics


@dataclass
class CacheEntry:
    """Represents a cached query result."""
    query_hash: str
    response: QueryResponse
    created_at: float
    last_accessed: float
    access_count: int
    ttl: int
    size_bytes: int
    
    @property
    def is_expired(self) -> bool:
        """Check if the cache entry is expired."""
        return time.time() - self.created_at > self.ttl
    
    @property
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds."""
        return time.time() - self.created_at


@dataclass
class CacheStatistics:
    """Statistics about cache performance."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    entry_count: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return (self.hits / total) if total > 0 else 0.0


class QueryCache:
    """
    Intelligent query result caching system.
    
    Features:
    - TTL-based cache expiration
    - LRU eviction policy
    - Memory usage tracking and limits
    - Thread-safe operations
    - Cache warming and preloading
    - Intelligent cache key generation
    - Cache invalidation patterns
    """
    
    def __init__(self, 
                 max_size_mb: int = 256,
                 default_ttl: int = 3600,
                 max_entries: int = 10000,
                 cleanup_interval: int = 300):
        """
        Initialize the query cache.
        
        Args:
            max_size_mb: Maximum cache size in megabytes
            default_ttl: Default TTL in seconds
            max_entries: Maximum number of cache entries
            cleanup_interval: Cleanup interval in seconds
        """
        self.logger = logging.getLogger(__name__)
        
        # Cache configuration
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.max_entries = max_entries
        self.cleanup_interval = cleanup_interval
        
        # Cache storage (OrderedDict for LRU behavior)
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = CacheStatistics()
        
        # Background cleanup
        self._cleanup_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cache-cleanup")
        self._last_cleanup = time.time()
        
        # Cache invalidation patterns
        self.invalidation_patterns = set()
        
        self.logger.info(f"Query cache initialized: max_size={max_size_mb}MB, max_entries={max_entries}, ttl={default_ttl}s")
    
    def get(self, request: QueryRequest) -> Optional[QueryResponse]:
        """
        Get cached query result.
        
        Args:
            request: Query request
            
        Returns:
            Cached query response or None if not found/expired
        """
        if not request.use_cache:
            return None
        
        cache_key = self._generate_cache_key(request)
        
        with self._lock:
            entry = self._cache.get(cache_key)
            
            if entry is None:
                self.stats.misses += 1
                return None
            
            # Check if expired
            if entry.is_expired:
                self._remove_entry(cache_key)
                self.stats.misses += 1
                return None
            
            # Update access statistics
            entry.last_accessed = time.time()
            entry.access_count += 1
            
            # Move to end (most recently used)
            self._cache.move_to_end(cache_key)
            
            self.stats.hits += 1
            
            # Clone response and mark as from cache
            cached_response = self._clone_response(entry.response)
            cached_response.from_cache = True
            cached_response.query_id = f"cached_{cache_key[:8]}"
            
            self.logger.debug(f"Cache hit for query: {request.query[:50]}...")
            return cached_response
    
    def put(self, request: QueryRequest, response: QueryResponse) -> bool:
        """
        Store query result in cache.
        
        Args:
            request: Query request
            response: Query response to cache
            
        Returns:
            True if successfully cached, False otherwise
        """
        if not request.use_cache or self._should_skip_caching(request, response):
            return False
        
        cache_key = self._generate_cache_key(request)
        ttl = request.cache_ttl or self.default_ttl
        
        # Estimate size
        estimated_size = self._estimate_response_size(response)
        
        # Skip if response is too large
        if estimated_size > self.max_size_bytes // 10:  # Don't cache if > 10% of total cache
            self.logger.warning(f"Skipping cache for large response: {estimated_size} bytes")
            return False
        
        with self._lock:
            # Remove existing entry if present
            if cache_key in self._cache:
                self._remove_entry(cache_key)
            
            # Ensure we have space
            self._ensure_cache_space(estimated_size)
            
            # Create cache entry
            entry = CacheEntry(
                query_hash=cache_key,
                response=self._clone_response(response),
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=0,
                ttl=ttl,
                size_bytes=estimated_size
            )
            
            # Store in cache
            self._cache[cache_key] = entry
            
            # Update statistics
            self.stats.size_bytes += estimated_size
            self.stats.entry_count += 1
            
            self.logger.debug(f"Cached query result: {cache_key[:8]} (size: {estimated_size} bytes, TTL: {ttl}s)")
            
            # Trigger cleanup if needed
            self._maybe_cleanup()
            
            return True
    
    def invalidate(self, pattern: Optional[str] = None, node_ids: Optional[List[str]] = None):
        """
        Invalidate cache entries based on patterns or affected nodes.
        
        Args:
            pattern: Pattern to match cache keys
            node_ids: List of node IDs that were modified
        """
        with self._lock:
            keys_to_remove = []
            
            if pattern:
                # Match cache keys by pattern
                for cache_key in self._cache.keys():
                    if pattern in cache_key:
                        keys_to_remove.append(cache_key)
            
            if node_ids:
                # Invalidate entries that might contain these nodes
                # This is a simplified approach - in practice, you'd track dependencies
                for cache_key in self._cache.keys():
                    entry = self._cache[cache_key]
                    # Check if any result contains the modified nodes
                    for result in entry.response.results:
                        if result.node_id in node_ids:
                            keys_to_remove.append(cache_key)
                            break
            
            # Remove identified entries
            for key in set(keys_to_remove):
                self._remove_entry(key)
                self.stats.evictions += 1
            
            if keys_to_remove:
                self.logger.info(f"Invalidated {len(keys_to_remove)} cache entries")
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            entry_count = len(self._cache)
            self._cache.clear()
            self.stats = CacheStatistics()
            self.logger.info(f"Cache cleared: {entry_count} entries removed")
    
    def _generate_cache_key(self, request: QueryRequest) -> str:
        """
        Generate a cache key for the query request using efficient method.
        
        Args:
            request: Query request
            
        Returns:
            Hash-based cache key
        """
        # Create key components for efficient hashing
        key_components = [
            request.query.strip().lower(),
            request.query_type.value,
            str(request.limit),
            str(request.offset),
            str(request.similarity_threshold),
            str(request.max_depth),
            str(request.include_metadata),
            str(request.include_relationships),
            str(request.include_embeddings)
        ]
        
        # Add filters (sorted for consistency)
        if request.filters:
            filter_strs = []
            for f in sorted(request.filters, key=lambda x: (x.field, x.operator)):
                filter_strs.append(f"{f.field}:{f.operator}:{f.value}")
            key_components.append('|'.join(filter_strs))
        
        # Add sort criteria
        if request.sort_by:
            sort_strs = [f"{s.field}:{s.order.value}" for s in request.sort_by]
            key_components.append('|'.join(sort_strs))
        
        # Add aggregations
        if request.aggregations:
            agg_strs = []
            for a in sorted(request.aggregations, key=lambda x: x.type.value):
                agg_strs.append(f"{a.type.value}:{a.field}")
            key_components.append('|'.join(agg_strs))
        
        # Add context if present
        if request.context:
            key_components.append(request.context)
        
        # Generate hash
        cache_str = '||'.join(str(c) for c in key_components)
        return hashlib.sha256(cache_str.encode()).hexdigest()
    
    def _should_skip_caching(self, request: QueryRequest, response: QueryResponse) -> bool:
        """
        Determine if a query result should be skipped from caching.
        
        Args:
            request: Query request
            response: Query response
            
        Returns:
            True if should skip caching
        """
        # Skip if no results
        if not response.results:
            return True
        
        # Skip if query took too little time (likely from index)
        if response.execution_time_ms < 10:
            return True
        
        # Skip if response is error or incomplete
        if hasattr(response, 'error') and response.error:
            return True
        
        # Skip certain query types
        if request.query_type.value in ['aggregation']:
            # Aggregations might change frequently
            return False  # Actually, cache aggregations too
        
        return False
    
    def _estimate_response_size(self, response: QueryResponse) -> int:
        """
        Estimate the size of a query response in bytes using efficient calculation.
        
        Args:
            response: Query response
            
        Returns:
            Estimated size in bytes
        """
        size = 200  # Base response overhead
        
        for result in response.results:
            # Content size (UTF-8 encoding)
            size += len(result.content.encode('utf-8'))
            
            # Node metadata
            if result.metadata:
                size += len(str(result.metadata).encode('utf-8'))
            
            # Relationships (estimated)
            if result.relationships:
                size += len(result.relationships) * 150  # Average relationship size
            
            # Other fields
            size += len(result.node_id) * 2
            size += 50  # Other fields overhead
        
        # Aggregations
        for agg in response.aggregations:
            size += 100  # Average aggregation size
        
        return size
    
    def _clone_response(self, response: QueryResponse) -> QueryResponse:
        """
        Create a deep copy of query response.
        
        Args:
            response: Original response
            
        Returns:
            Cloned response
        """
        try:
            return pickle.loads(pickle.dumps(response))
        except Exception as e:
            self.logger.error(f"Failed to clone response: {e}")
            return response
    
    def _ensure_cache_space(self, needed_bytes: int):
        """
        Ensure there's enough space in cache by evicting entries if needed.
        
        Args:
            needed_bytes: Number of bytes needed
        """
        # Check if we need to make space
        while (self.stats.size_bytes + needed_bytes > self.max_size_bytes or 
               len(self._cache) >= self.max_entries):
            
            if not self._cache:
                break
            
            # Remove least recently used entry
            lru_key = next(iter(self._cache))
            self._remove_entry(lru_key)
            self.stats.evictions += 1
    
    def _remove_entry(self, cache_key: str):
        """
        Remove a cache entry and update statistics.
        
        Args:
            cache_key: Cache key to remove
        """
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            self.stats.size_bytes -= entry.size_bytes
            self.stats.entry_count -= 1
            del self._cache[cache_key]
    
    def _maybe_cleanup(self):
        """
        Trigger background cleanup if needed.
        """
        current_time = time.time()
        if current_time - self._last_cleanup > self.cleanup_interval:
            self._cleanup_executor.submit(self._cleanup_expired)
            self._last_cleanup = current_time
    
    def _cleanup_expired(self):
        """
        Remove expired entries from cache.
        """
        with self._lock:
            expired_keys = []
            current_time = time.time()
            
            for cache_key, entry in self._cache.items():
                if entry.is_expired:
                    expired_keys.append(cache_key)
            
            for key in expired_keys:
                self._remove_entry(key)
                self.stats.evictions += 1
            
            if expired_keys:
                self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            # Calculate additional metrics
            total_requests = self.stats.hits + self.stats.misses
            avg_entry_size = (self.stats.size_bytes / self.stats.entry_count) if self.stats.entry_count > 0 else 0
            
            # Get cache age distribution
            ages = []
            access_counts = []
            for entry in self._cache.values():
                ages.append(entry.age_seconds)
                access_counts.append(entry.access_count)
            
            return {
                'hit_rate': self.stats.hit_rate,
                'total_requests': total_requests,
                'cache_hits': self.stats.hits,
                'cache_misses': self.stats.misses,
                'evictions': self.stats.evictions,
                'current_size_mb': self.stats.size_bytes / (1024 * 1024),
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'entry_count': self.stats.entry_count,
                'max_entries': self.max_entries,
                'avg_entry_size_kb': avg_entry_size / 1024,
                'oldest_entry_age_hours': max(ages) / 3600 if ages else 0,
                'avg_access_count': sum(access_counts) / len(access_counts) if access_counts else 0
            }
    
    def warm_cache(self, popular_queries: List[QueryRequest]):
        """
        Warm the cache with popular queries.
        
        Args:
            popular_queries: List of popular query requests to pre-execute
        """
        self.logger.info(f"Warming cache with {len(popular_queries)} popular queries")
        # This would be implemented by the query engine to pre-execute popular queries
        # For now, just log the intent
        pass
    
    def shutdown(self):
        """Clean shutdown of cache resources."""
        self._cleanup_executor.shutdown(wait=True)
        self.clear()
        self.logger.info("Query cache shutdown complete")