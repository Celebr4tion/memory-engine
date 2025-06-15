"""
Comprehensive health checking system for Memory Engine components.
"""

import asyncio
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Awaitable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    component: str
    status: HealthStatus
    message: str
    response_time: float
    timestamp: float
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class HealthCheck:
    """Base health check class."""
    
    def __init__(self, name: str, timeout: float = 10.0, 
                 critical: bool = True, interval: float = 30.0):
        self.name = name
        self.timeout = timeout
        self.critical = critical
        self.interval = interval
        self.last_result: Optional[HealthCheckResult] = None
        self.consecutive_failures = 0
        self.max_failures = 3
    
    async def check(self) -> HealthCheckResult:
        """Perform health check."""
        start_time = time.time()
        
        try:
            # Run the actual health check with timeout
            is_healthy = await asyncio.wait_for(
                self._check_health(),
                timeout=self.timeout
            )
            
            response_time = time.time() - start_time
            
            if is_healthy:
                self.consecutive_failures = 0
                status = HealthStatus.HEALTHY
                message = f"{self.name} is healthy"
            else:
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.max_failures:
                    status = HealthStatus.UNHEALTHY
                    message = f"{self.name} has failed {self.consecutive_failures} consecutive checks"
                else:
                    status = HealthStatus.DEGRADED
                    message = f"{self.name} is degraded ({self.consecutive_failures} failures)"
            
            result = HealthCheckResult(
                component=self.name,
                status=status,
                message=message,
                response_time=response_time,
                timestamp=time.time(),
                details=await self._get_details()
            )
            
        except asyncio.TimeoutError:
            self.consecutive_failures += 1
            result = HealthCheckResult(
                component=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"{self.name} health check timed out after {self.timeout}s",
                response_time=self.timeout,
                timestamp=time.time()
            )
            
        except Exception as e:
            self.consecutive_failures += 1
            result = HealthCheckResult(
                component=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"{self.name} health check failed: {str(e)}",
                response_time=time.time() - start_time,
                timestamp=time.time(),
                details={"error": str(e)}
            )
        
        self.last_result = result
        return result
    
    async def _check_health(self) -> bool:
        """Override this method to implement actual health check."""
        raise NotImplementedError("Subclasses must implement _check_health")
    
    async def _get_details(self) -> Dict[str, Any]:
        """Get additional health check details."""
        return {}


class DatabaseHealthCheck(HealthCheck):
    """Health check for database connections."""
    
    def __init__(self, name: str, storage_backend: Any, **kwargs):
        super().__init__(name, **kwargs)
        self.storage_backend = storage_backend
    
    async def _check_health(self) -> bool:
        """Check database connectivity and responsiveness."""
        try:
            # Test basic connectivity
            if hasattr(self.storage_backend, 'test_connection'):
                return await self.storage_backend.test_connection()
            
            # Fallback to simple query
            if hasattr(self.storage_backend, 'execute_query'):
                result = await self.storage_backend.execute_query("SELECT 1")
                return result is not None
            
            return True
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    async def _get_details(self) -> Dict[str, Any]:
        """Get database-specific details."""
        details = {}
        
        try:
            if hasattr(self.storage_backend, 'get_connection_info'):
                details['connection_info'] = await self.storage_backend.get_connection_info()
            
            if hasattr(self.storage_backend, 'get_stats'):
                details['stats'] = await self.storage_backend.get_stats()
                
        except Exception as e:
            details['error'] = str(e)
        
        return details


class LLMProviderHealthCheck(HealthCheck):
    """Health check for LLM providers."""
    
    def __init__(self, name: str, llm_provider: Any, **kwargs):
        super().__init__(name, **kwargs)
        self.llm_provider = llm_provider
    
    async def _check_health(self) -> bool:
        """Check LLM provider availability and responsiveness."""
        try:
            if hasattr(self.llm_provider, 'health_check'):
                health_result = await self.llm_provider.health_check()
                return health_result.get('connected', False)
            
            # Fallback to simple completion test
            if hasattr(self.llm_provider, 'generate_completion'):
                response = await self.llm_provider.generate_completion(
                    "test", max_tokens=1
                )
                return response is not None
            
            return True
            
        except Exception as e:
            logger.error(f"LLM provider health check failed: {e}")
            return False
    
    async def _get_details(self) -> Dict[str, Any]:
        """Get LLM provider specific details."""
        details = {}
        
        try:
            if hasattr(self.llm_provider, 'get_provider_info'):
                details['provider_info'] = await self.llm_provider.get_provider_info()
            
            if hasattr(self.llm_provider, 'get_model_info'):
                details['model_info'] = await self.llm_provider.get_model_info()
                
        except Exception as e:
            details['error'] = str(e)
        
        return details


class EmbeddingProviderHealthCheck(HealthCheck):
    """Health check for embedding providers."""
    
    def __init__(self, name: str, embedding_provider: Any, **kwargs):
        super().__init__(name, **kwargs)
        self.embedding_provider = embedding_provider
    
    async def _check_health(self) -> bool:
        """Check embedding provider availability."""
        try:
            if hasattr(self.embedding_provider, 'test_connection'):
                return await self.embedding_provider.test_connection()
            
            # Fallback to simple embedding test
            if hasattr(self.embedding_provider, 'generate_embedding'):
                embedding = await self.embedding_provider.generate_embedding("test")
                return embedding is not None
            
            return True
            
        except Exception as e:
            logger.error(f"Embedding provider health check failed: {e}")
            return False


class VectorStoreHealthCheck(HealthCheck):
    """Health check for vector stores."""
    
    def __init__(self, name: str, vector_store: Any, **kwargs):
        super().__init__(name, **kwargs)
        self.vector_store = vector_store
    
    async def _check_health(self) -> bool:
        """Check vector store connectivity and operations."""
        try:
            if hasattr(self.vector_store, 'test_connection'):
                return await self.vector_store.test_connection()
            
            # Test basic operations
            if hasattr(self.vector_store, 'search'):
                # Simple search test
                results = await self.vector_store.search([0.1] * 384, k=1)
                return True  # If no exception, it's working
            
            return True
            
        except Exception as e:
            logger.error(f"Vector store health check failed: {e}")
            return False


class MemoryHealthCheck(HealthCheck):
    """Health check for memory usage."""
    
    def __init__(self, name: str, memory_manager: Any = None, **kwargs):
        super().__init__(name, **kwargs)
        self.memory_manager = memory_manager
    
    async def _check_health(self) -> bool:
        """Check memory usage and pressure."""
        try:
            if self.memory_manager:
                stats = self.memory_manager.get_stats()
                # Consider unhealthy if memory usage > 90%
                return stats.memory_percent < 90.0
            
            # Fallback to basic memory check
            import psutil
            memory = psutil.virtual_memory()
            return memory.percent < 90.0
            
        except Exception as e:
            logger.error(f"Memory health check failed: {e}")
            return False
    
    async def _get_details(self) -> Dict[str, Any]:
        """Get memory usage details."""
        details = {}
        
        try:
            if self.memory_manager:
                details['memory_stats'] = self.memory_manager.get_detailed_stats()
            else:
                import psutil
                memory = psutil.virtual_memory()
                details['memory_stats'] = {
                    'total_mb': memory.total / (1024 * 1024),
                    'used_mb': memory.used / (1024 * 1024),
                    'available_mb': memory.available / (1024 * 1024),
                    'percent': memory.percent
                }
                
        except Exception as e:
            details['error'] = str(e)
        
        return details


class CustomHealthCheck(HealthCheck):
    """Custom health check with user-defined check function."""
    
    def __init__(self, name: str, check_function: Callable[[], Awaitable[bool]], 
                 details_function: Optional[Callable[[], Awaitable[Dict[str, Any]]]] = None,
                 **kwargs):
        super().__init__(name, **kwargs)
        self.check_function = check_function
        self.details_function = details_function
    
    async def _check_health(self) -> bool:
        """Execute custom health check function."""
        return await self.check_function()
    
    async def _get_details(self) -> Dict[str, Any]:
        """Get custom health check details."""
        if self.details_function:
            return await self.details_function()
        return {}


class HealthChecker:
    """Central health checking coordinator."""
    
    def __init__(self, auto_start: bool = True, check_interval: float = 30.0):
        self.check_interval = check_interval
        self._checks: Dict[str, HealthCheck] = {}
        self._results: Dict[str, HealthCheckResult] = {}
        self._lock = asyncio.Lock()
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        
        if auto_start:
            self.start_monitoring()
    
    def register_check(self, health_check: HealthCheck):
        """Register a health check."""
        self._checks[health_check.name] = health_check
    
    def unregister_check(self, name: str):
        """Unregister a health check."""
        self._checks.pop(name, None)
        self._results.pop(name, None)
    
    async def run_check(self, name: str) -> Optional[HealthCheckResult]:
        """Run a specific health check."""
        check = self._checks.get(name)
        if not check:
            return None
        
        result = await check.check()
        
        async with self._lock:
            self._results[name] = result
        
        return result
    
    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        if not self._checks:
            return {}
        
        # Run checks concurrently
        tasks = [
            self.run_check(name) for name in self._checks.keys()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        final_results = {}
        for name, result in zip(self._checks.keys(), results):
            if isinstance(result, Exception):
                final_results[name] = HealthCheckResult(
                    component=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed with exception: {str(result)}",
                    response_time=0.0,
                    timestamp=time.time()
                )
            elif result:
                final_results[name] = result
        
        return final_results
    
    def get_last_results(self) -> Dict[str, HealthCheckResult]:
        """Get the last health check results."""
        return dict(self._results)
    
    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status."""
        if not self._results:
            return HealthStatus.UNKNOWN
        
        statuses = [result.status for result in self._results.values()]
        
        # If any critical component is unhealthy, system is unhealthy
        critical_checks = [
            check for check in self._checks.values() if check.critical
        ]
        
        for check in critical_checks:
            result = self._results.get(check.name)
            if result and result.status == HealthStatus.UNHEALTHY:
                return HealthStatus.UNHEALTHY
        
        # If any component is degraded, system is degraded
        if HealthStatus.DEGRADED in statuses or HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.DEGRADED
        
        # If all components are healthy
        if all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        
        return HealthStatus.UNKNOWN
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""
        results = self.get_last_results()
        overall_status = self.get_overall_status()
        
        summary = {
            'overall_status': overall_status.value,
            'timestamp': time.time(),
            'total_checks': len(self._checks),
            'checks': {}
        }
        
        # Count by status
        status_counts = {status.value: 0 for status in HealthStatus}
        
        for name, result in results.items():
            summary['checks'][name] = {
                'status': result.status.value,
                'message': result.message,
                'response_time': result.response_time,
                'timestamp': result.timestamp,
                'critical': self._checks[name].critical,
                'details': result.details
            }
            status_counts[result.status.value] += 1
        
        summary['status_counts'] = status_counts
        
        return summary
    
    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self._running:
            return
        
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self):
        """Stop continuous health monitoring."""
        self._running = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
    
    async def _monitoring_loop(self):
        """Continuous monitoring loop."""
        while self._running:
            try:
                await self.run_all_checks()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring loop error: {e}")
                await asyncio.sleep(5)  # Brief pause before retrying