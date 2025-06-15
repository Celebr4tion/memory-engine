"""
Health check endpoints for Memory Engine system monitoring.

This module provides comprehensive health checks for all system components
including databases, external services, and internal subsystems.
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta, UTC
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass, field
import psutil
import httpx

from memory_core.config import get_config


class HealthStatus(Enum):
    """Health check status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Individual health check result."""

    component: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    response_time_ms: float = 0.0


@dataclass
class SystemHealthReport:
    """Complete system health report."""

    overall_status: HealthStatus
    timestamp: datetime
    checks: List[HealthCheckResult]
    summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_healthy(self) -> bool:
        """Check if system is healthy."""
        return self.overall_status == HealthStatus.HEALTHY

    @property
    def unhealthy_components(self) -> List[str]:
        """Get list of unhealthy components."""
        return [check.component for check in self.checks if check.status == HealthStatus.UNHEALTHY]


class HealthChecker:
    """
    Comprehensive health checker for Memory Engine components.

    Monitors the health of:
    - Database connections (JanusGraph, Milvus)
    - External APIs (Gemini, OpenAI)
    - System resources (CPU, memory, disk)
    - Internal services (embedding manager, query engine)
    """

    def __init__(self):
        """Initialize the health checker."""
        self.logger = logging.getLogger(__name__)
        self.config = get_config()
        self._last_check: Optional[SystemHealthReport] = None
        self._check_cache_duration = timedelta(seconds=30)  # Cache for 30 seconds

    async def check_system_health(self, force_refresh: bool = False) -> SystemHealthReport:
        """
        Perform comprehensive system health check.

        Args:
            force_refresh: Force fresh check ignoring cache

        Returns:
            SystemHealthReport with complete health status
        """
        # Use cached result if available and recent
        if (
            not force_refresh
            and self._last_check
            and datetime.now(UTC) - self._last_check.timestamp < self._check_cache_duration
        ):
            return self._last_check

        start_time = time.time()

        # Run all health checks concurrently
        check_tasks = [
            self._check_janusgraph(),
            self._check_milvus(),
            self._check_gemini_api(),
            self._check_system_resources(),
            self._check_memory_usage(),
            self._check_disk_space(),
        ]

        self.logger.info("Starting comprehensive health check")

        try:
            results = await asyncio.gather(*check_tasks, return_exceptions=True)

            # Convert exceptions to unhealthy results
            checks = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    component_name = [
                        "janusgraph",
                        "milvus",
                        "gemini_api",
                        "system_resources",
                        "memory",
                        "disk",
                    ][i]
                    checks.append(
                        HealthCheckResult(
                            component=component_name,
                            status=HealthStatus.UNHEALTHY,
                            message=f"Health check failed: {str(result)}",
                            details={"error": str(result)},
                        )
                    )
                else:
                    checks.append(result)

            # Determine overall status
            overall_status = self._calculate_overall_status(checks)

            # Calculate performance summary
            total_time_ms = (time.time() - start_time) * 1000
            summary = {
                "total_checks": len(checks),
                "healthy_count": len([c for c in checks if c.status == HealthStatus.HEALTHY]),
                "degraded_count": len([c for c in checks if c.status == HealthStatus.DEGRADED]),
                "unhealthy_count": len([c for c in checks if c.status == HealthStatus.UNHEALTHY]),
                "check_duration_ms": total_time_ms,
            }

            report = SystemHealthReport(
                overall_status=overall_status,
                timestamp=datetime.now(UTC),
                checks=checks,
                summary=summary,
            )

            self._last_check = report
            self.logger.info(
                f"Health check completed in {total_time_ms:.2f}ms - Status: {overall_status.value}"
            )

            return report

        except Exception as e:
            self.logger.error(f"Health check system error: {e}")
            return SystemHealthReport(
                overall_status=HealthStatus.UNHEALTHY,
                timestamp=datetime.now(UTC),
                checks=[
                    HealthCheckResult(
                        component="health_checker",
                        status=HealthStatus.UNHEALTHY,
                        message=f"Health check system error: {str(e)}",
                    )
                ],
            )

    async def _check_janusgraph(self) -> HealthCheckResult:
        """Check JanusGraph database connectivity."""
        start_time = time.time()

        try:
            from memory_core.db.janusgraph_storage import JanusGraphStorage

            # Test connection
            storage = JanusGraphStorage()
            connected = await storage.test_connection()

            response_time = (time.time() - start_time) * 1000

            if connected:
                return HealthCheckResult(
                    component="janusgraph",
                    status=HealthStatus.HEALTHY,
                    message="JanusGraph connection successful",
                    details={
                        "host": self.config.config.janusgraph.host,
                        "port": self.config.config.janusgraph.port,
                    },
                    response_time_ms=response_time,
                )
            else:
                return HealthCheckResult(
                    component="janusgraph",
                    status=HealthStatus.UNHEALTHY,
                    message="JanusGraph connection failed",
                    response_time_ms=response_time,
                )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component="janusgraph",
                status=HealthStatus.UNHEALTHY,
                message=f"JanusGraph check error: {str(e)}",
                details={"error": str(e)},
                response_time_ms=response_time,
            )

    async def _check_milvus(self) -> HealthCheckResult:
        """Check Milvus vector database connectivity."""
        start_time = time.time()

        try:
            from memory_core.embeddings.vector_store import VectorStoreMilvus

            # Test connection
            vector_store = VectorStoreMilvus()
            connected = vector_store.connect()

            response_time = (time.time() - start_time) * 1000

            if connected:
                # Test basic operations
                try:
                    collections = vector_store.client.list_collections()
                    return HealthCheckResult(
                        component="milvus",
                        status=HealthStatus.HEALTHY,
                        message="Milvus connection and operations successful",
                        details={
                            "host": self.config.config.milvus.host,
                            "port": self.config.config.milvus.port,
                            "collections_count": len(collections),
                        },
                        response_time_ms=response_time,
                    )
                except Exception as op_error:
                    return HealthCheckResult(
                        component="milvus",
                        status=HealthStatus.DEGRADED,
                        message=f"Milvus connected but operations limited: {str(op_error)}",
                        response_time_ms=response_time,
                    )
            else:
                return HealthCheckResult(
                    component="milvus",
                    status=HealthStatus.UNHEALTHY,
                    message="Milvus connection failed",
                    response_time_ms=response_time,
                )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component="milvus",
                status=HealthStatus.UNHEALTHY,
                message=f"Milvus check error: {str(e)}",
                details={"error": str(e)},
                response_time_ms=response_time,
            )

    async def _check_gemini_api(self) -> HealthCheckResult:
        """Check Gemini API connectivity and quota."""
        start_time = time.time()

        try:
            api_key = self.config.config.api.google_api_key
            if not api_key:
                return HealthCheckResult(
                    component="gemini_api",
                    status=HealthStatus.DEGRADED,
                    message="Gemini API key not configured",
                    response_time_ms=(time.time() - start_time) * 1000,
                )

            # Simple API test (lightweight request)
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent",
                    headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
                    json={"contents": [{"parts": [{"text": "Hi"}]}]},
                    timeout=10.0,
                )

                response_time = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    return HealthCheckResult(
                        component="gemini_api",
                        status=HealthStatus.HEALTHY,
                        message="Gemini API responding normally",
                        details={"api_available": True},
                        response_time_ms=response_time,
                    )
                elif response.status_code == 429:
                    return HealthCheckResult(
                        component="gemini_api",
                        status=HealthStatus.DEGRADED,
                        message="Gemini API rate limited",
                        details={"status_code": response.status_code},
                        response_time_ms=response_time,
                    )
                else:
                    return HealthCheckResult(
                        component="gemini_api",
                        status=HealthStatus.UNHEALTHY,
                        message=f"Gemini API error: {response.status_code}",
                        details={"status_code": response.status_code},
                        response_time_ms=response_time,
                    )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component="gemini_api",
                status=HealthStatus.UNHEALTHY,
                message=f"Gemini API check error: {str(e)}",
                details={"error": str(e)},
                response_time_ms=response_time,
            )

    async def _check_system_resources(self) -> HealthCheckResult:
        """Check system CPU and load."""
        start_time = time.time()

        try:
            # Get CPU usage over 1 second
            cpu_percent = psutil.cpu_percent(interval=1)
            load_avg = psutil.getloadavg()
            cpu_count = psutil.cpu_count()

            response_time = (time.time() - start_time) * 1000

            # Determine status based on CPU usage
            if cpu_percent < 70:
                status = HealthStatus.HEALTHY
                message = "System resources normal"
            elif cpu_percent < 90:
                status = HealthStatus.DEGRADED
                message = "System CPU usage elevated"
            else:
                status = HealthStatus.UNHEALTHY
                message = "System CPU usage critical"

            return HealthCheckResult(
                component="system_resources",
                status=status,
                message=message,
                details={
                    "cpu_percent": cpu_percent,
                    "load_1m": load_avg[0],
                    "load_5m": load_avg[1],
                    "load_15m": load_avg[2],
                    "cpu_count": cpu_count,
                },
                response_time_ms=response_time,
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component="system_resources",
                status=HealthStatus.UNHEALTHY,
                message=f"System resources check error: {str(e)}",
                details={"error": str(e)},
                response_time_ms=response_time,
            )

    async def _check_memory_usage(self) -> HealthCheckResult:
        """Check system memory usage."""
        start_time = time.time()

        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()

            response_time = (time.time() - start_time) * 1000

            # Determine status based on memory usage
            if memory.percent < 80:
                status = HealthStatus.HEALTHY
                message = "Memory usage normal"
            elif memory.percent < 95:
                status = HealthStatus.DEGRADED
                message = "Memory usage elevated"
            else:
                status = HealthStatus.UNHEALTHY
                message = "Memory usage critical"

            return HealthCheckResult(
                component="memory",
                status=status,
                message=message,
                details={
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "memory_total_gb": memory.total / (1024**3),
                    "swap_percent": swap.percent,
                    "swap_used_gb": swap.used / (1024**3),
                },
                response_time_ms=response_time,
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component="memory",
                status=HealthStatus.UNHEALTHY,
                message=f"Memory check error: {str(e)}",
                details={"error": str(e)},
                response_time_ms=response_time,
            )

    async def _check_disk_space(self) -> HealthCheckResult:
        """Check disk space usage."""
        start_time = time.time()

        try:
            disk = psutil.disk_usage("/")

            response_time = (time.time() - start_time) * 1000

            # Determine status based on disk usage
            disk_percent = (disk.used / disk.total) * 100

            if disk_percent < 80:
                status = HealthStatus.HEALTHY
                message = "Disk space normal"
            elif disk_percent < 95:
                status = HealthStatus.DEGRADED
                message = "Disk space elevated"
            else:
                status = HealthStatus.UNHEALTHY
                message = "Disk space critical"

            return HealthCheckResult(
                component="disk",
                status=status,
                message=message,
                details={
                    "disk_percent": disk_percent,
                    "disk_free_gb": disk.free / (1024**3),
                    "disk_total_gb": disk.total / (1024**3),
                    "disk_used_gb": disk.used / (1024**3),
                },
                response_time_ms=response_time,
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component="disk",
                status=HealthStatus.UNHEALTHY,
                message=f"Disk space check error: {str(e)}",
                details={"error": str(e)},
                response_time_ms=response_time,
            )

    def _calculate_overall_status(self, checks: List[HealthCheckResult]) -> HealthStatus:
        """Calculate overall system status from individual checks."""
        if not checks:
            return HealthStatus.UNKNOWN

        unhealthy_count = len([c for c in checks if c.status == HealthStatus.UNHEALTHY])
        degraded_count = len([c for c in checks if c.status == HealthStatus.DEGRADED])

        # If any component is unhealthy, system is unhealthy
        if unhealthy_count > 0:
            return HealthStatus.UNHEALTHY

        # If any component is degraded, system is degraded
        if degraded_count > 0:
            return HealthStatus.DEGRADED

        # All components healthy
        return HealthStatus.HEALTHY

    async def check_component_health(self, component: str) -> HealthCheckResult:
        """
        Check health of a specific component.

        Args:
            component: Component name to check

        Returns:
            HealthCheckResult for the component
        """
        component_checks = {
            "janusgraph": self._check_janusgraph,
            "milvus": self._check_milvus,
            "gemini_api": self._check_gemini_api,
            "system_resources": self._check_system_resources,
            "memory": self._check_memory_usage,
            "disk": self._check_disk_space,
        }

        if component not in component_checks:
            return HealthCheckResult(
                component=component,
                status=HealthStatus.UNKNOWN,
                message=f"Unknown component: {component}",
            )

        try:
            return await component_checks[component]()
        except Exception as e:
            return HealthCheckResult(
                component=component,
                status=HealthStatus.UNHEALTHY,
                message=f"Component check error: {str(e)}",
                details={"error": str(e)},
            )


# Global health checker instance
_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get the global health checker instance."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker
