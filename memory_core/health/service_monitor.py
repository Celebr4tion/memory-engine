"""
Service monitoring and dependency health verification.
"""

import asyncio
import aiohttp
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import socket

logger = logging.getLogger(__name__)


class ServiceHealth(Enum):
    """Service health states."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    UNREACHABLE = "unreachable"


@dataclass
class ServiceStatus:
    """Status of a monitored service."""

    name: str
    health: ServiceHealth
    response_time: float
    last_check: float
    message: str
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class ServiceMonitor:
    """Monitor external services and dependencies."""

    def __init__(self, timeout: float = 10.0, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def check_http_service(
        self, name: str, url: str, expected_status: int = 200, health_path: str = None
    ) -> ServiceStatus:
        """Check HTTP service health."""
        start_time = time.time()

        # Use health endpoint if specified
        check_url = url + health_path if health_path else url

        for attempt in range(self.max_retries):
            try:
                session = await self._get_session()

                async with session.get(check_url) as response:
                    response_time = time.time() - start_time

                    if response.status == expected_status:
                        # Try to get additional health info from response
                        details = {}
                        try:
                            if response.content_type == "application/json":
                                details = await response.json()
                        except:
                            pass

                        return ServiceStatus(
                            name=name,
                            health=ServiceHealth.HEALTHY,
                            response_time=response_time,
                            last_check=time.time(),
                            message=f"Service responding with status {response.status}",
                            details=details,
                        )
                    else:
                        return ServiceStatus(
                            name=name,
                            health=ServiceHealth.DEGRADED,
                            response_time=response_time,
                            last_check=time.time(),
                            message=f"Service returned status {response.status}, expected {expected_status}",
                        )

            except asyncio.TimeoutError:
                if attempt == self.max_retries - 1:
                    return ServiceStatus(
                        name=name,
                        health=ServiceHealth.UNREACHABLE,
                        response_time=self.timeout,
                        last_check=time.time(),
                        message=f"Service timed out after {self.timeout}s",
                    )
                await asyncio.sleep(1)  # Brief delay before retry

            except Exception as e:
                if attempt == self.max_retries - 1:
                    return ServiceStatus(
                        name=name,
                        health=ServiceHealth.UNHEALTHY,
                        response_time=time.time() - start_time,
                        last_check=time.time(),
                        message=f"Service check failed: {str(e)}",
                    )
                await asyncio.sleep(1)

        # Should not reach here
        return ServiceStatus(
            name=name,
            health=ServiceHealth.UNKNOWN,
            response_time=0,
            last_check=time.time(),
            message="Unknown error occurred",
        )

    async def check_tcp_service(self, name: str, host: str, port: int) -> ServiceStatus:
        """Check TCP service connectivity."""
        start_time = time.time()

        try:
            # Test TCP connectivity
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=self.timeout
            )

            writer.close()
            await writer.wait_closed()

            response_time = time.time() - start_time

            return ServiceStatus(
                name=name,
                health=ServiceHealth.HEALTHY,
                response_time=response_time,
                last_check=time.time(),
                message=f"TCP connection to {host}:{port} successful",
            )

        except asyncio.TimeoutError:
            return ServiceStatus(
                name=name,
                health=ServiceHealth.UNREACHABLE,
                response_time=self.timeout,
                last_check=time.time(),
                message=f"TCP connection to {host}:{port} timed out",
            )

        except Exception as e:
            return ServiceStatus(
                name=name,
                health=ServiceHealth.UNHEALTHY,
                response_time=time.time() - start_time,
                last_check=time.time(),
                message=f"TCP connection to {host}:{port} failed: {str(e)}",
            )

    async def check_janusgraph_service(self, name: str, url: str) -> ServiceStatus:
        """Check JanusGraph service health."""
        # JanusGraph WebSocket endpoint health check
        return await self.check_tcp_service(
            name,
            url.split("://")[1].split(":")[0],  # Extract host
            int(url.split(":")[-1]),  # Extract port
        )

    async def check_milvus_service(self, name: str, host: str, port: int = 19530) -> ServiceStatus:
        """Check Milvus service health."""
        return await self.check_tcp_service(name, host, port)

    async def check_ollama_service(
        self, name: str, base_url: str = "http://localhost:11434"
    ) -> ServiceStatus:
        """Check Ollama service health."""
        return await self.check_http_service(name, base_url, health_path="/api/tags")

    async def check_database_service(self, name: str, connection_string: str) -> ServiceStatus:
        """Check database service connectivity."""
        start_time = time.time()

        try:
            if "sqlite://" in connection_string:
                # SQLite - check file access
                import os

                db_path = connection_string.replace("sqlite://", "")

                if os.path.exists(db_path) or db_path == ":memory:":
                    return ServiceStatus(
                        name=name,
                        health=ServiceHealth.HEALTHY,
                        response_time=time.time() - start_time,
                        last_check=time.time(),
                        message="SQLite database accessible",
                    )
                else:
                    return ServiceStatus(
                        name=name,
                        health=ServiceHealth.UNHEALTHY,
                        response_time=time.time() - start_time,
                        last_check=time.time(),
                        message=f"SQLite database file not found: {db_path}",
                    )

            elif "postgresql://" in connection_string or "mysql://" in connection_string:
                # For other databases, extract host and port for TCP check
                parts = connection_string.split("://")[1].split("/")[0]
                if "@" in parts:
                    parts = parts.split("@")[1]

                if ":" in parts:
                    host, port = parts.split(":")
                    port = int(port)
                else:
                    host = parts
                    port = 5432 if "postgresql" in connection_string else 3306

                return await self.check_tcp_service(name, host, port)

            else:
                return ServiceStatus(
                    name=name,
                    health=ServiceHealth.UNKNOWN,
                    response_time=time.time() - start_time,
                    last_check=time.time(),
                    message="Unsupported database type",
                )

        except Exception as e:
            return ServiceStatus(
                name=name,
                health=ServiceHealth.UNHEALTHY,
                response_time=time.time() - start_time,
                last_check=time.time(),
                message=f"Database service check failed: {str(e)}",
            )

    async def check_api_service(
        self, name: str, api_key: str, base_url: str, test_endpoint: str = None
    ) -> ServiceStatus:
        """Check external API service (OpenAI, Anthropic, etc.)."""
        start_time = time.time()

        try:
            session = await self._get_session()
            headers = {"Authorization": f"Bearer {api_key}"}

            # Use specific test endpoint or a simple models endpoint
            if test_endpoint:
                url = base_url + test_endpoint
            elif "openai" in base_url.lower():
                url = base_url + "/v1/models"
            elif "anthropic" in base_url.lower() or "claude" in base_url.lower():
                # Anthropic doesn't have a simple health endpoint
                # Return healthy if we can create headers (API key exists)
                return ServiceStatus(
                    name=name,
                    health=ServiceHealth.HEALTHY,
                    response_time=time.time() - start_time,
                    last_check=time.time(),
                    message="API key configured for Anthropic service",
                )
            else:
                url = base_url

            async with session.get(url, headers=headers) as response:
                response_time = time.time() - start_time

                if response.status in [200, 401]:  # 401 means service is up but auth failed
                    health = (
                        ServiceHealth.HEALTHY if response.status == 200 else ServiceHealth.DEGRADED
                    )
                    message = (
                        "API service accessible"
                        if response.status == 200
                        else "API service accessible but authentication may be invalid"
                    )

                    return ServiceStatus(
                        name=name,
                        health=health,
                        response_time=response_time,
                        last_check=time.time(),
                        message=message,
                    )
                else:
                    return ServiceStatus(
                        name=name,
                        health=ServiceHealth.DEGRADED,
                        response_time=response_time,
                        last_check=time.time(),
                        message=f"API service returned status {response.status}",
                    )

        except Exception as e:
            return ServiceStatus(
                name=name,
                health=ServiceHealth.UNHEALTHY,
                response_time=time.time() - start_time,
                last_check=time.time(),
                message=f"API service check failed: {str(e)}",
            )

    async def check_all_services(
        self, service_configs: List[Dict[str, Any]]
    ) -> Dict[str, ServiceStatus]:
        """Check multiple services concurrently."""
        tasks = []

        for config in service_configs:
            service_type = config.get("type")
            name = config.get("name")

            if service_type == "http":
                task = self.check_http_service(
                    name,
                    config["url"],
                    config.get("expected_status", 200),
                    config.get("health_path"),
                )
            elif service_type == "tcp":
                task = self.check_tcp_service(name, config["host"], config["port"])
            elif service_type == "janusgraph":
                task = self.check_janusgraph_service(name, config["url"])
            elif service_type == "milvus":
                task = self.check_milvus_service(name, config["host"], config.get("port", 19530))
            elif service_type == "ollama":
                task = self.check_ollama_service(
                    name, config.get("base_url", "http://localhost:11434")
                )
            elif service_type == "database":
                task = self.check_database_service(name, config["connection_string"])
            elif service_type == "api":
                task = self.check_api_service(
                    name, config["api_key"], config["base_url"], config.get("test_endpoint")
                )
            else:
                # Create a failed status for unknown service type
                task = asyncio.create_task(asyncio.sleep(0))  # Dummy task
                tasks.append((name, "unknown", task))
                continue

            tasks.append((name, service_type, task))

        # Wait for all checks to complete
        results = {}

        for name, service_type, task in tasks:
            try:
                if service_type == "unknown":
                    result = ServiceStatus(
                        name=name,
                        health=ServiceHealth.UNKNOWN,
                        response_time=0,
                        last_check=time.time(),
                        message=f"Unknown service type: {service_type}",
                    )
                else:
                    result = await task

                results[name] = result

            except Exception as e:
                results[name] = ServiceStatus(
                    name=name,
                    health=ServiceHealth.UNHEALTHY,
                    response_time=0,
                    last_check=time.time(),
                    message=f"Service check exception: {str(e)}",
                )

        return results

    async def get_system_dependencies_status(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get status of all system dependencies."""
        service_configs = []

        # Extract service configurations from system config

        # Storage backends
        storage_config = config.get("storage", {})
        if storage_config.get("backend") == "janusgraph":
            service_configs.append(
                {
                    "name": "janusgraph",
                    "type": "janusgraph",
                    "url": storage_config.get("janusgraph", {}).get(
                        "url", "ws://localhost:8182/gremlin"
                    ),
                }
            )

        # Vector stores
        vector_config = config.get("vector_store", {})
        if vector_config.get("provider") == "milvus":
            milvus_config = vector_config.get("milvus", {})
            service_configs.append(
                {
                    "name": "milvus",
                    "type": "milvus",
                    "host": milvus_config.get("host", "localhost"),
                    "port": milvus_config.get("port", 19530),
                }
            )

        # LLM providers
        llm_config = config.get("llm", {})

        # Ollama
        if "ollama" in llm_config:
            ollama_config = llm_config["ollama"]
            service_configs.append(
                {
                    "name": "ollama",
                    "type": "ollama",
                    "base_url": ollama_config.get("base_url", "http://localhost:11434"),
                }
            )

        # OpenAI
        if "openai" in llm_config:
            openai_config = llm_config["openai"]
            if openai_config.get("api_key"):
                service_configs.append(
                    {
                        "name": "openai_api",
                        "type": "api",
                        "api_key": openai_config["api_key"],
                        "base_url": openai_config.get("base_url", "https://api.openai.com"),
                        "test_endpoint": "/v1/models",
                    }
                )

        # Anthropic
        if "anthropic" in llm_config:
            anthropic_config = llm_config["anthropic"]
            if anthropic_config.get("api_key"):
                service_configs.append(
                    {
                        "name": "anthropic_api",
                        "type": "api",
                        "api_key": anthropic_config["api_key"],
                        "base_url": anthropic_config.get("base_url", "https://api.anthropic.com"),
                    }
                )

        # Check all services
        service_results = await self.check_all_services(service_configs)

        # Calculate overall dependency health
        overall_health = ServiceHealth.HEALTHY
        healthy_count = sum(
            1 for status in service_results.values() if status.health == ServiceHealth.HEALTHY
        )
        total_count = len(service_results)

        if total_count == 0:
            overall_health = ServiceHealth.UNKNOWN
        elif healthy_count == total_count:
            overall_health = ServiceHealth.HEALTHY
        elif healthy_count >= total_count * 0.8:  # 80% healthy
            overall_health = ServiceHealth.DEGRADED
        else:
            overall_health = ServiceHealth.UNHEALTHY

        return {
            "overall_health": overall_health.value,
            "healthy_services": healthy_count,
            "total_services": total_count,
            "services": {
                name: {
                    "health": status.health.value,
                    "response_time": status.response_time,
                    "message": status.message,
                    "last_check": status.last_check,
                    "details": status.details,
                }
                for name, status in service_results.items()
            },
            "timestamp": time.time(),
        }

    async def close(self):
        """Close the service monitor and cleanup resources."""
        if self._session and not self._session.closed:
            await self._session.close()
