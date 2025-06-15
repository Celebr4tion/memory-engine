"""
Advanced connection pooling for database backends.
"""

import asyncio
import threading
import time
import weakref
from typing import Any, Dict, Optional, List, Callable, AsyncGenerator, Protocol
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection states."""

    IDLE = "idle"
    ACTIVE = "active"
    BROKEN = "broken"
    CLOSED = "closed"


@dataclass
class PoolConfig:
    """Connection pool configuration."""

    min_connections: int = 2
    max_connections: int = 20
    connection_timeout: float = 30.0
    idle_timeout: float = 300.0  # 5 minutes
    max_lifetime: float = 3600.0  # 1 hour
    retry_attempts: int = 3
    retry_delay: float = 1.0
    health_check_interval: float = 60.0
    enable_monitoring: bool = True


@dataclass
class ConnectionMetrics:
    """Connection pool metrics."""

    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    broken_connections: int = 0
    connections_created: int = 0
    connections_closed: int = 0
    connection_errors: int = 0
    avg_connection_time: float = 0.0
    avg_idle_time: float = 0.0


class Connection:
    """Wrapper for database connections with metadata."""

    def __init__(self, connection: Any, pool: "ConnectionPool"):
        self.connection = connection
        self.pool = weakref.ref(pool)
        self.state = ConnectionState.IDLE
        self.created_at = time.time()
        self.last_used = self.created_at
        self.use_count = 0
        self.id = id(self)

    async def execute(self, query: str, *args, **kwargs):
        """Execute query on connection."""
        if self.state != ConnectionState.ACTIVE:
            raise RuntimeError(f"Connection is not active: {self.state}")

        try:
            self.last_used = time.time()
            self.use_count += 1

            # Execute based on connection type
            if hasattr(self.connection, "execute"):
                return await self.connection.execute(query, *args, **kwargs)
            elif hasattr(self.connection, "submit"):
                # Gremlin connection
                return await self.connection.submit(query, *args, **kwargs)
            else:
                raise NotImplementedError("Unknown connection type")

        except Exception as e:
            self.state = ConnectionState.BROKEN
            raise

    async def close(self):
        """Close the connection."""
        if self.state != ConnectionState.CLOSED:
            try:
                if hasattr(self.connection, "close"):
                    await self.connection.close()
                elif hasattr(self.connection, "disconnect"):
                    await self.connection.disconnect()
            finally:
                self.state = ConnectionState.CLOSED

    def is_expired(self, max_lifetime: float, idle_timeout: float) -> bool:
        """Check if connection is expired."""
        now = time.time()
        lifetime_expired = (now - self.created_at) > max_lifetime
        idle_expired = (now - self.last_used) > idle_timeout
        return lifetime_expired or (self.state == ConnectionState.IDLE and idle_expired)

    def is_healthy(self) -> bool:
        """Check if connection is healthy."""
        return self.state in (ConnectionState.IDLE, ConnectionState.ACTIVE)


class ConnectionFactory(Protocol):
    """Protocol for connection factories."""

    async def create_connection(self) -> Any:
        """Create a new connection."""
        ...

    async def validate_connection(self, connection: Any) -> bool:
        """Validate if connection is healthy."""
        ...


class ConnectionPool:
    """Advanced connection pool with monitoring and optimization."""

    def __init__(self, factory: ConnectionFactory, config: PoolConfig):
        self.factory = factory
        self.config = config
        self.metrics = ConnectionMetrics()

        self._connections: List[Connection] = []
        self._semaphore = asyncio.Semaphore(config.max_connections)
        self._lock = asyncio.Lock()
        self._closed = False
        self._monitor_task: Optional[asyncio.Task] = None

        # Start monitoring if enabled
        if config.enable_monitoring:
            self._start_monitoring()

    def _start_monitoring(self):
        """Start connection pool monitoring."""

        async def monitor_loop():
            while not self._closed:
                try:
                    await asyncio.sleep(self.config.health_check_interval)
                    await self._health_check()
                    await self._cleanup_expired()
                    self._update_metrics()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Connection pool monitoring error: {e}")

        self._monitor_task = asyncio.create_task(monitor_loop())

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[Connection, None]:
        """Acquire a connection from the pool."""
        if self._closed:
            raise RuntimeError("Connection pool is closed")

        async with self._semaphore:
            connection = await self._get_connection()
            try:
                yield connection
            finally:
                await self._return_connection(connection)

    async def _get_connection(self) -> Connection:
        """Get an available connection."""
        start_time = time.time()

        async with self._lock:
            # Try to find an idle connection
            for conn in self._connections:
                if conn.state == ConnectionState.IDLE and conn.is_healthy():
                    conn.state = ConnectionState.ACTIVE
                    self.metrics.avg_connection_time = (
                        self.metrics.avg_connection_time + (time.time() - start_time)
                    ) / 2
                    return conn

            # Create new connection if under limit
            if len(self._connections) < self.config.max_connections:
                try:
                    raw_conn = await asyncio.wait_for(
                        self.factory.create_connection(), timeout=self.config.connection_timeout
                    )

                    connection = Connection(raw_conn, self)
                    connection.state = ConnectionState.ACTIVE
                    self._connections.append(connection)

                    self.metrics.connections_created += 1
                    self.metrics.avg_connection_time = (
                        self.metrics.avg_connection_time + (time.time() - start_time)
                    ) / 2

                    return connection

                except Exception as e:
                    self.metrics.connection_errors += 1
                    logger.error(f"Failed to create connection: {e}")
                    raise

            # Wait for connection to become available
            # This is a simplified implementation - in practice you'd use a queue
            raise RuntimeError("No connections available and pool is at maximum capacity")

    async def _return_connection(self, connection: Connection):
        """Return connection to the pool."""
        async with self._lock:
            if connection.state == ConnectionState.BROKEN:
                await self._remove_connection(connection)
            else:
                connection.state = ConnectionState.IDLE
                idle_start = time.time()
                self.metrics.avg_idle_time = (
                    self.metrics.avg_idle_time + (idle_start - connection.last_used)
                ) / 2

    async def _remove_connection(self, connection: Connection):
        """Remove connection from pool."""
        if connection in self._connections:
            self._connections.remove(connection)
            await connection.close()
            self.metrics.connections_closed += 1

    async def _health_check(self):
        """Perform health check on connections."""
        async with self._lock:
            unhealthy_connections = []

            for conn in self._connections:
                if not conn.is_healthy():
                    unhealthy_connections.append(conn)
                elif conn.state == ConnectionState.IDLE:
                    # Validate idle connections
                    try:
                        is_valid = await self.factory.validate_connection(conn.connection)
                        if not is_valid:
                            conn.state = ConnectionState.BROKEN
                            unhealthy_connections.append(conn)
                    except Exception:
                        conn.state = ConnectionState.BROKEN
                        unhealthy_connections.append(conn)

            # Remove unhealthy connections
            for conn in unhealthy_connections:
                await self._remove_connection(conn)

    async def _cleanup_expired(self):
        """Clean up expired connections."""
        async with self._lock:
            expired_connections = [
                conn
                for conn in self._connections
                if conn.is_expired(self.config.max_lifetime, self.config.idle_timeout)
                and conn.state == ConnectionState.IDLE
            ]

            for conn in expired_connections:
                await self._remove_connection(conn)

            # Ensure minimum connections
            await self._ensure_minimum_connections()

    async def _ensure_minimum_connections(self):
        """Ensure minimum number of connections."""
        current_count = len([c for c in self._connections if c.is_healthy()])

        if current_count < self.config.min_connections:
            needed = self.config.min_connections - current_count
            for _ in range(needed):
                try:
                    raw_conn = await self.factory.create_connection()
                    connection = Connection(raw_conn, self)
                    self._connections.append(connection)
                    self.metrics.connections_created += 1
                except Exception as e:
                    logger.error(f"Failed to create minimum connection: {e}")
                    break

    def _update_metrics(self):
        """Update connection metrics."""
        self.metrics.total_connections = len(self._connections)
        self.metrics.active_connections = len(
            [c for c in self._connections if c.state == ConnectionState.ACTIVE]
        )
        self.metrics.idle_connections = len(
            [c for c in self._connections if c.state == ConnectionState.IDLE]
        )
        self.metrics.broken_connections = len(
            [c for c in self._connections if c.state == ConnectionState.BROKEN]
        )

    def get_metrics(self) -> ConnectionMetrics:
        """Get pool metrics."""
        self._update_metrics()
        return self.metrics

    async def close(self):
        """Close the connection pool."""
        self._closed = True

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        async with self._lock:
            for connection in self._connections:
                await connection.close()
            self._connections.clear()


class ConnectionPoolManager:
    """Manager for multiple connection pools."""

    def __init__(self):
        self._pools: Dict[str, ConnectionPool] = {}
        self._lock = asyncio.Lock()

    async def create_pool(
        self, name: str, factory: ConnectionFactory, config: PoolConfig
    ) -> ConnectionPool:
        """Create a new connection pool."""
        async with self._lock:
            if name in self._pools:
                raise ValueError(f"Pool '{name}' already exists")

            pool = ConnectionPool(factory, config)
            self._pools[name] = pool
            return pool

    async def get_pool(self, name: str) -> Optional[ConnectionPool]:
        """Get connection pool by name."""
        return self._pools.get(name)

    async def close_pool(self, name: str):
        """Close and remove connection pool."""
        async with self._lock:
            pool = self._pools.pop(name, None)
            if pool:
                await pool.close()

    async def close_all(self):
        """Close all connection pools."""
        async with self._lock:
            for pool in self._pools.values():
                await pool.close()
            self._pools.clear()

    def get_all_metrics(self) -> Dict[str, ConnectionMetrics]:
        """Get metrics for all pools."""
        return {name: pool.get_metrics() for name, pool in self._pools.items()}


# Factory implementations for different backends


class JanusGraphConnectionFactory:
    """Connection factory for JanusGraph."""

    def __init__(self, url: str, **kwargs):
        self.url = url
        self.kwargs = kwargs

    async def create_connection(self) -> Any:
        """Create JanusGraph connection."""
        try:
            from gremlin_python.driver import client
            from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection

            # Create connection based on configuration
            if "traversal_source" in self.kwargs:
                # WebSocket connection
                connection = DriverRemoteConnection(self.url, **self.kwargs)
            else:
                # Client connection
                connection = client.Client(self.url, **self.kwargs)

            return connection
        except ImportError:
            raise RuntimeError("gremlin-python is required for JanusGraph connections")

    async def validate_connection(self, connection: Any) -> bool:
        """Validate JanusGraph connection."""
        try:
            if hasattr(connection, "submit"):
                # Simple health check query
                result = await connection.submit("g.V().limit(1).count()")
                return True
            return True
        except Exception:
            return False


class SQLiteConnectionFactory:
    """Connection factory for SQLite."""

    def __init__(self, database_path: str, **kwargs):
        self.database_path = database_path
        self.kwargs = kwargs

    async def create_connection(self) -> Any:
        """Create SQLite connection."""
        try:
            import aiosqlite

            connection = await aiosqlite.connect(self.database_path, **self.kwargs)
            return connection
        except ImportError:
            raise RuntimeError("aiosqlite is required for SQLite connections")

    async def validate_connection(self, connection: Any) -> bool:
        """Validate SQLite connection."""
        try:
            await connection.execute("SELECT 1")
            return True
        except Exception:
            return False
