"""
Backend migration utilities for Memory Engine.
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import os
import tempfile

logger = logging.getLogger(__name__)


class MigrationStrategy(Enum):
    """Migration strategies."""

    INCREMENTAL = "incremental"  # Migrate in small batches
    BULK = "bulk"  # Migrate all at once
    STREAMING = "streaming"  # Stream data during migration
    SNAPSHOT = "snapshot"  # Create snapshot first, then migrate


@dataclass
class MigrationConfig:
    """Configuration for backend migration."""

    source_backend: str
    target_backend: str
    strategy: MigrationStrategy = MigrationStrategy.INCREMENTAL
    batch_size: int = 1000
    include_embeddings: bool = True
    include_metadata: bool = True
    verify_migration: bool = True
    cleanup_source: bool = False
    progress_callback: Optional[Callable[[float, str], None]] = None
    temp_dir: Optional[str] = None


@dataclass
class MigrationResult:
    """Result of a migration operation."""

    success: bool
    source_count: int
    target_count: int
    duration: float
    errors: List[str]
    warnings: List[str]
    details: Dict[str, Any]


class BackendMigrator:
    """Migrate data between different storage backends."""

    def __init__(self, config: MigrationConfig):
        self.config = config
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self._progress = 0.0
        self._status = "initialized"

    async def migrate(self, source_engine: Any, target_engine: Any) -> MigrationResult:
        """Perform backend migration."""
        start_time = time.time()
        self._update_progress(0.0, "Starting migration")

        try:
            # Validate source and target
            await self._validate_backends(source_engine, target_engine)
            self._update_progress(5.0, "Validated backends")

            # Count source data
            source_count = await self._count_source_data(source_engine)
            self._update_progress(10.0, f"Found {source_count} items in source")

            # Choose migration strategy
            if self.config.strategy == MigrationStrategy.INCREMENTAL:
                target_count = await self._migrate_incremental(
                    source_engine, target_engine, source_count
                )
            elif self.config.strategy == MigrationStrategy.BULK:
                target_count = await self._migrate_bulk(source_engine, target_engine)
            elif self.config.strategy == MigrationStrategy.STREAMING:
                target_count = await self._migrate_streaming(source_engine, target_engine)
            elif self.config.strategy == MigrationStrategy.SNAPSHOT:
                target_count = await self._migrate_snapshot(source_engine, target_engine)
            else:
                raise ValueError(f"Unknown migration strategy: {self.config.strategy}")

            self._update_progress(85.0, "Migration completed, verifying")

            # Verify migration if requested
            if self.config.verify_migration:
                await self._verify_migration(
                    source_engine, target_engine, source_count, target_count
                )

            self._update_progress(95.0, "Verification completed")

            # Cleanup source if requested
            if self.config.cleanup_source:
                await self._cleanup_source(source_engine)
                self.warnings.append("Source data was cleaned up as requested")

            self._update_progress(100.0, "Migration completed successfully")

            return MigrationResult(
                success=True,
                source_count=source_count,
                target_count=target_count,
                duration=time.time() - start_time,
                errors=self.errors,
                warnings=self.warnings,
                details={
                    "strategy": self.config.strategy.value,
                    "batch_size": self.config.batch_size,
                    "include_embeddings": self.config.include_embeddings,
                    "include_metadata": self.config.include_metadata,
                },
            )

        except Exception as e:
            error_msg = f"Migration failed: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg)

            return MigrationResult(
                success=False,
                source_count=0,
                target_count=0,
                duration=time.time() - start_time,
                errors=self.errors,
                warnings=self.warnings,
                details={"error": str(e)},
            )

    async def _validate_backends(self, source_engine: Any, target_engine: Any):
        """Validate source and target backends."""
        # Check source connectivity
        if hasattr(source_engine, "test_connection"):
            if not await source_engine.test_connection():
                raise RuntimeError("Cannot connect to source backend")

        # Check target connectivity
        if hasattr(target_engine, "test_connection"):
            if not await target_engine.test_connection():
                raise RuntimeError("Cannot connect to target backend")

        # Check if backends are different
        source_type = type(source_engine).__name__
        target_type = type(target_engine).__name__

        if source_type == target_type:
            self.warnings.append("Source and target backends are the same type")

    async def _count_source_data(self, source_engine: Any) -> int:
        """Count items in source backend."""
        try:
            if hasattr(source_engine, "count_all_nodes"):
                return await source_engine.count_all_nodes()
            elif hasattr(source_engine, "get_all_knowledge_nodes"):
                nodes = await source_engine.get_all_knowledge_nodes()
                return len(nodes) if nodes else 0
            else:
                # Fallback - try to get all nodes and count
                nodes = await self._get_all_source_nodes(source_engine)
                return len(nodes)
        except Exception as e:
            logger.warning(f"Could not count source data: {e}")
            return 0

    async def _migrate_incremental(
        self, source_engine: Any, target_engine: Any, total_count: int
    ) -> int:
        """Migrate data incrementally in batches."""
        migrated_count = 0
        batch_size = self.config.batch_size

        # Get all nodes in batches
        offset = 0
        while True:
            batch = await self._get_source_batch(source_engine, offset, batch_size)
            if not batch:
                break

            # Migrate batch
            batch_count = await self._migrate_batch(batch, target_engine)
            migrated_count += batch_count

            # Update progress
            progress = min(80.0, (migrated_count / total_count) * 70.0 + 10.0)
            self._update_progress(progress, f"Migrated {migrated_count}/{total_count} items")

            offset += batch_size

        return migrated_count

    async def _migrate_bulk(self, source_engine: Any, target_engine: Any) -> int:
        """Migrate all data at once."""
        # Get all source data
        all_nodes = await self._get_all_source_nodes(source_engine)
        self._update_progress(30.0, f"Retrieved {len(all_nodes)} nodes from source")

        # Migrate all at once
        migrated_count = await self._migrate_batch(all_nodes, target_engine)
        self._update_progress(80.0, f"Migrated {migrated_count} nodes to target")

        return migrated_count

    async def _migrate_streaming(self, source_engine: Any, target_engine: Any) -> int:
        """Stream data during migration."""
        migrated_count = 0

        # Use async generator if available
        if hasattr(source_engine, "stream_all_nodes"):
            async for node in source_engine.stream_all_nodes():
                await self._migrate_single_node(node, target_engine)
                migrated_count += 1

                if migrated_count % 100 == 0:
                    progress = min(80.0, (migrated_count / 1000) * 70.0 + 10.0)
                    self._update_progress(progress, f"Streamed {migrated_count} nodes")
        else:
            # Fallback to batch streaming
            return await self._migrate_incremental(source_engine, target_engine, 0)

        return migrated_count

    async def _migrate_snapshot(self, source_engine: Any, target_engine: Any) -> int:
        """Create snapshot first, then migrate."""
        # Create temporary snapshot
        temp_dir = self.config.temp_dir or tempfile.gettempdir()
        snapshot_file = os.path.join(temp_dir, f"migration_snapshot_{int(time.time())}.json")

        try:
            # Export to snapshot
            self._update_progress(20.0, "Creating snapshot")
            await self._create_snapshot(source_engine, snapshot_file)

            # Import from snapshot
            self._update_progress(50.0, "Importing from snapshot")
            migrated_count = await self._import_from_snapshot(snapshot_file, target_engine)

            return migrated_count

        finally:
            # Cleanup snapshot
            if os.path.exists(snapshot_file):
                os.remove(snapshot_file)

    async def _get_all_source_nodes(self, source_engine: Any) -> List[Any]:
        """Get all nodes from source backend."""
        if hasattr(source_engine, "get_all_knowledge_nodes"):
            return await source_engine.get_all_knowledge_nodes()
        elif hasattr(source_engine, "query_all_nodes"):
            return await source_engine.query_all_nodes()
        else:
            raise NotImplementedError(f"Don't know how to get all nodes from {type(source_engine)}")

    async def _get_source_batch(self, source_engine: Any, offset: int, limit: int) -> List[Any]:
        """Get a batch of nodes from source backend."""
        if hasattr(source_engine, "get_knowledge_nodes_batch"):
            return await source_engine.get_knowledge_nodes_batch(offset, limit)
        elif hasattr(source_engine, "query_nodes_batch"):
            return await source_engine.query_nodes_batch(offset, limit)
        else:
            # Fallback - get all and slice
            all_nodes = await self._get_all_source_nodes(source_engine)
            return all_nodes[offset : offset + limit]

    async def _migrate_batch(self, nodes: List[Any], target_engine: Any) -> int:
        """Migrate a batch of nodes to target backend."""
        migrated_count = 0

        for node in nodes:
            try:
                await self._migrate_single_node(node, target_engine)
                migrated_count += 1
            except Exception as e:
                error_msg = f"Failed to migrate node {getattr(node, 'id', 'unknown')}: {str(e)}"
                self.errors.append(error_msg)
                logger.error(error_msg)

        return migrated_count

    async def _migrate_single_node(self, node: Any, target_engine: Any):
        """Migrate a single node to target backend."""
        try:
            # Convert node to dict format
            node_data = self._node_to_dict(node)

            # Create node in target
            if hasattr(target_engine, "create_knowledge_node"):
                await target_engine.create_knowledge_node(**node_data)
            elif hasattr(target_engine, "add_node"):
                await target_engine.add_node(node_data)
            else:
                raise NotImplementedError(f"Don't know how to create node in {type(target_engine)}")

        except Exception as e:
            raise RuntimeError(f"Failed to migrate node: {str(e)}")

    def _node_to_dict(self, node: Any) -> Dict[str, Any]:
        """Convert node to dictionary format."""
        if hasattr(node, "to_dict"):
            node_data = node.to_dict()
        elif hasattr(node, "__dict__"):
            node_data = node.__dict__.copy()
        elif isinstance(node, dict):
            node_data = node.copy()
        else:
            raise ValueError(f"Cannot convert node to dict: {type(node)}")

        # Filter data based on configuration
        if not self.config.include_embeddings:
            node_data.pop("embedding", None)
            node_data.pop("embeddings", None)

        if not self.config.include_metadata:
            node_data.pop("metadata", None)
            node_data.pop("meta", None)

        return node_data

    async def _create_snapshot(self, source_engine: Any, snapshot_file: str):
        """Create snapshot of source data."""
        all_nodes = await self._get_all_source_nodes(source_engine)

        # Convert to serializable format
        snapshot_data = {
            "timestamp": time.time(),
            "source_backend": self.config.source_backend,
            "node_count": len(all_nodes),
            "nodes": [self._node_to_dict(node) for node in all_nodes],
        }

        # Write to file
        with open(snapshot_file, "w") as f:
            json.dump(snapshot_data, f, indent=2, default=str)

    async def _import_from_snapshot(self, snapshot_file: str, target_engine: Any) -> int:
        """Import data from snapshot file."""
        with open(snapshot_file, "r") as f:
            snapshot_data = json.load(f)

        nodes = snapshot_data.get("nodes", [])
        migrated_count = 0

        for node_data in nodes:
            try:
                await self._migrate_single_node(node_data, target_engine)
                migrated_count += 1
            except Exception as e:
                error_msg = f"Failed to import node from snapshot: {str(e)}"
                self.errors.append(error_msg)

        return migrated_count

    async def _verify_migration(
        self, source_engine: Any, target_engine: Any, source_count: int, target_count: int
    ):
        """Verify migration was successful."""
        if source_count != target_count:
            warning_msg = f"Count mismatch: source={source_count}, target={target_count}"
            self.warnings.append(warning_msg)
            logger.warning(warning_msg)

        # Sample verification - check a few random nodes
        try:
            sample_nodes = await self._get_source_batch(source_engine, 0, min(10, source_count))

            for node in sample_nodes:
                node_data = self._node_to_dict(node)
                node_id = node_data.get("id") or node_data.get("node_id")

                if node_id:
                    # Check if node exists in target
                    if hasattr(target_engine, "get_knowledge_node"):
                        target_node = await target_engine.get_knowledge_node(node_id)
                        if not target_node:
                            self.warnings.append(f"Node {node_id} not found in target")

        except Exception as e:
            warning_msg = f"Verification check failed: {str(e)}"
            self.warnings.append(warning_msg)
            logger.warning(warning_msg)

    async def _cleanup_source(self, source_engine: Any):
        """Cleanup source backend after migration."""
        try:
            if hasattr(source_engine, "clear_all_data"):
                await source_engine.clear_all_data()
            elif hasattr(source_engine, "delete_all_nodes"):
                await source_engine.delete_all_nodes()
            else:
                self.warnings.append("Source cleanup not supported for this backend type")
        except Exception as e:
            error_msg = f"Source cleanup failed: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg)

    def _update_progress(self, progress: float, status: str):
        """Update migration progress."""
        self._progress = progress
        self._status = status

        if self.config.progress_callback:
            try:
                self.config.progress_callback(progress, status)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

        logger.info(f"Migration progress: {progress:.1f}% - {status}")


# Convenience functions for common migrations


async def migrate_janusgraph_to_sqlite(
    janusgraph_engine: Any, sqlite_engine: Any, **kwargs
) -> MigrationResult:
    """Migrate from JanusGraph to SQLite."""
    config = MigrationConfig(
        source_backend="janusgraph",
        target_backend="sqlite",
        strategy=MigrationStrategy.INCREMENTAL,
        **kwargs,
    )

    migrator = BackendMigrator(config)
    return await migrator.migrate(janusgraph_engine, sqlite_engine)


async def migrate_sqlite_to_json(sqlite_engine: Any, json_engine: Any, **kwargs) -> MigrationResult:
    """Migrate from SQLite to JSON file."""
    config = MigrationConfig(
        source_backend="sqlite",
        target_backend="json_file",
        strategy=MigrationStrategy.BULK,
        **kwargs,
    )

    migrator = BackendMigrator(config)
    return await migrator.migrate(sqlite_engine, json_engine)


async def migrate_json_to_janusgraph(
    json_engine: Any, janusgraph_engine: Any, **kwargs
) -> MigrationResult:
    """Migrate from JSON file to JanusGraph."""
    config = MigrationConfig(
        source_backend="json_file",
        target_backend="janusgraph",
        strategy=MigrationStrategy.STREAMING,
        **kwargs,
    )

    migrator = BackendMigrator(config)
    return await migrator.migrate(json_engine, janusgraph_engine)
