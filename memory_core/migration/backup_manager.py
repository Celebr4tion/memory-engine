"""
Backup and restore functionality for Memory Engine.
"""

import os
import json
import gzip
import shutil
import tempfile
import time
import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class BackupStrategy(Enum):
    """Backup strategies."""

    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"


class CompressionType(Enum):
    """Compression types."""

    NONE = "none"
    GZIP = "gzip"
    BZIP2 = "bzip2"
    LZMA = "lzma"


@dataclass
class BackupConfig:
    """Configuration for backup operations."""

    strategy: BackupStrategy = BackupStrategy.FULL
    compression: CompressionType = CompressionType.GZIP
    include_embeddings: bool = True
    include_metadata: bool = True
    include_relationships: bool = True
    verify_backup: bool = True
    retention_days: int = 30
    max_backup_size_gb: float = 10.0
    progress_callback: Optional[Callable[[float, str], None]] = None


@dataclass
class BackupInfo:
    """Information about a backup."""

    backup_id: str
    timestamp: float
    strategy: BackupStrategy
    compression: CompressionType
    file_path: str
    file_size_mb: float
    node_count: int
    relationship_count: int
    checksum: str
    metadata: Dict[str, Any]


class BackupManager:
    """Manage backups and restores for Memory Engine."""

    def __init__(self, config: BackupConfig, backup_directory: str):
        self.config = config
        self.backup_directory = backup_directory
        self._ensure_backup_directory()
        self._backup_index: Dict[str, BackupInfo] = {}
        self._load_backup_index()

    def _ensure_backup_directory(self):
        """Ensure backup directory exists."""
        os.makedirs(self.backup_directory, exist_ok=True)

    def _load_backup_index(self):
        """Load backup index from disk."""
        index_file = os.path.join(self.backup_directory, "backup_index.json")

        if os.path.exists(index_file):
            try:
                with open(index_file, "r") as f:
                    data = json.load(f)

                for backup_id, backup_data in data.items():
                    self._backup_index[backup_id] = BackupInfo(
                        backup_id=backup_data["backup_id"],
                        timestamp=backup_data["timestamp"],
                        strategy=BackupStrategy(backup_data["strategy"]),
                        compression=CompressionType(backup_data["compression"]),
                        file_path=backup_data["file_path"],
                        file_size_mb=backup_data["file_size_mb"],
                        node_count=backup_data["node_count"],
                        relationship_count=backup_data["relationship_count"],
                        checksum=backup_data["checksum"],
                        metadata=backup_data.get("metadata", {}),
                    )

            except Exception as e:
                logger.error(f"Failed to load backup index: {e}")

    def _save_backup_index(self):
        """Save backup index to disk."""
        index_file = os.path.join(self.backup_directory, "backup_index.json")

        try:
            data = {}
            for backup_id, backup_info in self._backup_index.items():
                data[backup_id] = {
                    "backup_id": backup_info.backup_id,
                    "timestamp": backup_info.timestamp,
                    "strategy": backup_info.strategy.value,
                    "compression": backup_info.compression.value,
                    "file_path": backup_info.file_path,
                    "file_size_mb": backup_info.file_size_mb,
                    "node_count": backup_info.node_count,
                    "relationship_count": backup_info.relationship_count,
                    "checksum": backup_info.checksum,
                    "metadata": backup_info.metadata,
                }

            with open(index_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save backup index: {e}")

    async def create_backup(self, engine: Any, backup_id: str = None) -> BackupInfo:
        """Create a backup of the knowledge graph."""
        start_time = time.time()

        if not backup_id:
            backup_id = f"backup_{int(time.time())}"

        self._update_progress(0.0, "Starting backup")

        try:
            # Get data from engine
            self._update_progress(10.0, "Retrieving data from engine")
            nodes = await self._get_all_nodes(engine)
            relationships = (
                await self._get_all_relationships(engine)
                if self.config.include_relationships
                else []
            )

            self._update_progress(
                30.0, f"Retrieved {len(nodes)} nodes and {len(relationships)} relationships"
            )

            # Create backup data structure
            backup_data = {
                "backup_info": {
                    "backup_id": backup_id,
                    "timestamp": time.time(),
                    "strategy": self.config.strategy.value,
                    "compression": self.config.compression.value,
                    "node_count": len(nodes),
                    "relationship_count": len(relationships),
                    "config": {
                        "include_embeddings": self.config.include_embeddings,
                        "include_metadata": self.config.include_metadata,
                        "include_relationships": self.config.include_relationships,
                    },
                },
                "nodes": nodes,
                "relationships": relationships,
            }

            self._update_progress(50.0, "Creating backup file")

            # Write backup file
            backup_file = os.path.join(self.backup_directory, f"{backup_id}.backup")
            await self._write_backup_file(backup_data, backup_file)

            self._update_progress(80.0, "Backup file created")

            # Calculate file size and checksum
            file_size_mb = os.path.getsize(backup_file) / (1024 * 1024)
            checksum = await self._calculate_checksum(backup_file)

            # Create backup info
            backup_info = BackupInfo(
                backup_id=backup_id,
                timestamp=backup_data["backup_info"]["timestamp"],
                strategy=self.config.strategy,
                compression=self.config.compression,
                file_path=backup_file,
                file_size_mb=file_size_mb,
                node_count=len(nodes),
                relationship_count=len(relationships),
                checksum=checksum,
                metadata={
                    "duration": time.time() - start_time,
                    "source_engine": type(engine).__name__,
                },
            )

            # Verify backup if requested
            if self.config.verify_backup:
                self._update_progress(90.0, "Verifying backup")
                await self._verify_backup(backup_file)

            # Update index
            self._backup_index[backup_id] = backup_info
            self._save_backup_index()

            # Cleanup old backups
            await self._cleanup_old_backups()

            self._update_progress(100.0, "Backup completed successfully")

            logger.info(f"Backup created: {backup_id} ({file_size_mb:.2f} MB)")
            return backup_info

        except Exception as e:
            logger.error(f"Backup failed: {e}")
            raise

    async def restore_backup(
        self, backup_id: str, engine: Any, clear_existing: bool = False
    ) -> Dict[str, Any]:
        """Restore from a backup."""
        start_time = time.time()

        if backup_id not in self._backup_index:
            raise ValueError(f"Backup {backup_id} not found")

        backup_info = self._backup_index[backup_id]

        try:
            self._update_progress(0.0, f"Starting restore from {backup_id}")

            # Verify backup file exists
            if not os.path.exists(backup_info.file_path):
                raise FileNotFoundError(f"Backup file not found: {backup_info.file_path}")

            # Verify backup integrity
            self._update_progress(10.0, "Verifying backup integrity")
            await self._verify_backup(backup_info.file_path)

            # Read backup data
            self._update_progress(20.0, "Reading backup data")
            backup_data = await self._read_backup_file(backup_info.file_path)

            # Clear existing data if requested
            if clear_existing:
                self._update_progress(30.0, "Clearing existing data")
                await self._clear_engine_data(engine)

            # Restore nodes
            self._update_progress(40.0, "Restoring nodes")
            nodes = backup_data.get("nodes", [])
            restored_nodes = await self._restore_nodes(nodes, engine)

            # Restore relationships
            self._update_progress(70.0, "Restoring relationships")
            relationships = backup_data.get("relationships", [])
            restored_relationships = await self._restore_relationships(relationships, engine)

            self._update_progress(100.0, "Restore completed successfully")

            result = {
                "success": True,
                "backup_id": backup_id,
                "restored_nodes": restored_nodes,
                "restored_relationships": restored_relationships,
                "duration": time.time() - start_time,
            }

            logger.info(
                f"Restore completed: {backup_id} ({restored_nodes} nodes, {restored_relationships} relationships)"
            )
            return result

        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return {"success": False, "error": str(e), "duration": time.time() - start_time}

    async def list_backups(self) -> List[BackupInfo]:
        """List all available backups."""
        return sorted(self._backup_index.values(), key=lambda x: x.timestamp, reverse=True)

    async def delete_backup(self, backup_id: str) -> bool:
        """Delete a backup."""
        if backup_id not in self._backup_index:
            return False

        backup_info = self._backup_index[backup_id]

        try:
            # Delete backup file
            if os.path.exists(backup_info.file_path):
                os.remove(backup_info.file_path)

            # Remove from index
            del self._backup_index[backup_id]
            self._save_backup_index()

            logger.info(f"Backup deleted: {backup_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete backup {backup_id}: {e}")
            return False

    async def _get_all_nodes(self, engine: Any) -> List[Dict[str, Any]]:
        """Get all nodes from engine."""
        try:
            if hasattr(engine, "get_all_knowledge_nodes"):
                nodes = await engine.get_all_knowledge_nodes()
            elif hasattr(engine, "query_all_nodes"):
                nodes = await engine.query_all_nodes()
            else:
                raise NotImplementedError("Engine doesn't support node retrieval")

            # Convert to dict format and filter
            result = []
            for node in nodes:
                node_data = self._node_to_dict(node)
                result.append(node_data)

            return result

        except Exception as e:
            logger.error(f"Failed to get nodes: {e}")
            return []

    async def _get_all_relationships(self, engine: Any) -> List[Dict[str, Any]]:
        """Get all relationships from engine."""
        try:
            if hasattr(engine, "get_all_relationships"):
                relationships = await engine.get_all_relationships()
            elif hasattr(engine, "query_all_relationships"):
                relationships = await engine.query_all_relationships()
            else:
                logger.warning("Engine doesn't support relationship retrieval")
                return []

            # Convert to dict format
            return [self._relationship_to_dict(rel) for rel in relationships]

        except Exception as e:
            logger.warning(f"Failed to get relationships: {e}")
            return []

    def _node_to_dict(self, node: Any) -> Dict[str, Any]:
        """Convert node to dictionary."""
        if hasattr(node, "to_dict"):
            node_data = node.to_dict()
        elif hasattr(node, "__dict__"):
            node_data = node.__dict__.copy()
        elif isinstance(node, dict):
            node_data = node.copy()
        else:
            node_data = {"content": str(node)}

        # Apply backup filters
        if not self.config.include_embeddings:
            node_data.pop("embedding", None)
            node_data.pop("embeddings", None)

        if not self.config.include_metadata:
            node_data.pop("metadata", None)
            node_data.pop("meta", None)

        return node_data

    def _relationship_to_dict(self, relationship: Any) -> Dict[str, Any]:
        """Convert relationship to dictionary."""
        if hasattr(relationship, "to_dict"):
            return relationship.to_dict()
        elif hasattr(relationship, "__dict__"):
            return relationship.__dict__.copy()
        elif isinstance(relationship, dict):
            return relationship.copy()
        else:
            return {"type": str(relationship)}

    async def _write_backup_file(self, data: Dict, file_path: str):
        """Write backup data to file with optional compression."""
        temp_file = file_path + ".tmp"

        try:
            if self.config.compression == CompressionType.GZIP:
                with gzip.open(temp_file, "wt", encoding="utf-8") as f:
                    json.dump(data, f, default=str)
            elif self.config.compression == CompressionType.BZIP2:
                import bz2

                with bz2.open(temp_file, "wt", encoding="utf-8") as f:
                    json.dump(data, f, default=str)
            elif self.config.compression == CompressionType.LZMA:
                import lzma

                with lzma.open(temp_file, "wt", encoding="utf-8") as f:
                    json.dump(data, f, default=str)
            else:
                # No compression
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, default=str)

            # Atomic move
            shutil.move(temp_file, file_path)

        except Exception as e:
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise

    async def _read_backup_file(self, file_path: str) -> Dict:
        """Read backup data from file with decompression."""
        try:
            # Try to detect compression from file header
            with open(file_path, "rb") as f:
                header = f.read(3)

            if header[:2] == b"\\x1f\\x8b":  # GZIP
                with gzip.open(file_path, "rt", encoding="utf-8") as f:
                    return json.load(f)
            elif header == b"BZh":  # BZIP2
                import bz2

                with bz2.open(file_path, "rt", encoding="utf-8") as f:
                    return json.load(f)
            elif header[:3] == b"\\xfd7z":  # LZMA
                import lzma

                with lzma.open(file_path, "rt", encoding="utf-8") as f:
                    return json.load(f)
            else:
                # No compression
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)

        except Exception as e:
            logger.error(f"Failed to read backup file {file_path}: {e}")
            raise

    async def _calculate_checksum(self, file_path: str) -> str:
        """Calculate MD5 checksum of backup file."""
        import hashlib

        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    async def _verify_backup(self, file_path: str):
        """Verify backup file integrity."""
        try:
            # Try to read the file
            data = await self._read_backup_file(file_path)

            # Verify structure
            if not isinstance(data, dict):
                raise ValueError("Invalid backup format: not a dictionary")

            if "backup_info" not in data:
                raise ValueError("Invalid backup format: missing backup_info")

            if "nodes" not in data:
                raise ValueError("Invalid backup format: missing nodes")

            logger.debug(f"Backup verification passed: {file_path}")

        except Exception as e:
            raise ValueError(f"Backup verification failed: {e}")

    async def _clear_engine_data(self, engine: Any):
        """Clear all data from engine."""
        try:
            if hasattr(engine, "clear_all_data"):
                await engine.clear_all_data()
            elif hasattr(engine, "delete_all_nodes"):
                await engine.delete_all_nodes()
            else:
                logger.warning("Engine doesn't support data clearing")
        except Exception as e:
            logger.error(f"Failed to clear engine data: {e}")
            raise

    async def _restore_nodes(self, nodes: List[Dict], engine: Any) -> int:
        """Restore nodes to engine."""
        restored_count = 0

        for node_data in nodes:
            try:
                if hasattr(engine, "create_knowledge_node"):
                    await engine.create_knowledge_node(**node_data)
                elif hasattr(engine, "add_node"):
                    await engine.add_node(node_data)
                else:
                    raise NotImplementedError("Engine doesn't support node creation")

                restored_count += 1

            except Exception as e:
                logger.error(f"Failed to restore node: {e}")

        return restored_count

    async def _restore_relationships(self, relationships: List[Dict], engine: Any) -> int:
        """Restore relationships to engine."""
        restored_count = 0

        for rel_data in relationships:
            try:
                if hasattr(engine, "create_relationship"):
                    await engine.create_relationship(**rel_data)
                elif hasattr(engine, "add_relationship"):
                    await engine.add_relationship(rel_data)
                else:
                    logger.warning("Engine doesn't support relationship creation")
                    continue

                restored_count += 1

            except Exception as e:
                logger.error(f"Failed to restore relationship: {e}")

        return restored_count

    async def _cleanup_old_backups(self):
        """Clean up old backups based on retention policy."""
        if self.config.retention_days <= 0:
            return

        cutoff_time = time.time() - (self.config.retention_days * 24 * 3600)

        for backup_id, backup_info in list(self._backup_index.items()):
            if backup_info.timestamp < cutoff_time:
                logger.info(f"Deleting old backup: {backup_id}")
                await self.delete_backup(backup_id)

    def _update_progress(self, progress: float, status: str):
        """Update backup/restore progress."""
        if self.config.progress_callback:
            try:
                self.config.progress_callback(progress, status)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

        logger.info(f"Backup progress: {progress:.1f}% - {status}")
