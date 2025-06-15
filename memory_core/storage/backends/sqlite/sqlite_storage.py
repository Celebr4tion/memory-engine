"""
SQLite storage implementation for graph storage.

This module provides a SQLite-based implementation of the GraphStorageInterface
using relational tables to store graph data. Good balance between simplicity
and performance for single-user deployments.
"""

import sqlite3
import json
import uuid
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import asyncio
import aiosqlite

from memory_core.storage.interfaces.graph_storage_interface import GraphStorageInterface
from memory_core.model.knowledge_node import KnowledgeNode
from memory_core.model.relationship import Relationship


class SqliteStorage(GraphStorageInterface):
    """
    SQLite-based implementation of the GraphStorageInterface.

    This implementation stores graph data in SQLite database tables,
    providing a good balance between simplicity and performance for
    single-user deployments.
    """

    def __init__(self, database_path: str = "./data/knowledge.db"):
        """
        Initialize SqliteStorage with database path.

        Args:
            database_path: Path to the SQLite database file
        """
        self.database_path = Path(database_path)
        self.logger = logging.getLogger(__name__)

        # Performance caches
        self._relationship_cache = {}
        self._content_index_cache = {}

        # Connection state
        self._connected = False
        self._db_connection = None

    # Connection Management
    async def connect(self) -> None:
        """Establish connection to the SQLite database."""
        if self._connected:
            return

        try:
            # Create directory if it doesn't exist
            self.database_path.parent.mkdir(parents=True, exist_ok=True)

            # Open database connection
            self._db_connection = await aiosqlite.connect(str(self.database_path))

            # Enable foreign key constraints
            await self._db_connection.execute("PRAGMA foreign_keys = ON")

            # Create tables
            await self._create_tables()

            self._connected = True
            self.logger.info(f"Connected to SQLite database at {self.database_path}")

        except Exception as e:
            self.logger.error(f"Failed to connect to SQLite database: {e}")
            raise

    async def close(self) -> None:
        """Close connection to the SQLite database."""
        if not self._connected:
            return

        try:
            if self._db_connection:
                await self._db_connection.close()
                self._db_connection = None

            self._connected = False
            self.logger.info("Disconnected from SQLite database")

        except Exception as e:
            self.logger.error(f"Error closing SQLite database: {e}")

    async def test_connection(self) -> bool:
        """Test if the connection to storage backend is working."""
        try:
            # Test database file access
            self.database_path.parent.mkdir(parents=True, exist_ok=True)

            # Try to open and query the database
            async with aiosqlite.connect(str(self.database_path)) as db:
                await db.execute("SELECT 1")

            return True
        except Exception as e:
            self.logger.error(f"SQLite connection test failed: {e}")
            return False

    async def is_available(self) -> bool:
        """Check if the storage backend is available."""
        return await self.test_connection()

    def connect_sync(self) -> None:
        """Establish connection to the storage backend (synchronous)."""
        return self._run_async_in_sync_context(self.connect())

    def close_sync(self) -> None:
        """Close connection to the storage backend (synchronous)."""
        return self._run_async_in_sync_context(self.close())

    def test_connection_sync(self) -> bool:
        """Test if the connection to storage backend is working (synchronous)."""
        return self._run_async_in_sync_context(self.test_connection())

    def is_available_sync(self) -> bool:
        """Check if the storage backend is available (synchronous)."""
        return self.test_connection_sync()

    # Raw Node Operations
    async def create_node(self, node_data: Dict[str, Any]) -> str:
        """Create a new node in the graph."""
        await self.connect()

        # Validate required fields
        required_fields = [
            "content",
            "source",
            "creation_timestamp",
            "rating_richness",
            "rating_truthfulness",
            "rating_stability",
        ]
        missing_fields = [field for field in required_fields if field not in node_data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        node_id = str(uuid.uuid4())

        try:
            await self._db_connection.execute(
                """
                INSERT INTO nodes (node_id, content, source, creation_timestamp, 
                                 rating_richness, rating_truthfulness, rating_stability, 
                                 properties)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    node_id,
                    node_data["content"],
                    node_data["source"],
                    node_data["creation_timestamp"],
                    node_data["rating_richness"],
                    node_data["rating_truthfulness"],
                    node_data["rating_stability"],
                    json.dumps({k: v for k, v in node_data.items() if k not in required_fields}),
                ),
            )

            await self._db_connection.commit()

            self.logger.debug(f"Created node {node_id}")
            return node_id

        except Exception as e:
            await self._db_connection.rollback()
            self.logger.error(f"Error creating node: {e}")
            raise

    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a node by its ID."""
        await self.connect()

        try:
            cursor = await self._db_connection.execute(
                """
                SELECT node_id, content, source, creation_timestamp, 
                       rating_richness, rating_truthfulness, rating_stability, properties
                FROM nodes WHERE node_id = ?
            """,
                (node_id,),
            )

            row = await cursor.fetchone()
            if not row:
                return None

            node_data = {
                "node_id": row[0],
                "content": row[1],
                "source": row[2],
                "creation_timestamp": row[3],
                "rating_richness": row[4],
                "rating_truthfulness": row[5],
                "rating_stability": row[6],
            }

            # Add additional properties
            if row[7]:
                properties = json.loads(row[7])
                node_data.update(properties)

            return node_data

        except Exception as e:
            self.logger.error(f"Error getting node {node_id}: {e}")
            raise

    async def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """Update properties of an existing node."""
        await self.connect()

        try:
            # Build update query for known fields
            known_fields = [
                "content",
                "source",
                "creation_timestamp",
                "rating_richness",
                "rating_truthfulness",
                "rating_stability",
            ]
            update_fields = []
            values = []

            for field in known_fields:
                if field in properties:
                    update_fields.append(f"{field} = ?")
                    values.append(properties[field])

            # Handle additional properties
            additional_props = {k: v for k, v in properties.items() if k not in known_fields}
            if additional_props:
                update_fields.append("properties = ?")
                values.append(json.dumps(additional_props))

            if not update_fields:
                return True  # Nothing to update

            values.append(node_id)

            query = f"UPDATE nodes SET {', '.join(update_fields)} WHERE node_id = ?"
            await self._db_connection.execute(query, values)
            await self._db_connection.commit()

            self.logger.debug(f"Updated node {node_id}")
            return True

        except Exception as e:
            await self._db_connection.rollback()
            self.logger.error(f"Error updating node {node_id}: {e}")
            return False

    async def delete_node(self, node_id: str) -> bool:
        """Delete a node from the graph."""
        await self.connect()

        try:
            # Delete connected edges first (foreign key constraints will handle this)
            await self._db_connection.execute(
                "DELETE FROM edges WHERE from_node_id = ? OR to_node_id = ?", (node_id, node_id)
            )

            # Delete the node
            cursor = await self._db_connection.execute(
                "DELETE FROM nodes WHERE node_id = ?", (node_id,)
            )
            await self._db_connection.commit()

            if cursor.rowcount > 0:
                self.logger.debug(f"Deleted node {node_id}")
                return True
            else:
                return False

        except Exception as e:
            await self._db_connection.rollback()
            self.logger.error(f"Error deleting node {node_id}: {e}")
            return False

    # Raw Edge Operations
    async def create_edge(
        self, from_node_id: str, to_node_id: str, relation_type: str, properties: Dict[str, Any]
    ) -> str:
        """Create a new edge between two nodes."""
        await self.connect()

        # Validate required fields
        required_fields = ["timestamp", "confidence_score", "version"]
        missing_fields = [field for field in required_fields if field not in properties]
        if missing_fields:
            raise ValueError(f"Missing required edge fields: {missing_fields}")

        edge_id = str(uuid.uuid4())

        try:
            await self._db_connection.execute(
                """
                INSERT INTO edges (edge_id, from_node_id, to_node_id, relation_type, 
                                 timestamp, confidence_score, version, properties)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    edge_id,
                    from_node_id,
                    to_node_id,
                    relation_type,
                    properties["timestamp"],
                    properties["confidence_score"],
                    properties["version"],
                    json.dumps({k: v for k, v in properties.items() if k not in required_fields}),
                ),
            )

            await self._db_connection.commit()

            self.logger.debug(f"Created edge {edge_id} from {from_node_id} to {to_node_id}")
            return edge_id

        except Exception as e:
            await self._db_connection.rollback()
            self.logger.error(f"Error creating edge: {e}")
            raise

    async def get_edge(self, edge_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an edge by its ID."""
        await self.connect()

        try:
            cursor = await self._db_connection.execute(
                """
                SELECT edge_id, from_node_id, to_node_id, relation_type, 
                       timestamp, confidence_score, version, properties
                FROM edges WHERE edge_id = ?
            """,
                (edge_id,),
            )

            row = await cursor.fetchone()
            if not row:
                return None

            edge_data = {
                "edge_id": row[0],
                "from_id": row[1],
                "to_id": row[2],
                "relation_type": row[3],
                "timestamp": row[4],
                "confidence_score": row[5],
                "version": row[6],
            }

            # Add additional properties
            if row[7]:
                properties = json.loads(row[7])
                edge_data.update(properties)

            return edge_data

        except Exception as e:
            self.logger.error(f"Error getting edge {edge_id}: {e}")
            raise

    async def update_edge(self, edge_id: str, properties: Dict[str, Any]) -> bool:
        """Update properties of an existing edge."""
        await self.connect()

        try:
            # Build update query for known fields
            known_fields = ["timestamp", "confidence_score", "version", "relation_type"]
            update_fields = []
            values = []

            for field in known_fields:
                if field in properties:
                    update_fields.append(f"{field} = ?")
                    values.append(properties[field])

            # Handle additional properties
            additional_props = {k: v for k, v in properties.items() if k not in known_fields}
            if additional_props:
                update_fields.append("properties = ?")
                values.append(json.dumps(additional_props))

            if not update_fields:
                return True  # Nothing to update

            values.append(edge_id)

            query = f"UPDATE edges SET {', '.join(update_fields)} WHERE edge_id = ?"
            await self._db_connection.execute(query, values)
            await self._db_connection.commit()

            self.logger.debug(f"Updated edge {edge_id}")
            return True

        except Exception as e:
            await self._db_connection.rollback()
            self.logger.error(f"Error updating edge {edge_id}: {e}")
            return False

    async def delete_edge(self, edge_id: str) -> bool:
        """Delete an edge from the graph."""
        await self.connect()

        try:
            cursor = await self._db_connection.execute(
                "DELETE FROM edges WHERE edge_id = ?", (edge_id,)
            )
            await self._db_connection.commit()

            if cursor.rowcount > 0:
                self.logger.debug(f"Deleted edge {edge_id}")
                return True
            else:
                return False

        except Exception as e:
            await self._db_connection.rollback()
            self.logger.error(f"Error deleting edge {edge_id}: {e}")
            return False

    # Graph Traversal Operations
    async def find_neighbors(
        self, node_id: str, relation_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Find neighboring nodes of a given node."""
        await self.connect()

        try:
            if relation_type:
                cursor = await self._db_connection.execute(
                    """
                    SELECT edge_id, from_node_id, to_node_id, relation_type, 
                           timestamp, confidence_score, version, properties
                    FROM edges 
                    WHERE (from_node_id = ? OR to_node_id = ?) AND relation_type = ?
                """,
                    (node_id, node_id, relation_type),
                )
            else:
                cursor = await self._db_connection.execute(
                    """
                    SELECT edge_id, from_node_id, to_node_id, relation_type, 
                           timestamp, confidence_score, version, properties
                    FROM edges 
                    WHERE from_node_id = ? OR to_node_id = ?
                """,
                    (node_id, node_id),
                )

            rows = await cursor.fetchall()
            neighbors = []

            for row in rows:
                edge_data = {
                    "edge_id": row[0],
                    "from_id": row[1],
                    "to_id": row[2],
                    "relation_type": row[3],
                    "timestamp": row[4],
                    "confidence_score": row[5],
                    "version": row[6],
                }

                # Add additional properties
                if row[7]:
                    properties = json.loads(row[7])
                    edge_data.update(properties)

                neighbors.append(edge_data)

            return neighbors

        except Exception as e:
            self.logger.error(f"Error finding neighbors for node {node_id}: {e}")
            raise

    async def get_outgoing_relationships(self, node_id: str) -> List[Dict[str, Any]]:
        """Get all outgoing relationships from a node."""
        await self.connect()

        try:
            cursor = await self._db_connection.execute(
                """
                SELECT edge_id, from_node_id, to_node_id, relation_type, 
                       timestamp, confidence_score, version, properties
                FROM edges WHERE from_node_id = ?
            """,
                (node_id,),
            )

            rows = await cursor.fetchall()
            relationships = []

            for row in rows:
                edge_data = {
                    "edge_id": row[0],
                    "from_id": row[1],
                    "to_id": row[2],
                    "relation_type": row[3],
                    "timestamp": row[4],
                    "confidence_score": row[5],
                    "version": row[6],
                }

                # Add additional properties
                if row[7]:
                    properties = json.loads(row[7])
                    edge_data.update(properties)

                relationships.append(edge_data)

            return relationships

        except Exception as e:
            self.logger.error(f"Error getting outgoing relationships for {node_id}: {e}")
            return []

    async def get_incoming_relationships(self, node_id: str) -> List[Dict[str, Any]]:
        """Get all incoming relationships to a node."""
        await self.connect()

        try:
            cursor = await self._db_connection.execute(
                """
                SELECT edge_id, from_node_id, to_node_id, relation_type, 
                       timestamp, confidence_score, version, properties
                FROM edges WHERE to_node_id = ?
            """,
                (node_id,),
            )

            rows = await cursor.fetchall()
            relationships = []

            for row in rows:
                edge_data = {
                    "edge_id": row[0],
                    "from_id": row[1],
                    "to_id": row[2],
                    "relation_type": row[3],
                    "timestamp": row[4],
                    "confidence_score": row[5],
                    "version": row[6],
                }

                # Add additional properties
                if row[7]:
                    properties = json.loads(row[7])
                    edge_data.update(properties)

                relationships.append(edge_data)

            return relationships

        except Exception as e:
            self.logger.error(f"Error getting incoming relationships for {node_id}: {e}")
            return []

    async def find_shortest_path(
        self, from_node_id: str, to_node_id: str, max_hops: int = 5
    ) -> List[str]:
        """Find the shortest path between two nodes using BFS."""
        if from_node_id == to_node_id:
            return [from_node_id]

        await self.connect()

        # Check if both nodes exist
        if not await self.get_node(from_node_id) or not await self.get_node(to_node_id):
            return []

        try:
            visited = set()
            queue = deque([(from_node_id, [from_node_id])])

            for hop in range(max_hops):
                if not queue:
                    break

                level_size = len(queue)
                for _ in range(level_size):
                    current_node, path = queue.popleft()

                    if current_node in visited:
                        continue
                    visited.add(current_node)

                    # Get neighbors
                    neighbors = await self._get_neighbor_node_ids(current_node)

                    for neighbor_id in neighbors:
                        if neighbor_id == to_node_id:
                            return path + [neighbor_id]

                        if neighbor_id not in visited:
                            queue.append((neighbor_id, path + [neighbor_id]))

            return []  # No path found

        except Exception as e:
            self.logger.error(
                f"Error finding shortest path from {from_node_id} to {to_node_id}: {e}"
            )
            return []

    # Query Operations
    async def find_nodes_by_content(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Find nodes by content search."""
        await self.connect()

        try:
            cursor = await self._db_connection.execute(
                """
                SELECT node_id, content, source, creation_timestamp, 
                       rating_richness, rating_truthfulness, rating_stability, properties
                FROM nodes 
                WHERE content LIKE ? 
                LIMIT ?
            """,
                (f"%{query}%", limit),
            )

            rows = await cursor.fetchall()
            results = []

            for row in rows:
                node_data = {
                    "node_id": row[0],
                    "content": row[1],
                    "source": row[2],
                    "creation_timestamp": row[3],
                    "rating_richness": row[4],
                    "rating_truthfulness": row[5],
                    "rating_stability": row[6],
                }

                # Add additional properties
                if row[7]:
                    properties = json.loads(row[7])
                    node_data.update(properties)

                results.append(node_data)

            return results

        except Exception as e:
            self.logger.error(f"Error searching nodes by content '{query}': {e}")
            return []

    async def get_relationships_for_node(
        self, node_id: str, max_depth: int = 1
    ) -> List[Dict[str, Any]]:
        """Get relationships for a node up to a certain depth."""
        if max_depth == 1:
            outgoing = await self.get_outgoing_relationships(node_id)
            incoming = await self.get_incoming_relationships(node_id)
            return outgoing + incoming
        else:
            # Multi-hop traversal - simplified implementation
            all_relationships = []
            visited_nodes = set()
            current_level = {node_id}

            for depth in range(max_depth):
                if not current_level:
                    break

                next_level = set()

                for current_node in current_level:
                    if current_node in visited_nodes:
                        continue

                    visited_nodes.add(current_node)
                    node_relationships = await self.get_relationships_for_node(current_node, 1)

                    for rel in node_relationships:
                        rel["hop_distance"] = depth + 1
                        all_relationships.append(rel)

                        # Add connected nodes to next level
                        if rel["from_id"] == current_node:
                            next_level.add(rel["to_id"])
                        else:
                            next_level.add(rel["from_id"])

                current_level = next_level

            return all_relationships

    # Domain Model Operations
    async def save_knowledge_node(self, node: KnowledgeNode) -> str:
        """Save a KnowledgeNode to storage."""
        node_data = {
            "content": node.content,
            "source": node.source,
            "creation_timestamp": node.creation_timestamp,
            "rating_richness": node.rating_richness,
            "rating_truthfulness": node.rating_truthfulness,
            "rating_stability": node.rating_stability,
        }

        if node.node_id is None:
            node_id = await self.create_node(node_data)
            node.node_id = node_id
            return node_id
        else:
            await self.update_node(node.node_id, node_data)
            return node.node_id

    async def get_knowledge_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """Retrieve a KnowledgeNode by ID."""
        node_data = await self.get_node(node_id)
        if not node_data:
            return None

        return KnowledgeNode(
            content=node_data["content"],
            source=node_data["source"],
            creation_timestamp=node_data["creation_timestamp"],
            rating_richness=node_data["rating_richness"],
            rating_truthfulness=node_data["rating_truthfulness"],
            rating_stability=node_data["rating_stability"],
            node_id=node_id,
        )

    async def delete_knowledge_node(self, node_id: str) -> bool:
        """Delete a KnowledgeNode from storage."""
        return await self.delete_node(node_id)

    async def save_relationship(self, relationship: Relationship) -> str:
        """Save a Relationship to storage."""
        edge_metadata = {
            "timestamp": relationship.timestamp,
            "confidence_score": relationship.confidence_score,
            "version": relationship.version,
        }

        if relationship.edge_id is None:
            edge_id = await self.create_edge(
                relationship.from_id, relationship.to_id, relationship.relation_type, edge_metadata
            )
            relationship.edge_id = edge_id
            return edge_id
        else:
            await self.update_edge(relationship.edge_id, edge_metadata)
            return relationship.edge_id

    async def get_relationship(self, edge_id: str) -> Optional[Relationship]:
        """Retrieve a Relationship by ID."""
        edge_data = await self.get_edge(edge_id)
        if not edge_data:
            return None

        return Relationship(
            from_id=edge_data["from_id"],
            to_id=edge_data["to_id"],
            relation_type=edge_data["relation_type"],
            timestamp=edge_data["timestamp"],
            confidence_score=edge_data["confidence_score"],
            version=edge_data["version"],
            edge_id=edge_id,
        )

    async def delete_relationship(self, edge_id: str) -> bool:
        """Delete a Relationship from storage."""
        return await self.delete_edge(edge_id)

    # Utility Operations
    async def clear_all_data(self) -> bool:
        """Clear all data from the storage backend."""
        await self.connect()

        try:
            await self._db_connection.execute("DELETE FROM edges")
            await self._db_connection.execute("DELETE FROM nodes")
            await self._db_connection.commit()

            self.logger.info("All data cleared from SQLite database")
            return True
        except Exception as e:
            await self._db_connection.rollback()
            self.logger.error(f"Error clearing data: {e}")
            return False

    async def merge_nodes(self, node_id1: str, node_id2: str) -> bool:
        """Merge two nodes into one."""
        await self.connect()

        try:
            # Check if both nodes exist
            node1 = await self.get_node(node_id1)
            node2 = await self.get_node(node_id2)

            if not node1 or not node2:
                return False

            # Update all edges that reference node2 to reference node1
            await self._db_connection.execute(
                """
                UPDATE edges SET from_node_id = ? 
                WHERE from_node_id = ? AND to_node_id != ?
            """,
                (node_id1, node_id2, node_id1),
            )  # Avoid self-loops

            await self._db_connection.execute(
                """
                UPDATE edges SET to_node_id = ? 
                WHERE to_node_id = ? AND from_node_id != ?
            """,
                (node_id1, node_id2, node_id1),
            )  # Avoid self-loops

            # Delete node2
            await self._db_connection.execute("DELETE FROM nodes WHERE node_id = ?", (node_id2,))

            await self._db_connection.commit()

            self.logger.info(f"Successfully merged nodes {node_id1} and {node_id2}")
            return True

        except Exception as e:
            await self._db_connection.rollback()
            self.logger.error(f"Error merging nodes: {e}")
            return False

    # Performance and Caching
    def clear_caches(self) -> None:
        """Clear any internal caches."""
        self._relationship_cache.clear()
        self._content_index_cache.clear()
        self.logger.debug("SQLite storage caches cleared")

    def get_traversal_statistics(self) -> Dict[str, int]:
        """Get traversal statistics for performance monitoring."""
        return {
            "relationship_cache_entries": len(self._relationship_cache),
            "content_index_cache_entries": len(self._content_index_cache),
        }

    # Synchronous versions for backwards compatibility
    def create_node_sync(self, node_data: Dict[str, Any]) -> str:
        """Create a new node in the graph (synchronous)."""
        return self._run_async_in_sync_context(self.create_node(node_data))

    def get_node_sync(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a node by its ID (synchronous)."""
        return self._run_async_in_sync_context(self.get_node(node_id))

    def save_knowledge_node_sync(self, node: KnowledgeNode) -> str:
        """Save a KnowledgeNode to storage (synchronous)."""
        return self._run_async_in_sync_context(self.save_knowledge_node(node))

    def get_knowledge_node_sync(self, node_id: str) -> Optional[KnowledgeNode]:
        """Retrieve a KnowledgeNode by ID (synchronous)."""
        return self._run_async_in_sync_context(self.get_knowledge_node(node_id))

    # Private helper methods
    async def _create_tables(self):
        """Create database tables if they don't exist."""
        # Create nodes table
        await self._db_connection.execute(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                creation_timestamp REAL NOT NULL,
                rating_richness REAL NOT NULL,
                rating_truthfulness REAL NOT NULL,
                rating_stability REAL NOT NULL,
                properties TEXT,  -- JSON string for additional properties
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Create edges table
        await self._db_connection.execute(
            """
            CREATE TABLE IF NOT EXISTS edges (
                edge_id TEXT PRIMARY KEY,
                from_node_id TEXT NOT NULL,
                to_node_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                confidence_score REAL NOT NULL,
                version INTEGER NOT NULL,
                properties TEXT,  -- JSON string for additional properties
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (from_node_id) REFERENCES nodes (node_id) ON DELETE CASCADE,
                FOREIGN KEY (to_node_id) REFERENCES nodes (node_id) ON DELETE CASCADE
            )
        """
        )

        # Create indexes for performance
        await self._db_connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_nodes_content ON nodes (content)
        """
        )

        await self._db_connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_edges_from_node ON edges (from_node_id)
        """
        )

        await self._db_connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_edges_to_node ON edges (to_node_id)
        """
        )

        await self._db_connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_edges_relation_type ON edges (relation_type)
        """
        )

        await self._db_connection.commit()

    async def _get_neighbor_node_ids(self, node_id: str) -> List[str]:
        """Get neighbor node IDs efficiently."""
        try:
            cursor = await self._db_connection.execute(
                """
                SELECT DISTINCT 
                    CASE 
                        WHEN from_node_id = ? THEN to_node_id 
                        ELSE from_node_id 
                    END as neighbor_id
                FROM edges 
                WHERE from_node_id = ? OR to_node_id = ?
            """,
                (node_id, node_id, node_id),
            )

            rows = await cursor.fetchall()
            return [row[0] for row in rows]

        except Exception as e:
            self.logger.error(f"Error getting neighbors for {node_id}: {e}")
            return []

    def _run_async_in_sync_context(self, coroutine, timeout=10):
        """Helper function to run an async coroutine in a sync context."""

        def run_in_thread():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(coroutine)
                finally:
                    loop.close()
            except Exception as e:
                self.logger.error(f"Error in thread: {e}")
                raise

        try:
            with ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=timeout)
        except Exception as e:
            self.logger.error(f"Error running async in sync context: {e}")
            raise
