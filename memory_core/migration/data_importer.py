"""
Data import utilities for Memory Engine.
"""

import json
import csv
import xml.etree.ElementTree as ET
import time
import asyncio
from typing import Dict, List, Any, Optional, Iterator
from dataclasses import dataclass
from enum import Enum
import logging
import os
import glob

logger = logging.getLogger(__name__)


@dataclass
class ImportConfig:
    """Configuration for data import."""

    validate_data: bool = True
    merge_duplicates: bool = True
    batch_size: int = 1000
    skip_errors: bool = True
    update_existing: bool = False
    preserve_ids: bool = True


class DataImporter:
    """Import data into Memory Engine from various formats."""

    def __init__(self, config: ImportConfig):
        self.config = config
        self._imported_count = 0
        self._error_count = 0
        self._skipped_count = 0
        self.errors: List[str] = []

    async def import_from_file(
        self, file_path: str, engine: Any, file_format: str = None
    ) -> Dict[str, Any]:
        """Import data from a file."""
        start_time = time.time()

        try:
            # Detect format if not specified
            if not file_format:
                file_format = self._detect_format(file_path)

            # Import based on format
            if file_format == "json":
                await self._import_json(file_path, engine)
            elif file_format == "csv":
                await self._import_csv(file_path, engine)
            elif file_format == "xml":
                await self._import_xml(file_path, engine)
            elif file_format == "graphml":
                await self._import_graphml(file_path, engine)
            elif file_format == "cypher":
                await self._import_cypher(file_path, engine)
            elif file_format == "gremlin":
                await self._import_gremlin(file_path, engine)
            elif file_format == "rdf":
                await self._import_rdf(file_path, engine)
            elif file_format == "networkx":
                await self._import_networkx(file_path, engine)
            else:
                raise ValueError(f"Unsupported import format: {file_format}")

            return {
                "success": True,
                "imported_count": self._imported_count,
                "error_count": self._error_count,
                "skipped_count": self._skipped_count,
                "duration": time.time() - start_time,
                "errors": self.errors,
            }

        except Exception as e:
            logger.error(f"Import failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "duration": time.time() - start_time,
                "errors": self.errors,
            }

    async def import_from_directory(self, directory_path: str, engine: Any) -> Dict[str, Any]:
        """Import data from multiple files in a directory."""
        start_time = time.time()
        results = []

        try:
            # Find all supported files
            patterns = [
                "*.json",
                "*.csv",
                "*.xml",
                "*.graphml",
                "*.cypher",
                "*.gremlin",
                "*.rdf",
                "*.pkl",
            ]
            files = []

            for pattern in patterns:
                files.extend(glob.glob(os.path.join(directory_path, pattern)))

            if not files:
                return {
                    "success": False,
                    "error": "No supported files found in directory",
                    "duration": time.time() - start_time,
                }

            # Import each file
            for file_path in sorted(files):
                try:
                    result = await self.import_from_file(file_path, engine)
                    results.append({"file": os.path.basename(file_path), "result": result})
                except Exception as e:
                    logger.error(f"Failed to import {file_path}: {e}")
                    results.append(
                        {
                            "file": os.path.basename(file_path),
                            "result": {"success": False, "error": str(e)},
                        }
                    )

            # Aggregate results
            total_imported = sum(r["result"].get("imported_count", 0) for r in results)
            total_errors = sum(r["result"].get("error_count", 0) for r in results)
            successful_files = sum(1 for r in results if r["result"].get("success", False))

            return {
                "success": successful_files > 0,
                "files_processed": len(results),
                "successful_files": successful_files,
                "total_imported": total_imported,
                "total_errors": total_errors,
                "duration": time.time() - start_time,
                "file_results": results,
            }

        except Exception as e:
            logger.error(f"Directory import failed: {e}")
            return {"success": False, "error": str(e), "duration": time.time() - start_time}

    def _detect_format(self, file_path: str) -> str:
        """Detect file format from extension and content."""
        _, ext = os.path.splitext(file_path.lower())

        if ext == ".json":
            return "json"
        elif ext == ".csv":
            return "csv"
        elif ext == ".xml":
            return "xml"
        elif ext == ".graphml":
            return "graphml"
        elif ext in [".cypher", ".cql"]:
            return "cypher"
        elif ext in [".gremlin", ".groovy"]:
            return "gremlin"
        elif ext in [".rdf", ".ttl", ".n3"]:
            return "rdf"
        elif ext in [".pkl", ".pickle"]:
            return "networkx"
        else:
            # Try to detect from content
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()

                    if first_line.startswith("{") or first_line.startswith("["):
                        return "json"
                    elif first_line.startswith("<?xml") or first_line.startswith("<graphml"):
                        return "graphml" if "graphml" in first_line else "xml"
                    elif first_line.startswith("CREATE") or first_line.startswith("MATCH"):
                        return "cypher"
                    elif first_line.startswith("g.") or "addV" in first_line:
                        return "gremlin"
                    elif "@prefix" in first_line or first_line.endswith(" ."):
                        return "rdf"
                    else:
                        return "csv"  # Default fallback
            except:
                return "json"  # Final fallback

    async def _import_json(self, file_path: str, engine: Any):
        """Import from JSON format."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle different JSON structures
        if isinstance(data, dict):
            if "nodes" in data:
                # Standard export format
                await self._import_nodes(data.get("nodes", []), engine)
                await self._import_relationships(data.get("relationships", []), engine)
            elif "items" in data:
                # Batch export format
                await self._import_nodes(data["items"], engine)
            else:
                # Single node
                await self._import_single_node(data, engine)
        elif isinstance(data, list):
            # List of nodes
            await self._import_nodes(data, engine)

    async def _import_nodes(self, nodes: List[Dict], engine: Any):
        """Import a list of nodes."""
        batch = []

        for node_data in nodes:
            try:
                if self.config.validate_data:
                    node_data = self._validate_node_data(node_data)

                batch.append(node_data)

                if len(batch) >= self.config.batch_size:
                    await self._import_node_batch(batch, engine)
                    batch = []

            except Exception as e:
                self._error_count += 1
                error_msg = f"Failed to process node: {str(e)}"
                self.errors.append(error_msg)

                if not self.config.skip_errors:
                    raise

        # Import remaining batch
        if batch:
            await self._import_node_batch(batch, engine)

    async def _import_node_batch(self, nodes: List[Dict], engine: Any):
        """Import a batch of nodes."""
        for node_data in nodes:
            try:
                await self._import_single_node(node_data, engine)
            except Exception as e:
                self._error_count += 1
                error_msg = f"Failed to import node: {str(e)}"
                self.errors.append(error_msg)

                if not self.config.skip_errors:
                    raise

    async def _import_single_node(self, node_data: Dict, engine: Any):
        """Import a single node."""
        try:
            # Check for existing node if merge duplicates
            if self.config.merge_duplicates or self.config.update_existing:
                node_id = node_data.get("id") or node_data.get("node_id")
                if node_id:
                    existing_node = await self._get_existing_node(engine, node_id)
                    if existing_node:
                        if self.config.update_existing:
                            await self._update_node(engine, node_id, node_data)
                            self._imported_count += 1
                            return
                        elif self.config.merge_duplicates:
                            self._skipped_count += 1
                            return

            # Create new node
            if hasattr(engine, "create_knowledge_node"):
                await engine.create_knowledge_node(**node_data)
            elif hasattr(engine, "add_node"):
                await engine.add_node(node_data)
            else:
                raise NotImplementedError("Engine doesn't support node creation")

            self._imported_count += 1

        except Exception as e:
            raise RuntimeError(f"Failed to import node: {str(e)}")

    async def _import_relationships(self, relationships: List[Dict], engine: Any):
        """Import relationships."""
        for rel_data in relationships:
            try:
                if self.config.validate_data:
                    rel_data = self._validate_relationship_data(rel_data)

                await self._import_single_relationship(rel_data, engine)

            except Exception as e:
                self._error_count += 1
                error_msg = f"Failed to import relationship: {str(e)}"
                self.errors.append(error_msg)

                if not self.config.skip_errors:
                    raise

    async def _import_single_relationship(self, rel_data: Dict, engine: Any):
        """Import a single relationship."""
        try:
            if hasattr(engine, "create_relationship"):
                await engine.create_relationship(**rel_data)
            elif hasattr(engine, "add_relationship"):
                await engine.add_relationship(rel_data)
            else:
                logger.warning("Engine doesn't support relationship creation")

        except Exception as e:
            raise RuntimeError(f"Failed to import relationship: {str(e)}")

    def _validate_node_data(self, node_data: Dict) -> Dict:
        """Validate and clean node data."""
        if not isinstance(node_data, dict):
            raise ValueError("Node data must be a dictionary")

        # Ensure required fields
        if not node_data.get("content") and not node_data.get("text"):
            if "id" not in node_data:
                raise ValueError("Node must have content, text, or id")

        # Clean data
        cleaned = {}
        for key, value in node_data.items():
            if value is not None:
                cleaned[key] = value

        return cleaned

    def _validate_relationship_data(self, rel_data: Dict) -> Dict:
        """Validate and clean relationship data."""
        if not isinstance(rel_data, dict):
            raise ValueError("Relationship data must be a dictionary")

        # Ensure required fields
        source_id = rel_data.get("source_id") or rel_data.get("from_id")
        target_id = rel_data.get("target_id") or rel_data.get("to_id")

        if not source_id or not target_id:
            raise ValueError("Relationship must have source and target IDs")

        # Standardize field names
        cleaned = rel_data.copy()
        if "from_id" in cleaned:
            cleaned["source_id"] = cleaned.pop("from_id")
        if "to_id" in cleaned:
            cleaned["target_id"] = cleaned.pop("to_id")

        return cleaned

    async def _get_existing_node(self, engine: Any, node_id: str) -> Optional[Any]:
        """Check if node exists."""
        try:
            if hasattr(engine, "get_knowledge_node"):
                return await engine.get_knowledge_node(node_id)
            elif hasattr(engine, "get_node"):
                return await engine.get_node(node_id)
        except:
            pass
        return None

    async def _update_node(self, engine: Any, node_id: str, node_data: Dict):
        """Update existing node."""
        try:
            if hasattr(engine, "update_knowledge_node"):
                await engine.update_knowledge_node(node_id, **node_data)
            elif hasattr(engine, "update_node"):
                await engine.update_node(node_id, node_data)
            else:
                logger.warning("Engine doesn't support node updates")
        except Exception as e:
            raise RuntimeError(f"Failed to update node {node_id}: {str(e)}")

    async def _import_csv(self, file_path: str, engine: Any):
        """Import from CSV format."""
        with open(file_path, "r", encoding="utf-8") as f:
            # Try to detect if this is nodes or relationships CSV
            first_line = f.readline()
            f.seek(0)

            if any(
                field in first_line.lower()
                for field in ["source_id", "target_id", "from_id", "to_id"]
            ):
                await self._import_relationships_csv(file_path, engine)
            else:
                await self._import_nodes_csv(file_path, engine)

    async def _import_nodes_csv(self, file_path: str, engine: Any):
        """Import nodes from CSV."""
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            nodes = []

            for row in reader:
                # Clean empty values
                node_data = {}
                for key, value in row.items():
                    if value and value.strip():
                        # Try to parse JSON values
                        if value.startswith(("{", "[")):
                            try:
                                node_data[key] = json.loads(value)
                            except:
                                node_data[key] = value
                        else:
                            node_data[key] = value

                nodes.append(node_data)

            await self._import_nodes(nodes, engine)

    async def _import_relationships_csv(self, file_path: str, engine: Any):
        """Import relationships from CSV."""
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            relationships = []

            for row in reader:
                # Clean empty values
                rel_data = {}
                for key, value in row.items():
                    if value and value.strip():
                        # Try to parse JSON values
                        if value.startswith(("{", "[")):
                            try:
                                rel_data[key] = json.loads(value)
                            except:
                                rel_data[key] = value
                        else:
                            rel_data[key] = value

                relationships.append(rel_data)

            await self._import_relationships(relationships, engine)

    async def _import_xml(self, file_path: str, engine: Any):
        """Import from XML format."""
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Find nodes
        nodes_elem = root.find("nodes")
        if nodes_elem is not None:
            nodes = []
            for node_elem in nodes_elem.findall("node"):
                node_data = self._xml_to_dict(node_elem)
                nodes.append(node_data)

            await self._import_nodes(nodes, engine)

        # Find relationships
        rels_elem = root.find("relationships")
        if rels_elem is not None:
            relationships = []
            for rel_elem in rels_elem.findall("relationship"):
                rel_data = self._xml_to_dict(rel_elem)
                relationships.append(rel_data)

            await self._import_relationships(relationships, engine)

    def _xml_to_dict(self, elem: ET.Element) -> Dict:
        """Convert XML element to dictionary."""
        result = {}

        for child in elem:
            if len(child) == 0:
                # Leaf node
                result[child.tag] = child.text
            else:
                # Has children
                if child.tag not in result:
                    result[child.tag] = []
                result[child.tag].append(self._xml_to_dict(child))

        return result

    async def _import_graphml(self, file_path: str, engine: Any):
        """Import from GraphML format."""
        try:
            import networkx as nx

            # Read GraphML
            G = nx.read_graphml(file_path)

            # Convert to Memory Engine format
            nodes = []
            for node_id, attrs in G.nodes(data=True):
                node_data = {"id": node_id, **attrs}
                nodes.append(node_data)

            relationships = []
            for source, target, attrs in G.edges(data=True):
                rel_data = {"source_id": source, "target_id": target, **attrs}
                relationships.append(rel_data)

            await self._import_nodes(nodes, engine)
            await self._import_relationships(relationships, engine)

        except ImportError:
            raise RuntimeError("NetworkX is required for GraphML import")

    async def _import_cypher(self, file_path: str, engine: Any):
        """Import from Cypher statements."""
        # This would require a Cypher parser
        # For now, we'll just log that it's not implemented
        logger.warning("Cypher import not fully implemented - requires Cypher parser")
        raise NotImplementedError("Cypher import requires additional parsing logic")

    async def _import_gremlin(self, file_path: str, engine: Any):
        """Import from Gremlin statements."""
        # This would require a Gremlin parser
        logger.warning("Gremlin import not fully implemented - requires Gremlin parser")
        raise NotImplementedError("Gremlin import requires additional parsing logic")

    async def _import_rdf(self, file_path: str, engine: Any):
        """Import from RDF format."""
        # This would require an RDF parser like rdflib
        logger.warning("RDF import not fully implemented - requires RDF parser")
        raise NotImplementedError("RDF import requires rdflib or similar library")

    async def _import_networkx(self, file_path: str, engine: Any):
        """Import from NetworkX pickle format."""
        try:
            import pickle
            import networkx as nx

            # Load NetworkX graph
            with open(file_path, "rb") as f:
                G = pickle.load(f)

            if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
                raise ValueError("File does not contain a NetworkX graph")

            # Convert to Memory Engine format
            nodes = []
            for node_id, attrs in G.nodes(data=True):
                node_data = {"id": str(node_id), **attrs}
                nodes.append(node_data)

            relationships = []
            for source, target, attrs in G.edges(data=True):
                rel_data = {"source_id": str(source), "target_id": str(target), **attrs}
                relationships.append(rel_data)

            await self._import_nodes(nodes, engine)
            await self._import_relationships(relationships, engine)

        except ImportError:
            raise RuntimeError("NetworkX is required for NetworkX pickle import")
