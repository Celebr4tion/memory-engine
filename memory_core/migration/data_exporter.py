"""
Data export utilities for Memory Engine.
"""

import json
import csv
import xml.etree.ElementTree as ET
import time
import asyncio
from typing import Dict, List, Any, Optional, TextIO
from dataclasses import dataclass
from enum import Enum
import logging
import os

logger = logging.getLogger(__name__)


class ExportFormat(Enum):
    """Supported export formats."""

    JSON = "json"
    CSV = "csv"
    XML = "xml"
    GRAPHML = "graphml"
    CYPHER = "cypher"
    GREMLIN = "gremlin"
    RDF = "rdf"
    NETWORKX = "networkx"


@dataclass
class ExportConfig:
    """Configuration for data export."""

    format: ExportFormat
    include_embeddings: bool = False
    include_metadata: bool = True
    include_relationships: bool = True
    pretty_format: bool = True
    batch_size: int = 1000
    max_file_size_mb: int = 100
    split_large_files: bool = True


class DataExporter:
    """Export Memory Engine data to various formats."""

    def __init__(self, config: ExportConfig):
        self.config = config
        self._exported_count = 0
        self._file_count = 0

    async def export_knowledge_graph(self, engine: Any, output_path: str) -> Dict[str, Any]:
        """Export complete knowledge graph."""
        start_time = time.time()

        try:
            # Get all nodes and relationships
            nodes = await self._get_all_nodes(engine)
            relationships = (
                await self._get_all_relationships(engine)
                if self.config.include_relationships
                else []
            )

            # Export based on format
            if self.config.format == ExportFormat.JSON:
                await self._export_json(nodes, relationships, output_path)
            elif self.config.format == ExportFormat.CSV:
                await self._export_csv(nodes, relationships, output_path)
            elif self.config.format == ExportFormat.XML:
                await self._export_xml(nodes, relationships, output_path)
            elif self.config.format == ExportFormat.GRAPHML:
                await self._export_graphml(nodes, relationships, output_path)
            elif self.config.format == ExportFormat.CYPHER:
                await self._export_cypher(nodes, relationships, output_path)
            elif self.config.format == ExportFormat.GREMLIN:
                await self._export_gremlin(nodes, relationships, output_path)
            elif self.config.format == ExportFormat.RDF:
                await self._export_rdf(nodes, relationships, output_path)
            elif self.config.format == ExportFormat.NETWORKX:
                await self._export_networkx(nodes, relationships, output_path)
            else:
                raise ValueError(f"Unsupported export format: {self.config.format}")

            return {
                "success": True,
                "nodes_exported": len(nodes),
                "relationships_exported": len(relationships),
                "files_created": self._file_count,
                "duration": time.time() - start_time,
                "output_path": output_path,
            }

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return {"success": False, "error": str(e), "duration": time.time() - start_time}

    async def _get_all_nodes(self, engine: Any) -> List[Dict[str, Any]]:
        """Get all nodes from the engine."""
        try:
            if hasattr(engine, "get_all_knowledge_nodes"):
                nodes = await engine.get_all_knowledge_nodes()
            elif hasattr(engine, "query_all_nodes"):
                nodes = await engine.query_all_nodes()
            else:
                raise NotImplementedError("Engine doesn't support node retrieval")

            # Convert to dict format
            return [self._node_to_dict(node) for node in nodes]

        except Exception as e:
            logger.error(f"Failed to get nodes: {e}")
            return []

    async def _get_all_relationships(self, engine: Any) -> List[Dict[str, Any]]:
        """Get all relationships from the engine."""
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

        # Apply export filters
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

    async def _export_json(self, nodes: List[Dict], relationships: List[Dict], output_path: str):
        """Export to JSON format."""
        data = {
            "metadata": {
                "export_time": time.time(),
                "format": "json",
                "node_count": len(nodes),
                "relationship_count": len(relationships),
            },
            "nodes": nodes,
            "relationships": relationships,
        }

        if self._should_split_file(data, output_path):
            await self._export_json_split(data, output_path)
        else:
            with open(output_path, "w", encoding="utf-8") as f:
                if self.config.pretty_format:
                    json.dump(data, f, indent=2, ensure_ascii=False, default=str)
                else:
                    json.dump(data, f, ensure_ascii=False, default=str)
            self._file_count = 1

    async def _export_json_split(self, data: Dict, output_path: str):
        """Export JSON in multiple files."""
        base_path = output_path.rsplit(".", 1)[0]

        # Export metadata
        metadata_file = f"{base_path}_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(data["metadata"], f, indent=2, default=str)

        # Export nodes in batches
        nodes = data["nodes"]
        node_files = await self._split_and_export_json(nodes, f"{base_path}_nodes", "nodes")

        # Export relationships in batches
        relationships = data["relationships"]
        rel_files = await self._split_and_export_json(
            relationships, f"{base_path}_relationships", "relationships"
        )

        self._file_count = 1 + len(node_files) + len(rel_files)

    async def _split_and_export_json(
        self, items: List[Dict], base_path: str, item_type: str
    ) -> List[str]:
        """Split items into multiple JSON files."""
        files = []
        batch_size = self.config.batch_size

        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            file_path = f"{base_path}_{i // batch_size + 1}.json"

            batch_data = {
                "type": item_type,
                "batch_index": i // batch_size + 1,
                "item_count": len(batch),
                "items": batch,
            }

            with open(file_path, "w", encoding="utf-8") as f:
                if self.config.pretty_format:
                    json.dump(batch_data, f, indent=2, ensure_ascii=False, default=str)
                else:
                    json.dump(batch_data, f, ensure_ascii=False, default=str)

            files.append(file_path)

        return files

    async def _export_csv(self, nodes: List[Dict], relationships: List[Dict], output_path: str):
        """Export to CSV format."""
        base_path = output_path.rsplit(".", 1)[0]

        # Export nodes
        nodes_file = f"{base_path}_nodes.csv"
        await self._export_nodes_csv(nodes, nodes_file)

        # Export relationships
        if relationships:
            relationships_file = f"{base_path}_relationships.csv"
            await self._export_relationships_csv(relationships, relationships_file)
            self._file_count = 2
        else:
            self._file_count = 1

    async def _export_nodes_csv(self, nodes: List[Dict], output_path: str):
        """Export nodes to CSV."""
        if not nodes:
            return

        # Get all possible fields
        all_fields = set()
        for node in nodes:
            all_fields.update(node.keys())

        all_fields = sorted(all_fields)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields)
            writer.writeheader()

            for node in nodes:
                # Flatten complex fields
                row = {}
                for field in all_fields:
                    value = node.get(field, "")
                    if isinstance(value, (dict, list)):
                        row[field] = json.dumps(value, default=str)
                    else:
                        row[field] = str(value) if value is not None else ""

                writer.writerow(row)

    async def _export_relationships_csv(self, relationships: List[Dict], output_path: str):
        """Export relationships to CSV."""
        if not relationships:
            return

        # Get all possible fields
        all_fields = set()
        for rel in relationships:
            all_fields.update(rel.keys())

        all_fields = sorted(all_fields)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields)
            writer.writeheader()

            for rel in relationships:
                row = {}
                for field in all_fields:
                    value = rel.get(field, "")
                    if isinstance(value, (dict, list)):
                        row[field] = json.dumps(value, default=str)
                    else:
                        row[field] = str(value) if value is not None else ""

                writer.writerow(row)

    async def _export_xml(self, nodes: List[Dict], relationships: List[Dict], output_path: str):
        """Export to XML format."""
        root = ET.Element("knowledge_graph")

        # Add metadata
        metadata = ET.SubElement(root, "metadata")
        ET.SubElement(metadata, "export_time").text = str(time.time())
        ET.SubElement(metadata, "node_count").text = str(len(nodes))
        ET.SubElement(metadata, "relationship_count").text = str(len(relationships))

        # Add nodes
        nodes_elem = ET.SubElement(root, "nodes")
        for node in nodes:
            node_elem = ET.SubElement(nodes_elem, "node")
            self._dict_to_xml(node, node_elem)

        # Add relationships
        if relationships:
            rels_elem = ET.SubElement(root, "relationships")
            for rel in relationships:
                rel_elem = ET.SubElement(rels_elem, "relationship")
                self._dict_to_xml(rel, rel_elem)

        # Write to file
        if self.config.pretty_format:
            self._indent_xml(root)

        tree = ET.ElementTree(root)
        tree.write(output_path, encoding="utf-8", xml_declaration=True)
        self._file_count = 1

    def _dict_to_xml(self, data: Dict, parent: ET.Element):
        """Convert dictionary to XML elements."""
        for key, value in data.items():
            elem = ET.SubElement(parent, str(key))
            if isinstance(value, dict):
                self._dict_to_xml(value, elem)
            elif isinstance(value, list):
                for item in value:
                    item_elem = ET.SubElement(elem, "item")
                    if isinstance(item, dict):
                        self._dict_to_xml(item, item_elem)
                    else:
                        item_elem.text = str(item)
            else:
                elem.text = str(value) if value is not None else ""

    def _indent_xml(self, elem: ET.Element, level: int = 0):
        """Add indentation to XML for pretty printing."""
        indent = "\\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = indent + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = indent
            for elem in elem:
                self._indent_xml(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = indent
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = indent

    async def _export_graphml(self, nodes: List[Dict], relationships: List[Dict], output_path: str):
        """Export to GraphML format."""
        root = ET.Element(
            "graphml",
            {
                "xmlns": "http://graphml.graphdrawing.org/xmlns",
                "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
                "xsi:schemaLocation": "http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd",
            },
        )

        # Define keys for node and edge attributes
        node_keys = set()
        edge_keys = set()

        for node in nodes:
            node_keys.update(node.keys())

        for rel in relationships:
            edge_keys.update(rel.keys())

        # Add key definitions
        key_id = 0
        key_mapping = {}

        for key in sorted(node_keys):
            key_elem = ET.SubElement(
                root,
                "key",
                {"id": f"d{key_id}", "for": "node", "attr.name": str(key), "attr.type": "string"},
            )
            key_mapping[f"node_{key}"] = f"d{key_id}"
            key_id += 1

        for key in sorted(edge_keys):
            key_elem = ET.SubElement(
                root,
                "key",
                {"id": f"d{key_id}", "for": "edge", "attr.name": str(key), "attr.type": "string"},
            )
            key_mapping[f"edge_{key}"] = f"d{key_id}"
            key_id += 1

        # Create graph
        graph = ET.SubElement(root, "graph", {"id": "G", "edgedefault": "directed"})

        # Add nodes
        for i, node in enumerate(nodes):
            node_id = node.get("id") or node.get("node_id") or f"n{i}"
            node_elem = ET.SubElement(graph, "node", {"id": str(node_id)})

            for key, value in node.items():
                if key in ["id", "node_id"]:
                    continue

                data_elem = ET.SubElement(
                    node_elem, "data", {"key": key_mapping.get(f"node_{key}", f"d{key_id}")}
                )

                if isinstance(value, (dict, list)):
                    data_elem.text = json.dumps(value, default=str)
                else:
                    data_elem.text = str(value) if value is not None else ""

        # Add edges
        for i, rel in enumerate(relationships):
            source = rel.get("source_id") or rel.get("from_id") or f"n{i}"
            target = rel.get("target_id") or rel.get("to_id") or f"n{i+1}"
            edge_id = rel.get("id") or f"e{i}"

            edge_elem = ET.SubElement(
                graph, "edge", {"id": str(edge_id), "source": str(source), "target": str(target)}
            )

            for key, value in rel.items():
                if key in ["id", "source_id", "target_id", "from_id", "to_id"]:
                    continue

                data_elem = ET.SubElement(
                    edge_elem, "data", {"key": key_mapping.get(f"edge_{key}", f"d{key_id}")}
                )

                if isinstance(value, (dict, list)):
                    data_elem.text = json.dumps(value, default=str)
                else:
                    data_elem.text = str(value) if value is not None else ""

        # Write to file
        if self.config.pretty_format:
            self._indent_xml(root)

        tree = ET.ElementTree(root)
        tree.write(output_path, encoding="utf-8", xml_declaration=True)
        self._file_count = 1

    async def _export_cypher(self, nodes: List[Dict], relationships: List[Dict], output_path: str):
        """Export to Cypher statements."""
        with open(output_path, "w", encoding="utf-8") as f:
            # Write header comment
            f.write("// Knowledge Graph Export - Cypher Statements\\n")
            f.write(f"// Generated at: {time.ctime()}\\n")
            f.write(f"// Nodes: {len(nodes)}, Relationships: {len(relationships)}\\n\\n")

            # Create nodes
            f.write("// Create nodes\\n")
            for i, node in enumerate(nodes):
                node_id = node.get("id") or node.get("node_id") or f"n{i}"
                labels = node.get("labels", ["Node"])
                if isinstance(labels, str):
                    labels = [labels]

                label_str = ":".join(labels)

                # Build properties
                props = {}
                for key, value in node.items():
                    if key not in ["id", "node_id", "labels"]:
                        if isinstance(value, str):
                            props[key] = f"'{value}'"
                        elif isinstance(value, (dict, list)):
                            props[key] = f"'{json.dumps(value, default=str)}'"
                        else:
                            props[key] = str(value)

                props_str = ", ".join([f"{k}: {v}" for k, v in props.items()])

                f.write(f"CREATE (n{node_id}:{label_str} {{{props_str}}});\\n")

            f.write("\\n// Create relationships\\n")
            for i, rel in enumerate(relationships):
                source = rel.get("source_id") or rel.get("from_id") or f"n{i}"
                target = rel.get("target_id") or rel.get("to_id") or f"n{i+1}"
                rel_type = rel.get("type", "RELATED_TO")

                # Build properties
                props = {}
                for key, value in rel.items():
                    if key not in ["source_id", "target_id", "from_id", "to_id", "type"]:
                        if isinstance(value, str):
                            props[key] = f"'{value}'"
                        elif isinstance(value, (dict, list)):
                            props[key] = f"'{json.dumps(value, default=str)}'"
                        else:
                            props[key] = str(value)

                props_str = ", ".join([f"{k}: {v}" for k, v in props.items()])
                props_clause = f" {{{props_str}}}" if props_str else ""

                f.write(f"MATCH (a), (b) WHERE a.id = '{source}' AND b.id = '{target}' ")
                f.write(f"CREATE (a)-[:{rel_type}{props_clause}]->(b);\\n")

        self._file_count = 1

    async def _export_gremlin(self, nodes: List[Dict], relationships: List[Dict], output_path: str):
        """Export to Gremlin statements."""
        with open(output_path, "w", encoding="utf-8") as f:
            # Write header comment
            f.write("// Knowledge Graph Export - Gremlin Statements\\n")
            f.write(f"// Generated at: {time.ctime()}\\n")
            f.write(f"// Nodes: {len(nodes)}, Relationships: {len(relationships)}\\n\\n")

            # Create nodes
            f.write("// Create vertices\\n")
            for i, node in enumerate(nodes):
                node_id = node.get("id") or node.get("node_id") or f"n{i}"
                label = node.get("label", "Node")

                f.write(f"g.addV('{label}').property(id, '{node_id}')")

                for key, value in node.items():
                    if key not in ["id", "node_id", "label"]:
                        if isinstance(value, str):
                            f.write(f".property('{key}', '{value}')")
                        elif isinstance(value, (dict, list)):
                            f.write(f".property('{key}', '{json.dumps(value, default=str)}')")
                        else:
                            f.write(f".property('{key}', {value})")

                f.write(";\\n")

            f.write("\\n// Create edges\\n")
            for rel in relationships:
                source = rel.get("source_id") or rel.get("from_id")
                target = rel.get("target_id") or rel.get("to_id")
                label = rel.get("type", "edge")

                if source and target:
                    f.write(f"g.V('{source}').addE('{label}').to(g.V('{target}'))")

                    for key, value in rel.items():
                        if key not in ["source_id", "target_id", "from_id", "to_id", "type"]:
                            if isinstance(value, str):
                                f.write(f".property('{key}', '{value}')")
                            elif isinstance(value, (dict, list)):
                                f.write(f".property('{key}', '{json.dumps(value, default=str)}')")
                            else:
                                f.write(f".property('{key}', {value})")

                    f.write(";\\n")

        self._file_count = 1

    async def _export_rdf(self, nodes: List[Dict], relationships: List[Dict], output_path: str):
        """Export to RDF format."""
        with open(output_path, "w", encoding="utf-8") as f:
            # Write RDF header
            f.write("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\\n")
            f.write("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\\n")
            f.write("@prefix kg: <http://memory-engine.org/kg#> .\\n\\n")

            # Export nodes as RDF triples
            for i, node in enumerate(nodes):
                node_id = node.get("id") or node.get("node_id") or f"n{i}"
                uri = f"kg:node_{node_id}"

                f.write(f"{uri} rdf:type kg:Node .\\n")

                for key, value in node.items():
                    if key not in ["id", "node_id"]:
                        predicate = f"kg:{key}"

                        if isinstance(value, str):
                            f.write(f'{uri} {predicate} \\"{value}\\" .\\n')
                        elif isinstance(value, (int, float)):
                            f.write(f"{uri} {predicate} {value} .\\n")
                        else:
                            f.write(f'{uri} {predicate} \\"{str(value)}\\" .\\n')

                f.write("\\n")

            # Export relationships as RDF triples
            for rel in relationships:
                source = rel.get("source_id") or rel.get("from_id")
                target = rel.get("target_id") or rel.get("to_id")

                if source and target:
                    source_uri = f"kg:node_{source}"
                    target_uri = f"kg:node_{target}"
                    predicate = f"kg:{rel.get('type', 'relatedTo')}"

                    f.write(f"{source_uri} {predicate} {target_uri} .\\n")

        self._file_count = 1

    async def _export_networkx(
        self, nodes: List[Dict], relationships: List[Dict], output_path: str
    ):
        """Export to NetworkX-compatible format."""
        try:
            import networkx as nx
            import pickle

            # Create NetworkX graph
            G = nx.DiGraph()

            # Add nodes
            for node in nodes:
                node_id = node.get("id") or node.get("node_id")
                if node_id:
                    # Remove complex objects that can't be pickled
                    attrs = {}
                    for key, value in node.items():
                        if key not in ["id", "node_id"]:
                            if isinstance(value, (str, int, float, bool)):
                                attrs[key] = value
                            else:
                                attrs[key] = str(value)

                    G.add_node(node_id, **attrs)

            # Add edges
            for rel in relationships:
                source = rel.get("source_id") or rel.get("from_id")
                target = rel.get("target_id") or rel.get("to_id")

                if source and target and source in G.nodes and target in G.nodes:
                    # Remove complex objects
                    attrs = {}
                    for key, value in rel.items():
                        if key not in ["source_id", "target_id", "from_id", "to_id"]:
                            if isinstance(value, (str, int, float, bool)):
                                attrs[key] = value
                            else:
                                attrs[key] = str(value)

                    G.add_edge(source, target, **attrs)

            # Save as pickle
            with open(output_path, "wb") as f:
                pickle.dump(G, f)

            self._file_count = 1

        except ImportError:
            raise RuntimeError("NetworkX is required for NetworkX export format")

    def _should_split_file(self, data: Any, output_path: str) -> bool:
        """Check if file should be split based on size."""
        if not self.config.split_large_files:
            return False

        # Estimate file size
        try:
            import sys

            estimated_size = sys.getsizeof(json.dumps(data, default=str))
            max_size = self.config.max_file_size_mb * 1024 * 1024
            return estimated_size > max_size
        except:
            return False
