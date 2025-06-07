"""
Enhanced MCP endpoint implementation for the Memory Engine.

This module extends the basic MCP interface with advanced querying capabilities,
knowledge synthesis, bulk operations, and analytics.
"""

import json
import logging
import time
import uuid
import asyncio
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import math

from memory_core.core.knowledge_engine import KnowledgeEngine
from memory_core.ingestion.advanced_extractor import extract_knowledge_units, process_extracted_units
from memory_core.ingestion.relationship_extractor import analyze_and_create_relationships
from memory_core.embeddings.embedding_manager import EmbeddingManager
from memory_core.embeddings.vector_store import VectorStoreMilvus
from memory_core.model.knowledge_node import KnowledgeNode
from memory_core.model.relationship import Relationship


class EnhancedMemoryEngineMCP:
    """
    Enhanced MCP interface for the Memory Engine with advanced querying capabilities.
    
    This class extends the basic MCP interface with:
    - Advanced graph queries (multi-hop traversal, pattern matching)
    - Knowledge synthesis (summaries, question answering)
    - Bulk operations (batch processing, export)
    - Analytics (coverage analysis, quality metrics)
    """
    
    def __init__(self, host: str = "localhost", port: int = 8182):
        """
        Initialize the enhanced MCP interface.
        
        Args:
            host: JanusGraph server host
            port: JanusGraph server port
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize the KnowledgeEngine
        self.engine = KnowledgeEngine(host=host, port=port, enable_versioning=True)
        self.engine.connect()
        
        # Initialize the vector store and embedding manager
        self.vector_store = None
        self.embedding_manager = None
        
        try:
            self.vector_store = VectorStoreMilvus(host="localhost", port=19530)
            if self.vector_store.connect():
                self.embedding_manager = EmbeddingManager(self.vector_store)
                self.logger.info("Vector storage and embedding manager initialized")
            else:
                self.logger.warning("Failed to connect to vector store")
        except Exception as e:
            self.logger.error(f"Error initializing vector storage: {str(e)}")
        
        # Progress tracking for bulk operations
        self.bulk_operations = {}
    
    # ============================================================================
    # ADVANCED GRAPH QUERIES
    # ============================================================================
    
    def multi_hop_traversal(self, start_node_id: str, max_hops: int = 3, 
                           relation_filter: Optional[List[str]] = None,
                           min_confidence: float = 0.0) -> Dict[str, Any]:
        """
        Perform multi-hop traversal from a starting node.
        
        Args:
            start_node_id: Starting node ID
            max_hops: Maximum number of hops to traverse
            relation_filter: Optional list of relation types to include
            min_confidence: Minimum confidence score for relationships
            
        Returns:
            Dictionary with traversal results and paths
        """
        try:
            # Validate starting node exists
            try:
                start_node = self.engine.get_node(start_node_id)
            except ValueError:
                return {
                    "status": "error",
                    "message": f"Starting node {start_node_id} not found"
                }
            
            # Perform breadth-first traversal
            visited = set()
            current_level = [(start_node_id, [])]  # (node_id, path)
            all_paths = []
            nodes_by_distance = defaultdict(list)
            
            for hop in range(max_hops + 1):
                if not current_level:
                    break
                
                next_level = []
                
                for node_id, path in current_level:
                    if node_id in visited:
                        continue
                    
                    visited.add(node_id)
                    nodes_by_distance[hop].append(node_id)
                    
                    # Record the path to this node
                    if path:
                        all_paths.append(path + [node_id])
                    
                    # Get outgoing relationships if not at max hops
                    if hop < max_hops:
                        try:
                            relationships = self.engine.get_outgoing_relationships(node_id)
                            
                            for rel in relationships:
                                # Apply filters
                                if relation_filter and rel.relation_type not in relation_filter:
                                    continue
                                if rel.confidence_score < min_confidence:
                                    continue
                                
                                # Add to next level
                                new_path = path + [node_id, f"--{rel.relation_type}-->"]
                                next_level.append((rel.to_id, new_path))
                        except Exception as e:
                            self.logger.warning(f"Error getting relationships for {node_id}: {e}")
                
                current_level = next_level
            
            # Get node details for visited nodes
            node_details = {}
            for node_id in visited:
                try:
                    node = self.engine.get_node(node_id)
                    node_details[node_id] = {
                        "content": node.content[:100] + "..." if len(node.content) > 100 else node.content,
                        "source": node.source,
                        "rating_truthfulness": node.rating_truthfulness
                    }
                except Exception as e:
                    self.logger.warning(f"Error getting details for node {node_id}: {e}")
            
            return {
                "status": "success",
                "start_node": start_node_id,
                "max_hops": max_hops,
                "total_nodes_found": len(visited),
                "nodes_by_distance": dict(nodes_by_distance),
                "paths": all_paths[:50],  # Limit paths for response size
                "node_details": node_details
            }
            
        except Exception as e:
            self.logger.error(f"Error in multi-hop traversal: {str(e)}")
            return {
                "status": "error",
                "message": f"Multi-hop traversal failed: {str(e)}"
            }
    
    def extract_subgraph(self, topic_keywords: List[str], max_nodes: int = 50,
                        min_relevance: float = 0.7) -> Dict[str, Any]:
        """
        Extract a subgraph around specific topics.
        
        Args:
            topic_keywords: Keywords defining the topic of interest
            max_nodes: Maximum number of nodes to include
            min_relevance: Minimum relevance score for inclusion
            
        Returns:
            Dictionary with subgraph nodes and relationships
        """
        try:
            if not self.embedding_manager:
                return {
                    "status": "error", 
                    "message": "Embedding manager not available for subgraph extraction"
                }
            
            # Search for nodes related to topic keywords
            topic_query = " ".join(topic_keywords)
            similar_node_ids = self.embedding_manager.search_similar_nodes(
                topic_query, top_k=min(max_nodes * 2, 100)
            )
            
            if not similar_node_ids:
                return {
                    "status": "no_results",
                    "message": "No nodes found for the specified topic",
                    "topic_keywords": topic_keywords
                }
            
            # Get detailed similarity scores and filter by relevance
            relevant_nodes = []
            for node_id in similar_node_ids:
                try:
                    node = self.engine.get_node(node_id)
                    # Simple relevance calculation based on keyword presence
                    content_lower = node.content.lower()
                    keyword_matches = sum(1 for kw in topic_keywords if kw.lower() in content_lower)
                    relevance = keyword_matches / len(topic_keywords)
                    
                    if relevance >= min_relevance:
                        relevant_nodes.append({
                            "node_id": node_id,
                            "relevance": relevance,
                            "content": node.content,
                            "source": node.source
                        })
                except Exception as e:
                    self.logger.warning(f"Error processing node {node_id}: {e}")
            
            # Sort by relevance and limit
            relevant_nodes.sort(key=lambda x: x["relevance"], reverse=True)
            relevant_nodes = relevant_nodes[:max_nodes]
            
            # Extract relationships between relevant nodes
            relevant_node_ids = {node["node_id"] for node in relevant_nodes}
            subgraph_relationships = []
            
            for node_data in relevant_nodes:
                node_id = node_data["node_id"]
                try:
                    # Get outgoing relationships
                    outgoing = self.engine.get_outgoing_relationships(node_id)
                    for rel in outgoing:
                        if rel.to_id in relevant_node_ids:
                            subgraph_relationships.append({
                                "from_id": rel.from_id,
                                "to_id": rel.to_id,
                                "relation_type": rel.relation_type,
                                "confidence_score": rel.confidence_score
                            })
                except Exception as e:
                    self.logger.warning(f"Error getting relationships for {node_id}: {e}")
            
            return {
                "status": "success",
                "topic_keywords": topic_keywords,
                "total_nodes": len(relevant_nodes),
                "total_relationships": len(subgraph_relationships),
                "nodes": relevant_nodes,
                "relationships": subgraph_relationships,
                "subgraph_density": len(subgraph_relationships) / max(1, len(relevant_nodes))
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting subgraph: {str(e)}")
            return {
                "status": "error",
                "message": f"Subgraph extraction failed: {str(e)}"
            }
    
    def pattern_matching(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """
        Find nodes and relationships matching specific patterns.
        
        Args:
            pattern: Pattern specification including node and relationship constraints
            
        Returns:
            Dictionary with matching subgraphs
        """
        try:
            # Parse pattern specification
            node_constraints = pattern.get("nodes", {})
            relationship_constraints = pattern.get("relationships", {})
            max_results = pattern.get("max_results", 20)
            
            matches = []
            
            # Simple pattern matching implementation
            # This is a basic version - could be extended with more sophisticated pattern matching
            
            # Find nodes matching node constraints
            candidate_nodes = []
            
            if "content_contains" in node_constraints:
                content_filter = node_constraints["content_contains"]
                if self.embedding_manager:
                    # Use semantic search for content matching
                    similar_node_ids = self.embedding_manager.search_similar_nodes(
                        content_filter, top_k=100
                    )
                    
                    for node_id in similar_node_ids:
                        try:
                            node = self.engine.get_node(node_id)
                            # Check additional constraints
                            if self._node_matches_constraints(node, node_constraints):
                                candidate_nodes.append(node)
                        except Exception as e:
                            self.logger.warning(f"Error checking node {node_id}: {e}")
            
            # Apply relationship constraints to find matching patterns
            for node in candidate_nodes[:max_results]:
                try:
                    # Check if this node participates in the required relationship pattern
                    if self._check_relationship_pattern(node.node_id, relationship_constraints):
                        matches.append({
                            "root_node": {
                                "node_id": node.node_id,
                                "content": node.content[:100] + "..." if len(node.content) > 100 else node.content,
                                "source": node.source
                            },
                            "pattern_score": self._calculate_pattern_score(node, pattern)
                        })
                except Exception as e:
                    self.logger.warning(f"Error processing pattern match for {node.node_id}: {e}")
            
            # Sort by pattern score
            matches.sort(key=lambda x: x["pattern_score"], reverse=True)
            
            return {
                "status": "success",
                "pattern": pattern,
                "total_matches": len(matches),
                "matches": matches[:max_results]
            }
            
        except Exception as e:
            self.logger.error(f"Error in pattern matching: {str(e)}")
            return {
                "status": "error",
                "message": f"Pattern matching failed: {str(e)}"
            }
    
    def temporal_query(self, start_time: Optional[float] = None, 
                      end_time: Optional[float] = None,
                      operation_type: str = "nodes_created") -> Dict[str, Any]:
        """
        Query knowledge at specific time periods.
        
        Args:
            start_time: Start timestamp (Unix time)
            end_time: End timestamp (Unix time)
            operation_type: Type of temporal query ('nodes_created', 'changes', 'snapshots')
            
        Returns:
            Dictionary with temporal query results
        """
        try:
            if not self.engine.revision_manager:
                return {
                    "status": "error",
                    "message": "Versioning not enabled - temporal queries unavailable"
                }
            
            # Set default time range if not provided
            if end_time is None:
                end_time = time.time()
            if start_time is None:
                start_time = end_time - (7 * 24 * 3600)  # Default to last 7 days
            
            results = []
            
            if operation_type == "nodes_created":
                # Find nodes created in the time range
                # This is a simplified implementation - would need actual temporal indexing
                try:
                    # Query revision log for node creations
                    all_revisions = self.engine.storage.query_vertices({
                        'label': 'RevisionLog',
                        'change_type': 'create',
                        'object_type': 'node'
                    })
                    
                    for revision in all_revisions:
                        rev_time = revision.get('timestamp', 0)
                        if start_time <= rev_time <= end_time:
                            results.append({
                                "node_id": revision.get('object_id'),
                                "timestamp": rev_time,
                                "change_type": revision.get('change_type'),
                                "formatted_time": datetime.fromtimestamp(rev_time).isoformat()
                            })
                            
                except Exception as e:
                    self.logger.warning(f"Error querying revision log: {e}")
                    # Fallback: filter nodes by creation timestamp
                    results = self._filter_nodes_by_timestamp(start_time, end_time)
            
            elif operation_type == "changes":
                # Find all changes in the time range
                try:
                    all_revisions = self.engine.storage.query_vertices({
                        'label': 'RevisionLog'
                    })
                    
                    for revision in all_revisions:
                        rev_time = revision.get('timestamp', 0)
                        if start_time <= rev_time <= end_time:
                            results.append({
                                "object_id": revision.get('object_id'),
                                "object_type": revision.get('object_type'),
                                "change_type": revision.get('change_type'),
                                "timestamp": rev_time,
                                "formatted_time": datetime.fromtimestamp(rev_time).isoformat()
                            })
                except Exception as e:
                    self.logger.error(f"Error querying changes: {e}")
            
            elif operation_type == "snapshots":
                # Find snapshots in the time range
                try:
                    snapshots = self.engine.revision_manager.get_all_snapshots()
                    for snapshot in snapshots:
                        snap_time = snapshot.get('timestamp', 0)
                        if start_time <= snap_time <= end_time:
                            results.append({
                                "snapshot_id": snapshot.get('snapshot_id'),
                                "timestamp": snap_time,
                                "formatted_time": datetime.fromtimestamp(snap_time).isoformat()
                            })
                except Exception as e:
                    self.logger.error(f"Error querying snapshots: {e}")
            
            # Sort results by timestamp
            results.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            
            return {
                "status": "success",
                "operation_type": operation_type,
                "start_time": start_time,
                "end_time": end_time,
                "time_range_days": (end_time - start_time) / (24 * 3600),
                "total_results": len(results),
                "results": results[:100]  # Limit results
            }
            
        except Exception as e:
            self.logger.error(f"Error in temporal query: {str(e)}")
            return {
                "status": "error",
                "message": f"Temporal query failed: {str(e)}"
            }
    
    # ============================================================================
    # KNOWLEDGE SYNTHESIS ENDPOINTS
    # ============================================================================
    
    def synthesize_knowledge(self, node_ids: List[str], 
                           synthesis_type: str = "summary") -> Dict[str, Any]:
        """
        Combine multiple nodes into coherent responses.
        
        Args:
            node_ids: List of node IDs to synthesize
            synthesis_type: Type of synthesis ('summary', 'comparison', 'timeline')
            
        Returns:
            Dictionary with synthesized knowledge
        """
        try:
            if not node_ids:
                return {
                    "status": "error",
                    "message": "No node IDs provided for synthesis"
                }
            
            # Retrieve all nodes
            nodes = []
            for node_id in node_ids:
                try:
                    node = self.engine.get_node(node_id)
                    nodes.append(node)
                except ValueError:
                    self.logger.warning(f"Node {node_id} not found")
            
            if not nodes:
                return {
                    "status": "error",
                    "message": "No valid nodes found for synthesis"
                }
            
            if synthesis_type == "summary":
                return self._synthesize_summary(nodes)
            elif synthesis_type == "comparison":
                return self._synthesize_comparison(nodes)
            elif synthesis_type == "timeline":
                return self._synthesize_timeline(nodes)
            else:
                return {
                    "status": "error",
                    "message": f"Unknown synthesis type: {synthesis_type}"
                }
                
        except Exception as e:
            self.logger.error(f"Error in knowledge synthesis: {str(e)}")
            return {
                "status": "error",
                "message": f"Knowledge synthesis failed: {str(e)}"
            }
    
    def answer_question(self, question: str, max_hops: int = 2,
                       top_k_nodes: int = 10) -> Dict[str, Any]:
        """
        Answer questions using graph traversal and knowledge synthesis.
        
        Args:
            question: Question to answer
            max_hops: Maximum traversal depth
            top_k_nodes: Number of initial nodes to consider
            
        Returns:
            Dictionary with answer and supporting evidence
        """
        try:
            if not self.embedding_manager:
                return {
                    "status": "error",
                    "message": "Embedding manager not available for question answering"
                }
            
            # Find relevant nodes using semantic search
            relevant_node_ids = self.embedding_manager.search_similar_nodes(
                question, top_k=top_k_nodes
            )
            
            if not relevant_node_ids:
                return {
                    "status": "no_results",
                    "message": "No relevant knowledge found for the question",
                    "question": question
                }
            
            # Expand knowledge using multi-hop traversal
            expanded_knowledge = []
            all_relevant_nodes = set()
            
            for node_id in relevant_node_ids:
                # Get the node and its immediate context
                traversal_result = self.multi_hop_traversal(
                    node_id, max_hops=max_hops, min_confidence=0.5
                )
                
                if traversal_result["status"] == "success":
                    for distance, node_list in traversal_result["nodes_by_distance"].items():
                        all_relevant_nodes.update(node_list)
            
            # Collect evidence from relevant nodes
            evidence = []
            for node_id in list(all_relevant_nodes)[:20]:  # Limit for performance
                try:
                    node = self.engine.get_node(node_id)
                    evidence.append({
                        "node_id": node_id,
                        "content": node.content,
                        "source": node.source,
                        "confidence": node.rating_truthfulness
                    })
                except Exception as e:
                    self.logger.warning(f"Error getting evidence from node {node_id}: {e}")
            
            # Sort evidence by confidence
            evidence.sort(key=lambda x: x["confidence"], reverse=True)
            
            # Generate answer based on evidence
            answer = self._generate_answer_from_evidence(question, evidence[:10])
            
            return {
                "status": "success",
                "question": question,
                "answer": answer,
                "evidence_count": len(evidence),
                "evidence": evidence[:5],  # Return top 5 pieces of evidence
                "confidence_score": self._calculate_answer_confidence(evidence[:10])
            }
            
        except Exception as e:
            self.logger.error(f"Error answering question: {str(e)}")
            return {
                "status": "error",
                "message": f"Question answering failed: {str(e)}"
            }
    
    def find_contradictions(self, topic_keywords: Optional[List[str]] = None,
                           confidence_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Find contradictions in the knowledge base.
        
        Args:
            topic_keywords: Optional topic filter
            confidence_threshold: Minimum confidence for contradiction detection
            
        Returns:
            Dictionary with potential contradictions found
        """
        try:
            contradictions = []
            
            # Get nodes to analyze
            if topic_keywords and self.embedding_manager:
                # Focus on specific topic
                topic_query = " ".join(topic_keywords)
                candidate_node_ids = self.embedding_manager.search_similar_nodes(
                    topic_query, top_k=50
                )
            else:
                # Analyze all high-confidence nodes (limited for performance)
                candidate_node_ids = self._get_high_confidence_nodes(limit=100)
            
            # Compare nodes pairwise for contradictions
            nodes_data = {}
            for node_id in candidate_node_ids:
                try:
                    node = self.engine.get_node(node_id)
                    if node.rating_truthfulness >= confidence_threshold:
                        nodes_data[node_id] = node
                except Exception as e:
                    self.logger.warning(f"Error getting node {node_id}: {e}")
            
            # Simple contradiction detection based on opposite statements
            contradiction_keywords = {
                "positive": ["is", "are", "has", "have", "can", "does", "will", "true", "correct"],
                "negative": ["is not", "are not", "has no", "have no", "cannot", "does not", "will not", "false", "incorrect"]
            }
            
            node_list = list(nodes_data.items())
            for i, (node_id1, node1) in enumerate(node_list):
                for j, (node_id2, node2) in enumerate(node_list[i+1:], i+1):
                    # Check for potential contradiction
                    contradiction_score = self._detect_contradiction(node1.content, node2.content)
                    
                    if contradiction_score > 0.7:  # Threshold for contradiction
                        contradictions.append({
                            "node1": {
                                "id": node_id1,
                                "content": node1.content[:200],
                                "source": node1.source,
                                "confidence": node1.rating_truthfulness
                            },
                            "node2": {
                                "id": node_id2,
                                "content": node2.content[:200],
                                "source": node2.source,
                                "confidence": node2.rating_truthfulness
                            },
                            "contradiction_score": contradiction_score,
                            "detected_patterns": self._get_contradiction_patterns(node1.content, node2.content)
                        })
            
            # Sort by contradiction score
            contradictions.sort(key=lambda x: x["contradiction_score"], reverse=True)
            
            return {
                "status": "success",
                "topic_keywords": topic_keywords,
                "nodes_analyzed": len(nodes_data),
                "contradictions_found": len(contradictions),
                "contradictions": contradictions[:10]  # Limit results
            }
            
        except Exception as e:
            self.logger.error(f"Error finding contradictions: {str(e)}")
            return {
                "status": "error",
                "message": f"Contradiction detection failed: {str(e)}"
            }
    
    # ============================================================================
    # HELPER METHODS
    # ============================================================================
    
    def _node_matches_constraints(self, node: KnowledgeNode, constraints: Dict[str, Any]) -> bool:
        """Check if a node matches the given constraints."""
        try:
            if "min_truthfulness" in constraints:
                if node.rating_truthfulness < constraints["min_truthfulness"]:
                    return False
            
            if "source_contains" in constraints:
                if constraints["source_contains"].lower() not in node.source.lower():
                    return False
            
            if "content_length_min" in constraints:
                if len(node.content) < constraints["content_length_min"]:
                    return False
            
            return True
        except Exception:
            return False
    
    def _check_relationship_pattern(self, node_id: str, constraints: Dict[str, Any]) -> bool:
        """Check if a node participates in the required relationship pattern."""
        try:
            if "outgoing_relation_type" in constraints:
                outgoing = self.engine.get_outgoing_relationships(node_id)
                required_type = constraints["outgoing_relation_type"]
                if not any(rel.relation_type == required_type for rel in outgoing):
                    return False
            
            if "incoming_relation_type" in constraints:
                incoming = self.engine.get_incoming_relationships(node_id)
                required_type = constraints["incoming_relation_type"]
                if not any(rel.relation_type == required_type for rel in incoming):
                    return False
            
            return True
        except Exception:
            return False
    
    def _calculate_pattern_score(self, node: KnowledgeNode, pattern: Dict[str, Any]) -> float:
        """Calculate how well a node matches the pattern."""
        score = 0.0
        
        # Base score from node quality
        score += node.rating_truthfulness * 0.3
        score += node.rating_richness * 0.2
        
        # Additional scoring based on pattern specificity
        if "content_contains" in pattern.get("nodes", {}):
            content_lower = node.content.lower()
            search_term = pattern["nodes"]["content_contains"].lower()
            if search_term in content_lower:
                score += 0.5
        
        return min(1.0, score)
    
    def _filter_nodes_by_timestamp(self, start_time: float, end_time: float) -> List[Dict[str, Any]]:
        """Filter nodes by creation timestamp (fallback method)."""
        results = []
        # This is a simplified implementation
        # In a real system, you'd have proper temporal indexing
        return results
    
    def _synthesize_summary(self, nodes: List[KnowledgeNode]) -> Dict[str, Any]:
        """Generate a summary from multiple nodes."""
        try:
            # Extract key information
            sources = set(node.source for node in nodes)
            total_content_length = sum(len(node.content) for node in nodes)
            avg_confidence = sum(node.rating_truthfulness for node in nodes) / len(nodes)
            
            # Group nodes by topic/source for better summarization
            content_by_source = defaultdict(list)
            for node in nodes:
                content_by_source[node.source].append(node.content)
            
            # Generate summary points
            summary_points = []
            for source, contents in content_by_source.items():
                # Simple summarization: take first sentence of each content
                for content in contents:
                    first_sentence = content.split('.')[0] + '.'
                    if len(first_sentence) > 20:  # Meaningful sentence
                        summary_points.append(first_sentence)
            
            return {
                "status": "success",
                "synthesis_type": "summary",
                "nodes_processed": len(nodes),
                "sources": list(sources),
                "average_confidence": avg_confidence,
                "summary_points": summary_points[:10],
                "total_content_length": total_content_length
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Summary synthesis failed: {str(e)}"
            }
    
    def _synthesize_comparison(self, nodes: List[KnowledgeNode]) -> Dict[str, Any]:
        """Generate a comparison between multiple nodes."""
        try:
            if len(nodes) < 2:
                return {
                    "status": "error",
                    "message": "At least 2 nodes required for comparison"
                }
            
            # Compare key attributes
            comparisons = []
            
            for i, node1 in enumerate(nodes):
                for j, node2 in enumerate(nodes[i+1:], i+1):
                    comparison = {
                        "node1": {"id": node1.node_id, "source": node1.source},
                        "node2": {"id": node2.node_id, "source": node2.source},
                        "confidence_diff": abs(node1.rating_truthfulness - node2.rating_truthfulness),
                        "source_match": node1.source == node2.source,
                        "content_similarity": self._calculate_content_similarity(node1.content, node2.content)
                    }
                    comparisons.append(comparison)
            
            return {
                "status": "success",
                "synthesis_type": "comparison",
                "nodes_compared": len(nodes),
                "pairwise_comparisons": len(comparisons),
                "comparisons": comparisons
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Comparison synthesis failed: {str(e)}"
            }
    
    def _synthesize_timeline(self, nodes: List[KnowledgeNode]) -> Dict[str, Any]:
        """Generate a timeline from multiple nodes."""
        try:
            # Sort nodes by creation timestamp
            sorted_nodes = sorted(nodes, key=lambda x: x.creation_timestamp)
            
            timeline_events = []
            for node in sorted_nodes:
                timeline_events.append({
                    "timestamp": node.creation_timestamp,
                    "date": datetime.fromtimestamp(node.creation_timestamp).isoformat(),
                    "content": node.content[:100] + "..." if len(node.content) > 100 else node.content,
                    "source": node.source,
                    "node_id": node.node_id
                })
            
            # Calculate time span
            if len(sorted_nodes) > 1:
                time_span = sorted_nodes[-1].creation_timestamp - sorted_nodes[0].creation_timestamp
                time_span_days = time_span / (24 * 3600)
            else:
                time_span_days = 0
            
            return {
                "status": "success",
                "synthesis_type": "timeline",
                "nodes_in_timeline": len(nodes),
                "time_span_days": time_span_days,
                "timeline_events": timeline_events
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Timeline synthesis failed: {str(e)}"
            }
    
    def _generate_answer_from_evidence(self, question: str, evidence: List[Dict[str, Any]]) -> str:
        """Generate an answer based on evidence (simplified implementation)."""
        if not evidence:
            return "Insufficient evidence to answer the question."
        
        # Simple answer generation based on highest confidence evidence
        best_evidence = evidence[0]
        
        # Extract relevant sentences from the evidence
        content = best_evidence["content"]
        sentences = content.split('.')
        
        # Return the first meaningful sentence as a basic answer
        for sentence in sentences:
            if len(sentence.strip()) > 20:
                return sentence.strip() + "."
        
        return "Based on available evidence: " + content[:200] + "..."
    
    def _calculate_answer_confidence(self, evidence: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for an answer based on evidence."""
        if not evidence:
            return 0.0
        
        # Average confidence weighted by evidence quality
        total_confidence = sum(e["confidence"] for e in evidence)
        return total_confidence / len(evidence)
    
    def _get_high_confidence_nodes(self, limit: int = 100) -> List[str]:
        """Get high confidence nodes (simplified implementation)."""
        # This would need proper indexing in a real implementation
        return []
    
    def _detect_contradiction(self, content1: str, content2: str) -> float:
        """Detect contradiction between two pieces of content."""
        # Simplified contradiction detection
        # In a real implementation, this would use NLP techniques
        
        # Look for direct negations
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        # Simple pattern: one says "is" and another says "is not"
        contradiction_score = 0.0
        
        if "is not" in content1.lower() and " is " in content2.lower():
            contradiction_score += 0.5
        elif " is " in content1.lower() and "is not" in content2.lower():
            contradiction_score += 0.5
        
        # Check for opposite terms
        opposites = [
            ("true", "false"), ("correct", "incorrect"), ("right", "wrong"),
            ("good", "bad"), ("positive", "negative"), ("yes", "no")
        ]
        
        for term1, term2 in opposites:
            if term1 in words1 and term2 in words2:
                contradiction_score += 0.3
            elif term2 in words1 and term1 in words2:
                contradiction_score += 0.3
        
        return min(1.0, contradiction_score)
    
    def _get_contradiction_patterns(self, content1: str, content2: str) -> List[str]:
        """Get specific contradiction patterns detected."""
        patterns = []
        
        if "is not" in content1.lower() and " is " in content2.lower():
            patterns.append("Negation vs Affirmation")
        
        # Add more pattern detection logic here
        
        return patterns
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two pieces of content."""
        # Simple word overlap similarity
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    # ============================================================================
    # BULK OPERATIONS
    # ============================================================================
    
    def start_bulk_ingestion(self, operation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Start a bulk ingestion operation with progress tracking.
        
        Args:
            operation_id: Optional custom operation ID
            
        Returns:
            Dictionary with operation details and tracking ID
        """
        try:
            if operation_id is None:
                operation_id = str(uuid.uuid4())
            
            # Initialize operation tracking
            self.bulk_operations[operation_id] = {
                "operation_type": "bulk_ingestion",
                "status": "initialized",
                "created_at": time.time(),
                "total_items": 0,
                "processed_items": 0,
                "failed_items": 0,
                "node_ids": [],
                "errors": []
            }
            
            return {
                "status": "success",
                "operation_id": operation_id,
                "message": "Bulk ingestion operation initialized"
            }
            
        except Exception as e:
            self.logger.error(f"Error starting bulk ingestion: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to start bulk ingestion: {str(e)}"
            }
    
    def add_to_bulk_ingestion(self, operation_id: str, texts: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Add texts to an ongoing bulk ingestion operation.
        
        Args:
            operation_id: Operation ID from start_bulk_ingestion
            texts: List of dictionaries with 'text' and 'source' fields
            
        Returns:
            Dictionary with processing results
        """
        try:
            if operation_id not in self.bulk_operations:
                return {
                    "status": "error",
                    "message": f"Bulk operation {operation_id} not found"
                }
            
            operation = self.bulk_operations[operation_id]
            operation["status"] = "processing"
            operation["total_items"] += len(texts)
            
            # Process texts
            for text_data in texts:
                try:
                    text = text_data.get("text", "")
                    source = text_data.get("source", "Bulk Input")
                    
                    if not text.strip():
                        operation["failed_items"] += 1
                        operation["errors"].append("Empty text provided")
                        continue
                    
                    # Extract and process knowledge
                    units = extract_knowledge_units(text)
                    node_ids = process_extracted_units(
                        units=units,
                        source_label=source,
                        storage=self.engine.storage,
                        embedding_manager=self.embedding_manager
                    )
                    
                    operation["node_ids"].extend(node_ids)
                    operation["processed_items"] += 1
                    
                except Exception as e:
                    operation["failed_items"] += 1
                    operation["errors"].append(f"Processing error: {str(e)}")
                    self.logger.warning(f"Error processing text in bulk operation: {e}")
            
            # Update progress
            progress = operation["processed_items"] / max(1, operation["total_items"])
            operation["progress"] = progress
            
            if operation["processed_items"] + operation["failed_items"] >= operation["total_items"]:
                operation["status"] = "completed"
                operation["completed_at"] = time.time()
            
            return {
                "status": "success",
                "operation_id": operation_id,
                "progress": progress,
                "processed": operation["processed_items"],
                "failed": operation["failed_items"],
                "total": operation["total_items"],
                "nodes_created": len(operation["node_ids"])
            }
            
        except Exception as e:
            self.logger.error(f"Error in bulk ingestion: {str(e)}")
            return {
                "status": "error",
                "message": f"Bulk ingestion failed: {str(e)}"
            }
    
    def get_bulk_operation_status(self, operation_id: str) -> Dict[str, Any]:
        """
        Get the status of a bulk operation.
        
        Args:
            operation_id: Operation ID to check
            
        Returns:
            Dictionary with operation status and progress
        """
        try:
            if operation_id not in self.bulk_operations:
                return {
                    "status": "error",
                    "message": f"Bulk operation {operation_id} not found"
                }
            
            operation = self.bulk_operations[operation_id]
            
            # Calculate timing information
            elapsed_time = time.time() - operation["created_at"]
            
            result = {
                "status": "success",
                "operation_id": operation_id,
                "operation_status": operation["status"],
                "progress": operation.get("progress", 0.0),
                "processed_items": operation["processed_items"],
                "failed_items": operation["failed_items"],
                "total_items": operation["total_items"],
                "nodes_created": len(operation["node_ids"]),
                "elapsed_time_seconds": elapsed_time,
                "errors": operation["errors"][-10:]  # Last 10 errors
            }
            
            if operation["status"] == "completed":
                result["completed_at"] = operation.get("completed_at")
                result["total_duration"] = operation.get("completed_at", time.time()) - operation["created_at"]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting bulk operation status: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to get operation status: {str(e)}"
            }
    
    def export_subgraph(self, node_ids: List[str], format_type: str = "json",
                       include_relationships: bool = True) -> Dict[str, Any]:
        """
        Export a subgraph in various formats.
        
        Args:
            node_ids: List of node IDs to include in export
            format_type: Export format ('json', 'cypher', 'graphml', 'csv')
            include_relationships: Whether to include relationships in export
            
        Returns:
            Dictionary with exported data
        """
        try:
            if not node_ids:
                return {
                    "status": "error",
                    "message": "No node IDs provided for export"
                }
            
            # Collect node data
            nodes_data = []
            for node_id in node_ids:
                try:
                    node = self.engine.get_node(node_id)
                    node_data = {
                        "id": node.node_id,
                        "content": node.content,
                        "source": node.source,
                        "creation_timestamp": node.creation_timestamp,
                        "rating_richness": node.rating_richness,
                        "rating_truthfulness": node.rating_truthfulness,
                        "rating_stability": node.rating_stability
                    }
                    nodes_data.append(node_data)
                except Exception as e:
                    self.logger.warning(f"Error exporting node {node_id}: {e}")
            
            # Collect relationship data
            relationships_data = []
            if include_relationships:
                node_id_set = set(node_ids)
                for node_id in node_ids:
                    try:
                        outgoing = self.engine.get_outgoing_relationships(node_id)
                        for rel in outgoing:
                            if rel.to_id in node_id_set:  # Only include internal relationships
                                rel_data = {
                                    "from_id": rel.from_id,
                                    "to_id": rel.to_id,
                                    "relation_type": rel.relation_type,
                                    "confidence_score": rel.confidence_score,
                                    "timestamp": rel.timestamp
                                }
                                relationships_data.append(rel_data)
                    except Exception as e:
                        self.logger.warning(f"Error exporting relationships for {node_id}: {e}")
            
            # Format the export
            if format_type == "json":
                exported_data = {
                    "nodes": nodes_data,
                    "relationships": relationships_data,
                    "export_metadata": {
                        "timestamp": time.time(),
                        "node_count": len(nodes_data),
                        "relationship_count": len(relationships_data)
                    }
                }
                return {
                    "status": "success",
                    "format": format_type,
                    "data": exported_data
                }
            
            elif format_type == "cypher":
                cypher_statements = []
                
                # Create nodes
                for node in nodes_data:
                    cypher = f"CREATE (n{node['id'].replace('-', '_')}:KnowledgeNode {{"
                    properties = []
                    for key, value in node.items():
                        if isinstance(value, str):
                            escaped_value = value.replace("'", "\\'")
                            properties.append(f"{key}: '{escaped_value}'")
                        else:
                            properties.append(f"{key}: {value}")
                    cypher += ", ".join(properties) + "})"
                    cypher_statements.append(cypher)
                
                # Create relationships
                for rel in relationships_data:
                    cypher = f"MATCH (a:KnowledgeNode {{id: '{rel['from_id']}'}}), (b:KnowledgeNode {{id: '{rel['to_id']}'}})"
                    cypher += f" CREATE (a)-[:{rel['relation_type']} {{confidence: {rel['confidence_score']}}}]->(b)"
                    cypher_statements.append(cypher)
                
                return {
                    "status": "success",
                    "format": format_type,
                    "data": cypher_statements
                }
            
            elif format_type == "csv":
                # Simple CSV format
                csv_data = {
                    "nodes_csv": self._convert_to_csv(nodes_data),
                    "relationships_csv": self._convert_to_csv(relationships_data) if relationships_data else ""
                }
                return {
                    "status": "success",
                    "format": format_type,
                    "data": csv_data
                }
            
            else:
                return {
                    "status": "error",
                    "message": f"Unsupported export format: {format_type}"
                }
                
        except Exception as e:
            self.logger.error(f"Error exporting subgraph: {str(e)}")
            return {
                "status": "error",
                "message": f"Subgraph export failed: {str(e)}"
            }
    
    def bulk_create_relationships(self, relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create multiple relationships in bulk.
        
        Args:
            relationships: List of relationship specifications
            
        Returns:
            Dictionary with creation results
        """
        try:
            created_relationships = []
            failed_relationships = []
            
            for rel_spec in relationships:
                try:
                    # Validate required fields
                    required_fields = ["from_id", "to_id", "relation_type"]
                    if not all(field in rel_spec for field in required_fields):
                        failed_relationships.append({
                            "spec": rel_spec,
                            "error": "Missing required fields"
                        })
                        continue
                    
                    # Create relationship object
                    relationship = Relationship(
                        from_id=rel_spec["from_id"],
                        to_id=rel_spec["to_id"],
                        relation_type=rel_spec["relation_type"],
                        confidence_score=rel_spec.get("confidence_score", 0.7),
                        timestamp=rel_spec.get("timestamp", time.time())
                    )
                    
                    # Save relationship
                    edge_id = self.engine.save_relationship(relationship)
                    created_relationships.append({
                        "edge_id": edge_id,
                        "from_id": rel_spec["from_id"],
                        "to_id": rel_spec["to_id"],
                        "relation_type": rel_spec["relation_type"]
                    })
                    
                except Exception as e:
                    failed_relationships.append({
                        "spec": rel_spec,
                        "error": str(e)
                    })
                    self.logger.warning(f"Error creating relationship: {e}")
            
            return {
                "status": "success",
                "created_count": len(created_relationships),
                "failed_count": len(failed_relationships),
                "created_relationships": created_relationships,
                "failed_relationships": failed_relationships[:10]  # Limit for response size
            }
            
        except Exception as e:
            self.logger.error(f"Error in bulk relationship creation: {str(e)}")
            return {
                "status": "error",
                "message": f"Bulk relationship creation failed: {str(e)}"
            }
    
    # ============================================================================
    # ANALYTICS ENDPOINTS
    # ============================================================================
    
    def analyze_knowledge_coverage(self, domains: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze knowledge coverage across domains and sources.
        
        Args:
            domains: Optional list of domains to analyze
            
        Returns:
            Dictionary with coverage analysis
        """
        try:
            # This is a simplified implementation
            # In a real system, you'd have proper domain indexing
            
            analysis = {
                "total_nodes": 0,
                "domains": defaultdict(int),
                "sources": defaultdict(int),
                "quality_distribution": {"high": 0, "medium": 0, "low": 0},
                "content_length_stats": {"min": float('inf'), "max": 0, "avg": 0},
                "temporal_distribution": defaultdict(int)
            }
            
            # Sample analysis - in practice you'd query the actual graph
            sample_node_ids = self._get_sample_nodes(limit=1000)
            
            content_lengths = []
            
            for node_id in sample_node_ids:
                try:
                    node = self.engine.get_node(node_id)
                    analysis["total_nodes"] += 1
                    
                    # Source analysis
                    analysis["sources"][node.source] += 1
                    
                    # Quality distribution
                    avg_rating = (node.rating_truthfulness + node.rating_richness + node.rating_stability) / 3
                    if avg_rating >= 0.8:
                        analysis["quality_distribution"]["high"] += 1
                    elif avg_rating >= 0.6:
                        analysis["quality_distribution"]["medium"] += 1
                    else:
                        analysis["quality_distribution"]["low"] += 1
                    
                    # Content length statistics
                    content_length = len(node.content)
                    content_lengths.append(content_length)
                    analysis["content_length_stats"]["min"] = min(analysis["content_length_stats"]["min"], content_length)
                    analysis["content_length_stats"]["max"] = max(analysis["content_length_stats"]["max"], content_length)
                    
                    # Temporal distribution (by month)
                    month_key = datetime.fromtimestamp(node.creation_timestamp).strftime("%Y-%m")
                    analysis["temporal_distribution"][month_key] += 1
                    
                except Exception as e:
                    self.logger.warning(f"Error analyzing node {node_id}: {e}")
            
            # Calculate average content length
            if content_lengths:
                analysis["content_length_stats"]["avg"] = sum(content_lengths) / len(content_lengths)
            
            # Convert defaultdicts to regular dicts for JSON serialization
            analysis["domains"] = dict(analysis["domains"])
            analysis["sources"] = dict(analysis["sources"])
            analysis["temporal_distribution"] = dict(analysis["temporal_distribution"])
            
            return {
                "status": "success",
                "analysis": analysis,
                "analyzed_nodes": len(sample_node_ids)
            }
            
        except Exception as e:
            self.logger.error(f"Error in coverage analysis: {str(e)}")
            return {
                "status": "error",
                "message": f"Coverage analysis failed: {str(e)}"
            }
    
    def calculate_relationship_metrics(self) -> Dict[str, Any]:
        """
        Calculate relationship density and network metrics.
        
        Returns:
            Dictionary with relationship metrics
        """
        try:
            metrics = {
                "total_relationships": 0,
                "relationship_types": defaultdict(int),
                "density": 0.0,
                "avg_confidence": 0.0,
                "confidence_distribution": {"high": 0, "medium": 0, "low": 0}
            }
            
            # Sample relationships for analysis
            sample_node_ids = self._get_sample_nodes(limit=500)
            total_nodes = len(sample_node_ids)
            
            confidence_scores = []
            
            for node_id in sample_node_ids:
                try:
                    relationships = self.engine.get_outgoing_relationships(node_id)
                    
                    for rel in relationships:
                        metrics["total_relationships"] += 1
                        metrics["relationship_types"][rel.relation_type] += 1
                        
                        confidence_scores.append(rel.confidence_score)
                        
                        # Confidence distribution
                        if rel.confidence_score >= 0.8:
                            metrics["confidence_distribution"]["high"] += 1
                        elif rel.confidence_score >= 0.6:
                            metrics["confidence_distribution"]["medium"] += 1
                        else:
                            metrics["confidence_distribution"]["low"] += 1
                            
                except Exception as e:
                    self.logger.warning(f"Error analyzing relationships for {node_id}: {e}")
            
            # Calculate metrics
            if total_nodes > 1:
                max_possible_edges = total_nodes * (total_nodes - 1)
                metrics["density"] = metrics["total_relationships"] / max_possible_edges
            
            if confidence_scores:
                metrics["avg_confidence"] = sum(confidence_scores) / len(confidence_scores)
            
            # Convert defaultdict to regular dict
            metrics["relationship_types"] = dict(metrics["relationship_types"])
            
            return {
                "status": "success",
                "metrics": metrics,
                "analyzed_nodes": total_nodes
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating relationship metrics: {str(e)}")
            return {
                "status": "error",
                "message": f"Relationship metrics calculation failed: {str(e)}"
            }
    
    def analyze_quality_scores(self) -> Dict[str, Any]:
        """
        Analyze quality score distributions across the knowledge base.
        
        Returns:
            Dictionary with quality analysis
        """
        try:
            quality_analysis = {
                "truthfulness": {"scores": [], "avg": 0.0, "distribution": {}},
                "richness": {"scores": [], "avg": 0.0, "distribution": {}},
                "stability": {"scores": [], "avg": 0.0, "distribution": {}}
            }
            
            # Sample nodes for analysis
            sample_node_ids = self._get_sample_nodes(limit=1000)
            
            for node_id in sample_node_ids:
                try:
                    node = self.engine.get_node(node_id)
                    
                    # Collect scores
                    quality_analysis["truthfulness"]["scores"].append(node.rating_truthfulness)
                    quality_analysis["richness"]["scores"].append(node.rating_richness)
                    quality_analysis["stability"]["scores"].append(node.rating_stability)
                    
                except Exception as e:
                    self.logger.warning(f"Error analyzing quality for node {node_id}: {e}")
            
            # Calculate statistics for each quality dimension
            for dimension in ["truthfulness", "richness", "stability"]:
                scores = quality_analysis[dimension]["scores"]
                
                if scores:
                    # Average
                    quality_analysis[dimension]["avg"] = sum(scores) / len(scores)
                    
                    # Distribution (bins: 0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0)
                    distribution = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
                    
                    for score in scores:
                        if score < 0.2:
                            distribution["0.0-0.2"] += 1
                        elif score < 0.4:
                            distribution["0.2-0.4"] += 1
                        elif score < 0.6:
                            distribution["0.4-0.6"] += 1
                        elif score < 0.8:
                            distribution["0.6-0.8"] += 1
                        else:
                            distribution["0.8-1.0"] += 1
                    
                    quality_analysis[dimension]["distribution"] = distribution
                    
                    # Remove raw scores from output for efficiency
                    del quality_analysis[dimension]["scores"]
            
            return {
                "status": "success",
                "quality_analysis": quality_analysis,
                "analyzed_nodes": len(sample_node_ids)
            }
            
        except Exception as e:
            self.logger.error(f"Error in quality analysis: {str(e)}")
            return {
                "status": "error",
                "message": f"Quality analysis failed: {str(e)}"
            }
    
    def analyze_knowledge_evolution(self, time_periods: int = 12) -> Dict[str, Any]:
        """
        Analyze how knowledge has evolved over time.
        
        Args:
            time_periods: Number of time periods to analyze
            
        Returns:
            Dictionary with evolution analysis
        """
        try:
            if not self.engine.revision_manager:
                return {
                    "status": "error",
                    "message": "Versioning not enabled - evolution analysis unavailable"
                }
            
            # Calculate time period boundaries
            end_time = time.time()
            period_duration = (30 * 24 * 3600)  # 30 days per period
            start_time = end_time - (time_periods * period_duration)
            
            evolution_data = []
            
            for period in range(time_periods):
                period_start = start_time + (period * period_duration)
                period_end = period_start + period_duration
                
                # Analyze this time period
                period_analysis = {
                    "period": period + 1,
                    "start_date": datetime.fromtimestamp(period_start).isoformat(),
                    "end_date": datetime.fromtimestamp(period_end).isoformat(),
                    "nodes_created": 0,
                    "nodes_updated": 0,
                    "relationships_created": 0,
                    "avg_quality": 0.0
                }
                
                # Query revision log for this period
                try:
                    # This is simplified - would need proper temporal queries
                    temporal_result = self.temporal_query(
                        start_time=period_start,
                        end_time=period_end,
                        operation_type="changes"
                    )
                    
                    if temporal_result["status"] == "success":
                        for change in temporal_result["results"]:
                            if change["change_type"] == "create":
                                if change["object_type"] == "node":
                                    period_analysis["nodes_created"] += 1
                                elif change["object_type"] == "edge":
                                    period_analysis["relationships_created"] += 1
                            elif change["change_type"] == "update" and change["object_type"] == "node":
                                period_analysis["nodes_updated"] += 1
                    
                except Exception as e:
                    self.logger.warning(f"Error analyzing period {period}: {e}")
                
                evolution_data.append(period_analysis)
            
            # Calculate trends
            if len(evolution_data) > 1:
                trends = {
                    "nodes_created_trend": self._calculate_trend([p["nodes_created"] for p in evolution_data]),
                    "nodes_updated_trend": self._calculate_trend([p["nodes_updated"] for p in evolution_data]),
                    "relationships_created_trend": self._calculate_trend([p["relationships_created"] for p in evolution_data])
                }
            else:
                trends = {}
            
            return {
                "status": "success",
                "time_periods": time_periods,
                "period_duration_days": period_duration / (24 * 3600),
                "evolution_data": evolution_data,
                "trends": trends
            }
            
        except Exception as e:
            self.logger.error(f"Error in evolution analysis: {str(e)}")
            return {
                "status": "error",
                "message": f"Evolution analysis failed: {str(e)}"
            }
    
    # ============================================================================
    # ENHANCED COMMAND ROUTING
    # ============================================================================
    
    def execute_mcp_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an enhanced MCP command.
        
        Args:
            command: Dictionary with command details
            
        Returns:
            Response dictionary based on the command
        """
        try:
            if not command or 'action' not in command:
                return {
                    "status": "error",
                    "message": "Invalid command format: 'action' field is required"
                }
            
            action = command['action']
            
            # Advanced Graph Queries
            if action == 'multi_hop_traversal':
                return self.multi_hop_traversal(
                    start_node_id=command.get('start_node_id'),
                    max_hops=command.get('max_hops', 3),
                    relation_filter=command.get('relation_filter'),
                    min_confidence=command.get('min_confidence', 0.0)
                )
            
            elif action == 'extract_subgraph':
                return self.extract_subgraph(
                    topic_keywords=command.get('topic_keywords', []),
                    max_nodes=command.get('max_nodes', 50),
                    min_relevance=command.get('min_relevance', 0.7)
                )
            
            elif action == 'pattern_matching':
                return self.pattern_matching(command.get('pattern', {}))
            
            elif action == 'temporal_query':
                return self.temporal_query(
                    start_time=command.get('start_time'),
                    end_time=command.get('end_time'),
                    operation_type=command.get('operation_type', 'nodes_created')
                )
            
            # Knowledge Synthesis
            elif action == 'synthesize_knowledge':
                return self.synthesize_knowledge(
                    node_ids=command.get('node_ids', []),
                    synthesis_type=command.get('synthesis_type', 'summary')
                )
            
            elif action == 'answer_question':
                return self.answer_question(
                    question=command.get('question', ''),
                    max_hops=command.get('max_hops', 2),
                    top_k_nodes=command.get('top_k_nodes', 10)
                )
            
            elif action == 'find_contradictions':
                return self.find_contradictions(
                    topic_keywords=command.get('topic_keywords'),
                    confidence_threshold=command.get('confidence_threshold', 0.8)
                )
            
            # Bulk Operations
            elif action == 'start_bulk_ingestion':
                return self.start_bulk_ingestion(command.get('operation_id'))
            
            elif action == 'add_to_bulk_ingestion':
                return self.add_to_bulk_ingestion(
                    operation_id=command.get('operation_id'),
                    texts=command.get('texts', [])
                )
            
            elif action == 'get_bulk_operation_status':
                return self.get_bulk_operation_status(command.get('operation_id'))
            
            elif action == 'export_subgraph':
                return self.export_subgraph(
                    node_ids=command.get('node_ids', []),
                    format_type=command.get('format', 'json'),
                    include_relationships=command.get('include_relationships', True)
                )
            
            elif action == 'bulk_create_relationships':
                return self.bulk_create_relationships(command.get('relationships', []))
            
            # Analytics
            elif action == 'analyze_knowledge_coverage':
                return self.analyze_knowledge_coverage(command.get('domains'))
            
            elif action == 'calculate_relationship_metrics':
                return self.calculate_relationship_metrics()
            
            elif action == 'analyze_quality_scores':
                return self.analyze_quality_scores()
            
            elif action == 'analyze_knowledge_evolution':
                return self.analyze_knowledge_evolution(command.get('time_periods', 12))
            
            else:
                return {
                    "status": "error",
                    "message": f"Unknown action: {action}"
                }
                
        except Exception as e:
            self.logger.error(f"Error executing MCP command: {str(e)}")
            return {
                "status": "error",
                "message": f"Command execution error: {str(e)}"
            }
    
    # ============================================================================
    # UTILITY METHODS
    # ============================================================================
    
    def _get_sample_nodes(self, limit: int = 100) -> List[str]:
        """Get a sample of node IDs for analysis."""
        # This is a simplified implementation
        # In practice, you'd have proper sampling strategies
        try:
            # Try to get some actual node IDs
            # This is a placeholder - would need proper graph traversal
            return []
        except Exception:
            return []
    
    def _convert_to_csv(self, data: List[Dict[str, Any]]) -> str:
        """Convert list of dictionaries to CSV format."""
        if not data:
            return ""
        
        import io
        import csv
        
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
        
        return output.getvalue()
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a series of values."""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple trend calculation
        first_half = sum(values[:len(values)//2]) / (len(values)//2)
        second_half = sum(values[len(values)//2:]) / (len(values) - len(values)//2)
        
        if second_half > first_half * 1.1:
            return "increasing"
        elif second_half < first_half * 0.9:
            return "decreasing"
        else:
            return "stable"
    
    def close(self):
        """Close connections to databases."""
        if self.vector_store:
            try:
                self.vector_store.disconnect()
            except:
                pass
        
        if self.engine:
            try:
                self.engine.disconnect()
            except:
                pass