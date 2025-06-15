"""
Knowledge Gap Detection Engine

Provides automated detection of knowledge gaps and missing information
in the knowledge graph, with suggestions for filling identified gaps.
"""

import logging
import time
import re
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from collections import defaultdict, Counter

from memory_core.query.query_engine import AdvancedQueryEngine
from memory_core.query.query_types import QueryRequest, QueryType
from memory_core.model.knowledge_node import KnowledgeNode
from memory_core.model.relationship import Relationship


class GapType(Enum):
    """Types of knowledge gaps."""

    CONTENT_GAP = "content_gap"  # Missing content/information
    RELATIONSHIP_GAP = "relationship_gap"  # Missing connections
    DOMAIN_GAP = "domain_gap"  # Missing domain coverage
    TEMPORAL_GAP = "temporal_gap"  # Missing temporal information
    DEPTH_GAP = "depth_gap"  # Insufficient detail
    BREADTH_GAP = "breadth_gap"  # Missing related topics
    FACTUAL_GAP = "factual_gap"  # Missing factual information
    PROCEDURAL_GAP = "procedural_gap"  # Missing process/method information


class GapSeverity(Enum):
    """Severity levels for knowledge gaps."""

    CRITICAL = "critical"  # Major gap affecting core functionality
    HIGH = "high"  # Important gap needing attention
    MEDIUM = "medium"  # Moderate gap worth addressing
    LOW = "low"  # Minor gap with low impact


@dataclass
class KnowledgeGap:
    """Identified knowledge gap."""

    gap_id: str
    gap_type: GapType
    severity: GapSeverity
    description: str
    affected_area: str
    missing_elements: List[str]
    related_nodes: List[str]
    confidence: float
    detection_method: str
    suggested_actions: List[str]
    potential_sources: List[str]
    priority_score: float
    detected_at: datetime


@dataclass
class GapAnalysis:
    """Comprehensive gap analysis result."""

    analyzed_domain: str
    total_gaps_found: int
    gaps_by_type: Dict[GapType, int]
    gaps_by_severity: Dict[GapSeverity, int]
    critical_gaps: List[KnowledgeGap]
    gap_coverage_score: float  # 0-1, higher is better coverage
    recommendations: List[str]
    analysis_confidence: float


@dataclass
class GapDetectionReport:
    """Knowledge gap detection report."""

    gap_analyses: List[GapAnalysis]
    total_gaps_detected: int
    overall_coverage_score: float
    priority_gaps: List[KnowledgeGap]
    global_recommendations: List[str]
    detection_time_ms: float


class ContentGapDetector:
    """Detects content and information gaps."""

    def __init__(self, query_engine: AdvancedQueryEngine):
        self.query_engine = query_engine
        self.logger = logging.getLogger(__name__)

        # Expected content patterns for different domains
        self.content_expectations = {
            "definition": [
                "what is",
                "definition",
                "meaning",
                "refers to",
                "describes",
                "characterize",
                "define",
                "explain",
            ],
            "example": [
                "example",
                "instance",
                "case",
                "illustration",
                "demonstration",
                "sample",
                "for instance",
                "such as",
            ],
            "cause": [
                "cause",
                "reason",
                "because",
                "due to",
                "results from",
                "stems from",
                "leads to",
                "triggers",
            ],
            "effect": [
                "effect",
                "result",
                "consequence",
                "outcome",
                "impact",
                "leads to",
                "causes",
                "produces",
            ],
            "process": [
                "process",
                "method",
                "procedure",
                "steps",
                "how to",
                "approach",
                "technique",
                "way to",
            ],
            "comparison": [
                "compare",
                "contrast",
                "difference",
                "similar",
                "unlike",
                "versus",
                "compared to",
                "in contrast",
            ],
        }

    def detect_content_gaps(self, domain_nodes: List[KnowledgeNode]) -> List[KnowledgeGap]:
        """
        Detect content gaps in a set of domain nodes.

        Args:
            domain_nodes: Nodes representing a knowledge domain

        Returns:
            List of detected content gaps
        """
        gaps = []

        try:
            # Analyze content completeness
            completeness_gaps = self._analyze_content_completeness(domain_nodes)
            gaps.extend(completeness_gaps)

            # Analyze missing essential information
            essential_gaps = self._analyze_missing_essential_info(domain_nodes)
            gaps.extend(essential_gaps)

            # Analyze depth gaps
            depth_gaps = self._analyze_depth_gaps(domain_nodes)
            gaps.extend(depth_gaps)

            # Analyze missing examples and illustrations
            example_gaps = self._analyze_missing_examples(domain_nodes)
            gaps.extend(example_gaps)

            return gaps

        except Exception as e:
            self.logger.error(f"Error detecting content gaps: {e}")
            return []

    def _analyze_content_completeness(self, nodes: List[KnowledgeNode]) -> List[KnowledgeGap]:
        """Analyze completeness of content across nodes."""
        gaps = []

        # Group nodes by topic/concept
        topic_groups = self._group_nodes_by_topic(nodes)

        for topic, topic_nodes in topic_groups.items():
            # Check for missing content types
            missing_types = self._find_missing_content_types(topic_nodes)

            for missing_type in missing_types:
                gap = KnowledgeGap(
                    gap_id=f"content_completeness_{topic}_{missing_type}",
                    gap_type=GapType.CONTENT_GAP,
                    severity=self._assess_content_gap_severity(missing_type, len(topic_nodes)),
                    description=f"Missing {missing_type} information for {topic}",
                    affected_area=topic,
                    missing_elements=[missing_type],
                    related_nodes=[node.node_id for node in topic_nodes],
                    confidence=0.7,
                    detection_method="content_type_analysis",
                    suggested_actions=[
                        f"Add {missing_type} information",
                        f"Research {missing_type} aspects of {topic}",
                    ],
                    potential_sources=[
                        f"Academic sources on {topic}",
                        f"Reference materials for {topic}",
                    ],
                    priority_score=self._calculate_priority_score(missing_type, len(topic_nodes)),
                    detected_at=datetime.now(),
                )
                gaps.append(gap)

        return gaps

    def _group_nodes_by_topic(self, nodes: List[KnowledgeNode]) -> Dict[str, List[KnowledgeNode]]:
        """Group nodes by their main topic or concept."""
        topic_groups = defaultdict(list)

        for node in nodes:
            # Extract main topic from content or metadata
            topic = self._extract_main_topic(node)
            topic_groups[topic].append(node)

        return dict(topic_groups)

    def _extract_main_topic(self, node: KnowledgeNode) -> str:
        """Extract the main topic from a node."""
        # Check metadata first
        metadata = node.metadata or {}

        if "topic" in metadata:
            return str(metadata["topic"])

        if "domain" in metadata:
            return str(metadata["domain"])

        if "category" in metadata:
            return str(metadata["category"])

        # Extract from content
        content_words = node.content.split()[:10]  # First 10 words

        # Look for proper nouns or key terms
        import re

        proper_nouns = re.findall(r"\b[A-Z][a-z]+\b", " ".join(content_words))

        if proper_nouns:
            return proper_nouns[0]

        # Fallback to node type or generic
        return node.node_type or "general"

    def _find_missing_content_types(self, nodes: List[KnowledgeNode]) -> List[str]:
        """Find missing content types in a group of nodes."""
        # Analyze content for presence of different types
        present_types = set()

        for node in nodes:
            content_lower = node.content.lower()

            for content_type, indicators in self.content_expectations.items():
                if any(indicator in content_lower for indicator in indicators):
                    present_types.add(content_type)

        # Determine expected types based on domain
        expected_types = {"definition", "example"}  # Minimum expected

        # Add more expectations based on content
        combined_content = " ".join(node.content.lower() for node in nodes)

        if any(word in combined_content for word in ["process", "method", "procedure"]):
            expected_types.add("process")

        if any(word in combined_content for word in ["cause", "effect", "result"]):
            expected_types.update(["cause", "effect"])

        # Find missing types
        missing_types = expected_types - present_types
        return list(missing_types)

    def _assess_content_gap_severity(self, missing_type: str, node_count: int) -> GapSeverity:
        """Assess severity of a content gap."""
        # Critical gaps
        if missing_type == "definition" and node_count > 0:
            return GapSeverity.CRITICAL

        # High priority gaps
        if missing_type in ["example", "process"] and node_count >= 3:
            return GapSeverity.HIGH

        # Medium priority gaps
        if missing_type in ["cause", "effect"] and node_count >= 2:
            return GapSeverity.MEDIUM

        # Default to low
        return GapSeverity.LOW

    def _calculate_priority_score(self, missing_type: str, node_count: int) -> float:
        """Calculate priority score for a gap."""
        base_scores = {
            "definition": 0.9,
            "example": 0.7,
            "process": 0.8,
            "cause": 0.6,
            "effect": 0.6,
            "comparison": 0.5,
        }

        base_score = base_scores.get(missing_type, 0.4)

        # Adjust for node count (more nodes = higher priority)
        node_factor = min(node_count / 5, 1.0)

        return base_score * (0.7 + 0.3 * node_factor)

    def _analyze_missing_essential_info(self, nodes: List[KnowledgeNode]) -> List[KnowledgeGap]:
        """Analyze missing essential information."""
        gaps = []

        # Look for incomplete information patterns
        for node in nodes:
            content = node.content

            # Check for incomplete sentences or thoughts
            if self._has_incomplete_information(content):
                gap = KnowledgeGap(
                    gap_id=f"essential_info_{node.node_id}",
                    gap_type=GapType.CONTENT_GAP,
                    severity=GapSeverity.MEDIUM,
                    description=f"Incomplete essential information in {node.node_id}",
                    affected_area=node.node_type or "content",
                    missing_elements=["complete_information"],
                    related_nodes=[node.node_id],
                    confidence=0.6,
                    detection_method="incomplete_information_analysis",
                    suggested_actions=[
                        "Complete incomplete statements",
                        "Add missing details and context",
                    ],
                    potential_sources=["Original source verification"],
                    priority_score=0.6,
                    detected_at=datetime.now(),
                )
                gaps.append(gap)

        return gaps

    def _has_incomplete_information(self, content: str) -> bool:
        """Check if content has incomplete information patterns."""
        # Look for incomplete patterns
        incomplete_patterns = [
            r"\.\.\.",  # Ellipsis
            r"\[?\?\]?",  # Question marks indicating uncertainty
            r"\b(?:unclear|unknown|uncertain|incomplete|partial)\b",
            r"\b(?:need more|requires additional|insufficient)\b",
            r"\b(?:to be determined|TBD|TODO)\b",
        ]

        for pattern in incomplete_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True

        # Check for very short content that might be incomplete
        sentences = re.split(r"[.!?]+", content)
        meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if len(meaningful_sentences) < 2 and len(content) < 100:
            return True

        return False

    def _analyze_depth_gaps(self, nodes: List[KnowledgeNode]) -> List[KnowledgeGap]:
        """Analyze depth gaps - insufficient detail."""
        gaps = []

        for node in nodes:
            depth_score = self._assess_content_depth(node.content)

            if depth_score < 0.4:  # Low depth threshold
                gap = KnowledgeGap(
                    gap_id=f"depth_gap_{node.node_id}",
                    gap_type=GapType.DEPTH_GAP,
                    severity=GapSeverity.MEDIUM,
                    description=f"Insufficient detail in {node.node_id}",
                    affected_area=node.node_type or "content",
                    missing_elements=["detailed_information"],
                    related_nodes=[node.node_id],
                    confidence=0.7,
                    detection_method="depth_analysis",
                    suggested_actions=[
                        "Add more detailed explanations",
                        "Include supporting information and context",
                    ],
                    potential_sources=["Detailed references", "Expert sources"],
                    priority_score=0.5 + depth_score * 0.3,
                    detected_at=datetime.now(),
                )
                gaps.append(gap)

        return gaps

    def _assess_content_depth(self, content: str) -> float:
        """Assess the depth of content."""
        depth_score = 0.0

        # Length factor
        length_factor = min(len(content) / 500, 1.0)
        depth_score += length_factor * 0.3

        # Sentence complexity
        sentences = re.split(r"[.!?]+", content)
        if sentences:
            avg_sentence_length = len(content.split()) / len(sentences)
            complexity_factor = min(avg_sentence_length / 15, 1.0)
            depth_score += complexity_factor * 0.2

        # Detail indicators
        detail_indicators = [
            "specifically",
            "detailed",
            "comprehensive",
            "thorough",
            "in particular",
            "furthermore",
            "moreover",
            "additionally",
        ]

        detail_count = sum(1 for indicator in detail_indicators if indicator in content.lower())
        detail_factor = min(detail_count / 3, 1.0)
        depth_score += detail_factor * 0.2

        # Technical depth
        technical_patterns = [
            r"\b[A-Z]{2,}\b",  # Acronyms
            r"\d+(?:\.\d+)?",  # Numbers
            r"\b\w+ly\b",  # Adverbs (often technical)
        ]

        technical_count = sum(len(re.findall(pattern, content)) for pattern in technical_patterns)
        technical_factor = min(technical_count / 10, 1.0)
        depth_score += technical_factor * 0.3

        return min(depth_score, 1.0)

    def _analyze_missing_examples(self, nodes: List[KnowledgeNode]) -> List[KnowledgeGap]:
        """Analyze missing examples and illustrations."""
        gaps = []

        # Group nodes and check for examples
        topic_groups = self._group_nodes_by_topic(nodes)

        for topic, topic_nodes in topic_groups.items():
            if len(topic_nodes) < 2:  # Skip single nodes
                continue

            has_examples = any(self._has_examples(node.content) for node in topic_nodes)

            if not has_examples:
                gap = KnowledgeGap(
                    gap_id=f"examples_gap_{topic}",
                    gap_type=GapType.CONTENT_GAP,
                    severity=GapSeverity.MEDIUM,
                    description=f"Missing examples for {topic}",
                    affected_area=topic,
                    missing_elements=["examples", "illustrations"],
                    related_nodes=[node.node_id for node in topic_nodes],
                    confidence=0.8,
                    detection_method="example_analysis",
                    suggested_actions=[
                        f"Add concrete examples for {topic}",
                        f"Include practical illustrations",
                    ],
                    potential_sources=[f"Case studies on {topic}", "Practical guides"],
                    priority_score=0.6,
                    detected_at=datetime.now(),
                )
                gaps.append(gap)

        return gaps

    def _has_examples(self, content: str) -> bool:
        """Check if content contains examples."""
        example_indicators = [
            "example",
            "instance",
            "case",
            "illustration",
            "demonstration",
            "for example",
            "such as",
            "for instance",
            "e.g.",
            "like",
        ]

        content_lower = content.lower()
        return any(indicator in content_lower for indicator in example_indicators)


class RelationshipGapDetector:
    """Detects missing relationships and connections."""

    def __init__(self, query_engine: AdvancedQueryEngine):
        self.query_engine = query_engine
        self.logger = logging.getLogger(__name__)

    def detect_relationship_gaps(self, nodes: List[KnowledgeNode]) -> List[KnowledgeGap]:
        """
        Detect missing relationships between nodes.

        Args:
            nodes: Nodes to analyze for relationship gaps

        Returns:
            List of detected relationship gaps
        """
        gaps = []

        try:
            # Find isolated nodes
            isolation_gaps = self._find_isolated_nodes(nodes)
            gaps.extend(isolation_gaps)

            # Find missing logical connections
            logical_gaps = self._find_missing_logical_connections(nodes)
            gaps.extend(logical_gaps)

            # Find missing hierarchical relationships
            hierarchy_gaps = self._find_missing_hierarchical_relationships(nodes)
            gaps.extend(hierarchy_gaps)

            return gaps

        except Exception as e:
            self.logger.error(f"Error detecting relationship gaps: {e}")
            return []

    def _find_isolated_nodes(self, nodes: List[KnowledgeNode]) -> List[KnowledgeGap]:
        """Find nodes with no or very few relationships."""
        gaps = []

        for node in nodes:
            relationship_count = (
                len(node.relationships)
                if hasattr(node, "relationships") and node.relationships
                else 0
            )

            if relationship_count == 0:
                gap = KnowledgeGap(
                    gap_id=f"isolation_gap_{node.node_id}",
                    gap_type=GapType.RELATIONSHIP_GAP,
                    severity=GapSeverity.HIGH,
                    description=f"Isolated node with no relationships: {node.node_id}",
                    affected_area="connectivity",
                    missing_elements=["relationships", "connections"],
                    related_nodes=[node.node_id],
                    confidence=0.9,
                    detection_method="isolation_analysis",
                    suggested_actions=[
                        "Identify relevant relationships for this node",
                        "Connect to related concepts and entities",
                    ],
                    potential_sources=["Related knowledge sources", "Domain expertise"],
                    priority_score=0.8,
                    detected_at=datetime.now(),
                )
                gaps.append(gap)

            elif relationship_count <= 2:  # Very few relationships
                gap = KnowledgeGap(
                    gap_id=f"weak_connectivity_{node.node_id}",
                    gap_type=GapType.RELATIONSHIP_GAP,
                    severity=GapSeverity.MEDIUM,
                    description=f"Weakly connected node: {node.node_id}",
                    affected_area="connectivity",
                    missing_elements=["additional_relationships"],
                    related_nodes=[node.node_id],
                    confidence=0.7,
                    detection_method="weak_connectivity_analysis",
                    suggested_actions=["Add more relationships to improve connectivity"],
                    potential_sources=["Domain knowledge", "Related concepts"],
                    priority_score=0.6,
                    detected_at=datetime.now(),
                )
                gaps.append(gap)

        return gaps

    def _find_missing_logical_connections(self, nodes: List[KnowledgeNode]) -> List[KnowledgeGap]:
        """Find missing logical connections based on content analysis."""
        gaps = []

        # Analyze content for logical relationship indicators
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i + 1 :]:
                # Check if nodes should be connected but aren't
                should_connect = self._should_nodes_be_connected(node1, node2)
                are_connected = self._are_nodes_connected(node1, node2)

                if should_connect and not are_connected:
                    connection_type = self._suggest_connection_type(node1, node2)

                    gap = KnowledgeGap(
                        gap_id=f"logical_connection_{node1.node_id}_{node2.node_id}",
                        gap_type=GapType.RELATIONSHIP_GAP,
                        severity=GapSeverity.MEDIUM,
                        description=f"Missing logical connection between {node1.node_id} and {node2.node_id}",
                        affected_area="logical_connectivity",
                        missing_elements=[f"{connection_type}_relationship"],
                        related_nodes=[node1.node_id, node2.node_id],
                        confidence=0.6,
                        detection_method="logical_connection_analysis",
                        suggested_actions=[
                            f"Add {connection_type} relationship",
                            "Verify logical connection validity",
                        ],
                        potential_sources=["Domain analysis", "Expert review"],
                        priority_score=0.5,
                        detected_at=datetime.now(),
                    )
                    gaps.append(gap)

        return gaps

    def _should_nodes_be_connected(self, node1: KnowledgeNode, node2: KnowledgeNode) -> bool:
        """Determine if two nodes should be connected based on content similarity."""
        # Simple content similarity check
        content1_words = set(node1.content.lower().split())
        content2_words = set(node2.content.lower().split())

        if len(content1_words) == 0 or len(content2_words) == 0:
            return False

        # Calculate Jaccard similarity
        intersection = len(content1_words & content2_words)
        union = len(content1_words | content2_words)

        similarity = intersection / union if union > 0 else 0

        # Threshold for potential connection
        return similarity > 0.3

    def _are_nodes_connected(self, node1: KnowledgeNode, node2: KnowledgeNode) -> bool:
        """Check if two nodes are already connected."""
        # Check if node1 has relationships to node2
        if hasattr(node1, "relationships") and node1.relationships:
            for rel in node1.relationships:
                if rel.target_id == node2.node_id or rel.source_id == node2.node_id:
                    return True

        # Check if node2 has relationships to node1
        if hasattr(node2, "relationships") and node2.relationships:
            for rel in node2.relationships:
                if rel.target_id == node1.node_id or rel.source_id == node1.node_id:
                    return True

        return False

    def _suggest_connection_type(self, node1: KnowledgeNode, node2: KnowledgeNode) -> str:
        """Suggest the type of connection between two nodes."""
        # Analyze content to suggest relationship type
        content1 = node1.content.lower()
        content2 = node2.content.lower()

        # Look for hierarchical indicators
        if any(word in content1 for word in ["part of", "component", "element"]):
            return "part_of"

        if any(word in content2 for word in ["part of", "component", "element"]):
            return "contains"

        # Look for causal indicators
        if any(word in content1 for word in ["causes", "leads to", "results in"]):
            return "causes"

        # Look for similarity indicators
        if any(word in content1 for word in ["similar", "like", "comparable"]):
            return "similar_to"

        # Default to generic relationship
        return "related_to"

    def _find_missing_hierarchical_relationships(
        self, nodes: List[KnowledgeNode]
    ) -> List[KnowledgeGap]:
        """Find missing hierarchical relationships."""
        gaps = []

        # Look for nodes that mention hierarchical concepts without proper relationships
        for node in nodes:
            content_lower = node.content.lower()

            # Check for hierarchical language without corresponding relationships
            hierarchical_indicators = [
                "part of",
                "component of",
                "element of",
                "subset of",
                "category of",
                "type of",
                "kind of",
                "belongs to",
            ]

            has_hierarchical_language = any(
                indicator in content_lower for indicator in hierarchical_indicators
            )

            if has_hierarchical_language:
                # Check if node has appropriate hierarchical relationships
                has_hierarchical_rels = self._has_hierarchical_relationships(node)

                if not has_hierarchical_rels:
                    gap = KnowledgeGap(
                        gap_id=f"hierarchy_gap_{node.node_id}",
                        gap_type=GapType.RELATIONSHIP_GAP,
                        severity=GapSeverity.MEDIUM,
                        description=f"Missing hierarchical relationships for {node.node_id}",
                        affected_area="hierarchy",
                        missing_elements=["hierarchical_relationships"],
                        related_nodes=[node.node_id],
                        confidence=0.7,
                        detection_method="hierarchical_analysis",
                        suggested_actions=[
                            "Add appropriate hierarchical relationships",
                            "Identify parent/child concepts",
                        ],
                        potential_sources=["Ontology sources", "Domain hierarchies"],
                        priority_score=0.6,
                        detected_at=datetime.now(),
                    )
                    gaps.append(gap)

        return gaps

    def _has_hierarchical_relationships(self, node: KnowledgeNode) -> bool:
        """Check if node has hierarchical relationships."""
        if not hasattr(node, "relationships") or not node.relationships:
            return False

        hierarchical_types = [
            "part_of",
            "contains",
            "is_a",
            "instance_of",
            "subclass_of",
            "parent_of",
            "child_of",
            "belongs_to",
        ]

        for rel in node.relationships:
            if rel.relationship_type in hierarchical_types:
                return True

        return False


class DomainGapDetector:
    """Detects missing domain coverage and breadth gaps."""

    def __init__(self, query_engine: AdvancedQueryEngine):
        self.query_engine = query_engine
        self.logger = logging.getLogger(__name__)

    def detect_domain_gaps(
        self, nodes: List[KnowledgeNode], expected_domains: List[str] = None
    ) -> List[KnowledgeGap]:
        """
        Detect domain coverage gaps.

        Args:
            nodes: Nodes to analyze for domain coverage
            expected_domains: Expected domains that should be covered

        Returns:
            List of detected domain gaps
        """
        gaps = []

        try:
            # Analyze current domain coverage
            current_domains = self._analyze_current_domains(nodes)

            # Find missing domains
            if expected_domains:
                missing_domains = set(expected_domains) - set(current_domains.keys())

                for missing_domain in missing_domains:
                    gap = KnowledgeGap(
                        gap_id=f"domain_gap_{missing_domain}",
                        gap_type=GapType.DOMAIN_GAP,
                        severity=GapSeverity.HIGH,
                        description=f"Missing domain coverage: {missing_domain}",
                        affected_area=missing_domain,
                        missing_elements=[missing_domain],
                        related_nodes=[],
                        confidence=0.8,
                        detection_method="domain_coverage_analysis",
                        suggested_actions=[
                            f"Add content for {missing_domain} domain",
                            f"Research {missing_domain} knowledge",
                        ],
                        potential_sources=[
                            f"{missing_domain} domain experts",
                            f"{missing_domain} literature",
                        ],
                        priority_score=0.8,
                        detected_at=datetime.now(),
                    )
                    gaps.append(gap)

            # Find undercovered domains
            undercovered_gaps = self._find_undercovered_domains(current_domains)
            gaps.extend(undercovered_gaps)

            return gaps

        except Exception as e:
            self.logger.error(f"Error detecting domain gaps: {e}")
            return []

    def _analyze_current_domains(
        self, nodes: List[KnowledgeNode]
    ) -> Dict[str, List[KnowledgeNode]]:
        """Analyze current domain coverage."""
        domain_coverage = defaultdict(list)

        for node in nodes:
            # Extract domain from metadata or content
            domain = self._extract_domain(node)
            domain_coverage[domain].append(node)

        return dict(domain_coverage)

    def _extract_domain(self, node: KnowledgeNode) -> str:
        """Extract domain from node."""
        metadata = node.metadata or {}

        # Check metadata
        if "domain" in metadata:
            return str(metadata["domain"])

        if "category" in metadata:
            return str(metadata["category"])

        # Extract from node type
        if node.node_type:
            return node.node_type

        # Extract from content keywords
        domain_keywords = {
            "technology": ["software", "computer", "digital", "tech", "algorithm"],
            "science": ["research", "study", "experiment", "theory", "hypothesis"],
            "business": ["company", "market", "finance", "customer", "revenue"],
            "health": ["medical", "health", "patient", "treatment", "diagnosis"],
            "education": ["learning", "student", "teaching", "academic", "curriculum"],
        }

        content_lower = node.content.lower()

        for domain, keywords in domain_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                return domain

        return "general"

    def _find_undercovered_domains(
        self, domain_coverage: Dict[str, List[KnowledgeNode]]
    ) -> List[KnowledgeGap]:
        """Find domains with insufficient coverage."""
        gaps = []

        # Calculate coverage statistics
        total_nodes = sum(len(nodes) for nodes in domain_coverage.values())

        for domain, nodes in domain_coverage.items():
            coverage_ratio = len(nodes) / total_nodes if total_nodes > 0 else 0

            # Consider domain undercovered if it has very few nodes
            if len(nodes) <= 2 and coverage_ratio < 0.1:
                gap = KnowledgeGap(
                    gap_id=f"undercovered_domain_{domain}",
                    gap_type=GapType.DOMAIN_GAP,
                    severity=GapSeverity.MEDIUM,
                    description=f"Undercovered domain: {domain} ({len(nodes)} nodes)",
                    affected_area=domain,
                    missing_elements=[f"additional_{domain}_content"],
                    related_nodes=[node.node_id for node in nodes],
                    confidence=0.7,
                    detection_method="coverage_analysis",
                    suggested_actions=[f"Expand {domain} content", f"Add more {domain} knowledge"],
                    potential_sources=[f"{domain} resources", f"{domain} experts"],
                    priority_score=0.6,
                    detected_at=datetime.now(),
                )
                gaps.append(gap)

        return gaps


class KnowledgeGapDetector:
    """
    Main Knowledge Gap Detection Engine.

    Provides comprehensive automated detection of knowledge gaps
    across content, relationships, domains, and other dimensions.
    """

    def __init__(self, query_engine: AdvancedQueryEngine):
        """
        Initialize the Knowledge Gap Detector.

        Args:
            query_engine: Query engine for accessing knowledge graph
        """
        self.query_engine = query_engine

        # Initialize detectors
        self.content_detector = ContentGapDetector(query_engine)
        self.relationship_detector = RelationshipGapDetector(query_engine)
        self.domain_detector = DomainGapDetector(query_engine)

        self.logger = logging.getLogger(__name__)

        # Statistics
        self.stats = {
            "gap_detections_performed": 0,
            "total_gaps_detected": 0,
            "gaps_by_type": {gap_type.value: 0 for gap_type in GapType},
            "gaps_by_severity": {severity.value: 0 for severity in GapSeverity},
            "avg_detection_time_ms": 0.0,
            "avg_coverage_score": 0.0,
        }

    def detect_knowledge_gaps(
        self,
        nodes: List[KnowledgeNode],
        domain_name: str = None,
        gap_types: List[GapType] = None,
        expected_domains: List[str] = None,
    ) -> GapAnalysis:
        """
        Perform comprehensive knowledge gap detection.

        Args:
            nodes: Knowledge nodes to analyze
            domain_name: Name of the domain being analyzed
            gap_types: Types of gaps to detect (all if None)
            expected_domains: Expected domains for coverage analysis

        Returns:
            GapAnalysis with comprehensive results
        """
        start_time = time.time()

        if gap_types is None:
            gap_types = list(GapType)

        try:
            self.logger.info(f"Detecting knowledge gaps in {len(nodes)} nodes")

            all_gaps = []

            # Content gap detection
            if any(gt in gap_types for gt in [GapType.CONTENT_GAP, GapType.DEPTH_GAP]):
                content_gaps = self.content_detector.detect_content_gaps(nodes)
                all_gaps.extend(content_gaps)

            # Relationship gap detection
            if GapType.RELATIONSHIP_GAP in gap_types:
                relationship_gaps = self.relationship_detector.detect_relationship_gaps(nodes)
                all_gaps.extend(relationship_gaps)

            # Domain gap detection
            if any(gt in gap_types for gt in [GapType.DOMAIN_GAP, GapType.BREADTH_GAP]):
                domain_gaps = self.domain_detector.detect_domain_gaps(nodes, expected_domains)
                all_gaps.extend(domain_gaps)

            # Analyze gaps
            gaps_by_type = self._count_gaps_by_type(all_gaps)
            gaps_by_severity = self._count_gaps_by_severity(all_gaps)

            # Identify critical gaps
            critical_gaps = [gap for gap in all_gaps if gap.severity == GapSeverity.CRITICAL]

            # Calculate coverage score
            coverage_score = self._calculate_coverage_score(all_gaps, len(nodes))

            # Generate recommendations
            recommendations = self._generate_recommendations(all_gaps, coverage_score)

            # Calculate analysis confidence
            analysis_confidence = self._calculate_analysis_confidence(all_gaps, len(nodes))

            processing_time = (time.time() - start_time) * 1000

            analysis = GapAnalysis(
                analyzed_domain=domain_name or "unknown",
                total_gaps_found=len(all_gaps),
                gaps_by_type=gaps_by_type,
                gaps_by_severity=gaps_by_severity,
                critical_gaps=critical_gaps,
                gap_coverage_score=coverage_score,
                recommendations=recommendations,
                analysis_confidence=analysis_confidence,
            )

            # Update statistics
            self._update_statistics(analysis, processing_time)

            self.logger.info(f"Gap detection completed: {len(all_gaps)} gaps found")
            return analysis

        except Exception as e:
            self.logger.error(f"Error detecting knowledge gaps: {e}")
            return self._create_error_analysis(domain_name, start_time)

    def detect_gaps_multiple_domains(
        self, domain_nodes: Dict[str, List[KnowledgeNode]]
    ) -> GapDetectionReport:
        """
        Detect gaps across multiple domains and generate comprehensive report.

        Args:
            domain_nodes: Dictionary mapping domain names to node lists

        Returns:
            GapDetectionReport with results across all domains
        """
        start_time = time.time()

        try:
            self.logger.info(f"Detecting gaps across {len(domain_nodes)} domains")

            gap_analyses = []
            all_gaps = []

            # Analyze each domain
            for domain_name, nodes in domain_nodes.items():
                domain_analysis = self.detect_knowledge_gaps(nodes, domain_name)
                gap_analyses.append(domain_analysis)

                # Collect all gaps for global analysis
                all_gaps.extend(self._get_gaps_from_analysis(domain_analysis))

            # Calculate overall metrics
            total_gaps = len(all_gaps)
            overall_coverage = np.mean([analysis.gap_coverage_score for analysis in gap_analyses])

            # Identify priority gaps across all domains
            priority_gaps = sorted(all_gaps, key=lambda g: g.priority_score, reverse=True)[:10]

            # Generate global recommendations
            global_recommendations = self._generate_global_recommendations(gap_analyses)

            processing_time = (time.time() - start_time) * 1000

            report = GapDetectionReport(
                gap_analyses=gap_analyses,
                total_gaps_detected=total_gaps,
                overall_coverage_score=overall_coverage,
                priority_gaps=priority_gaps,
                global_recommendations=global_recommendations,
                detection_time_ms=processing_time,
            )

            self.logger.info(f"Multi-domain gap detection completed: {total_gaps} total gaps")
            return report

        except Exception as e:
            self.logger.error(f"Error in multi-domain gap detection: {e}")
            return GapDetectionReport(
                gap_analyses=[],
                total_gaps_detected=0,
                overall_coverage_score=0.0,
                priority_gaps=[],
                global_recommendations=[],
                detection_time_ms=(time.time() - start_time) * 1000,
            )

    def _count_gaps_by_type(self, gaps: List[KnowledgeGap]) -> Dict[GapType, int]:
        """Count gaps by type."""
        type_counts = {gap_type: 0 for gap_type in GapType}

        for gap in gaps:
            type_counts[gap.gap_type] += 1

        return type_counts

    def _count_gaps_by_severity(self, gaps: List[KnowledgeGap]) -> Dict[GapSeverity, int]:
        """Count gaps by severity."""
        severity_counts = {severity: 0 for severity in GapSeverity}

        for gap in gaps:
            severity_counts[gap.severity] += 1

        return severity_counts

    def _calculate_coverage_score(self, gaps: List[KnowledgeGap], node_count: int) -> float:
        """Calculate coverage score (0-1, higher is better)."""
        if node_count == 0:
            return 0.0

        # Base score starts high and decreases with gaps
        base_score = 1.0

        # Penalty for each gap based on severity
        severity_penalties = {
            GapSeverity.CRITICAL: 0.2,
            GapSeverity.HIGH: 0.1,
            GapSeverity.MEDIUM: 0.05,
            GapSeverity.LOW: 0.02,
        }

        total_penalty = 0.0
        for gap in gaps:
            penalty = severity_penalties.get(gap.severity, 0.02)
            total_penalty += penalty

        # Normalize penalty by node count
        normalized_penalty = total_penalty / max(node_count / 10, 1)

        coverage_score = max(0.0, base_score - normalized_penalty)
        return coverage_score

    def _generate_recommendations(
        self, gaps: List[KnowledgeGap], coverage_score: float
    ) -> List[str]:
        """Generate recommendations based on detected gaps."""
        recommendations = []

        if not gaps:
            recommendations.append("No significant knowledge gaps detected")
            return recommendations

        # Priority recommendations based on gap severity
        critical_gaps = [g for g in gaps if g.severity == GapSeverity.CRITICAL]
        if critical_gaps:
            recommendations.append(
                "Address critical gaps immediately to prevent knowledge degradation"
            )

        # Type-specific recommendations
        gap_types = {gap.gap_type for gap in gaps}

        if GapType.CONTENT_GAP in gap_types:
            recommendations.append("Improve content completeness and depth")

        if GapType.RELATIONSHIP_GAP in gap_types:
            recommendations.append("Enhance knowledge connectivity through better relationships")

        if GapType.DOMAIN_GAP in gap_types:
            recommendations.append("Expand domain coverage to address knowledge breadth")

        # Coverage-based recommendations
        if coverage_score < 0.5:
            recommendations.append(
                "Comprehensive knowledge enhancement needed - coverage below 50%"
            )
        elif coverage_score < 0.7:
            recommendations.append("Moderate knowledge enhancement recommended")

        return recommendations[:5]  # Top 5 recommendations

    def _calculate_analysis_confidence(self, gaps: List[KnowledgeGap], node_count: int) -> float:
        """Calculate confidence in the gap analysis."""
        if node_count == 0:
            return 0.0

        # Base confidence from node count
        node_factor = min(node_count / 20, 1.0)  # More nodes = higher confidence

        # Confidence from gap detection consistency
        if gaps:
            avg_gap_confidence = np.mean([gap.confidence for gap in gaps])
        else:
            avg_gap_confidence = 0.8  # High confidence when no gaps found

        # Combined confidence
        analysis_confidence = (node_factor * 0.4) + (avg_gap_confidence * 0.6)

        return analysis_confidence

    def _get_gaps_from_analysis(self, analysis: GapAnalysis) -> List[KnowledgeGap]:
        """Extract all gaps from a gap analysis."""
        # In a full implementation, this would access the gaps stored in the analysis
        # For now, return the critical gaps as a proxy
        return analysis.critical_gaps

    def _generate_global_recommendations(self, analyses: List[GapAnalysis]) -> List[str]:
        """Generate global recommendations across all domain analyses."""
        recommendations = []

        # Analyze overall patterns
        total_gaps = sum(analysis.total_gaps_found for analysis in analyses)
        avg_coverage = np.mean([analysis.gap_coverage_score for analysis in analyses])

        if total_gaps > len(analyses) * 5:  # More than 5 gaps per domain on average
            recommendations.append("Implement systematic knowledge enhancement process")

        if avg_coverage < 0.6:
            recommendations.append("Focus on improving overall knowledge coverage")

        # Critical gap patterns
        critical_gap_count = sum(len(analysis.critical_gaps) for analysis in analyses)
        if critical_gap_count > 0:
            recommendations.append("Prioritize resolution of critical knowledge gaps")

        # Domain-specific patterns
        domain_coverages = {
            analysis.analyzed_domain: analysis.gap_coverage_score for analysis in analyses
        }

        worst_domain = min(domain_coverages.items(), key=lambda x: x[1])
        if worst_domain[1] < 0.4:
            recommendations.append(f"Focus enhancement efforts on {worst_domain[0]} domain")

        return recommendations[:5]  # Top 5 global recommendations

    def _create_error_analysis(self, domain_name: str, start_time: float) -> GapAnalysis:
        """Create error analysis when detection fails."""
        return GapAnalysis(
            analyzed_domain=domain_name or "error",
            total_gaps_found=0,
            gaps_by_type={gap_type: 0 for gap_type in GapType},
            gaps_by_severity={severity: 0 for severity in GapSeverity},
            critical_gaps=[],
            gap_coverage_score=0.0,
            recommendations=["Gap detection failed - review input data"],
            analysis_confidence=0.0,
        )

    def _update_statistics(self, analysis: GapAnalysis, processing_time: float):
        """Update detector statistics."""
        self.stats["gap_detections_performed"] += 1
        self.stats["total_gaps_detected"] += analysis.total_gaps_found

        # Update gap type distribution
        for gap_type, count in analysis.gaps_by_type.items():
            self.stats["gaps_by_type"][gap_type.value] += count

        # Update gap severity distribution
        for severity, count in analysis.gaps_by_severity.items():
            self.stats["gaps_by_severity"][severity.value] += count

        # Update average processing time
        total_time = self.stats["avg_detection_time_ms"] * (
            self.stats["gap_detections_performed"] - 1
        )
        self.stats["avg_detection_time_ms"] = (total_time + processing_time) / self.stats[
            "gap_detections_performed"
        ]

        # Update average coverage score
        total_coverage = self.stats["avg_coverage_score"] * (
            self.stats["gap_detections_performed"] - 1
        )
        self.stats["avg_coverage_score"] = (
            total_coverage + analysis.gap_coverage_score
        ) / self.stats["gap_detections_performed"]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get knowledge gap detector statistics.

        Returns:
            Dictionary with comprehensive statistics
        """
        return {
            "gap_detection": self.stats.copy(),
            "query_engine": self.query_engine.get_statistics(),
        }
