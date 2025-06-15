"""
Contradiction Resolution Engine

Provides automated workflows for resolving contradictions between knowledge nodes,
including conflict detection, resolution strategies, and consensus building.
"""

import logging
import time
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


class ResolutionStrategy(Enum):
    """Strategies for resolving contradictions."""

    SOURCE_AUTHORITY = "source_authority"  # Prefer higher authority sources
    CONSENSUS_VOTING = "consensus_voting"  # Use majority consensus
    TEMPORAL_PREFERENCE = "temporal_preference"  # Prefer more recent information
    CONFIDENCE_WEIGHTED = "confidence_weighted"  # Weight by confidence scores
    EXPERT_REVIEW = "expert_review"  # Flag for human expert review
    EVIDENCE_BASED = "evidence_based"  # Resolve based on supporting evidence
    HYBRID_APPROACH = "hybrid_approach"  # Combine multiple strategies


class ContradictionSeverity(Enum):
    """Severity levels for contradictions."""

    CRITICAL = "critical"  # Fundamental factual contradictions
    HIGH = "high"  # Important contradictions affecting key information
    MEDIUM = "medium"  # Moderate contradictions with limited impact
    LOW = "low"  # Minor contradictions or interpretation differences


class ResolutionStatus(Enum):
    """Status of contradiction resolution."""

    RESOLVED = "resolved"  # Contradiction successfully resolved
    PARTIALLY_RESOLVED = "partially_resolved"  # Some aspects resolved
    UNRESOLVED = "unresolved"  # Could not resolve contradiction
    REQUIRES_REVIEW = "requires_review"  # Needs human expert review
    DEFERRED = "deferred"  # Resolution deferred for later


@dataclass
class ContradictionCase:
    """A specific contradiction case between nodes."""

    case_id: str
    conflicting_nodes: List[str]
    contradiction_type: str
    severity: ContradictionSeverity
    conflicting_claims: List[str]
    evidence_for: Dict[str, List[str]]  # node_id -> supporting evidence
    evidence_against: Dict[str, List[str]]  # node_id -> contradicting evidence
    context: str
    detected_at: datetime
    confidence: float


@dataclass
class ResolutionAction:
    """An action taken to resolve a contradiction."""

    action_type: str  # 'prefer_node', 'merge_information', 'flag_for_review', etc.
    target_nodes: List[str]
    reasoning: str
    confidence: float
    supporting_data: Dict[str, Any]


@dataclass
class ResolutionResult:
    """Result of contradiction resolution process."""

    case_id: str
    resolution_status: ResolutionStatus
    strategy_used: ResolutionStrategy
    resolution_actions: List[ResolutionAction]
    final_recommendation: str
    confidence: float
    reasoning: str
    resolved_at: datetime
    resolution_metadata: Dict[str, Any]


@dataclass
class ResolutionReport:
    """Comprehensive contradiction resolution report."""

    resolution_results: List[ResolutionResult]
    total_contradictions: int
    resolved_count: int
    unresolved_count: int
    strategy_effectiveness: Dict[ResolutionStrategy, float]
    common_contradiction_types: List[Tuple[str, int]]
    recommendations: List[str]
    processing_time_ms: float


class ContradictionDetector:
    """Detects contradictions between knowledge nodes."""

    def __init__(self, query_engine: AdvancedQueryEngine):
        self.query_engine = query_engine
        self.logger = logging.getLogger(__name__)

        # Contradiction patterns
        self.contradiction_patterns = {
            "factual": [
                ("is", "is not"),
                ("true", "false"),
                ("exists", "does not exist"),
                ("has", "does not have"),
                ("can", "cannot"),
                ("will", "will not"),
            ],
            "numerical": [
                ("increase", "decrease"),
                ("more than", "less than"),
                ("higher", "lower"),
                ("greater", "smaller"),
            ],
            "temporal": [
                ("before", "after"),
                ("earlier", "later"),
                ("past", "future"),
                ("old", "new"),
            ],
            "qualitative": [
                ("good", "bad"),
                ("positive", "negative"),
                ("beneficial", "harmful"),
                ("effective", "ineffective"),
            ],
        }

    def detect_contradictions(self, nodes: List[KnowledgeNode]) -> List[ContradictionCase]:
        """
        Detect contradictions between knowledge nodes.

        Args:
            nodes: List of knowledge nodes to analyze

        Returns:
            List of detected contradiction cases
        """
        try:
            self.logger.info(f"Detecting contradictions among {len(nodes)} nodes")

            contradictions = []

            # Compare each pair of nodes
            for i, node1 in enumerate(nodes):
                for node2 in nodes[i + 1 :]:
                    case = self._analyze_node_pair(node1, node2)
                    if case:
                        contradictions.append(case)

            # Group and analyze related contradictions
            grouped_contradictions = self._group_related_contradictions(contradictions)

            self.logger.info(f"Found {len(grouped_contradictions)} contradiction cases")
            return grouped_contradictions

        except Exception as e:
            self.logger.error(f"Error detecting contradictions: {e}")
            return []

    def _analyze_node_pair(
        self, node1: KnowledgeNode, node2: KnowledgeNode
    ) -> Optional[ContradictionCase]:
        """Analyze a pair of nodes for contradictions."""
        try:
            content1 = node1.content.lower()
            content2 = node2.content.lower()

            # Check for direct contradictions
            contradictions = []

            for contradiction_type, patterns in self.contradiction_patterns.items():
                for positive, negative in patterns:
                    if self._has_contradiction_pattern(content1, content2, positive, negative):
                        contradictions.append(
                            {
                                "type": contradiction_type,
                                "pattern": (positive, negative),
                                "confidence": self._calculate_pattern_confidence(
                                    content1, content2, positive, negative
                                ),
                            }
                        )

            if not contradictions:
                return None

            # Find the most significant contradiction
            best_contradiction = max(contradictions, key=lambda x: x["confidence"])

            if best_contradiction["confidence"] < 0.5:
                return None

            # Create contradiction case
            case_id = f"contradiction_{node1.node_id}_{node2.node_id}_{int(time.time())}"

            severity = self._assess_contradiction_severity(best_contradiction, node1, node2)

            case = ContradictionCase(
                case_id=case_id,
                conflicting_nodes=[node1.node_id, node2.node_id],
                contradiction_type=best_contradiction["type"],
                severity=severity,
                conflicting_claims=[
                    self._extract_conflicting_claim(node1.content, best_contradiction["pattern"]),
                    self._extract_conflicting_claim(node2.content, best_contradiction["pattern"]),
                ],
                evidence_for={
                    node1.node_id: [node1.content[:200] + "..."],
                    node2.node_id: [node2.content[:200] + "..."],
                },
                evidence_against={},
                context=f"{best_contradiction['type']} contradiction involving {best_contradiction['pattern']}",
                detected_at=datetime.now(),
                confidence=best_contradiction["confidence"],
            )

            return case

        except Exception as e:
            self.logger.error(f"Error analyzing node pair {node1.node_id}, {node2.node_id}: {e}")
            return None

    def _has_contradiction_pattern(
        self, content1: str, content2: str, positive: str, negative: str
    ) -> bool:
        """Check if two contents have contradictory patterns."""
        return (positive in content1 and negative in content2) or (
            negative in content1 and positive in content2
        )

    def _calculate_pattern_confidence(
        self, content1: str, content2: str, positive: str, negative: str
    ) -> float:
        """Calculate confidence in pattern-based contradiction."""
        confidence = 0.5  # Base confidence

        # Check for exact pattern matches
        if positive in content1 and negative in content2:
            confidence += 0.3
        elif negative in content1 and positive in content2:
            confidence += 0.3

        # Check for context around patterns
        import re

        # Look for strong assertion words around patterns
        strong_indicators = ["definitely", "certainly", "always", "never", "absolutely"]

        for indicator in strong_indicators:
            if indicator in content1 or indicator in content2:
                confidence += 0.1
                break

        # Check for negation strength
        negation_indicators = ["not", "never", "cannot", "impossible", "false"]
        negation_count = sum(
            1 for indicator in negation_indicators if indicator in content1 or indicator in content2
        )

        confidence += min(negation_count * 0.05, 0.2)

        return min(confidence, 1.0)

    def _assess_contradiction_severity(
        self, contradiction: Dict[str, Any], node1: KnowledgeNode, node2: KnowledgeNode
    ) -> ContradictionSeverity:
        """Assess the severity of a contradiction."""
        severity_score = 0.0

        # Base severity on contradiction type
        type_severities = {"factual": 0.8, "numerical": 0.7, "temporal": 0.5, "qualitative": 0.4}

        severity_score += type_severities.get(contradiction["type"], 0.5)

        # Consider confidence
        severity_score += contradiction["confidence"] * 0.2

        # Consider node importance (simple heuristic based on content length)
        content_importance = (len(node1.content) + len(node2.content)) / 1000
        severity_score += min(content_importance * 0.1, 0.2)

        # Map to severity levels
        if severity_score >= 0.8:
            return ContradictionSeverity.CRITICAL
        elif severity_score >= 0.6:
            return ContradictionSeverity.HIGH
        elif severity_score >= 0.4:
            return ContradictionSeverity.MEDIUM
        else:
            return ContradictionSeverity.LOW

    def _extract_conflicting_claim(self, content: str, pattern: Tuple[str, str]) -> str:
        """Extract the specific conflicting claim from content."""
        # Find sentences containing the pattern
        import re

        sentences = re.split(r"[.!?]+", content)

        for sentence in sentences:
            if any(word in sentence.lower() for word in pattern):
                return sentence.strip()[:200]  # Limit length

        return content[:100] + "..."  # Fallback

    def _group_related_contradictions(
        self, contradictions: List[ContradictionCase]
    ) -> List[ContradictionCase]:
        """Group related contradictions to avoid duplicates."""
        # For now, just return unique contradictions
        # In a full implementation, this would merge related cases
        seen_pairs = set()
        unique_contradictions = []

        for case in contradictions:
            pair = tuple(sorted(case.conflicting_nodes))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                unique_contradictions.append(case)

        return unique_contradictions


class ResolutionStrategyEngine:
    """Implements different strategies for resolving contradictions."""

    def __init__(self, query_engine: AdvancedQueryEngine):
        self.query_engine = query_engine
        self.logger = logging.getLogger(__name__)

    def resolve_contradiction(
        self, case: ContradictionCase, strategy: ResolutionStrategy
    ) -> ResolutionResult:
        """
        Resolve a contradiction using the specified strategy.

        Args:
            case: Contradiction case to resolve
            strategy: Resolution strategy to use

        Returns:
            ResolutionResult with resolution outcome
        """
        try:
            self.logger.info(f"Resolving contradiction {case.case_id} using {strategy.value}")

            if strategy == ResolutionStrategy.SOURCE_AUTHORITY:
                return self._resolve_by_source_authority(case)
            elif strategy == ResolutionStrategy.CONSENSUS_VOTING:
                return self._resolve_by_consensus(case)
            elif strategy == ResolutionStrategy.TEMPORAL_PREFERENCE:
                return self._resolve_by_temporal_preference(case)
            elif strategy == ResolutionStrategy.CONFIDENCE_WEIGHTED:
                return self._resolve_by_confidence(case)
            elif strategy == ResolutionStrategy.EVIDENCE_BASED:
                return self._resolve_by_evidence(case)
            elif strategy == ResolutionStrategy.HYBRID_APPROACH:
                return self._resolve_by_hybrid_approach(case)
            else:
                return self._create_review_result(case, strategy)

        except Exception as e:
            self.logger.error(f"Error resolving contradiction {case.case_id}: {e}")
            return self._create_error_result(case, strategy)

    def _resolve_by_source_authority(self, case: ContradictionCase) -> ResolutionResult:
        """Resolve based on source authority."""
        # Get nodes and assess their source reliability
        node_authorities = {}

        for node_id in case.conflicting_nodes:
            # This would normally query for the actual node and assess authority
            # For now, use a placeholder authority score
            authority_score = self._get_node_authority_score(node_id)
            node_authorities[node_id] = authority_score

        # Select the node with highest authority
        preferred_node = max(node_authorities.items(), key=lambda x: x[1])

        if preferred_node[1] > 0.7:  # High authority threshold
            action = ResolutionAction(
                action_type="prefer_node",
                target_nodes=[preferred_node[0]],
                reasoning=f"Node {preferred_node[0]} has higher source authority ({preferred_node[1]:.2f})",
                confidence=preferred_node[1],
                supporting_data={"authority_scores": node_authorities},
            )

            return ResolutionResult(
                case_id=case.case_id,
                resolution_status=ResolutionStatus.RESOLVED,
                strategy_used=ResolutionStrategy.SOURCE_AUTHORITY,
                resolution_actions=[action],
                final_recommendation=f"Accept information from {preferred_node[0]} based on higher source authority",
                confidence=preferred_node[1],
                reasoning=action.reasoning,
                resolved_at=datetime.now(),
                resolution_metadata={"preferred_authority": preferred_node[1]},
            )
        else:
            return self._create_review_result(case, ResolutionStrategy.SOURCE_AUTHORITY)

    def _resolve_by_consensus(self, case: ContradictionCase) -> ResolutionResult:
        """Resolve based on consensus from similar nodes."""
        # Find similar nodes and check their consensus
        consensus_data = self._gather_consensus_data(case)

        if consensus_data["consensus_strength"] > 0.6:
            preferred_position = consensus_data["majority_position"]

            action = ResolutionAction(
                action_type="prefer_consensus",
                target_nodes=[preferred_position],
                reasoning=f"Consensus supports position from {preferred_position}",
                confidence=consensus_data["consensus_strength"],
                supporting_data=consensus_data,
            )

            return ResolutionResult(
                case_id=case.case_id,
                resolution_status=ResolutionStatus.RESOLVED,
                strategy_used=ResolutionStrategy.CONSENSUS_VOTING,
                resolution_actions=[action],
                final_recommendation=f"Accept consensus position from {preferred_position}",
                confidence=consensus_data["consensus_strength"],
                reasoning=action.reasoning,
                resolved_at=datetime.now(),
                resolution_metadata=consensus_data,
            )
        else:
            return self._create_review_result(case, ResolutionStrategy.CONSENSUS_VOTING)

    def _resolve_by_temporal_preference(self, case: ContradictionCase) -> ResolutionResult:
        """Resolve by preferring more recent information."""
        node_timestamps = {}

        for node_id in case.conflicting_nodes:
            timestamp = self._get_node_timestamp(node_id)
            node_timestamps[node_id] = timestamp

        # Find the most recent node
        if any(ts for ts in node_timestamps.values()):
            most_recent = max(
                node_timestamps.items(), key=lambda x: x[1] or datetime.min, default=(None, None)
            )

            if most_recent[1]:
                action = ResolutionAction(
                    action_type="prefer_recent",
                    target_nodes=[most_recent[0]],
                    reasoning=f"Node {most_recent[0]} contains more recent information",
                    confidence=0.7,
                    supporting_data={"timestamps": node_timestamps},
                )

                return ResolutionResult(
                    case_id=case.case_id,
                    resolution_status=ResolutionStatus.RESOLVED,
                    strategy_used=ResolutionStrategy.TEMPORAL_PREFERENCE,
                    resolution_actions=[action],
                    final_recommendation=f"Accept more recent information from {most_recent[0]}",
                    confidence=0.7,
                    reasoning=action.reasoning,
                    resolved_at=datetime.now(),
                    resolution_metadata={"temporal_data": node_timestamps},
                )

        return self._create_review_result(case, ResolutionStrategy.TEMPORAL_PREFERENCE)

    def _resolve_by_confidence(self, case: ContradictionCase) -> ResolutionResult:
        """Resolve based on confidence scores."""
        node_confidences = {}

        for node_id in case.conflicting_nodes:
            confidence = self._get_node_confidence(node_id)
            node_confidences[node_id] = confidence

        # Select highest confidence node
        highest_confidence = max(node_confidences.items(), key=lambda x: x[1])

        if highest_confidence[1] > 0.7:
            action = ResolutionAction(
                action_type="prefer_confident",
                target_nodes=[highest_confidence[0]],
                reasoning=f"Node {highest_confidence[0]} has higher confidence score",
                confidence=highest_confidence[1],
                supporting_data={"confidence_scores": node_confidences},
            )

            return ResolutionResult(
                case_id=case.case_id,
                resolution_status=ResolutionStatus.RESOLVED,
                strategy_used=ResolutionStrategy.CONFIDENCE_WEIGHTED,
                resolution_actions=[action],
                final_recommendation=f"Accept information from {highest_confidence[0]} based on higher confidence",
                confidence=highest_confidence[1],
                reasoning=action.reasoning,
                resolved_at=datetime.now(),
                resolution_metadata={"confidence_data": node_confidences},
            )

        return self._create_review_result(case, ResolutionStrategy.CONFIDENCE_WEIGHTED)

    def _resolve_by_evidence(self, case: ContradictionCase) -> ResolutionResult:
        """Resolve based on supporting evidence."""
        evidence_analysis = self._analyze_supporting_evidence(case)

        if evidence_analysis["clear_winner"]:
            winner = evidence_analysis["preferred_node"]

            action = ResolutionAction(
                action_type="prefer_evidence",
                target_nodes=[winner],
                reasoning=f"Node {winner} has stronger supporting evidence",
                confidence=evidence_analysis["confidence"],
                supporting_data=evidence_analysis,
            )

            return ResolutionResult(
                case_id=case.case_id,
                resolution_status=ResolutionStatus.RESOLVED,
                strategy_used=ResolutionStrategy.EVIDENCE_BASED,
                resolution_actions=[action],
                final_recommendation=f"Accept {winner} based on stronger evidence",
                confidence=evidence_analysis["confidence"],
                reasoning=action.reasoning,
                resolved_at=datetime.now(),
                resolution_metadata=evidence_analysis,
            )

        return self._create_review_result(case, ResolutionStrategy.EVIDENCE_BASED)

    def _resolve_by_hybrid_approach(self, case: ContradictionCase) -> ResolutionResult:
        """Resolve using a combination of strategies."""
        # Try multiple strategies and combine results
        strategies = [
            ResolutionStrategy.SOURCE_AUTHORITY,
            ResolutionStrategy.TEMPORAL_PREFERENCE,
            ResolutionStrategy.CONFIDENCE_WEIGHTED,
        ]

        strategy_results = {}
        for strategy in strategies:
            result = self.resolve_contradiction(case, strategy)
            if result.resolution_status == ResolutionStatus.RESOLVED:
                strategy_results[strategy] = result

        if strategy_results:
            # Combine results with weighted approach
            node_scores = defaultdict(float)

            for strategy, result in strategy_results.items():
                if result.resolution_actions:
                    preferred_node = result.resolution_actions[0].target_nodes[0]
                    weight = self._get_strategy_weight(strategy)
                    node_scores[preferred_node] += result.confidence * weight

            if node_scores:
                best_node = max(node_scores.items(), key=lambda x: x[1])

                action = ResolutionAction(
                    action_type="hybrid_resolution",
                    target_nodes=[best_node[0]],
                    reasoning=f"Hybrid approach favors {best_node[0]} (score: {best_node[1]:.2f})",
                    confidence=min(best_node[1], 1.0),
                    supporting_data={
                        "strategy_results": strategy_results,
                        "node_scores": dict(node_scores),
                    },
                )

                return ResolutionResult(
                    case_id=case.case_id,
                    resolution_status=ResolutionStatus.RESOLVED,
                    strategy_used=ResolutionStrategy.HYBRID_APPROACH,
                    resolution_actions=[action],
                    final_recommendation=f"Accept {best_node[0]} based on hybrid analysis",
                    confidence=min(best_node[1], 1.0),
                    reasoning=action.reasoning,
                    resolved_at=datetime.now(),
                    resolution_metadata={"hybrid_analysis": dict(node_scores)},
                )

        return self._create_review_result(case, ResolutionStrategy.HYBRID_APPROACH)

    def _get_node_authority_score(self, node_id: str) -> float:
        """Get authority score for a node (placeholder implementation)."""
        # This would normally query the source reliability engine
        return 0.7  # Placeholder

    def _get_node_timestamp(self, node_id: str) -> Optional[datetime]:
        """Get timestamp for a node (placeholder implementation)."""
        # This would normally query the node metadata
        return datetime.now()  # Placeholder

    def _get_node_confidence(self, node_id: str) -> float:
        """Get confidence score for a node (placeholder implementation)."""
        # This would normally query the node metadata
        return 0.7  # Placeholder

    def _gather_consensus_data(self, case: ContradictionCase) -> Dict[str, Any]:
        """Gather consensus data from similar nodes."""
        # This would normally search for similar nodes and analyze their positions
        return {
            "consensus_strength": 0.6,
            "majority_position": case.conflicting_nodes[0],
            "supporting_nodes": 5,
            "total_nodes_analyzed": 8,
        }

    def _analyze_supporting_evidence(self, case: ContradictionCase) -> Dict[str, Any]:
        """Analyze supporting evidence for each position."""
        # This would normally analyze evidence quality and quantity
        return {
            "clear_winner": True,
            "preferred_node": case.conflicting_nodes[0],
            "confidence": 0.7,
            "evidence_strength": {node: 0.7 for node in case.conflicting_nodes},
        }

    def _get_strategy_weight(self, strategy: ResolutionStrategy) -> float:
        """Get weight for a strategy in hybrid approach."""
        weights = {
            ResolutionStrategy.SOURCE_AUTHORITY: 0.4,
            ResolutionStrategy.TEMPORAL_PREFERENCE: 0.3,
            ResolutionStrategy.CONFIDENCE_WEIGHTED: 0.3,
        }
        return weights.get(strategy, 0.2)

    def _create_review_result(
        self, case: ContradictionCase, strategy: ResolutionStrategy
    ) -> ResolutionResult:
        """Create result requiring review."""
        return ResolutionResult(
            case_id=case.case_id,
            resolution_status=ResolutionStatus.REQUIRES_REVIEW,
            strategy_used=strategy,
            resolution_actions=[],
            final_recommendation="Contradiction requires human expert review",
            confidence=0.0,
            reasoning=f"Strategy {strategy.value} could not resolve contradiction automatically",
            resolved_at=datetime.now(),
            resolution_metadata={"requires_expert": True},
        )

    def _create_error_result(
        self, case: ContradictionCase, strategy: ResolutionStrategy
    ) -> ResolutionResult:
        """Create error result."""
        return ResolutionResult(
            case_id=case.case_id,
            resolution_status=ResolutionStatus.UNRESOLVED,
            strategy_used=strategy,
            resolution_actions=[],
            final_recommendation="Resolution failed due to error",
            confidence=0.0,
            reasoning="Error occurred during resolution process",
            resolved_at=datetime.now(),
            resolution_metadata={"error": True},
        )


class ContradictionResolver:
    """
    Main Contradiction Resolution Engine.

    Provides automated workflows for detecting and resolving contradictions
    between knowledge nodes using various resolution strategies.
    """

    def __init__(self, query_engine: AdvancedQueryEngine):
        """
        Initialize the Contradiction Resolver.

        Args:
            query_engine: Query engine for accessing knowledge graph
        """
        self.query_engine = query_engine

        # Initialize components
        self.detector = ContradictionDetector(query_engine)
        self.strategy_engine = ResolutionStrategyEngine(query_engine)

        self.logger = logging.getLogger(__name__)

        # Statistics
        self.stats = {
            "contradictions_detected": 0,
            "contradictions_resolved": 0,
            "resolution_success_rate": 0.0,
            "strategy_usage": {strategy.value: 0 for strategy in ResolutionStrategy},
            "avg_resolution_time_ms": 0.0,
            "severity_distribution": {severity.value: 0 for severity in ContradictionSeverity},
        }

    def resolve_contradictions(
        self,
        nodes: List[KnowledgeNode],
        strategy: ResolutionStrategy = ResolutionStrategy.HYBRID_APPROACH,
    ) -> ResolutionReport:
        """
        Detect and resolve contradictions in a set of knowledge nodes.

        Args:
            nodes: Knowledge nodes to analyze
            strategy: Primary resolution strategy to use

        Returns:
            ResolutionReport with comprehensive results
        """
        start_time = time.time()

        try:
            self.logger.info(f"Starting contradiction resolution for {len(nodes)} nodes")

            # Detect contradictions
            contradictions = self.detector.detect_contradictions(nodes)

            if not contradictions:
                self.logger.info("No contradictions detected")
                return ResolutionReport(
                    resolution_results=[],
                    total_contradictions=0,
                    resolved_count=0,
                    unresolved_count=0,
                    strategy_effectiveness={},
                    common_contradiction_types=[],
                    recommendations=[],
                    processing_time_ms=(time.time() - start_time) * 1000,
                )

            # Resolve each contradiction
            resolution_results = []

            for case in contradictions:
                self.logger.debug(f"Resolving contradiction {case.case_id}")

                result = self.strategy_engine.resolve_contradiction(case, strategy)
                resolution_results.append(result)

                # Update statistics
                self._update_case_statistics(case, result)

            # Generate report
            report = self._generate_resolution_report(
                contradictions, resolution_results, start_time
            )

            # Update overall statistics
            self._update_statistics(report)

            self.logger.info(
                f"Contradiction resolution completed: {report.resolved_count}/{report.total_contradictions} resolved"
            )
            return report

        except Exception as e:
            self.logger.error(f"Error in contradiction resolution: {e}")
            return ResolutionReport(
                resolution_results=[],
                total_contradictions=0,
                resolved_count=0,
                unresolved_count=0,
                strategy_effectiveness={},
                common_contradiction_types=[],
                recommendations=[],
                processing_time_ms=(time.time() - start_time) * 1000,
            )

    def resolve_single_contradiction(
        self,
        case: ContradictionCase,
        strategy: ResolutionStrategy = ResolutionStrategy.HYBRID_APPROACH,
    ) -> ResolutionResult:
        """
        Resolve a single contradiction case.

        Args:
            case: Contradiction case to resolve
            strategy: Resolution strategy to use

        Returns:
            ResolutionResult with resolution outcome
        """
        try:
            result = self.strategy_engine.resolve_contradiction(case, strategy)
            self._update_case_statistics(case, result)
            return result

        except Exception as e:
            self.logger.error(f"Error resolving single contradiction {case.case_id}: {e}")
            return self.strategy_engine._create_error_result(case, strategy)

    def _generate_resolution_report(
        self,
        contradictions: List[ContradictionCase],
        results: List[ResolutionResult],
        start_time: float,
    ) -> ResolutionReport:
        """Generate comprehensive resolution report."""
        resolved_count = sum(1 for r in results if r.resolution_status == ResolutionStatus.RESOLVED)
        unresolved_count = len(results) - resolved_count

        # Calculate strategy effectiveness
        strategy_effectiveness = {}
        strategy_counts = Counter(r.strategy_used for r in results)
        strategy_successes = Counter(
            r.strategy_used for r in results if r.resolution_status == ResolutionStatus.RESOLVED
        )

        for strategy in ResolutionStrategy:
            total = strategy_counts.get(strategy, 0)
            successes = strategy_successes.get(strategy, 0)
            strategy_effectiveness[strategy] = successes / total if total > 0 else 0.0

        # Find common contradiction types
        contradiction_types = Counter(case.contradiction_type for case in contradictions)
        common_types = contradiction_types.most_common(5)

        # Generate recommendations
        recommendations = self._generate_recommendations(contradictions, results)

        processing_time = (time.time() - start_time) * 1000

        return ResolutionReport(
            resolution_results=results,
            total_contradictions=len(contradictions),
            resolved_count=resolved_count,
            unresolved_count=unresolved_count,
            strategy_effectiveness=strategy_effectiveness,
            common_contradiction_types=common_types,
            recommendations=recommendations,
            processing_time_ms=processing_time,
        )

    def _generate_recommendations(
        self, contradictions: List[ContradictionCase], results: List[ResolutionResult]
    ) -> List[str]:
        """Generate recommendations based on resolution results."""
        recommendations = []

        unresolved_count = sum(
            1 for r in results if r.resolution_status != ResolutionStatus.RESOLVED
        )
        total_count = len(results)

        if unresolved_count > total_count * 0.3:
            recommendations.append(
                "High number of unresolved contradictions - consider improving source quality"
            )

        # Analyze common contradiction types
        type_counts = Counter(case.contradiction_type for case in contradictions)
        most_common_type = type_counts.most_common(1)

        if most_common_type and most_common_type[0][1] > total_count * 0.4:
            recommendations.append(
                f"Focus on addressing {most_common_type[0][0]} contradictions - most frequent type"
            )

        # Analyze severity distribution
        critical_count = sum(
            1 for case in contradictions if case.severity == ContradictionSeverity.CRITICAL
        )

        if critical_count > 0:
            recommendations.append(
                "Critical contradictions detected - prioritize resolution immediately"
            )

        # Strategy effectiveness recommendations
        strategy_success_rates = {}
        for result in results:
            strategy = result.strategy_used
            if strategy not in strategy_success_rates:
                strategy_success_rates[strategy] = []
            strategy_success_rates[strategy].append(
                result.resolution_status == ResolutionStatus.RESOLVED
            )

        best_strategy = None
        best_rate = 0.0

        for strategy, successes in strategy_success_rates.items():
            rate = sum(successes) / len(successes) if successes else 0.0
            if rate > best_rate:
                best_rate = rate
                best_strategy = strategy

        if best_strategy and best_rate > 0.7:
            recommendations.append(
                f"Consider using {best_strategy.value} strategy more frequently - highest success rate"
            )

        return recommendations[:5]  # Top 5 recommendations

    def _update_case_statistics(self, case: ContradictionCase, result: ResolutionResult):
        """Update statistics for individual case resolution."""
        self.stats["contradictions_detected"] += 1

        if result.resolution_status == ResolutionStatus.RESOLVED:
            self.stats["contradictions_resolved"] += 1

        self.stats["strategy_usage"][result.strategy_used.value] += 1
        self.stats["severity_distribution"][case.severity.value] += 1

    def _update_statistics(self, report: ResolutionReport):
        """Update overall engine statistics."""
        # Update success rate
        if self.stats["contradictions_detected"] > 0:
            self.stats["resolution_success_rate"] = (
                self.stats["contradictions_resolved"] / self.stats["contradictions_detected"]
            )

        # Update average processing time (simple moving average)
        current_avg = self.stats["avg_resolution_time_ms"]
        new_time = report.processing_time_ms

        if current_avg == 0.0:
            self.stats["avg_resolution_time_ms"] = new_time
        else:
            self.stats["avg_resolution_time_ms"] = (current_avg + new_time) / 2

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get contradiction resolver statistics.

        Returns:
            Dictionary with comprehensive statistics
        """
        return {
            "contradiction_resolution": self.stats.copy(),
            "query_engine": self.query_engine.get_statistics(),
        }
