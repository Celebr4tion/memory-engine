"""
Knowledge Quality Enhancement Engine

Main orchestration engine that integrates all quality enhancement components
including quality assessment, cross-validation, source reliability, gap detection,
and contradiction resolution to provide comprehensive quality improvements.
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
from memory_core.query.query_types import QueryRequest, QueryType, QueryResponse
from memory_core.model.knowledge_node import KnowledgeNode

from .quality_assessment import (
    QualityAssessmentEngine,
    QualityAssessment,
    QualityLevel,
    QualityDimension,
)
from .cross_validation import CrossValidationEngine, CrossValidationResult, ValidationStatus
from .source_reliability import SourceReliabilityEngine, SourceReliabilityScore, ReliabilityLevel
from .gap_detection import KnowledgeGapDetector, GapAnalysis, GapSeverity
from .contradiction_resolution import ContradictionResolver, ResolutionReport, ResolutionStatus


class EnhancementPriority(Enum):
    """Priority levels for quality enhancement actions."""

    CRITICAL = "critical"  # Must be addressed immediately
    HIGH = "high"  # Should be addressed soon
    MEDIUM = "medium"  # Should be addressed when possible
    LOW = "low"  # Can be addressed later
    DEFERRED = "deferred"  # Can be deferred indefinitely


class EnhancementAction(Enum):
    """Types of quality enhancement actions."""

    IMPROVE_CONTENT = "improve_content"  # Enhance content quality
    RESOLVE_CONTRADICTION = "resolve_contradiction"  # Resolve conflicts
    FILL_GAP = "fill_gap"  # Fill knowledge gaps
    UPDATE_SOURCE = "update_source"  # Improve source reliability
    VALIDATE_CLAIM = "validate_claim"  # Cross-validate information
    ENHANCE_RELATIONSHIPS = "enhance_relationships"  # Improve connections
    UPDATE_METADATA = "update_metadata"  # Improve metadata quality


@dataclass
class QualityScore:
    """Comprehensive quality score for a knowledge node."""

    node_id: str
    overall_score: float  # 0.0 to 1.0
    content_score: float
    structural_score: float
    temporal_score: float
    reliability_score: float
    validation_score: float
    quality_level: QualityLevel
    confidence: float
    last_assessed: datetime


@dataclass
class EnhancementRecommendation:
    """Recommendation for quality enhancement."""

    node_id: str
    action_type: EnhancementAction
    priority: EnhancementPriority
    description: str
    expected_improvement: float  # Expected score improvement
    confidence: float
    supporting_evidence: List[str]
    estimated_effort: str  # 'low', 'medium', 'high'
    dependencies: List[str]  # Other nodes that might be affected


@dataclass
class QualityEnhancementReport:
    """Comprehensive quality enhancement report."""

    total_nodes_analyzed: int
    quality_scores: List[QualityScore]
    enhancement_recommendations: List[EnhancementRecommendation]
    quality_distribution: Dict[QualityLevel, int]
    critical_issues: List[str]
    gap_analysis: Optional[GapAnalysis]
    contradiction_report: Optional[ResolutionReport]
    overall_quality_trend: str  # 'improving', 'declining', 'stable'
    processing_time_ms: float
    next_assessment_recommended: datetime


class QualityRanker:
    """Provides quality-based ranking for query results."""

    def __init__(self, quality_engine: "KnowledgeQualityEnhancementEngine"):
        self.quality_engine = quality_engine
        self.logger = logging.getLogger(__name__)

        # Ranking weights for different quality dimensions
        self.ranking_weights = {
            "content_quality": 0.25,
            "reliability": 0.30,
            "validation": 0.20,
            "temporal": 0.15,
            "structural": 0.10,
        }

    def rank_nodes_by_quality(
        self, nodes: List[KnowledgeNode], query_context: Optional[str] = None
    ) -> List[Tuple[KnowledgeNode, float]]:
        """
        Rank nodes by their quality scores.

        Args:
            nodes: List of nodes to rank
            query_context: Optional context for relevance weighting

        Returns:
            List of (node, quality_score) tuples sorted by quality
        """
        try:
            ranked_nodes = []

            for node in nodes:
                quality_score = self._calculate_node_quality_score(node, query_context)
                ranked_nodes.append((node, quality_score))

            # Sort by quality score (descending)
            ranked_nodes.sort(key=lambda x: x[1], reverse=True)

            return ranked_nodes

        except Exception as e:
            self.logger.error(f"Error ranking nodes by quality: {e}")
            return [(node, 0.5) for node in nodes]  # Fallback to neutral scores

    def enhance_query_response(self, response: QueryResponse) -> QueryResponse:
        """
        Enhance query response with quality-based ranking.

        Args:
            response: Original query response

        Returns:
            Enhanced query response with quality ranking
        """
        try:
            if not response.results:
                return response

            # Convert results to nodes for ranking
            nodes = []
            result_map = {}

            for result in response.results:
                # Create temporary node from result data using correct constructor
                metadata = result.metadata or {}
                node = KnowledgeNode(
                    content=result.content,
                    source=metadata.get("source", "unknown"),
                    node_id=result.node_id,
                    rating_richness=metadata.get("rating_richness", 0.5),
                    rating_truthfulness=metadata.get("rating_truthfulness", 0.5),
                    rating_stability=metadata.get("rating_stability", 0.5),
                )

                # Add metadata and node_type as attributes
                node.metadata = metadata
                node.node_type = result.node_type or "document"

                nodes.append(node)
                result_map[result.node_id] = result

            # Rank nodes by quality
            ranked_nodes = self.rank_nodes_by_quality(nodes, response.query)

            # Update results with quality ranking
            enhanced_results = []
            for node, quality_score in ranked_nodes:
                original_result = result_map[node.node_id]

                # Add quality information to metadata
                enhanced_metadata = original_result.metadata or {}
                enhanced_metadata["quality_score"] = quality_score
                enhanced_metadata["quality_ranked"] = True

                # Update the result
                original_result.metadata = enhanced_metadata
                enhanced_results.append(original_result)

            # Create enhanced response
            enhanced_response = QueryResponse(
                results=enhanced_results,
                total_count=response.total_count,
                returned_count=response.returned_count,
                execution_time_ms=response.execution_time_ms,
            )

            # Copy other attributes
            enhanced_response.query_id = response.query_id
            enhanced_response.from_cache = response.from_cache
            enhanced_response.has_more = response.has_more
            enhanced_response.next_offset = response.next_offset
            enhanced_response.query = response.query

            # Set up metadata - ensure it's always a dictionary
            enhanced_response.metadata = getattr(response, "metadata", {}) or {}

            # Add quality enhancement metadata
            enhanced_response.metadata["quality_enhanced"] = True
            enhanced_response.metadata["quality_ranking_applied"] = True

            return enhanced_response

        except Exception as e:
            self.logger.error(f"Error enhancing query response: {e}")
            return response  # Return original response on error

    def _calculate_node_quality_score(
        self, node: KnowledgeNode, query_context: Optional[str] = None
    ) -> float:
        """Calculate comprehensive quality score for a node."""
        try:
            # Get or calculate quality metrics
            quality_assessment = self.quality_engine._get_cached_quality_assessment(node.node_id)

            if not quality_assessment:
                # Quick assessment if not cached
                quality_assessment = self.quality_engine.quality_assessor.assess_node_quality(node)

            # Get reliability score
            reliability_score = self.quality_engine._get_cached_reliability_score(node.node_id)

            if not reliability_score:
                reliability_score = (
                    self.quality_engine.source_reliability.assess_source_reliability(node)
                )

            # Calculate component scores
            scores = {
                "content_quality": quality_assessment.metrics.get(
                    QualityDimension.CONTENT_QUALITY, type("", (), {"score": 0.5})
                ).score,
                "reliability": reliability_score.overall_score,
                "validation": self._get_validation_score(node.node_id),
                "temporal": quality_assessment.metrics.get(
                    QualityDimension.TEMPORAL_QUALITY, type("", (), {"score": 0.5})
                ).score,
                "structural": quality_assessment.metrics.get(
                    QualityDimension.STRUCTURAL_QUALITY, type("", (), {"score": 0.5})
                ).score,
            }

            # Calculate weighted score
            weighted_score = sum(
                scores[dimension] * weight for dimension, weight in self.ranking_weights.items()
            )

            # Apply context bonus if relevant
            if query_context:
                context_bonus = self._calculate_context_relevance_bonus(node, query_context)
                weighted_score = min(1.0, weighted_score + context_bonus)

            return weighted_score

        except Exception as e:
            self.logger.error(f"Error calculating quality score for node {node.node_id}: {e}")
            return 0.5  # Neutral score on error

    def _get_validation_score(self, node_id: str) -> float:
        """Get validation score for a node."""
        # This would normally check cross-validation results
        # Placeholder implementation
        return 0.7

    def _calculate_context_relevance_bonus(self, node: KnowledgeNode, query_context: str) -> float:
        """Calculate bonus score based on query context relevance."""
        # Simple keyword matching for context relevance
        query_words = set(query_context.lower().split())
        content_words = set(node.content.lower().split())

        overlap = len(query_words & content_words)
        total_query_words = len(query_words)

        if total_query_words == 0:
            return 0.0

        relevance_ratio = overlap / total_query_words
        return min(0.1, relevance_ratio * 0.2)  # Max 10% bonus


class KnowledgeQualityEnhancementEngine:
    """
    Main Knowledge Quality Enhancement Engine.

    Orchestrates all quality enhancement components to provide comprehensive
    quality assessment, improvement recommendations, and automated enhancements.
    """

    def __init__(self, query_engine: AdvancedQueryEngine):
        """
        Initialize the Knowledge Quality Enhancement Engine.

        Args:
            query_engine: Query engine for accessing knowledge graph
        """
        self.query_engine = query_engine

        # Initialize component engines
        self.quality_assessor = QualityAssessmentEngine(query_engine)
        self.cross_validator = CrossValidationEngine(query_engine)
        self.source_reliability = SourceReliabilityEngine(query_engine)
        self.gap_detector = KnowledgeGapDetector(query_engine)
        self.contradiction_resolver = ContradictionResolver(query_engine)

        # Initialize quality ranker
        self.quality_ranker = QualityRanker(self)

        self.logger = logging.getLogger(__name__)

        # Caches for performance
        self.quality_cache = {}
        self.reliability_cache = {}
        self.validation_cache = {}

        # Statistics
        self.stats = {
            "enhancements_performed": 0,
            "nodes_enhanced": 0,
            "quality_improvements": 0,
            "contradictions_resolved": 0,
            "gaps_filled": 0,
            "avg_quality_improvement": 0.0,
            "processing_time_total_ms": 0.0,
        }

    def enhance_knowledge_quality(
        self, nodes: List[KnowledgeNode], perform_enhancements: bool = False
    ) -> QualityEnhancementReport:
        """
        Perform comprehensive quality enhancement analysis and optionally apply improvements.

        Args:
            nodes: Knowledge nodes to enhance
            perform_enhancements: Whether to apply recommended enhancements

        Returns:
            QualityEnhancementReport with analysis and recommendations
        """
        start_time = time.time()

        try:
            self.logger.info(f"Starting quality enhancement for {len(nodes)} nodes")

            # Assess quality for all nodes
            quality_scores = self._assess_node_qualities(nodes)

            # Perform cross-validation
            validation_results = self._perform_cross_validation(nodes)

            # Assess source reliability
            reliability_scores = self._assess_source_reliability(nodes)

            # Detect knowledge gaps
            gap_analysis = self._detect_knowledge_gaps(nodes)

            # Detect and resolve contradictions
            contradiction_report = self._resolve_contradictions(nodes)

            # Generate enhancement recommendations
            recommendations = self._generate_enhancement_recommendations(
                nodes, quality_scores, validation_results, reliability_scores, gap_analysis
            )

            # Apply enhancements if requested
            if perform_enhancements:
                self._apply_enhancements(recommendations)

            # Calculate quality distribution
            quality_distribution = self._calculate_quality_distribution(quality_scores)

            # Identify critical issues
            critical_issues = self._identify_critical_issues(
                quality_scores, validation_results, contradiction_report
            )

            # Assess overall quality trend
            quality_trend = self._assess_quality_trend(quality_scores)

            processing_time = (time.time() - start_time) * 1000

            report = QualityEnhancementReport(
                total_nodes_analyzed=len(nodes),
                quality_scores=quality_scores,
                enhancement_recommendations=recommendations,
                quality_distribution=quality_distribution,
                critical_issues=critical_issues,
                gap_analysis=gap_analysis,
                contradiction_report=contradiction_report,
                overall_quality_trend=quality_trend,
                processing_time_ms=processing_time,
                next_assessment_recommended=self._calculate_next_assessment_time(quality_scores),
            )

            # Update statistics
            self._update_statistics(report, processing_time)

            self.logger.info(f"Quality enhancement completed in {processing_time:.1f}ms")
            return report

        except Exception as e:
            self.logger.error(f"Error in quality enhancement: {e}")
            return self._create_error_report(nodes, start_time)

    def get_quality_score(self, node: KnowledgeNode) -> QualityScore:
        """
        Get comprehensive quality score for a single node.

        Args:
            node: Knowledge node to assess

        Returns:
            QualityScore with comprehensive metrics
        """
        try:
            # Assess quality
            quality_assessment = self.quality_assessor.assess_node_quality(node)

            # Assess reliability
            reliability_score = self.source_reliability.assess_source_reliability(node)

            # Get validation score
            validation_score = self._get_node_validation_score(node)

            # Calculate component scores
            content_score = quality_assessment.metrics.get(
                QualityDimension.CONTENT_QUALITY, type("", (), {"score": 0.5})
            ).score
            structural_score = quality_assessment.metrics.get(
                QualityDimension.STRUCTURAL_QUALITY, type("", (), {"score": 0.5})
            ).score
            temporal_score = quality_assessment.metrics.get(
                QualityDimension.TEMPORAL_QUALITY, type("", (), {"score": 0.5})
            ).score

            # Calculate overall score
            overall_score = np.mean(
                [
                    content_score,
                    structural_score,
                    temporal_score,
                    reliability_score.overall_score,
                    validation_score,
                ]
            )

            return QualityScore(
                node_id=node.node_id,
                overall_score=overall_score,
                content_score=content_score,
                structural_score=structural_score,
                temporal_score=temporal_score,
                reliability_score=reliability_score.overall_score,
                validation_score=validation_score,
                quality_level=self._score_to_quality_level(overall_score),
                confidence=quality_assessment.assessment_confidence,
                last_assessed=datetime.now(),
            )

        except Exception as e:
            self.logger.error(f"Error getting quality score for node {node.node_id}: {e}")
            return self._create_error_quality_score(node.node_id)

    def enhance_query_with_quality_ranking(self, request: QueryRequest) -> QueryResponse:
        """
        Execute query with quality-based result ranking.

        Args:
            request: Query request

        Returns:
            Query response with quality-enhanced ranking
        """
        try:
            # Execute original query
            response = self.query_engine.query(request)

            # Enhance with quality ranking
            enhanced_response = self.quality_ranker.enhance_query_response(response)

            return enhanced_response

        except Exception as e:
            self.logger.error(f"Error enhancing query with quality ranking: {e}")
            # Fall back to original query if enhancement fails
            return self.query_engine.query(request)

    def _assess_node_qualities(self, nodes: List[KnowledgeNode]) -> List[QualityScore]:
        """Assess quality for all nodes."""
        quality_scores = []

        for node in nodes:
            score = self.get_quality_score(node)
            quality_scores.append(score)

            # Cache the result
            self.quality_cache[node.node_id] = score

        return quality_scores

    def _perform_cross_validation(self, nodes: List[KnowledgeNode]) -> List[CrossValidationResult]:
        """Perform cross-validation for all nodes."""
        all_results = []

        for node in nodes:
            results = self.cross_validator.validate_node(node)
            all_results.extend(results)

        return all_results

    def _assess_source_reliability(
        self, nodes: List[KnowledgeNode]
    ) -> List[SourceReliabilityScore]:
        """Assess source reliability for all nodes."""
        reliability_scores = []

        for node in nodes:
            score = self.source_reliability.assess_source_reliability(node)
            reliability_scores.append(score)

            # Cache the result
            self.reliability_cache[node.node_id] = score

        return reliability_scores

    def _detect_knowledge_gaps(self, nodes: List[KnowledgeNode]) -> GapAnalysis:
        """Detect knowledge gaps in the node set."""
        return self.gap_detector.detect_knowledge_gaps(nodes)

    def _resolve_contradictions(self, nodes: List[KnowledgeNode]) -> ResolutionReport:
        """Detect and resolve contradictions."""
        return self.contradiction_resolver.resolve_contradictions(nodes)

    def _generate_enhancement_recommendations(
        self,
        nodes: List[KnowledgeNode],
        quality_scores: List[QualityScore],
        validation_results: List[CrossValidationResult],
        reliability_scores: List[SourceReliabilityScore],
        gap_analysis: GapAnalysis,
    ) -> List[EnhancementRecommendation]:
        """Generate comprehensive enhancement recommendations."""
        recommendations = []

        # Content quality recommendations
        for score in quality_scores:
            if score.content_score < 0.6:
                recommendations.append(
                    EnhancementRecommendation(
                        node_id=score.node_id,
                        action_type=EnhancementAction.IMPROVE_CONTENT,
                        priority=(
                            EnhancementPriority.HIGH
                            if score.content_score < 0.4
                            else EnhancementPriority.MEDIUM
                        ),
                        description=f"Improve content quality (current score: {score.content_score:.2f})",
                        expected_improvement=0.6 - score.content_score,
                        confidence=0.8,
                        supporting_evidence=[
                            f"Content score below threshold: {score.content_score:.2f}"
                        ],
                        estimated_effort="medium",
                        dependencies=[],
                    )
                )

        # Validation recommendations
        for result in validation_results:
            if result.validation_status == ValidationStatus.CONFLICTED:
                recommendations.append(
                    EnhancementRecommendation(
                        node_id=result.target_node_id,
                        action_type=EnhancementAction.RESOLVE_CONTRADICTION,
                        priority=EnhancementPriority.HIGH,
                        description="Resolve validation conflicts",
                        expected_improvement=0.3,
                        confidence=0.7,
                        supporting_evidence=result.key_conflicts,
                        estimated_effort="high",
                        dependencies=[],
                    )
                )

        # Gap filling recommendations
        for gap in gap_analysis.critical_gaps:
            recommendations.append(
                EnhancementRecommendation(
                    node_id=gap.related_nodes[0] if gap.related_nodes else "general",
                    action_type=EnhancementAction.FILL_GAP,
                    priority=(
                        EnhancementPriority.HIGH
                        if gap.severity == GapSeverity.CRITICAL
                        else EnhancementPriority.MEDIUM
                    ),
                    description=f"Fill knowledge gap: {gap.description}",
                    expected_improvement=0.4,
                    confidence=gap.confidence,
                    supporting_evidence=[gap.description],
                    estimated_effort="high",
                    dependencies=gap.related_nodes,
                )
            )

        # Source reliability recommendations
        for score in reliability_scores:
            if score.reliability_level in [
                ReliabilityLevel.QUESTIONABLE,
                ReliabilityLevel.UNRELIABLE,
            ]:
                recommendations.append(
                    EnhancementRecommendation(
                        node_id=score.source_identifier,
                        action_type=EnhancementAction.UPDATE_SOURCE,
                        priority=EnhancementPriority.MEDIUM,
                        description=f"Improve source reliability ({score.reliability_level.value})",
                        expected_improvement=0.3,
                        confidence=score.assessment_confidence,
                        supporting_evidence=score.improvement_suggestions,
                        estimated_effort="medium",
                        dependencies=[],
                    )
                )

        # Sort by priority and expected improvement
        recommendations.sort(key=lambda r: (r.priority.value, -r.expected_improvement))

        return recommendations[:20]  # Top 20 recommendations

    def _apply_enhancements(self, recommendations: List[EnhancementRecommendation]):
        """Apply enhancement recommendations (placeholder implementation)."""
        # This would implement actual enhancement actions
        # For now, just log the actions that would be taken

        applied_count = 0
        for rec in recommendations:
            if rec.priority in [EnhancementPriority.CRITICAL, EnhancementPriority.HIGH]:
                self.logger.info(f"Would apply enhancement: {rec.description} for {rec.node_id}")
                applied_count += 1

        self.stats["enhancements_performed"] += applied_count

    def _calculate_quality_distribution(
        self, quality_scores: List[QualityScore]
    ) -> Dict[QualityLevel, int]:
        """Calculate distribution of quality levels."""
        distribution = {level: 0 for level in QualityLevel}

        for score in quality_scores:
            distribution[score.quality_level] += 1

        return distribution

    def _identify_critical_issues(
        self,
        quality_scores: List[QualityScore],
        validation_results: List[CrossValidationResult],
        contradiction_report: ResolutionReport,
    ) -> List[str]:
        """Identify critical quality issues."""
        issues = []

        # Critical quality scores
        critical_quality_count = sum(
            1 for score in quality_scores if score.quality_level == QualityLevel.CRITICAL
        )

        if critical_quality_count > 0:
            issues.append(f"{critical_quality_count} nodes have critical quality issues")

        # Validation conflicts
        conflict_count = sum(
            1
            for result in validation_results
            if result.validation_status == ValidationStatus.CONFLICTED
        )

        if conflict_count > 0:
            issues.append(f"{conflict_count} validation conflicts detected")

        # Unresolved contradictions
        if contradiction_report and contradiction_report.unresolved_count > 0:
            issues.append(f"{contradiction_report.unresolved_count} unresolved contradictions")

        return issues

    def _assess_quality_trend(self, quality_scores: List[QualityScore]) -> str:
        """Assess overall quality trend (placeholder implementation)."""
        # This would compare with historical data
        # For now, return based on current quality distribution

        excellent_count = sum(
            1 for score in quality_scores if score.quality_level == QualityLevel.EXCELLENT
        )
        total_count = len(quality_scores)

        if excellent_count / total_count > 0.5:
            return "stable"
        elif excellent_count / total_count > 0.2:
            return "improving"
        else:
            return "declining"

    def _calculate_next_assessment_time(self, quality_scores: List[QualityScore]) -> datetime:
        """Calculate when next assessment should be performed."""
        # Base assessment frequency on quality levels
        critical_count = sum(
            1 for score in quality_scores if score.quality_level == QualityLevel.CRITICAL
        )

        if critical_count > 0:
            # More frequent assessment for critical issues
            next_assessment = datetime.now().replace(hour=datetime.now().hour + 24)  # 24 hours
        else:
            # Standard weekly assessment
            next_assessment = datetime.now().replace(day=datetime.now().day + 7)  # 7 days

        return next_assessment

    def _get_node_validation_score(self, node: KnowledgeNode) -> float:
        """Get validation score for a node."""
        # Check cache first
        if node.node_id in self.validation_cache:
            return self.validation_cache[node.node_id]

        # Perform quick validation
        validation_results = self.cross_validator.validate_node(node)

        if not validation_results:
            score = 0.5  # Neutral score if no validation performed
        else:
            # Calculate average consensus score
            consensus_scores = [result.consensus_score for result in validation_results]
            score = np.mean(consensus_scores) if consensus_scores else 0.5

        # Cache the result
        self.validation_cache[node.node_id] = score

        return score

    def _score_to_quality_level(self, score: float) -> QualityLevel:
        """Convert numeric score to quality level."""
        if score >= 0.9:
            return QualityLevel.EXCELLENT
        elif score >= 0.7:
            return QualityLevel.GOOD
        elif score >= 0.5:
            return QualityLevel.FAIR
        elif score >= 0.3:
            return QualityLevel.POOR
        else:
            return QualityLevel.CRITICAL

    def _get_cached_quality_assessment(self, node_id: str) -> Optional[Any]:
        """Get cached quality assessment."""
        return self.quality_cache.get(node_id)

    def _get_cached_reliability_score(self, node_id: str) -> Optional[SourceReliabilityScore]:
        """Get cached reliability score."""
        return self.reliability_cache.get(node_id)

    def _create_error_quality_score(self, node_id: str) -> QualityScore:
        """Create error quality score."""
        return QualityScore(
            node_id=node_id,
            overall_score=0.0,
            content_score=0.0,
            structural_score=0.0,
            temporal_score=0.0,
            reliability_score=0.0,
            validation_score=0.0,
            quality_level=QualityLevel.CRITICAL,
            confidence=0.0,
            last_assessed=datetime.now(),
        )

    def _create_error_report(
        self, nodes: List[KnowledgeNode], start_time: float
    ) -> QualityEnhancementReport:
        """Create error report when enhancement fails."""
        return QualityEnhancementReport(
            total_nodes_analyzed=len(nodes),
            quality_scores=[],
            enhancement_recommendations=[],
            quality_distribution={level: 0 for level in QualityLevel},
            critical_issues=["Quality enhancement failed due to error"],
            gap_analysis=None,
            contradiction_report=None,
            overall_quality_trend="unknown",
            processing_time_ms=(time.time() - start_time) * 1000,
            next_assessment_recommended=datetime.now(),
        )

    def _update_statistics(self, report: QualityEnhancementReport, processing_time: float):
        """Update engine statistics."""
        self.stats["nodes_enhanced"] += report.total_nodes_analyzed
        self.stats["processing_time_total_ms"] += processing_time

        # Update quality improvements
        high_quality_count = report.quality_distribution.get(
            QualityLevel.EXCELLENT, 0
        ) + report.quality_distribution.get(QualityLevel.GOOD, 0)

        if report.total_nodes_analyzed > 0:
            quality_ratio = high_quality_count / report.total_nodes_analyzed
            self.stats["avg_quality_improvement"] = quality_ratio

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics from all quality enhancement components.

        Returns:
            Dictionary with comprehensive statistics
        """
        return {
            "quality_enhancement": self.stats.copy(),
            "quality_assessment": self.quality_assessor.get_statistics(),
            "cross_validation": self.cross_validator.get_statistics(),
            "source_reliability": self.source_reliability.get_statistics(),
            "gap_detection": self.gap_detector.get_statistics(),
            "contradiction_resolution": self.contradiction_resolver.get_statistics(),
            "query_engine": self.query_engine.get_statistics(),
        }
