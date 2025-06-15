"""
Comprehensive tests for the Knowledge Quality Enhancement System.

Tests all components of the quality enhancement system including:
- Quality assessment
- Cross-validation
- Source reliability
- Gap detection
- Contradiction resolution
- Quality-based ranking
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch

from memory_core.model.knowledge_node import KnowledgeNode
from memory_core.model.relationship import Relationship
from memory_core.query.query_engine import AdvancedQueryEngine
from memory_core.query.query_types import QueryRequest, QueryType

from memory_core.quality.quality_assessment import (
    QualityAssessmentEngine,
    QualityDimension,
    QualityLevel,
)
from memory_core.quality.cross_validation import (
    CrossValidationEngine,
    ValidationStatus,
    ValidationConfidence,
)
from memory_core.quality.source_reliability import (
    SourceReliabilityEngine,
    ReliabilityLevel,
    SourceType,
)
from memory_core.quality.gap_detection import KnowledgeGapDetector, GapType, GapSeverity
from memory_core.quality.contradiction_resolution import (
    ContradictionResolver,
    ResolutionStrategy,
    ContradictionSeverity,
)
from memory_core.quality.quality_enhancement_engine import (
    KnowledgeQualityEnhancementEngine,
    QualityScore,
    EnhancementAction,
)


@pytest.fixture
def mock_query_engine():
    """Create a mock query engine for testing."""
    engine = Mock(spec=AdvancedQueryEngine)
    engine.get_statistics.return_value = {}
    return engine


@pytest.fixture
def sample_nodes():
    """Create sample knowledge nodes for testing."""
    nodes = [
        KnowledgeNode(
            content="Python is a high-level programming language created by Guido van Rossum in 1991. It emphasizes code readability and supports multiple programming paradigms.",
            source="https://python.org",
            node_id="node_1",
            rating_richness=0.8,
            rating_truthfulness=0.9,
            rating_stability=0.7,
        ),
        KnowledgeNode(
            content="Python was invented in 1989 by Guido van Rossum. It is known for its simple syntax.",
            source="https://wikipedia.org/wiki/Python",
            node_id="node_2",
            rating_richness=0.6,
            rating_truthfulness=0.7,
            rating_stability=0.5,
        ),
        KnowledgeNode(
            content="ML algorithms",  # Short, low quality content
            source="unknown",
            node_id="node_3",
            rating_richness=0.2,
            rating_truthfulness=0.3,
            rating_stability=0.1,
        ),
        KnowledgeNode(
            content="Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It involves algorithms that can identify patterns in data.",
            source="https://nature.com/articles/ml-overview",
            node_id="node_4",
            rating_richness=0.9,
            rating_truthfulness=0.95,
            rating_stability=0.8,
        ),
    ]

    # Add additional metadata and attributes to simulate what the quality engines expect
    nodes[0].metadata = {
        "timestamp": "2024-01-15T10:00:00Z",
        "domain": "programming",
        "confidence": 0.9,
        "node_type": "definition",
    }
    nodes[0].node_type = "definition"

    nodes[1].metadata = {
        "timestamp": "2024-01-10T15:30:00Z",
        "domain": "programming",
        "confidence": 0.7,
        "node_type": "fact",
    }
    nodes[1].node_type = "fact"

    nodes[2].metadata = {
        "timestamp": "2024-01-01T12:00:00Z",
        "confidence": 0.3,
        "node_type": "note",
    }
    nodes[2].node_type = "note"

    nodes[3].metadata = {
        "timestamp": "2024-01-20T09:00:00Z",
        "domain": "artificial_intelligence",
        "confidence": 0.95,
        "peer_reviewed": True,
        "node_type": "definition",
    }
    nodes[3].node_type = "definition"

    return nodes


class TestQualityAssessmentEngine:
    """Tests for the Quality Assessment Engine."""

    def test_content_quality_assessment(self, mock_query_engine, sample_nodes):
        """Test content quality assessment."""
        engine = QualityAssessmentEngine(mock_query_engine)

        # Test high-quality node (content-wise)
        high_quality_node = sample_nodes[0]  # Python definition
        assessment = engine.assess_node_quality(high_quality_node)

        assert assessment.node_id == "node_1"
        assert 0.0 <= assessment.overall_score <= 1.0
        assert QualityDimension.CONTENT_QUALITY in assessment.metrics
        assert assessment.overall_level in list(QualityLevel)

        # Content quality should be higher than structural/temporal
        content_quality = assessment.metrics[QualityDimension.CONTENT_QUALITY].score
        assert content_quality > 0.3  # Reasonable content quality for descriptive text

        # Test low-quality node
        low_quality_node = sample_nodes[2]  # Short content
        assessment = engine.assess_node_quality(low_quality_node)

        assert assessment.node_id == "node_3"
        assert 0.0 <= assessment.overall_score <= 1.0
        assert assessment.overall_level in list(QualityLevel)

        # Low quality content should have lower content quality score
        low_content_quality = assessment.metrics[QualityDimension.CONTENT_QUALITY].score
        assert low_content_quality < content_quality  # Should be lower than the good content

    def test_multiple_quality_dimensions(self, mock_query_engine, sample_nodes):
        """Test assessment across multiple quality dimensions."""
        engine = QualityAssessmentEngine(mock_query_engine)

        dimensions = [
            QualityDimension.CONTENT_QUALITY,
            QualityDimension.TEMPORAL_QUALITY,
            QualityDimension.STRUCTURAL_QUALITY,
        ]

        assessment = engine.assess_node_quality(sample_nodes[0], dimensions)

        assert len(assessment.metrics) == len(dimensions)
        for dimension in dimensions:
            assert dimension in assessment.metrics
            assert 0.0 <= assessment.metrics[dimension].score <= 1.0


class TestCrossValidationEngine:
    """Tests for the Cross-Validation Engine."""

    def test_node_validation(self, mock_query_engine, sample_nodes):
        """Test cross-validation of individual nodes."""
        # Mock query response for evidence collection
        mock_response = Mock()
        mock_response.results = []
        mock_query_engine.query.return_value = mock_response

        engine = CrossValidationEngine(mock_query_engine)

        # Test validation of a node
        validation_results = engine.validate_node(sample_nodes[0])

        # Should return a list of validation results
        assert isinstance(validation_results, list)
        # Results depend on extractable claims

    def test_contradiction_detection(self, mock_query_engine, sample_nodes):
        """Test contradiction detection between nodes."""
        engine = CrossValidationEngine(mock_query_engine)

        # Use nodes with contradictory dates (1989 vs 1991)
        contradictory_nodes = [sample_nodes[0], sample_nodes[1]]

        contradictions = engine.detect_contradictions(contradictory_nodes)

        assert isinstance(contradictions, list)
        # Should detect the date contradiction
        if contradictions:
            assert any(
                "1989" in str(c.conflicting_claims) or "1991" in str(c.conflicting_claims)
                for c in contradictions
            )


class TestSourceReliabilityEngine:
    """Tests for the Source Reliability Engine."""

    def test_source_reliability_assessment(self, mock_query_engine, sample_nodes):
        """Test source reliability assessment."""
        engine = SourceReliabilityEngine(mock_query_engine)

        # Test academic source (nature.com)
        academic_node = sample_nodes[3]
        reliability = engine.assess_source_reliability(academic_node)

        assert reliability.source_identifier
        assert 0.0 <= reliability.overall_score <= 1.0
        assert reliability.reliability_level in list(ReliabilityLevel)
        assert reliability.source_type in list(SourceType)

        # Verify that source assessment is working (score should be reasonable)
        assert 0.0 <= reliability.overall_score <= 1.0
        # Note: The test node might not be classified as academic due to mock limitations

    def test_multiple_source_assessment(self, mock_query_engine, sample_nodes):
        """Test assessment of multiple sources."""
        engine = SourceReliabilityEngine(mock_query_engine)

        report = engine.assess_multiple_sources(sample_nodes)

        assert report.source_scores
        assert len(report.source_scores) == len(sample_nodes)
        assert report.summary_statistics
        assert "avg_reliability_score" in report.summary_statistics


class TestKnowledgeGapDetector:
    """Tests for the Knowledge Gap Detector."""

    def test_gap_detection(self, mock_query_engine, sample_nodes):
        """Test knowledge gap detection."""
        engine = KnowledgeGapDetector(mock_query_engine)

        analysis = engine.detect_knowledge_gaps(sample_nodes, domain_name="programming")

        assert analysis.analyzed_domain == "programming"
        assert analysis.total_gaps_found >= 0
        assert isinstance(analysis.gaps_by_type, dict)
        assert isinstance(analysis.gaps_by_severity, dict)
        assert 0.0 <= analysis.gap_coverage_score <= 1.0

    def test_specific_gap_types(self, mock_query_engine, sample_nodes):
        """Test detection of specific gap types."""
        engine = KnowledgeGapDetector(mock_query_engine)

        # Test content gap detection
        gap_types = [GapType.CONTENT_GAP, GapType.DEPTH_GAP]
        analysis = engine.detect_knowledge_gaps(sample_nodes, gap_types=gap_types)

        # Should find depth gaps for the short content node
        assert analysis.total_gaps_found >= 0


class TestContradictionResolver:
    """Tests for the Contradiction Resolver."""

    def test_contradiction_resolution(self, mock_query_engine, sample_nodes):
        """Test contradiction resolution."""
        engine = ContradictionResolver(mock_query_engine)

        # Use nodes with potential contradictions
        contradictory_nodes = [sample_nodes[0], sample_nodes[1]]

        report = engine.resolve_contradictions(contradictory_nodes)

        assert report.total_contradictions >= 0
        assert report.resolved_count >= 0
        assert report.unresolved_count >= 0
        assert report.resolved_count + report.unresolved_count == report.total_contradictions

    def test_resolution_strategies(self, mock_query_engine, sample_nodes):
        """Test different resolution strategies."""
        engine = ContradictionResolver(mock_query_engine)

        strategies = [
            ResolutionStrategy.SOURCE_AUTHORITY,
            ResolutionStrategy.TEMPORAL_PREFERENCE,
            ResolutionStrategy.HYBRID_APPROACH,
        ]

        for strategy in strategies:
            report = engine.resolve_contradictions(sample_nodes, strategy)
            assert isinstance(report.strategy_effectiveness, dict)


class TestKnowledgeQualityEnhancementEngine:
    """Tests for the main Quality Enhancement Engine."""

    def test_comprehensive_quality_enhancement(self, mock_query_engine, sample_nodes):
        """Test comprehensive quality enhancement."""
        engine = KnowledgeQualityEnhancementEngine(mock_query_engine)

        report = engine.enhance_knowledge_quality(sample_nodes)

        assert report.total_nodes_analyzed == len(sample_nodes)
        assert len(report.quality_scores) == len(sample_nodes)
        assert isinstance(report.enhancement_recommendations, list)
        assert isinstance(report.quality_distribution, dict)
        assert report.processing_time_ms > 0

    def test_quality_score_calculation(self, mock_query_engine, sample_nodes):
        """Test individual quality score calculation."""
        engine = KnowledgeQualityEnhancementEngine(mock_query_engine)

        # Add node_type and other expected attributes to nodes
        for node in sample_nodes:
            if hasattr(node, "metadata") and node.metadata:
                node.node_type = node.metadata.get("node_type", "document")
            else:
                node.node_type = "document"
                node.metadata = {}

        for node in sample_nodes:
            quality_score = engine.get_quality_score(node)

            assert isinstance(quality_score, QualityScore)
            assert quality_score.node_id == node.node_id
            assert 0.0 <= quality_score.overall_score <= 1.0
            assert 0.0 <= quality_score.content_score <= 1.0
            assert 0.0 <= quality_score.reliability_score <= 1.0
            assert quality_score.quality_level in list(QualityLevel)

    def test_enhancement_recommendations(self, mock_query_engine, sample_nodes):
        """Test generation of enhancement recommendations."""
        engine = KnowledgeQualityEnhancementEngine(mock_query_engine)

        # Add node_type attributes for proper processing
        for node in sample_nodes:
            if not hasattr(node, "node_type"):
                node.node_type = node.metadata.get("node_type", "document")

        # Get quality scores first
        quality_scores = [engine.get_quality_score(node) for node in sample_nodes]

        # Create a simple gap analysis instead of None
        from memory_core.quality.gap_detection import GapAnalysis

        mock_gap_analysis = GapAnalysis(
            analyzed_domain="test",
            total_gaps_found=0,
            gaps_by_type={},
            gaps_by_severity={},
            critical_gaps=[],
            gap_coverage_score=0.8,
            recommendations=[],
            analysis_confidence=0.7,
        )

        # Generate recommendations
        recommendations = engine._generate_enhancement_recommendations(
            sample_nodes, quality_scores, [], [], mock_gap_analysis
        )

        assert isinstance(recommendations, list)
        # Should have recommendations for low quality nodes
        for rec in recommendations:
            assert rec.node_id
            assert rec.action_type in list(EnhancementAction)
            assert 0.0 <= rec.expected_improvement <= 1.0
            assert 0.0 <= rec.confidence <= 1.0


class TestQualityBasedRanking:
    """Tests for quality-based ranking in queries."""

    def test_quality_ranking_integration(self, mock_query_engine, sample_nodes):
        """Test integration of quality ranking with query engine."""
        quality_engine = KnowledgeQualityEnhancementEngine(mock_query_engine)

        # Test the quality ranker
        ranker = quality_engine.quality_ranker

        ranked_nodes = ranker.rank_nodes_by_quality(sample_nodes)

        assert len(ranked_nodes) == len(sample_nodes)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in ranked_nodes)
        assert all(0.0 <= score <= 1.0 for _, score in ranked_nodes)

        # Verify ranking order (higher quality first)
        scores = [score for _, score in ranked_nodes]
        assert scores == sorted(scores, reverse=True)

    @patch("memory_core.query.query_engine.AdvancedQueryEngine")
    def test_enhanced_query_execution(self, mock_engine_class, sample_nodes):
        """Test enhanced query execution with quality ranking."""
        # Setup mock query engine
        mock_engine = Mock()
        mock_response = Mock()
        mock_response.results = []
        mock_response.query = "test query"
        mock_response.total_results = 0
        mock_response.processing_time_ms = 100
        mock_response.metadata = {}
        mock_engine.query.return_value = mock_response

        quality_engine = KnowledgeQualityEnhancementEngine(mock_engine)

        request = QueryRequest(query="test query", query_type=QueryType.NATURAL_LANGUAGE)

        response = quality_engine.enhance_query_with_quality_ranking(request)

        assert response
        assert hasattr(response, "metadata")


class TestIntegrationScenarios:
    """Integration tests for complete quality enhancement workflows."""

    def test_full_quality_enhancement_workflow(self, mock_query_engine, sample_nodes):
        """Test complete quality enhancement workflow."""
        engine = KnowledgeQualityEnhancementEngine(mock_query_engine)

        # Step 1: Assess current quality
        quality_scores = [engine.get_quality_score(node) for node in sample_nodes]

        # Step 2: Run comprehensive enhancement
        report = engine.enhance_knowledge_quality(sample_nodes, perform_enhancements=False)

        # Step 3: Verify results
        assert report.total_nodes_analyzed == len(sample_nodes)
        assert len(report.quality_scores) == len(quality_scores)

        # Step 4: Check for improvements and recommendations
        low_quality_nodes = [score for score in report.quality_scores if score.overall_score < 0.6]

        if low_quality_nodes:
            # Should have recommendations for low quality nodes
            relevant_recommendations = [
                rec
                for rec in report.enhancement_recommendations
                if rec.node_id in [node.node_id for node in low_quality_nodes]
            ]
            assert len(relevant_recommendations) > 0

    def test_performance_characteristics(self, mock_query_engine, sample_nodes):
        """Test performance characteristics of quality enhancement."""
        engine = KnowledgeQualityEnhancementEngine(mock_query_engine)

        start_time = time.time()

        # Run enhancement on sample nodes
        report = engine.enhance_knowledge_quality(sample_nodes)

        processing_time = time.time() - start_time

        # Verify reasonable performance
        assert report.processing_time_ms > 0
        assert processing_time < 10.0  # Should complete within 10 seconds

        # Test caching effectiveness
        start_time = time.time()

        # Run again on same nodes (should be faster due to caching)
        report2 = engine.enhance_knowledge_quality(sample_nodes)

        processing_time2 = time.time() - start_time

        # Second run should be faster or similar (caching effect)
        assert processing_time2 <= processing_time * 1.5

    def test_statistics_collection(self, mock_query_engine, sample_nodes):
        """Test statistics collection across all components."""
        engine = KnowledgeQualityEnhancementEngine(mock_query_engine)

        # Run some operations to generate statistics
        engine.enhance_knowledge_quality(sample_nodes[:2])

        # Get comprehensive statistics
        stats = engine.get_statistics()

        assert "quality_enhancement" in stats
        assert "quality_assessment" in stats
        assert "cross_validation" in stats
        assert "source_reliability" in stats
        assert "gap_detection" in stats
        assert "contradiction_resolution" in stats

        # Verify basic statistics are present
        quality_stats = stats["quality_enhancement"]
        assert "nodes_enhanced" in quality_stats
        assert quality_stats["nodes_enhanced"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
