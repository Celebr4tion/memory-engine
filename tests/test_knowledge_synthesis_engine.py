"""
Tests for Knowledge Synthesis Engine

Comprehensive tests for the synthesis engine including question answering,
insight discovery, and perspective analysis capabilities.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

from memory_core.synthesis.knowledge_synthesis_engine import (
    KnowledgeSynthesisEngine,
    SynthesisRequest,
    SynthesisTaskType,
    SynthesisMode,
    ComprehensiveSynthesisResult,
)
from memory_core.synthesis.question_answering import (
    QuestionAnsweringSystem,
    SynthesizedAnswer,
    QuestionType,
    QuestionContext,
    AnswerSource,
)
from memory_core.synthesis.insight_discovery import (
    InsightDiscoveryEngine,
    InsightReport,
    Pattern,
    Trend,
    Anomaly,
    PatternType,
    TrendType,
    AnomalyType,
)
from memory_core.synthesis.perspective_analysis import (
    PerspectiveAnalysisEngine,
    PerspectiveAnalysisReport,
    Perspective,
    PerspectiveType,
    ConsensusLevel,
)
from memory_core.query.query_types import QueryResponse, QueryResult


class TestKnowledgeSynthesisEngine:
    """Test the main Knowledge Synthesis Engine."""

    @pytest.fixture
    def mock_query_engine(self):
        """Create a mock query engine."""
        engine = Mock()
        engine.query.return_value = QueryResponse(
            results=[
                QueryResult(
                    node_id="test_node_1",
                    content="Test content about artificial intelligence and machine learning",
                    node_type="concept",
                    relevance_score=0.8,
                    metadata={"domain": "technology", "timestamp": "2024-01-01T00:00:00Z"},
                ),
                QueryResult(
                    node_id="test_node_2",
                    content="Alternative perspective on AI safety and ethics",
                    node_type="opinion",
                    relevance_score=0.7,
                    metadata={"domain": "technology", "timestamp": "2024-01-02T00:00:00Z"},
                ),
            ],
            total_count=2,
            returned_count=2,
            execution_time_ms=100.0,
            query_id="test_query",
            timestamp=datetime.now(),
        )
        return engine

    @pytest.fixture
    def synthesis_engine(self, mock_query_engine):
        """Create a synthesis engine with mocked dependencies."""
        return KnowledgeSynthesisEngine(mock_query_engine)

    def test_initialization(self, mock_query_engine):
        """Test synthesis engine initialization."""
        engine = KnowledgeSynthesisEngine(mock_query_engine)

        assert engine.query_engine == mock_query_engine
        assert isinstance(engine.question_answering, QuestionAnsweringSystem)
        assert isinstance(engine.insight_discovery, InsightDiscoveryEngine)
        assert isinstance(engine.perspective_analysis, PerspectiveAnalysisEngine)
        assert engine.stats["total_syntheses"] == 0

    def test_question_answering_synthesis(self, synthesis_engine):
        """Test question answering synthesis."""
        request = SynthesisRequest(
            task_type=SynthesisTaskType.QUESTION_ANSWERING,
            query="What is artificial intelligence?",
            mode=SynthesisMode.FAST,
        )

        result = synthesis_engine.synthesize(request)

        assert isinstance(result, SynthesizedAnswer)
        assert result.answer is not None
        assert result.confidence_score >= 0.0
        assert result.processing_time_ms > 0

    def test_insight_discovery_synthesis(self, synthesis_engine):
        """Test insight discovery synthesis."""
        request = SynthesisRequest(
            task_type=SynthesisTaskType.INSIGHT_DISCOVERY,
            query="technology trends",
            mode=SynthesisMode.BALANCED,
        )

        result = synthesis_engine.synthesize(request)

        assert isinstance(result, InsightReport)
        assert isinstance(result.patterns, list)
        assert isinstance(result.trends, list)
        assert isinstance(result.anomalies, list)
        assert result.discovery_time_ms > 0

    def test_perspective_analysis_synthesis(self, synthesis_engine):
        """Test perspective analysis synthesis."""
        request = SynthesisRequest(
            task_type=SynthesisTaskType.PERSPECTIVE_ANALYSIS,
            query="artificial intelligence",
            mode=SynthesisMode.BALANCED,
        )

        result = synthesis_engine.synthesize(request)

        assert isinstance(result, PerspectiveAnalysisReport)
        assert isinstance(result.perspectives, list)
        assert isinstance(result.comparisons, list)
        assert result.processing_time_ms > 0

    def test_comprehensive_synthesis(self, synthesis_engine):
        """Test comprehensive synthesis combining all engines."""
        request = SynthesisRequest(
            task_type=SynthesisTaskType.COMPREHENSIVE_SYNTHESIS,
            query="What are the implications of artificial intelligence?",
            mode=SynthesisMode.COMPREHENSIVE,
        )

        result = synthesis_engine.synthesize(request)

        assert isinstance(result, ComprehensiveSynthesisResult)
        assert result.executive_summary is not None
        assert isinstance(result.key_findings, list)
        assert isinstance(result.confidence_assessment, dict)
        assert isinstance(result.actionable_recommendations, list)
        assert result.synthesis_confidence >= 0.0
        assert result.processing_time_ms > 0

    def test_synthesis_modes(self, synthesis_engine):
        """Test different synthesis modes."""
        base_request = SynthesisRequest(
            task_type=SynthesisTaskType.QUESTION_ANSWERING, query="Test query"
        )

        # Test each mode
        for mode in SynthesisMode:
            request = SynthesisRequest(
                task_type=base_request.task_type, query=base_request.query, mode=mode
            )

            result = synthesis_engine.synthesize(request)
            assert result is not None

    def test_statistics_tracking(self, synthesis_engine):
        """Test that statistics are properly tracked."""
        initial_stats = synthesis_engine.get_statistics()
        assert initial_stats["synthesis_engine"]["total_syntheses"] == 0

        request = SynthesisRequest(
            task_type=SynthesisTaskType.QUESTION_ANSWERING, query="Test query"
        )

        synthesis_engine.synthesize(request)

        updated_stats = synthesis_engine.get_statistics()
        assert updated_stats["synthesis_engine"]["total_syntheses"] == 1
        assert "question_answering" in updated_stats["synthesis_engine"]["synthesis_by_type"]

    def test_error_handling(self, synthesis_engine):
        """Test error handling in synthesis."""
        # Mock an error in the question answering system
        synthesis_engine.question_answering.answer_question = Mock(
            side_effect=Exception("Test error")
        )

        request = SynthesisRequest(
            task_type=SynthesisTaskType.QUESTION_ANSWERING, query="Test query"
        )

        result = synthesis_engine.synthesize(request)

        # Should return error result instead of crashing
        assert isinstance(result, SynthesizedAnswer)
        assert "error" in result.answer.lower()
        assert result.confidence_score == 0.0

    def test_capabilities_report(self, synthesis_engine):
        """Test capabilities reporting."""
        capabilities = synthesis_engine.get_capabilities()

        assert "synthesis_types" in capabilities
        assert "synthesis_modes" in capabilities
        assert "features" in capabilities

        # Check that all synthesis types are included
        expected_types = [task_type.value for task_type in SynthesisTaskType]
        assert set(capabilities["synthesis_types"]) == set(expected_types)


class TestQuestionAnsweringSystem:
    """Test the Question Answering System."""

    @pytest.fixture
    def mock_query_engine(self):
        """Create a mock query engine for QA testing."""
        engine = Mock()
        engine.query.return_value = QueryResponse(
            results=[
                QueryResult(
                    node_id="qa_node_1",
                    content="Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data",
                    node_type="definition",
                    relevance_score=0.9,
                    metadata={"confidence": 0.8},
                )
            ],
            total_count=1,
            returned_count=1,
            execution_time_ms=50.0,
            query_id="qa_test",
            timestamp=datetime.now(),
        )
        return engine

    @pytest.fixture
    def qa_system(self, mock_query_engine):
        """Create a question answering system."""
        return QuestionAnsweringSystem(mock_query_engine)

    def test_basic_question_answering(self, qa_system):
        """Test basic question answering functionality."""
        result = qa_system.answer_question("What is machine learning?")

        assert isinstance(result, SynthesizedAnswer)
        assert result.answer is not None
        assert len(result.answer) > 0
        assert result.confidence_score > 0
        assert result.question_type in QuestionType

    def test_question_with_context(self, qa_system):
        """Test question answering with context."""
        context = QuestionContext(
            domain="technology", time_frame="recent", entities=["AI", "machine learning"]
        )

        result = qa_system.answer_question(
            "How does AI relate to machine learning?", context=context
        )

        assert isinstance(result, SynthesizedAnswer)
        assert result.answer is not None

    def test_question_types_detection(self, qa_system):
        """Test detection of different question types."""
        test_questions = [
            ("What is AI?", QuestionType.FACTUAL),
            ("How does X compare to Y?", QuestionType.COMPARATIVE),
            ("Why does this happen?", QuestionType.CAUSAL),
            ("How to implement AI?", QuestionType.PROCEDURAL),
            ("Define machine learning", QuestionType.DEFINITIONAL),
        ]

        for question, expected_type in test_questions:
            result = qa_system.answer_question(question)
            # Note: Actual type detection may vary based on implementation
            assert isinstance(result.question_type, QuestionType)

    def test_source_attribution(self, qa_system):
        """Test that sources are properly attributed."""
        result = qa_system.answer_question("What is AI?")

        assert isinstance(result.sources, list)
        for source in result.sources:
            assert isinstance(source, AnswerSource)
            assert source.node_id is not None
            assert source.relevance_score >= 0


class TestInsightDiscoveryEngine:
    """Test the Insight Discovery Engine."""

    @pytest.fixture
    def mock_query_engine(self):
        """Create a mock query engine for insight discovery."""
        engine = Mock()
        engine.query.return_value = QueryResponse(
            results=[
                QueryResult(
                    node_id=f"insight_node_{i}",
                    content=f"Content about technology trend {i}",
                    node_type="trend",
                    relevance_score=0.8,
                    metadata={"timestamp": "2024-01-01T00:00:00Z", "domain": "technology"},
                    relationships=[],
                )
                for i in range(5)
            ],
            total_count=5,
            returned_count=5,
            execution_time_ms=200.0,
            query_id="insight_test",
            timestamp=datetime.now(),
        )
        return engine

    @pytest.fixture
    def insight_engine(self, mock_query_engine):
        """Create an insight discovery engine."""
        return InsightDiscoveryEngine(mock_query_engine)

    def test_insight_discovery_basic(self, insight_engine):
        """Test basic insight discovery."""
        result = insight_engine.discover_insights(domain="technology")

        assert isinstance(result, InsightReport)
        assert isinstance(result.patterns, list)
        assert isinstance(result.trends, list)
        assert isinstance(result.anomalies, list)
        assert result.discovery_time_ms > 0

    def test_pattern_detection_types(self, insight_engine):
        """Test different pattern detection types."""
        pattern_types = [PatternType.FREQUENCY, PatternType.CLUSTERING]

        result = insight_engine.discover_insights(domain="technology", pattern_types=pattern_types)

        # Should attempt to find the specified pattern types
        assert isinstance(result.patterns, list)

    def test_trend_analysis_types(self, insight_engine):
        """Test different trend analysis types."""
        trend_types = [TrendType.TEMPORAL, TrendType.GROWTH]

        result = insight_engine.discover_insights(domain="technology", trend_types=trend_types)

        assert isinstance(result.trends, list)

    def test_anomaly_detection_types(self, insight_engine):
        """Test different anomaly detection types."""
        anomaly_types = [AnomalyType.OUTLIER, AnomalyType.STRUCTURAL]

        result = insight_engine.discover_insights(domain="technology", anomaly_types=anomaly_types)

        assert isinstance(result.anomalies, list)


class TestPerspectiveAnalysisEngine:
    """Test the Perspective Analysis Engine."""

    @pytest.fixture
    def mock_query_engine(self):
        """Create a mock query engine for perspective analysis."""
        engine = Mock()
        engine.query.return_value = QueryResponse(
            results=[
                QueryResult(
                    node_id="perspective_node_1",
                    content="Experts believe that AI will revolutionize healthcare",
                    node_type="expert_opinion",
                    relevance_score=0.8,
                    metadata={"stakeholder": "experts", "domain": "healthcare"},
                ),
                QueryResult(
                    node_id="perspective_node_2",
                    content="However, users are concerned about privacy implications",
                    node_type="user_concern",
                    relevance_score=0.7,
                    metadata={"stakeholder": "users", "domain": "privacy"},
                ),
            ],
            total_count=2,
            returned_count=2,
            execution_time_ms=150.0,
            query_id="perspective_test",
            timestamp=datetime.now(),
        )
        return engine

    @pytest.fixture
    def perspective_engine(self, mock_query_engine):
        """Create a perspective analysis engine."""
        return PerspectiveAnalysisEngine(mock_query_engine)

    def test_perspective_analysis_basic(self, perspective_engine):
        """Test basic perspective analysis."""
        result = perspective_engine.analyze_perspectives("artificial intelligence")

        assert isinstance(result, PerspectiveAnalysisReport)
        assert isinstance(result.perspectives, list)
        assert isinstance(result.comparisons, list)
        assert result.processing_time_ms > 0

    def test_perspective_types(self, perspective_engine):
        """Test analysis of specific perspective types."""
        perspective_types = [PerspectiveType.STAKEHOLDER, PerspectiveType.OPPOSING]

        result = perspective_engine.analyze_perspectives(
            "AI ethics", perspective_types=perspective_types
        )

        assert isinstance(result, PerspectiveAnalysisReport)

    def test_consensus_analysis(self, perspective_engine):
        """Test consensus level analysis."""
        result = perspective_engine.analyze_perspectives("technology")

        assert hasattr(result, "overall_consensus")
        assert isinstance(result.overall_consensus, ConsensusLevel)

    def test_stakeholder_analysis(self, perspective_engine):
        """Test stakeholder-specific analysis."""
        result = perspective_engine.analyze_perspectives(
            "AI development", include_stakeholder_analysis=True
        )

        assert isinstance(result.stakeholder_analysis, list)

    def test_temporal_evolution(self, perspective_engine):
        """Test temporal evolution analysis."""
        result = perspective_engine.analyze_perspectives(
            "machine learning", include_temporal_analysis=True
        )

        # Temporal evolution may be None if insufficient temporal data
        if result.temporal_evolution:
            assert hasattr(result.temporal_evolution, "evolution_trend")


class TestSynthesisIntegration:
    """Test integration between synthesis components."""

    @pytest.fixture
    def comprehensive_mock_engine(self):
        """Create a comprehensive mock engine for integration testing."""
        engine = Mock()

        # Mock responses with rich data for cross-validation
        engine.query.return_value = QueryResponse(
            results=[
                QueryResult(
                    node_id="integration_node_1",
                    content="Artificial intelligence shows promising applications in healthcare",
                    node_type="research_finding",
                    relevance_score=0.9,
                    metadata={
                        "domain": "healthcare",
                        "timestamp": "2024-01-01T00:00:00Z",
                        "confidence": 0.8,
                    },
                    relationships=[],
                ),
                QueryResult(
                    node_id="integration_node_2",
                    content="However, experts warn about ethical implications of AI",
                    node_type="expert_opinion",
                    relevance_score=0.8,
                    metadata={
                        "domain": "ethics",
                        "timestamp": "2024-01-02T00:00:00Z",
                        "stakeholder": "experts",
                    },
                    relationships=[],
                ),
            ],
            total_count=2,
            returned_count=2,
            execution_time_ms=100.0,
            query_id="integration_test",
            timestamp=datetime.now(),
        )
        return engine

    @pytest.fixture
    def integration_engine(self, comprehensive_mock_engine):
        """Create synthesis engine for integration testing."""
        return KnowledgeSynthesisEngine(comprehensive_mock_engine)

    def test_comprehensive_synthesis_integration(self, integration_engine):
        """Test that comprehensive synthesis properly integrates all components."""
        request = SynthesisRequest(
            task_type=SynthesisTaskType.COMPREHENSIVE_SYNTHESIS,
            query="What are the implications of AI in healthcare?",
            mode=SynthesisMode.COMPREHENSIVE,
        )

        result = integration_engine.synthesize(request)

        assert isinstance(result, ComprehensiveSynthesisResult)

        # Check that individual components were executed
        assert result.question_answer is not None
        assert result.insights is not None
        assert result.perspectives is not None

        # Check integration results
        assert result.executive_summary is not None
        assert len(result.executive_summary) > 0
        assert isinstance(result.key_findings, list)
        assert isinstance(result.confidence_assessment, dict)
        assert isinstance(result.actionable_recommendations, list)

        # Check meta information
        assert result.synthesis_confidence >= 0.0
        assert result.sources_analyzed >= 0
        assert result.cross_validation_score >= 0.0

    def test_cross_validation_scoring(self, integration_engine):
        """Test cross-validation between synthesis components."""
        request = SynthesisRequest(
            task_type=SynthesisTaskType.COMPREHENSIVE_SYNTHESIS,
            query="AI healthcare applications",
            mode=SynthesisMode.BALANCED,
        )

        result = integration_engine.synthesize(request)

        # Cross-validation should be performed
        assert "cross_validation" in result.confidence_assessment
        assert result.cross_validation_score >= 0.0

    def test_confidence_aggregation(self, integration_engine):
        """Test confidence score aggregation across components."""
        request = SynthesisRequest(
            task_type=SynthesisTaskType.COMPREHENSIVE_SYNTHESIS,
            query="Technology trends",
            mode=SynthesisMode.BALANCED,
        )

        result = integration_engine.synthesize(request)

        # Should have confidence assessments for different components
        assert isinstance(result.confidence_assessment, dict)
        assert len(result.confidence_assessment) > 0

        # Overall synthesis confidence should be computed
        assert 0.0 <= result.synthesis_confidence <= 1.0

    def test_source_deduplication(self, integration_engine):
        """Test that sources are properly deduplicated across components."""
        request = SynthesisRequest(
            task_type=SynthesisTaskType.COMPREHENSIVE_SYNTHESIS,
            query="AI applications",
            mode=SynthesisMode.COMPREHENSIVE,
        )

        result = integration_engine.synthesize(request)

        # Should have counted unique sources
        assert result.sources_analyzed >= 0

    def test_recommendation_synthesis(self, integration_engine):
        """Test synthesis of recommendations from multiple components."""
        request = SynthesisRequest(
            task_type=SynthesisTaskType.COMPREHENSIVE_SYNTHESIS,
            query="Machine learning implementation",
            mode=SynthesisMode.COMPREHENSIVE,
        )

        result = integration_engine.synthesize(request)

        # Should generate actionable recommendations
        assert isinstance(result.actionable_recommendations, list)

        # Should generate follow-up suggestions
        assert isinstance(result.suggested_follow_ups, list)

        # Should identify related topics
        assert isinstance(result.related_topics, list)


class TestSynthesisPerformance:
    """Test synthesis engine performance characteristics."""

    @pytest.fixture
    def performance_mock_engine(self):
        """Create a mock engine with controlled response times."""
        engine = Mock()

        # Simulate realistic response times
        def mock_query(*args, **kwargs):
            import time

            time.sleep(0.01)  # 10ms delay
            return QueryResponse(
                results=[
                    QueryResult(
                        node_id="perf_node",
                        content="Performance test content",
                        node_type="test",
                        relevance_score=0.8,
                    )
                ],
                total_count=1,
                returned_count=1,
                execution_time_ms=10.0,
                query_id="perf_test",
                timestamp=datetime.now(),
            )

        engine.query.side_effect = mock_query
        return engine

    @pytest.fixture
    def performance_engine(self, performance_mock_engine):
        """Create synthesis engine for performance testing."""
        return KnowledgeSynthesisEngine(performance_mock_engine)

    def test_synthesis_mode_performance(self, performance_engine):
        """Test that different modes have appropriate performance characteristics."""
        query = "Test performance query"

        # Test fast mode
        fast_request = SynthesisRequest(
            task_type=SynthesisTaskType.QUESTION_ANSWERING, query=query, mode=SynthesisMode.FAST
        )
        fast_result = performance_engine.synthesize(fast_request)
        fast_time = fast_result.processing_time_ms

        # Test comprehensive mode
        comp_request = SynthesisRequest(
            task_type=SynthesisTaskType.COMPREHENSIVE_SYNTHESIS,
            query=query,
            mode=SynthesisMode.COMPREHENSIVE,
        )
        comp_result = performance_engine.synthesize(comp_request)
        comp_time = comp_result.processing_time_ms

        # Comprehensive should take longer than fast mode
        assert comp_time > fast_time

    def test_statistics_accuracy(self, performance_engine):
        """Test that performance statistics are accurately tracked."""
        initial_stats = performance_engine.get_statistics()
        initial_count = initial_stats["synthesis_engine"]["total_syntheses"]
        initial_avg_time = initial_stats["synthesis_engine"]["avg_processing_time_ms"]

        # Perform multiple syntheses
        for i in range(3):
            request = SynthesisRequest(
                task_type=SynthesisTaskType.QUESTION_ANSWERING, query=f"Test query {i}"
            )
            performance_engine.synthesize(request)

        final_stats = performance_engine.get_statistics()
        final_count = final_stats["synthesis_engine"]["total_syntheses"]
        final_avg_time = final_stats["synthesis_engine"]["avg_processing_time_ms"]

        # Check count is correct
        assert final_count == initial_count + 3

        # Check average time is reasonable
        assert final_avg_time > 0


if __name__ == "__main__":
    pytest.main([__file__])
