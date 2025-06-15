"""
Knowledge Synthesis Engine Example

Demonstrates the comprehensive knowledge synthesis capabilities including
question answering, insight discovery, and perspective analysis.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_core.synthesis import (
    KnowledgeSynthesisEngine,
    SynthesisRequest,
    SynthesisTaskType,
    SynthesisMode,
)
from memory_core.query.query_engine import AdvancedQueryEngine
from memory_core.db.graph_storage_adapter import GraphStorageAdapter
from memory_core.embeddings.embedding_manager import EmbeddingManager
from memory_core.synthesis.question_answering import QuestionContext
from memory_core.synthesis.insight_discovery import PatternType, TrendType, AnomalyType
from memory_core.synthesis.perspective_analysis import PerspectiveType


def create_mock_components():
    """Create mock components for demonstration."""
    from unittest.mock import Mock
    from memory_core.query.query_types import QueryResponse, QueryResult
    from datetime import datetime

    # Create mock query engine
    mock_query_engine = Mock()

    # Setup realistic mock responses
    mock_query_engine.query.return_value = QueryResponse(
        results=[
            QueryResult(
                node_id="ai_concept_1",
                content="Artificial Intelligence (AI) is transforming healthcare through improved diagnostics and personalized treatment plans. Machine learning algorithms can analyze medical images with high accuracy.",
                node_type="research_finding",
                relevance_score=0.9,
                metadata={
                    "domain": "healthcare",
                    "timestamp": "2024-01-15T10:00:00Z",
                    "source": "medical_journal",
                    "confidence": 0.85,
                },
            ),
            QueryResult(
                node_id="ai_concern_1",
                content="However, experts raise concerns about AI bias in healthcare decisions and the need for explainable AI systems to maintain trust.",
                node_type="expert_opinion",
                relevance_score=0.8,
                metadata={
                    "domain": "ai_ethics",
                    "timestamp": "2024-01-20T14:30:00Z",
                    "stakeholder": "experts",
                    "concern_level": "moderate",
                },
            ),
            QueryResult(
                node_id="ai_application_1",
                content="Recent studies show AI applications in radiology have achieved 95% accuracy in detecting certain cancers, supporting radiologists in making faster diagnoses.",
                node_type="clinical_study",
                relevance_score=0.87,
                metadata={
                    "domain": "medical_imaging",
                    "timestamp": "2024-01-25T09:15:00Z",
                    "accuracy_rate": 0.95,
                    "study_size": 10000,
                },
            ),
            QueryResult(
                node_id="ai_trend_1",
                content="The adoption of AI in healthcare is accelerating, with investment increasing by 40% year-over-year in medical AI startups.",
                node_type="market_trend",
                relevance_score=0.75,
                metadata={
                    "domain": "healthcare_market",
                    "timestamp": "2024-02-01T16:45:00Z",
                    "growth_rate": 0.4,
                    "trend_direction": "increasing",
                },
            ),
        ],
        total_count=4,
        returned_count=4,
        execution_time_ms=120.0,
        query_id="demo_query",
        timestamp=datetime.now(),
    )

    mock_query_engine.get_statistics.return_value = {
        "total_queries": 1,
        "avg_execution_time_ms": 120.0,
    }

    return mock_query_engine


def demonstrate_question_answering():
    """Demonstrate question answering capabilities."""
    print("\n" + "=" * 60)
    print("🤖 QUESTION ANSWERING DEMONSTRATION")
    print("=" * 60)

    mock_query_engine = create_mock_components()
    synthesis_engine = KnowledgeSynthesisEngine(mock_query_engine)

    # Create question answering request
    context = QuestionContext(
        domain="healthcare", entities=["AI", "artificial intelligence", "medical diagnostics"]
    )

    request = SynthesisRequest(
        task_type=SynthesisTaskType.QUESTION_ANSWERING,
        query="How is artificial intelligence being used in healthcare?",
        context=context,
        mode=SynthesisMode.BALANCED,
    )

    result = synthesis_engine.synthesize(request)

    print(f"Question: {request.query}")
    print(f"Answer: {result.answer}")
    print(f"Confidence: {result.confidence_score:.2%}")
    print(f"Question Type: {result.question_type.value}")
    print(f"Sources Used: {len(result.sources)}")
    print(f"Processing Time: {result.processing_time_ms:.1f}ms")

    if result.sources:
        print("\nKey Sources:")
        for i, source in enumerate(result.sources[:3], 1):
            print(f"  {i}. {source.content_snippet[:100]}...")
            print(f"     Relevance: {source.relevance_score:.2%}")

    if result.follow_up_questions:
        print("\nSuggested Follow-up Questions:")
        for i, question in enumerate(result.follow_up_questions[:3], 1):
            print(f"  {i}. {question}")


def demonstrate_insight_discovery():
    """Demonstrate insight discovery capabilities."""
    print("\n" + "=" * 60)
    print("🔍 INSIGHT DISCOVERY DEMONSTRATION")
    print("=" * 60)

    mock_query_engine = create_mock_components()
    synthesis_engine = KnowledgeSynthesisEngine(mock_query_engine)

    request = SynthesisRequest(
        task_type=SynthesisTaskType.INSIGHT_DISCOVERY,
        query="healthcare AI trends",
        mode=SynthesisMode.BALANCED,
        pattern_types=[PatternType.FREQUENCY, PatternType.CLUSTERING],
        trend_types=[TrendType.GROWTH, TrendType.TEMPORAL],
        anomaly_types=[AnomalyType.OUTLIER, AnomalyType.STRUCTURAL],
    )

    result = synthesis_engine.synthesize(request)

    print(f"Domain Analyzed: {request.query}")
    print(f"Discovery Time: {result.discovery_time_ms:.1f}ms")
    print(f"Entities Analyzed: {result.total_entities_analyzed}")

    print(f"\nPatterns Discovered: {len(result.patterns)}")
    for i, pattern in enumerate(result.patterns[:3], 1):
        print(f"  {i}. {pattern.description}")
        print(f"     Type: {pattern.pattern_type.value}")
        print(f"     Confidence: {pattern.confidence:.2%}")
        print(f"     Support: {pattern.support} instances")

    print(f"\nTrends Identified: {len(result.trends)}")
    for i, trend in enumerate(result.trends[:3], 1):
        print(f"  {i}. {trend.description}")
        print(f"     Type: {trend.trend_type.value}")
        print(f"     Strength: {trend.strength:.2%}")
        print(f"     Direction: {trend.direction}")

    print(f"\nAnomalies Detected: {len(result.anomalies)}")
    for i, anomaly in enumerate(result.anomalies[:3], 1):
        print(f"  {i}. {anomaly.description}")
        print(f"     Type: {anomaly.anomaly_type.value}")
        print(f"     Severity: {anomaly.severity:.2%}")

    if result.recommendations:
        print("\nRecommendations:")
        for i, rec in enumerate(result.recommendations[:3], 1):
            print(f"  {i}. {rec}")


def demonstrate_perspective_analysis():
    """Demonstrate perspective analysis capabilities."""
    print("\n" + "=" * 60)
    print("👥 PERSPECTIVE ANALYSIS DEMONSTRATION")
    print("=" * 60)

    mock_query_engine = create_mock_components()
    synthesis_engine = KnowledgeSynthesisEngine(mock_query_engine)

    request = SynthesisRequest(
        task_type=SynthesisTaskType.PERSPECTIVE_ANALYSIS,
        query="AI in healthcare",
        mode=SynthesisMode.COMPREHENSIVE,
        perspective_types=[
            PerspectiveType.STAKEHOLDER,
            PerspectiveType.OPPOSING,
            PerspectiveType.COMPLEMENTARY,
        ],
        include_temporal_analysis=True,
        include_stakeholder_analysis=True,
    )

    result = synthesis_engine.synthesize(request)

    print(f"Topic Analyzed: {request.query}")
    print(f"Processing Time: {result.processing_time_ms:.1f}ms")
    print(f"Analysis Confidence: {result.analysis_confidence:.2%}")
    print(f"Overall Consensus: {result.overall_consensus.value.replace('_', ' ').title()}")

    print(f"\nPerspectives Found: {len(result.perspectives)}")
    for i, perspective in enumerate(result.perspectives[:3], 1):
        print(f"  {i}. {perspective.perspective_type.value.title()} Perspective")
        print(f"     Viewpoint: {perspective.viewpoint[:100]}...")
        print(f"     Confidence: {perspective.confidence_score:.2%}")
        print(f"     Sources: {len(perspective.supporting_evidence)}")

        if perspective.key_claims:
            print(f"     Key Claims: {len(perspective.key_claims)}")
            for claim in perspective.key_claims[:2]:
                print(f"       - {claim[:80]}...")

    print(f"\nComparisons Made: {len(result.comparisons)}")
    for i, comparison in enumerate(result.comparisons[:2], 1):
        print(f"  {i}. Comparison of {len(comparison.perspectives)} perspectives")
        print(f"     Consensus Level: {comparison.consensus_level.value.replace('_', ' ').title()}")
        print(f"     Consensus Areas: {len(comparison.consensus_areas)}")
        print(f"     Disagreement Areas: {len(comparison.disagreement_areas)}")

    if result.key_insights:
        print("\nKey Insights:")
        for i, insight in enumerate(result.key_insights[:3], 1):
            print(f"  {i}. {insight}")

    if result.stakeholder_analysis:
        print(f"\nStakeholder Analysis: {len(result.stakeholder_analysis)} groups")
        for stakeholder in result.stakeholder_analysis[:2]:
            print(f"  - {stakeholder.stakeholder_group.title()}")
            print(f"    Influence Level: {stakeholder.influence_level:.2%}")
            if stakeholder.interests:
                print(f"    Key Interests: {', '.join(stakeholder.interests[:2])}")


def demonstrate_comprehensive_synthesis():
    """Demonstrate comprehensive synthesis combining all engines."""
    print("\n" + "=" * 80)
    print("🧠 COMPREHENSIVE SYNTHESIS DEMONSTRATION")
    print("=" * 80)

    mock_query_engine = create_mock_components()
    synthesis_engine = KnowledgeSynthesisEngine(mock_query_engine)

    request = SynthesisRequest(
        task_type=SynthesisTaskType.COMPREHENSIVE_SYNTHESIS,
        query="What are the opportunities and challenges of AI in healthcare?",
        mode=SynthesisMode.COMPREHENSIVE,
        context=QuestionContext(
            domain="healthcare",
            entities=["artificial intelligence", "medical diagnostics", "patient care"],
        ),
        include_temporal_analysis=True,
        include_stakeholder_analysis=True,
    )

    result = synthesis_engine.synthesize(request)

    print(f"Query: {request.query}")
    print(f"Processing Mode: {request.mode.value.title()}")
    print(f"Total Processing Time: {result.processing_time_ms:.1f}ms")
    print(f"Synthesis Confidence: {result.synthesis_confidence:.2%}")
    print(f"Unique Sources Analyzed: {result.sources_analyzed}")
    print(f"Cross-Validation Score: {result.cross_validation_score:.2%}")

    print("\n" + "-" * 50)
    print("EXECUTIVE SUMMARY")
    print("-" * 50)
    print(result.executive_summary)

    if result.key_findings:
        print("\n" + "-" * 50)
        print("KEY FINDINGS")
        print("-" * 50)
        for i, finding in enumerate(result.key_findings, 1):
            print(f"{i}. {finding}")

    if result.confidence_assessment:
        print("\n" + "-" * 50)
        print("CONFIDENCE ASSESSMENT")
        print("-" * 50)
        for component, confidence in result.confidence_assessment.items():
            print(f"  {component.replace('_', ' ').title()}: {confidence:.2%}")

    if result.actionable_recommendations:
        print("\n" + "-" * 50)
        print("ACTIONABLE RECOMMENDATIONS")
        print("-" * 50)
        for i, rec in enumerate(result.actionable_recommendations, 1):
            print(f"{i}. {rec}")

    if result.suggested_follow_ups:
        print("\n" + "-" * 50)
        print("SUGGESTED FOLLOW-UPS")
        print("-" * 50)
        for i, follow_up in enumerate(result.suggested_follow_ups, 1):
            print(f"{i}. {follow_up}")

    if result.related_topics:
        print("\n" + "-" * 50)
        print("RELATED TOPICS")
        print("-" * 50)
        print(f"  {', '.join(result.related_topics)}")

    # Show individual component results
    if result.question_answer:
        print("\n" + "-" * 30)
        print("QUESTION ANSWERING RESULT")
        print("-" * 30)
        print(f"Answer: {result.question_answer.answer[:200]}...")
        print(f"Confidence: {result.question_answer.confidence_score:.2%}")

    if result.insights:
        print("\n" + "-" * 30)
        print("INSIGHTS DISCOVERED")
        print("-" * 30)
        total_insights = (
            len(result.insights.patterns)
            + len(result.insights.trends)
            + len(result.insights.anomalies)
        )
        print(f"Total Insights: {total_insights}")
        print(f"Patterns: {len(result.insights.patterns)}")
        print(f"Trends: {len(result.insights.trends)}")
        print(f"Anomalies: {len(result.insights.anomalies)}")

    if result.perspectives:
        print("\n" + "-" * 30)
        print("PERSPECTIVES ANALYZED")
        print("-" * 30)
        print(f"Perspectives: {len(result.perspectives.perspectives)}")
        print(f"Comparisons: {len(result.perspectives.comparisons)}")
        print(
            f"Consensus Level: {result.perspectives.overall_consensus.value.replace('_', ' ').title()}"
        )


def demonstrate_engine_capabilities():
    """Demonstrate engine capabilities and statistics."""
    print("\n" + "=" * 60)
    print("⚙️  ENGINE CAPABILITIES")
    print("=" * 60)

    mock_query_engine = create_mock_components()
    synthesis_engine = KnowledgeSynthesisEngine(mock_query_engine)

    capabilities = synthesis_engine.get_capabilities()

    print("Synthesis Types:")
    for i, task_type in enumerate(capabilities["synthesis_types"], 1):
        print(f"  {i}. {task_type.replace('_', ' ').title()}")

    print("\nSynthesis Modes:")
    for i, mode in enumerate(capabilities["synthesis_modes"], 1):
        print(f"  {i}. {mode.title()}")

    print("\nPerspective Types:")
    for i, ptype in enumerate(capabilities["perspective_types"], 1):
        print(f"  {i}. {ptype.replace('_', ' ').title()}")

    print("\nPattern Types:")
    for i, ptype in enumerate(capabilities["pattern_types"], 1):
        print(f"  {i}. {ptype.replace('_', ' ').title()}")

    print("\nKey Features:")
    for feature, enabled in capabilities["features"].items():
        status = "✓" if enabled else "✗"
        print(f"  {status} {feature.replace('_', ' ').title()}")

    # Show statistics after running some syntheses
    stats = synthesis_engine.get_statistics()
    print(f"\nEngine Statistics:")
    print(f"  Total Syntheses: {stats['synthesis_engine']['total_syntheses']}")
    print(f"  Average Processing Time: {stats['synthesis_engine']['avg_processing_time_ms']:.1f}ms")
    print(f"  Average Confidence: {stats['synthesis_engine']['avg_confidence_score']:.2%}")


def main():
    """Run all demonstrations."""
    print("🧠 KNOWLEDGE SYNTHESIS ENGINE DEMONSTRATION")
    print("=" * 80)
    print("This demonstration shows the comprehensive knowledge synthesis")
    print("capabilities including question answering, insight discovery,")
    print("and perspective analysis using simulated data.")

    try:
        # Run individual component demonstrations
        demonstrate_question_answering()
        demonstrate_insight_discovery()
        demonstrate_perspective_analysis()

        # Run comprehensive synthesis
        demonstrate_comprehensive_synthesis()

        # Show engine capabilities
        demonstrate_engine_capabilities()

        print("\n" + "=" * 80)
        print("✅ DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("The Knowledge Synthesis Engine demonstrates:")
        print("• Intelligent question answering with source attribution")
        print("• Pattern, trend, and anomaly discovery")
        print("• Multi-perspective analysis with consensus detection")
        print("• Comprehensive synthesis with cross-validation")
        print("• Configurable processing modes and capabilities")

    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
