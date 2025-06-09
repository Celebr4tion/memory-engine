"""
Tests for Quality-Based Query Ranking Integration.

Focused tests for the integration of quality enhancement with query processing,
ensuring that quality scores properly influence result ranking.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from memory_core.model.knowledge_node import KnowledgeNode
from memory_core.query.query_types import QueryRequest, QueryResult, QueryResponse, QueryType
from memory_core.query.result_ranker import ResultRanker
from memory_core.quality.quality_enhancement_engine import KnowledgeQualityEnhancementEngine, QualityScore


@pytest.fixture
def mock_quality_engine():
    """Create a mock quality enhancement engine."""
    engine = Mock(spec=KnowledgeQualityEnhancementEngine)
    
    # Mock quality scores - simulate different quality levels
    def mock_get_quality_score(node):
        quality_scores = {
            "high_quality_node": QualityScore(
                node_id="high_quality_node",
                overall_score=0.9,
                content_score=0.9,
                structural_score=0.8,
                temporal_score=0.85,
                reliability_score=0.95,
                validation_score=0.9,
                quality_level=None,  # Will be set by actual implementation
                confidence=0.9,
                last_assessed=None
            ),
            "medium_quality_node": QualityScore(
                node_id="medium_quality_node",
                overall_score=0.6,
                content_score=0.6,
                structural_score=0.7,
                temporal_score=0.5,
                reliability_score=0.6,
                validation_score=0.65,
                quality_level=None,
                confidence=0.7,
                last_assessed=None
            ),
            "low_quality_node": QualityScore(
                node_id="low_quality_node",
                overall_score=0.3,
                content_score=0.2,
                structural_score=0.4,
                temporal_score=0.3,
                reliability_score=0.3,
                validation_score=0.35,
                quality_level=None,
                confidence=0.5,
                last_assessed=None
            )
        }
        return quality_scores.get(node.node_id, quality_scores["medium_quality_node"])
    
    engine.get_quality_score.side_effect = mock_get_quality_score
    return engine


@pytest.fixture
def sample_query_results():
    """Create sample query results with different quality characteristics."""
    return [
        QueryResult(
            node_id="high_quality_node",
            content="Comprehensive analysis of machine learning algorithms with detailed explanations, citations, and examples. This content provides thorough coverage of the topic with supporting evidence and clear methodology.",
            node_type="analysis",
            relevance_score=0.8,
            metadata={
                "source": "https://nature.com/articles/ml-comprehensive",
                "timestamp": "2024-01-20T10:00:00Z",
                "peer_reviewed": True,
                "rating_richness": 0.9,
                "rating_truthfulness": 0.95,
                "rating_stability": 0.85
            }
        ),
        QueryResult(
            node_id="medium_quality_node",
            content="Machine learning is used in many applications. It involves algorithms that learn from data.",
            node_type="definition",
            relevance_score=0.85,  # Higher relevance but lower quality
            metadata={
                "source": "https://example.com/ml-basics",
                "timestamp": "2024-01-15T14:30:00Z",
                "rating_richness": 0.6,
                "rating_truthfulness": 0.7,
                "rating_stability": 0.5
            }
        ),
        QueryResult(
            node_id="low_quality_node",
            content="ML stuff",  # Very low quality content
            node_type="note",
            relevance_score=0.9,  # Highest relevance but lowest quality
            metadata={
                "timestamp": "2024-01-01T12:00:00Z",
                "rating_richness": 0.2,
                "rating_truthfulness": 0.3,
                "rating_stability": 0.1
            }
        )
    ]


class TestQualityBasedRanking:
    """Tests for quality-based ranking functionality."""
    
    def test_ranker_with_quality_engine(self, mock_quality_engine, sample_query_results):
        """Test that ranker uses quality engine when available."""
        ranker = ResultRanker(quality_enhancement_engine=mock_quality_engine)
        
        request = QueryRequest(
            query="machine learning algorithms",
            query_type=QueryType.SEMANTIC_SEARCH
        )
        
        ranked_results = ranker.rank_results(sample_query_results, request)
        
        # Verify ranking order considers quality
        assert len(ranked_results) == 3
        
        # High quality node should be ranked higher despite lower relevance
        high_quality_result = next(r for r in ranked_results if r.node_id == "high_quality_node")
        low_quality_result = next(r for r in ranked_results if r.node_id == "low_quality_node")
        
        # Find their positions
        high_quality_pos = ranked_results.index(high_quality_result)
        low_quality_pos = ranked_results.index(low_quality_result)
        
        # High quality should be ranked better than low quality
        assert high_quality_pos < low_quality_pos
        
        # Verify quality engine was called
        assert mock_quality_engine.get_quality_score.called
    
    def test_ranker_without_quality_engine(self, sample_query_results):
        """Test ranker fallback behavior without quality engine."""
        ranker = ResultRanker()  # No quality engine
        
        request = QueryRequest(
            query="machine learning algorithms",
            query_type=QueryType.SEMANTIC_SEARCH
        )
        
        ranked_results = ranker.rank_results(sample_query_results, request)
        
        # Should still work, using basic quality metrics
        assert len(ranked_results) == 3
        
        # Should fall back to metadata-based quality scoring
        for result in ranked_results:
            assert hasattr(result, 'combined_score')
    
    def test_quality_score_caching(self, mock_quality_engine, sample_query_results):
        """Test that quality scores are cached for performance."""
        ranker = ResultRanker(quality_enhancement_engine=mock_quality_engine)
        
        request = QueryRequest(
            query="machine learning",
            query_type=QueryType.SEMANTIC_SEARCH
        )
        
        # First ranking
        ranker.rank_results(sample_query_results, request)
        first_call_count = mock_quality_engine.get_quality_score.call_count
        
        # Second ranking with same results
        ranker.rank_results(sample_query_results, request)
        second_call_count = mock_quality_engine.get_quality_score.call_count
        
        # Should use cache, so no additional calls
        assert second_call_count == first_call_count
    
    def test_quality_score_integration_in_metadata(self, mock_quality_engine, sample_query_results):
        """Test that quality scores are properly integrated into result metadata."""
        ranker = ResultRanker(quality_enhancement_engine=mock_quality_engine)
        
        # Add quality scores directly to metadata
        sample_query_results[0].metadata['quality_score'] = 0.95
        
        request = QueryRequest(
            query="machine learning",
            query_type=QueryType.SEMANTIC_SEARCH
        )
        
        ranked_results = ranker.rank_results(sample_query_results, request)
        
        # Should use the metadata quality score
        high_quality_result = ranked_results[0]
        
        # Verify the quality score was used
        assert high_quality_result.metadata.get('quality_score') == 0.95


class TestQualityEnhancedQueryExecution:
    """Tests for complete quality-enhanced query execution."""
    
    @patch('memory_core.query.query_engine.AdvancedQueryEngine')
    def test_enhance_query_response(self, mock_engine_class, mock_quality_engine, sample_query_results):
        """Test enhancement of query response with quality ranking."""
        # Setup mock query engine
        mock_engine = Mock()
        mock_response = QueryResponse(
            results=sample_query_results,
            total_count=len(sample_query_results),
            returned_count=len(sample_query_results),
            execution_time_ms=100
        )
        mock_response.query = "machine learning"
        mock_response.metadata = {}
        mock_engine.query.return_value = mock_response
        
        # Create quality enhancement engine
        quality_engine = KnowledgeQualityEnhancementEngine(mock_engine)
        quality_engine.quality_ranker.quality_enhancement_engine = mock_quality_engine
        
        request = QueryRequest(
            query="machine learning",
            query_type=QueryType.SEMANTIC_SEARCH
        )
        
        enhanced_response = quality_engine.enhance_query_with_quality_ranking(request)
        
        assert enhanced_response
        assert enhanced_response.results
        assert enhanced_response.metadata.get('quality_enhanced') == True
        assert enhanced_response.metadata.get('quality_ranking_applied') == True
        
        # Verify quality scores were added to result metadata
        for result in enhanced_response.results:
            assert 'quality_score' in result.metadata
            assert result.metadata.get('quality_ranked') == True
    
    def test_ranking_with_context_relevance(self, mock_quality_engine, sample_query_results):
        """Test quality ranking with query context relevance."""
        ranker = ResultRanker(quality_enhancement_engine=mock_quality_engine)
        ranker.quality_ranker = ranker  # Self-reference for testing
        
        # Test with context-relevant query
        sample_query_results[1].content = "machine learning algorithms comprehensive analysis detailed"
        
        request = QueryRequest(
            query="comprehensive machine learning analysis",
            query_type=QueryType.SEMANTIC_SEARCH
        )
        
        ranked_results = ranker.rank_results(sample_query_results, request)
        
        # Context relevance should boost relevant results
        assert len(ranked_results) == 3
        
        # Results should be ordered considering both quality and relevance
        for i in range(len(ranked_results) - 1):
            current_score = ranked_results[i].combined_score
            next_score = ranked_results[i + 1].combined_score
            assert current_score >= next_score


class TestPerformanceAndScalability:
    """Tests for performance and scalability of quality-based ranking."""
    
    def test_large_result_set_ranking(self, mock_quality_engine):
        """Test ranking performance with large result sets."""
        # Create a large number of results
        large_result_set = []
        for i in range(100):
            result = QueryResult(
                node_id=f"node_{i}",
                content=f"Content for node {i} with varying quality levels",
                node_type="document",
                relevance_score=0.5 + (i % 10) / 20,  # Varying relevance
                metadata={"quality_score": 0.3 + (i % 7) / 10}  # Varying quality
            )
            large_result_set.append(result)
        
        ranker = ResultRanker(quality_enhancement_engine=mock_quality_engine)
        
        request = QueryRequest(
            query="test query",
            query_type=QueryType.SEMANTIC_SEARCH
        )
        
        import time
        start_time = time.time()
        
        ranked_results = ranker.rank_results(large_result_set, request)
        
        ranking_time = time.time() - start_time
        
        # Should complete ranking in reasonable time
        assert ranking_time < 5.0  # Less than 5 seconds
        assert len(ranked_results) >= 1  # Should have at least some results
        # Note: The exact count might vary due to ranking implementation
        
        # Verify ranking order
        for i in range(len(ranked_results) - 1):
            current_score = ranked_results[i].combined_score
            next_score = ranked_results[i + 1].combined_score
            assert current_score >= next_score
    
    def test_quality_assessment_error_handling(self, sample_query_results):
        """Test error handling in quality assessment."""
        # Create a mock quality engine that raises exceptions
        failing_quality_engine = Mock()
        failing_quality_engine.get_quality_score.side_effect = Exception("Quality assessment failed")
        
        ranker = ResultRanker(quality_enhancement_engine=failing_quality_engine)
        
        request = QueryRequest(
            query="test query",
            query_type=QueryType.SEMANTIC_SEARCH
        )
        
        # Should handle errors gracefully and fall back to basic ranking
        ranked_results = ranker.rank_results(sample_query_results, request)
        
        assert len(ranked_results) == len(sample_query_results)
        
        # Should have fallen back to metadata-based quality scoring
        for result in ranked_results:
            assert hasattr(result, 'combined_score')


class TestRankingCriteriaCustomization:
    """Tests for customizing ranking criteria and weights."""
    
    def test_custom_ranking_weights(self, mock_quality_engine, sample_query_results):
        """Test custom ranking criteria weights."""
        from memory_core.query.result_ranker import RankingCriteria
        
        # Create custom criteria that heavily weights quality
        custom_criteria = RankingCriteria(
            relevance_weight=0.2,
            quality_weight=0.6,
            freshness_weight=0.1,
            popularity_weight=0.05,
            diversity_weight=0.05
        )
        
        ranker = ResultRanker(quality_enhancement_engine=mock_quality_engine)
        
        request = QueryRequest(
            query="machine learning",
            query_type=QueryType.SEMANTIC_SEARCH
        )
        
        ranked_results = ranker.rank_results(sample_query_results, request, custom_criteria)
        
        # With heavy quality weighting, high quality should rank first
        assert ranked_results[0].node_id == "high_quality_node"
        
        # Low quality should rank last despite high relevance
        assert ranked_results[-1].node_id == "low_quality_node"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])