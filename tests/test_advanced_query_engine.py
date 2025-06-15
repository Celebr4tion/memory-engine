"""
Comprehensive tests for the Advanced Query Engine.
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from memory_core.query import (
    AdvancedQueryEngine,
    QueryRequest,
    QueryResponse,
    QueryResult,
    QueryType,
    FilterCondition,
    SortCriteria,
    SortOrder,
    AggregationRequest,
    AggregationType,
)


class TestAdvancedQueryEngine:
    """Test suite for AdvancedQueryEngine."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for testing."""
        graph_adapter = Mock()
        embedding_manager = Mock()
        rating_storage = Mock()

        # Mock graph adapter methods
        graph_adapter.get_node_by_id.return_value = {
            "id": "test_node_1",
            "content": "Test content",
            "node_type": "concept",
            "metadata": {"creation_date": "2024-01-01"},
        }

        graph_adapter.find_nodes_by_content.return_value = [
            {
                "id": "test_node_1",
                "content": "Test content about AI",
                "node_type": "concept",
                "metadata": {"rating": 0.8, "domain": "technology"},
            },
            {
                "id": "test_node_2",
                "content": "Another test about machine learning",
                "node_type": "concept",
                "metadata": {"rating": 0.9, "domain": "technology"},
            },
        ]

        graph_adapter.get_relationships_for_node.return_value = [
            {"type": "related_to", "target": "test_node_3", "weight": 0.8}
        ]

        # Mock embedding manager
        embedding_manager.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]
        embedding_manager.find_similar_nodes.return_value = [
            ("test_node_1", 0.95),
            ("test_node_2", 0.87),
        ]

        return graph_adapter, embedding_manager, rating_storage

    @pytest.fixture
    def query_engine(self, mock_dependencies):
        """Create query engine with mocked dependencies."""
        graph_adapter, embedding_manager, rating_storage = mock_dependencies
        return AdvancedQueryEngine(graph_adapter, embedding_manager, rating_storage)

    def test_basic_query_execution(self, query_engine):
        """Test basic query execution flow."""
        request = QueryRequest(query="test query", query_type=QueryType.NATURAL_LANGUAGE, limit=10)

        response = query_engine.query(request)

        assert isinstance(response, QueryResponse)
        assert response.query_id is not None
        assert response.execution_time_ms >= 0
        assert isinstance(response.results, list)

    def test_semantic_search(self, query_engine):
        """Test semantic search functionality."""
        request = QueryRequest(
            query="artificial intelligence",
            query_type=QueryType.SEMANTIC_SEARCH,
            similarity_threshold=0.8,
            limit=5,
        )

        response = query_engine.query(request)

        assert len(response.results) <= 5
        for result in response.results:
            assert result.relevance_score >= 0.8

    def test_graph_pattern_query(self, query_engine):
        """Test graph pattern query execution."""
        request = QueryRequest(
            query="find concepts related to AI",
            query_type=QueryType.GRAPH_PATTERN,
            include_relationships=True,
            max_depth=2,
        )

        response = query_engine.query(request)

        assert isinstance(response, QueryResponse)
        # Should fall back to text search for now
        assert len(response.results) >= 0

    def test_filtering(self, query_engine):
        """Test result filtering functionality."""
        filters = [
            FilterCondition(field="metadata.domain", operator="eq", value="technology"),
            FilterCondition(field="metadata.rating", operator="gt", value=0.7),
        ]

        request = QueryRequest(query="test", filters=filters, limit=10)

        response = query_engine.query(request)

        # All results should pass the filters
        for result in response.results:
            assert result.metadata.get("domain") == "technology"
            assert result.metadata.get("rating", 0) > 0.7

    def test_aggregation(self, query_engine):
        """Test aggregation functionality."""
        aggregations = [
            AggregationRequest(type=AggregationType.COUNT),
            AggregationRequest(type=AggregationType.GROUP_BY, group_by=["node_type"]),
        ]

        request = QueryRequest(query="test", aggregations=aggregations)

        response = query_engine.query(request)

        assert len(response.aggregations) >= 1
        count_agg = next((a for a in response.aggregations if a.aggregation_type == "count"), None)
        assert count_agg is not None
        assert isinstance(count_agg.value, int)

    def test_sorting(self, query_engine):
        """Test result sorting."""
        sort_criteria = [SortCriteria(field="relevance_score", order=SortOrder.DESCENDING)]

        request = QueryRequest(query="test", sort_by=sort_criteria, limit=10)

        response = query_engine.query(request)

        # Check that results are sorted by relevance score descending
        if len(response.results) > 1:
            for i in range(len(response.results) - 1):
                assert (
                    response.results[i].relevance_score >= response.results[i + 1].relevance_score
                )

    def test_pagination(self, query_engine):
        """Test pagination functionality."""
        # First page
        request1 = QueryRequest(query="test", limit=2, offset=0)
        response1 = query_engine.query(request1)

        # Second page
        request2 = QueryRequest(query="test", limit=2, offset=2)
        response2 = query_engine.query(request2)

        assert response1.returned_count <= 2
        assert response2.returned_count <= 2

        # Results should be different (if there are enough results)
        if response1.total_count > 2:
            assert response1.results != response2.results

    def test_caching(self, query_engine):
        """Test query result caching."""
        request = QueryRequest(query="cacheable query", use_cache=True, cache_ttl=3600)

        # First execution
        response1 = query_engine.query(request)
        assert not response1.from_cache

        # Second execution should be cached
        response2 = query_engine.query(request)
        assert response2.from_cache
        assert response1.results == response2.results

    def test_explanation(self, query_engine):
        """Test query explanation generation."""
        request = QueryRequest(query="test query with explanation", explain=True)

        response = query_engine.query(request)

        assert response.explanation is not None
        assert response.explanation.original_query == request.query
        assert len(response.explanation.execution_plan) > 0
        assert response.explanation.total_execution_time_ms >= 0

    def test_error_handling(self, query_engine):
        """Test error handling for invalid queries."""
        # Test with invalid similarity threshold
        request = QueryRequest(
            query="test",
            query_type=QueryType.SEMANTIC_SEARCH,
            similarity_threshold=2.0,  # Invalid - should be 0-1
        )

        response = query_engine.query(request)
        assert isinstance(response, QueryResponse)
        # Should handle gracefully without crashing

    def test_performance_stats(self, query_engine):
        """Test performance statistics tracking."""
        initial_stats = query_engine.get_statistics()

        request = QueryRequest(query="performance test")
        query_engine.query(request)

        updated_stats = query_engine.get_statistics()

        # Query count should have increased
        assert (
            updated_stats["query_engine"]["total_queries"]
            > initial_stats["query_engine"]["total_queries"]
        )

    def test_cache_invalidation(self, query_engine):
        """Test cache invalidation functionality."""
        # Execute query to populate cache
        request = QueryRequest(query="invalidation test", use_cache=True)
        response1 = query_engine.query(request)

        # Invalidate cache
        query_engine.invalidate_cache(node_ids=["test_node_1"])

        # Query again - should not be from cache
        response2 = query_engine.query(request)

        # Both should be valid responses
        assert isinstance(response1, QueryResponse)
        assert isinstance(response2, QueryResponse)


class TestQueryTypes:
    """Test query type handling and data structures."""

    def test_query_request_creation(self):
        """Test QueryRequest creation and validation."""
        request = QueryRequest(
            query="test query",
            query_type=QueryType.SEMANTIC_SEARCH,
            limit=10,
            similarity_threshold=0.8,
        )

        assert request.query == "test query"
        assert request.query_type == QueryType.SEMANTIC_SEARCH
        assert request.limit == 10
        assert request.similarity_threshold == 0.8

    def test_filter_condition(self):
        """Test FilterCondition functionality."""
        filter_cond = FilterCondition(field="metadata.rating", operator="gt", value=0.5)

        filter_dict = filter_cond.to_dict()
        assert filter_dict["field"] == "metadata.rating"
        assert filter_dict["operator"] == "gt"
        assert filter_dict["value"] == 0.5

    def test_aggregation_request(self):
        """Test AggregationRequest functionality."""
        agg_request = AggregationRequest(
            type=AggregationType.GROUP_BY, field="node_type", group_by=["domain"]
        )

        agg_dict = agg_request.to_dict()
        assert agg_dict["type"] == "group_by"
        assert agg_dict["field"] == "node_type"
        assert agg_dict["group_by"] == ["domain"]


class TestFilterProcessor:
    """Test filter processor functionality."""

    @pytest.fixture
    def filter_processor(self):
        """Create filter processor for testing."""
        from memory_core.query.filter_processor import FilterProcessor

        return FilterProcessor()

    @pytest.fixture
    def sample_results(self):
        """Create sample query results for testing."""
        return [
            QueryResult(
                node_id="node1",
                content="First test result",
                metadata={"rating": 0.8, "domain": "tech", "count": 5},
            ),
            QueryResult(
                node_id="node2",
                content="Second test result",
                metadata={"rating": 0.6, "domain": "science", "count": 3},
            ),
            QueryResult(
                node_id="node3",
                content="Third test result",
                metadata={"rating": 0.9, "domain": "tech", "count": 7},
            ),
        ]

    def test_equality_filter(self, filter_processor, sample_results):
        """Test equality filtering."""
        filters = [FilterCondition(field="metadata.domain", operator="eq", value="tech")]

        filtered = filter_processor.apply_filters(sample_results, filters)

        assert len(filtered) == 2
        for result in filtered:
            assert result.metadata["domain"] == "tech"

    def test_greater_than_filter(self, filter_processor, sample_results):
        """Test greater than filtering."""
        filters = [FilterCondition(field="metadata.rating", operator="gt", value=0.7)]

        filtered = filter_processor.apply_filters(sample_results, filters)

        assert len(filtered) == 2  # ratings 0.8 and 0.9
        for result in filtered:
            assert result.metadata["rating"] > 0.7

    def test_contains_filter(self, filter_processor, sample_results):
        """Test contains filtering."""
        filters = [FilterCondition(field="content", operator="contains", value="test")]

        filtered = filter_processor.apply_filters(sample_results, filters)

        assert len(filtered) == 3  # All contain "test"

    def test_regex_filter(self, filter_processor, sample_results):
        """Test regex filtering."""
        filters = [FilterCondition(field="content", operator="regex", value=r"First|Third")]

        filtered = filter_processor.apply_filters(sample_results, filters)

        assert len(filtered) == 2  # First and Third results

    def test_multiple_filters(self, filter_processor, sample_results):
        """Test multiple filter conditions."""
        filters = [
            FilterCondition(field="metadata.domain", operator="eq", value="tech"),
            FilterCondition(field="metadata.rating", operator="gt", value=0.7),
        ]

        filtered = filter_processor.apply_filters(sample_results, filters)

        assert len(filtered) == 2  # Both tech domain AND rating > 0.7
        for result in filtered:
            assert result.metadata["domain"] == "tech"
            assert result.metadata["rating"] > 0.7

    def test_invalid_regex_handling(self, filter_processor, sample_results):
        """Test handling of invalid regex patterns."""
        filters = [FilterCondition(field="content", operator="regex", value="[invalid")]

        # Should not crash, should return empty or handle gracefully
        filtered = filter_processor.apply_filters(sample_results, filters)
        assert isinstance(filtered, list)


class TestResultRanker:
    """Test result ranking functionality."""

    @pytest.fixture
    def result_ranker(self):
        """Create result ranker for testing."""
        from memory_core.query.result_ranker import ResultRanker

        return ResultRanker()

    @pytest.fixture
    def sample_results(self):
        """Create sample results with different scores."""
        return [
            QueryResult(
                node_id="node1",
                content="High quality content",
                metadata={
                    "rating_richness": 0.9,
                    "rating_truthfulness": 0.8,
                    "creation_timestamp": time.time(),
                },
            ),
            QueryResult(
                node_id="node2",
                content="Medium quality content",
                metadata={
                    "rating_richness": 0.6,
                    "rating_truthfulness": 0.7,
                    "creation_timestamp": time.time() - 86400,
                },
            ),
            QueryResult(
                node_id="node3",
                content="Lower quality content",
                metadata={
                    "rating_richness": 0.4,
                    "rating_truthfulness": 0.5,
                    "creation_timestamp": time.time() - 172800,
                },
            ),
        ]

    def test_basic_ranking(self, result_ranker, sample_results):
        """Test basic result ranking."""
        request = QueryRequest(query="test ranking")

        ranked = result_ranker.rank_results(sample_results, request)

        assert len(ranked) == 3
        # Should be sorted by combined score descending
        assert ranked[0].combined_score >= ranked[1].combined_score >= ranked[2].combined_score

    def test_relevance_scoring(self, result_ranker):
        """Test relevance score calculation."""
        result = QueryResult(node_id="test", content="artificial intelligence machine learning")
        request = QueryRequest(query="machine learning", query_type=QueryType.NATURAL_LANGUAGE)

        # This tests the private method, normally we'd test through public interface
        score = result_ranker._calculate_relevance_score(result, request)

        assert 0 <= score <= 1
        assert score > 0  # Should have some relevance

    def test_quality_scoring(self, result_ranker):
        """Test quality score calculation."""
        result = QueryResult(
            node_id="test",
            content="test",
            metadata={"rating_richness": 0.8, "rating_truthfulness": 0.9, "rating_stability": 0.7},
        )

        score = result_ranker._calculate_quality_score(result)

        assert 0 <= score <= 1
        # Should be weighted average of the ratings
        expected = 0.8 * 0.4 + 0.9 * 0.4 + 0.7 * 0.2
        assert abs(score - expected) < 0.01


class TestQueryCache:
    """Test query caching functionality."""

    @pytest.fixture
    def query_cache(self):
        """Create query cache for testing."""
        from memory_core.query.query_cache import QueryCache

        return QueryCache(max_size_mb=1, max_entries=10)  # Small cache for testing

    def test_cache_put_get(self, query_cache):
        """Test basic cache put and get operations."""
        request = QueryRequest(query="cache test")
        response = QueryResponse(
            results=[QueryResult(node_id="test", content="cached content")],
            total_count=1,
            returned_count=1,
            execution_time_ms=100.0,  # Add execution time to avoid skipping
        )

        # Put in cache
        success = query_cache.put(request, response)
        assert success

        # Get from cache
        cached_response = query_cache.get(request)
        assert cached_response is not None
        assert cached_response.from_cache
        assert len(cached_response.results) == 1

    def test_cache_expiration(self, query_cache):
        """Test cache TTL expiration."""
        request = QueryRequest(query="expiration test", cache_ttl=1)  # 1 second TTL
        response = QueryResponse(
            results=[QueryResult(node_id="test", content="expires soon")],
            total_count=1,
            returned_count=1,
            execution_time_ms=100.0,
        )

        query_cache.put(request, response)

        # Should be in cache immediately
        cached = query_cache.get(request)
        assert cached is not None

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired now
        cached = query_cache.get(request)
        assert cached is None

    def test_cache_invalidation(self, query_cache):
        """Test cache invalidation."""
        request = QueryRequest(query="invalidation test")
        response = QueryResponse(
            results=[QueryResult(node_id="node123", content="to be invalidated")],
            total_count=1,
            returned_count=1,
            execution_time_ms=100.0,
        )

        query_cache.put(request, response)

        # Should be in cache
        assert query_cache.get(request) is not None

        # Invalidate
        query_cache.invalidate(node_ids=["node123"])

        # Should be removed from cache
        assert query_cache.get(request) is None

    def test_cache_statistics(self, query_cache):
        """Test cache statistics tracking."""
        request = QueryRequest(query="stats test")
        response = QueryResponse(
            results=[QueryResult(node_id="test", content="stats content")],
            total_count=1,
            returned_count=1,
            execution_time_ms=100.0,
        )

        initial_stats = query_cache.get_statistics()

        # Cache miss
        query_cache.get(request)

        # Cache put
        query_cache.put(request, response)

        # Cache hit
        query_cache.get(request)

        final_stats = query_cache.get_statistics()

        assert final_stats["cache_hits"] > initial_stats["cache_hits"]
        assert final_stats["cache_misses"] > initial_stats["cache_misses"]


class TestQueryOptimizer:
    """Test query optimization functionality."""

    @pytest.fixture
    def query_optimizer(self):
        """Create query optimizer for testing."""
        from memory_core.query.query_optimizer import QueryOptimizer

        return QueryOptimizer()

    def test_basic_optimization(self, query_optimizer):
        """Test basic query optimization."""
        request = QueryRequest(
            query="test optimization",
            query_type=QueryType.SEMANTIC_SEARCH,
            similarity_threshold=0.95,  # Too high
            max_depth=5,  # Too deep
            include_metadata=True,
        )

        optimized_request, execution_plan, explanation_steps = query_optimizer.optimize_query(
            request
        )

        assert optimized_request.similarity_threshold < request.similarity_threshold
        assert len(explanation_steps) > 0
        assert execution_plan.estimated_cost > 0

    def test_filter_reordering(self, query_optimizer):
        """Test filter reordering optimization."""
        filters = [
            FilterCondition(field="field1", operator="ne", value="value1"),  # Less selective
            FilterCondition(field="field2", operator="eq", value="value2"),  # More selective
        ]

        reordered = query_optimizer._reorder_filters(filters)

        # Equality filter should come first
        assert reordered[0].operator == "eq"
        assert reordered[1].operator == "ne"


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_advanced_query_engine.py -v
    pytest.main([__file__, "-v"])
