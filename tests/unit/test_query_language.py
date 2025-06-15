"""Unit tests for GraphQL-like query language."""

import pytest
from memory_core.orchestrator.query_language import (
    QueryBuilder,
    QueryValidator,
    QueryType,
    FilterOperator,
    GraphQLQuery,
    QueryFilter,
    SortOrder,
    QueryParseError,
    QueryValidationError
)


class TestQueryBuilder:
    """Test the query builder functionality."""
    
    def test_simple_node_query(self):
        """Test building a simple node query."""
        builder = QueryBuilder()
        query = (builder
                .query_type(QueryType.NODES)
                .select(['id', 'content', 'created_at'])
                .limit(10)
                .build())
        
        assert query.query_type == QueryType.NODES
        assert query.fields == ['id', 'content', 'created_at']
        assert query.limit == 10
        assert query.filters == []
    
    def test_query_with_filters(self):
        """Test building a query with filters."""
        builder = QueryBuilder()
        query = (builder
                .query_type(QueryType.NODES)
                .select(['id', 'content'])
                .filter('content', FilterOperator.CONTAINS, 'AI')
                .filter('created_at', FilterOperator.GT, '2024-01-01')
                .build())
        
        assert len(query.filters) == 2
        assert query.filters[0].field == 'content'
        assert query.filters[0].operator == FilterOperator.CONTAINS
        assert query.filters[0].value == 'AI'
        assert query.filters[1].field == 'created_at'
        assert query.filters[1].operator == FilterOperator.GT
    
    def test_relationship_query(self):
        """Test building a relationship query."""
        builder = QueryBuilder()
        query = (builder
                .query_type(QueryType.RELATIONSHIPS)
                .select(['source_id', 'target_id', 'relationship_type'])
                .filter('relationship_type', FilterOperator.EQ, 'references')
                .limit(50)
                .build())
        
        assert query.query_type == QueryType.RELATIONSHIPS
        assert 'relationship_type' in query.fields
        assert query.limit == 50
    
    def test_query_with_sorting(self):
        """Test building a query with sorting."""
        builder = QueryBuilder()
        query = (builder
                .query_type(QueryType.NODES)
                .select(['id', 'content', 'relevance_score'])
                .sort('relevance_score', SortOrder.DESC)
                .sort('created_at', SortOrder.ASC)
                .limit(20)
                .build())
        
        assert len(query.sort) == 2
        assert query.sort[0]['field'] == 'relevance_score'
        assert query.sort[0]['order'] == SortOrder.DESC
        assert query.sort[1]['field'] == 'created_at'
        assert query.sort[1]['order'] == SortOrder.ASC
    
    def test_query_with_offset(self):
        """Test building a query with pagination."""
        builder = QueryBuilder()
        query = (builder
                .query_type(QueryType.NODES)
                .select(['id', 'content'])
                .offset(100)
                .limit(20)
                .build())
        
        assert query.offset == 100
        assert query.limit == 20
    
    def test_aggregate_query(self):
        """Test building an aggregate query."""
        builder = QueryBuilder()
        query = (builder
                .query_type(QueryType.AGGREGATE)
                .aggregate('count', 'id', alias='total_nodes')
                .aggregate('avg', 'relevance_score', alias='avg_score')
                .group_by(['category', 'source'])
                .build())
        
        assert query.query_type == QueryType.AGGREGATE
        assert len(query.aggregations) == 2
        assert query.aggregations[0]['function'] == 'count'
        assert query.aggregations[0]['field'] == 'id'
        assert query.aggregations[0]['alias'] == 'total_nodes'
        assert query.group_by == ['category', 'source']
    
    def test_chain_filters_with_or(self):
        """Test building complex filter chains."""
        builder = QueryBuilder()
        query = (builder
                .query_type(QueryType.NODES)
                .select(['id', 'content'])
                .filter('content', FilterOperator.CONTAINS, 'machine learning')
                .or_filter('content', FilterOperator.CONTAINS, 'deep learning')
                .build())
        
        # This would need more complex filter structure in real implementation
        assert len(query.filters) >= 2


class TestQueryValidator:
    """Test query validation functionality."""
    
    def test_valid_node_query(self):
        """Test validation of a valid node query."""
        query = GraphQLQuery(
            query_type=QueryType.NODES,
            fields=['id', 'content'],
            filters=[
                QueryFilter(
                    field='content',
                    operator=FilterOperator.CONTAINS,
                    value='test'
                )
            ],
            limit=10
        )
        
        validator = QueryValidator()
        # Should not raise exception
        validator.validate(query)
    
    def test_invalid_query_type(self):
        """Test validation fails for invalid query type."""
        query = GraphQLQuery(
            query_type=None,  # Invalid
            fields=['id'],
            limit=10
        )
        
        validator = QueryValidator()
        with pytest.raises(QueryValidationError):
            validator.validate(query)
    
    def test_missing_fields(self):
        """Test validation fails when fields are missing."""
        query = GraphQLQuery(
            query_type=QueryType.NODES,
            fields=[],  # Empty fields
            limit=10
        )
        
        validator = QueryValidator()
        with pytest.raises(QueryValidationError):
            validator.validate(query)
    
    def test_invalid_limit(self):
        """Test validation of query limits."""
        query = GraphQLQuery(
            query_type=QueryType.NODES,
            fields=['id'],
            limit=-1  # Invalid negative limit
        )
        
        validator = QueryValidator()
        with pytest.raises(QueryValidationError):
            validator.validate(query)
    
    def test_excessive_limit(self):
        """Test validation fails for excessive limits."""
        query = GraphQLQuery(
            query_type=QueryType.NODES,
            fields=['id'],
            limit=10000  # Too high
        )
        
        validator = QueryValidator()
        with pytest.raises(QueryValidationError):
            validator.validate(query)
    
    def test_invalid_filter_operator(self):
        """Test validation of filter operators."""
        query = GraphQLQuery(
            query_type=QueryType.NODES,
            fields=['id'],
            filters=[
                QueryFilter(
                    field='content',
                    operator='INVALID',  # Not a valid operator
                    value='test'
                )
            ]
        )
        
        validator = QueryValidator()
        with pytest.raises(QueryValidationError):
            validator.validate(query)


class TestQueryParsing:
    """Test query parsing from string format."""
    
    def test_parse_simple_query(self):
        """Test parsing a simple string query."""
        query_str = """
        {
            nodes {
                id
                content
                created_at
            }
        }
        """
        
        from memory_core.orchestrator.query_language import parse_query
        
        query = parse_query(query_str)
        assert query.query_type == QueryType.NODES
        assert 'id' in query.fields
        assert 'content' in query.fields
        assert 'created_at' in query.fields
    
    def test_parse_query_with_filters(self):
        """Test parsing a query with filters."""
        query_str = """
        {
            nodes(
                filter: {
                    content: {contains: "AI"},
                    created_at: {gt: "2024-01-01"}
                },
                limit: 20
            ) {
                id
                content
            }
        }
        """
        
        from memory_core.orchestrator.query_language import parse_query
        
        query = parse_query(query_str)
        assert query.query_type == QueryType.NODES
        assert query.limit == 20
        assert len(query.filters) == 2
    
    def test_parse_invalid_syntax(self):
        """Test parsing fails for invalid syntax."""
        query_str = """
        {
            invalid syntax here
        }
        """
        
        from memory_core.orchestrator.query_language import parse_query
        
        with pytest.raises(QueryParseError):
            parse_query(query_str)


class TestQueryExecution:
    """Test query execution simulation."""
    
    @pytest.mark.asyncio
    async def test_execute_node_query(self):
        """Test executing a node query."""
        from memory_core.orchestrator.query_language import GraphQLQueryProcessor
        
        # Mock knowledge engine
        class MockEngine:
            async def search_nodes(self, **kwargs):
                return [
                    {'id': '1', 'content': 'Test node 1'},
                    {'id': '2', 'content': 'Test node 2'}
                ]
        
        processor = GraphQLQueryProcessor(MockEngine())
        
        query = GraphQLQuery(
            query_type=QueryType.NODES,
            fields=['id', 'content'],
            limit=10
        )
        
        results = await processor.execute(query)
        assert len(results) == 2
        assert results[0]['id'] == '1'
    
    @pytest.mark.asyncio
    async def test_execute_with_field_selection(self):
        """Test field selection in query execution."""
        from memory_core.orchestrator.query_language import GraphQLQueryProcessor
        
        # Mock engine with more fields
        class MockEngine:
            async def search_nodes(self, **kwargs):
                return [
                    {
                        'id': '1',
                        'content': 'Test content',
                        'metadata': {'author': 'test'},
                        'embedding': [0.1, 0.2, 0.3]
                    }
                ]
        
        processor = GraphQLQueryProcessor(MockEngine())
        
        # Query only specific fields
        query = GraphQLQuery(
            query_type=QueryType.NODES,
            fields=['id', 'content'],  # Not requesting metadata or embedding
            limit=10
        )
        
        results = await processor.execute(query)
        assert len(results) == 1
        assert 'id' in results[0]
        assert 'content' in results[0]
        assert 'metadata' not in results[0]  # Should be filtered out
        assert 'embedding' not in results[0]  # Should be filtered out