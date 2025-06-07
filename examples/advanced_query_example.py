"""
Advanced Query Engine Example

Demonstrates the full capabilities of the Memory Engine's Advanced Query Engine,
including natural language processing, filtering, aggregations, caching, and explanations.
"""

import asyncio
from datetime import datetime
from memory_core.query import (
    AdvancedQueryEngine, QueryRequest, QueryType, 
    FilterCondition, SortCriteria, SortOrder, 
    AggregationRequest, AggregationType
)
from memory_core.db.graph_storage_adapter import GraphStorageAdapter
from memory_core.embeddings.embedding_manager import EmbeddingManager


async def main():
    """Demonstrate Advanced Query Engine capabilities."""
    
    print("🚀 Advanced Query Engine Example")
    print("=" * 50)
    
    # Initialize components (mock for example)
    # In real usage, these would be properly configured
    graph_adapter = MockGraphAdapter()
    embedding_manager = MockEmbeddingManager()
    
    # Create the Advanced Query Engine
    query_engine = AdvancedQueryEngine(
        graph_adapter=graph_adapter,
        embedding_manager=embedding_manager,
        rating_storage=None
    )
    
    print("\n✨ Query Engine initialized successfully!")
    
    # Example 1: Natural Language Query
    print("\n" + "="*50)
    print("📝 Example 1: Natural Language Query")
    print("="*50)
    
    nl_request = QueryRequest(
        query="Find concepts related to machine learning with high quality ratings",
        query_type=QueryType.NATURAL_LANGUAGE,
        limit=10,
        explain=True
    )
    
    print(f"Query: {nl_request.query}")
    nl_response = query_engine.query(nl_request)
    
    print(f"Results found: {nl_response.total_count}")
    print(f"Execution time: {nl_response.execution_time_ms:.2f}ms")
    print(f"From cache: {nl_response.from_cache}")
    
    if nl_response.explanation:
        print(f"Query steps: {len(nl_response.explanation.execution_plan)}")
        for i, step in enumerate(nl_response.explanation.execution_plan[:3]):
            print(f"  {i+1}. {step.description} ({step.execution_time_ms:.1f}ms)")
    
    # Example 2: Semantic Search with Filters
    print("\n" + "="*50)
    print("🔍 Example 2: Semantic Search with Filters")
    print("="*50)
    
    semantic_request = QueryRequest(
        query="artificial intelligence deep learning",
        query_type=QueryType.SEMANTIC_SEARCH,
        similarity_threshold=0.8,
        filters=[
            FilterCondition(field="metadata.domain", operator="eq", value="technology"),
            FilterCondition(field="metadata.rating", operator="gt", value=0.7)
        ],
        sort_by=[
            SortCriteria(field="relevance_score", order=SortOrder.DESCENDING)
        ],
        limit=5
    )
    
    print(f"Query: {semantic_request.query}")
    print(f"Similarity threshold: {semantic_request.similarity_threshold}")
    print(f"Filters: {len(semantic_request.filters)}")
    
    semantic_response = query_engine.query(semantic_request)
    
    print(f"Results found: {semantic_response.returned_count}")
    print(f"Execution time: {semantic_response.execution_time_ms:.2f}ms")
    
    for i, result in enumerate(semantic_response.results[:3]):
        print(f"  {i+1}. {result.content[:50]}... (score: {result.relevance_score:.3f})")
    
    # Example 3: Complex Aggregation Query
    print("\n" + "="*50)
    print("📊 Example 3: Aggregation Query")
    print("="*50)
    
    agg_request = QueryRequest(
        query="programming languages",
        aggregations=[
            AggregationRequest(type=AggregationType.COUNT),
            AggregationRequest(
                type=AggregationType.GROUP_BY, 
                group_by=["node_type"]
            ),
            AggregationRequest(
                type=AggregationType.AVERAGE,
                field="metadata.rating"
            )
        ]
    )
    
    print(f"Query: {agg_request.query}")
    print(f"Aggregations: {len(agg_request.aggregations)}")
    
    agg_response = query_engine.query(agg_request)
    
    print(f"Aggregation results: {len(agg_response.aggregations)}")
    for agg in agg_response.aggregations:
        if agg.group_key:
            print(f"  {agg.aggregation_type} by {agg.field}: {agg.group_key} = {agg.value}")
        else:
            print(f"  {agg.aggregation_type}: {agg.value}")
    
    # Example 4: Advanced Filtering with Multiple Operators
    print("\n" + "="*50)
    print("🎯 Example 4: Advanced Filtering")
    print("="*50)
    
    filter_request = QueryRequest(
        query="software development",
        filters=[
            FilterCondition(field="content", operator="contains", value="Python"),
            FilterCondition(field="metadata.creation_date", operator="gt", value="2023-01-01"),
            FilterCondition(field="metadata.tags", operator="in", value=["programming", "tutorial"]),
            FilterCondition(field="metadata.rating", operator="between", value=[0.5, 1.0])
        ],
        sort_by=[
            SortCriteria(field="metadata.rating", order=SortOrder.DESCENDING),
            SortCriteria(field="relevance_score", order=SortOrder.DESCENDING)
        ],
        limit=15
    )
    
    print(f"Query: {filter_request.query}")
    print(f"Applied filters:")
    for f in filter_request.filters:
        print(f"  - {f.field} {f.operator} {f.value}")
    
    filter_response = query_engine.query(filter_request)
    
    print(f"Results found: {filter_response.returned_count}")
    print(f"Total matching: {filter_response.total_count}")
    
    # Example 5: Cache Performance Demo
    print("\n" + "="*50)
    print("⚡ Example 5: Cache Performance")
    print("="*50)
    
    cache_request = QueryRequest(
        query="cached query example",
        use_cache=True,
        cache_ttl=3600
    )
    
    # First execution (cache miss)
    print("First execution (cache miss):")
    start_time = datetime.now()
    response1 = query_engine.query(cache_request)
    end_time = datetime.now()
    first_duration = (end_time - start_time).total_seconds() * 1000
    
    print(f"  Time: {first_duration:.2f}ms")
    print(f"  From cache: {response1.from_cache}")
    
    # Second execution (cache hit)
    print("Second execution (cache hit):")
    start_time = datetime.now()
    response2 = query_engine.query(cache_request)
    end_time = datetime.now()
    second_duration = (end_time - start_time).total_seconds() * 1000
    
    print(f"  Time: {second_duration:.2f}ms")
    print(f"  From cache: {response2.from_cache}")
    print(f"  Speed improvement: {(first_duration/second_duration):.1f}x faster")
    
    # Example 6: Query Explanation
    print("\n" + "="*50)
    print("🔍 Example 6: Query Explanation")
    print("="*50)
    
    explain_request = QueryRequest(
        query="machine learning algorithms with examples",
        query_type=QueryType.NATURAL_LANGUAGE,
        explain=True,
        include_relationships=True,
        max_depth=2
    )
    
    explain_response = query_engine.query(explain_request)
    
    if explain_response.explanation:
        explanation = explain_response.explanation
        print(f"Original query: {explanation.original_query}")
        print(f"Total execution time: {explanation.total_execution_time_ms:.2f}ms")
        print(f"Optimizations applied: {len(explanation.optimizations)}")
        
        print("\nExecution plan:")
        for i, step in enumerate(explanation.execution_plan):
            print(f"  {i+1}. {step.step_name}: {step.description}")
            if step.execution_time_ms:
                print(f"     Time: {step.execution_time_ms:.2f}ms")
            if step.optimizations_applied:
                print(f"     Optimizations: {', '.join(step.optimizations_applied)}")
    
    # Example 7: Performance Statistics
    print("\n" + "="*50)
    print("📈 Example 7: Performance Statistics")
    print("="*50)
    
    stats = query_engine.get_statistics()
    
    print("Query Engine Statistics:")
    print(f"  Total queries: {stats['query_engine']['total_queries']}")
    print(f"  Cache hit rate: {stats['cache']['hit_rate']:.1%}")
    print(f"  Average execution time: {stats['query_engine']['average_execution_time_ms']:.2f}ms")
    
    if 'ranking' in stats:
        print(f"  Results ranked: {stats['ranking'].get('total_queries_ranked', 0)}")
    
    if 'optimization' in stats:
        print(f"  Queries optimized: {stats['optimization'].get('total_optimized_queries', 0)}")
    
    print("\n✅ Advanced Query Engine examples completed!")
    print("\nKey Features Demonstrated:")
    print("  ✓ Natural language query processing")
    print("  ✓ Semantic search with similarity thresholds")
    print("  ✓ Complex filtering with multiple operators")
    print("  ✓ Aggregations and grouping")
    print("  ✓ Result ranking and sorting")
    print("  ✓ Intelligent caching for performance")
    print("  ✓ Query optimization and explanation")
    print("  ✓ Performance monitoring and statistics")


class MockGraphAdapter:
    """Mock graph adapter for demonstration."""
    
    def get_node_by_id(self, node_id):
        return {
            'id': node_id,
            'content': f'Sample content for node {node_id}',
            'node_type': 'concept',
            'metadata': {
                'rating': 0.85,
                'domain': 'technology',
                'creation_date': '2024-01-15',
                'tags': ['programming', 'tutorial']
            }
        }
    
    def find_nodes_by_content(self, query, limit=50):
        """Return sample nodes for demonstration."""
        return [
            {
                'id': f'node_{i}',
                'content': f'Sample content about {query} - result {i}',
                'node_type': 'concept',
                'metadata': {
                    'rating': 0.7 + (i * 0.05),
                    'domain': 'technology',
                    'creation_date': '2024-01-15',
                    'tags': ['programming', 'example']
                }
            }
            for i in range(min(limit, 20))
        ]
    
    def get_relationships_for_node(self, node_id, max_depth=3):
        return [
            {'type': 'related_to', 'target': f'related_{node_id}', 'weight': 0.8}
        ]


class MockEmbeddingManager:
    """Mock embedding manager for demonstration."""
    
    def generate_embeddings(self, texts):
        return [[0.1 * i, 0.2 * i, 0.3 * i] for i in range(len(texts))]
    
    def find_similar_nodes(self, query_embedding, top_k=50, similarity_threshold=0.7):
        return [
            (f'node_{i}', 0.9 - (i * 0.05))
            for i in range(min(top_k, 15))
            if 0.9 - (i * 0.05) >= similarity_threshold
        ]


if __name__ == "__main__":
    asyncio.run(main())