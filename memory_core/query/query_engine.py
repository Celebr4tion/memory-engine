"""
Advanced Query Engine for Memory Engine

Main query engine that orchestrates natural language processing, query optimization,
result ranking, caching, and explanation generation.
"""

import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from memory_core.db.graph_storage_adapter import GraphStorageAdapter
from memory_core.embeddings.embedding_manager import EmbeddingManager
from memory_core.rating import rating_system

from .query_types import (
    QueryRequest, QueryResponse, QueryResult, QueryExplanation, 
    QueryExplanationStep, QueryType, QueryStatistics, AggregationResult
)
from .natural_language_processor import NaturalLanguageQueryProcessor
from .query_optimizer import QueryOptimizer
from .result_ranker import ResultRanker
from .query_cache import QueryCache
from .filter_processor import FilterProcessor, AggregationProcessor
from .query_explainer import QueryExplainer


class AdvancedQueryEngine:
    """
    Advanced query engine providing sophisticated graph query capabilities.
    
    Features:
    - Natural language to graph query translation
    - Query optimization for common patterns
    - Result ranking by relevance and quality scores
    - Query result caching for performance
    - Support for complex filters and aggregations
    - Query explanation for transparency
    """
    
    def __init__(self, 
                 graph_adapter: GraphStorageAdapter,
                 embedding_manager: EmbeddingManager,
                 rating_storage=None,
                 quality_enhancement_engine=None):
        """
        Initialize the advanced query engine.
        
        Args:
            graph_adapter: Graph storage adapter for data access
            embedding_manager: Embedding manager for semantic searches
            rating_storage: Rating storage for quality scoring (optional)
            quality_enhancement_engine: Quality enhancement engine for advanced quality scoring (optional)
        """
        self.logger = logging.getLogger(__name__)
        
        # Core dependencies
        self.graph_adapter = graph_adapter
        self.embedding_manager = embedding_manager
        self.rating_storage = rating_storage
        self.quality_enhancement_engine = quality_enhancement_engine
        
        # Query processing components
        self.nlp_processor = NaturalLanguageQueryProcessor()
        self.optimizer = QueryOptimizer()
        self.ranker = ResultRanker(quality_enhancement_engine)
        self.cache = QueryCache()
        self.filter_processor = FilterProcessor()
        self.aggregation_processor = AggregationProcessor(self.filter_processor)
        self.explainer = QueryExplainer()
        
        # Statistics
        self.stats = QueryStatistics()
        
        self.logger.info("Advanced Query Engine initialized")
    
    def query(self, request: QueryRequest) -> QueryResponse:
        """
        Execute a query request and return results.
        
        Args:
            request: Query request to execute
            
        Returns:
            Query response with results and metadata
        """
        start_time = time.time()
        query_id = str(uuid.uuid4())
        
        self.logger.info(f"Executing query [{query_id}]: {request.query[:100]}...")
        
        try:
            # Step 1: Check cache first
            cached_response = self.cache.get(request)
            if cached_response:
                cached_response.query_id = query_id
                self.stats.cache_hits += 1
                self.logger.info(f"Query [{query_id}] served from cache")
                return cached_response
            
            self.stats.cache_misses += 1
            
            # Step 2: Process natural language query
            explanation_steps = []
            
            if request.query_type == QueryType.NATURAL_LANGUAGE:
                nlp_start = time.time()
                parsed_query = self.nlp_processor.process_query(request.query, request.context)
                
                # Update request based on NLP results
                request = self._enhance_request_with_nlp(request, parsed_query)
                
                nlp_step = QueryExplanationStep(
                    step_name="natural_language_processing",
                    description="Parse natural language query",
                    operation="nlp_parse",
                    execution_time_ms=(time.time() - nlp_start) * 1000,
                    details={
                        "intent": parsed_query.intent,
                        "entities": parsed_query.entities,
                        "confidence": parsed_query.confidence,
                        "detected_type": parsed_query.query_type.value
                    }
                )
                explanation_steps.append(nlp_step)
            
            # Step 3: Query optimization
            opt_start = time.time()
            optimized_request, execution_plan, opt_steps = self.optimizer.optimize_query(request)
            explanation_steps.extend(opt_steps)
            
            # Step 4: Execute query
            exec_start = time.time()
            raw_results = self._execute_query(optimized_request, execution_plan)
            
            exec_step = QueryExplanationStep(
                step_name="query_execution",
                description="Execute optimized query against graph",
                operation="graph_query",
                input_size=None,
                output_size=len(raw_results),
                execution_time_ms=(time.time() - exec_start) * 1000,
                details={"results_found": len(raw_results)}
            )
            explanation_steps.append(exec_step)
            
            # Step 4.5: Apply filters
            if optimized_request.filters:
                filter_start = time.time()
                pre_filter_count = len(raw_results)
                raw_results = self.filter_processor.apply_filters(raw_results, optimized_request.filters)
                
                filter_step = QueryExplanationStep(
                    step_name="filter_application",
                    description=f"Apply {len(optimized_request.filters)} filters",
                    operation="filter",
                    input_size=pre_filter_count,
                    output_size=len(raw_results),
                    execution_time_ms=(time.time() - filter_start) * 1000,
                    details={"filters_applied": len(optimized_request.filters)}
                )
                explanation_steps.append(filter_step)
            
            # Step 5: Rank results
            rank_start = time.time()
            ranked_results = self.ranker.rank_results(raw_results, optimized_request)
            
            rank_step = QueryExplanationStep(
                step_name="result_ranking",
                description="Rank results by relevance and quality",
                operation="ranking",
                input_size=len(raw_results),
                output_size=len(ranked_results),
                execution_time_ms=(time.time() - rank_start) * 1000,
                details={
                    "ranking_criteria": "relevance + quality + freshness",
                    "top_score": ranked_results[0].combined_score if ranked_results else 0
                }
            )
            explanation_steps.append(rank_step)
            
            # Step 6: Apply pagination
            paginated_results, total_count = self._apply_pagination(ranked_results, optimized_request)
            
            # Step 7: Process aggregations if requested
            aggregations = []
            if optimized_request.aggregations:
                agg_start = time.time()
                aggregations = self.aggregation_processor.process_aggregations(
                    ranked_results, optimized_request.aggregations
                )
                
                agg_step = QueryExplanationStep(
                    step_name="aggregation",
                    description=f"Compute {len(optimized_request.aggregations)} aggregations",
                    operation="aggregate",
                    execution_time_ms=(time.time() - agg_start) * 1000,
                    details={"aggregation_count": len(aggregations)}
                )
                explanation_steps.append(agg_step)
            
            # Step 8: Create response
            total_time = (time.time() - start_time) * 1000
            
            response = QueryResponse(
                results=paginated_results,
                total_count=total_count,
                returned_count=len(paginated_results),
                aggregations=aggregations,
                execution_time_ms=total_time,
                from_cache=False,
                query_id=query_id,
                timestamp=datetime.now(),
                has_more=(optimized_request.offset + len(paginated_results)) < total_count,
                next_offset=optimized_request.offset + len(paginated_results) if 
                          (optimized_request.offset + len(paginated_results)) < total_count else None
            )
            
            # Step 9: Add explanation if requested
            if optimized_request.explain:
                response.explanation = self.explainer.generate_explanation(
                    request=request,
                    response=response,
                    execution_steps=explanation_steps
                )
            
            # Step 10: Cache the result
            self.cache.put(request, response)
            
            # Update statistics
            self.stats.total_queries += 1
            self._update_performance_stats(total_time)
            
            self.logger.info(f"Query [{query_id}] completed in {total_time:.2f}ms, {len(paginated_results)} results")
            return response
            
        except ConnectionError as e:
            self.logger.error(f"Query [{query_id}] failed - Database connection error: {e}")
            return self._create_error_response("Database connection failed", query_id, start_time, str(e))
        except TimeoutError as e:
            self.logger.error(f"Query [{query_id}] failed - Query timeout: {e}")
            return self._create_error_response("Query execution timeout", query_id, start_time, str(e))
        except ValueError as e:
            self.logger.error(f"Query [{query_id}] failed - Invalid query parameters: {e}")
            return self._create_error_response("Invalid query parameters", query_id, start_time, str(e))
        except Exception as e:
            self.logger.error(f"Query [{query_id}] failed - Unexpected error: {e}")
            return self._create_error_response("Internal server error", query_id, start_time, str(e))
    
    def _enhance_request_with_nlp(self, request: QueryRequest, parsed_query) -> QueryRequest:
        """
        Enhance query request with NLP parsing results.
        
        Args:
            request: Original request
            parsed_query: Parsed query from NLP
            
        Returns:
            Enhanced request
        """
        # Update query type if NLP detected a more specific type
        if parsed_query.query_type != QueryType.NATURAL_LANGUAGE:
            request.query_type = parsed_query.query_type
        
        # Add extracted filters
        if hasattr(parsed_query, 'filters') and parsed_query.filters:
            request.filters.extend(parsed_query.filters)
        
        # Update similarity threshold for semantic searches
        if parsed_query.query_type == QueryType.SEMANTIC_SEARCH:
            if not request.similarity_threshold or request.similarity_threshold == 0.7:  # Default
                request.similarity_threshold = 0.75  # Slightly higher for NLP queries
        
        return request
    
    def _execute_query(self, request: QueryRequest, execution_plan) -> List[QueryResult]:
        """
        Execute the actual query against the graph database.
        
        Args:
            request: Optimized query request
            execution_plan: Query execution plan
            
        Returns:
            List of raw query results
        """
        results = []
        
        try:
            if request.query_type == QueryType.SEMANTIC_SEARCH:
                results = self._execute_semantic_search(request)
            elif request.query_type == QueryType.GRAPH_PATTERN:
                results = self._execute_graph_pattern_query(request)
            elif request.query_type == QueryType.RELATIONSHIP_SEARCH:
                results = self._execute_relationship_search(request)
            elif request.query_type == QueryType.AGGREGATION:
                results = self._execute_aggregation_query(request)
            else:
                # Default to text search
                results = self._execute_text_search(request)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            return []
    
    def _execute_semantic_search(self, request: QueryRequest) -> List[QueryResult]:
        """Execute semantic similarity search."""
        try:
            # Get query embedding
            query_embedding = self.embedding_manager.generate_embeddings([request.query])[0]
            
            # Search for similar nodes
            similar_nodes = self.embedding_manager.find_similar_nodes(
                query_embedding,
                top_k=request.limit or 50,
                similarity_threshold=request.similarity_threshold
            )
            
            results = []
            for node_id, similarity_score in similar_nodes:
                # Get node details from graph
                node_data = self.graph_adapter.get_node_by_id(node_id)
                if node_data:
                    result = QueryResult(
                        node_id=node_id,
                        content=node_data.get('content', ''),
                        node_type=node_data.get('node_type'),
                        relevance_score=similarity_score,
                        metadata=node_data.get('metadata', {})
                    )
                    
                    # Add relationships if requested
                    if request.include_relationships:
                        relationships = self.graph_adapter.get_relationships_for_node(
                            node_id, max_depth=request.max_depth
                        )
                        result.relationships = relationships
                    
                    results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            return []
    
    def _execute_graph_pattern_query(self, request: QueryRequest) -> List[QueryResult]:
        """Execute graph pattern matching query."""
        # This would implement actual graph pattern matching
        # For now, fall back to text search
        return self._execute_text_search(request)
    
    def _execute_relationship_search(self, request: QueryRequest) -> List[QueryResult]:
        """Execute relationship-focused search."""
        try:
            # Extract entities from query for relationship search
            # This is a simplified implementation
            nodes = self.graph_adapter.find_nodes_by_content(
                request.query,
                limit=request.limit or 20
            )
            
            results = []
            for node in nodes:
                # Get relationships for each node
                relationships = self.graph_adapter.get_relationships_for_node(
                    node['id'], max_depth=request.max_depth
                )
                
                if relationships:  # Only include nodes with relationships
                    result = QueryResult(
                        node_id=node['id'],
                        content=node.get('content', ''),
                        node_type=node.get('node_type'),
                        metadata=node.get('metadata', {}),
                        relationships=relationships
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Relationship search failed: {e}")
            return []
    
    def _execute_aggregation_query(self, request: QueryRequest) -> List[QueryResult]:
        """Execute aggregation query."""
        # Aggregations are handled separately, return empty results
        return []
    
    def _execute_text_search(self, request: QueryRequest) -> List[QueryResult]:
        """Execute basic text search as fallback."""
        try:
            nodes = self.graph_adapter.find_nodes_by_content(
                request.query,
                limit=request.limit or 50
            )
            
            results = []
            for node in nodes:
                result = QueryResult(
                    node_id=node['id'],
                    content=node.get('content', ''),
                    node_type=node.get('node_type'),
                    metadata=node.get('metadata', {})
                )
                
                # Add relationships if requested
                if request.include_relationships:
                    relationships = self.graph_adapter.get_relationships_for_node(
                        node['id'], max_depth=request.max_depth
                    )
                    result.relationships = relationships
                
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Text search failed: {e}")
            return []
    
    def _apply_pagination(self, results: List[QueryResult], request: QueryRequest) -> Tuple[List[QueryResult], int]:
        """
        Apply pagination to results.
        
        Args:
            results: All results
            request: Query request with pagination params
            
        Returns:
            Tuple of (paginated_results, total_count)
        """
        total_count = len(results)
        
        if request.limit:
            start_idx = request.offset
            end_idx = start_idx + request.limit
            paginated_results = results[start_idx:end_idx]
        else:
            paginated_results = results[request.offset:] if request.offset else results
        
        return paginated_results, total_count
    
    def _create_error_response(self, error_message: str, query_id: str, start_time: float, error_details: str = "") -> QueryResponse:
        """
        Create an error response for failed queries.
        
        Args:
            error_message: Human-readable error message
            query_id: Query identifier
            start_time: Query start time
            error_details: Detailed error information
            
        Returns:
            Error response
        """
        return QueryResponse(
            results=[],
            total_count=0,
            returned_count=0,
            execution_time_ms=(time.time() - start_time) * 1000,
            query_id=query_id,
            timestamp=datetime.now(),
            explanation=QueryExplanation(
                original_query="",
                parsed_query={"error": error_message, "details": error_details},
                translation_steps=[f"Error: {error_message}"],
                execution_plan=[],
                optimizations=[],
                total_execution_time_ms=(time.time() - start_time) * 1000,
                cache_hit=False
            ) if error_details else None
        )
    
    
    def _update_performance_stats(self, execution_time_ms: float):
        """
        Update performance statistics.
        
        Args:
            execution_time_ms: Query execution time in milliseconds
        """
        # Update average execution time
        total_time = self.stats.average_execution_time_ms * (self.stats.total_queries - 1)
        self.stats.average_execution_time_ms = (total_time + execution_time_ms) / self.stats.total_queries
        
        # Update performance metrics
        if execution_time_ms < 100:
            self.stats.performance_metrics['fast_queries'] = self.stats.performance_metrics.get('fast_queries', 0) + 1
        elif execution_time_ms < 1000:
            self.stats.performance_metrics['medium_queries'] = self.stats.performance_metrics.get('medium_queries', 0) + 1
        else:
            self.stats.performance_metrics['slow_queries'] = self.stats.performance_metrics.get('slow_queries', 0) + 1
    
    def query_with_quality_enhancement(self, request: QueryRequest) -> QueryResponse:
        """
        Execute a query with enhanced quality-based ranking.
        
        Args:
            request: Query request to execute
            
        Returns:
            Query response with quality-enhanced ranking
        """
        if self.quality_enhancement_engine:
            return self.quality_enhancement_engine.enhance_query_with_quality_ranking(request)
        else:
            # Fall back to standard query if no quality enhancement engine
            self.logger.warning("Quality enhancement engine not available, falling back to standard query")
            return self.query(request)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive query engine statistics.
        
        Returns:
            Dictionary with statistics from all components
        """
        return {
            'query_engine': self.stats.to_dict(),
            'cache': self.cache.get_statistics(),
            'ranking': self.ranker.get_ranking_statistics(),
            'optimization': self.optimizer.get_query_statistics()
        }
    
    def invalidate_cache(self, node_ids: Optional[List[str]] = None):
        """
        Invalidate cache entries when nodes are modified.
        
        Args:
            node_ids: List of modified node IDs
        """
        self.cache.invalidate(node_ids=node_ids)
    
    def shutdown(self):
        """Clean shutdown of query engine."""
        self.cache.shutdown()
        self.logger.info("Advanced Query Engine shutdown complete")