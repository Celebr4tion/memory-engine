"""
Query Optimizer for Advanced Query Engine

Optimizes graph queries for performance by identifying common patterns,
rewriting queries, and applying performance optimizations.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import hashlib

from .query_types import (
    QueryRequest,
    QueryType,
    FilterCondition,
    GraphPattern,
    QueryExplanationStep,
)


@dataclass
class OptimizationRule:
    """Represents a query optimization rule."""

    name: str
    description: str
    pattern: str
    replacement: str
    conditions: List[str]
    estimated_improvement: float  # Expected performance improvement (0-1)


@dataclass
class QueryPlan:
    """Represents an optimized query execution plan."""

    steps: List[Dict[str, Any]]
    estimated_cost: float
    optimizations_applied: List[str]
    index_hints: List[str]
    parallel_execution: bool = False


class QueryOptimizer:
    """
    Optimizes graph queries for better performance.

    Features:
    - Pattern recognition for common query types
    - Query rewriting for performance
    - Index usage optimization
    - Parallel execution planning
    - Cost-based optimization
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Statistics for optimization decisions
        self.query_stats = defaultdict(list)
        self.pattern_frequency = Counter()
        self.performance_history = {}

        # Optimization rules
        self.optimization_rules = self._load_optimization_rules()

        # Common query patterns cache
        self.pattern_cache = {}

        # Index suggestions
        self.suggested_indexes = set()

    def optimize_query(
        self, request: QueryRequest
    ) -> Tuple[QueryRequest, QueryPlan, List[QueryExplanationStep]]:
        """
        Optimize a query request for better performance.

        Args:
            request: Original query request

        Returns:
            Tuple of (optimized_request, execution_plan, explanation_steps)
        """
        start_time = time.time()
        explanation_steps = []

        # Step 1: Analyze query pattern
        pattern_step = QueryExplanationStep(
            step_name="pattern_analysis",
            description="Analyze query pattern and classify",
            operation="analyze",
        )

        query_pattern = self._analyze_query_pattern(request)
        pattern_step.details = {"pattern": query_pattern, "query_type": request.query_type.value}
        pattern_step.execution_time_ms = (time.time() - start_time) * 1000
        explanation_steps.append(pattern_step)

        # Step 2: Apply optimizations
        opt_start = time.time()
        optimized_request = self._apply_optimizations(request, query_pattern)

        opt_step = QueryExplanationStep(
            step_name="query_optimization",
            description="Apply optimization rules and rewrite query",
            operation="optimize",
        )

        # Identify applied optimizations
        applied_optimizations = self._get_applied_optimizations(request, optimized_request)
        opt_step.optimizations_applied = applied_optimizations
        opt_step.execution_time_ms = (time.time() - opt_start) * 1000
        explanation_steps.append(opt_step)

        # Step 3: Create execution plan
        plan_start = time.time()
        execution_plan = self._create_execution_plan(optimized_request, query_pattern)

        plan_step = QueryExplanationStep(
            step_name="execution_planning",
            description="Create optimized execution plan",
            operation="plan",
            details={
                "estimated_cost": execution_plan.estimated_cost,
                "parallel": execution_plan.parallel_execution,
            },
        )
        plan_step.execution_time_ms = (time.time() - plan_start) * 1000
        explanation_steps.append(plan_step)

        # Update statistics
        self._update_statistics(request, query_pattern)

        total_time = (time.time() - start_time) * 1000
        self.logger.info(
            f"Query optimization completed in {total_time:.2f}ms, {len(applied_optimizations)} optimizations applied"
        )

        return optimized_request, execution_plan, explanation_steps

    def _analyze_query_pattern(self, request: QueryRequest) -> str:
        """
        Analyze the query to identify common patterns.

        Args:
            request: Query request

        Returns:
            Pattern identifier string
        """
        # Create a pattern signature based on query characteristics
        pattern_components = []

        # Query type
        pattern_components.append(f"type:{request.query_type.value}")

        # Number and types of filters
        if request.filters:
            filter_types = [f.operator for f in request.filters]
            pattern_components.append(
                f"filters:{len(request.filters)}:{','.join(sorted(set(filter_types)))}"
            )

        # Sorting
        if request.sort_by:
            sort_fields = [s.field for s in request.sort_by]
            pattern_components.append(f"sort:{','.join(sort_fields)}")

        # Aggregations
        if request.aggregations:
            agg_types = [a.type.value for a in request.aggregations]
            pattern_components.append(f"agg:{','.join(sorted(agg_types))}")

        # Query complexity indicators
        if request.include_relationships:
            pattern_components.append("with_rels")

        if request.max_depth > 1:
            pattern_components.append(f"depth:{request.max_depth}")

        # Limit size category
        if request.limit:
            if request.limit <= 10:
                pattern_components.append("small_limit")
            elif request.limit <= 100:
                pattern_components.append("medium_limit")
            else:
                pattern_components.append("large_limit")

        pattern = "|".join(pattern_components)

        # Generate a hash for long patterns
        if len(pattern) > 100:
            pattern_hash = hashlib.md5(pattern.encode()).hexdigest()[:8]
            pattern = f"complex:{pattern_hash}"

        return pattern

    def _apply_optimizations(self, request: QueryRequest, pattern: str) -> QueryRequest:
        """
        Apply optimization rules to the query request.

        Args:
            request: Original request
            pattern: Query pattern

        Returns:
            Optimized request
        """
        optimized = QueryRequest(
            query=request.query,
            query_type=request.query_type,
            filters=request.filters.copy(),
            sort_by=request.sort_by.copy(),
            limit=request.limit,
            offset=request.offset,
            include_metadata=request.include_metadata,
            include_relationships=request.include_relationships,
            include_embeddings=request.include_embeddings,
            aggregations=request.aggregations.copy(),
            similarity_threshold=request.similarity_threshold,
            max_depth=request.max_depth,
            explain=request.explain,
            use_cache=request.use_cache,
            cache_ttl=request.cache_ttl,
            context=request.context,
            user_id=request.user_id,
            session_id=request.session_id,
        )

        # Optimization 1: Adjust similarity threshold for semantic searches
        if request.query_type == QueryType.SEMANTIC_SEARCH:
            if request.similarity_threshold > 0.9:
                optimized.similarity_threshold = 0.85  # Slightly more permissive for better recall
            elif request.similarity_threshold < 0.5:
                optimized.similarity_threshold = (
                    0.6  # Slightly more restrictive for better precision
                )

        # Optimization 2: Limit depth for complex traversals
        if request.include_relationships and request.max_depth > 3:
            if not request.limit or request.limit > 100:
                optimized.max_depth = min(3, request.max_depth)  # Limit depth for large result sets

        # Optimization 3: Optimize metadata inclusion
        if request.query_type == QueryType.AGGREGATION:
            optimized.include_metadata = False  # Metadata not needed for aggregations
            optimized.include_relationships = False

        # Optimization 4: Reorder filters for efficiency
        if len(request.filters) > 1:
            optimized.filters = self._reorder_filters(request.filters)

        # Optimization 5: Adjust cache TTL based on query type
        if request.query_type == QueryType.AGGREGATION:
            optimized.cache_ttl = max(7200, request.cache_ttl)  # Longer cache for aggregations
        elif request.query_type == QueryType.SEMANTIC_SEARCH:
            optimized.cache_ttl = min(
                1800, request.cache_ttl
            )  # Shorter cache for semantic searches

        # Optimization 6: Enable parallel execution for complex queries
        if self._should_use_parallel_execution(request, pattern):
            # Mark for parallel execution (would be used by execution engine)
            pass

        return optimized

    def _reorder_filters(self, filters: List[FilterCondition]) -> List[FilterCondition]:
        """
        Reorder filters for optimal execution (most selective first).

        Args:
            filters: Original filter list

        Returns:
            Reordered filter list
        """
        # Define selectivity order (most selective first)
        selectivity_order = {
            "eq": 1,  # Equality is most selective
            "in": 2,  # IN clause
            "regex": 3,  # Regex patterns
            "contains": 4,  # Contains searches
            "gt": 5,  # Range comparisons
            "gte": 5,
            "lt": 5,
            "lte": 5,
            "ne": 6,  # Not equal is least selective
            "not_in": 7,
        }

        # Sort by selectivity, then by field name for consistency
        return sorted(filters, key=lambda f: (selectivity_order.get(f.operator, 10), f.field))

    def _should_use_parallel_execution(self, request: QueryRequest, pattern: str) -> bool:
        """
        Determine if a query should use parallel execution.

        Args:
            request: Query request
            pattern: Query pattern

        Returns:
            True if parallel execution is recommended
        """
        # Use parallel execution for:
        # 1. Large result sets with relationships
        # 2. Complex aggregations
        # 3. Deep traversals

        if request.limit and request.limit > 1000:
            return True

        if request.include_relationships and request.max_depth > 2:
            return True

        if len(request.aggregations) > 2:
            return True

        if request.query_type == QueryType.SEMANTIC_SEARCH and not request.limit:
            return True

        return False

    def _create_execution_plan(self, request: QueryRequest, pattern: str) -> QueryPlan:
        """
        Create an optimized execution plan for the query.

        Args:
            request: Optimized query request
            pattern: Query pattern

        Returns:
            Query execution plan
        """
        steps = []
        estimated_cost = 0.0
        optimizations_applied = []
        index_hints = []

        # Step 1: Initial filtering
        if request.filters:
            filter_cost = len(request.filters) * 0.1
            steps.append(
                {
                    "operation": "filter",
                    "description": f"Apply {len(request.filters)} filters",
                    "estimated_cost": filter_cost,
                    "details": {"filter_count": len(request.filters)},
                }
            )
            estimated_cost += filter_cost

            # Suggest indexes for filter fields
            for filter_cond in request.filters:
                index_hints.append(f"index_on_{filter_cond.field}")

        # Step 2: Query execution based on type
        if request.query_type == QueryType.SEMANTIC_SEARCH:
            vector_cost = 2.0  # Vector searches are more expensive
            steps.append(
                {
                    "operation": "vector_search",
                    "description": "Perform semantic similarity search",
                    "estimated_cost": vector_cost,
                    "details": {"similarity_threshold": request.similarity_threshold},
                }
            )
            estimated_cost += vector_cost
            optimizations_applied.append("vector_index_optimization")

        elif request.query_type == QueryType.GRAPH_PATTERN:
            graph_cost = 1.5 * (request.max_depth or 1)
            steps.append(
                {
                    "operation": "graph_traversal",
                    "description": f"Graph pattern matching (depth: {request.max_depth})",
                    "estimated_cost": graph_cost,
                    "details": {"max_depth": request.max_depth},
                }
            )
            estimated_cost += graph_cost

        elif request.query_type == QueryType.AGGREGATION:
            agg_cost = len(request.aggregations) * 0.5
            steps.append(
                {
                    "operation": "aggregation",
                    "description": f"Compute {len(request.aggregations)} aggregations",
                    "estimated_cost": agg_cost,
                    "details": {"aggregation_count": len(request.aggregations)},
                }
            )
            estimated_cost += agg_cost
            optimizations_applied.append("aggregation_pushdown")

        # Step 3: Relationship expansion (if needed)
        if request.include_relationships:
            rel_cost = 0.5 * (request.max_depth or 1)
            steps.append(
                {
                    "operation": "expand_relationships",
                    "description": f"Expand relationships (depth: {request.max_depth})",
                    "estimated_cost": rel_cost,
                    "details": {"max_depth": request.max_depth},
                }
            )
            estimated_cost += rel_cost
            index_hints.append("relationship_index")

        # Step 4: Sorting
        if request.sort_by:
            sort_cost = 0.3 * len(request.sort_by)
            steps.append(
                {
                    "operation": "sort",
                    "description": f"Sort by {len(request.sort_by)} criteria",
                    "estimated_cost": sort_cost,
                    "details": {"sort_fields": [s.field for s in request.sort_by]},
                }
            )
            estimated_cost += sort_cost

            # Suggest indexes for sort fields
            for sort_criteria in request.sort_by:
                index_hints.append(f"index_on_{sort_criteria.field}")

        # Step 5: Pagination
        if request.limit:
            pagination_cost = 0.1
            steps.append(
                {
                    "operation": "paginate",
                    "description": f"Apply limit {request.limit} offset {request.offset}",
                    "estimated_cost": pagination_cost,
                    "details": {"limit": request.limit, "offset": request.offset},
                }
            )
            estimated_cost += pagination_cost
            optimizations_applied.append("limit_pushdown")

        # Determine if parallel execution is beneficial
        parallel_execution = self._should_use_parallel_execution(request, pattern)
        if parallel_execution:
            optimizations_applied.append("parallel_execution")
            estimated_cost *= 0.7  # Parallel execution reduces overall cost

        return QueryPlan(
            steps=steps,
            estimated_cost=estimated_cost,
            optimizations_applied=optimizations_applied,
            index_hints=list(set(index_hints)),  # Remove duplicates
            parallel_execution=parallel_execution,
        )

    def _get_applied_optimizations(
        self, original: QueryRequest, optimized: QueryRequest
    ) -> List[str]:
        """
        Identify which optimizations were applied by comparing requests.

        Args:
            original: Original request
            optimized: Optimized request

        Returns:
            List of optimization names
        """
        optimizations = []

        if original.similarity_threshold != optimized.similarity_threshold:
            optimizations.append("similarity_threshold_tuning")

        if original.max_depth != optimized.max_depth:
            optimizations.append("depth_limiting")

        if original.include_metadata != optimized.include_metadata:
            optimizations.append("metadata_exclusion")

        if original.cache_ttl != optimized.cache_ttl:
            optimizations.append("cache_ttl_optimization")

        if original.filters != optimized.filters:
            optimizations.append("filter_reordering")

        return optimizations

    def _update_statistics(self, request: QueryRequest, pattern: str):
        """
        Update query statistics for future optimization decisions.

        Args:
            request: Query request
            pattern: Query pattern
        """
        self.pattern_frequency[pattern] += 1

        # Store pattern with query characteristics
        query_signature = f"{pattern}|{request.query_type.value}"
        if query_signature not in self.query_stats:
            self.query_stats[query_signature] = []

        self.query_stats[query_signature].append(
            {
                "timestamp": time.time(),
                "filters": len(request.filters),
                "limit": request.limit,
                "has_relationships": request.include_relationships,
                "max_depth": request.max_depth,
            }
        )

        # Keep only recent statistics (last 1000 queries per pattern)
        if len(self.query_stats[query_signature]) > 1000:
            self.query_stats[query_signature] = self.query_stats[query_signature][-1000:]

    def _load_optimization_rules(self) -> List[OptimizationRule]:
        """
        Load predefined optimization rules.

        Returns:
            List of optimization rules
        """
        return [
            OptimizationRule(
                name="limit_pushdown",
                description="Push LIMIT clause down to reduce intermediate results",
                pattern=".*limit.*",
                replacement="optimized_limit",
                conditions=["has_limit", "no_aggregation"],
                estimated_improvement=0.3,
            ),
            OptimizationRule(
                name="filter_pushdown",
                description="Push filters as early as possible in execution",
                pattern=".*filter.*",
                replacement="early_filter",
                conditions=["has_filters"],
                estimated_improvement=0.4,
            ),
            OptimizationRule(
                name="index_selection",
                description="Select optimal indexes for query execution",
                pattern=".*",
                replacement="with_index_hints",
                conditions=["has_indexed_fields"],
                estimated_improvement=0.5,
            ),
            OptimizationRule(
                name="parallel_aggregation",
                description="Use parallel execution for aggregations",
                pattern=".*aggregation.*",
                replacement="parallel_agg",
                conditions=["multiple_aggregations"],
                estimated_improvement=0.6,
            ),
            OptimizationRule(
                name="semantic_threshold_tuning",
                description="Optimize similarity thresholds for semantic searches",
                pattern=".*semantic.*",
                replacement="tuned_threshold",
                conditions=["semantic_search"],
                estimated_improvement=0.2,
            ),
        ]

    def get_optimization_suggestions(self, pattern: str) -> List[str]:
        """
        Get optimization suggestions for a query pattern.

        Args:
            pattern: Query pattern

        Returns:
            List of optimization suggestions
        """
        suggestions = []

        # Frequency-based suggestions
        if self.pattern_frequency[pattern] > 10:
            suggestions.append(
                f"Consider creating an optimized query template for pattern '{pattern}' (used {self.pattern_frequency[pattern]} times)"
            )

        # Performance-based suggestions
        if pattern in self.performance_history:
            avg_time = self.performance_history[pattern].get("avg_execution_time", 0)
            if avg_time > 1000:  # More than 1 second
                suggestions.append(
                    "Consider adding indexes or limiting result size for better performance"
                )

        # Index suggestions
        if self.suggested_indexes:
            suggestions.append(
                f"Consider creating indexes: {', '.join(list(self.suggested_indexes)[:3])}"
            )

        return suggestions

    def get_query_statistics(self) -> Dict[str, Any]:
        """
        Get query optimization statistics.

        Returns:
            Dictionary with statistics
        """
        total_queries = sum(self.pattern_frequency.values())
        most_common = self.pattern_frequency.most_common(5)

        return {
            "total_optimized_queries": total_queries,
            "unique_patterns": len(self.pattern_frequency),
            "most_common_patterns": [{"pattern": p, "count": c} for p, c in most_common],
            "suggested_indexes": list(self.suggested_indexes),
            "optimization_rules_loaded": len(self.optimization_rules),
        }
