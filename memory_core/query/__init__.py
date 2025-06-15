"""
Advanced Query Engine for Memory Engine

This module provides sophisticated graph query capabilities including:
- Natural language to graph query translation
- Query optimization for common patterns
- Result ranking by relevance and quality scores
- Query result caching for performance
- Support for complex filters and aggregations
- Query explanation for transparency
"""

from .query_engine import AdvancedQueryEngine
from .query_types import (
    QueryRequest,
    QueryResponse,
    QueryExplanation,
    QueryResult,
    QueryType,
    FilterCondition,
    SortCriteria,
    SortOrder,
    AggregationRequest,
    AggregationType,
)
from .natural_language_processor import NaturalLanguageQueryProcessor
from .query_optimizer import QueryOptimizer
from .result_ranker import ResultRanker
from .query_cache import QueryCache
from .filter_processor import FilterProcessor, AggregationProcessor
from .query_explainer import QueryExplainer

__all__ = [
    "AdvancedQueryEngine",
    "QueryRequest",
    "QueryResponse",
    "QueryExplanation",
    "QueryResult",
    "QueryType",
    "FilterCondition",
    "SortCriteria",
    "SortOrder",
    "AggregationRequest",
    "AggregationType",
    "NaturalLanguageQueryProcessor",
    "QueryOptimizer",
    "ResultRanker",
    "QueryCache",
    "FilterProcessor",
    "AggregationProcessor",
    "QueryExplainer",
]
