"""
Query types and data structures for the advanced query engine.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum
import time
from datetime import datetime


class QueryType(Enum):
    """Types of supported queries."""

    NATURAL_LANGUAGE = "natural_language"
    GRAPH_PATTERN = "graph_pattern"
    SEMANTIC_SEARCH = "semantic_search"
    RELATIONSHIP_SEARCH = "relationship_search"
    AGGREGATION = "aggregation"
    HYBRID = "hybrid"


class SortOrder(Enum):
    """Sort order for query results."""

    ASCENDING = "asc"
    DESCENDING = "desc"


class AggregationType(Enum):
    """Types of aggregations supported."""

    COUNT = "count"
    SUM = "sum"
    AVERAGE = "avg"
    MIN = "min"
    MAX = "max"
    GROUP_BY = "group_by"


@dataclass
class FilterCondition:
    """Represents a filter condition for queries."""

    field: str
    operator: str  # eq, ne, gt, lt, gte, lte, in, not_in, contains, regex
    value: Any
    case_sensitive: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "operator": self.operator,
            "value": self.value,
            "case_sensitive": self.case_sensitive,
        }


@dataclass
class SortCriteria:
    """Represents sorting criteria for query results."""

    field: str
    order: SortOrder = SortOrder.DESCENDING

    def to_dict(self) -> Dict[str, Any]:
        return {"field": self.field, "order": self.order.value}


@dataclass
class AggregationRequest:
    """Represents an aggregation request."""

    type: AggregationType
    field: Optional[str] = None
    group_by: Optional[List[str]] = None
    having: Optional[List[FilterCondition]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "field": self.field,
            "group_by": self.group_by,
            "having": [h.to_dict() for h in (self.having or [])],
        }


@dataclass
class QueryRequest:
    """Represents a query request to the advanced query engine."""

    query: str
    query_type: QueryType = QueryType.NATURAL_LANGUAGE
    filters: List[FilterCondition] = field(default_factory=list)
    sort_by: List[SortCriteria] = field(default_factory=list)
    limit: Optional[int] = None
    offset: int = 0
    include_metadata: bool = True
    include_relationships: bool = True
    include_embeddings: bool = False
    aggregations: List[AggregationRequest] = field(default_factory=list)
    similarity_threshold: float = 0.7
    max_depth: int = 3
    explain: bool = False
    use_cache: bool = True
    cache_ttl: int = 3600  # Cache TTL in seconds

    # Query context for better understanding
    context: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "query_type": self.query_type.value,
            "filters": [f.to_dict() for f in self.filters],
            "sort_by": [s.to_dict() for s in self.sort_by],
            "limit": self.limit,
            "offset": self.offset,
            "include_metadata": self.include_metadata,
            "include_relationships": self.include_relationships,
            "include_embeddings": self.include_embeddings,
            "aggregations": [a.to_dict() for a in self.aggregations],
            "similarity_threshold": self.similarity_threshold,
            "max_depth": self.max_depth,
            "explain": self.explain,
            "use_cache": self.use_cache,
            "cache_ttl": self.cache_ttl,
            "context": self.context,
            "user_id": self.user_id,
            "session_id": self.session_id,
        }


@dataclass
class QueryExplanationStep:
    """Represents a step in query execution explanation."""

    step_name: str
    description: str
    operation: str
    input_size: Optional[int] = None
    output_size: Optional[int] = None
    execution_time_ms: Optional[float] = None
    optimizations_applied: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_name": self.step_name,
            "description": self.description,
            "operation": self.operation,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "execution_time_ms": self.execution_time_ms,
            "optimizations_applied": self.optimizations_applied,
            "details": self.details,
        }


@dataclass
class QueryExplanation:
    """Provides explanation of how a query was executed."""

    original_query: str
    parsed_query: Dict[str, Any]
    translation_steps: List[str]
    execution_plan: List[QueryExplanationStep]
    optimizations: List[str]
    total_execution_time_ms: float
    cache_hit: bool = False
    cache_key: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "parsed_query": self.parsed_query,
            "translation_steps": self.translation_steps,
            "execution_plan": [step.to_dict() for step in self.execution_plan],
            "optimizations": self.optimizations,
            "total_execution_time_ms": self.total_execution_time_ms,
            "cache_hit": self.cache_hit,
            "cache_key": self.cache_key,
        }


@dataclass
class QueryResult:
    """Represents a single query result item."""

    node_id: str
    content: str
    node_type: Optional[str] = None
    relevance_score: float = 0.0
    quality_score: float = 0.0
    combined_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    explanation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "content": self.content,
            "node_type": self.node_type,
            "relevance_score": self.relevance_score,
            "quality_score": self.quality_score,
            "combined_score": self.combined_score,
            "metadata": self.metadata,
            "relationships": self.relationships,
            "embedding": self.embedding,
            "explanation": self.explanation,
        }


@dataclass
class AggregationResult:
    """Represents the result of an aggregation operation."""

    aggregation_type: str
    field: Optional[str]
    value: Any
    group_key: Optional[str] = None
    count: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aggregation_type": self.aggregation_type,
            "field": self.field,
            "value": self.value,
            "group_key": self.group_key,
            "count": self.count,
        }


@dataclass
class QueryResponse:
    """Response from the advanced query engine."""

    results: List[QueryResult]
    total_count: int
    returned_count: int
    aggregations: List[AggregationResult] = field(default_factory=list)
    explanation: Optional[QueryExplanation] = None
    execution_time_ms: float = 0.0
    from_cache: bool = False
    query_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    # Pagination info
    has_more: bool = False
    next_offset: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "total_count": self.total_count,
            "returned_count": self.returned_count,
            "aggregations": [a.to_dict() for a in self.aggregations],
            "explanation": self.explanation.to_dict() if self.explanation else None,
            "execution_time_ms": self.execution_time_ms,
            "from_cache": self.from_cache,
            "query_id": self.query_id,
            "timestamp": self.timestamp.isoformat(),
            "has_more": self.has_more,
            "next_offset": self.next_offset,
        }


@dataclass
class GraphPattern:
    """Represents a graph pattern for pattern-based queries."""

    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    constraints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"nodes": self.nodes, "edges": self.edges, "constraints": self.constraints}


@dataclass
class QueryStatistics:
    """Statistics about query execution."""

    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    average_execution_time_ms: float = 0.0
    most_common_patterns: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total) if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hit_rate,
            "average_execution_time_ms": self.average_execution_time_ms,
            "most_common_patterns": self.most_common_patterns,
            "performance_metrics": self.performance_metrics,
        }
