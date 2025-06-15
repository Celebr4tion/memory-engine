"""
GraphQL-like Query Language Support for Memory Engine

This module provides a GraphQL-inspired query language for complex knowledge queries:
- Flexible field selection and projection
- Nested relationship traversal
- Complex filtering and aggregation
- Custom query optimization
"""

import json
import re
from typing import Dict, List, Any, Optional, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries supported."""

    NODES = "nodes"
    RELATIONSHIPS = "relationships"
    GRAPH = "graph"
    AGGREGATION = "aggregation"
    SEARCH = "search"


class FilterOperator(Enum):
    """Filter operators for query conditions."""

    EQ = "eq"  # equals
    NE = "ne"  # not equals
    GT = "gt"  # greater than
    GTE = "gte"  # greater than or equal
    LT = "lt"  # less than
    LTE = "lte"  # less than or equal
    IN = "in"  # in list
    NOT_IN = "not_in"  # not in list
    CONTAINS = "contains"  # string contains
    STARTS_WITH = "starts_with"  # string starts with
    ENDS_WITH = "ends_with"  # string ends with
    REGEX = "regex"  # regex match
    EXISTS = "exists"  # field exists
    NULL = "null"  # field is null


@dataclass
class FilterCondition:
    """Filter condition for queries."""

    field: str
    operator: FilterOperator
    value: Any

    def to_dict(self) -> Dict[str, Any]:
        return {"field": self.field, "operator": self.operator.value, "value": self.value}


@dataclass
class QueryProjection:
    """Field projection for query results."""

    fields: List[str] = field(default_factory=list)
    exclude_fields: List[str] = field(default_factory=list)
    nested_projections: Dict[str, "QueryProjection"] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {"fields": self.fields, "exclude_fields": self.exclude_fields}
        if self.nested_projections:
            result["nested_projections"] = {
                k: v.to_dict() for k, v in self.nested_projections.items()
            }
        return result


@dataclass
class QueryAggregation:
    """Aggregation specification for queries."""

    field: str
    operation: str  # count, sum, avg, min, max, group_by
    alias: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"field": self.field, "operation": self.operation, "alias": self.alias}


@dataclass
class QuerySort:
    """Sort specification for queries."""

    field: str
    ascending: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {"field": self.field, "ascending": self.ascending}


@dataclass
class QuerySpec:
    """Complete query specification."""

    query_type: QueryType
    filters: List[FilterCondition] = field(default_factory=list)
    projection: Optional[QueryProjection] = None
    aggregations: List[QueryAggregation] = field(default_factory=list)
    sorts: List[QuerySort] = field(default_factory=list)
    limit: Optional[int] = None
    offset: Optional[int] = None
    include_metadata: bool = True

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "query_type": self.query_type.value,
            "filters": [f.to_dict() for f in self.filters],
            "aggregations": [a.to_dict() for a in self.aggregations],
            "sorts": [s.to_dict() for s in self.sorts],
            "limit": self.limit,
            "offset": self.offset,
            "include_metadata": self.include_metadata,
        }
        if self.projection:
            result["projection"] = self.projection.to_dict()
        return result


class QueryValidator:
    """Validates query specifications."""

    def __init__(self):
        self.supported_fields = {
            "nodes": [
                "id",
                "content",
                "source",
                "timestamp",
                "rating_truthfulness",
                "rating_importance",
                "rating_novelty",
                "tags",
                "metadata",
            ],
            "relationships": [
                "id",
                "source_id",
                "target_id",
                "relationship_type",
                "confidence",
                "metadata",
                "timestamp",
            ],
            "graph": ["nodes", "relationships", "metadata"],
        }

        self.supported_aggregations = ["count", "sum", "avg", "min", "max", "group_by"]

    def validate_query(self, query_spec: QuerySpec) -> List[str]:
        """Validate query specification and return list of errors."""
        errors = []

        # Validate query type
        if query_spec.query_type not in QueryType:
            errors.append(f"Invalid query type: {query_spec.query_type}")

        # Validate filters
        for filter_condition in query_spec.filters:
            if filter_condition.operator not in FilterOperator:
                errors.append(f"Invalid filter operator: {filter_condition.operator}")

            # Validate field exists for query type
            query_type_str = query_spec.query_type.value
            if (
                query_type_str in self.supported_fields
                and filter_condition.field not in self.supported_fields[query_type_str]
            ):
                errors.append(
                    f"Invalid field '{filter_condition.field}' for query type '{query_type_str}'"
                )

        # Validate aggregations
        for aggregation in query_spec.aggregations:
            if aggregation.operation not in self.supported_aggregations:
                errors.append(f"Invalid aggregation operation: {aggregation.operation}")

        # Validate projection
        if query_spec.projection:
            query_type_str = query_spec.query_type.value
            if query_type_str in self.supported_fields:
                for field_name in query_spec.projection.fields:
                    if field_name not in self.supported_fields[query_type_str]:
                        errors.append(
                            f"Invalid projection field '{field_name}' for query type '{query_type_str}'"
                        )

        # Validate limit and offset
        if query_spec.limit is not None and query_spec.limit < 0:
            errors.append("Limit must be non-negative")

        if query_spec.offset is not None and query_spec.offset < 0:
            errors.append("Offset must be non-negative")

        return errors

    def validate_query_syntax(self, query_string: str) -> List[str]:
        """Validate query string syntax."""
        errors = []

        try:
            # Basic JSON validation
            json.loads(query_string)
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON syntax: {e}")

        return errors


class QueryBuilder:
    """Builder for constructing query specifications."""

    def __init__(self):
        self.reset()

    def reset(self) -> "QueryBuilder":
        """Reset builder to initial state."""
        self._query_type = QueryType.NODES
        self._filters = []
        self._projection = None
        self._aggregations = []
        self._sorts = []
        self._limit = None
        self._offset = None
        self._include_metadata = True
        return self

    def query_type(self, query_type: QueryType) -> "QueryBuilder":
        """Set query type."""
        self._query_type = query_type
        return self

    def filter(self, field: str, operator: FilterOperator, value: Any) -> "QueryBuilder":
        """Add filter condition."""
        self._filters.append(FilterCondition(field, operator, value))
        return self

    def where(self, field: str, value: Any) -> "QueryBuilder":
        """Add equality filter (convenience method)."""
        return self.filter(field, FilterOperator.EQ, value)

    def where_in(self, field: str, values: List[Any]) -> "QueryBuilder":
        """Add 'in' filter (convenience method)."""
        return self.filter(field, FilterOperator.IN, values)

    def where_contains(self, field: str, value: str) -> "QueryBuilder":
        """Add contains filter (convenience method)."""
        return self.filter(field, FilterOperator.CONTAINS, value)

    def where_greater_than(self, field: str, value: Union[int, float]) -> "QueryBuilder":
        """Add greater than filter (convenience method)."""
        return self.filter(field, FilterOperator.GT, value)

    def select(self, *fields: str) -> "QueryBuilder":
        """Set projection fields."""
        if not self._projection:
            self._projection = QueryProjection()
        self._projection.fields.extend(fields)
        return self

    def exclude(self, *fields: str) -> "QueryBuilder":
        """Set excluded fields."""
        if not self._projection:
            self._projection = QueryProjection()
        self._projection.exclude_fields.extend(fields)
        return self

    def aggregate(self, field: str, operation: str, alias: Optional[str] = None) -> "QueryBuilder":
        """Add aggregation."""
        self._aggregations.append(QueryAggregation(field, operation, alias))
        return self

    def count(self, field: str = "*", alias: str = "count") -> "QueryBuilder":
        """Add count aggregation (convenience method)."""
        return self.aggregate(field, "count", alias)

    def sum(self, field: str, alias: Optional[str] = None) -> "QueryBuilder":
        """Add sum aggregation (convenience method)."""
        return self.aggregate(field, "sum", alias)

    def avg(self, field: str, alias: Optional[str] = None) -> "QueryBuilder":
        """Add average aggregation (convenience method)."""
        return self.aggregate(field, "avg", alias)

    def group_by(self, field: str) -> "QueryBuilder":
        """Add group by aggregation (convenience method)."""
        return self.aggregate(field, "group_by")

    def order_by(self, field: str, ascending: bool = True) -> "QueryBuilder":
        """Add sort order."""
        self._sorts.append(QuerySort(field, ascending))
        return self

    def limit(self, limit: int) -> "QueryBuilder":
        """Set result limit."""
        self._limit = limit
        return self

    def offset(self, offset: int) -> "QueryBuilder":
        """Set result offset."""
        self._offset = offset
        return self

    def include_metadata(self, include: bool = True) -> "QueryBuilder":
        """Set whether to include metadata."""
        self._include_metadata = include
        return self

    def build(self) -> QuerySpec:
        """Build query specification."""
        return QuerySpec(
            query_type=self._query_type,
            filters=self._filters.copy(),
            projection=self._projection,
            aggregations=self._aggregations.copy(),
            sorts=self._sorts.copy(),
            limit=self._limit,
            offset=self._offset,
            include_metadata=self._include_metadata,
        )


class QueryProcessor(ABC):
    """Abstract base class for query processors."""

    @abstractmethod
    async def execute_query(self, query_spec: QuerySpec) -> Dict[str, Any]:
        """Execute query and return results."""
        pass


class GraphQLQueryProcessor(QueryProcessor):
    """Processes GraphQL-like queries against the knowledge engine."""

    def __init__(self, knowledge_engine: Any):
        self.knowledge_engine = knowledge_engine
        self.validator = QueryValidator()

    async def execute_query(self, query_spec: QuerySpec) -> Dict[str, Any]:
        """Execute query specification against knowledge engine."""
        # Validate query
        validation_errors = self.validator.validate_query(query_spec)
        if validation_errors:
            return {"success": False, "errors": validation_errors}

        try:
            if query_spec.query_type == QueryType.NODES:
                return await self._execute_node_query(query_spec)
            elif query_spec.query_type == QueryType.RELATIONSHIPS:
                return await self._execute_relationship_query(query_spec)
            elif query_spec.query_type == QueryType.GRAPH:
                return await self._execute_graph_query(query_spec)
            elif query_spec.query_type == QueryType.AGGREGATION:
                return await self._execute_aggregation_query(query_spec)
            elif query_spec.query_type == QueryType.SEARCH:
                return await self._execute_search_query(query_spec)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported query type: {query_spec.query_type}",
                }

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return {"success": False, "error": str(e)}

    async def _execute_node_query(self, query_spec: QuerySpec) -> Dict[str, Any]:
        """Execute node query."""
        # Get all nodes (simplified - would use actual knowledge engine methods)
        nodes = await self.knowledge_engine.get_all_nodes()

        # Apply filters
        filtered_nodes = self._apply_filters(nodes, query_spec.filters)

        # Apply sorting
        sorted_nodes = self._apply_sorting(filtered_nodes, query_spec.sorts)

        # Apply pagination
        paginated_nodes = self._apply_pagination(sorted_nodes, query_spec.limit, query_spec.offset)

        # Apply projection
        projected_nodes = self._apply_projection(paginated_nodes, query_spec.projection)

        return {
            "success": True,
            "data": projected_nodes,
            "total_count": len(filtered_nodes),
            "returned_count": len(projected_nodes),
        }

    async def _execute_relationship_query(self, query_spec: QuerySpec) -> Dict[str, Any]:
        """Execute relationship query."""
        # Get all relationships (simplified)
        relationships = await self.knowledge_engine.get_all_relationships()

        # Apply filters
        filtered_relationships = self._apply_filters(relationships, query_spec.filters)

        # Apply sorting
        sorted_relationships = self._apply_sorting(filtered_relationships, query_spec.sorts)

        # Apply pagination
        paginated_relationships = self._apply_pagination(
            sorted_relationships, query_spec.limit, query_spec.offset
        )

        # Apply projection
        projected_relationships = self._apply_projection(
            paginated_relationships, query_spec.projection
        )

        return {
            "success": True,
            "data": projected_relationships,
            "total_count": len(filtered_relationships),
            "returned_count": len(projected_relationships),
        }

    async def _execute_graph_query(self, query_spec: QuerySpec) -> Dict[str, Any]:
        """Execute graph query (nodes + relationships)."""
        nodes_result = await self._execute_node_query(
            QuerySpec(
                QueryType.NODES,
                query_spec.filters,
                query_spec.projection,
                [],
                query_spec.sorts,
                query_spec.limit,
                query_spec.offset,
            )
        )

        relationships_result = await self._execute_relationship_query(
            QuerySpec(
                QueryType.RELATIONSHIPS,
                query_spec.filters,
                query_spec.projection,
                [],
                query_spec.sorts,
                query_spec.limit,
                query_spec.offset,
            )
        )

        return {
            "success": True,
            "data": {
                "nodes": nodes_result.get("data", []),
                "relationships": relationships_result.get("data", []),
            },
            "metadata": {
                "node_count": nodes_result.get("total_count", 0),
                "relationship_count": relationships_result.get("total_count", 0),
            },
        }

    async def _execute_aggregation_query(self, query_spec: QuerySpec) -> Dict[str, Any]:
        """Execute aggregation query."""
        # Get base data
        if query_spec.query_type == QueryType.NODES:
            data = await self.knowledge_engine.get_all_nodes()
        else:
            data = await self.knowledge_engine.get_all_relationships()

        # Apply filters
        filtered_data = self._apply_filters(data, query_spec.filters)

        # Apply aggregations
        aggregation_results = {}
        for aggregation in query_spec.aggregations:
            result = self._apply_aggregation(filtered_data, aggregation)
            alias = aggregation.alias or f"{aggregation.operation}_{aggregation.field}"
            aggregation_results[alias] = result

        return {"success": True, "data": aggregation_results, "total_records": len(filtered_data)}

    async def _execute_search_query(self, query_spec: QuerySpec) -> Dict[str, Any]:
        """Execute semantic search query."""
        # Extract search terms from filters
        search_terms = []
        for filter_cond in query_spec.filters:
            if filter_cond.field == "content" and filter_cond.operator == FilterOperator.CONTAINS:
                search_terms.append(filter_cond.value)

        if not search_terms:
            return {
                "success": False,
                "error": "Search query requires content filter with contains operator",
            }

        # Perform semantic search
        search_results = await self.knowledge_engine.semantic_search(
            " ".join(search_terms), limit=query_spec.limit or 10
        )

        # Apply projection
        projected_results = self._apply_projection(search_results, query_spec.projection)

        return {
            "success": True,
            "data": projected_results,
            "search_terms": search_terms,
            "returned_count": len(projected_results),
        }

    def _apply_filters(
        self, data: List[Dict[str, Any]], filters: List[FilterCondition]
    ) -> List[Dict[str, Any]]:
        """Apply filter conditions to data."""
        if not filters:
            return data

        filtered_data = []
        for item in data:
            if self._item_matches_filters(item, filters):
                filtered_data.append(item)

        return filtered_data

    def _item_matches_filters(self, item: Dict[str, Any], filters: List[FilterCondition]) -> bool:
        """Check if item matches all filter conditions."""
        for filter_cond in filters:
            if not self._evaluate_filter(item, filter_cond):
                return False
        return True

    def _evaluate_filter(self, item: Dict[str, Any], filter_cond: FilterCondition) -> bool:
        """Evaluate single filter condition against item."""
        field_value = item.get(filter_cond.field)
        filter_value = filter_cond.value

        if filter_cond.operator == FilterOperator.EQ:
            return field_value == filter_value
        elif filter_cond.operator == FilterOperator.NE:
            return field_value != filter_value
        elif filter_cond.operator == FilterOperator.GT:
            return field_value is not None and field_value > filter_value
        elif filter_cond.operator == FilterOperator.GTE:
            return field_value is not None and field_value >= filter_value
        elif filter_cond.operator == FilterOperator.LT:
            return field_value is not None and field_value < filter_value
        elif filter_cond.operator == FilterOperator.LTE:
            return field_value is not None and field_value <= filter_value
        elif filter_cond.operator == FilterOperator.IN:
            return field_value in filter_value
        elif filter_cond.operator == FilterOperator.NOT_IN:
            return field_value not in filter_value
        elif filter_cond.operator == FilterOperator.CONTAINS:
            return (
                isinstance(field_value, str)
                and isinstance(filter_value, str)
                and filter_value.lower() in field_value.lower()
            )
        elif filter_cond.operator == FilterOperator.STARTS_WITH:
            return (
                isinstance(field_value, str)
                and isinstance(filter_value, str)
                and field_value.lower().startswith(filter_value.lower())
            )
        elif filter_cond.operator == FilterOperator.ENDS_WITH:
            return (
                isinstance(field_value, str)
                and isinstance(filter_value, str)
                and field_value.lower().endswith(filter_value.lower())
            )
        elif filter_cond.operator == FilterOperator.REGEX:
            return (
                isinstance(field_value, str)
                and isinstance(filter_value, str)
                and re.search(filter_value, field_value) is not None
            )
        elif filter_cond.operator == FilterOperator.EXISTS:
            return filter_cond.field in item
        elif filter_cond.operator == FilterOperator.NULL:
            return field_value is None

        return False

    def _apply_sorting(
        self, data: List[Dict[str, Any]], sorts: List[QuerySort]
    ) -> List[Dict[str, Any]]:
        """Apply sorting to data."""
        if not sorts:
            return data

        def sort_key(item):
            # Create sort key tuple
            key_values = []
            for sort_spec in sorts:
                value = item.get(sort_spec.field)
                # Handle None values
                if value is None:
                    value = "" if sort_spec.ascending else "zzz"
                key_values.append(value if sort_spec.ascending else self._reverse_sort_value(value))
            return tuple(key_values)

        return sorted(data, key=sort_key)

    def _reverse_sort_value(self, value: Any) -> Any:
        """Create reverse sort value for descending order."""
        if isinstance(value, (int, float)):
            return -value
        elif isinstance(value, str):
            # Create reverse string comparison
            return "".join(chr(255 - ord(c)) for c in value)
        else:
            return value

    def _apply_pagination(
        self, data: List[Dict[str, Any]], limit: Optional[int], offset: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Apply pagination to data."""
        start = offset or 0
        end = start + limit if limit else None
        return data[start:end]

    def _apply_projection(
        self, data: List[Dict[str, Any]], projection: Optional[QueryProjection]
    ) -> List[Dict[str, Any]]:
        """Apply field projection to data."""
        if not projection:
            return data

        projected_data = []
        for item in data:
            projected_item = {}

            # Include specified fields
            if projection.fields:
                for field in projection.fields:
                    if field in item:
                        projected_item[field] = item[field]
            else:
                # Include all fields if none specified
                projected_item = item.copy()

            # Exclude specified fields
            for field in projection.exclude_fields:
                projected_item.pop(field, None)

            projected_data.append(projected_item)

        return projected_data

    def _apply_aggregation(self, data: List[Dict[str, Any]], aggregation: QueryAggregation) -> Any:
        """Apply aggregation to data."""
        field_values = [
            item.get(aggregation.field)
            for item in data
            if aggregation.field in item and item[aggregation.field] is not None
        ]

        if aggregation.operation == "count":
            if aggregation.field == "*":
                return len(data)
            else:
                return len(field_values)
        elif aggregation.operation == "sum":
            return sum(v for v in field_values if isinstance(v, (int, float)))
        elif aggregation.operation == "avg":
            numeric_values = [v for v in field_values if isinstance(v, (int, float))]
            return sum(numeric_values) / len(numeric_values) if numeric_values else 0
        elif aggregation.operation == "min":
            return min(field_values) if field_values else None
        elif aggregation.operation == "max":
            return max(field_values) if field_values else None
        elif aggregation.operation == "group_by":
            groups = {}
            for item in data:
                group_key = item.get(aggregation.field, "null")
                if group_key not in groups:
                    groups[group_key] = []
                groups[group_key].append(item)
            return groups

        return None


# Convenience functions for common query patterns
def create_node_query() -> QueryBuilder:
    """Create query builder for nodes."""
    return QueryBuilder().query_type(QueryType.NODES)


def create_relationship_query() -> QueryBuilder:
    """Create query builder for relationships."""
    return QueryBuilder().query_type(QueryType.RELATIONSHIPS)


def create_search_query(search_term: str) -> QueryBuilder:
    """Create query builder for semantic search."""
    return (
        QueryBuilder()
        .query_type(QueryType.SEARCH)
        .filter("content", FilterOperator.CONTAINS, search_term)
    )


def create_aggregation_query() -> QueryBuilder:
    """Create query builder for aggregations."""
    return QueryBuilder().query_type(QueryType.AGGREGATION)
