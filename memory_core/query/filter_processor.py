"""
Advanced Filter and Aggregation Processor

Handles complex filtering logic and aggregation operations for the query engine.
"""

import logging
import re
import statistics
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, Counter

from .query_types import FilterCondition, AggregationRequest, QueryResult, AggregationResult, AggregationType


@dataclass
class FilterContext:
    """Context for filter evaluation."""
    case_sensitive: bool = True
    null_handling: str = "exclude"  # exclude, include, as_value
    type_coercion: bool = True


class FilterProcessor:
    """
    Advanced filter processor supporting complex filter logic.
    
    Features:
    - Multiple comparison operators
    - Nested field access
    - Type coercion and validation
    - Regex pattern matching
    - Date/time range filtering
    - Array/list operations
    - Custom filter functions
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Operator implementations
        self.operators = {
            'eq': self._op_equals,
            'ne': self._op_not_equals,
            'gt': self._op_greater_than,
            'gte': self._op_greater_than_equal,
            'lt': self._op_less_than,
            'lte': self._op_less_than_equal,
            'in': self._op_in,
            'not_in': self._op_not_in,
            'contains': self._op_contains,
            'not_contains': self._op_not_contains,
            'starts_with': self._op_starts_with,
            'ends_with': self._op_ends_with,
            'regex': self._op_regex,
            'exists': self._op_exists,
            'not_exists': self._op_not_exists,
            'empty': self._op_empty,
            'not_empty': self._op_not_empty,
            'between': self._op_between,
            'date_range': self._op_date_range,
            'array_contains': self._op_array_contains,
            'array_length': self._op_array_length,
        }
        
        # Type conversion functions
        self.type_converters = {
            'int': lambda x: int(x) if x is not None else None,
            'float': lambda x: float(x) if x is not None else None,
            'str': lambda x: str(x) if x is not None else None,
            'bool': lambda x: bool(x) if x is not None else None,
            'datetime': self._convert_to_datetime,
        }
    
    def apply_filters(self, 
                     results: List[QueryResult], 
                     filters: List[FilterCondition],
                     context: Optional[FilterContext] = None) -> List[QueryResult]:
        """
        Apply filters to query results.
        
        Args:
            results: List of query results to filter
            filters: List of filter conditions
            context: Filter evaluation context
            
        Returns:
            Filtered list of query results
        """
        if not filters:
            return results
        
        context = context or FilterContext()
        filtered_results = []
        
        for result in results:
            if self._evaluate_filters(result, filters, context):
                filtered_results.append(result)
        
        self.logger.debug(f"Filtered {len(results)} results to {len(filtered_results)}")
        return filtered_results
    
    def _evaluate_filters(self, 
                         result: QueryResult, 
                         filters: List[FilterCondition],
                         context: FilterContext) -> bool:
        """
        Evaluate all filters against a single result.
        
        Args:
            result: Query result to evaluate
            filters: List of filter conditions
            context: Filter context
            
        Returns:
            True if result passes all filters
        """
        for filter_condition in filters:
            if not self._evaluate_single_filter(result, filter_condition, context):
                return False
        return True
    
    def _evaluate_single_filter(self, 
                               result: QueryResult, 
                               filter_cond: FilterCondition,
                               context: FilterContext) -> bool:
        """
        Evaluate a single filter condition against a result.
        
        Args:
            result: Query result to evaluate
            filter_cond: Filter condition
            context: Filter context
            
        Returns:
            True if result passes the filter
        """
        try:
            # Extract field value from result
            field_value = self._extract_field_value(result, filter_cond.field)
            
            # Handle null values
            if field_value is None:
                if context.null_handling == "exclude":
                    return False
                elif context.null_handling == "include":
                    return True
                # else: treat as actual value (None)
            
            # Get operator function
            operator_func = self.operators.get(filter_cond.operator)
            if not operator_func:
                self.logger.warning(f"Unknown operator: {filter_cond.operator}")
                return True  # Skip unknown operators
            
            # Apply type coercion if needed
            if context.type_coercion:
                field_value, filter_value = self._coerce_types(field_value, filter_cond.value)
            else:
                filter_value = filter_cond.value
            
            # Apply case sensitivity
            if isinstance(field_value, str) and isinstance(filter_value, str):
                if not filter_cond.case_sensitive:
                    field_value = field_value.lower()
                    filter_value = filter_value.lower()
            
            # Evaluate the condition
            return operator_func(field_value, filter_value)
            
        except Exception as e:
            self.logger.error(f"Error evaluating filter {filter_cond.field} {filter_cond.operator} {filter_cond.value}: {e}")
            return True  # Skip problematic filters
    
    def _extract_field_value(self, result: QueryResult, field_path: str) -> Any:
        """
        Extract field value from result using dot notation for nested fields.
        
        Args:
            result: Query result
            field_path: Field path (e.g., 'metadata.creation_date')
            
        Returns:
            Field value or None if not found
        """
        # Special fields
        if field_path == 'content':
            return result.content
        elif field_path == 'node_id':
            return result.node_id
        elif field_path == 'node_type':
            return result.node_type
        elif field_path == 'relevance_score':
            return result.relevance_score
        elif field_path == 'quality_score':
            return result.quality_score
        elif field_path == 'combined_score':
            return result.combined_score
        
        # Nested field access
        path_parts = field_path.split('.')
        current_value = result
        
        for part in path_parts:
            if hasattr(current_value, part):
                current_value = getattr(current_value, part)
            elif isinstance(current_value, dict) and part in current_value:
                current_value = current_value[part]
            else:
                return None
        
        return current_value
    
    def _coerce_types(self, field_value: Any, filter_value: Any) -> tuple:
        """
        Coerce field and filter values to compatible types.
        
        Args:
            field_value: Value from the result
            filter_value: Value from the filter
            
        Returns:
            Tuple of (coerced_field_value, coerced_filter_value)
        """
        if field_value is None or filter_value is None:
            return field_value, filter_value
        
        # If types already match, return as is
        if type(field_value) == type(filter_value):
            return field_value, filter_value
        
        # Try to convert filter value to field value type
        try:
            if isinstance(field_value, (int, float)) and isinstance(filter_value, str):
                if isinstance(field_value, int):
                    return field_value, int(filter_value)
                else:
                    return field_value, float(filter_value)
            elif isinstance(field_value, str) and isinstance(filter_value, (int, float)):
                return field_value, str(filter_value)
            elif isinstance(field_value, bool) and isinstance(filter_value, str):
                return field_value, filter_value.lower() in ('true', '1', 'yes', 'on')
        except (ValueError, TypeError):
            pass
        
        return field_value, filter_value
    
    def _convert_to_datetime(self, value: Any) -> Optional[datetime]:
        """Convert value to datetime object."""
        if isinstance(value, datetime):
            return value
        elif isinstance(value, (int, float)):
            return datetime.fromtimestamp(value)
        elif isinstance(value, str):
            # Try common datetime formats
            formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y/%m/%d %H:%M:%S',
                '%Y/%m/%d'
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue
        return None
    
    # Operator implementations
    def _op_equals(self, field_value: Any, filter_value: Any) -> bool:
        return field_value == filter_value
    
    def _op_not_equals(self, field_value: Any, filter_value: Any) -> bool:
        return field_value != filter_value
    
    def _op_greater_than(self, field_value: Any, filter_value: Any) -> bool:
        try:
            return field_value > filter_value
        except TypeError:
            return False
    
    def _op_greater_than_equal(self, field_value: Any, filter_value: Any) -> bool:
        try:
            return field_value >= filter_value
        except TypeError:
            return False
    
    def _op_less_than(self, field_value: Any, filter_value: Any) -> bool:
        try:
            return field_value < filter_value
        except TypeError:
            return False
    
    def _op_less_than_equal(self, field_value: Any, filter_value: Any) -> bool:
        try:
            return field_value <= filter_value
        except TypeError:
            return False
    
    def _op_in(self, field_value: Any, filter_value: Any) -> bool:
        if not isinstance(filter_value, (list, tuple, set)):
            return field_value == filter_value
        return field_value in filter_value
    
    def _op_not_in(self, field_value: Any, filter_value: Any) -> bool:
        if not isinstance(filter_value, (list, tuple, set)):
            return field_value != filter_value
        return field_value not in filter_value
    
    def _op_contains(self, field_value: Any, filter_value: Any) -> bool:
        if isinstance(field_value, str) and isinstance(filter_value, str):
            return filter_value in field_value
        elif isinstance(field_value, (list, tuple)):
            return filter_value in field_value
        return False
    
    def _op_not_contains(self, field_value: Any, filter_value: Any) -> bool:
        return not self._op_contains(field_value, filter_value)
    
    def _op_starts_with(self, field_value: Any, filter_value: Any) -> bool:
        if isinstance(field_value, str) and isinstance(filter_value, str):
            return field_value.startswith(filter_value)
        return False
    
    def _op_ends_with(self, field_value: Any, filter_value: Any) -> bool:
        if isinstance(field_value, str) and isinstance(filter_value, str):
            return field_value.endswith(filter_value)
        return False
    
    def _op_regex(self, field_value: Any, filter_value: Any) -> bool:
        if isinstance(field_value, str) and isinstance(filter_value, str):
            try:
                # Validate regex pattern for security
                if len(filter_value) > 1000:  # Prevent extremely long patterns
                    return False
                
                # Compile pattern to validate it
                pattern = re.compile(filter_value)
                return pattern.search(field_value) is not None
            except re.error as e:
                self.logger.warning(f"Invalid regex pattern '{filter_value}': {e}")
                return False
        return False
    
    def _op_exists(self, field_value: Any, filter_value: Any) -> bool:
        return field_value is not None
    
    def _op_not_exists(self, field_value: Any, filter_value: Any) -> bool:
        return field_value is None
    
    def _op_empty(self, field_value: Any, filter_value: Any) -> bool:
        if field_value is None:
            return True
        if isinstance(field_value, (str, list, tuple, dict)):
            return len(field_value) == 0
        return False
    
    def _op_not_empty(self, field_value: Any, filter_value: Any) -> bool:
        return not self._op_empty(field_value, filter_value)
    
    def _op_between(self, field_value: Any, filter_value: Any) -> bool:
        if not isinstance(filter_value, (list, tuple)) or len(filter_value) != 2:
            return False
        try:
            return filter_value[0] <= field_value <= filter_value[1]
        except TypeError:
            return False
    
    def _op_date_range(self, field_value: Any, filter_value: Any) -> bool:
        field_date = self._convert_to_datetime(field_value)
        if field_date is None:
            return False
        
        if isinstance(filter_value, dict):
            start_date = self._convert_to_datetime(filter_value.get('start'))
            end_date = self._convert_to_datetime(filter_value.get('end'))
            
            if start_date and end_date:
                return start_date <= field_date <= end_date
            elif start_date:
                return field_date >= start_date
            elif end_date:
                return field_date <= end_date
        
        return False
    
    def _op_array_contains(self, field_value: Any, filter_value: Any) -> bool:
        if isinstance(field_value, (list, tuple)):
            return filter_value in field_value
        return False
    
    def _op_array_length(self, field_value: Any, filter_value: Any) -> bool:
        if isinstance(field_value, (list, tuple, str)):
            if isinstance(filter_value, dict):
                length = len(field_value)
                op = filter_value.get('op', 'eq')
                value = filter_value.get('value', 0)
                
                if op == 'eq':
                    return length == value
                elif op == 'gt':
                    return length > value
                elif op == 'lt':
                    return length < value
                elif op == 'gte':
                    return length >= value
                elif op == 'lte':
                    return length <= value
            else:
                return len(field_value) == filter_value
        return False


class AggregationProcessor:
    """
    Advanced aggregation processor supporting complex aggregation operations.
    
    Features:
    - Basic aggregations (count, sum, avg, min, max)
    - Group-by operations
    - Having clauses
    - Nested aggregations
    - Statistical functions
    - Custom aggregation functions
    """
    
    def __init__(self, filter_processor: FilterProcessor):
        self.filter_processor = filter_processor
        self.logger = logging.getLogger(__name__)
        
        # Aggregation function implementations
        self.aggregation_functions = {
            AggregationType.COUNT: self._agg_count,
            AggregationType.SUM: self._agg_sum,
            AggregationType.AVERAGE: self._agg_average,
            AggregationType.MIN: self._agg_min,
            AggregationType.MAX: self._agg_max,
            AggregationType.GROUP_BY: self._agg_group_by,
        }
    
    def process_aggregations(self, 
                           results: List[QueryResult],
                           aggregations: List[AggregationRequest]) -> List[AggregationResult]:
        """
        Process aggregation requests on query results.
        
        Args:
            results: Query results to aggregate
            aggregations: List of aggregation requests
            
        Returns:
            List of aggregation results
        """
        if not aggregations:
            return []
        
        aggregation_results = []
        
        for agg_request in aggregations:
            try:
                # Apply having filters if specified
                filtered_results = results
                if agg_request.having:
                    filtered_results = self.filter_processor.apply_filters(
                        results, agg_request.having
                    )
                
                # Process the aggregation
                agg_func = self.aggregation_functions.get(agg_request.type)
                if agg_func:
                    agg_results = agg_func(filtered_results, agg_request)
                    aggregation_results.extend(agg_results)
                else:
                    self.logger.warning(f"Unknown aggregation type: {agg_request.type}")
                    
            except Exception as e:
                self.logger.error(f"Error processing aggregation {agg_request.type}: {e}")
        
        return aggregation_results
    
    def _agg_count(self, results: List[QueryResult], request: AggregationRequest) -> List[AggregationResult]:
        """Count aggregation."""
        return [AggregationResult(
            aggregation_type='count',
            field=request.field,
            value=len(results)
        )]
    
    def _agg_sum(self, results: List[QueryResult], request: AggregationRequest) -> List[AggregationResult]:
        """Sum aggregation."""
        if not request.field:
            return []
        
        values = []
        for result in results:
            value = self.filter_processor._extract_field_value(result, request.field)
            if isinstance(value, (int, float)):
                values.append(value)
        
        return [AggregationResult(
            aggregation_type='sum',
            field=request.field,
            value=sum(values),
            count=len(values)
        )]
    
    def _agg_average(self, results: List[QueryResult], request: AggregationRequest) -> List[AggregationResult]:
        """Average aggregation."""
        if not request.field:
            return []
        
        values = []
        for result in results:
            value = self.filter_processor._extract_field_value(result, request.field)
            if isinstance(value, (int, float)):
                values.append(value)
        
        avg_value = statistics.mean(values) if values else 0
        
        return [AggregationResult(
            aggregation_type='avg',
            field=request.field,
            value=avg_value,
            count=len(values)
        )]
    
    def _agg_min(self, results: List[QueryResult], request: AggregationRequest) -> List[AggregationResult]:
        """Minimum aggregation."""
        if not request.field:
            return []
        
        values = []
        for result in results:
            value = self.filter_processor._extract_field_value(result, request.field)
            if value is not None:
                values.append(value)
        
        min_value = min(values) if values else None
        
        return [AggregationResult(
            aggregation_type='min',
            field=request.field,
            value=min_value,
            count=len(values)
        )]
    
    def _agg_max(self, results: List[QueryResult], request: AggregationRequest) -> List[AggregationResult]:
        """Maximum aggregation."""
        if not request.field:
            return []
        
        values = []
        for result in results:
            value = self.filter_processor._extract_field_value(result, request.field)
            if value is not None:
                values.append(value)
        
        max_value = max(values) if values else None
        
        return [AggregationResult(
            aggregation_type='max',
            field=request.field,
            value=max_value,
            count=len(values)
        )]
    
    def _agg_group_by(self, results: List[QueryResult], request: AggregationRequest) -> List[AggregationResult]:
        """Group by aggregation."""
        if not request.group_by:
            return []
        
        # Group results by specified fields
        groups = defaultdict(list)
        
        for result in results:
            group_key_parts = []
            for field in request.group_by:
                value = self.filter_processor._extract_field_value(result, field)
                group_key_parts.append(str(value) if value is not None else 'null')
            
            group_key = '|'.join(group_key_parts)
            groups[group_key].append(result)
        
        # Create aggregation results for each group
        agg_results = []
        for group_key, group_results in groups.items():
            agg_results.append(AggregationResult(
                aggregation_type='group_by',
                field='|'.join(request.group_by),
                value=len(group_results),
                group_key=group_key,
                count=len(group_results)
            ))
        
        return agg_results