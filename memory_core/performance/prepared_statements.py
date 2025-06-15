"""
Prepared statement support for graph backends with optimization.
"""

import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import threading
import logging

logger = logging.getLogger(__name__)


class StatementType(Enum):
    """Types of prepared statements."""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    TRAVERSAL = "traversal"
    BULK_OPERATION = "bulk_operation"


@dataclass
class PreparedStatement:
    """Prepared statement with metadata."""
    statement_id: str
    query_template: str
    statement_type: StatementType
    parameter_names: List[str]
    created_at: float
    use_count: int = 0
    last_used: float = 0.0
    avg_execution_time: float = 0.0
    compiled_query: Optional[Any] = None
    
    def record_execution(self, execution_time: float):
        """Record statement execution metrics."""
        self.use_count += 1
        self.last_used = time.time()
        if self.avg_execution_time == 0:
            self.avg_execution_time = execution_time
        else:
            self.avg_execution_time = (
                (self.avg_execution_time * (self.use_count - 1) + execution_time) / self.use_count
            )


class PreparedStatementManager:
    """Manager for prepared statements with caching and optimization."""
    
    def __init__(self, max_statements: int = 1000, cleanup_interval: int = 300):
        self.max_statements = max_statements
        self.cleanup_interval = cleanup_interval
        self._statements: Dict[str, PreparedStatement] = {}
        self._lock = threading.RLock()
        self._last_cleanup = time.time()
    
    def prepare(self, query_template: str, statement_type: StatementType, 
                parameter_names: List[str] = None) -> str:
        """Prepare a statement and return its ID."""
        statement_id = self._generate_statement_id(query_template)
        
        with self._lock:
            if statement_id in self._statements:
                return statement_id
            
            # Clean up old statements if needed
            if len(self._statements) >= self.max_statements:
                self._cleanup_old_statements()
            
            # Create new prepared statement
            statement = PreparedStatement(
                statement_id=statement_id,
                query_template=query_template,
                statement_type=statement_type,
                parameter_names=parameter_names or [],
                created_at=time.time()
            )
            
            # Compile query if possible
            statement.compiled_query = self._compile_query(query_template, statement_type)
            
            self._statements[statement_id] = statement
            return statement_id
    
    def execute(self, statement_id: str, parameters: Dict[str, Any], 
                connection: Any) -> Any:
        """Execute a prepared statement."""
        with self._lock:
            statement = self._statements.get(statement_id)
            if not statement:
                raise ValueError(f"Prepared statement {statement_id} not found")
            
            start_time = time.time()
            try:
                result = self._execute_statement(statement, parameters, connection)
                execution_time = time.time() - start_time
                statement.record_execution(execution_time)
                return result
            except Exception as e:
                logger.error(f"Error executing prepared statement {statement_id}: {e}")
                raise
    
    def _execute_statement(self, statement: PreparedStatement, 
                          parameters: Dict[str, Any], connection: Any) -> Any:
        """Execute the prepared statement."""
        try:
            # Use compiled query if available
            if statement.compiled_query:
                return self._execute_compiled(statement.compiled_query, parameters, connection)
            
            # Otherwise, substitute parameters in template
            query = self._substitute_parameters(statement.query_template, parameters)
            
            # Execute based on statement type
            if statement.statement_type == StatementType.TRAVERSAL:
                return self._execute_traversal(query, connection)
            else:
                return self._execute_query(query, connection)
                
        except Exception as e:
            logger.error(f"Statement execution failed: {e}")
            raise
    
    def _execute_compiled(self, compiled_query: Any, parameters: Dict[str, Any], 
                         connection: Any) -> Any:
        """Execute compiled query."""
        # Implementation depends on backend type
        if hasattr(connection, 'submit'):
            # Gremlin/JanusGraph
            return connection.submit(compiled_query, parameters)
        elif hasattr(connection, 'execute'):
            # SQL-based
            return connection.execute(compiled_query, parameters)
        else:
            raise NotImplementedError("Unknown connection type for compiled query")
    
    def _execute_traversal(self, query: str, connection: Any) -> Any:
        """Execute graph traversal query."""
        if hasattr(connection, 'submit'):
            return connection.submit(query)
        else:
            raise NotImplementedError("Connection does not support traversal queries")
    
    def _execute_query(self, query: str, connection: Any) -> Any:
        """Execute regular query."""
        if hasattr(connection, 'execute'):
            return connection.execute(query)
        elif hasattr(connection, 'submit'):
            return connection.submit(query)
        else:
            raise NotImplementedError("Unknown connection type for query execution")
    
    def _compile_query(self, query_template: str, statement_type: StatementType) -> Optional[Any]:
        """Compile query template if possible."""
        try:
            # Gremlin query compilation
            if statement_type == StatementType.TRAVERSAL:
                return self._compile_gremlin_query(query_template)
            
            # SQL query compilation would go here
            return None
            
        except Exception as e:
            logger.debug(f"Could not compile query: {e}")
            return None
    
    def _compile_gremlin_query(self, query_template: str) -> Optional[str]:
        """Compile Gremlin query template."""
        # Basic optimization - remove unnecessary whitespace
        compiled = ' '.join(query_template.split())
        
        # Could add more sophisticated optimizations:
        # - Query rewriting for better performance
        # - Index hint injection
        # - Subquery optimization
        
        return compiled
    
    def _substitute_parameters(self, template: str, parameters: Dict[str, Any]) -> str:
        """Substitute parameters in query template."""
        query = template
        for param_name, param_value in parameters.items():
            placeholder = f"${{{param_name}}}"
            
            # Format value based on type
            if isinstance(param_value, str):
                formatted_value = f"'{param_value}'"
            elif isinstance(param_value, (int, float)):
                formatted_value = str(param_value)
            elif isinstance(param_value, bool):
                formatted_value = str(param_value).lower()
            elif isinstance(param_value, list):
                formatted_value = str(param_value)
            else:
                formatted_value = f"'{str(param_value)}'"
            
            query = query.replace(placeholder, formatted_value)
        
        return query
    
    def _generate_statement_id(self, query_template: str) -> str:
        """Generate unique statement ID."""
        return hashlib.md5(query_template.encode()).hexdigest()
    
    def _cleanup_old_statements(self):
        """Remove old unused statements."""
        current_time = time.time()
        
        # Sort by last used time and remove oldest
        statements_by_usage = sorted(
            self._statements.items(),
            key=lambda x: x[1].last_used or x[1].created_at
        )
        
        # Remove 20% of statements
        remove_count = max(1, len(statements_by_usage) // 5)
        for statement_id, _ in statements_by_usage[:remove_count]:
            del self._statements[statement_id]
        
        logger.info(f"Cleaned up {remove_count} prepared statements")
    
    def get_statement_stats(self) -> Dict[str, Any]:
        """Get prepared statement statistics."""
        with self._lock:
            stats = {
                'total_statements': len(self._statements),
                'statement_types': {},
                'top_statements': [],
                'avg_execution_times': {}
            }
            
            # Count by type
            for statement in self._statements.values():
                stmt_type = statement.statement_type.value
                stats['statement_types'][stmt_type] = stats['statement_types'].get(stmt_type, 0) + 1
            
            # Top used statements
            top_statements = sorted(
                self._statements.values(),
                key=lambda x: x.use_count,
                reverse=True
            )[:10]
            
            stats['top_statements'] = [
                {
                    'id': stmt.statement_id,
                    'type': stmt.statement_type.value,
                    'use_count': stmt.use_count,
                    'avg_execution_time': stmt.avg_execution_time
                }
                for stmt in top_statements
            ]
            
            return stats
    
    def clear(self):
        """Clear all prepared statements."""
        with self._lock:
            self._statements.clear()


class QueryBuilder:
    """Builder for creating optimized queries."""
    
    def __init__(self, backend_type: str):
        self.backend_type = backend_type
        self._query_parts = []
        self._parameters = {}
        self._statement_type = StatementType.SELECT
    
    def select(self, *fields) -> 'QueryBuilder':
        """Add SELECT clause."""
        self._statement_type = StatementType.SELECT
        if self.backend_type == 'gremlin':
            self._query_parts.append("g.V()")
        elif self.backend_type == 'sql':
            field_list = ', '.join(fields) if fields else '*'
            self._query_parts.append(f"SELECT {field_list}")
        return self
    
    def where(self, condition: str, **params) -> 'QueryBuilder':
        """Add WHERE clause."""
        if self.backend_type == 'gremlin':
            self._query_parts.append(f".has({condition})")
        elif self.backend_type == 'sql':
            self._query_parts.append(f"WHERE {condition}")
        
        self._parameters.update(params)
        return self
    
    def limit(self, count: int) -> 'QueryBuilder':
        """Add LIMIT clause."""
        if self.backend_type == 'gremlin':
            self._query_parts.append(f".limit({count})")
        elif self.backend_type == 'sql':
            self._query_parts.append(f"LIMIT {count}")
        return self
    
    def order_by(self, field: str, direction: str = 'ASC') -> 'QueryBuilder':
        """Add ORDER BY clause."""
        if self.backend_type == 'gremlin':
            order_direction = 'incr' if direction.upper() == 'ASC' else 'decr'
            self._query_parts.append(f".order().by('{field}', {order_direction})")
        elif self.backend_type == 'sql':
            self._query_parts.append(f"ORDER BY {field} {direction}")
        return self
    
    def build(self) -> Tuple[str, Dict[str, Any], StatementType]:
        """Build the query."""
        if self.backend_type == 'gremlin':
            query = ''.join(self._query_parts)
            if self._statement_type == StatementType.SELECT:
                self._statement_type = StatementType.TRAVERSAL
        elif self.backend_type == 'sql':
            query = ' '.join(self._query_parts)
        else:
            raise ValueError(f"Unsupported backend type: {self.backend_type}")
        
        return query, self._parameters, self._statement_type


class BatchQueryExecutor:
    """Executor for batch queries with optimization."""
    
    def __init__(self, statement_manager: PreparedStatementManager):
        self.statement_manager = statement_manager
    
    async def execute_batch(self, queries: List[Tuple[str, Dict[str, Any]]], 
                           connection: Any) -> List[Any]:
        """Execute batch of queries."""
        results = []
        
        # Group queries by template for better prepared statement reuse
        query_groups = self._group_queries(queries)
        
        for template, query_params_list in query_groups.items():
            # Prepare statement once per template
            statement_id = self.statement_manager.prepare(
                template, 
                StatementType.SELECT  # Assuming SELECT for now
            )
            
            # Execute all queries with this template
            for params in query_params_list:
                try:
                    result = self.statement_manager.execute(statement_id, params, connection)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch query failed: {e}")
                    results.append(None)
        
        return results
    
    def _group_queries(self, queries: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group queries by template."""
        groups = {}
        
        for query_template, params in queries:
            if query_template not in groups:
                groups[query_template] = []
            groups[query_template].append(params)
        
        return groups