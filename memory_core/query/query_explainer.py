"""
Query Explanation System for Advanced Query Engine

Provides detailed explanations of how queries are processed, optimized, and executed.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from .query_types import (
    QueryRequest, QueryResponse, QueryExplanation, QueryExplanationStep, 
    QueryType, QueryResult
)


@dataclass
class ExplanationContext:
    """Context for generating query explanations."""
    include_timing: bool = True
    include_optimization_details: bool = True
    include_ranking_details: bool = True
    include_execution_plan: bool = True
    verbosity_level: int = 2  # 1=basic, 2=detailed, 3=verbose


class QueryExplainer:
    """
    Comprehensive query explanation system.
    
    Features:
    - Step-by-step query execution explanation
    - Performance analysis and bottleneck identification
    - Optimization recommendation
    - Visual execution plan representation
    - Natural language explanations
    - Interactive explanation queries
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_explanation(self, 
                           request: QueryRequest,
                           response: QueryResponse,
                           execution_steps: List[QueryExplanationStep],
                           context: Optional[ExplanationContext] = None) -> QueryExplanation:
        """
        Generate comprehensive query explanation.
        
        Args:
            request: Original query request
            response: Query response
            execution_steps: List of execution steps
            context: Explanation context
            
        Returns:
            Complete query explanation
        """
        context = context or ExplanationContext()
        
        # Parse the original query for explanation
        parsed_query = self._parse_query_for_explanation(request)
        
        # Generate translation steps
        translation_steps = self._generate_translation_steps(request, execution_steps)
        
        # Generate optimization summary
        optimizations = self._extract_optimizations(execution_steps)
        
        # Calculate total execution time
        total_time = sum(step.execution_time_ms or 0 for step in execution_steps)
        
        explanation = QueryExplanation(
            original_query=request.query,
            parsed_query=parsed_query,
            translation_steps=translation_steps,
            execution_plan=execution_steps,
            optimizations=optimizations,
            total_execution_time_ms=total_time,
            cache_hit=response.from_cache,
            cache_key=f"cached_{response.query_id}" if response.from_cache else None
        )
        
        return explanation
    
    def generate_natural_language_explanation(self, 
                                            explanation: QueryExplanation,
                                            context: Optional[ExplanationContext] = None) -> str:
        """
        Generate natural language explanation of query execution.
        
        Args:
            explanation: Query explanation object
            context: Explanation context
            
        Returns:
            Natural language explanation string
        """
        context = context or ExplanationContext()
        explanation_parts = []
        
        # Query interpretation
        explanation_parts.append(self._explain_query_interpretation(explanation))
        
        # Execution overview
        explanation_parts.append(self._explain_execution_overview(explanation, context))
        
        # Performance analysis
        if context.include_timing:
            explanation_parts.append(self._explain_performance_analysis(explanation))
        
        # Optimization details
        if context.include_optimization_details and explanation.optimizations:
            explanation_parts.append(self._explain_optimizations(explanation))
        
        # Result summary
        explanation_parts.append(self._explain_result_summary(explanation))
        
        return "\n\n".join(explanation_parts)
    
    def generate_visual_execution_plan(self, explanation: QueryExplanation) -> Dict[str, Any]:
        """
        Generate visual representation of execution plan.
        
        Args:
            explanation: Query explanation
            
        Returns:
            Dictionary representing visual execution plan
        """
        plan = {
            "type": "execution_plan",
            "query": explanation.original_query,
            "total_time_ms": explanation.total_execution_time_ms,
            "steps": []
        }
        
        for i, step in enumerate(explanation.execution_plan):
            step_data = {
                "step_number": i + 1,
                "name": step.step_name,
                "description": step.description,
                "operation": step.operation,
                "time_ms": step.execution_time_ms,
                "time_percentage": (step.execution_time_ms / explanation.total_execution_time_ms * 100) 
                                 if explanation.total_execution_time_ms > 0 else 0,
                "input_size": step.input_size,
                "output_size": step.output_size,
                "optimizations": step.optimizations_applied,
                "details": step.details
            }
            plan["steps"].append(step_data)
        
        return plan
    
    def analyze_performance_bottlenecks(self, explanation: QueryExplanation) -> List[Dict[str, Any]]:
        """
        Analyze execution plan for performance bottlenecks.
        
        Args:
            explanation: Query explanation
            
        Returns:
            List of identified bottlenecks with recommendations
        """
        bottlenecks = []
        total_time = explanation.total_execution_time_ms
        
        if total_time == 0:
            return bottlenecks
        
        for step in explanation.execution_plan:
            step_time = step.execution_time_ms or 0
            time_percentage = (step_time / total_time) * 100
            
            # Identify slow steps
            if time_percentage > 30:  # Step takes more than 30% of total time
                bottleneck = {
                    "type": "slow_step",
                    "step_name": step.step_name,
                    "time_ms": step_time,
                    "time_percentage": time_percentage,
                    "description": f"Step '{step.step_name}' is consuming {time_percentage:.1f}% of execution time",
                    "recommendations": self._get_step_optimization_recommendations(step)
                }
                bottlenecks.append(bottleneck)
            
            # Check for inefficient operations
            if step.operation == "filter" and step.input_size and step.output_size:
                filter_ratio = step.output_size / step.input_size
                if filter_ratio < 0.1:  # Filter eliminates >90% of results
                    bottleneck = {
                        "type": "inefficient_filter",
                        "step_name": step.step_name,
                        "filter_ratio": filter_ratio,
                        "description": f"Filter is very selective ({filter_ratio:.1%} pass rate), consider applying earlier",
                        "recommendations": [
                            "Move selective filters earlier in execution",
                            "Consider adding indexes for filter fields",
                            "Optimize filter conditions for better performance"
                        ]
                    }
                    bottlenecks.append(bottleneck)
        
        # Check for missing optimizations
        if explanation.total_execution_time_ms > 1000 and len(explanation.optimizations) < 2:
            bottleneck = {
                "type": "missing_optimizations",
                "description": "Query is slow but few optimizations were applied",
                "recommendations": [
                    "Consider adding more specific filters",
                    "Reduce result set size with LIMIT",
                    "Use more selective query patterns",
                    "Check if appropriate indexes exist"
                ]
            }
            bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    def generate_query_optimization_suggestions(self, 
                                              request: QueryRequest,
                                              explanation: QueryExplanation) -> List[str]:
        """
        Generate suggestions for optimizing the query.
        
        Args:
            request: Original query request
            explanation: Query explanation
            
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        
        # Query-specific suggestions
        if request.query_type == QueryType.NATURAL_LANGUAGE:
            suggestions.append("Consider using more specific query types (semantic_search, graph_pattern) for better performance")
        
        if not request.limit:
            suggestions.append("Add a LIMIT to reduce result set size and improve performance")
        
        if request.include_relationships and request.max_depth > 2:
            suggestions.append("Consider reducing max_depth for relationship traversal to improve performance")
        
        if len(request.filters) == 0 and explanation.total_execution_time_ms > 500:
            suggestions.append("Add filters to reduce the search space and improve query performance")
        
        # Performance-based suggestions
        if explanation.total_execution_time_ms > 2000:
            suggestions.append("Query is slow (>2s). Consider breaking it into smaller, more specific queries")
        
        if not explanation.cache_hit and explanation.total_execution_time_ms > 100:
            suggestions.append("Enable caching for frequently executed queries")
        
        # Result-based suggestions
        execution_plan = explanation.execution_plan
        if execution_plan:
            result_step = next((s for s in execution_plan if s.operation == "graph_query"), None)
            if result_step and result_step.output_size and result_step.output_size > 1000:
                suggestions.append("Large result set detected. Consider adding more selective filters")
        
        return suggestions
    
    def _parse_query_for_explanation(self, request: QueryRequest) -> Dict[str, Any]:
        """Parse query request for explanation purposes."""
        return {
            "query_text": request.query,
            "query_type": request.query_type.value,
            "filters_count": len(request.filters),
            "has_aggregations": len(request.aggregations) > 0,
            "includes_relationships": request.include_relationships,
            "max_depth": request.max_depth,
            "limit": request.limit,
            "similarity_threshold": request.similarity_threshold
        }
    
    def _generate_translation_steps(self, 
                                  request: QueryRequest,
                                  execution_steps: List[QueryExplanationStep]) -> List[str]:
        """Generate human-readable translation steps."""
        steps = []
        
        # Initial query interpretation
        if request.query_type == QueryType.NATURAL_LANGUAGE:
            steps.append(f"Interpreted natural language query: '{request.query}'")
            nlp_step = next((s for s in execution_steps if s.step_name == "natural_language_processing"), None)
            if nlp_step and nlp_step.details:
                intent = nlp_step.details.get("intent", "unknown")
                entities = nlp_step.details.get("entities", [])
                steps.append(f"Detected intent: {intent}")
                if entities:
                    steps.append(f"Identified entities: {', '.join(entities)}")
        
        # Query optimization
        opt_step = next((s for s in execution_steps if s.step_name == "query_optimization"), None)
        if opt_step and opt_step.optimizations_applied:
            steps.append(f"Applied optimizations: {', '.join(opt_step.optimizations_applied)}")
        
        # Execution strategy
        exec_step = next((s for s in execution_steps if s.step_name == "query_execution"), None)
        if exec_step:
            steps.append(f"Executed {request.query_type.value} against graph database")
        
        # Filtering
        filter_step = next((s for s in execution_steps if s.step_name == "filter_application"), None)
        if filter_step:
            steps.append(f"Applied {len(request.filters)} filters to results")
        
        # Ranking
        rank_step = next((s for s in execution_steps if s.step_name == "result_ranking"), None)
        if rank_step:
            steps.append("Ranked results by relevance and quality scores")
        
        # Aggregation
        agg_step = next((s for s in execution_steps if s.step_name == "aggregation"), None)
        if agg_step:
            steps.append(f"Computed {len(request.aggregations)} aggregations")
        
        return steps
    
    def _extract_optimizations(self, execution_steps: List[QueryExplanationStep]) -> List[str]:
        """Extract all optimizations from execution steps."""
        optimizations = []
        for step in execution_steps:
            if step.optimizations_applied:
                optimizations.extend(step.optimizations_applied)
        return list(set(optimizations))  # Remove duplicates
    
    def _explain_query_interpretation(self, explanation: QueryExplanation) -> str:
        """Generate explanation of query interpretation."""
        parsed = explanation.parsed_query
        
        text = f"**Query Interpretation:**\n"
        text += f"Original query: '{explanation.original_query}'\n"
        text += f"Query type: {parsed.get('query_type', 'unknown')}\n"
        
        if parsed.get('filters_count', 0) > 0:
            text += f"Filters applied: {parsed['filters_count']}\n"
        
        if parsed.get('has_aggregations'):
            text += "Includes aggregations\n"
        
        if parsed.get('includes_relationships'):
            text += f"Includes relationships (max depth: {parsed.get('max_depth', 'unlimited')})\n"
        
        return text
    
    def _explain_execution_overview(self, explanation: QueryExplanation, context: ExplanationContext) -> str:
        """Generate execution overview explanation."""
        text = f"**Execution Overview:**\n"
        
        if explanation.cache_hit:
            text += "✓ Result served from cache\n"
        else:
            text += f"Executed in {explanation.total_execution_time_ms:.1f}ms across {len(explanation.execution_plan)} steps\n"
        
        # Step summary
        if context.verbosity_level >= 2:
            text += "\nExecution steps:\n"
            for i, step in enumerate(explanation.execution_plan, 1):
                time_info = f" ({step.execution_time_ms:.1f}ms)" if step.execution_time_ms else ""
                text += f"{i}. {step.description}{time_info}\n"
        
        return text
    
    def _explain_performance_analysis(self, explanation: QueryExplanation) -> str:
        """Generate performance analysis explanation."""
        text = f"**Performance Analysis:**\n"
        
        total_time = explanation.total_execution_time_ms
        
        if total_time < 100:
            text += "✓ Excellent performance (< 100ms)\n"
        elif total_time < 500:
            text += "✓ Good performance (< 500ms)\n"
        elif total_time < 1000:
            text += "⚠ Moderate performance (< 1s)\n"
        else:
            text += "⚠ Slow performance (> 1s)\n"
        
        # Identify slowest step
        slowest_step = max(explanation.execution_plan, 
                          key=lambda s: s.execution_time_ms or 0,
                          default=None)
        
        if slowest_step and slowest_step.execution_time_ms:
            percentage = (slowest_step.execution_time_ms / total_time) * 100
            text += f"Slowest step: {slowest_step.step_name} ({percentage:.1f}% of total time)\n"
        
        return text
    
    def _explain_optimizations(self, explanation: QueryExplanation) -> str:
        """Generate optimization explanation."""
        text = f"**Applied Optimizations:**\n"
        
        for optimization in explanation.optimizations:
            text += f"• {optimization}\n"
        
        if not explanation.optimizations:
            text += "No optimizations were applied\n"
        
        return text
    
    def _explain_result_summary(self, explanation: QueryExplanation) -> str:
        """Generate result summary explanation."""
        # This would need access to the response object
        # For now, provide a basic summary
        text = f"**Result Summary:**\n"
        
        exec_step = next((s for s in explanation.execution_plan 
                         if s.step_name == "query_execution"), None)
        
        if exec_step and exec_step.details:
            results_found = exec_step.details.get("results_found", 0)
            text += f"Found {results_found} matching results\n"
        
        rank_step = next((s for s in explanation.execution_plan 
                         if s.step_name == "result_ranking"), None)
        
        if rank_step and rank_step.details:
            top_score = rank_step.details.get("top_score", 0)
            text += f"Top result score: {top_score:.3f}\n"
        
        return text
    
    def _get_step_optimization_recommendations(self, step: QueryExplanationStep) -> List[str]:
        """Get optimization recommendations for a specific step."""
        recommendations = []
        
        if step.operation == "vector_search":
            recommendations.extend([
                "Consider increasing similarity threshold to reduce search space",
                "Use more specific query terms",
                "Limit result size with appropriate LIMIT clause"
            ])
        elif step.operation == "graph_traversal":
            recommendations.extend([
                "Reduce max_depth if deep traversal isn't necessary",
                "Add more selective filters before traversal",
                "Consider using relationship-specific queries"
            ])
        elif step.operation == "filter":
            recommendations.extend([
                "Ensure indexes exist for filter fields",
                "Reorder filters to put most selective first",
                "Consider combining multiple filters into compound conditions"
            ])
        elif step.operation == "ranking":
            recommendations.extend([
                "Reduce ranking complexity for large result sets",
                "Consider pre-computing quality scores",
                "Use simpler ranking criteria for real-time queries"
            ])
        
        return recommendations