"""
Natural Language Query Processor

Converts natural language queries into structured graph queries using LLM.
"""

import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from memory_core.config.config_manager import get_config
from memory_core.llm.providers.gemini import GeminiLLMProvider
from .query_types import QueryType, FilterCondition, SortCriteria, SortOrder, GraphPattern


@dataclass
class ParsedQuery:
    """Result of natural language query parsing."""
    intent: str
    entities: List[str]
    relationships: List[str]
    constraints: List[str]
    query_type: QueryType
    graph_pattern: Optional[GraphPattern] = None
    semantic_keywords: List[str] = None
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'intent': self.intent,
            'entities': self.entities,
            'relationships': self.relationships,
            'constraints': self.constraints,
            'query_type': self.query_type.value,
            'graph_pattern': self.graph_pattern.to_dict() if self.graph_pattern else None,
            'semantic_keywords': self.semantic_keywords,
            'confidence': self.confidence
        }


class NaturalLanguageQueryProcessor:
    """
    Processes natural language queries and converts them to structured graph queries.
    
    Uses modular LLM provider system to understand query intent and extract relevant graph patterns.
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize LLM provider
        api_key = self.config.config.api.google_api_key
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not configured for natural language processing")
        
        llm_config = {
            'api_key': api_key,
            'model_name': self.config.config.llm.model,
            'temperature': self.config.config.llm.temperature,
            'max_tokens': self.config.config.llm.max_tokens,
            'timeout': self.config.config.llm.timeout,
        }
        
        self.llm_provider = GeminiLLMProvider(llm_config)
        
        # Common patterns for query classification
        self.query_patterns = {
            'find_nodes': [
                r'find|search|get|show|list',
                r'nodes?|entities?|items?|things?',
            ],
            'find_relationships': [
                r'connect|relation|link|associate',
                r'between|from|to|with',
            ],
            'count_aggregation': [
                r'how many|count|number of',
            ],
            'similarity_search': [
                r'similar|like|related|comparable',
            ],
            'complex_pattern': [
                r'where|when|that|which',
                r'and|or|not',
            ]
        }
    
    async def process_query(self, query: str, context: Optional[str] = None) -> ParsedQuery:
        """
        Process a natural language query and extract structured information.
        
        Args:
            query: Natural language query string
            context: Optional context to help with understanding
            
        Returns:
            ParsedQuery with extracted information
        """
        self.logger.info(f"Processing natural language query: {query[:100]}...")
        
        # First, try rule-based classification for simple patterns
        basic_classification = self._classify_query_basic(query)
        
        # Use LLM for complex parsing
        llm_result = await self._parse_with_llm(query, context)
        
        # Combine results
        parsed_query = self._combine_parsing_results(query, basic_classification, llm_result)
        
        self.logger.info(f"Parsed query intent: {parsed_query.intent}, type: {parsed_query.query_type}")
        return parsed_query
    
    def _classify_query_basic(self, query: str) -> Dict[str, Any]:
        """
        Basic rule-based query classification for common patterns.
        
        Args:
            query: Query string
            
        Returns:
            Basic classification results
        """
        query_lower = query.lower()
        classification = {
            'patterns_matched': [],
            'likely_type': QueryType.NATURAL_LANGUAGE,
            'confidence': 0.0
        }
        
        for pattern_type, patterns in self.query_patterns.items():
            matches = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    matches += 1
            
            if matches > 0:
                classification['patterns_matched'].append(pattern_type)
                classification['confidence'] += matches * 0.2
        
        # Determine most likely query type
        if 'find_nodes' in classification['patterns_matched']:
            if 'similar' in query_lower:
                classification['likely_type'] = QueryType.SEMANTIC_SEARCH
            else:
                classification['likely_type'] = QueryType.GRAPH_PATTERN
        elif 'find_relationships' in classification['patterns_matched']:
            classification['likely_type'] = QueryType.RELATIONSHIP_SEARCH
        elif 'count_aggregation' in classification['patterns_matched']:
            classification['likely_type'] = QueryType.AGGREGATION
        elif 'similarity_search' in classification['patterns_matched']:
            classification['likely_type'] = QueryType.SEMANTIC_SEARCH
        
        classification['confidence'] = min(classification['confidence'], 1.0)
        return classification
    
    async def _parse_with_llm(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Use LLM to parse complex natural language queries.
        
        Args:
            query: Natural language query
            context: Optional context
            
        Returns:
            LLM parsing results
        """
        try:
            # Ensure LLM provider is connected
            if not self.llm_provider.is_connected:
                await self.llm_provider.connect()
            
            # Use the LLM provider's natural language query parsing method
            result = await self.llm_provider.parse_natural_language_query(query, context)
            return result
            
        except Exception as e:
            self.logger.error(f"Error parsing query with LLM: {e}")
            return {
                'intent': 'search',
                'entities': [],
                'relationships': [],
                'constraints': [],
                'query_type': 'natural_language',
                'confidence': 0.0
            }
    
    
    def _combine_parsing_results(self, original_query: str, basic: Dict[str, Any], llm: Dict[str, Any]) -> ParsedQuery:
        """
        Combine basic pattern matching with LLM results.
        
        Args:
            original_query: Original query string
            basic: Basic pattern matching results
            llm: LLM parsing results
            
        Returns:
            Combined ParsedQuery
        """
        # Use LLM results as primary, fall back to basic for missing info
        intent = llm.get('intent', 'search')
        entities = llm.get('entities', [])
        relationships = llm.get('relationships', [])
        constraints = llm.get('constraints', [])
        semantic_keywords = llm.get('semantic_keywords', [])
        
        # Determine query type (prefer LLM, but validate against basic patterns)
        llm_type_str = llm.get('query_type', 'natural_language')
        try:
            query_type = QueryType(llm_type_str)
        except ValueError:
            query_type = basic.get('likely_type', QueryType.NATURAL_LANGUAGE)
        
        # Create graph pattern if provided
        graph_pattern = None
        if 'graph_pattern' in llm and llm['graph_pattern']:
            gp_data = llm['graph_pattern']
            graph_pattern = GraphPattern(
                nodes=gp_data.get('nodes', []),
                edges=gp_data.get('edges', []),
                constraints=gp_data.get('constraints', [])
            )
        
        # Combine confidence scores
        basic_confidence = basic.get('confidence', 0.0)
        llm_confidence = llm.get('confidence', 0.0)
        combined_confidence = (basic_confidence + llm_confidence * 2) / 3  # Weight LLM higher
        
        return ParsedQuery(
            intent=intent,
            entities=entities,
            relationships=relationships,
            constraints=constraints,
            query_type=query_type,
            graph_pattern=graph_pattern,
            semantic_keywords=semantic_keywords or entities,  # Fall back to entities
            confidence=combined_confidence
        )
    
    def extract_filters_from_query(self, query: str) -> List[FilterCondition]:
        """
        Extract filter conditions from natural language query.
        
        Args:
            query: Natural language query
            
        Returns:
            List of extracted filter conditions
        """
        filters = []
        query_lower = query.lower()
        
        # Common filter patterns
        patterns = [
            (r'created after (\d{4})', 'creation_date', 'gt'),
            (r'created before (\d{4})', 'creation_date', 'lt'),
            (r'with rating above ([\d.]+)', 'rating', 'gt'),
            (r'with rating below ([\d.]+)', 'rating', 'lt'),
            (r'type is (\w+)', 'node_type', 'eq'),
            (r'contains? ["\']([^"\']+)["\']', 'content', 'contains'),
        ]
        
        for pattern, field, operator in patterns:
            matches = re.finditer(pattern, query_lower)
            for match in matches:
                value = match.group(1)
                # Convert numeric values
                if operator in ['gt', 'lt', 'gte', 'lte']:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                
                filters.append(FilterCondition(
                    field=field,
                    operator=operator,
                    value=value
                ))
        
        return filters
    
    def suggest_query_improvements(self, query: str, parsed: ParsedQuery) -> List[str]:
        """
        Suggest improvements to make the query more effective.
        
        Args:
            query: Original query
            parsed: Parsed query result
            
        Returns:
            List of suggestion strings
        """
        suggestions = []
        
        if parsed.confidence < 0.5:
            suggestions.append("Consider making your query more specific")
        
        if not parsed.entities and parsed.query_type != QueryType.AGGREGATION:
            suggestions.append("Try including specific entities or concepts you're looking for")
        
        if len(query.split()) < 3:
            suggestions.append("Add more context to improve search accuracy")
        
        if parsed.query_type == QueryType.SEMANTIC_SEARCH and not parsed.semantic_keywords:
            suggestions.append("Include key terms that describe what you're looking for")
        
        return suggestions