"""
Natural Language Query Processor

Converts natural language queries into structured graph queries using LLM.
"""

import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from google import genai
from google.genai import types

from memory_core.config.config_manager import get_config
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
    
    Uses LLM to understand query intent and extract relevant graph patterns.
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize Gemini client
        api_key = self.config.config.api.google_api_key
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not configured for natural language processing")
        
        self.client = genai.Client(api_key=api_key)
        self.model = self.config.config.llm.model
        
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
    
    def process_query(self, query: str, context: Optional[str] = None) -> ParsedQuery:
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
        llm_result = self._parse_with_llm(query, context)
        
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
    
    def _parse_with_llm(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Use LLM to parse complex natural language queries.
        
        Args:
            query: Natural language query
            context: Optional context
            
        Returns:
            LLM parsing results
        """
        prompt = self._create_parsing_prompt(query, context)
        
        try:
            gen_config = types.GenerateContentConfig(
                temperature=0.1,  # Low temperature for consistent parsing
                top_p=0.9,
                max_output_tokens=2048,
                response_mime_type="application/json"
            )
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=gen_config
            )
            
            result = json.loads(response.text)
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
    
    def _create_parsing_prompt(self, query: str, context: Optional[str] = None) -> str:
        """
        Create a prompt for LLM to parse the natural language query.
        
        Args:
            query: Natural language query
            context: Optional context
            
        Returns:
            Formatted prompt string
        """
        context_part = f"\nContext: {context}" if context else ""
        
        return f"""You are an expert at parsing natural language queries for a knowledge graph database.
Analyze the following query and extract structured information.

Query: "{query}"{context_part}

Return a JSON object with the following structure:
{{
    "intent": "<primary intent: search, find_relationships, aggregate, compare, etc.>",
    "entities": ["<list of entities/concepts mentioned>"],
    "relationships": ["<list of relationships mentioned>"],
    "constraints": ["<list of constraints/filters mentioned>"],
    "query_type": "<one of: natural_language, graph_pattern, semantic_search, relationship_search, aggregation, hybrid>",
    "semantic_keywords": ["<key terms for semantic search>"],
    "confidence": <float between 0 and 1>,
    "graph_pattern": {{
        "nodes": [
            {{"type": "<node type>", "properties": {{"key": "value"}}, "variable": "<variable name>"}}
        ],
        "edges": [
            {{"type": "<relationship type>", "from": "<source variable>", "to": "<target variable>", "properties": {{}}}}
        ],
        "constraints": ["<gremlin-style constraints>"]
    }},
    "filters": [
        {{"field": "<field name>", "operator": "<eq|ne|gt|lt|contains|regex>", "value": "<filter value>"}}
    ],
    "sort_criteria": [
        {{"field": "<field name>", "order": "<asc|desc>"}}
    ],
    "aggregations": [
        {{"type": "<count|sum|avg|min|max>", "field": "<field name>"}}
    ]
}}

Guidelines:
- For similarity/semantic queries, focus on semantic_keywords
- For relationship queries, identify source and target entities
- For aggregation queries, identify what to count/sum/etc
- Extract any mentioned filters, sorting, or constraints
- Be conservative with confidence scores
- Use graph_pattern for structured queries that can be represented as node-edge patterns

Examples:
- "Find nodes similar to 'artificial intelligence'" → semantic_search
- "Show relationships between Python and machine learning" → relationship_search  
- "Count how many programming languages are mentioned" → aggregation
- "Find all concepts related to databases created after 2020" → graph_pattern with constraints
"""
    
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