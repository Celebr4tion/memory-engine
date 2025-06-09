"""
Insight Discovery Engine for Knowledge Synthesis

Provides pattern detection, trend analysis, and anomaly detection capabilities
to discover insights and patterns across knowledge domains.
"""

import logging
import time
import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, Counter

from memory_core.query.query_engine import AdvancedQueryEngine
from memory_core.query.query_types import QueryRequest, QueryType
from memory_core.model.knowledge_node import KnowledgeNode
from memory_core.model.relationship import Relationship


class PatternType(Enum):
    """Types of patterns that can be detected."""
    FREQUENCY = "frequency"  # Frequent co-occurrences
    CLUSTERING = "clustering"  # Dense clusters of related concepts
    HIERARCHY = "hierarchy"  # Hierarchical relationships
    SEQUENCE = "sequence"  # Sequential patterns
    CORRELATION = "correlation"  # Correlated attributes
    EMERGENCE = "emergence"  # Emerging patterns over time
    SIMILARITY = "similarity"  # Similar concept groups


class TrendType(Enum):
    """Types of trends that can be analyzed."""
    TEMPORAL = "temporal"  # Changes over time
    GROWTH = "growth"  # Growth patterns
    DECLINE = "decline"  # Decline patterns
    CYCLICAL = "cyclical"  # Cyclical patterns
    SEASONAL = "seasonal"  # Seasonal variations
    SUDDEN_CHANGE = "sudden_change"  # Sudden changes or shifts


class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    OUTLIER = "outlier"  # Statistical outliers
    STRUCTURAL = "structural"  # Structural anomalies in graph
    TEMPORAL = "temporal"  # Temporal anomalies
    BEHAVIORAL = "behavioral"  # Behavioral anomalies
    SEMANTIC = "semantic"  # Semantic inconsistencies


@dataclass
class Pattern:
    """Detected pattern structure."""
    pattern_id: str
    pattern_type: PatternType
    description: str
    elements: List[str]  # Node IDs involved
    confidence: float
    support: int  # Number of instances
    metadata: Dict[str, Any]
    discovered_at: datetime


@dataclass
class Trend:
    """Detected trend structure."""
    trend_id: str
    trend_type: TrendType
    description: str
    entities: List[str]
    time_range: Tuple[datetime, datetime]
    strength: float  # Trend strength (0-1)
    direction: str  # 'increasing', 'decreasing', 'stable', 'cyclical'
    confidence: float
    data_points: List[Dict[str, Any]]
    metadata: Dict[str, Any]


@dataclass
class Anomaly:
    """Detected anomaly structure."""
    anomaly_id: str
    anomaly_type: AnomalyType
    description: str
    affected_entities: List[str]
    severity: float  # 0-1 scale
    confidence: float
    detected_at: datetime
    context: Dict[str, Any]
    suggested_actions: List[str]


@dataclass
class InsightReport:
    """Comprehensive insight discovery report."""
    patterns: List[Pattern]
    trends: List[Trend]
    anomalies: List[Anomaly]
    summary: str
    discovery_time_ms: float
    total_entities_analyzed: int
    confidence_distribution: Dict[str, int]
    recommendations: List[str]


class PatternDetector:
    """Detects various patterns in the knowledge graph."""
    
    def __init__(self, query_engine: AdvancedQueryEngine):
        self.query_engine = query_engine
        self.logger = logging.getLogger(__name__)
    
    def detect_patterns(self, domain: Optional[str] = None, 
                       pattern_types: List[PatternType] = None,
                       min_support: int = 3) -> List[Pattern]:
        """
        Detect patterns in the knowledge graph.
        
        Args:
            domain: Optional domain filter
            pattern_types: Types of patterns to detect
            min_support: Minimum support for pattern validity
            
        Returns:
            List of detected patterns
        """
        if pattern_types is None:
            pattern_types = list(PatternType)
        
        patterns = []
        
        try:
            # Get sample of nodes for analysis
            nodes = self._get_analysis_nodes(domain, limit=500)
            
            for pattern_type in pattern_types:
                if pattern_type == PatternType.FREQUENCY:
                    patterns.extend(self._detect_frequency_patterns(nodes, min_support))
                elif pattern_type == PatternType.CLUSTERING:
                    patterns.extend(self._detect_clustering_patterns(nodes))
                elif pattern_type == PatternType.HIERARCHY:
                    patterns.extend(self._detect_hierarchy_patterns(nodes))
                elif pattern_type == PatternType.SEQUENCE:
                    patterns.extend(self._detect_sequence_patterns(nodes))
                elif pattern_type == PatternType.CORRELATION:
                    patterns.extend(self._detect_correlation_patterns(nodes))
                elif pattern_type == PatternType.SIMILARITY:
                    patterns.extend(self._detect_similarity_patterns(nodes))
            
            self.logger.info(f"Detected {len(patterns)} patterns")
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting patterns: {e}")
            return []
    
    def _get_analysis_nodes(self, domain: Optional[str], limit: int) -> List[Dict[str, Any]]:
        """Get nodes for pattern analysis."""
        try:
            query = domain if domain else "*"
            
            request = QueryRequest(
                query=query,
                query_type=QueryType.SEMANTIC_SEARCH if domain else QueryType.GRAPH_PATTERN,
                limit=limit,
                include_relationships=True,
                max_depth=1
            )
            
            response = self.query_engine.query(request)
            
            nodes = []
            for result in response.results:
                nodes.append({
                    'id': result.node_id,
                    'content': result.content,
                    'type': result.node_type,
                    'metadata': result.metadata or {},
                    'relationships': result.relationships or []
                })
            
            return nodes
            
        except Exception as e:
            self.logger.error(f"Error getting analysis nodes: {e}")
            return []
    
    def _detect_frequency_patterns(self, nodes: List[Dict[str, Any]], 
                                  min_support: int) -> List[Pattern]:
        """Detect frequency-based patterns (co-occurrence patterns)."""
        patterns = []
        
        try:
            # Extract terms from all nodes
            term_sets = []
            node_map = {}
            
            for node in nodes:
                terms = self._extract_terms(node['content'])
                if len(terms) >= 2:
                    term_sets.append(set(terms))
                    node_map[len(term_sets) - 1] = node['id']
            
            # Find frequent itemsets using Apriori-like approach
            frequent_pairs = self._find_frequent_pairs(term_sets, min_support)
            
            for pair, support in frequent_pairs.items():
                if support >= min_support:
                    # Find nodes containing this pattern
                    pattern_nodes = []
                    for i, term_set in enumerate(term_sets):
                        if pair.issubset(term_set):
                            pattern_nodes.append(node_map[i])
                    
                    pattern = Pattern(
                        pattern_id=f"freq_{hash(pair)}",
                        pattern_type=PatternType.FREQUENCY,
                        description=f"Frequent co-occurrence of {', '.join(pair)}",
                        elements=pattern_nodes,
                        confidence=min(support / len(term_sets), 1.0),
                        support=support,
                        metadata={'terms': list(pair), 'frequency': support},
                        discovered_at=datetime.now()
                    )
                    patterns.append(pattern)
            
        except Exception as e:
            self.logger.error(f"Error detecting frequency patterns: {e}")
        
        return patterns
    
    def _detect_clustering_patterns(self, nodes: List[Dict[str, Any]]) -> List[Pattern]:
        """Detect clustering patterns in the graph."""
        patterns = []
        
        try:
            # Build adjacency information
            adjacency = defaultdict(set)
            
            for node in nodes:
                node_id = node['id']
                for rel in node['relationships']:
                    if rel.target_id != node_id:
                        adjacency[node_id].add(rel.target_id)
                    if rel.source_id != node_id:
                        adjacency[node_id].add(rel.source_id)
            
            # Find dense clusters using simple clustering
            clusters = self._find_dense_clusters(adjacency, min_cluster_size=3)
            
            for i, cluster in enumerate(clusters):
                if len(cluster) >= 3:  # Minimum cluster size
                    # Calculate cluster density
                    density = self._calculate_cluster_density(cluster, adjacency)
                    
                    if density >= 0.4:  # Minimum density threshold
                        pattern = Pattern(
                            pattern_id=f"cluster_{i}",
                            pattern_type=PatternType.CLUSTERING,
                            description=f"Dense cluster of {len(cluster)} related entities",
                            elements=list(cluster),
                            confidence=density,
                            support=len(cluster),
                            metadata={'density': density, 'size': len(cluster)},
                            discovered_at=datetime.now()
                        )
                        patterns.append(pattern)
            
        except Exception as e:
            self.logger.error(f"Error detecting clustering patterns: {e}")
        
        return patterns
    
    def _detect_hierarchy_patterns(self, nodes: List[Dict[str, Any]]) -> List[Pattern]:
        """Detect hierarchical patterns."""
        patterns = []
        
        try:
            # Look for hierarchical relationships
            hierarchical_rels = ['is_a', 'part_of', 'contains', 'child_of', 'parent_of']
            
            hierarchy_groups = defaultdict(list)
            
            for node in nodes:
                for rel in node['relationships']:
                    if rel.relationship_type.lower() in hierarchical_rels:
                        key = f"{rel.relationship_type}_{rel.source_id}"
                        hierarchy_groups[key].append(rel.target_id)
            
            # Find significant hierarchies
            for group_key, children in hierarchy_groups.items():
                if len(children) >= 3:  # At least 3 children
                    rel_type, parent_id = group_key.split('_', 1)
                    
                    pattern = Pattern(
                        pattern_id=f"hierarchy_{hash(group_key)}",
                        pattern_type=PatternType.HIERARCHY,
                        description=f"Hierarchical structure with {len(children)} {rel_type} relationships",
                        elements=[parent_id] + children,
                        confidence=0.8,  # High confidence for explicit hierarchical relationships
                        support=len(children),
                        metadata={
                            'relationship_type': rel_type,
                            'parent': parent_id,
                            'children': children
                        },
                        discovered_at=datetime.now()
                    )
                    patterns.append(pattern)
            
        except Exception as e:
            self.logger.error(f"Error detecting hierarchy patterns: {e}")
        
        return patterns
    
    def _detect_sequence_patterns(self, nodes: List[Dict[str, Any]]) -> List[Pattern]:
        """Detect sequential patterns."""
        patterns = []
        
        try:
            # Look for temporal or procedural sequences
            sequence_indicators = ['next', 'then', 'after', 'before', 'follows', 'precedes']
            sequential_rels = ['follows', 'precedes', 'next', 'before', 'after']
            
            sequences = defaultdict(list)
            
            for node in nodes:
                # Check content for sequence indicators
                content_lower = node['content'].lower()
                if any(indicator in content_lower for indicator in sequence_indicators):
                    # Check relationships for sequential connections
                    for rel in node['relationships']:
                        if rel.relationship_type.lower() in sequential_rels:
                            seq_key = f"sequence_{rel.relationship_type}"
                            sequences[seq_key].append((rel.source_id, rel.target_id))
            
            # Build sequence chains
            for seq_type, connections in sequences.items():
                if len(connections) >= 2:
                    chains = self._build_sequence_chains(connections)
                    
                    for chain in chains:
                        if len(chain) >= 3:  # Minimum chain length
                            pattern = Pattern(
                                pattern_id=f"sequence_{hash(tuple(chain))}",
                                pattern_type=PatternType.SEQUENCE,
                                description=f"Sequential pattern: {' → '.join(chain[:3])}...",
                                elements=chain,
                                confidence=0.7,
                                support=len(chain),
                                metadata={'sequence_type': seq_type, 'chain_length': len(chain)},
                                discovered_at=datetime.now()
                            )
                            patterns.append(pattern)
            
        except Exception as e:
            self.logger.error(f"Error detecting sequence patterns: {e}")
        
        return patterns
    
    def _detect_correlation_patterns(self, nodes: List[Dict[str, Any]]) -> List[Pattern]:
        """Detect correlation patterns in node attributes."""
        patterns = []
        
        try:
            # Group nodes by attributes
            attribute_groups = defaultdict(lambda: defaultdict(list))
            
            for node in nodes:
                metadata = node['metadata']
                node_type = node['type']
                
                # Group by type and attributes
                if node_type:
                    attribute_groups[node_type]['nodes'].append(node['id'])
                
                # Group by metadata attributes
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float)):
                        attribute_groups[f"attr_{key}"][str(value)].append(node['id'])
            
            # Find correlated attributes
            for attr_name, value_groups in attribute_groups.items():
                if attr_name.startswith('attr_') and len(value_groups) >= 2:
                    # Check for correlation between attribute values and other patterns
                    for value, node_ids in value_groups.items():
                        if len(node_ids) >= 3:  # Significant group size
                            
                            # Calculate correlation strength
                            correlation = len(node_ids) / len(nodes)
                            
                            if correlation >= 0.1:  # Minimum correlation threshold
                                pattern = Pattern(
                                    pattern_id=f"corr_{hash(f'{attr_name}_{value}')}",
                                    pattern_type=PatternType.CORRELATION,
                                    description=f"Correlation: nodes with {attr_name}={value}",
                                    elements=node_ids,
                                    confidence=correlation,
                                    support=len(node_ids),
                                    metadata={
                                        'attribute': attr_name.replace('attr_', ''),
                                        'value': value,
                                        'correlation_strength': correlation
                                    },
                                    discovered_at=datetime.now()
                                )
                                patterns.append(pattern)
            
        except Exception as e:
            self.logger.error(f"Error detecting correlation patterns: {e}")
        
        return patterns
    
    def _detect_similarity_patterns(self, nodes: List[Dict[str, Any]]) -> List[Pattern]:
        """Detect similarity patterns using content analysis."""
        patterns = []
        
        try:
            # Simple similarity detection based on content overlap
            similarity_groups = []
            processed = set()
            
            for i, node1 in enumerate(nodes):
                if node1['id'] in processed:
                    continue
                
                similar_nodes = [node1['id']]
                terms1 = set(self._extract_terms(node1['content']))
                
                for j, node2 in enumerate(nodes[i+1:], i+1):
                    if node2['id'] in processed:
                        continue
                    
                    terms2 = set(self._extract_terms(node2['content']))
                    
                    # Calculate Jaccard similarity
                    if terms1 and terms2:
                        intersection = len(terms1 & terms2)
                        union = len(terms1 | terms2)
                        similarity = intersection / union if union > 0 else 0
                        
                        if similarity >= 0.3:  # Similarity threshold
                            similar_nodes.append(node2['id'])
                            processed.add(node2['id'])
                
                if len(similar_nodes) >= 3:  # Minimum group size
                    # Calculate average similarity within group
                    avg_similarity = self._calculate_group_similarity(
                        similar_nodes, nodes
                    )
                    
                    pattern = Pattern(
                        pattern_id=f"similarity_{hash(tuple(sorted(similar_nodes)))}",
                        pattern_type=PatternType.SIMILARITY,
                        description=f"Similar content group of {len(similar_nodes)} entities",
                        elements=similar_nodes,
                        confidence=avg_similarity,
                        support=len(similar_nodes),
                        metadata={
                            'avg_similarity': avg_similarity,
                            'group_size': len(similar_nodes)
                        },
                        discovered_at=datetime.now()
                    )
                    patterns.append(pattern)
                    
                    similarity_groups.append(similar_nodes)
                    for node_id in similar_nodes:
                        processed.add(node_id)
            
        except Exception as e:
            self.logger.error(f"Error detecting similarity patterns: {e}")
        
        return patterns
    
    def _extract_terms(self, content: str) -> List[str]:
        """Extract meaningful terms from content."""
        import re
        
        # Simple term extraction
        content = content.lower()
        # Remove punctuation and split
        terms = re.findall(r'\b[a-zA-Z]{3,}\b', content)
        
        # Remove common stop words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'through', 'during',
            'this', 'that', 'these', 'those', 'what', 'which', 'who', 'when',
            'where', 'why', 'how', 'can', 'could', 'would', 'should', 'will',
            'shall', 'may', 'might', 'must', 'have', 'has', 'had', 'do', 'does',
            'did', 'was', 'were', 'been', 'being', 'are', 'is'
        }
        
        filtered_terms = [term for term in terms if term not in stop_words]
        return filtered_terms
    
    def _find_frequent_pairs(self, term_sets: List[Set[str]], 
                           min_support: int) -> Dict[frozenset, int]:
        """Find frequent term pairs using Apriori-like approach."""
        # Count all individual terms
        term_counts = Counter()
        for term_set in term_sets:
            for term in term_set:
                term_counts[term] += 1
        
        # Get frequent individual terms
        frequent_terms = {term for term, count in term_counts.items() 
                         if count >= min_support}
        
        # Count frequent pairs
        pair_counts = Counter()
        for term_set in term_sets:
            # Only consider pairs of frequent terms
            frequent_in_set = term_set & frequent_terms
            if len(frequent_in_set) >= 2:
                for term1 in frequent_in_set:
                    for term2 in frequent_in_set:
                        if term1 < term2:  # Avoid duplicates
                            pair = frozenset([term1, term2])
                            pair_counts[pair] += 1
        
        return dict(pair_counts)
    
    def _find_dense_clusters(self, adjacency: Dict[str, Set[str]], 
                           min_cluster_size: int) -> List[Set[str]]:
        """Find dense clusters using simple clustering algorithm."""
        clusters = []
        visited = set()
        
        for node in adjacency:
            if node in visited:
                continue
            
            # Start a new cluster
            cluster = set()
            to_visit = [node]
            
            while to_visit:
                current = to_visit.pop()
                if current in visited:
                    continue
                
                visited.add(current)
                cluster.add(current)
                
                # Add densely connected neighbors
                neighbors = adjacency.get(current, set())
                for neighbor in neighbors:
                    if neighbor not in visited:
                        # Check if neighbor is densely connected to cluster
                        neighbor_connections = adjacency.get(neighbor, set())
                        common_connections = len(neighbor_connections & cluster)
                        
                        if common_connections >= min(len(cluster) * 0.3, 2):
                            to_visit.append(neighbor)
            
            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)
        
        return clusters
    
    def _calculate_cluster_density(self, cluster: Set[str], 
                                 adjacency: Dict[str, Set[str]]) -> float:
        """Calculate the density of a cluster."""
        if len(cluster) < 2:
            return 0.0
        
        total_possible_edges = len(cluster) * (len(cluster) - 1) // 2
        actual_edges = 0
        
        cluster_list = list(cluster)
        for i, node1 in enumerate(cluster_list):
            for node2 in cluster_list[i+1:]:
                if node2 in adjacency.get(node1, set()):
                    actual_edges += 1
        
        return actual_edges / total_possible_edges if total_possible_edges > 0 else 0.0
    
    def _build_sequence_chains(self, connections: List[Tuple[str, str]]) -> List[List[str]]:
        """Build sequence chains from connections."""
        # Build adjacency for sequences
        next_nodes = defaultdict(list)
        prev_nodes = defaultdict(list)
        
        for source, target in connections:
            next_nodes[source].append(target)
            prev_nodes[target].append(source)
        
        # Find chain starts (nodes with no predecessors)
        starts = [node for node in next_nodes if node not in prev_nodes]
        
        chains = []
        for start in starts:
            chain = self._follow_chain(start, next_nodes)
            if len(chain) >= 2:
                chains.append(chain)
        
        return chains
    
    def _follow_chain(self, start: str, next_nodes: Dict[str, List[str]]) -> List[str]:
        """Follow a chain from start node."""
        chain = [start]
        current = start
        visited = set([start])
        
        while current in next_nodes:
            # Take first unvisited next node to avoid cycles
            next_candidates = [n for n in next_nodes[current] if n not in visited]
            if not next_candidates:
                break
            
            current = next_candidates[0]
            chain.append(current)
            visited.add(current)
            
            # Prevent infinite loops
            if len(chain) > 20:
                break
        
        return chain
    
    def _calculate_group_similarity(self, node_ids: List[str], 
                                  nodes: List[Dict[str, Any]]) -> float:
        """Calculate average similarity within a group of nodes."""
        node_contents = {}
        for node in nodes:
            if node['id'] in node_ids:
                node_contents[node['id']] = set(self._extract_terms(node['content']))
        
        if len(node_contents) < 2:
            return 0.0
        
        total_similarity = 0.0
        comparisons = 0
        
        node_list = list(node_contents.keys())
        for i, node1 in enumerate(node_list):
            for node2 in node_list[i+1:]:
                terms1 = node_contents[node1]
                terms2 = node_contents[node2]
                
                if terms1 and terms2:
                    intersection = len(terms1 & terms2)
                    union = len(terms1 | terms2)
                    similarity = intersection / union if union > 0 else 0
                    total_similarity += similarity
                    comparisons += 1
        
        return total_similarity / comparisons if comparisons > 0 else 0.0


class TrendAnalyzer:
    """Analyzes trends over time in the knowledge graph."""
    
    def __init__(self, query_engine: AdvancedQueryEngine):
        self.query_engine = query_engine
        self.logger = logging.getLogger(__name__)
    
    def analyze_trends(self, domain: Optional[str] = None,
                      time_range: Optional[Tuple[datetime, datetime]] = None,
                      trend_types: List[TrendType] = None) -> List[Trend]:
        """
        Analyze trends in the knowledge graph.
        
        Args:
            domain: Optional domain filter
            time_range: Optional time range for analysis
            trend_types: Types of trends to analyze
            
        Returns:
            List of detected trends
        """
        if trend_types is None:
            trend_types = [TrendType.TEMPORAL, TrendType.GROWTH, TrendType.DECLINE]
        
        trends = []
        
        try:
            # Get temporal data
            temporal_data = self._get_temporal_data(domain, time_range)
            
            if not temporal_data:
                self.logger.warning("No temporal data available for trend analysis")
                return trends
            
            for trend_type in trend_types:
                if trend_type == TrendType.TEMPORAL:
                    trends.extend(self._analyze_temporal_trends(temporal_data))
                elif trend_type == TrendType.GROWTH:
                    trends.extend(self._analyze_growth_trends(temporal_data))
                elif trend_type == TrendType.DECLINE:
                    trends.extend(self._analyze_decline_trends(temporal_data))
                elif trend_type == TrendType.CYCLICAL:
                    trends.extend(self._analyze_cyclical_trends(temporal_data))
                elif trend_type == TrendType.SUDDEN_CHANGE:
                    trends.extend(self._analyze_sudden_changes(temporal_data))
            
            self.logger.info(f"Analyzed {len(trends)} trends")
            return trends
            
        except Exception as e:
            self.logger.error(f"Error analyzing trends: {e}")
            return []
    
    def _get_temporal_data(self, domain: Optional[str], 
                          time_range: Optional[Tuple[datetime, datetime]]) -> Dict[str, List[Dict[str, Any]]]:
        """Get temporal data for trend analysis."""
        temporal_data = defaultdict(list)
        
        try:
            # Get nodes with temporal metadata
            query = domain if domain else "*"
            
            request = QueryRequest(
                query=query,
                query_type=QueryType.SEMANTIC_SEARCH if domain else QueryType.GRAPH_PATTERN,
                limit=1000,  # Larger sample for trend analysis
                include_relationships=False
            )
            
            response = self.query_engine.query(request)
            
            for result in response.results:
                metadata = result.metadata or {}
                
                # Look for temporal information
                timestamp = None
                
                # Check various timestamp fields
                for field in ['timestamp', 'created_at', 'date', 'time', 'modified_at']:
                    if field in metadata:
                        try:
                            if isinstance(metadata[field], str):
                                timestamp = datetime.fromisoformat(metadata[field].replace('Z', '+00:00'))
                            elif isinstance(metadata[field], (int, float)):
                                timestamp = datetime.fromtimestamp(metadata[field])
                            break
                        except:
                            continue
                
                if timestamp:
                    # Filter by time range if specified
                    if time_range:
                        start_time, end_time = time_range
                        if not (start_time <= timestamp <= end_time):
                            continue
                    
                    # Group by entity type or domain
                    entity_key = result.node_type or 'unknown'
                    
                    temporal_data[entity_key].append({
                        'node_id': result.node_id,
                        'timestamp': timestamp,
                        'content_length': len(result.content),
                        'metadata': metadata
                    })
            
            # Sort each entity's data by timestamp
            for entity_key in temporal_data:
                temporal_data[entity_key].sort(key=lambda x: x['timestamp'])
            
            return dict(temporal_data)
            
        except Exception as e:
            self.logger.error(f"Error getting temporal data: {e}")
            return {}
    
    def _analyze_temporal_trends(self, temporal_data: Dict[str, List[Dict[str, Any]]]) -> List[Trend]:
        """Analyze general temporal trends."""
        trends = []
        
        for entity_type, data_points in temporal_data.items():
            if len(data_points) < 3:  # Need minimum data points
                continue
            
            try:
                # Analyze activity over time
                time_series = self._create_time_series(data_points, 'count')
                
                if len(time_series) >= 3:
                    # Calculate trend direction and strength
                    direction, strength = self._calculate_trend_direction(time_series)
                    
                    if strength >= 0.3:  # Minimum trend strength
                        start_time = min(dp['timestamp'] for dp in data_points)
                        end_time = max(dp['timestamp'] for dp in data_points)
                        
                        trend = Trend(
                            trend_id=f"temporal_{entity_type}_{hash(str(time_series))}",
                            trend_type=TrendType.TEMPORAL,
                            description=f"{direction.title()} activity trend for {entity_type}",
                            entities=[entity_type],
                            time_range=(start_time, end_time),
                            strength=strength,
                            direction=direction,
                            confidence=min(strength + 0.2, 1.0),
                            data_points=time_series,
                            metadata={
                                'entity_type': entity_type,
                                'total_data_points': len(data_points),
                                'trend_strength': strength
                            }
                        )
                        trends.append(trend)
                
            except Exception as e:
                self.logger.warning(f"Error analyzing temporal trends for {entity_type}: {e}")
                continue
        
        return trends
    
    def _analyze_growth_trends(self, temporal_data: Dict[str, List[Dict[str, Any]]]) -> List[Trend]:
        """Analyze growth trends."""
        trends = []
        
        for entity_type, data_points in temporal_data.items():
            if len(data_points) < 5:  # Need more data for growth analysis
                continue
            
            try:
                # Create cumulative count time series
                cumulative_series = self._create_cumulative_series(data_points)
                
                # Check for growth patterns
                growth_rate = self._calculate_growth_rate(cumulative_series)
                
                if growth_rate > 0.1:  # Minimum growth rate
                    start_time = min(dp['timestamp'] for dp in data_points)
                    end_time = max(dp['timestamp'] for dp in data_points)
                    
                    trend = Trend(
                        trend_id=f"growth_{entity_type}_{hash(str(cumulative_series))}",
                        trend_type=TrendType.GROWTH,
                        description=f"Growth trend in {entity_type} (rate: {growth_rate:.2f})",
                        entities=[entity_type],
                        time_range=(start_time, end_time),
                        strength=min(growth_rate, 1.0),
                        direction='increasing',
                        confidence=min(growth_rate + 0.3, 1.0),
                        data_points=cumulative_series,
                        metadata={
                            'growth_rate': growth_rate,
                            'entity_type': entity_type
                        }
                    )
                    trends.append(trend)
                
            except Exception as e:
                self.logger.warning(f"Error analyzing growth trends for {entity_type}: {e}")
                continue
        
        return trends
    
    def _analyze_decline_trends(self, temporal_data: Dict[str, List[Dict[str, Any]]]) -> List[Trend]:
        """Analyze decline trends."""
        trends = []
        
        for entity_type, data_points in temporal_data.items():
            if len(data_points) < 5:
                continue
            
            try:
                # Look for periods of declining activity
                time_series = self._create_time_series(data_points, 'count', bucket_size='week')
                
                if len(time_series) >= 4:
                    # Check for sustained decline
                    decline_periods = self._find_decline_periods(time_series)
                    
                    for period in decline_periods:
                        if period['duration'] >= 3 and period['decline_rate'] > 0.2:
                            trend = Trend(
                                trend_id=f"decline_{entity_type}_{hash(str(period))}",
                                trend_type=TrendType.DECLINE,
                                description=f"Decline in {entity_type} activity",
                                entities=[entity_type],
                                time_range=(period['start'], period['end']),
                                strength=period['decline_rate'],
                                direction='decreasing',
                                confidence=min(period['decline_rate'] + 0.2, 1.0),
                                data_points=period['data_points'],
                                metadata={
                                    'decline_rate': period['decline_rate'],
                                    'duration_periods': period['duration']
                                }
                            )
                            trends.append(trend)
                
            except Exception as e:
                self.logger.warning(f"Error analyzing decline trends for {entity_type}: {e}")
                continue
        
        return trends
    
    def _analyze_cyclical_trends(self, temporal_data: Dict[str, List[Dict[str, Any]]]) -> List[Trend]:
        """Analyze cyclical trends."""
        trends = []
        
        for entity_type, data_points in temporal_data.items():
            if len(data_points) < 10:  # Need more data for cycle detection
                continue
            
            try:
                # Create time series and look for cycles
                time_series = self._create_time_series(data_points, 'count', bucket_size='day')
                
                if len(time_series) >= 10:
                    cycles = self._detect_cycles(time_series)
                    
                    for cycle in cycles:
                        if cycle['confidence'] >= 0.4:
                            start_time = min(dp['timestamp'] for dp in data_points)
                            end_time = max(dp['timestamp'] for dp in data_points)
                            
                            trend = Trend(
                                trend_id=f"cyclical_{entity_type}_{hash(str(cycle))}",
                                trend_type=TrendType.CYCLICAL,
                                description=f"Cyclical pattern in {entity_type} (period: {cycle['period']} days)",
                                entities=[entity_type],
                                time_range=(start_time, end_time),
                                strength=cycle['amplitude'],
                                direction='cyclical',
                                confidence=cycle['confidence'],
                                data_points=time_series,
                                metadata={
                                    'cycle_period': cycle['period'],
                                    'amplitude': cycle['amplitude']
                                }
                            )
                            trends.append(trend)
                
            except Exception as e:
                self.logger.warning(f"Error analyzing cyclical trends for {entity_type}: {e}")
                continue
        
        return trends
    
    def _analyze_sudden_changes(self, temporal_data: Dict[str, List[Dict[str, Any]]]) -> List[Trend]:
        """Analyze sudden changes or shifts."""
        trends = []
        
        for entity_type, data_points in temporal_data.items():
            if len(data_points) < 6:
                continue
            
            try:
                time_series = self._create_time_series(data_points, 'count', bucket_size='day')
                
                # Detect sudden changes
                changes = self._detect_sudden_changes(time_series)
                
                for change in changes:
                    if change['magnitude'] >= 2.0:  # Significant change threshold
                        trend = Trend(
                            trend_id=f"sudden_change_{entity_type}_{hash(str(change))}",
                            trend_type=TrendType.SUDDEN_CHANGE,
                            description=f"Sudden {change['direction']} in {entity_type}",
                            entities=[entity_type],
                            time_range=(change['timestamp'] - timedelta(days=1), 
                                      change['timestamp'] + timedelta(days=1)),
                            strength=min(change['magnitude'] / 5, 1.0),
                            direction=change['direction'],
                            confidence=min(change['magnitude'] / 3, 1.0),
                            data_points=[change],
                            metadata={
                                'change_magnitude': change['magnitude'],
                                'change_timestamp': change['timestamp'].isoformat()
                            }
                        )
                        trends.append(trend)
                
            except Exception as e:
                self.logger.warning(f"Error analyzing sudden changes for {entity_type}: {e}")
                continue
        
        return trends
    
    def _create_time_series(self, data_points: List[Dict[str, Any]], 
                          metric: str, bucket_size: str = 'day') -> List[Dict[str, Any]]:
        """Create time series from data points."""
        # Group data points by time buckets
        buckets = defaultdict(list)
        
        for dp in data_points:
            timestamp = dp['timestamp']
            
            # Create bucket key based on bucket size
            if bucket_size == 'day':
                bucket_key = timestamp.date()
            elif bucket_size == 'week':
                # Get start of week
                days_since_monday = timestamp.weekday()
                week_start = timestamp.date() - timedelta(days=days_since_monday)
                bucket_key = week_start
            elif bucket_size == 'month':
                bucket_key = timestamp.replace(day=1).date()
            else:
                bucket_key = timestamp.date()
            
            buckets[bucket_key].append(dp)
        
        # Create time series
        time_series = []
        for bucket_key, bucket_data in sorted(buckets.items()):
            if metric == 'count':
                value = len(bucket_data)
            elif metric == 'avg_length':
                value = sum(dp['content_length'] for dp in bucket_data) / len(bucket_data)
            else:
                value = len(bucket_data)  # Default to count
            
            time_series.append({
                'timestamp': datetime.combine(bucket_key, datetime.min.time()),
                'value': value,
                'count': len(bucket_data)
            })
        
        return time_series
    
    def _create_cumulative_series(self, data_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create cumulative time series."""
        cumulative_series = []
        cumulative_count = 0
        
        # Group by day and create cumulative counts
        daily_counts = defaultdict(int)
        for dp in data_points:
            day = dp['timestamp'].date()
            daily_counts[day] += 1
        
        for day, count in sorted(daily_counts.items()):
            cumulative_count += count
            cumulative_series.append({
                'timestamp': datetime.combine(day, datetime.min.time()),
                'value': cumulative_count,
                'daily_count': count
            })
        
        return cumulative_series
    
    def _calculate_trend_direction(self, time_series: List[Dict[str, Any]]) -> Tuple[str, float]:
        """Calculate trend direction and strength."""
        if len(time_series) < 2:
            return 'stable', 0.0
        
        values = [point['value'] for point in time_series]
        
        # Simple linear regression to find trend
        n = len(values)
        x = list(range(n))
        
        # Calculate slope
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        if n * sum_x2 - sum_x ** 2 == 0:
            return 'stable', 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        
        # Normalize slope to get strength
        if max(values) == min(values):
            strength = 0.0
        else:
            strength = abs(slope) / (max(values) - min(values)) * len(values)
        
        # Determine direction
        if slope > 0.1:
            direction = 'increasing'
        elif slope < -0.1:
            direction = 'decreasing'
        else:
            direction = 'stable'
        
        return direction, min(strength, 1.0)
    
    def _calculate_growth_rate(self, cumulative_series: List[Dict[str, Any]]) -> float:
        """Calculate growth rate from cumulative series."""
        if len(cumulative_series) < 2:
            return 0.0
        
        start_value = cumulative_series[0]['value']
        end_value = cumulative_series[-1]['value']
        
        if start_value == 0:
            return 1.0 if end_value > 0 else 0.0
        
        # Calculate compound growth rate
        periods = len(cumulative_series) - 1
        growth_rate = (end_value / start_value) ** (1 / periods) - 1
        
        return max(0, growth_rate)
    
    def _find_decline_periods(self, time_series: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find periods of sustained decline."""
        decline_periods = []
        
        # Look for consecutive periods of decline
        current_decline = None
        
        for i in range(1, len(time_series)):
            current_value = time_series[i]['value']
            previous_value = time_series[i-1]['value']
            
            if current_value < previous_value:
                # Start or continue decline
                if current_decline is None:
                    current_decline = {
                        'start': time_series[i-1]['timestamp'],
                        'start_value': previous_value,
                        'data_points': [time_series[i-1], time_series[i]]
                    }
                else:
                    current_decline['data_points'].append(time_series[i])
            else:
                # End decline if it exists
                if current_decline is not None:
                    current_decline['end'] = time_series[i-1]['timestamp']
                    current_decline['end_value'] = time_series[i-1]['value']
                    current_decline['duration'] = len(current_decline['data_points'])
                    
                    # Calculate decline rate
                    if current_decline['start_value'] > 0:
                        decline_amount = current_decline['start_value'] - current_decline['end_value']
                        current_decline['decline_rate'] = decline_amount / current_decline['start_value']
                    else:
                        current_decline['decline_rate'] = 0.0
                    
                    decline_periods.append(current_decline)
                    current_decline = None
        
        # Handle decline that continues to the end
        if current_decline is not None:
            current_decline['end'] = time_series[-1]['timestamp']
            current_decline['end_value'] = time_series[-1]['value']
            current_decline['duration'] = len(current_decline['data_points'])
            
            if current_decline['start_value'] > 0:
                decline_amount = current_decline['start_value'] - current_decline['end_value']
                current_decline['decline_rate'] = decline_amount / current_decline['start_value']
            else:
                current_decline['decline_rate'] = 0.0
            
            decline_periods.append(current_decline)
        
        return decline_periods
    
    def _detect_cycles(self, time_series: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect cyclical patterns in time series."""
        cycles = []
        
        values = [point['value'] for point in time_series]
        
        # Simple cycle detection using autocorrelation
        for period in range(2, len(values) // 2):
            correlation = self._calculate_autocorrelation(values, period)
            
            if correlation > 0.3:  # Correlation threshold
                # Calculate amplitude
                amplitude = self._calculate_cycle_amplitude(values, period)
                
                cycle = {
                    'period': period,
                    'correlation': correlation,
                    'amplitude': amplitude,
                    'confidence': correlation
                }
                cycles.append(cycle)
        
        # Sort by confidence and return top cycles
        cycles.sort(key=lambda x: x['confidence'], reverse=True)
        return cycles[:3]  # Top 3 cycles
    
    def _calculate_autocorrelation(self, values: List[float], lag: int) -> float:
        """Calculate autocorrelation for given lag."""
        if lag >= len(values):
            return 0.0
        
        n = len(values) - lag
        if n <= 0:
            return 0.0
        
        mean_val = sum(values) / len(values)
        
        # Calculate correlation
        numerator = sum((values[i] - mean_val) * (values[i + lag] - mean_val) for i in range(n))
        denominator = sum((val - mean_val) ** 2 for val in values)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _calculate_cycle_amplitude(self, values: List[float], period: int) -> float:
        """Calculate amplitude of cyclical pattern."""
        if period >= len(values):
            return 0.0
        
        # Calculate average values for each position in the cycle
        cycle_averages = []
        for pos in range(period):
            positions = [values[i] for i in range(pos, len(values), period)]
            if positions:
                cycle_averages.append(sum(positions) / len(positions))
        
        if not cycle_averages:
            return 0.0
        
        # Amplitude is the range of the cycle
        return max(cycle_averages) - min(cycle_averages)
    
    def _detect_sudden_changes(self, time_series: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect sudden changes in time series."""
        changes = []
        
        values = [point['value'] for point in time_series]
        
        # Calculate moving average and standard deviation
        window = min(5, len(values) // 3)
        if window < 2:
            return changes
        
        for i in range(window, len(values) - window):
            # Calculate before and after windows
            before_window = values[i-window:i]
            after_window = values[i:i+window]
            
            before_avg = sum(before_window) / len(before_window)
            after_avg = sum(after_window) / len(after_window)
            
            # Calculate change magnitude
            if before_avg != 0:
                change_ratio = abs(after_avg - before_avg) / before_avg
            else:
                change_ratio = 1.0 if after_avg > 0 else 0.0
            
            # Detect significant changes
            if change_ratio > 0.5:  # 50% change threshold
                change = {
                    'timestamp': time_series[i]['timestamp'],
                    'magnitude': change_ratio,
                    'direction': 'increase' if after_avg > before_avg else 'decrease',
                    'before_value': before_avg,
                    'after_value': after_avg
                }
                changes.append(change)
        
        return changes


class AnomalyDetector:
    """Detects anomalies in the knowledge graph."""
    
    def __init__(self, query_engine: AdvancedQueryEngine):
        self.query_engine = query_engine
        self.logger = logging.getLogger(__name__)
    
    def detect_anomalies(self, domain: Optional[str] = None,
                        anomaly_types: List[AnomalyType] = None) -> List[Anomaly]:
        """
        Detect anomalies in the knowledge graph.
        
        Args:
            domain: Optional domain filter
            anomaly_types: Types of anomalies to detect
            
        Returns:
            List of detected anomalies
        """
        if anomaly_types is None:
            anomaly_types = [AnomalyType.OUTLIER, AnomalyType.STRUCTURAL, AnomalyType.SEMANTIC]
        
        anomalies = []
        
        try:
            # Get data for analysis
            nodes = self._get_analysis_nodes(domain, limit=500)
            
            for anomaly_type in anomaly_types:
                if anomaly_type == AnomalyType.OUTLIER:
                    anomalies.extend(self._detect_outlier_anomalies(nodes))
                elif anomaly_type == AnomalyType.STRUCTURAL:
                    anomalies.extend(self._detect_structural_anomalies(nodes))
                elif anomaly_type == AnomalyType.SEMANTIC:
                    anomalies.extend(self._detect_semantic_anomalies(nodes))
                elif anomaly_type == AnomalyType.TEMPORAL:
                    anomalies.extend(self._detect_temporal_anomalies(nodes))
                elif anomaly_type == AnomalyType.BEHAVIORAL:
                    anomalies.extend(self._detect_behavioral_anomalies(nodes))
            
            self.logger.info(f"Detected {len(anomalies)} anomalies")
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {e}")
            return []
    
    def _get_analysis_nodes(self, domain: Optional[str], limit: int) -> List[Dict[str, Any]]:
        """Get nodes for anomaly analysis."""
        try:
            query = domain if domain else "*"
            
            request = QueryRequest(
                query=query,
                query_type=QueryType.SEMANTIC_SEARCH if domain else QueryType.GRAPH_PATTERN,
                limit=limit,
                include_relationships=True,
                max_depth=1
            )
            
            response = self.query_engine.query(request)
            
            nodes = []
            for result in response.results:
                nodes.append({
                    'id': result.node_id,
                    'content': result.content,
                    'type': result.node_type,
                    'metadata': result.metadata or {},
                    'relationships': result.relationships or []
                })
            
            return nodes
            
        except Exception as e:
            self.logger.error(f"Error getting analysis nodes: {e}")
            return []
    
    def _detect_outlier_anomalies(self, nodes: List[Dict[str, Any]]) -> List[Anomaly]:
        """Detect statistical outlier anomalies."""
        anomalies = []
        
        try:
            # Analyze content length outliers
            content_lengths = [len(node['content']) for node in nodes]
            
            if content_lengths:
                outliers = self._find_statistical_outliers(content_lengths)
                
                for idx in outliers:
                    node = nodes[idx]
                    severity = self._calculate_outlier_severity(
                        content_lengths[idx], content_lengths
                    )
                    
                    anomaly = Anomaly(
                        anomaly_id=f"outlier_length_{node['id']}",
                        anomaly_type=AnomalyType.OUTLIER,
                        description=f"Content length outlier: {len(node['content'])} characters",
                        affected_entities=[node['id']],
                        severity=severity,
                        confidence=min(severity + 0.2, 1.0),
                        detected_at=datetime.now(),
                        context={
                            'content_length': len(node['content']),
                            'avg_length': sum(content_lengths) / len(content_lengths),
                            'outlier_type': 'content_length'
                        },
                        suggested_actions=[
                            "Review content for quality and relevance",
                            "Check if content should be split or merged"
                        ]
                    )
                    anomalies.append(anomaly)
            
            # Analyze relationship count outliers
            rel_counts = [len(node['relationships']) for node in nodes]
            
            if rel_counts:
                outliers = self._find_statistical_outliers(rel_counts)
                
                for idx in outliers:
                    node = nodes[idx]
                    severity = self._calculate_outlier_severity(
                        rel_counts[idx], rel_counts
                    )
                    
                    anomaly = Anomaly(
                        anomaly_id=f"outlier_relationships_{node['id']}",
                        anomaly_type=AnomalyType.OUTLIER,
                        description=f"Relationship count outlier: {len(node['relationships'])} relationships",
                        affected_entities=[node['id']],
                        severity=severity,
                        confidence=min(severity + 0.2, 1.0),
                        detected_at=datetime.now(),
                        context={
                            'relationship_count': len(node['relationships']),
                            'avg_relationships': sum(rel_counts) / len(rel_counts),
                            'outlier_type': 'relationship_count'
                        },
                        suggested_actions=[
                            "Review relationship structure",
                            "Check for over-connection or isolation"
                        ]
                    )
                    anomalies.append(anomaly)
            
        except Exception as e:
            self.logger.error(f"Error detecting outlier anomalies: {e}")
        
        return anomalies
    
    def _detect_structural_anomalies(self, nodes: List[Dict[str, Any]]) -> List[Anomaly]:
        """Detect structural anomalies in the graph."""
        anomalies = []
        
        try:
            # Find isolated nodes (no relationships)
            isolated_nodes = [node for node in nodes if not node['relationships']]
            
            if len(isolated_nodes) > len(nodes) * 0.1:  # More than 10% isolated
                severity = len(isolated_nodes) / len(nodes)
                
                anomaly = Anomaly(
                    anomaly_id=f"structural_isolation_{hash(str([n['id'] for n in isolated_nodes]))}",
                    anomaly_type=AnomalyType.STRUCTURAL,
                    description=f"High isolation: {len(isolated_nodes)} nodes with no relationships",
                    affected_entities=[node['id'] for node in isolated_nodes[:10]],  # Limit to first 10
                    severity=severity,
                    confidence=0.8,
                    detected_at=datetime.now(),
                    context={
                        'isolated_count': len(isolated_nodes),
                        'total_nodes': len(nodes),
                        'isolation_rate': severity
                    },
                    suggested_actions=[
                        "Review isolated nodes for missing relationships",
                        "Consider relationship extraction improvements"
                    ]
                )
                anomalies.append(anomaly)
            
            # Find nodes with unusual relationship patterns
            relationship_types = defaultdict(int)
            for node in nodes:
                for rel in node['relationships']:
                    relationship_types[rel.relationship_type] += 1
            
            # Find nodes with rare relationship types
            rare_threshold = max(1, len(nodes) * 0.01)  # 1% threshold
            rare_rel_types = {rel_type for rel_type, count in relationship_types.items() 
                            if count <= rare_threshold}
            
            if rare_rel_types:
                affected_nodes = []
                for node in nodes:
                    for rel in node['relationships']:
                        if rel.relationship_type in rare_rel_types:
                            affected_nodes.append(node['id'])
                            break
                
                if affected_nodes:
                    anomaly = Anomaly(
                        anomaly_id=f"structural_rare_relationships_{hash(str(rare_rel_types))}",
                        anomaly_type=AnomalyType.STRUCTURAL,
                        description=f"Nodes with rare relationship types: {', '.join(list(rare_rel_types)[:3])}",
                        affected_entities=affected_nodes[:10],
                        severity=0.6,
                        confidence=0.7,
                        detected_at=datetime.now(),
                        context={
                            'rare_relationship_types': list(rare_rel_types),
                            'affected_node_count': len(affected_nodes)
                        },
                        suggested_actions=[
                            "Review rare relationship types for accuracy",
                            "Consider relationship type standardization"
                        ]
                    )
                    anomalies.append(anomaly)
            
        except Exception as e:
            self.logger.error(f"Error detecting structural anomalies: {e}")
        
        return anomalies
    
    def _detect_semantic_anomalies(self, nodes: List[Dict[str, Any]]) -> List[Anomaly]:
        """Detect semantic anomalies and inconsistencies."""
        anomalies = []
        
        try:
            # Group nodes by type
            type_groups = defaultdict(list)
            for node in nodes:
                node_type = node['type'] or 'unknown'
                type_groups[node_type].append(node)
            
            # Analyze each type group for semantic consistency
            for node_type, group_nodes in type_groups.items():
                if len(group_nodes) < 3:  # Need minimum group size
                    continue
                
                # Check for content similarity within type
                similarities = self._calculate_group_semantic_similarity(group_nodes)
                
                if similarities['avg_similarity'] < 0.2:  # Low similarity threshold
                    # Find the most dissimilar nodes
                    outlier_nodes = self._find_semantic_outliers(group_nodes)
                    
                    if outlier_nodes:
                        anomaly = Anomaly(
                            anomaly_id=f"semantic_inconsistency_{node_type}_{hash(str([n['id'] for n in outlier_nodes]))}",
                            anomaly_type=AnomalyType.SEMANTIC,
                            description=f"Semantic inconsistency in {node_type} nodes",
                            affected_entities=[node['id'] for node in outlier_nodes],
                            severity=1.0 - similarities['avg_similarity'],
                            confidence=0.6,
                            detected_at=datetime.now(),
                            context={
                                'node_type': node_type,
                                'avg_similarity': similarities['avg_similarity'],
                                'group_size': len(group_nodes)
                            },
                            suggested_actions=[
                                "Review node type assignments",
                                "Check for content classification errors"
                            ]
                        )
                        anomalies.append(anomaly)
            
            # Check for duplicate or near-duplicate content
            duplicates = self._find_duplicate_content(nodes)
            
            for duplicate_group in duplicates:
                if len(duplicate_group) >= 2:
                    anomaly = Anomaly(
                        anomaly_id=f"semantic_duplicates_{hash(str([n['id'] for n in duplicate_group]))}",
                        anomaly_type=AnomalyType.SEMANTIC,
                        description=f"Potential duplicate content found ({len(duplicate_group)} nodes)",
                        affected_entities=[node['id'] for node in duplicate_group],
                        severity=0.7,
                        confidence=0.8,
                        detected_at=datetime.now(),
                        context={
                            'duplicate_count': len(duplicate_group),
                            'similarity_threshold': 0.8
                        },
                        suggested_actions=[
                            "Review for actual duplicates",
                            "Consider content deduplication"
                        ]
                    )
                    anomalies.append(anomaly)
            
        except Exception as e:
            self.logger.error(f"Error detecting semantic anomalies: {e}")
        
        return anomalies
    
    def _detect_temporal_anomalies(self, nodes: List[Dict[str, Any]]) -> List[Anomaly]:
        """Detect temporal anomalies."""
        anomalies = []
        
        try:
            # Extract temporal information
            temporal_nodes = []
            for node in nodes:
                metadata = node['metadata']
                
                # Look for timestamp fields
                timestamp = None
                for field in ['timestamp', 'created_at', 'date', 'modified_at']:
                    if field in metadata:
                        try:
                            if isinstance(metadata[field], str):
                                timestamp = datetime.fromisoformat(metadata[field].replace('Z', '+00:00'))
                            elif isinstance(metadata[field], (int, float)):
                                timestamp = datetime.fromtimestamp(metadata[field])
                            break
                        except:
                            continue
                
                if timestamp:
                    temporal_nodes.append({
                        'node': node,
                        'timestamp': timestamp
                    })
            
            if len(temporal_nodes) < 10:  # Need sufficient temporal data
                return anomalies
            
            # Find temporal outliers
            timestamps = [tn['timestamp'] for tn in temporal_nodes]
            
            # Check for future dates (impossible timestamps)
            now = datetime.now()
            future_nodes = [tn for tn in temporal_nodes if tn['timestamp'] > now]
            
            if future_nodes:
                anomaly = Anomaly(
                    anomaly_id=f"temporal_future_dates_{hash(str([tn['node']['id'] for tn in future_nodes]))}",
                    anomaly_type=AnomalyType.TEMPORAL,
                    description=f"Nodes with future timestamps ({len(future_nodes)} nodes)",
                    affected_entities=[tn['node']['id'] for tn in future_nodes],
                    severity=0.9,
                    confidence=0.95,
                    detected_at=datetime.now(),
                    context={
                        'future_date_count': len(future_nodes),
                        'current_time': now.isoformat()
                    },
                    suggested_actions=[
                        "Correct timestamp data",
                        "Review timestamp generation process"
                    ]
                )
                anomalies.append(anomaly)
            
            # Check for very old dates (potentially invalid)
            cutoff_date = datetime.now() - timedelta(days=365*10)  # 10 years ago
            very_old_nodes = [tn for tn in temporal_nodes if tn['timestamp'] < cutoff_date]
            
            if len(very_old_nodes) > len(temporal_nodes) * 0.05:  # More than 5%
                anomaly = Anomaly(
                    anomaly_id=f"temporal_very_old_{hash(str([tn['node']['id'] for tn in very_old_nodes]))}",
                    anomaly_type=AnomalyType.TEMPORAL,
                    description=f"Unusually old timestamps ({len(very_old_nodes)} nodes)",
                    affected_entities=[tn['node']['id'] for tn in very_old_nodes[:10]],
                    severity=0.6,
                    confidence=0.7,
                    detected_at=datetime.now(),
                    context={
                        'very_old_count': len(very_old_nodes),
                        'cutoff_date': cutoff_date.isoformat()
                    },
                    suggested_actions=[
                        "Verify timestamp accuracy",
                        "Check for data import issues"
                    ]
                )
                anomalies.append(anomaly)
            
        except Exception as e:
            self.logger.error(f"Error detecting temporal anomalies: {e}")
        
        return anomalies
    
    def _detect_behavioral_anomalies(self, nodes: List[Dict[str, Any]]) -> List[Anomaly]:
        """Detect behavioral anomalies (unusual patterns)."""
        anomalies = []
        
        try:
            # Analyze relationship behavior patterns
            relationship_patterns = defaultdict(list)
            
            for node in nodes:
                node_id = node['id']
                relationships = node['relationships']
                
                # Count relationship types
                rel_type_counts = Counter(rel.relationship_type for rel in relationships)
                
                # Create behavior signature
                behavior_signature = {
                    'total_relationships': len(relationships),
                    'unique_rel_types': len(rel_type_counts),
                    'most_common_rel_type': rel_type_counts.most_common(1)[0][0] if rel_type_counts else None,
                    'rel_type_distribution': dict(rel_type_counts)
                }
                
                relationship_patterns[node_id] = behavior_signature
            
            # Find nodes with unusual behavior patterns
            unusual_nodes = self._find_behavioral_outliers(relationship_patterns)
            
            for node_id, outlier_info in unusual_nodes.items():
                node = next((n for n in nodes if n['id'] == node_id), None)
                if node:
                    anomaly = Anomaly(
                        anomaly_id=f"behavioral_{node_id}_{hash(str(outlier_info))}",
                        anomaly_type=AnomalyType.BEHAVIORAL,
                        description=f"Unusual relationship behavior: {outlier_info['reason']}",
                        affected_entities=[node_id],
                        severity=outlier_info['severity'],
                        confidence=outlier_info['confidence'],
                        detected_at=datetime.now(),
                        context={
                            'behavior_pattern': relationship_patterns[node_id],
                            'outlier_reason': outlier_info['reason']
                        },
                        suggested_actions=[
                            "Review relationship patterns",
                            "Check for relationship extraction errors"
                        ]
                    )
                    anomalies.append(anomaly)
            
        except Exception as e:
            self.logger.error(f"Error detecting behavioral anomalies: {e}")
        
        return anomalies
    
    def _find_statistical_outliers(self, values: List[float]) -> List[int]:
        """Find statistical outliers using IQR method."""
        if len(values) < 4:
            return []
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        # Calculate quartiles
        q1_idx = n // 4
        q3_idx = 3 * n // 4
        
        q1 = sorted_values[q1_idx]
        q3 = sorted_values[q3_idx]
        
        iqr = q3 - q1
        if iqr == 0:
            return []
        
        # Calculate outlier boundaries
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Find outlier indices
        outliers = []
        for i, value in enumerate(values):
            if value < lower_bound or value > upper_bound:
                outliers.append(i)
        
        return outliers
    
    def _calculate_outlier_severity(self, value: float, all_values: List[float]) -> float:
        """Calculate severity of an outlier."""
        if not all_values:
            return 0.0
        
        mean_val = sum(all_values) / len(all_values)
        std_dev = (sum((v - mean_val) ** 2 for v in all_values) / len(all_values)) ** 0.5
        
        if std_dev == 0:
            return 0.0
        
        # Calculate z-score
        z_score = abs(value - mean_val) / std_dev
        
        # Convert to severity (0-1 scale)
        severity = min(z_score / 5, 1.0)  # Normalize to 0-1
        
        return severity
    
    def _calculate_group_semantic_similarity(self, nodes: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate semantic similarity within a group of nodes."""
        if len(nodes) < 2:
            return {'avg_similarity': 1.0, 'min_similarity': 1.0, 'max_similarity': 1.0}
        
        similarities = []
        
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                # Simple similarity based on content overlap
                content1 = set(node1['content'].lower().split())
                content2 = set(node2['content'].lower().split())
                
                if content1 or content2:
                    intersection = len(content1 & content2)
                    union = len(content1 | content2)
                    similarity = intersection / union if union > 0 else 0
                    similarities.append(similarity)
        
        if not similarities:
            return {'avg_similarity': 0.0, 'min_similarity': 0.0, 'max_similarity': 0.0}
        
        return {
            'avg_similarity': sum(similarities) / len(similarities),
            'min_similarity': min(similarities),
            'max_similarity': max(similarities)
        }
    
    def _find_semantic_outliers(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find semantic outliers within a group."""
        if len(nodes) < 3:
            return []
        
        outliers = []
        
        # Calculate each node's average similarity to others
        for i, node in enumerate(nodes):
            similarities_to_others = []
            
            for j, other_node in enumerate(nodes):
                if i != j:
                    content1 = set(node['content'].lower().split())
                    content2 = set(other_node['content'].lower().split())
                    
                    if content1 or content2:
                        intersection = len(content1 & content2)
                        union = len(content1 | content2)
                        similarity = intersection / union if union > 0 else 0
                        similarities_to_others.append(similarity)
            
            if similarities_to_others:
                avg_similarity = sum(similarities_to_others) / len(similarities_to_others)
                
                # Consider as outlier if average similarity is very low
                if avg_similarity < 0.15:  # Low similarity threshold
                    outliers.append(node)
        
        return outliers
    
    def _find_duplicate_content(self, nodes: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Find groups of nodes with duplicate or near-duplicate content."""
        duplicate_groups = []
        processed = set()
        
        for i, node1 in enumerate(nodes):
            if node1['id'] in processed:
                continue
            
            duplicates = [node1]
            content1 = set(node1['content'].lower().split())
            
            for node2 in nodes[i+1:]:
                if node2['id'] in processed:
                    continue
                
                content2 = set(node2['content'].lower().split())
                
                # Calculate similarity
                if content1 or content2:
                    intersection = len(content1 & content2)
                    union = len(content1 | content2)
                    similarity = intersection / union if union > 0 else 0
                    
                    if similarity >= 0.8:  # High similarity threshold for duplicates
                        duplicates.append(node2)
                        processed.add(node2['id'])
            
            if len(duplicates) > 1:
                duplicate_groups.append(duplicates)
                for node in duplicates:
                    processed.add(node['id'])
        
        return duplicate_groups
    
    def _find_behavioral_outliers(self, patterns: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Find nodes with unusual behavioral patterns."""
        outliers = {}
        
        if len(patterns) < 10:  # Need sufficient data for comparison
            return outliers
        
        # Analyze total relationship counts
        rel_counts = [pattern['total_relationships'] for pattern in patterns.values()]
        avg_rel_count = sum(rel_counts) / len(rel_counts)
        
        # Analyze unique relationship type counts
        unique_type_counts = [pattern['unique_rel_types'] for pattern in patterns.values()]
        avg_unique_types = sum(unique_type_counts) / len(unique_type_counts)
        
        for node_id, pattern in patterns.items():
            outlier_reasons = []
            severity_scores = []
            
            # Check for excessive relationships
            if pattern['total_relationships'] > avg_rel_count * 3:
                outlier_reasons.append("excessive_relationships")
                severity = min((pattern['total_relationships'] / avg_rel_count) / 5, 1.0)
                severity_scores.append(severity)
            
            # Check for too few relationships (but not zero)
            if 0 < pattern['total_relationships'] < avg_rel_count * 0.1:
                outlier_reasons.append("minimal_relationships")
                severity = 0.6
                severity_scores.append(severity)
            
            # Check for unusual relationship type diversity
            if pattern['unique_rel_types'] > avg_unique_types * 2:
                outlier_reasons.append("excessive_relationship_diversity")
                severity = min((pattern['unique_rel_types'] / avg_unique_types) / 3, 1.0)
                severity_scores.append(severity)
            
            if outlier_reasons and severity_scores:
                outliers[node_id] = {
                    'reason': ', '.join(outlier_reasons),
                    'severity': max(severity_scores),
                    'confidence': 0.7
                }
        
        return outliers


class InsightDiscoveryEngine:
    """
    Complete Insight Discovery Engine.
    
    Integrates pattern detection, trend analysis, and anomaly detection
    to provide comprehensive insights about knowledge domains.
    """
    
    def __init__(self, query_engine: AdvancedQueryEngine):
        """
        Initialize the Insight Discovery Engine.
        
        Args:
            query_engine: Query engine for accessing knowledge graph
        """
        self.query_engine = query_engine
        self.pattern_detector = PatternDetector(query_engine)
        self.trend_analyzer = TrendAnalyzer(query_engine)
        self.anomaly_detector = AnomalyDetector(query_engine)
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.stats = {
            'insights_generated': 0,
            'patterns_discovered': 0,
            'trends_analyzed': 0,
            'anomalies_detected': 0,
            'avg_processing_time_ms': 0.0
        }
    
    def discover_insights(self, domain: Optional[str] = None,
                         pattern_types: List[PatternType] = None,
                         trend_types: List[TrendType] = None,
                         anomaly_types: List[AnomalyType] = None,
                         time_range: Optional[Tuple[datetime, datetime]] = None) -> InsightReport:
        """
        Discover comprehensive insights for a domain.
        
        Args:
            domain: Optional domain filter
            pattern_types: Types of patterns to detect
            trend_types: Types of trends to analyze
            anomaly_types: Types of anomalies to detect
            time_range: Optional time range for temporal analysis
            
        Returns:
            InsightReport with comprehensive findings
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting insight discovery for domain: {domain or 'all'}")
            
            # Detect patterns
            patterns = self.pattern_detector.detect_patterns(
                domain=domain,
                pattern_types=pattern_types
            )
            
            # Analyze trends
            trends = self.trend_analyzer.analyze_trends(
                domain=domain,
                time_range=time_range,
                trend_types=trend_types
            )
            
            # Detect anomalies
            anomalies = self.anomaly_detector.detect_anomalies(
                domain=domain,
                anomaly_types=anomaly_types
            )
            
            # Generate summary and recommendations
            summary = self._generate_insight_summary(patterns, trends, anomalies)
            recommendations = self._generate_recommendations(patterns, trends, anomalies)
            
            # Calculate statistics
            processing_time = (time.time() - start_time) * 1000
            total_entities = len(set(
                [entity for pattern in patterns for entity in pattern.elements] +
                [entity for trend in trends for entity in trend.entities] +
                [entity for anomaly in anomalies for entity in anomaly.affected_entities]
            ))
            
            confidence_distribution = self._calculate_confidence_distribution(
                patterns, trends, anomalies
            )
            
            # Create insight report
            report = InsightReport(
                patterns=patterns,
                trends=trends,
                anomalies=anomalies,
                summary=summary,
                discovery_time_ms=processing_time,
                total_entities_analyzed=total_entities,
                confidence_distribution=confidence_distribution,
                recommendations=recommendations
            )
            
            # Update statistics
            self._update_statistics(report, start_time)
            
            self.logger.info(f"Insight discovery completed: {len(patterns)} patterns, "
                           f"{len(trends)} trends, {len(anomalies)} anomalies")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error during insight discovery: {e}")
            
            # Return empty report on error
            return InsightReport(
                patterns=[],
                trends=[],
                anomalies=[],
                summary=f"Insight discovery failed: {str(e)}",
                discovery_time_ms=(time.time() - start_time) * 1000,
                total_entities_analyzed=0,
                confidence_distribution={},
                recommendations=[]
            )
    
    def _generate_insight_summary(self, patterns: List[Pattern], 
                                 trends: List[Trend], 
                                 anomalies: List[Anomaly]) -> str:
        """Generate a summary of discovered insights."""
        summary_parts = []
        
        # Patterns summary
        if patterns:
            pattern_types = Counter(pattern.pattern_type.value for pattern in patterns)
            most_common_pattern = pattern_types.most_common(1)[0]
            summary_parts.append(
                f"Discovered {len(patterns)} patterns, primarily {most_common_pattern[0]} patterns"
            )
        else:
            summary_parts.append("No significant patterns detected")
        
        # Trends summary
        if trends:
            trend_types = Counter(trend.trend_type.value for trend in trends)
            most_common_trend = trend_types.most_common(1)[0]
            summary_parts.append(
                f"Identified {len(trends)} trends, mainly {most_common_trend[0]} trends"
            )
        else:
            summary_parts.append("No significant trends identified")
        
        # Anomalies summary
        if anomalies:
            high_severity_anomalies = [a for a in anomalies if a.severity > 0.7]
            summary_parts.append(
                f"Detected {len(anomalies)} anomalies ({len(high_severity_anomalies)} high-severity)"
            )
        else:
            summary_parts.append("No anomalies detected")
        
        return ". ".join(summary_parts) + "."
    
    def _generate_recommendations(self, patterns: List[Pattern], 
                                 trends: List[Trend], 
                                 anomalies: List[Anomaly]) -> List[str]:
        """Generate actionable recommendations based on insights."""
        recommendations = []
        
        # Pattern-based recommendations
        if patterns:
            frequent_patterns = [p for p in patterns if p.pattern_type == PatternType.FREQUENCY]
            if frequent_patterns:
                recommendations.append(
                    "Leverage frequent co-occurrence patterns for improved content organization"
                )
            
            clustering_patterns = [p for p in patterns if p.pattern_type == PatternType.CLUSTERING]
            if clustering_patterns:
                recommendations.append(
                    "Utilize identified clusters for better knowledge navigation"
                )
        
        # Trend-based recommendations
        if trends:
            growth_trends = [t for t in trends if t.trend_type == TrendType.GROWTH]
            if growth_trends:
                recommendations.append(
                    "Focus resources on growing knowledge domains"
                )
            
            decline_trends = [t for t in trends if t.trend_type == TrendType.DECLINE]
            if decline_trends:
                recommendations.append(
                    "Investigate declining areas for potential issues or optimization opportunities"
                )
        
        # Anomaly-based recommendations
        if anomalies:
            high_severity_anomalies = [a for a in anomalies if a.severity > 0.7]
            if high_severity_anomalies:
                recommendations.append(
                    "Address high-severity anomalies to improve data quality"
                )
            
            structural_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.STRUCTURAL]
            if structural_anomalies:
                recommendations.append(
                    "Review graph structure to resolve connectivity issues"
                )
        
        # General recommendations
        if not patterns and not trends:
            recommendations.append(
                "Consider increasing data collection to enable pattern and trend detection"
            )
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _calculate_confidence_distribution(self, patterns: List[Pattern], 
                                         trends: List[Trend], 
                                         anomalies: List[Anomaly]) -> Dict[str, int]:
        """Calculate confidence score distribution."""
        all_confidences = []
        
        # Collect all confidence scores
        all_confidences.extend([p.confidence for p in patterns])
        all_confidences.extend([t.confidence for t in trends])
        all_confidences.extend([a.confidence for a in anomalies])
        
        if not all_confidences:
            return {}
        
        # Create distribution buckets
        distribution = {
            'high': len([c for c in all_confidences if c >= 0.7]),
            'medium': len([c for c in all_confidences if 0.4 <= c < 0.7]),
            'low': len([c for c in all_confidences if c < 0.4])
        }
        
        return distribution
    
    def _update_statistics(self, report: InsightReport, start_time: float):
        """Update engine statistics."""
        self.stats['insights_generated'] += 1
        self.stats['patterns_discovered'] += len(report.patterns)
        self.stats['trends_analyzed'] += len(report.trends)
        self.stats['anomalies_detected'] += len(report.anomalies)
        
        # Update average processing time
        total_time = self.stats['avg_processing_time_ms'] * (self.stats['insights_generated'] - 1)
        self.stats['avg_processing_time_ms'] = (total_time + report.discovery_time_ms) / self.stats['insights_generated']
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get engine statistics.
        
        Returns:
            Dictionary with comprehensive statistics
        """
        return {
            'insight_discovery': self.stats.copy(),
            'query_engine': self.query_engine.get_statistics()
        }