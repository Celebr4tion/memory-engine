"""
Automated Quality Assessment Engine

Provides comprehensive automated quality assessment capabilities beyond
basic ratings, including content quality, structural quality, temporal
quality, and semantic consistency evaluation.
"""

import logging
import re
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from collections import Counter

from memory_core.query.query_engine import AdvancedQueryEngine
from memory_core.query.query_types import QueryRequest, QueryType
from memory_core.model.knowledge_node import KnowledgeNode
from memory_core.model.relationship import Relationship


class QualityDimension(Enum):
    """Different dimensions of quality assessment."""
    CONTENT_QUALITY = "content_quality"  # Content richness, clarity, completeness
    STRUCTURAL_QUALITY = "structural_quality"  # Graph connectivity, relationship quality
    TEMPORAL_QUALITY = "temporal_quality"  # Freshness, temporal consistency
    SEMANTIC_QUALITY = "semantic_quality"  # Semantic consistency, coherence
    SOURCE_QUALITY = "source_quality"  # Source credibility, provenance
    FACTUAL_ACCURACY = "factual_accuracy"  # Factual correctness, verifiability
    RELEVANCE_QUALITY = "relevance_quality"  # Domain relevance, context appropriateness


class QualityLevel(Enum):
    """Quality levels for assessment results."""
    EXCELLENT = "excellent"  # 90-100%
    GOOD = "good"  # 70-89%
    FAIR = "fair"  # 50-69%
    POOR = "poor"  # 30-49%
    CRITICAL = "critical"  # 0-29%


@dataclass
class QualityMetric:
    """Individual quality metric result."""
    dimension: QualityDimension
    score: float  # 0.0 to 1.0
    level: QualityLevel
    confidence: float
    details: Dict[str, Any]
    issues: List[str]
    recommendations: List[str]


@dataclass
class QualityAssessment:
    """Comprehensive quality assessment result."""
    node_id: str
    overall_score: float
    overall_level: QualityLevel
    metrics: Dict[QualityDimension, QualityMetric]
    assessment_confidence: float
    assessment_timestamp: datetime
    critical_issues: List[str]
    improvement_priorities: List[str]
    quality_trend: Optional[str] = None  # 'improving', 'declining', 'stable'


@dataclass
class QualityReport:
    """Quality assessment report for multiple nodes."""
    assessments: List[QualityAssessment]
    summary_statistics: Dict[str, float]
    quality_distribution: Dict[QualityLevel, int]
    common_issues: List[Tuple[str, int]]  # (issue, frequency)
    recommendations: List[str]
    assessment_time_ms: float


class ContentQualityAnalyzer:
    """Analyzes content quality dimensions."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Quality indicators
        self.quality_indicators = {
            'positive': [
                'detailed', 'comprehensive', 'thorough', 'complete', 'accurate',
                'verified', 'documented', 'researched', 'evidenced', 'cited',
                'peer-reviewed', 'validated', 'confirmed', 'established'
            ],
            'negative': [
                'unclear', 'incomplete', 'partial', 'unverified', 'speculative',
                'unconfirmed', 'preliminary', 'draft', 'incomplete', 'fragmentary',
                'vague', 'ambiguous', 'contradictory', 'inconsistent'
            ]
        }
        
        # Content patterns for quality assessment
        self.patterns = {
            'citations': r'\[(\d+)\]|\(([^)]+\d{4}[^)]*)\)',
            'urls': r'https?://[^\s]+',
            'numbers': r'\b\d+(?:\.\d+)?(?:%|kg|meters?|seconds?|minutes?|hours?|days?|years?)?\b',
            'dates': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b',
            'technical_terms': r'\b[A-Z][a-z]*(?:[A-Z][a-z]*)+\b',
            'bullet_points': r'^[\s]*[-*•]\s',
            'headings': r'^#+\s|^[A-Z][^.!?]*:$'
        }
    
    def analyze_content_quality(self, node: KnowledgeNode) -> QualityMetric:
        """
        Analyze content quality of a knowledge node.
        
        Args:
            node: Knowledge node to analyze
            
        Returns:
            QualityMetric for content quality
        """
        try:
            content = node.content
            metadata = node.metadata or {}
            
            # Calculate various content quality indicators
            length_score = self._assess_content_length(content)
            structure_score = self._assess_content_structure(content)
            richness_score = self._assess_content_richness(content)
            clarity_score = self._assess_content_clarity(content)
            completeness_score = self._assess_content_completeness(content, metadata)
            
            # Combine scores
            component_scores = {
                'length': length_score,
                'structure': structure_score,
                'richness': richness_score,
                'clarity': clarity_score,
                'completeness': completeness_score
            }
            
            overall_score = np.mean(list(component_scores.values()))
            
            # Identify issues and recommendations
            issues = self._identify_content_issues(content, component_scores)
            recommendations = self._generate_content_recommendations(component_scores, issues)
            
            # Determine quality level
            quality_level = self._score_to_level(overall_score)
            
            return QualityMetric(
                dimension=QualityDimension.CONTENT_QUALITY,
                score=overall_score,
                level=quality_level,
                confidence=self._calculate_content_confidence(content, component_scores),
                details=component_scores,
                issues=issues,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing content quality: {e}")
            return self._create_error_metric(QualityDimension.CONTENT_QUALITY)
    
    def _assess_content_length(self, content: str) -> float:
        """Assess content length appropriateness."""
        length = len(content)
        
        if length < 50:
            return 0.2  # Too short
        elif length < 100:
            return 0.5  # Short but acceptable
        elif length < 500:
            return 0.8  # Good length
        elif length < 2000:
            return 1.0  # Excellent length
        elif length < 5000:
            return 0.9  # Long but still good
        else:
            return 0.7  # Very long, may be too verbose
    
    def _assess_content_structure(self, content: str) -> float:
        """Assess content structural quality."""
        structure_score = 0.0
        
        # Check for headings
        if re.search(self.patterns['headings'], content, re.MULTILINE):
            structure_score += 0.2
        
        # Check for bullet points or lists
        if re.search(self.patterns['bullet_points'], content, re.MULTILINE):
            structure_score += 0.2
        
        # Check for paragraph structure
        paragraphs = content.split('\n\n')
        if len(paragraphs) > 1:
            structure_score += 0.2
        
        # Check for sentences
        sentences = re.split(r'[.!?]+', content)
        sentence_count = len([s for s in sentences if len(s.strip()) > 10])
        
        if sentence_count >= 2:
            structure_score += 0.2
        if sentence_count >= 5:
            structure_score += 0.2
        
        return min(structure_score, 1.0)
    
    def _assess_content_richness(self, content: str) -> float:
        """Assess content richness (citations, data, technical terms)."""
        richness_score = 0.0
        
        # Check for citations
        citations = re.findall(self.patterns['citations'], content)
        if citations:
            richness_score += min(len(citations) * 0.1, 0.3)
        
        # Check for URLs/links
        urls = re.findall(self.patterns['urls'], content)
        if urls:
            richness_score += min(len(urls) * 0.05, 0.2)
        
        # Check for numerical data
        numbers = re.findall(self.patterns['numbers'], content)
        if numbers:
            richness_score += min(len(numbers) * 0.02, 0.2)
        
        # Check for dates
        dates = re.findall(self.patterns['dates'], content)
        if dates:
            richness_score += min(len(dates) * 0.05, 0.15)
        
        # Check for technical terms
        tech_terms = re.findall(self.patterns['technical_terms'], content)
        if tech_terms:
            richness_score += min(len(tech_terms) * 0.01, 0.15)
        
        return min(richness_score, 1.0)
    
    def _assess_content_clarity(self, content: str) -> float:
        """Assess content clarity and readability."""
        clarity_score = 0.5  # Base score
        
        # Check for positive clarity indicators
        positive_count = sum(1 for indicator in self.quality_indicators['positive'] 
                           if indicator in content.lower())
        clarity_score += min(positive_count * 0.05, 0.3)
        
        # Check for negative clarity indicators
        negative_count = sum(1 for indicator in self.quality_indicators['negative'] 
                           if indicator in content.lower())
        clarity_score -= min(negative_count * 0.1, 0.4)
        
        # Simple readability check (average sentence length)
        sentences = re.split(r'[.!?]+', content)
        if sentences:
            words = content.split()
            avg_sentence_length = len(words) / len(sentences)
            
            if 10 <= avg_sentence_length <= 20:
                clarity_score += 0.2  # Good sentence length
            elif avg_sentence_length > 30:
                clarity_score -= 0.2  # Too long sentences
        
        return max(0.0, min(clarity_score, 1.0))
    
    def _assess_content_completeness(self, content: str, metadata: Dict[str, Any]) -> float:
        """Assess content completeness."""
        completeness_score = 0.5  # Base score
        
        # Check if content addresses key questions (who, what, when, where, why, how)
        key_indicators = {
            'what': ['is', 'are', 'definition', 'meaning', 'refers to'],
            'how': ['process', 'method', 'approach', 'technique', 'procedure'],
            'why': ['because', 'reason', 'cause', 'purpose', 'motivation'],
            'when': ['date', 'time', 'year', 'period', 'during'],
            'where': ['location', 'place', 'region', 'area', 'site']
        }
        
        content_lower = content.lower()
        addressed_questions = 0
        
        for question_type, indicators in key_indicators.items():
            if any(indicator in content_lower for indicator in indicators):
                addressed_questions += 1
        
        completeness_score += (addressed_questions / len(key_indicators)) * 0.4
        
        # Check metadata completeness
        important_metadata = ['source', 'timestamp', 'domain', 'type', 'confidence']
        metadata_completeness = sum(1 for field in important_metadata if field in metadata)
        completeness_score += (metadata_completeness / len(important_metadata)) * 0.1
        
        return min(completeness_score, 1.0)
    
    def _identify_content_issues(self, content: str, scores: Dict[str, float]) -> List[str]:
        """Identify specific content quality issues."""
        issues = []
        
        if scores['length'] < 0.5:
            if len(content) < 50:
                issues.append("Content too short - lacks sufficient detail")
            else:
                issues.append("Content too long - may be overly verbose")
        
        if scores['structure'] < 0.5:
            issues.append("Poor content structure - lacks clear organization")
        
        if scores['richness'] < 0.3:
            issues.append("Low content richness - lacks supporting data/references")
        
        if scores['clarity'] < 0.5:
            issues.append("Clarity issues - content may be unclear or ambiguous")
        
        if scores['completeness'] < 0.5:
            issues.append("Incomplete content - missing key information")
        
        return issues
    
    def _generate_content_recommendations(self, scores: Dict[str, float], issues: List[str]) -> List[str]:
        """Generate content improvement recommendations."""
        recommendations = []
        
        if scores['length'] < 0.5:
            if any('too short' in issue for issue in issues):
                recommendations.append("Add more detailed explanations and examples")
            else:
                recommendations.append("Condense content to improve readability")
        
        if scores['structure'] < 0.5:
            recommendations.append("Improve content organization with headings and lists")
        
        if scores['richness'] < 0.3:
            recommendations.append("Add citations, data, and supporting references")
        
        if scores['clarity'] < 0.5:
            recommendations.append("Clarify ambiguous statements and improve readability")
        
        if scores['completeness'] < 0.5:
            recommendations.append("Add missing information to provide complete coverage")
        
        return recommendations
    
    def _calculate_content_confidence(self, content: str, scores: Dict[str, float]) -> float:
        """Calculate confidence in content quality assessment."""
        # Base confidence on content length and score consistency
        length_factor = min(len(content) / 200, 1.0)  # More content = higher confidence
        
        # Score consistency (lower variance = higher confidence)
        score_values = list(scores.values())
        score_variance = np.var(score_values)
        consistency_factor = max(0.0, 1.0 - score_variance)
        
        return (length_factor * 0.4) + (consistency_factor * 0.6)
    
    def _score_to_level(self, score: float) -> QualityLevel:
        """Convert numeric score to quality level."""
        if score >= 0.9:
            return QualityLevel.EXCELLENT
        elif score >= 0.7:
            return QualityLevel.GOOD
        elif score >= 0.5:
            return QualityLevel.FAIR
        elif score >= 0.3:
            return QualityLevel.POOR
        else:
            return QualityLevel.CRITICAL
    
    def _create_error_metric(self, dimension: QualityDimension) -> QualityMetric:
        """Create error metric when analysis fails."""
        return QualityMetric(
            dimension=dimension,
            score=0.0,
            level=QualityLevel.CRITICAL,
            confidence=0.0,
            details={'error': True},
            issues=['Analysis failed due to error'],
            recommendations=['Review node data and retry analysis']
        )


class StructuralQualityAnalyzer:
    """Analyzes structural quality dimensions."""
    
    def __init__(self, query_engine: AdvancedQueryEngine):
        self.query_engine = query_engine
        self.logger = logging.getLogger(__name__)
    
    def analyze_structural_quality(self, node: KnowledgeNode) -> QualityMetric:
        """
        Analyze structural quality of a knowledge node.
        
        Args:
            node: Knowledge node to analyze
            
        Returns:
            QualityMetric for structural quality
        """
        try:
            # Get node relationships and connectivity info
            relationships = self._get_node_relationships(node.node_id)
            
            # Calculate structural quality indicators
            connectivity_score = self._assess_connectivity(relationships)
            relationship_quality_score = self._assess_relationship_quality(relationships)
            centrality_score = self._assess_centrality(node.node_id, relationships)
            diversity_score = self._assess_relationship_diversity(relationships)
            
            # Combine scores
            component_scores = {
                'connectivity': connectivity_score,
                'relationship_quality': relationship_quality_score,
                'centrality': centrality_score,
                'diversity': diversity_score
            }
            
            overall_score = np.mean(list(component_scores.values()))
            
            # Identify issues and recommendations
            issues = self._identify_structural_issues(component_scores, relationships)
            recommendations = self._generate_structural_recommendations(component_scores, issues)
            
            quality_level = self._score_to_level(overall_score)
            
            return QualityMetric(
                dimension=QualityDimension.STRUCTURAL_QUALITY,
                score=overall_score,
                level=quality_level,
                confidence=self._calculate_structural_confidence(relationships, component_scores),
                details=component_scores,
                issues=issues,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing structural quality: {e}")
            return self._create_error_metric()
    
    def _get_node_relationships(self, node_id: str) -> List[Relationship]:
        """Get all relationships for a node."""
        try:
            # This would normally query the graph database
            # For now, return empty list or mock data
            return []
        except Exception as e:
            self.logger.error(f"Error getting relationships for node {node_id}: {e}")
            return []
    
    def _assess_connectivity(self, relationships: List[Relationship]) -> float:
        """Assess node connectivity."""
        relationship_count = len(relationships)
        
        if relationship_count == 0:
            return 0.0  # Isolated node
        elif relationship_count == 1:
            return 0.3  # Poorly connected
        elif relationship_count <= 3:
            return 0.6  # Adequately connected
        elif relationship_count <= 7:
            return 0.9  # Well connected
        else:
            return 1.0  # Highly connected
    
    def _assess_relationship_quality(self, relationships: List[Relationship]) -> float:
        """Assess quality of relationships."""
        if not relationships:
            return 0.0
        
        quality_scores = []
        
        for rel in relationships:
            rel_score = 0.5  # Base score
            
            # Check relationship type specificity
            if rel.relationship_type in ['related_to', 'associated_with']:
                rel_score -= 0.2  # Generic relationships
            elif rel.relationship_type in ['part_of', 'instance_of', 'causes', 'enables']:
                rel_score += 0.3  # Specific relationships
            
            # Check for relationship metadata/confidence
            if hasattr(rel, 'confidence') and rel.confidence:
                rel_score += rel.confidence * 0.2
            
            quality_scores.append(min(rel_score, 1.0))
        
        return np.mean(quality_scores)
    
    def _assess_centrality(self, node_id: str, relationships: List[Relationship]) -> float:
        """Assess node centrality in the graph."""
        # Simple centrality measure based on relationship count and types
        if not relationships:
            return 0.0
        
        # Count incoming vs outgoing relationships
        incoming = sum(1 for rel in relationships if rel.target_id == node_id)
        outgoing = sum(1 for rel in relationships if rel.source_id == node_id)
        
        # Balanced nodes have better centrality
        total_rels = len(relationships)
        if total_rels == 0:
            return 0.0
        
        balance_ratio = min(incoming, outgoing) / max(incoming, outgoing) if max(incoming, outgoing) > 0 else 0
        
        # Scale by total connectivity
        centrality_score = (balance_ratio * 0.7) + (min(total_rels / 10, 1.0) * 0.3)
        
        return centrality_score
    
    def _assess_relationship_diversity(self, relationships: List[Relationship]) -> float:
        """Assess diversity of relationship types."""
        if not relationships:
            return 0.0
        
        # Count unique relationship types
        rel_types = [rel.relationship_type for rel in relationships]
        unique_types = len(set(rel_types))
        total_rels = len(relationships)
        
        # Higher diversity is better (up to a point)
        diversity_ratio = unique_types / total_rels
        
        if diversity_ratio >= 0.8:
            return 1.0  # Very diverse
        elif diversity_ratio >= 0.6:
            return 0.8  # Good diversity
        elif diversity_ratio >= 0.4:
            return 0.6  # Fair diversity
        elif diversity_ratio >= 0.2:
            return 0.4  # Poor diversity
        else:
            return 0.2  # Very poor diversity
    
    def _identify_structural_issues(self, scores: Dict[str, float], relationships: List[Relationship]) -> List[str]:
        """Identify structural quality issues."""
        issues = []
        
        if scores['connectivity'] < 0.3:
            if not relationships:
                issues.append("Node is isolated - no relationships found")
            else:
                issues.append("Poor connectivity - insufficient relationships")
        
        if scores['relationship_quality'] < 0.5:
            issues.append("Low relationship quality - generic or weak relationships")
        
        if scores['centrality'] < 0.3:
            issues.append("Low centrality - node is peripheral in the graph")
        
        if scores['diversity'] < 0.4:
            issues.append("Poor relationship diversity - limited relationship types")
        
        return issues
    
    def _generate_structural_recommendations(self, scores: Dict[str, float], issues: List[str]) -> List[str]:
        """Generate structural improvement recommendations."""
        recommendations = []
        
        if scores['connectivity'] < 0.3:
            recommendations.append("Add more relationships to improve connectivity")
        
        if scores['relationship_quality'] < 0.5:
            recommendations.append("Replace generic relationships with more specific ones")
        
        if scores['centrality'] < 0.3:
            recommendations.append("Link to more central nodes to improve position")
        
        if scores['diversity'] < 0.4:
            recommendations.append("Add diverse relationship types")
        
        return recommendations
    
    def _calculate_structural_confidence(self, relationships: List[Relationship], scores: Dict[str, float]) -> float:
        """Calculate confidence in structural quality assessment."""
        # Base confidence on relationship count and score consistency
        rel_count_factor = min(len(relationships) / 5, 1.0)
        
        score_values = list(scores.values())
        score_variance = np.var(score_values)
        consistency_factor = max(0.0, 1.0 - score_variance)
        
        return (rel_count_factor * 0.5) + (consistency_factor * 0.5)
    
    def _score_to_level(self, score: float) -> QualityLevel:
        """Convert numeric score to quality level."""
        if score >= 0.9:
            return QualityLevel.EXCELLENT
        elif score >= 0.7:
            return QualityLevel.GOOD
        elif score >= 0.5:
            return QualityLevel.FAIR
        elif score >= 0.3:
            return QualityLevel.POOR
        else:
            return QualityLevel.CRITICAL
    
    def _create_error_metric(self) -> QualityMetric:
        """Create error metric when analysis fails."""
        return QualityMetric(
            dimension=QualityDimension.STRUCTURAL_QUALITY,
            score=0.0,
            level=QualityLevel.CRITICAL,
            confidence=0.0,
            details={'error': True},
            issues=['Structural analysis failed'],
            recommendations=['Check node relationships and connectivity']
        )


class TemporalQualityAnalyzer:
    """Analyzes temporal quality dimensions."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_temporal_quality(self, node: KnowledgeNode) -> QualityMetric:
        """
        Analyze temporal quality of a knowledge node.
        
        Args:
            node: Knowledge node to analyze
            
        Returns:
            QualityMetric for temporal quality
        """
        try:
            metadata = node.metadata or {}
            
            # Calculate temporal quality indicators
            freshness_score = self._assess_freshness(metadata)
            temporal_consistency_score = self._assess_temporal_consistency(node.content, metadata)
            update_frequency_score = self._assess_update_frequency(metadata)
            temporal_relevance_score = self._assess_temporal_relevance(node.content)
            
            # Combine scores
            component_scores = {
                'freshness': freshness_score,
                'temporal_consistency': temporal_consistency_score,
                'update_frequency': update_frequency_score,
                'temporal_relevance': temporal_relevance_score
            }
            
            overall_score = np.mean(list(component_scores.values()))
            
            # Identify issues and recommendations
            issues = self._identify_temporal_issues(component_scores, metadata)
            recommendations = self._generate_temporal_recommendations(component_scores, issues)
            
            quality_level = self._score_to_level(overall_score)
            
            return QualityMetric(
                dimension=QualityDimension.TEMPORAL_QUALITY,
                score=overall_score,
                level=quality_level,
                confidence=self._calculate_temporal_confidence(metadata, component_scores),
                details=component_scores,
                issues=issues,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing temporal quality: {e}")
            return self._create_error_metric()
    
    def _assess_freshness(self, metadata: Dict[str, Any]) -> float:
        """Assess content freshness."""
        # Look for timestamp fields
        timestamp_fields = ['timestamp', 'created_at', 'updated_at', 'modified_at', 'date']
        
        latest_timestamp = None
        for field in timestamp_fields:
            if field in metadata:
                try:
                    if isinstance(metadata[field], str):
                        latest_timestamp = datetime.fromisoformat(metadata[field].replace('Z', '+00:00'))
                    elif isinstance(metadata[field], (int, float)):
                        latest_timestamp = datetime.fromtimestamp(metadata[field])
                    break
                except:
                    continue
        
        if not latest_timestamp:
            return 0.3  # No timestamp available
        
        # Calculate age
        now = datetime.now()
        if latest_timestamp.tzinfo:
            now = now.replace(tzinfo=latest_timestamp.tzinfo)
        
        age = now - latest_timestamp
        
        # Score based on age
        if age.days <= 30:
            return 1.0  # Very fresh
        elif age.days <= 90:
            return 0.8  # Fresh
        elif age.days <= 180:
            return 0.6  # Moderate
        elif age.days <= 365:
            return 0.4  # Old
        else:
            return 0.2  # Very old
    
    def _assess_temporal_consistency(self, content: str, metadata: Dict[str, Any]) -> float:
        """Assess temporal consistency in content."""
        # Extract dates from content
        date_patterns = [
            r'\b\d{4}\b',  # Years
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # Dates
            r'\b\d{4}-\d{2}-\d{2}\b'  # ISO dates
        ]
        
        found_dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, content)
            found_dates.extend(matches)
        
        if not found_dates:
            return 0.7  # No temporal references (neutral)
        
        # Check if dates are consistent with metadata timestamp
        metadata_year = None
        timestamp_fields = ['timestamp', 'created_at', 'updated_at', 'date']
        
        for field in timestamp_fields:
            if field in metadata:
                try:
                    if isinstance(metadata[field], str):
                        dt = datetime.fromisoformat(metadata[field].replace('Z', '+00:00'))
                        metadata_year = dt.year
                    break
                except:
                    continue
        
        if not metadata_year:
            return 0.6  # Can't verify consistency
        
        # Check for temporal consistency
        inconsistent_dates = 0
        total_dates = 0
        
        for date_str in found_dates:
            try:
                if len(date_str) == 4 and date_str.isdigit():  # Year
                    year = int(date_str)
                    if 1900 <= year <= datetime.now().year:  # Valid year range
                        total_dates += 1
                        if abs(year - metadata_year) > 5:  # Allow 5-year variance
                            inconsistent_dates += 1
            except:
                continue
        
        if total_dates == 0:
            return 0.7  # No valid dates found
        
        consistency_ratio = 1.0 - (inconsistent_dates / total_dates)
        return consistency_ratio
    
    def _assess_update_frequency(self, metadata: Dict[str, Any]) -> float:
        """Assess update frequency appropriateness."""
        # Look for update history in metadata
        update_fields = ['update_count', 'version', 'revision_count']
        
        for field in update_fields:
            if field in metadata:
                count = metadata[field]
                if isinstance(count, int):
                    if count == 0:
                        return 0.5  # Never updated
                    elif count <= 3:
                        return 0.7  # Occasionally updated
                    elif count <= 10:
                        return 0.9  # Regularly updated
                    else:
                        return 1.0  # Frequently updated
        
        # Check for multiple timestamp fields (indicating updates)
        timestamp_fields = ['created_at', 'updated_at', 'modified_at']
        present_timestamps = sum(1 for field in timestamp_fields if field in metadata)
        
        if present_timestamps >= 2:
            return 0.8  # Has update timestamps
        else:
            return 0.6  # Limited update information
    
    def _assess_temporal_relevance(self, content: str) -> float:
        """Assess temporal relevance of content."""
        # Look for temporal indicators
        temporal_indicators = {
            'current': ['current', 'currently', 'now', 'today', 'recent', 'latest', 'modern'],
            'historical': ['historical', 'past', 'former', 'previous', 'legacy', 'traditional'],
            'future': ['future', 'upcoming', 'planned', 'expected', 'projected', 'anticipated']
        }
        
        content_lower = content.lower()
        
        current_count = sum(1 for indicator in temporal_indicators['current'] 
                          if indicator in content_lower)
        historical_count = sum(1 for indicator in temporal_indicators['historical'] 
                             if indicator in content_lower)
        future_count = sum(1 for indicator in temporal_indicators['future'] 
                         if indicator in content_lower)
        
        total_indicators = current_count + historical_count + future_count
        
        if total_indicators == 0:
            return 0.7  # No clear temporal context
        
        # Prefer current and future content
        relevance_score = ((current_count * 1.0) + (future_count * 0.8) + (historical_count * 0.6)) / total_indicators
        
        return min(relevance_score, 1.0)
    
    def _identify_temporal_issues(self, scores: Dict[str, float], metadata: Dict[str, Any]) -> List[str]:
        """Identify temporal quality issues."""
        issues = []
        
        if scores['freshness'] < 0.4:
            issues.append("Content is outdated - may need refresh")
        
        if scores['temporal_consistency'] < 0.5:
            issues.append("Temporal inconsistencies detected in content")
        
        if scores['update_frequency'] < 0.5:
            issues.append("Content lacks update history")
        
        if scores['temporal_relevance'] < 0.5:
            issues.append("Content may not be temporally relevant")
        
        return issues
    
    def _generate_temporal_recommendations(self, scores: Dict[str, float], issues: List[str]) -> List[str]:
        """Generate temporal improvement recommendations."""
        recommendations = []
        
        if scores['freshness'] < 0.4:
            recommendations.append("Update content with recent information")
        
        if scores['temporal_consistency'] < 0.5:
            recommendations.append("Review and fix temporal inconsistencies")
        
        if scores['update_frequency'] < 0.5:
            recommendations.append("Establish regular update schedule")
        
        if scores['temporal_relevance'] < 0.5:
            recommendations.append("Add temporal context indicators")
        
        return recommendations
    
    def _calculate_temporal_confidence(self, metadata: Dict[str, Any], scores: Dict[str, float]) -> float:
        """Calculate confidence in temporal quality assessment."""
        # Base confidence on metadata completeness
        temporal_fields = ['timestamp', 'created_at', 'updated_at', 'date']
        metadata_completeness = sum(1 for field in temporal_fields if field in metadata)
        metadata_factor = metadata_completeness / len(temporal_fields)
        
        # Score consistency
        score_values = list(scores.values())
        score_variance = np.var(score_values)
        consistency_factor = max(0.0, 1.0 - score_variance)
        
        return (metadata_factor * 0.6) + (consistency_factor * 0.4)
    
    def _score_to_level(self, score: float) -> QualityLevel:
        """Convert numeric score to quality level."""
        if score >= 0.9:
            return QualityLevel.EXCELLENT
        elif score >= 0.7:
            return QualityLevel.GOOD
        elif score >= 0.5:
            return QualityLevel.FAIR
        elif score >= 0.3:
            return QualityLevel.POOR
        else:
            return QualityLevel.CRITICAL
    
    def _create_error_metric(self) -> QualityMetric:
        """Create error metric when analysis fails."""
        return QualityMetric(
            dimension=QualityDimension.TEMPORAL_QUALITY,
            score=0.0,
            level=QualityLevel.CRITICAL,
            confidence=0.0,
            details={'error': True},
            issues=['Temporal analysis failed'],
            recommendations=['Check node temporal metadata']
        )


class QualityAssessmentEngine:
    """
    Main Quality Assessment Engine.
    
    Provides comprehensive automated quality assessment capabilities
    across multiple quality dimensions.
    """
    
    def __init__(self, query_engine: AdvancedQueryEngine):
        """
        Initialize the Quality Assessment Engine.
        
        Args:
            query_engine: Query engine for accessing knowledge graph
        """
        self.query_engine = query_engine
        
        # Initialize analyzers
        self.content_analyzer = ContentQualityAnalyzer()
        self.structural_analyzer = StructuralQualityAnalyzer(query_engine)
        self.temporal_analyzer = TemporalQualityAnalyzer()
        
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.stats = {
            'assessments_performed': 0,
            'avg_assessment_time_ms': 0.0,
            'quality_distribution': {level.value: 0 for level in QualityLevel},
            'common_issues': Counter(),
            'improvement_trends': {}
        }
    
    def assess_node_quality(self, node: KnowledgeNode, 
                           dimensions: List[QualityDimension] = None) -> QualityAssessment:
        """
        Perform comprehensive quality assessment of a knowledge node.
        
        Args:
            node: Knowledge node to assess
            dimensions: Quality dimensions to assess (all if None)
            
        Returns:
            QualityAssessment with comprehensive results
        """
        start_time = time.time()
        
        if dimensions is None:
            dimensions = [
                QualityDimension.CONTENT_QUALITY,
                QualityDimension.STRUCTURAL_QUALITY,
                QualityDimension.TEMPORAL_QUALITY
            ]
        
        try:
            self.logger.info(f"Assessing quality for node {node.node_id}")
            
            # Perform assessment for each dimension
            metrics = {}
            
            if QualityDimension.CONTENT_QUALITY in dimensions:
                metrics[QualityDimension.CONTENT_QUALITY] = self.content_analyzer.analyze_content_quality(node)
            
            if QualityDimension.STRUCTURAL_QUALITY in dimensions:
                metrics[QualityDimension.STRUCTURAL_QUALITY] = self.structural_analyzer.analyze_structural_quality(node)
            
            if QualityDimension.TEMPORAL_QUALITY in dimensions:
                metrics[QualityDimension.TEMPORAL_QUALITY] = self.temporal_analyzer.analyze_temporal_quality(node)
            
            # Calculate overall quality score
            overall_score = np.mean([metric.score for metric in metrics.values()])
            overall_level = self._score_to_level(overall_score)
            
            # Collect critical issues and priorities
            critical_issues = []
            improvement_priorities = []
            
            for metric in metrics.values():
                if metric.level in [QualityLevel.CRITICAL, QualityLevel.POOR]:
                    critical_issues.extend(metric.issues)
                improvement_priorities.extend(metric.recommendations)
            
            # Calculate assessment confidence
            assessment_confidence = np.mean([metric.confidence for metric in metrics.values()])
            
            # Create assessment result
            assessment = QualityAssessment(
                node_id=node.node_id,
                overall_score=overall_score,
                overall_level=overall_level,
                metrics=metrics,
                assessment_confidence=assessment_confidence,
                assessment_timestamp=datetime.now(),
                critical_issues=critical_issues,
                improvement_priorities=improvement_priorities[:5]  # Top 5 priorities
            )
            
            # Update statistics
            processing_time = (time.time() - start_time) * 1000
            self._update_statistics(assessment, processing_time)
            
            self.logger.info(f"Quality assessment completed for {node.node_id}: {overall_level.value}")
            return assessment
            
        except Exception as e:
            self.logger.error(f"Error assessing node quality: {e}")
            return self._create_error_assessment(node.node_id, start_time)
    
    def assess_multiple_nodes(self, nodes: List[KnowledgeNode],
                             dimensions: List[QualityDimension] = None) -> QualityReport:
        """
        Assess quality for multiple nodes and generate comprehensive report.
        
        Args:
            nodes: List of knowledge nodes to assess
            dimensions: Quality dimensions to assess
            
        Returns:
            QualityReport with aggregated results
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Assessing quality for {len(nodes)} nodes")
            
            # Assess each node
            assessments = []
            for node in nodes:
                assessment = self.assess_node_quality(node, dimensions)
                assessments.append(assessment)
            
            # Generate summary statistics
            summary_stats = self._calculate_summary_statistics(assessments)
            
            # Calculate quality distribution
            quality_distribution = {level: 0 for level in QualityLevel}
            for assessment in assessments:
                quality_distribution[assessment.overall_level] += 1
            
            # Find common issues
            all_issues = []
            for assessment in assessments:
                all_issues.extend(assessment.critical_issues)
            
            common_issues = Counter(all_issues).most_common(10)
            
            # Generate recommendations
            recommendations = self._generate_global_recommendations(assessments)
            
            processing_time = (time.time() - start_time) * 1000
            
            report = QualityReport(
                assessments=assessments,
                summary_statistics=summary_stats,
                quality_distribution=quality_distribution,
                common_issues=common_issues,
                recommendations=recommendations,
                assessment_time_ms=processing_time
            )
            
            self.logger.info(f"Quality report generated for {len(nodes)} nodes in {processing_time:.1f}ms")
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating quality report: {e}")
            return QualityReport(
                assessments=[],
                summary_statistics={},
                quality_distribution={level: 0 for level in QualityLevel},
                common_issues=[],
                recommendations=[],
                assessment_time_ms=(time.time() - start_time) * 1000
            )
    
    def _calculate_summary_statistics(self, assessments: List[QualityAssessment]) -> Dict[str, float]:
        """Calculate summary statistics for assessments."""
        if not assessments:
            return {}
        
        overall_scores = [assessment.overall_score for assessment in assessments]
        confidence_scores = [assessment.assessment_confidence for assessment in assessments]
        
        # Calculate dimension-specific averages
        dimension_averages = {}
        for dimension in QualityDimension:
            dimension_scores = []
            for assessment in assessments:
                if dimension in assessment.metrics:
                    dimension_scores.append(assessment.metrics[dimension].score)
            
            if dimension_scores:
                dimension_averages[f'{dimension.value}_avg'] = np.mean(dimension_scores)
        
        stats = {
            'avg_overall_score': np.mean(overall_scores),
            'median_overall_score': np.median(overall_scores),
            'min_overall_score': np.min(overall_scores),
            'max_overall_score': np.max(overall_scores),
            'std_overall_score': np.std(overall_scores),
            'avg_confidence': np.mean(confidence_scores),
            'total_nodes_assessed': len(assessments)
        }
        
        stats.update(dimension_averages)
        return stats
    
    def _generate_global_recommendations(self, assessments: List[QualityAssessment]) -> List[str]:
        """Generate global recommendations based on assessment results."""
        recommendations = []
        
        if not assessments:
            return recommendations
        
        # Analyze overall quality distribution
        quality_counts = {level: 0 for level in QualityLevel}
        for assessment in assessments:
            quality_counts[assessment.overall_level] += 1
        
        total_nodes = len(assessments)
        
        # Generate recommendations based on quality patterns
        poor_quality_ratio = (quality_counts[QualityLevel.POOR] + quality_counts[QualityLevel.CRITICAL]) / total_nodes
        
        if poor_quality_ratio > 0.3:
            recommendations.append("Significant quality issues detected - implement systematic quality improvement process")
        
        if quality_counts[QualityLevel.CRITICAL] > 0:
            recommendations.append("Address critical quality issues immediately to prevent data degradation")
        
        # Analyze common dimension issues
        dimension_issues = {dimension: 0 for dimension in QualityDimension}
        for assessment in assessments:
            for dimension, metric in assessment.metrics.items():
                if metric.level in [QualityLevel.POOR, QualityLevel.CRITICAL]:
                    dimension_issues[dimension] += 1
        
        # Recommend focus areas
        max_issues_dimension = max(dimension_issues.items(), key=lambda x: x[1])
        if max_issues_dimension[1] > total_nodes * 0.2:
            dim_name = max_issues_dimension[0].value.replace('_', ' ')
            recommendations.append(f"Focus improvement efforts on {dim_name} - highest issue frequency")
        
        # General recommendations
        avg_confidence = np.mean([a.assessment_confidence for a in assessments])
        if avg_confidence < 0.6:
            recommendations.append("Improve metadata completeness to increase assessment confidence")
        
        return recommendations[:5]  # Top 5 recommendations
    
    def _score_to_level(self, score: float) -> QualityLevel:
        """Convert numeric score to quality level."""
        if score >= 0.9:
            return QualityLevel.EXCELLENT
        elif score >= 0.7:
            return QualityLevel.GOOD
        elif score >= 0.5:
            return QualityLevel.FAIR
        elif score >= 0.3:
            return QualityLevel.POOR
        else:
            return QualityLevel.CRITICAL
    
    def _create_error_assessment(self, node_id: str, start_time: float) -> QualityAssessment:
        """Create error assessment when analysis fails."""
        return QualityAssessment(
            node_id=node_id,
            overall_score=0.0,
            overall_level=QualityLevel.CRITICAL,
            metrics={},
            assessment_confidence=0.0,
            assessment_timestamp=datetime.now(),
            critical_issues=["Quality assessment failed due to error"],
            improvement_priorities=["Review node data and retry assessment"]
        )
    
    def _update_statistics(self, assessment: QualityAssessment, processing_time: float):
        """Update engine statistics."""
        self.stats['assessments_performed'] += 1
        
        # Update average processing time
        total_time = self.stats['avg_assessment_time_ms'] * (self.stats['assessments_performed'] - 1)
        self.stats['avg_assessment_time_ms'] = (total_time + processing_time) / self.stats['assessments_performed']
        
        # Update quality distribution
        self.stats['quality_distribution'][assessment.overall_level.value] += 1
        
        # Update common issues
        for issue in assessment.critical_issues:
            self.stats['common_issues'][issue] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get quality assessment engine statistics.
        
        Returns:
            Dictionary with comprehensive statistics
        """
        return {
            'quality_assessment': self.stats.copy(),
            'query_engine': self.query_engine.get_statistics()
        }