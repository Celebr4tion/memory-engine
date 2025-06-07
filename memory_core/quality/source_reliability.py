"""
Source Reliability Scoring Engine

Provides comprehensive source reliability assessment based on multiple factors
including authority, credibility, freshness, consistency, and validation history.
"""

import logging
import time
import re
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, Counter
from urllib.parse import urlparse

from memory_core.query.query_engine import AdvancedQueryEngine
from memory_core.query.query_types import QueryRequest, QueryType
from memory_core.model.knowledge_node import KnowledgeNode


class ReliabilityLevel(Enum):
    """Reliability levels for sources."""
    HIGHLY_RELIABLE = "highly_reliable"  # 90-100%
    RELIABLE = "reliable"  # 70-89%
    MODERATELY_RELIABLE = "moderately_reliable"  # 50-69%
    QUESTIONABLE = "questionable"  # 30-49%
    UNRELIABLE = "unreliable"  # 0-29%


class SourceType(Enum):
    """Types of information sources."""
    ACADEMIC = "academic"  # Peer-reviewed journals, academic institutions
    GOVERNMENTAL = "governmental"  # Government agencies, official reports
    NEWS_MEDIA = "news_media"  # Established news organizations
    PROFESSIONAL = "professional"  # Industry publications, professional organizations
    REFERENCE = "reference"  # Encyclopedia, reference works
    USER_GENERATED = "user_generated"  # Wikis, forums, blogs
    COMMERCIAL = "commercial"  # Company websites, marketing materials
    UNKNOWN = "unknown"  # Cannot determine source type


@dataclass
class ReliabilityMetric:
    """Individual reliability metric result."""
    metric_name: str
    score: float  # 0.0 to 1.0
    confidence: float
    evidence: List[str]
    weight: float  # Importance weight in overall calculation


@dataclass
class SourceReliabilityScore:
    """Comprehensive source reliability assessment."""
    source_identifier: str
    overall_score: float
    reliability_level: ReliabilityLevel
    source_type: SourceType
    metrics: Dict[str, ReliabilityMetric]
    assessment_confidence: float
    factors_considered: List[str]
    improvement_suggestions: List[str]
    last_assessed: datetime
    assessment_history: List[Dict[str, Any]] = None


@dataclass
class ReliabilityReport:
    """Source reliability assessment report."""
    source_scores: List[SourceReliabilityScore]
    summary_statistics: Dict[str, float]
    reliability_distribution: Dict[ReliabilityLevel, int]
    source_type_distribution: Dict[SourceType, int]
    recommendations: List[str]
    assessment_time_ms: float


class SourceIdentifier:
    """Identifies and categorizes information sources."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Source type patterns
        self.source_patterns = {
            SourceType.ACADEMIC: [
                r'\.edu/', r'journal', r'research', r'academic', r'university',
                r'\.org.*research', r'peer.?review', r'publication', r'doi:',
                r'arxiv', r'pubmed', r'scholar\.google'
            ],
            SourceType.GOVERNMENTAL: [
                r'\.gov/', r'\.mil/', r'government', r'ministry', r'department',
                r'agency', r'official', r'state\.', r'federal', r'national'
            ],
            SourceType.NEWS_MEDIA: [
                r'news', r'times', r'post', r'herald', r'guardian', r'reuters',
                r'associated.?press', r'cnn', r'bbc', r'npr', r'journalism'
            ],
            SourceType.PROFESSIONAL: [
                r'association', r'institute', r'society', r'professional',
                r'industry', r'trade', r'\.org.*professional'
            ],
            SourceType.REFERENCE: [
                r'encyclopedia', r'dictionary', r'reference', r'britannica',
                r'wikipedia', r'reference.?work'
            ],
            SourceType.USER_GENERATED: [
                r'wiki', r'blog', r'forum', r'reddit', r'stackoverflow',
                r'user.?generated', r'community', r'discussion'
            ],
            SourceType.COMMERCIAL: [
                r'\.com/', r'company', r'corporation', r'business', r'marketing',
                r'product', r'service', r'commercial'
            ]
        }
        
        # Authority indicators
        self.authority_indicators = {
            'high': [
                'peer-reviewed', 'published', 'editorial board', 'impact factor',
                'indexed', 'citation', 'expert', 'authority', 'established'
            ],
            'medium': [
                'reviewed', 'editorial', 'professional', 'certified', 'accredited'
            ],
            'low': [
                'opinion', 'personal', 'unofficial', 'unverified', 'amateur'
            ]
        }
    
    def identify_source(self, source_info: str, metadata: Dict[str, Any]) -> Tuple[str, SourceType]:
        """
        Identify and categorize a source.
        
        Args:
            source_info: Source information (URL, citation, etc.)
            metadata: Additional metadata about the source
            
        Returns:
            Tuple of (source_identifier, source_type)
        """
        try:
            # Clean and normalize source info
            source_identifier = self._normalize_source_identifier(source_info)
            
            # Determine source type
            source_type = self._classify_source_type(source_info.lower(), metadata)
            
            return source_identifier, source_type
            
        except Exception as e:
            self.logger.error(f"Error identifying source: {e}")
            return source_info, SourceType.UNKNOWN
    
    def _normalize_source_identifier(self, source_info: str) -> str:
        """Normalize source identifier for consistent tracking."""
        # If it's a URL, extract domain
        if source_info.startswith(('http://', 'https://')):
            try:
                parsed = urlparse(source_info)
                return f"{parsed.netloc}{parsed.path}".rstrip('/')
            except:
                return source_info
        
        # If it's a citation, extract key parts
        if 'doi:' in source_info.lower():
            doi_match = re.search(r'doi:\s*([^\s,]+)', source_info, re.IGNORECASE)
            if doi_match:
                return f"doi:{doi_match.group(1)}"
        
        # Clean up general source info
        cleaned = re.sub(r'\s+', ' ', source_info.strip())
        return cleaned[:200]  # Limit length
    
    def _classify_source_type(self, source_info: str, metadata: Dict[str, Any]) -> SourceType:
        """Classify the type of source."""
        # Check metadata first
        if 'source_type' in metadata:
            try:
                return SourceType(metadata['source_type'])
            except:
                pass
        
        # Pattern-based classification
        for source_type, patterns in self.source_patterns.items():
            for pattern in patterns:
                if re.search(pattern, source_info, re.IGNORECASE):
                    return source_type
        
        return SourceType.UNKNOWN


class AuthorityAnalyzer:
    """Analyzes source authority and credibility."""
    
    def __init__(self, query_engine: AdvancedQueryEngine):
        self.query_engine = query_engine
        self.logger = logging.getLogger(__name__)
        
        # Known high-authority sources
        self.authority_database = {
            'high': [
                'nature.com', 'science.org', 'nejm.org', 'thelancet.com',
                'ieee.org', 'acm.org', 'springer.com', 'elsevier.com',
                'nih.gov', 'who.int', 'cdc.gov', 'fda.gov'
            ],
            'medium': [
                'wikipedia.org', 'britannica.com', 'reuters.com', 'bbc.com',
                'npr.org', 'pbs.org', 'smithsonian.edu'
            ],
            'questionable': [
                'tabloid', 'conspiracy', 'unverified', 'clickbait'
            ]
        }
    
    def analyze_authority(self, source_identifier: str, source_type: SourceType,
                         content: str, metadata: Dict[str, Any]) -> ReliabilityMetric:
        """
        Analyze source authority.
        
        Args:
            source_identifier: Source identifier
            source_type: Type of source
            content: Content from the source
            metadata: Source metadata
            
        Returns:
            ReliabilityMetric for authority
        """
        try:
            authority_scores = []
            evidence = []
            
            # Check against known authority database
            db_score = self._check_authority_database(source_identifier)
            if db_score is not None:
                authority_scores.append(db_score)
                evidence.append(f"Known authority database match: {db_score:.2f}")
            
            # Source type authority
            type_score = self._get_source_type_authority(source_type)
            authority_scores.append(type_score)
            evidence.append(f"Source type authority ({source_type.value}): {type_score:.2f}")
            
            # Content-based authority indicators
            content_score = self._analyze_content_authority(content)
            authority_scores.append(content_score)
            evidence.append(f"Content authority indicators: {content_score:.2f}")
            
            # Metadata-based authority
            metadata_score = self._analyze_metadata_authority(metadata)
            authority_scores.append(metadata_score)
            evidence.append(f"Metadata authority indicators: {metadata_score:.2f}")
            
            # Calculate weighted average
            weights = [0.4, 0.2, 0.2, 0.2]  # Database, type, content, metadata
            overall_score = np.average(authority_scores, weights=weights)
            
            # Calculate confidence based on available evidence
            confidence = self._calculate_authority_confidence(
                source_identifier, source_type, len(evidence)
            )
            
            return ReliabilityMetric(
                metric_name="authority",
                score=overall_score,
                confidence=confidence,
                evidence=evidence,
                weight=0.3  # Authority weight in overall reliability
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing authority: {e}")
            return self._create_error_metric("authority")
    
    def _check_authority_database(self, source_identifier: str) -> Optional[float]:
        """Check source against authority database."""
        source_lower = source_identifier.lower()
        
        # Check high authority sources
        for domain in self.authority_database['high']:
            if domain in source_lower:
                return 0.9
        
        # Check medium authority sources
        for domain in self.authority_database['medium']:
            if domain in source_lower:
                return 0.7
        
        # Check questionable sources
        for indicator in self.authority_database['questionable']:
            if indicator in source_lower:
                return 0.2
        
        return None  # Not found in database
    
    def _get_source_type_authority(self, source_type: SourceType) -> float:
        """Get authority score based on source type."""
        type_scores = {
            SourceType.ACADEMIC: 0.9,
            SourceType.GOVERNMENTAL: 0.8,
            SourceType.PROFESSIONAL: 0.7,
            SourceType.NEWS_MEDIA: 0.6,
            SourceType.REFERENCE: 0.7,
            SourceType.USER_GENERATED: 0.3,
            SourceType.COMMERCIAL: 0.4,
            SourceType.UNKNOWN: 0.5
        }
        
        return type_scores.get(source_type, 0.5)
    
    def _analyze_content_authority(self, content: str) -> float:
        """Analyze content for authority indicators."""
        authority_score = 0.5  # Base score
        content_lower = content.lower()
        
        # Check for positive authority indicators
        for indicator in self.authority_database['high']:
            if indicator in content_lower:
                authority_score += 0.1
        
        # Check for citation patterns
        citation_patterns = [
            r'\[[0-9]+\]',  # Numbered citations
            r'\([^)]*\d{4}[^)]*\)',  # Year citations
            r'doi:',  # DOI references
            r'et al\.',  # Academic citations
        ]
        
        for pattern in citation_patterns:
            if re.search(pattern, content):
                authority_score += 0.05
        
        # Check for methodology indicators
        methodology_indicators = ['method', 'methodology', 'experiment', 'study', 'analysis']
        for indicator in methodology_indicators:
            if indicator in content_lower:
                authority_score += 0.02
        
        return min(authority_score, 1.0)
    
    def _analyze_metadata_authority(self, metadata: Dict[str, Any]) -> float:
        """Analyze metadata for authority indicators."""
        authority_score = 0.5  # Base score
        
        # Check for author credentials
        if 'author' in metadata:
            author_info = str(metadata['author']).lower()
            if any(indicator in author_info for indicator in ['dr.', 'ph.d', 'professor', 'phd']):
                authority_score += 0.2
        
        # Check for publication info
        if 'publication' in metadata:
            pub_info = str(metadata['publication']).lower()
            if any(indicator in pub_info for indicator in ['journal', 'review', 'proceedings']):
                authority_score += 0.1
        
        # Check for institutional affiliation
        if 'institution' in metadata:
            authority_score += 0.1
        
        # Check for peer review status
        if metadata.get('peer_reviewed', False):
            authority_score += 0.2
        
        return min(authority_score, 1.0)
    
    def _calculate_authority_confidence(self, source_identifier: str, 
                                      source_type: SourceType, evidence_count: int) -> float:
        """Calculate confidence in authority assessment."""
        confidence = 0.5  # Base confidence
        
        # Higher confidence for known sources
        if any(domain in source_identifier.lower() for domain in 
               self.authority_database['high'] + self.authority_database['medium']):
            confidence += 0.3
        
        # Higher confidence for specific source types
        if source_type in [SourceType.ACADEMIC, SourceType.GOVERNMENTAL]:
            confidence += 0.2
        
        # Confidence based on evidence
        confidence += min(evidence_count * 0.05, 0.2)
        
        return min(confidence, 1.0)
    
    def _create_error_metric(self, metric_name: str) -> ReliabilityMetric:
        """Create error metric when analysis fails."""
        return ReliabilityMetric(
            metric_name=metric_name,
            score=0.3,  # Conservative score
            confidence=0.1,
            evidence=[f"Error analyzing {metric_name}"],
            weight=0.0
        )


class ConsistencyAnalyzer:
    """Analyzes source consistency and validation history."""
    
    def __init__(self, query_engine: AdvancedQueryEngine):
        self.query_engine = query_engine
        self.logger = logging.getLogger(__name__)
    
    def analyze_consistency(self, source_identifier: str, 
                          node: KnowledgeNode) -> ReliabilityMetric:
        """
        Analyze source consistency.
        
        Args:
            source_identifier: Source identifier
            node: Current knowledge node
            
        Returns:
            ReliabilityMetric for consistency
        """
        try:
            evidence = []
            
            # Find other nodes from the same source
            same_source_nodes = self._find_same_source_nodes(source_identifier)
            
            if len(same_source_nodes) < 2:
                # Not enough data for consistency analysis
                return ReliabilityMetric(
                    metric_name="consistency",
                    score=0.6,  # Neutral score
                    confidence=0.3,
                    evidence=["Insufficient data for consistency analysis"],
                    weight=0.2
                )
            
            # Analyze consistency across nodes from same source
            consistency_scores = []
            
            # Content quality consistency
            quality_consistency = self._analyze_quality_consistency(same_source_nodes)
            consistency_scores.append(quality_consistency)
            evidence.append(f"Quality consistency: {quality_consistency:.2f}")
            
            # Temporal consistency
            temporal_consistency = self._analyze_temporal_consistency(same_source_nodes)
            consistency_scores.append(temporal_consistency)
            evidence.append(f"Temporal consistency: {temporal_consistency:.2f}")
            
            # Validation history
            validation_consistency = self._analyze_validation_history(same_source_nodes)
            consistency_scores.append(validation_consistency)
            evidence.append(f"Validation history: {validation_consistency:.2f}")
            
            # Overall consistency score
            overall_score = np.mean(consistency_scores)
            
            # Calculate confidence
            confidence = min(len(same_source_nodes) / 10, 1.0)  # More nodes = higher confidence
            
            return ReliabilityMetric(
                metric_name="consistency",
                score=overall_score,
                confidence=confidence,
                evidence=evidence,
                weight=0.2
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing consistency: {e}")
            return self._create_error_metric("consistency")
    
    def _find_same_source_nodes(self, source_identifier: str) -> List[KnowledgeNode]:
        """Find other nodes from the same source."""
        try:
            # This would normally query the database for nodes with matching source
            # For now, return empty list (would be implemented with actual graph queries)
            return []
        except Exception as e:
            self.logger.error(f"Error finding same source nodes: {e}")
            return []
    
    def _analyze_quality_consistency(self, nodes: List[KnowledgeNode]) -> float:
        """Analyze quality consistency across nodes from same source."""
        if len(nodes) < 2:
            return 0.6
        
        # This would analyze quality metrics across nodes
        # For now, return a placeholder score
        return 0.7
    
    def _analyze_temporal_consistency(self, nodes: List[KnowledgeNode]) -> float:
        """Analyze temporal consistency (regular updates, etc.)."""
        if len(nodes) < 2:
            return 0.6
        
        # This would analyze update patterns and temporal information
        # For now, return a placeholder score
        return 0.75
    
    def _analyze_validation_history(self, nodes: List[KnowledgeNode]) -> float:
        """Analyze validation history for the source."""
        if len(nodes) < 2:
            return 0.6
        
        # This would analyze cross-validation results for nodes from this source
        # For now, return a placeholder score
        return 0.8
    
    def _create_error_metric(self, metric_name: str) -> ReliabilityMetric:
        """Create error metric when analysis fails."""
        return ReliabilityMetric(
            metric_name=metric_name,
            score=0.3,
            confidence=0.1,
            evidence=[f"Error analyzing {metric_name}"],
            weight=0.0
        )


class FreshnessAnalyzer:
    """Analyzes source freshness and temporal relevance."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_freshness(self, metadata: Dict[str, Any], content: str) -> ReliabilityMetric:
        """
        Analyze source freshness.
        
        Args:
            metadata: Source metadata
            content: Source content
            
        Returns:
            ReliabilityMetric for freshness
        """
        try:
            evidence = []
            freshness_scores = []
            
            # Publication/creation date freshness
            pub_freshness = self._analyze_publication_freshness(metadata)
            if pub_freshness is not None:
                freshness_scores.append(pub_freshness)
                evidence.append(f"Publication freshness: {pub_freshness:.2f}")
            
            # Update freshness
            update_freshness = self._analyze_update_freshness(metadata)
            if update_freshness is not None:
                freshness_scores.append(update_freshness)
                evidence.append(f"Update freshness: {update_freshness:.2f}")
            
            # Content temporal relevance
            content_freshness = self._analyze_content_freshness(content)
            freshness_scores.append(content_freshness)
            evidence.append(f"Content temporal relevance: {content_freshness:.2f}")
            
            # Calculate overall freshness
            if freshness_scores:
                overall_score = np.mean(freshness_scores)
            else:
                overall_score = 0.5  # Default if no temporal data
            
            # Calculate confidence
            confidence = 0.8 if len(freshness_scores) >= 2 else 0.5
            
            return ReliabilityMetric(
                metric_name="freshness",
                score=overall_score,
                confidence=confidence,
                evidence=evidence,
                weight=0.15
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing freshness: {e}")
            return self._create_error_metric("freshness")
    
    def _analyze_publication_freshness(self, metadata: Dict[str, Any]) -> Optional[float]:
        """Analyze publication date freshness."""
        timestamp_fields = ['publication_date', 'created_at', 'date', 'timestamp']
        
        for field in timestamp_fields:
            if field in metadata:
                try:
                    if isinstance(metadata[field], str):
                        pub_date = datetime.fromisoformat(metadata[field].replace('Z', '+00:00'))
                    elif isinstance(metadata[field], (int, float)):
                        pub_date = datetime.fromtimestamp(metadata[field])
                    else:
                        continue
                    
                    # Calculate age in days
                    now = datetime.now()
                    if pub_date.tzinfo:
                        now = now.replace(tzinfo=pub_date.tzinfo)
                    
                    age_days = (now - pub_date).days
                    
                    # Score based on age
                    if age_days <= 30:
                        return 1.0  # Very fresh
                    elif age_days <= 90:
                        return 0.9  # Fresh
                    elif age_days <= 180:
                        return 0.7  # Moderately fresh
                    elif age_days <= 365:
                        return 0.5  # Somewhat dated
                    elif age_days <= 730:
                        return 0.3  # Old
                    else:
                        return 0.1  # Very old
                        
                except Exception:
                    continue
        
        return None  # No publication date found
    
    def _analyze_update_freshness(self, metadata: Dict[str, Any]) -> Optional[float]:
        """Analyze update/modification freshness."""
        update_fields = ['updated_at', 'modified_at', 'last_updated']
        
        for field in update_fields:
            if field in metadata:
                try:
                    if isinstance(metadata[field], str):
                        update_date = datetime.fromisoformat(metadata[field].replace('Z', '+00:00'))
                    elif isinstance(metadata[field], (int, float)):
                        update_date = datetime.fromtimestamp(metadata[field])
                    else:
                        continue
                    
                    # Calculate age since last update
                    now = datetime.now()
                    if update_date.tzinfo:
                        now = now.replace(tzinfo=update_date.tzinfo)
                    
                    age_days = (now - update_date).days
                    
                    # Score based on recency of updates
                    if age_days <= 7:
                        return 1.0  # Very recently updated
                    elif age_days <= 30:
                        return 0.9  # Recently updated
                    elif age_days <= 90:
                        return 0.7  # Moderately recent
                    elif age_days <= 180:
                        return 0.5  # Somewhat dated
                    else:
                        return 0.3  # Not recently updated
                        
                except Exception:
                    continue
        
        return None  # No update date found
    
    def _analyze_content_freshness(self, content: str) -> float:
        """Analyze content for temporal relevance indicators."""
        content_lower = content.lower()
        
        # Current/recent indicators
        recent_indicators = ['current', 'recent', 'latest', 'now', 'today', 'this year', '2024', '2023']
        recent_count = sum(1 for indicator in recent_indicators if indicator in content_lower)
        
        # Outdated indicators
        outdated_indicators = ['legacy', 'deprecated', 'obsolete', 'outdated', 'former', 'previous']
        outdated_count = sum(1 for indicator in outdated_indicators if indicator in content_lower)
        
        # Calculate freshness score
        freshness_score = 0.5  # Base score
        
        if recent_count > 0:
            freshness_score += min(recent_count * 0.1, 0.3)
        
        if outdated_count > 0:
            freshness_score -= min(outdated_count * 0.1, 0.3)
        
        return max(0.1, min(freshness_score, 1.0))
    
    def _create_error_metric(self, metric_name: str) -> ReliabilityMetric:
        """Create error metric when analysis fails."""
        return ReliabilityMetric(
            metric_name=metric_name,
            score=0.3,
            confidence=0.1,
            evidence=[f"Error analyzing {metric_name}"],
            weight=0.0
        )


class SourceReliabilityEngine:
    """
    Main Source Reliability Engine.
    
    Provides comprehensive source reliability scoring based on authority,
    consistency, freshness, and other quality factors.
    """
    
    def __init__(self, query_engine: AdvancedQueryEngine):
        """
        Initialize the Source Reliability Engine.
        
        Args:
            query_engine: Query engine for accessing knowledge graph
        """
        self.query_engine = query_engine
        
        # Initialize analyzers
        self.source_identifier = SourceIdentifier()
        self.authority_analyzer = AuthorityAnalyzer(query_engine)
        self.consistency_analyzer = ConsistencyAnalyzer(query_engine)
        self.freshness_analyzer = FreshnessAnalyzer()
        
        self.logger = logging.getLogger(__name__)
        
        # Cache for source scores
        self.source_cache = {}
        self.cache_expiry = timedelta(hours=24)  # Cache expires after 24 hours
        
        # Statistics
        self.stats = {
            'sources_assessed': 0,
            'avg_reliability_score': 0.0,
            'reliability_distribution': {level.value: 0 for level in ReliabilityLevel},
            'source_type_distribution': {stype.value: 0 for stype in SourceType},
            'cache_hit_rate': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def assess_source_reliability(self, node: KnowledgeNode) -> SourceReliabilityScore:
        """
        Assess the reliability of a source for a knowledge node.
        
        Args:
            node: Knowledge node to assess source reliability for
            
        Returns:
            SourceReliabilityScore with comprehensive assessment
        """
        try:
            # Extract source information
            source_info = self._extract_source_info(node)
            
            if not source_info:
                return self._create_unknown_source_score(node.node_id)
            
            # Identify and categorize source
            source_identifier, source_type = self.source_identifier.identify_source(
                source_info, node.metadata or {}
            )
            
            # Check cache first
            cached_score = self._check_cache(source_identifier)
            if cached_score:
                self.stats['cache_hits'] += 1
                return cached_score
            
            self.stats['cache_misses'] += 1
            
            self.logger.info(f"Assessing reliability for source: {source_identifier}")
            
            # Perform reliability analysis
            metrics = {}
            
            # Authority analysis
            metrics['authority'] = self.authority_analyzer.analyze_authority(
                source_identifier, source_type, node.content, node.metadata or {}
            )
            
            # Consistency analysis
            metrics['consistency'] = self.consistency_analyzer.analyze_consistency(
                source_identifier, node
            )
            
            # Freshness analysis
            metrics['freshness'] = self.freshness_analyzer.analyze_freshness(
                node.metadata or {}, node.content
            )
            
            # Calculate overall reliability score
            overall_score = self._calculate_overall_score(metrics)
            reliability_level = self._score_to_reliability_level(overall_score)
            
            # Calculate assessment confidence
            assessment_confidence = self._calculate_assessment_confidence(metrics)
            
            # Generate improvement suggestions
            improvement_suggestions = self._generate_improvement_suggestions(metrics, source_type)
            
            # Create reliability score
            reliability_score = SourceReliabilityScore(
                source_identifier=source_identifier,
                overall_score=overall_score,
                reliability_level=reliability_level,
                source_type=source_type,
                metrics=metrics,
                assessment_confidence=assessment_confidence,
                factors_considered=[metric.metric_name for metric in metrics.values()],
                improvement_suggestions=improvement_suggestions,
                last_assessed=datetime.now()
            )
            
            # Cache the result
            self._cache_score(source_identifier, reliability_score)
            
            # Update statistics
            self._update_statistics(reliability_score)
            
            self.logger.info(f"Source reliability assessed: {reliability_level.value} ({overall_score:.2f})")
            return reliability_score
            
        except Exception as e:
            self.logger.error(f"Error assessing source reliability: {e}")
            return self._create_error_score(node.node_id)
    
    def assess_multiple_sources(self, nodes: List[KnowledgeNode]) -> ReliabilityReport:
        """
        Assess reliability for multiple sources and generate a report.
        
        Args:
            nodes: List of knowledge nodes to assess
            
        Returns:
            ReliabilityReport with comprehensive results
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Assessing reliability for {len(nodes)} sources")
            
            source_scores = []
            
            # Assess each source
            for node in nodes:
                score = self.assess_source_reliability(node)
                source_scores.append(score)
            
            # Generate summary statistics
            summary_stats = self._calculate_summary_statistics(source_scores)
            
            # Calculate distributions
            reliability_distribution = {level: 0 for level in ReliabilityLevel}
            source_type_distribution = {stype: 0 for stype in SourceType}
            
            for score in source_scores:
                reliability_distribution[score.reliability_level] += 1
                source_type_distribution[score.source_type] += 1
            
            # Generate recommendations
            recommendations = self._generate_global_recommendations(source_scores)
            
            processing_time = (time.time() - start_time) * 1000
            
            report = ReliabilityReport(
                source_scores=source_scores,
                summary_statistics=summary_stats,
                reliability_distribution=reliability_distribution,
                source_type_distribution=source_type_distribution,
                recommendations=recommendations,
                assessment_time_ms=processing_time
            )
            
            self.logger.info(f"Reliability report generated for {len(nodes)} sources")
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating reliability report: {e}")
            return ReliabilityReport(
                source_scores=[],
                summary_statistics={},
                reliability_distribution={level: 0 for level in ReliabilityLevel},
                source_type_distribution={stype: 0 for stype in SourceType},
                recommendations=[],
                assessment_time_ms=(time.time() - start_time) * 1000
            )
    
    def _extract_source_info(self, node: KnowledgeNode) -> Optional[str]:
        """Extract source information from node metadata."""
        metadata = node.metadata or {}
        
        # Check common source fields
        source_fields = ['source', 'url', 'citation', 'reference', 'origin']
        
        for field in source_fields:
            if field in metadata and metadata[field]:
                return str(metadata[field])
        
        # Check for URL patterns in content
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, node.content)
        if urls:
            return urls[0]
        
        return None
    
    def _check_cache(self, source_identifier: str) -> Optional[SourceReliabilityScore]:
        """Check cache for existing reliability score."""
        if source_identifier in self.source_cache:
            cached_entry = self.source_cache[source_identifier]
            
            # Check if cache entry is still valid
            if datetime.now() - cached_entry['timestamp'] < self.cache_expiry:
                return cached_entry['score']
            else:
                # Remove expired entry
                del self.source_cache[source_identifier]
        
        return None
    
    def _cache_score(self, source_identifier: str, score: SourceReliabilityScore):
        """Cache reliability score."""
        self.source_cache[source_identifier] = {
            'score': score,
            'timestamp': datetime.now()
        }
        
        # Limit cache size
        if len(self.source_cache) > 1000:
            # Remove oldest entries
            oldest_key = min(self.source_cache.keys(), 
                           key=lambda k: self.source_cache[k]['timestamp'])
            del self.source_cache[oldest_key]
    
    def _calculate_overall_score(self, metrics: Dict[str, ReliabilityMetric]) -> float:
        """Calculate overall reliability score from individual metrics."""
        if not metrics:
            return 0.3  # Conservative default
        
        # Calculate weighted average
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for metric in metrics.values():
            weighted_score = metric.score * metric.weight * metric.confidence
            total_weighted_score += weighted_score
            total_weight += metric.weight * metric.confidence
        
        if total_weight == 0:
            return 0.3
        
        overall_score = total_weighted_score / total_weight
        return max(0.0, min(overall_score, 1.0))
    
    def _score_to_reliability_level(self, score: float) -> ReliabilityLevel:
        """Convert numeric score to reliability level."""
        if score >= 0.9:
            return ReliabilityLevel.HIGHLY_RELIABLE
        elif score >= 0.7:
            return ReliabilityLevel.RELIABLE
        elif score >= 0.5:
            return ReliabilityLevel.MODERATELY_RELIABLE
        elif score >= 0.3:
            return ReliabilityLevel.QUESTIONABLE
        else:
            return ReliabilityLevel.UNRELIABLE
    
    def _calculate_assessment_confidence(self, metrics: Dict[str, ReliabilityMetric]) -> float:
        """Calculate confidence in the overall assessment."""
        if not metrics:
            return 0.1
        
        # Average confidence across metrics
        confidences = [metric.confidence for metric in metrics.values()]
        avg_confidence = np.mean(confidences)
        
        # Boost confidence if we have multiple reliable metrics
        reliable_metrics = sum(1 for metric in metrics.values() if metric.confidence > 0.7)
        confidence_boost = min(reliable_metrics * 0.1, 0.2)
        
        return min(avg_confidence + confidence_boost, 1.0)
    
    def _generate_improvement_suggestions(self, metrics: Dict[str, ReliabilityMetric], 
                                        source_type: SourceType) -> List[str]:
        """Generate suggestions for improving source reliability."""
        suggestions = []
        
        # Analyze each metric for improvement opportunities
        for metric in metrics.values():
            if metric.score < 0.5:
                if metric.metric_name == 'authority':
                    suggestions.append("Seek sources with higher authority and credibility")
                elif metric.metric_name == 'consistency':
                    suggestions.append("Verify information consistency across multiple sources")
                elif metric.metric_name == 'freshness':
                    suggestions.append("Use more recent sources or verify current relevance")
        
        # Source type specific suggestions
        if source_type == SourceType.USER_GENERATED:
            suggestions.append("Consider supplementing with authoritative sources")
        elif source_type == SourceType.COMMERCIAL:
            suggestions.append("Verify information with independent sources")
        elif source_type == SourceType.UNKNOWN:
            suggestions.append("Identify and verify the source of information")
        
        return suggestions[:3]  # Limit to top 3 suggestions
    
    def _create_unknown_source_score(self, node_id: str) -> SourceReliabilityScore:
        """Create score for unknown source."""
        return SourceReliabilityScore(
            source_identifier=f"unknown_source_{node_id}",
            overall_score=0.3,
            reliability_level=ReliabilityLevel.QUESTIONABLE,
            source_type=SourceType.UNKNOWN,
            metrics={},
            assessment_confidence=0.2,
            factors_considered=[],
            improvement_suggestions=["Identify and document the source of information"],
            last_assessed=datetime.now()
        )
    
    def _create_error_score(self, node_id: str) -> SourceReliabilityScore:
        """Create error score when assessment fails."""
        return SourceReliabilityScore(
            source_identifier=f"error_source_{node_id}",
            overall_score=0.1,
            reliability_level=ReliabilityLevel.UNRELIABLE,
            source_type=SourceType.UNKNOWN,
            metrics={},
            assessment_confidence=0.1,
            factors_considered=[],
            improvement_suggestions=["Review source information and retry assessment"],
            last_assessed=datetime.now()
        )
    
    def _calculate_summary_statistics(self, scores: List[SourceReliabilityScore]) -> Dict[str, float]:
        """Calculate summary statistics for reliability scores."""
        if not scores:
            return {}
        
        overall_scores = [score.overall_score for score in scores]
        confidence_scores = [score.assessment_confidence for score in scores]
        
        return {
            'avg_reliability_score': np.mean(overall_scores),
            'median_reliability_score': np.median(overall_scores),
            'min_reliability_score': np.min(overall_scores),
            'max_reliability_score': np.max(overall_scores),
            'std_reliability_score': np.std(overall_scores),
            'avg_confidence': np.mean(confidence_scores),
            'total_sources_assessed': len(scores),
            'high_reliability_rate': sum(1 for s in scores if s.reliability_level in 
                                       [ReliabilityLevel.HIGHLY_RELIABLE, ReliabilityLevel.RELIABLE]) / len(scores)
        }
    
    def _generate_global_recommendations(self, scores: List[SourceReliabilityScore]) -> List[str]:
        """Generate global recommendations based on all assessed sources."""
        recommendations = []
        
        if not scores:
            return recommendations
        
        # Analyze reliability distribution
        unreliable_count = sum(1 for score in scores 
                             if score.reliability_level in [ReliabilityLevel.UNRELIABLE, ReliabilityLevel.QUESTIONABLE])
        
        if unreliable_count > len(scores) * 0.3:
            recommendations.append("Improve source quality - significant portion of sources are unreliable")
        
        # Analyze source type distribution
        source_types = [score.source_type for score in scores]
        user_generated_ratio = source_types.count(SourceType.USER_GENERATED) / len(source_types)
        
        if user_generated_ratio > 0.5:
            recommendations.append("Diversify source types - reduce reliance on user-generated content")
        
        # Analyze confidence patterns
        low_confidence_count = sum(1 for score in scores if score.assessment_confidence < 0.5)
        
        if low_confidence_count > len(scores) * 0.4:
            recommendations.append("Improve source metadata and documentation for better assessment")
        
        # Authority recommendations
        academic_ratio = source_types.count(SourceType.ACADEMIC) / len(source_types)
        
        if academic_ratio < 0.2:
            recommendations.append("Consider including more academic and authoritative sources")
        
        return recommendations[:5]  # Top 5 recommendations
    
    def _update_statistics(self, score: SourceReliabilityScore):
        """Update engine statistics."""
        self.stats['sources_assessed'] += 1
        
        # Update average reliability score
        total_score = self.stats['avg_reliability_score'] * (self.stats['sources_assessed'] - 1)
        self.stats['avg_reliability_score'] = (total_score + score.overall_score) / self.stats['sources_assessed']
        
        # Update distributions
        self.stats['reliability_distribution'][score.reliability_level.value] += 1
        self.stats['source_type_distribution'][score.source_type.value] += 1
        
        # Update cache hit rate
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        if total_requests > 0:
            self.stats['cache_hit_rate'] = self.stats['cache_hits'] / total_requests
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get source reliability engine statistics.
        
        Returns:
            Dictionary with comprehensive statistics
        """
        return {
            'source_reliability': self.stats.copy(),
            'cache_size': len(self.source_cache),
            'query_engine': self.query_engine.get_statistics()
        }