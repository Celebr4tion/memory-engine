"""
Cross-Validation Engine for Knowledge Quality Enhancement

Provides cross-validation capabilities between knowledge nodes to identify
inconsistencies, verify information accuracy, and improve overall knowledge
quality through comparative analysis.
"""

import logging
import re
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from collections import defaultdict, Counter

from memory_core.query.query_engine import AdvancedQueryEngine
from memory_core.query.query_types import QueryRequest, QueryType
from memory_core.model.knowledge_node import KnowledgeNode
from memory_core.model.relationship import Relationship


class ValidationStatus(Enum):
    """Status of cross-validation results."""

    VALIDATED = "validated"  # Information confirmed by multiple sources
    CONFLICTED = "conflicted"  # Contradictory information found
    INSUFFICIENT = "insufficient"  # Not enough information for validation
    PENDING = "pending"  # Validation in progress
    ERROR = "error"  # Validation failed due to error


class ValidationConfidence(Enum):
    """Confidence levels for validation results."""

    HIGH = "high"  # 85-100% confidence
    MEDIUM = "medium"  # 60-84% confidence
    LOW = "low"  # 30-59% confidence
    VERY_LOW = "very_low"  # 0-29% confidence


@dataclass
class ValidationEvidence:
    """Evidence supporting or contradicting a claim."""

    node_id: str
    content_snippet: str
    relevance_score: float
    reliability_score: float
    support_type: str  # 'supporting', 'contradicting', 'neutral'
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class CrossValidationResult:
    """Result of cross-validation analysis."""

    target_node_id: str
    validation_status: ValidationStatus
    confidence_level: ValidationConfidence
    consensus_score: float  # 0.0 to 1.0
    supporting_evidence: List[ValidationEvidence]
    contradicting_evidence: List[ValidationEvidence]
    neutral_evidence: List[ValidationEvidence]
    validation_summary: str
    key_conflicts: List[str]
    recommendations: List[str]
    validation_timestamp: datetime


@dataclass
class FactualClaim:
    """Extracted factual claim for validation."""

    claim_id: str
    claim_text: str
    claim_type: str  # 'factual', 'numerical', 'temporal', 'relational'
    confidence: float
    source_node_id: str
    extraction_method: str


@dataclass
class ValidationReport:
    """Comprehensive validation report."""

    validation_results: List[CrossValidationResult]
    overall_statistics: Dict[str, Any]
    quality_improvements: List[str]
    critical_conflicts: List[str]
    validation_coverage: float
    processing_time_ms: float


class FactualClaimExtractor:
    """Extracts factual claims from knowledge nodes for validation."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Patterns for different types of claims
        self.claim_patterns = {
            "factual": [
                r"([A-Z][^.!?]*(?:is|are|was|were|has|have)[^.!?]*)",
                r"([A-Z][^.!?]*(?:can|will|would|should|must)[^.!?]*)",
                r"([A-Z][^.!?]*(?:causes?|leads? to|results? in)[^.!?]*)",
            ],
            "numerical": [
                r"([^.!?]*\b\d+(?:\.\d+)?(?:%|kg|meters?|seconds?|minutes?|hours?|days?|years?|dollars?)\b[^.!?]*)",
                r"([^.!?]*\b(?:approximately|about|around|over|under)\s+\d+[^.!?]*)",
            ],
            "temporal": [
                r"([^.!?]*\b(?:in|during|since|before|after)\s+\d{4}[^.!?]*)",
                r"([^.!?]*\b(?:yesterday|today|tomorrow|recently|currently)[^.!?]*)",
            ],
            "relational": [
                r"([^.!?]*\b(?:related to|connected to|part of|belongs to)[^.!?]*)",
                r"([^.!?]*\b(?:similar to|different from|compared to)[^.!?]*)",
            ],
        }

        # Confidence indicators
        self.confidence_indicators = {
            "high": ["confirmed", "verified", "established", "proven", "documented"],
            "medium": ["likely", "probable", "suggests", "indicates", "appears"],
            "low": ["possibly", "might", "could", "seems", "unclear", "unconfirmed"],
        }

    def extract_claims(self, node: KnowledgeNode) -> List[FactualClaim]:
        """
        Extract factual claims from a knowledge node.

        Args:
            node: Knowledge node to extract claims from

        Returns:
            List of extracted factual claims
        """
        try:
            claims = []
            content = node.content

            # Extract claims for each type
            for claim_type, patterns in self.claim_patterns.items():
                type_claims = self._extract_claims_by_type(
                    content, claim_type, patterns, node.node_id
                )
                claims.extend(type_claims)

            # Remove duplicates and low-quality claims
            claims = self._deduplicate_claims(claims)
            claims = self._filter_quality_claims(claims)

            self.logger.debug(f"Extracted {len(claims)} claims from node {node.node_id}")
            return claims

        except Exception as e:
            self.logger.error(f"Error extracting claims from node {node.node_id}: {e}")
            return []

    def _extract_claims_by_type(
        self, content: str, claim_type: str, patterns: List[str], node_id: str
    ) -> List[FactualClaim]:
        """Extract claims of a specific type."""
        claims = []

        for pattern in patterns:
            import re

            matches = re.findall(pattern, content, re.IGNORECASE)

            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else (match[1] if len(match) > 1 else "")

                if len(match.strip()) > 10:  # Minimum claim length
                    confidence = self._calculate_claim_confidence(match, content)

                    claim = FactualClaim(
                        claim_id=f"{node_id}_{claim_type}_{hash(match)}",
                        claim_text=match.strip(),
                        claim_type=claim_type,
                        confidence=confidence,
                        source_node_id=node_id,
                        extraction_method="pattern_matching",
                    )
                    claims.append(claim)

        return claims

    def _calculate_claim_confidence(self, claim_text: str, full_content: str) -> float:
        """Calculate confidence in extracted claim."""
        claim_lower = claim_text.lower()
        confidence = 0.5  # Base confidence

        # Check for confidence indicators
        for level, indicators in self.confidence_indicators.items():
            for indicator in indicators:
                if indicator in claim_lower:
                    if level == "high":
                        confidence += 0.3
                    elif level == "medium":
                        confidence += 0.1
                    else:  # low
                        confidence -= 0.2
                    break

        # Boost confidence for specific, detailed claims
        if any(char.isdigit() for char in claim_text):
            confidence += 0.1  # Contains numbers

        if len(claim_text.split()) > 8:
            confidence += 0.1  # Detailed claim

        return max(0.1, min(confidence, 1.0))

    def _deduplicate_claims(self, claims: List[FactualClaim]) -> List[FactualClaim]:
        """Remove duplicate claims."""
        seen_claims = set()
        unique_claims = []

        for claim in claims:
            # Create a normalized version for comparison
            normalized = claim.claim_text.lower().strip()

            if normalized not in seen_claims and len(normalized) > 15:
                seen_claims.add(normalized)
                unique_claims.append(claim)

        return unique_claims

    def _filter_quality_claims(self, claims: List[FactualClaim]) -> List[FactualClaim]:
        """Filter out low-quality claims."""
        quality_claims = []

        for claim in claims:
            # Filter criteria
            if (
                claim.confidence >= 0.3
                and len(claim.claim_text) >= 15
                and len(claim.claim_text.split()) >= 3
            ):
                quality_claims.append(claim)

        return quality_claims


class EvidenceCollector:
    """Collects validation evidence for factual claims."""

    def __init__(self, query_engine: AdvancedQueryEngine):
        self.query_engine = query_engine
        self.logger = logging.getLogger(__name__)

    def collect_evidence(
        self, claim: FactualClaim, exclude_nodes: Set[str] = None
    ) -> List[ValidationEvidence]:
        """
        Collect evidence for or against a factual claim.

        Args:
            claim: Factual claim to validate
            exclude_nodes: Node IDs to exclude from search

        Returns:
            List of validation evidence
        """
        try:
            # Extract key terms from claim for search
            search_terms = self._extract_search_terms(claim.claim_text)

            # Search for relevant nodes
            relevant_nodes = self._search_relevant_nodes(search_terms, exclude_nodes)

            # Analyze each node for supporting/contradicting evidence
            evidence = []

            for node_data in relevant_nodes:
                node_evidence = self._analyze_node_evidence(claim, node_data)
                if node_evidence:
                    evidence.append(node_evidence)

            # Sort evidence by relevance and reliability
            evidence.sort(key=lambda e: e.relevance_score * e.reliability_score, reverse=True)

            return evidence[:20]  # Limit to top 20 pieces of evidence

        except Exception as e:
            self.logger.error(f"Error collecting evidence for claim {claim.claim_id}: {e}")
            return []

    def _extract_search_terms(self, claim_text: str) -> List[str]:
        """Extract key search terms from claim text."""
        import re

        # Remove common words and extract important terms
        stop_words = {
            "is",
            "are",
            "was",
            "were",
            "has",
            "have",
            "had",
            "can",
            "will",
            "would",
            "should",
            "must",
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "to",
            "of",
            "in",
            "on",
            "at",
            "by",
            "for",
            "with",
            "that",
            "this",
        }

        # Extract words, preserve important terms
        words = re.findall(r"\b[a-zA-Z]{3,}\b", claim_text.lower())

        # Filter stop words and get key terms
        key_terms = [word for word in words if word not in stop_words]

        # Also extract quoted phrases and proper nouns
        phrases = re.findall(r'"([^"]*)"', claim_text)
        proper_nouns = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", claim_text)

        search_terms = key_terms[:5] + phrases + proper_nouns  # Limit key terms
        return list(set(search_terms))  # Remove duplicates

    def _search_relevant_nodes(
        self, search_terms: List[str], exclude_nodes: Set[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for nodes relevant to the search terms."""
        try:
            # Create search query
            query = " ".join(search_terms[:3])  # Use top 3 terms

            request = QueryRequest(
                query=query,
                query_type=QueryType.SEMANTIC_SEARCH,
                limit=30,
                similarity_threshold=0.4,
                include_relationships=False,
            )

            response = self.query_engine.query(request)

            # Filter out excluded nodes
            relevant_nodes = []
            exclude_nodes = exclude_nodes or set()

            for result in response.results:
                if result.node_id not in exclude_nodes:
                    relevant_nodes.append(
                        {
                            "node_id": result.node_id,
                            "content": result.content,
                            "node_type": result.node_type,
                            "metadata": result.metadata or {},
                            "relevance_score": result.relevance_score or 0.5,
                        }
                    )

            return relevant_nodes

        except Exception as e:
            self.logger.error(f"Error searching relevant nodes: {e}")
            return []

    def _analyze_node_evidence(
        self, claim: FactualClaim, node_data: Dict[str, Any]
    ) -> Optional[ValidationEvidence]:
        """Analyze a node for evidence supporting or contradicting the claim."""
        try:
            content = node_data["content"]

            # Calculate relevance to claim
            relevance_score = self._calculate_relevance(claim.claim_text, content)

            if relevance_score < 0.3:  # Not relevant enough
                return None

            # Determine support type (supporting, contradicting, neutral)
            support_type, support_confidence = self._determine_support_type(
                claim.claim_text, content
            )

            # Calculate reliability score
            reliability_score = self._calculate_reliability(node_data)

            # Extract relevant content snippet
            content_snippet = self._extract_relevant_snippet(claim.claim_text, content)

            return ValidationEvidence(
                node_id=node_data["node_id"],
                content_snippet=content_snippet,
                relevance_score=relevance_score,
                reliability_score=reliability_score,
                support_type=support_type,
                confidence=support_confidence,
                metadata=node_data["metadata"],
            )

        except Exception as e:
            self.logger.error(f"Error analyzing node evidence: {e}")
            return None

    def _calculate_relevance(self, claim_text: str, content: str) -> float:
        """Calculate relevance between claim and content."""
        # Simple word overlap calculation
        claim_words = set(claim_text.lower().split())
        content_words = set(content.lower().split())

        if not claim_words or not content_words:
            return 0.0

        # Calculate Jaccard similarity
        intersection = len(claim_words & content_words)
        union = len(claim_words | content_words)

        jaccard_similarity = intersection / union if union > 0 else 0.0

        # Boost for exact phrase matches
        if claim_text.lower() in content.lower():
            jaccard_similarity += 0.3

        return min(jaccard_similarity, 1.0)

    def _determine_support_type(self, claim_text: str, content: str) -> Tuple[str, float]:
        """Determine if content supports, contradicts, or is neutral to claim."""
        claim_lower = claim_text.lower()
        content_lower = content.lower()

        # Look for contradiction indicators
        contradiction_indicators = [
            "however",
            "but",
            "although",
            "despite",
            "contrary to",
            "in contrast",
            "on the other hand",
            "nevertheless",
            "contradicts",
        ]

        # Look for support indicators
        support_indicators = [
            "confirms",
            "supports",
            "validates",
            "agrees",
            "consistent with",
            "furthermore",
            "moreover",
            "in addition",
            "similarly",
            "likewise",
        ]

        # Check for direct contradictions (negations)
        negation_patterns = [r"not\s+" + re.escape(word) for word in claim_lower.split()[:3]]

        contradiction_score = 0.0
        support_score = 0.0

        # Check for contradiction indicators
        for indicator in contradiction_indicators:
            if indicator in content_lower:
                contradiction_score += 0.2

        # Check for support indicators
        for indicator in support_indicators:
            if indicator in content_lower:
                support_score += 0.2

        # Check for negation patterns
        import re

        for pattern in negation_patterns:
            if re.search(pattern, content_lower):
                contradiction_score += 0.3

        # Check for similar statements (support)
        if self._calculate_relevance(claim_text, content) > 0.6:
            support_score += 0.3

        # Determine support type
        if contradiction_score > support_score and contradiction_score > 0.3:
            return "contradicting", contradiction_score
        elif support_score > contradiction_score and support_score > 0.3:
            return "supporting", support_score
        else:
            return "neutral", max(contradiction_score, support_score)

    def _calculate_reliability(self, node_data: Dict[str, Any]) -> float:
        """Calculate reliability score for a node."""
        metadata = node_data["metadata"]
        reliability = 0.5  # Base reliability

        # Source-based reliability
        if "source" in metadata:
            source = metadata["source"].lower()
            if any(indicator in source for indicator in ["journal", "research", "academic"]):
                reliability += 0.2
            elif any(indicator in source for indicator in ["wiki", "blog", "forum"]):
                reliability -= 0.1

        # Confidence-based reliability
        if "confidence" in metadata:
            try:
                confidence = float(metadata["confidence"])
                reliability += (confidence - 0.5) * 0.4  # Scale confidence contribution
            except:
                pass

        # Freshness-based reliability
        if "timestamp" in metadata:
            try:
                timestamp = datetime.fromisoformat(metadata["timestamp"].replace("Z", "+00:00"))
                age_days = (datetime.now().replace(tzinfo=timestamp.tzinfo) - timestamp).days

                if age_days <= 90:
                    reliability += 0.1  # Recent information
                elif age_days > 365:
                    reliability -= 0.1  # Old information
            except:
                pass

        return max(0.1, min(reliability, 1.0))

    def _extract_relevant_snippet(
        self, claim_text: str, content: str, max_length: int = 200
    ) -> str:
        """Extract the most relevant snippet from content."""
        # Split content into sentences
        import re

        sentences = re.split(r"[.!?]+", content)

        claim_words = set(claim_text.lower().split())

        # Find the sentence with the highest word overlap
        best_sentence = ""
        best_score = 0.0

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue

            sentence_words = set(sentence.lower().split())
            overlap = len(claim_words & sentence_words)

            if overlap > best_score:
                best_score = overlap
                best_sentence = sentence

        # Truncate if too long
        if len(best_sentence) > max_length:
            best_sentence = best_sentence[:max_length] + "..."

        return best_sentence if best_sentence else content[:max_length] + "..."


class ValidationAnalyzer:
    """Analyzes collected evidence to determine validation results."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze_validation(
        self, claim: FactualClaim, evidence: List[ValidationEvidence]
    ) -> CrossValidationResult:
        """
        Analyze validation evidence to determine validation result.

        Args:
            claim: Factual claim being validated
            evidence: Collected validation evidence

        Returns:
            CrossValidationResult with analysis
        """
        try:
            # Separate evidence by support type
            supporting = [e for e in evidence if e.support_type == "supporting"]
            contradicting = [e for e in evidence if e.support_type == "contradicting"]
            neutral = [e for e in evidence if e.support_type == "neutral"]

            # Calculate consensus score
            consensus_score = self._calculate_consensus_score(supporting, contradicting, neutral)

            # Determine validation status
            validation_status = self._determine_validation_status(consensus_score, evidence)

            # Determine confidence level
            confidence_level = self._determine_confidence_level(consensus_score, evidence)

            # Identify key conflicts
            key_conflicts = self._identify_key_conflicts(contradicting)

            # Generate validation summary
            validation_summary = self._generate_validation_summary(
                claim, supporting, contradicting, consensus_score
            )

            # Generate recommendations
            recommendations = self._generate_recommendations(
                validation_status, consensus_score, supporting, contradicting
            )

            return CrossValidationResult(
                target_node_id=claim.source_node_id,
                validation_status=validation_status,
                confidence_level=confidence_level,
                consensus_score=consensus_score,
                supporting_evidence=supporting,
                contradicting_evidence=contradicting,
                neutral_evidence=neutral,
                validation_summary=validation_summary,
                key_conflicts=key_conflicts,
                recommendations=recommendations,
                validation_timestamp=datetime.now(),
            )

        except Exception as e:
            self.logger.error(f"Error analyzing validation: {e}")
            return self._create_error_result(claim.source_node_id)

    def _calculate_consensus_score(
        self,
        supporting: List[ValidationEvidence],
        contradicting: List[ValidationEvidence],
        neutral: List[ValidationEvidence],
    ) -> float:
        """Calculate consensus score based on evidence."""
        if not supporting and not contradicting:
            return 0.5  # No clear evidence either way

        # Weight evidence by reliability and confidence
        supporting_weight = sum(e.reliability_score * e.confidence for e in supporting)
        contradicting_weight = sum(e.reliability_score * e.confidence for e in contradicting)

        total_weight = supporting_weight + contradicting_weight

        if total_weight == 0:
            return 0.5

        # Consensus score: 1.0 = full support, 0.0 = full contradiction, 0.5 = neutral
        consensus_score = supporting_weight / total_weight

        return consensus_score

    def _determine_validation_status(
        self, consensus_score: float, evidence: List[ValidationEvidence]
    ) -> ValidationStatus:
        """Determine validation status based on consensus score and evidence."""
        if not evidence:
            return ValidationStatus.INSUFFICIENT

        # Require minimum evidence quality
        high_quality_evidence = [e for e in evidence if e.reliability_score > 0.6]

        if len(high_quality_evidence) < 2:
            return ValidationStatus.INSUFFICIENT

        # Determine status based on consensus
        if consensus_score >= 0.75:
            return ValidationStatus.VALIDATED
        elif consensus_score <= 0.25:
            return ValidationStatus.CONFLICTED
        elif 0.4 <= consensus_score <= 0.6:
            return ValidationStatus.CONFLICTED  # Mixed evidence
        else:
            return ValidationStatus.INSUFFICIENT

    def _determine_confidence_level(
        self, consensus_score: float, evidence: List[ValidationEvidence]
    ) -> ValidationConfidence:
        """Determine confidence level based on evidence quality and consensus."""
        if not evidence:
            return ValidationConfidence.VERY_LOW

        # Calculate average evidence quality
        avg_reliability = np.mean([e.reliability_score for e in evidence])
        avg_confidence = np.mean([e.confidence for e in evidence])

        overall_quality = (avg_reliability + avg_confidence) / 2

        # Adjust for consensus strength
        consensus_strength = abs(consensus_score - 0.5) * 2  # 0.0 to 1.0

        confidence_score = (overall_quality * 0.6) + (consensus_strength * 0.4)

        if confidence_score >= 0.85:
            return ValidationConfidence.HIGH
        elif confidence_score >= 0.6:
            return ValidationConfidence.MEDIUM
        elif confidence_score >= 0.3:
            return ValidationConfidence.LOW
        else:
            return ValidationConfidence.VERY_LOW

    def _identify_key_conflicts(
        self, contradicting_evidence: List[ValidationEvidence]
    ) -> List[str]:
        """Identify key conflicts from contradicting evidence."""
        conflicts = []

        for evidence in contradicting_evidence[:3]:  # Top 3 conflicts
            conflict_desc = f"Contradiction in {evidence.node_id}: {evidence.content_snippet[:100]}"
            conflicts.append(conflict_desc)

        return conflicts

    def _generate_validation_summary(
        self,
        claim: FactualClaim,
        supporting: List[ValidationEvidence],
        contradicting: List[ValidationEvidence],
        consensus_score: float,
    ) -> str:
        """Generate a human-readable validation summary."""
        summary_parts = []

        # Claim description
        summary_parts.append(f"Validation of claim: {claim.claim_text[:100]}...")

        # Evidence summary
        if supporting and contradicting:
            summary_parts.append(
                f"Found {len(supporting)} supporting and {len(contradicting)} contradicting sources"
            )
        elif supporting:
            summary_parts.append(
                f"Found {len(supporting)} supporting sources with no contradictions"
            )
        elif contradicting:
            summary_parts.append(
                f"Found {len(contradicting)} contradicting sources with no support"
            )
        else:
            summary_parts.append("No clear supporting or contradicting evidence found")

        # Consensus description
        if consensus_score >= 0.75:
            summary_parts.append("Strong consensus supports the claim")
        elif consensus_score <= 0.25:
            summary_parts.append("Strong consensus contradicts the claim")
        else:
            summary_parts.append("Mixed or insufficient evidence for consensus")

        return ". ".join(summary_parts) + "."

    def _generate_recommendations(
        self,
        status: ValidationStatus,
        consensus_score: float,
        supporting: List[ValidationEvidence],
        contradicting: List[ValidationEvidence],
    ) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        if status == ValidationStatus.VALIDATED:
            if consensus_score < 0.9:
                recommendations.append(
                    "Consider adding more supporting evidence to strengthen validation"
                )

        elif status == ValidationStatus.CONFLICTED:
            recommendations.append("Investigate contradictory sources to resolve conflicts")
            recommendations.append("Consider seeking additional authoritative sources")

            if contradicting:
                recommendations.append(
                    "Review methodology and reliability of contradicting sources"
                )

        elif status == ValidationStatus.INSUFFICIENT:
            recommendations.append("Collect more evidence from reliable sources")
            recommendations.append("Expand search terms to find relevant information")

        # General recommendations
        if supporting and len(supporting) < 3:
            recommendations.append("Seek additional supporting sources for better validation")

        low_reliability_count = len(
            [e for e in supporting + contradicting if e.reliability_score < 0.5]
        )
        if low_reliability_count > 0:
            recommendations.append(
                "Improve source reliability by using more authoritative references"
            )

        return recommendations[:4]  # Limit to top 4 recommendations

    def _create_error_result(self, node_id: str) -> CrossValidationResult:
        """Create error result when validation fails."""
        return CrossValidationResult(
            target_node_id=node_id,
            validation_status=ValidationStatus.ERROR,
            confidence_level=ValidationConfidence.VERY_LOW,
            consensus_score=0.0,
            supporting_evidence=[],
            contradicting_evidence=[],
            neutral_evidence=[],
            validation_summary="Validation failed due to error",
            key_conflicts=[],
            recommendations=["Retry validation with different parameters"],
            validation_timestamp=datetime.now(),
        )


class CrossValidationEngine:
    """
    Main Cross-Validation Engine.

    Provides comprehensive cross-validation capabilities between knowledge nodes
    to identify inconsistencies, verify information accuracy, and improve
    overall knowledge quality.
    """

    def __init__(self, query_engine: AdvancedQueryEngine):
        """
        Initialize the Cross-Validation Engine.

        Args:
            query_engine: Query engine for accessing knowledge graph
        """
        self.query_engine = query_engine

        # Initialize components
        self.claim_extractor = FactualClaimExtractor()
        self.evidence_collector = EvidenceCollector(query_engine)
        self.validation_analyzer = ValidationAnalyzer()

        self.logger = logging.getLogger(__name__)

        # Statistics
        self.stats = {
            "validations_performed": 0,
            "claims_validated": 0,
            "conflicts_detected": 0,
            "avg_consensus_score": 0.0,
            "avg_processing_time_ms": 0.0,
            "validation_status_distribution": {status.value: 0 for status in ValidationStatus},
        }

    def validate_node(
        self, node: KnowledgeNode, validation_scope: str = "comprehensive"
    ) -> List[CrossValidationResult]:
        """
        Perform cross-validation for a knowledge node.

        Args:
            node: Knowledge node to validate
            validation_scope: 'comprehensive', 'factual', 'numerical', 'temporal'

        Returns:
            List of cross-validation results for each claim
        """
        start_time = time.time()

        try:
            self.logger.info(f"Starting cross-validation for node {node.node_id}")

            # Extract factual claims from the node
            claims = self.claim_extractor.extract_claims(node)

            # Filter claims based on validation scope
            if validation_scope != "comprehensive":
                claims = [c for c in claims if c.claim_type == validation_scope]

            if not claims:
                self.logger.warning(f"No claims found for validation in node {node.node_id}")
                return []

            # Validate each claim
            validation_results = []
            exclude_nodes = {node.node_id}  # Exclude the source node from evidence

            for claim in claims:
                self.logger.debug(f"Validating claim: {claim.claim_text[:50]}...")

                # Collect evidence for the claim
                evidence = self.evidence_collector.collect_evidence(claim, exclude_nodes)

                # Analyze validation
                validation_result = self.validation_analyzer.analyze_validation(claim, evidence)
                validation_results.append(validation_result)

                # Update statistics
                self._update_claim_statistics(validation_result)

            # Update overall statistics
            processing_time = (time.time() - start_time) * 1000
            self._update_statistics(validation_results, processing_time)

            self.logger.info(
                f"Cross-validation completed for {node.node_id}: "
                f"{len(validation_results)} claims validated"
            )

            return validation_results

        except Exception as e:
            self.logger.error(f"Error validating node {node.node_id}: {e}")
            return []

    def validate_multiple_nodes(
        self, nodes: List[KnowledgeNode], validation_scope: str = "comprehensive"
    ) -> ValidationReport:
        """
        Perform cross-validation for multiple nodes and generate a report.

        Args:
            nodes: List of knowledge nodes to validate
            validation_scope: Scope of validation

        Returns:
            ValidationReport with comprehensive results
        """
        start_time = time.time()

        try:
            self.logger.info(f"Starting batch validation for {len(nodes)} nodes")

            all_results = []

            # Validate each node
            for node in nodes:
                node_results = self.validate_node(node, validation_scope)
                all_results.extend(node_results)

            # Generate overall statistics
            overall_stats = self._calculate_overall_statistics(all_results)

            # Identify critical conflicts
            critical_conflicts = self._identify_critical_conflicts(all_results)

            # Generate quality improvements
            quality_improvements = self._generate_quality_improvements(all_results)

            # Calculate validation coverage
            validation_coverage = self._calculate_validation_coverage(nodes, all_results)

            processing_time = (time.time() - start_time) * 1000

            report = ValidationReport(
                validation_results=all_results,
                overall_statistics=overall_stats,
                quality_improvements=quality_improvements,
                critical_conflicts=critical_conflicts,
                validation_coverage=validation_coverage,
                processing_time_ms=processing_time,
            )

            self.logger.info(f"Batch validation completed: {len(all_results)} total validations")
            return report

        except Exception as e:
            self.logger.error(f"Error in batch validation: {e}")
            return ValidationReport(
                validation_results=[],
                overall_statistics={},
                quality_improvements=[],
                critical_conflicts=[],
                validation_coverage=0.0,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

    def detect_contradictions(self, nodes: List[KnowledgeNode]) -> List[CrossValidationResult]:
        """
        Detect contradictions between multiple knowledge nodes.

        Args:
            nodes: List of knowledge nodes to analyze for contradictions

        Returns:
            List of validation results showing contradictions
        """
        try:
            self.logger.info(f"Detecting contradictions among {len(nodes)} nodes")

            contradictions = []

            # Extract claims from all nodes
            all_claims = []
            for node in nodes:
                node_claims = self.claim_extractor.extract_claims(node)
                all_claims.extend(node_claims)

            # Group similar claims for comparison
            claim_groups = self._group_similar_claims(all_claims)

            # Analyze each group for contradictions
            for group in claim_groups:
                if len(group) >= 2:  # Need at least 2 claims to compare
                    group_contradictions = self._analyze_claim_group_contradictions(group)
                    contradictions.extend(group_contradictions)

            self.logger.info(f"Found {len(contradictions)} potential contradictions")
            return contradictions

        except Exception as e:
            self.logger.error(f"Error detecting contradictions: {e}")
            return []

    def _group_similar_claims(self, claims: List[FactualClaim]) -> List[List[FactualClaim]]:
        """Group similar claims together for comparison."""
        groups = []
        processed = set()

        for i, claim1 in enumerate(claims):
            if i in processed:
                continue

            group = [claim1]
            processed.add(i)

            for j, claim2 in enumerate(claims[i + 1 :], i + 1):
                if j in processed:
                    continue

                # Calculate similarity between claims
                similarity = self._calculate_claim_similarity(claim1.claim_text, claim2.claim_text)

                if similarity > 0.6:  # Similarity threshold
                    group.append(claim2)
                    processed.add(j)

            if len(group) >= 2:  # Only keep groups with multiple claims
                groups.append(group)

        return groups

    def _calculate_claim_similarity(self, claim1: str, claim2: str) -> float:
        """Calculate similarity between two claims."""
        # Simple word-based similarity
        words1 = set(claim1.lower().split())
        words2 = set(claim2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _analyze_claim_group_contradictions(
        self, claims: List[FactualClaim]
    ) -> List[CrossValidationResult]:
        """Analyze a group of similar claims for contradictions."""
        contradictions = []

        # Compare each pair of claims in the group
        for i, claim1 in enumerate(claims):
            for claim2 in claims[i + 1 :]:
                # Check if claims contradict each other
                if self._are_claims_contradictory(claim1, claim2):
                    # Create validation result showing contradiction
                    contradiction_result = CrossValidationResult(
                        target_node_id=claim1.source_node_id,
                        validation_status=ValidationStatus.CONFLICTED,
                        confidence_level=ValidationConfidence.MEDIUM,
                        consensus_score=0.0,  # Full contradiction
                        supporting_evidence=[],
                        contradicting_evidence=[
                            ValidationEvidence(
                                node_id=claim2.source_node_id,
                                content_snippet=claim2.claim_text,
                                relevance_score=0.9,
                                reliability_score=0.7,
                                support_type="contradicting",
                                confidence=0.8,
                                metadata={"contradiction_type": "direct"},
                            )
                        ],
                        neutral_evidence=[],
                        validation_summary=f"Direct contradiction found between claims from {claim1.source_node_id} and {claim2.source_node_id}",
                        key_conflicts=[
                            f"Contradictory claims: '{claim1.claim_text}' vs '{claim2.claim_text}'"
                        ],
                        recommendations=[
                            "Investigate source reliability",
                            "Seek additional authoritative sources",
                        ],
                        validation_timestamp=datetime.now(),
                    )
                    contradictions.append(contradiction_result)

        return contradictions

    def _are_claims_contradictory(self, claim1: FactualClaim, claim2: FactualClaim) -> bool:
        """Check if two claims are contradictory."""
        text1 = claim1.claim_text.lower()
        text2 = claim2.claim_text.lower()

        # Look for direct negations
        contradiction_patterns = [
            ("is", "is not"),
            ("are", "are not"),
            ("can", "cannot"),
            ("will", "will not"),
            ("true", "false"),
            ("yes", "no"),
            ("increases", "decreases"),
            ("improves", "worsens"),
        ]

        for positive, negative in contradiction_patterns:
            if positive in text1 and negative in text2:
                return True
            if negative in text1 and positive in text2:
                return True

        # Look for contradictory numerical claims
        if claim1.claim_type == "numerical" and claim2.claim_type == "numerical":
            return self._check_numerical_contradiction(text1, text2)

        return False

    def _check_numerical_contradiction(self, text1: str, text2: str) -> bool:
        """Check for contradictory numerical claims."""
        import re

        # Extract numbers from both texts
        numbers1 = re.findall(r"\d+(?:\.\d+)?", text1)
        numbers2 = re.findall(r"\d+(?:\.\d+)?", text2)

        if numbers1 and numbers2:
            try:
                # Simple check for significantly different numbers
                num1 = float(numbers1[0])
                num2 = float(numbers2[0])

                # Consider contradiction if numbers differ by more than 50%
                if max(num1, num2) > 0:
                    difference_ratio = abs(num1 - num2) / max(num1, num2)
                    return difference_ratio > 0.5
            except:
                pass

        return False

    def _calculate_overall_statistics(self, results: List[CrossValidationResult]) -> Dict[str, Any]:
        """Calculate overall statistics for validation results."""
        if not results:
            return {}

        # Status distribution
        status_counts = {status.value: 0 for status in ValidationStatus}
        for result in results:
            status_counts[result.validation_status.value] += 1

        # Consensus scores
        consensus_scores = [r.consensus_score for r in results]

        # Confidence levels
        confidence_counts = {level.value: 0 for level in ValidationConfidence}
        for result in results:
            confidence_counts[result.confidence_level.value] += 1

        return {
            "total_validations": len(results),
            "avg_consensus_score": np.mean(consensus_scores),
            "median_consensus_score": np.median(consensus_scores),
            "status_distribution": status_counts,
            "confidence_distribution": confidence_counts,
            "validation_rate": status_counts["validated"] / len(results),
            "conflict_rate": status_counts["conflicted"] / len(results),
        }

    def _identify_critical_conflicts(self, results: List[CrossValidationResult]) -> List[str]:
        """Identify critical conflicts requiring immediate attention."""
        critical_conflicts = []

        for result in results:
            if (
                result.validation_status == ValidationStatus.CONFLICTED
                and result.confidence_level
                in [ValidationConfidence.MEDIUM, ValidationConfidence.HIGH]
            ):

                conflict_desc = f"High-confidence conflict in {result.target_node_id}"
                if result.key_conflicts:
                    conflict_desc += f": {result.key_conflicts[0]}"

                critical_conflicts.append(conflict_desc)

        return critical_conflicts[:10]  # Top 10 critical conflicts

    def _generate_quality_improvements(self, results: List[CrossValidationResult]) -> List[str]:
        """Generate quality improvement suggestions based on validation results."""
        improvements = []

        # Analyze common issues
        insufficient_count = sum(
            1 for r in results if r.validation_status == ValidationStatus.INSUFFICIENT
        )
        conflict_count = sum(
            1 for r in results if r.validation_status == ValidationStatus.CONFLICTED
        )

        if insufficient_count > len(results) * 0.3:
            improvements.append("Increase content detail and add more supporting information")

        if conflict_count > len(results) * 0.2:
            improvements.append("Implement systematic conflict resolution process")

        # Analyze confidence patterns
        low_confidence_count = sum(
            1 for r in results if r.confidence_level == ValidationConfidence.LOW
        )

        if low_confidence_count > len(results) * 0.4:
            improvements.append("Improve source reliability and metadata completeness")

        return improvements[:5]  # Top 5 improvements

    def _calculate_validation_coverage(
        self, nodes: List[KnowledgeNode], results: List[CrossValidationResult]
    ) -> float:
        """Calculate what percentage of nodes were successfully validated."""
        if not nodes:
            return 0.0

        validated_nodes = set(result.target_node_id for result in results)
        return len(validated_nodes) / len(nodes)

    def _update_claim_statistics(self, result: CrossValidationResult):
        """Update statistics for individual claim validation."""
        self.stats["claims_validated"] += 1
        self.stats["validation_status_distribution"][result.validation_status.value] += 1

        if result.validation_status == ValidationStatus.CONFLICTED:
            self.stats["conflicts_detected"] += 1

    def _update_statistics(self, results: List[CrossValidationResult], processing_time: float):
        """Update engine statistics."""
        self.stats["validations_performed"] += 1

        # Update average processing time
        total_time = self.stats["avg_processing_time_ms"] * (
            self.stats["validations_performed"] - 1
        )
        self.stats["avg_processing_time_ms"] = (total_time + processing_time) / self.stats[
            "validations_performed"
        ]

        # Update average consensus score
        if results:
            consensus_scores = [r.consensus_score for r in results]
            avg_consensus = np.mean(consensus_scores)

            total_consensus = self.stats["avg_consensus_score"] * (
                self.stats["validations_performed"] - 1
            )
            self.stats["avg_consensus_score"] = (total_consensus + avg_consensus) / self.stats[
                "validations_performed"
            ]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get cross-validation engine statistics.

        Returns:
            Dictionary with comprehensive statistics
        """
        return {
            "cross_validation": self.stats.copy(),
            "query_engine": self.query_engine.get_statistics(),
        }
