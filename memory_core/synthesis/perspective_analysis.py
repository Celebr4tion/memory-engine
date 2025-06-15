"""
Perspective-Based Analysis Engine for Knowledge Synthesis

Provides capabilities for generating different viewpoints on topics,
comparing conflicting information, and identifying consensus and disagreements
across multiple knowledge sources.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from collections import defaultdict, Counter

from memory_core.query.query_engine import AdvancedQueryEngine
from memory_core.query.query_types import QueryRequest, QueryType
from memory_core.synthesis.question_answering import QuestionAnsweringSystem, QuestionContext


class PerspectiveType(Enum):
    """Types of perspectives that can be analyzed."""

    OPPOSING = "opposing"  # Conflicting viewpoints
    COMPLEMENTARY = "complementary"  # Supporting different aspects
    ALTERNATIVE = "alternative"  # Different approaches to same topic
    TEMPORAL = "temporal"  # Evolution of perspectives over time
    STAKEHOLDER = "stakeholder"  # Different stakeholder viewpoints
    METHODOLOGICAL = "methodological"  # Different methodological approaches


class ConsensusLevel(Enum):
    """Levels of consensus across perspectives."""

    STRONG_CONSENSUS = "strong_consensus"  # >80% agreement
    MODERATE_CONSENSUS = "moderate_consensus"  # 60-80% agreement
    WEAK_CONSENSUS = "weak_consensus"  # 40-60% agreement
    NO_CONSENSUS = "no_consensus"  # <40% agreement
    STRONG_DISAGREEMENT = "strong_disagreement"  # Active opposition


@dataclass
class Perspective:
    """Individual perspective on a topic."""

    perspective_id: str
    topic: str
    viewpoint: str
    supporting_evidence: List[str]  # Node IDs
    confidence_score: float
    source_count: int
    perspective_type: PerspectiveType
    metadata: Dict[str, Any]
    key_claims: List[str]
    evidence_quality: float


@dataclass
class PerspectiveComparison:
    """Comparison between multiple perspectives."""

    topic: str
    perspectives: List[Perspective]
    consensus_areas: List[str]
    disagreement_areas: List[str]
    consensus_level: ConsensusLevel
    comparison_matrix: Dict[str, Dict[str, float]]  # Similarity between perspectives
    synthesis_summary: str
    confidence_in_analysis: float


@dataclass
class StakeholderPerspective:
    """Perspective from a specific stakeholder group."""

    stakeholder_group: str
    perspective: Perspective
    interests: List[str]
    concerns: List[str]
    priorities: List[str]
    influence_level: float


@dataclass
class TemporalPerspectiveEvolution:
    """Evolution of perspectives over time."""

    topic: str
    time_periods: List[Tuple[datetime, datetime]]
    perspective_changes: List[Dict[str, Any]]
    evolution_trend: str  # 'converging', 'diverging', 'stable', 'cyclical'
    key_turning_points: List[Dict[str, Any]]
    current_trajectory: str


@dataclass
class PerspectiveAnalysisReport:
    """Comprehensive perspective analysis report."""

    topic: str
    perspectives: List[Perspective]
    comparisons: List[PerspectiveComparison]
    stakeholder_analysis: List[StakeholderPerspective]
    temporal_evolution: Optional[TemporalPerspectiveEvolution]
    overall_consensus: ConsensusLevel
    key_insights: List[str]
    recommendations: List[str]
    analysis_confidence: float
    processing_time_ms: float


class PerspectiveExtractor:
    """Extracts different perspectives on topics from knowledge sources."""

    def __init__(self, query_engine: AdvancedQueryEngine):
        self.query_engine = query_engine
        self.logger = logging.getLogger(__name__)

        # Perspective indicator keywords
        self.perspective_indicators = {
            PerspectiveType.OPPOSING: [
                "however",
                "but",
                "on the contrary",
                "conversely",
                "disagreed",
                "opposed",
                "contradicts",
                "refutes",
                "challenges",
                "disputes",
            ],
            PerspectiveType.COMPLEMENTARY: [
                "additionally",
                "furthermore",
                "moreover",
                "also",
                "supports",
                "confirms",
                "validates",
                "complements",
                "builds upon",
            ],
            PerspectiveType.ALTERNATIVE: [
                "alternatively",
                "another approach",
                "different method",
                "another way",
                "alternatively",
                "instead",
                "rather than",
            ],
            PerspectiveType.STAKEHOLDER: [
                "experts believe",
                "users think",
                "researchers suggest",
                "practitioners argue",
                "stakeholders view",
                "community feels",
            ],
            PerspectiveType.METHODOLOGICAL: [
                "methodology",
                "approach",
                "technique",
                "method",
                "framework",
                "model",
                "strategy",
                "paradigm",
            ],
        }

    def extract_perspectives(
        self,
        topic: str,
        perspective_types: List[PerspectiveType] = None,
        min_evidence_count: int = 2,
    ) -> List[Perspective]:
        """
        Extract different perspectives on a topic.

        Args:
            topic: Topic to analyze perspectives for
            perspective_types: Types of perspectives to extract
            min_evidence_count: Minimum evidence sources required

        Returns:
            List of extracted perspectives
        """
        if perspective_types is None:
            perspective_types = list(PerspectiveType)

        perspectives = []

        try:
            # Get comprehensive information about the topic
            topic_data = self._gather_topic_information(topic)

            if not topic_data:
                self.logger.warning(f"No information found for topic: {topic}")
                return perspectives

            # Extract perspectives for each type
            for perspective_type in perspective_types:
                type_perspectives = self._extract_type_specific_perspectives(
                    topic, topic_data, perspective_type, min_evidence_count
                )
                perspectives.extend(type_perspectives)

            # Remove duplicate perspectives
            perspectives = self._deduplicate_perspectives(perspectives)

            # Score and rank perspectives
            perspectives = self._score_perspectives(perspectives)

            self.logger.info(f"Extracted {len(perspectives)} perspectives for topic: {topic}")
            return perspectives

        except Exception as e:
            self.logger.error(f"Error extracting perspectives for {topic}: {e}")
            return []

    def _gather_topic_information(self, topic: str) -> List[Dict[str, Any]]:
        """Gather comprehensive information about a topic."""
        try:
            # Semantic search for topic
            request = QueryRequest(
                query=topic,
                query_type=QueryType.SEMANTIC_SEARCH,
                limit=50,
                include_relationships=True,
                similarity_threshold=0.6,
            )

            response = self.query_engine.query(request)

            topic_data = []
            for result in response.results:
                topic_data.append(
                    {
                        "node_id": result.node_id,
                        "content": result.content,
                        "node_type": result.node_type,
                        "metadata": result.metadata or {},
                        "relationships": result.relationships or [],
                        "relevance_score": result.relevance_score or 0.0,
                    }
                )

            return topic_data

        except Exception as e:
            self.logger.error(f"Error gathering topic information: {e}")
            return []

    def _extract_type_specific_perspectives(
        self,
        topic: str,
        topic_data: List[Dict[str, Any]],
        perspective_type: PerspectiveType,
        min_evidence_count: int,
    ) -> List[Perspective]:
        """Extract perspectives for a specific type."""
        perspectives = []

        try:
            if perspective_type == PerspectiveType.OPPOSING:
                perspectives.extend(
                    self._extract_opposing_perspectives(topic, topic_data, min_evidence_count)
                )
            elif perspective_type == PerspectiveType.COMPLEMENTARY:
                perspectives.extend(
                    self._extract_complementary_perspectives(topic, topic_data, min_evidence_count)
                )
            elif perspective_type == PerspectiveType.ALTERNATIVE:
                perspectives.extend(
                    self._extract_alternative_perspectives(topic, topic_data, min_evidence_count)
                )
            elif perspective_type == PerspectiveType.STAKEHOLDER:
                perspectives.extend(
                    self._extract_stakeholder_perspectives(topic, topic_data, min_evidence_count)
                )
            elif perspective_type == PerspectiveType.METHODOLOGICAL:
                perspectives.extend(
                    self._extract_methodological_perspectives(topic, topic_data, min_evidence_count)
                )
            elif perspective_type == PerspectiveType.TEMPORAL:
                perspectives.extend(
                    self._extract_temporal_perspectives(topic, topic_data, min_evidence_count)
                )

        except Exception as e:
            self.logger.error(f"Error extracting {perspective_type.value} perspectives: {e}")

        return perspectives

    def _extract_opposing_perspectives(
        self, topic: str, topic_data: List[Dict[str, Any]], min_evidence_count: int
    ) -> List[Perspective]:
        """Extract opposing viewpoints on a topic."""
        perspectives = []

        # Group content by opposing indicators
        opposing_groups = defaultdict(list)

        for data in topic_data:
            content = data["content"].lower()

            # Check for opposing language
            for indicator in self.perspective_indicators[PerspectiveType.OPPOSING]:
                if indicator in content:
                    # Extract the opposing viewpoint
                    viewpoint = self._extract_viewpoint_around_indicator(data["content"], indicator)

                    if viewpoint:
                        key = self._generate_perspective_key(viewpoint)
                        opposing_groups[key].append(
                            {"viewpoint": viewpoint, "evidence": data, "indicator": indicator}
                        )
                    break

        # Create perspectives from groups
        for group_key, group_items in opposing_groups.items():
            if len(group_items) >= min_evidence_count:
                # Combine viewpoints and evidence
                combined_viewpoint = self._combine_viewpoints(
                    [item["viewpoint"] for item in group_items]
                )

                evidence_ids = [item["evidence"]["node_id"] for item in group_items]
                key_claims = self._extract_key_claims(combined_viewpoint)

                perspective = Perspective(
                    perspective_id=f"opposing_{hash(group_key)}",
                    topic=topic,
                    viewpoint=combined_viewpoint,
                    supporting_evidence=evidence_ids,
                    confidence_score=self._calculate_perspective_confidence(group_items),
                    source_count=len(group_items),
                    perspective_type=PerspectiveType.OPPOSING,
                    metadata={
                        "indicators_found": list(set(item["indicator"] for item in group_items)),
                        "group_key": group_key,
                    },
                    key_claims=key_claims,
                    evidence_quality=self._assess_evidence_quality(group_items),
                )
                perspectives.append(perspective)

        return perspectives

    def _extract_complementary_perspectives(
        self, topic: str, topic_data: List[Dict[str, Any]], min_evidence_count: int
    ) -> List[Perspective]:
        """Extract complementary viewpoints that support each other."""
        perspectives = []

        # Group by complementary themes
        complementary_groups = defaultdict(list)

        for data in topic_data:
            content = data["content"].lower()

            # Check for complementary language
            for indicator in self.perspective_indicators[PerspectiveType.COMPLEMENTARY]:
                if indicator in content:
                    viewpoint = self._extract_viewpoint_around_indicator(data["content"], indicator)

                    if viewpoint:
                        # Group by semantic similarity
                        group_key = self._find_semantic_group(
                            viewpoint, complementary_groups.keys()
                        )
                        if not group_key:
                            group_key = self._generate_perspective_key(viewpoint)

                        complementary_groups[group_key].append(
                            {"viewpoint": viewpoint, "evidence": data, "indicator": indicator}
                        )
                    break

        # Create perspectives
        for group_key, group_items in complementary_groups.items():
            if len(group_items) >= min_evidence_count:
                combined_viewpoint = self._combine_viewpoints(
                    [item["viewpoint"] for item in group_items]
                )

                evidence_ids = [item["evidence"]["node_id"] for item in group_items]
                key_claims = self._extract_key_claims(combined_viewpoint)

                perspective = Perspective(
                    perspective_id=f"complementary_{hash(group_key)}",
                    topic=topic,
                    viewpoint=combined_viewpoint,
                    supporting_evidence=evidence_ids,
                    confidence_score=self._calculate_perspective_confidence(group_items),
                    source_count=len(group_items),
                    perspective_type=PerspectiveType.COMPLEMENTARY,
                    metadata={
                        "indicators_found": list(set(item["indicator"] for item in group_items))
                    },
                    key_claims=key_claims,
                    evidence_quality=self._assess_evidence_quality(group_items),
                )
                perspectives.append(perspective)

        return perspectives

    def _extract_alternative_perspectives(
        self, topic: str, topic_data: List[Dict[str, Any]], min_evidence_count: int
    ) -> List[Perspective]:
        """Extract alternative approaches or solutions."""
        perspectives = []

        # Look for alternative approaches
        alternative_groups = defaultdict(list)

        for data in topic_data:
            content = data["content"]
            content_lower = content.lower()

            # Check for alternative language
            for indicator in self.perspective_indicators[PerspectiveType.ALTERNATIVE]:
                if indicator in content_lower:
                    viewpoint = self._extract_viewpoint_around_indicator(content, indicator)

                    if viewpoint:
                        # Extract the specific alternative being proposed
                        alternative_approach = self._extract_alternative_approach(viewpoint)

                        if alternative_approach:
                            group_key = self._generate_perspective_key(alternative_approach)
                            alternative_groups[group_key].append(
                                {
                                    "viewpoint": viewpoint,
                                    "approach": alternative_approach,
                                    "evidence": data,
                                    "indicator": indicator,
                                }
                            )
                    break

        # Create perspectives
        for group_key, group_items in alternative_groups.items():
            if len(group_items) >= min_evidence_count:
                combined_viewpoint = self._combine_viewpoints(
                    [item["viewpoint"] for item in group_items]
                )

                evidence_ids = [item["evidence"]["node_id"] for item in group_items]
                key_claims = self._extract_key_claims(combined_viewpoint)

                perspective = Perspective(
                    perspective_id=f"alternative_{hash(group_key)}",
                    topic=topic,
                    viewpoint=combined_viewpoint,
                    supporting_evidence=evidence_ids,
                    confidence_score=self._calculate_perspective_confidence(group_items),
                    source_count=len(group_items),
                    perspective_type=PerspectiveType.ALTERNATIVE,
                    metadata={"approaches": list(set(item["approach"] for item in group_items))},
                    key_claims=key_claims,
                    evidence_quality=self._assess_evidence_quality(group_items),
                )
                perspectives.append(perspective)

        return perspectives

    def _extract_stakeholder_perspectives(
        self, topic: str, topic_data: List[Dict[str, Any]], min_evidence_count: int
    ) -> List[Perspective]:
        """Extract perspectives from different stakeholder groups."""
        perspectives = []

        # Define stakeholder patterns
        stakeholder_patterns = {
            "researchers": ["research", "study", "findings", "analysis", "investigation"],
            "practitioners": ["practice", "experience", "implementation", "application"],
            "users": ["user", "customer", "client", "end-user", "consumer"],
            "experts": ["expert", "specialist", "authority", "professional"],
            "organizations": ["company", "organization", "institution", "agency"],
        }

        stakeholder_groups = defaultdict(list)

        for data in topic_data:
            content = data["content"].lower()

            # Identify stakeholder group
            identified_stakeholder = None
            for stakeholder, patterns in stakeholder_patterns.items():
                if any(pattern in content for pattern in patterns):
                    identified_stakeholder = stakeholder
                    break

            if identified_stakeholder:
                # Check for perspective indicators
                for indicator in self.perspective_indicators[PerspectiveType.STAKEHOLDER]:
                    if indicator in content:
                        viewpoint = self._extract_viewpoint_around_indicator(
                            data["content"], indicator
                        )

                        if viewpoint:
                            stakeholder_groups[identified_stakeholder].append(
                                {"viewpoint": viewpoint, "evidence": data, "indicator": indicator}
                            )
                        break

        # Create perspectives for each stakeholder group
        for stakeholder, group_items in stakeholder_groups.items():
            if len(group_items) >= min_evidence_count:
                combined_viewpoint = self._combine_viewpoints(
                    [item["viewpoint"] for item in group_items]
                )

                evidence_ids = [item["evidence"]["node_id"] for item in group_items]
                key_claims = self._extract_key_claims(combined_viewpoint)

                perspective = Perspective(
                    perspective_id=f"stakeholder_{stakeholder}_{hash(str(group_items))}",
                    topic=topic,
                    viewpoint=combined_viewpoint,
                    supporting_evidence=evidence_ids,
                    confidence_score=self._calculate_perspective_confidence(group_items),
                    source_count=len(group_items),
                    perspective_type=PerspectiveType.STAKEHOLDER,
                    metadata={
                        "stakeholder_group": stakeholder,
                        "indicators_found": list(set(item["indicator"] for item in group_items)),
                    },
                    key_claims=key_claims,
                    evidence_quality=self._assess_evidence_quality(group_items),
                )
                perspectives.append(perspective)

        return perspectives

    def _extract_methodological_perspectives(
        self, topic: str, topic_data: List[Dict[str, Any]], min_evidence_count: int
    ) -> List[Perspective]:
        """Extract different methodological approaches."""
        perspectives = []

        # Look for methodological approaches
        method_groups = defaultdict(list)

        for data in topic_data:
            content = data["content"]
            content_lower = content.lower()

            # Check for methodological language
            for indicator in self.perspective_indicators[PerspectiveType.METHODOLOGICAL]:
                if indicator in content_lower:
                    viewpoint = self._extract_viewpoint_around_indicator(content, indicator)

                    if viewpoint:
                        # Extract the specific methodology
                        methodology = self._extract_methodology(viewpoint, indicator)

                        if methodology:
                            group_key = self._generate_perspective_key(methodology)
                            method_groups[group_key].append(
                                {
                                    "viewpoint": viewpoint,
                                    "methodology": methodology,
                                    "evidence": data,
                                    "indicator": indicator,
                                }
                            )
                    break

        # Create perspectives
        for group_key, group_items in method_groups.items():
            if len(group_items) >= min_evidence_count:
                combined_viewpoint = self._combine_viewpoints(
                    [item["viewpoint"] for item in group_items]
                )

                evidence_ids = [item["evidence"]["node_id"] for item in group_items]
                key_claims = self._extract_key_claims(combined_viewpoint)

                perspective = Perspective(
                    perspective_id=f"methodological_{hash(group_key)}",
                    topic=topic,
                    viewpoint=combined_viewpoint,
                    supporting_evidence=evidence_ids,
                    confidence_score=self._calculate_perspective_confidence(group_items),
                    source_count=len(group_items),
                    perspective_type=PerspectiveType.METHODOLOGICAL,
                    metadata={
                        "methodologies": list(set(item["methodology"] for item in group_items))
                    },
                    key_claims=key_claims,
                    evidence_quality=self._assess_evidence_quality(group_items),
                )
                perspectives.append(perspective)

        return perspectives

    def _extract_temporal_perspectives(
        self, topic: str, topic_data: List[Dict[str, Any]], min_evidence_count: int
    ) -> List[Perspective]:
        """Extract how perspectives have evolved over time."""
        perspectives = []

        # Group by time periods
        temporal_groups = defaultdict(list)

        for data in topic_data:
            metadata = data["metadata"]

            # Extract temporal information
            timestamp = None
            for field in ["timestamp", "created_at", "date", "modified_at"]:
                if field in metadata:
                    try:
                        if isinstance(metadata[field], str):
                            timestamp = datetime.fromisoformat(
                                metadata[field].replace("Z", "+00:00")
                            )
                        elif isinstance(metadata[field], (int, float)):
                            timestamp = datetime.fromtimestamp(metadata[field])
                        break
                    except:
                        continue

            if timestamp:
                # Group by year
                year = timestamp.year
                temporal_groups[year].append(data)

        # Create temporal perspectives for periods with sufficient data
        for year, year_data in temporal_groups.items():
            if len(year_data) >= min_evidence_count:
                # Extract dominant viewpoint for this period
                combined_content = " ".join(item["content"] for item in year_data)
                viewpoint = self._extract_dominant_viewpoint(combined_content, topic)

                if viewpoint:
                    evidence_ids = [item["node_id"] for item in year_data]
                    key_claims = self._extract_key_claims(viewpoint)

                    perspective = Perspective(
                        perspective_id=f"temporal_{year}_{hash(viewpoint)}",
                        topic=topic,
                        viewpoint=viewpoint,
                        supporting_evidence=evidence_ids,
                        confidence_score=len(year_data)
                        / max(len(temporal_groups.values()), key=len),
                        source_count=len(year_data),
                        perspective_type=PerspectiveType.TEMPORAL,
                        metadata={"time_period": year, "data_points": len(year_data)},
                        key_claims=key_claims,
                        evidence_quality=sum(item.get("relevance_score", 0.5) for item in year_data)
                        / len(year_data),
                    )
                    perspectives.append(perspective)

        return perspectives

    def _extract_viewpoint_around_indicator(self, content: str, indicator: str) -> Optional[str]:
        """Extract the viewpoint around a perspective indicator."""
        import re

        # Find sentences containing the indicator
        sentences = re.split(r"[.!?]+", content)

        for sentence in sentences:
            if indicator.lower() in sentence.lower():
                # Return the sentence and potentially the next one
                sentence_idx = sentences.index(sentence)

                viewpoint_parts = [sentence.strip()]

                # Add next sentence if it's short and relevant
                if sentence_idx + 1 < len(sentences) and len(sentences[sentence_idx + 1]) < 200:
                    viewpoint_parts.append(sentences[sentence_idx + 1].strip())

                viewpoint = " ".join(viewpoint_parts)

                # Clean up and return if meaningful
                if len(viewpoint.strip()) > 20:
                    return viewpoint.strip()

        return None

    def _generate_perspective_key(self, viewpoint: str) -> str:
        """Generate a key for grouping similar perspectives."""
        # Extract key terms for grouping
        import re

        # Remove common words and get key terms
        words = re.findall(r"\b[a-zA-Z]{4,}\b", viewpoint.lower())

        # Remove stop words
        stop_words = {
            "that",
            "this",
            "with",
            "from",
            "they",
            "have",
            "been",
            "were",
            "will",
            "would",
            "could",
            "should",
            "must",
            "might",
            "also",
            "some",
            "many",
            "most",
            "very",
            "more",
            "than",
            "such",
            "when",
            "where",
            "what",
            "which",
            "their",
            "there",
            "these",
            "those",
        }

        key_words = [word for word in words if word not in stop_words]

        # Use top 3 words as key
        return "_".join(sorted(key_words[:3]))

    def _find_semantic_group(self, viewpoint: str, existing_keys: List[str]) -> Optional[str]:
        """Find existing semantic group for a viewpoint."""
        # Simple similarity check
        viewpoint_words = set(viewpoint.lower().split())

        for key in existing_keys:
            key_words = set(key.split("_"))

            # Check for word overlap
            overlap = len(viewpoint_words & key_words)
            total_words = len(viewpoint_words | key_words)

            if total_words > 0 and overlap / total_words > 0.3:  # 30% similarity
                return key

        return None

    def _combine_viewpoints(self, viewpoints: List[str]) -> str:
        """Combine multiple viewpoints into a coherent perspective."""
        if not viewpoints:
            return ""

        if len(viewpoints) == 1:
            return viewpoints[0]

        # Simple combination - take the longest viewpoint and add key points from others
        main_viewpoint = max(viewpoints, key=len)

        # Extract key phrases from other viewpoints
        additional_points = []
        for viewpoint in viewpoints:
            if viewpoint != main_viewpoint:
                # Extract unique phrases
                phrases = self._extract_key_phrases(viewpoint)
                for phrase in phrases:
                    if phrase.lower() not in main_viewpoint.lower():
                        additional_points.append(phrase)

        # Combine
        if additional_points:
            combined = main_viewpoint + " Additionally, " + ". ".join(additional_points[:2])
            return combined
        else:
            return main_viewpoint

    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text."""
        import re

        # Split into clauses and phrases
        clauses = re.split(r"[,;]", text)

        # Filter meaningful clauses
        key_phrases = []
        for clause in clauses:
            clause = clause.strip()
            if 10 <= len(clause) <= 100:  # Reasonable length
                key_phrases.append(clause)

        return key_phrases[:3]  # Top 3 phrases

    def _extract_key_claims(self, viewpoint: str) -> List[str]:
        """Extract key claims from a viewpoint."""
        import re

        # Look for assertive statements
        sentences = re.split(r"[.!?]+", viewpoint)

        claims = []
        for sentence in sentences:
            sentence = sentence.strip()

            # Look for claim indicators
            claim_indicators = ["is", "are", "shows", "demonstrates", "proves", "indicates"]

            if len(sentence) > 15 and any(
                indicator in sentence.lower() for indicator in claim_indicators
            ):
                claims.append(sentence)

        return claims[:3]  # Top 3 claims

    def _extract_alternative_approach(self, viewpoint: str) -> Optional[str]:
        """Extract the specific alternative approach from a viewpoint."""
        import re

        # Look for approach/method descriptions
        approach_patterns = [
            r"approach is to ([^.]+)",
            r"method involves ([^.]+)",
            r"strategy is ([^.]+)",
            r"technique uses ([^.]+)",
            r"way to ([^.]+)",
        ]

        for pattern in approach_patterns:
            match = re.search(pattern, viewpoint.lower())
            if match:
                return match.group(1).strip()

        # Fallback - extract main action/concept
        words = viewpoint.split()
        if len(words) > 5:
            return " ".join(words[:10])  # First 10 words

        return None

    def _extract_methodology(self, viewpoint: str, indicator: str) -> Optional[str]:
        """Extract methodology description from viewpoint."""
        import re

        # Look for methodology descriptions around the indicator
        viewpoint_lower = viewpoint.lower()
        indicator_pos = viewpoint_lower.find(indicator.lower())

        if indicator_pos != -1:
            # Extract text around the indicator
            start = max(0, indicator_pos - 50)
            end = min(len(viewpoint), indicator_pos + len(indicator) + 100)

            methodology_text = viewpoint[start:end]

            # Extract the methodology name/description
            method_patterns = [
                r"using ([^,\.]+)",
                r"through ([^,\.]+)",
                r"by ([^,\.]+)",
                r"with ([^,\.]+)",
            ]

            for pattern in method_patterns:
                match = re.search(pattern, methodology_text.lower())
                if match:
                    return match.group(1).strip()

        return None

    def _extract_dominant_viewpoint(self, combined_content: str, topic: str) -> Optional[str]:
        """Extract the dominant viewpoint from combined content."""
        # Simple approach - find the most common themes
        import re

        sentences = re.split(r"[.!?]+", combined_content)

        # Find sentences most relevant to the topic
        topic_words = topic.lower().split()
        relevant_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and any(word in sentence.lower() for word in topic_words):
                relevant_sentences.append(sentence)

        if relevant_sentences:
            # Return the longest relevant sentence as the dominant viewpoint
            return max(relevant_sentences, key=len)

        return None

    def _calculate_perspective_confidence(self, group_items: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for a perspective."""
        # Base confidence from number of sources
        source_confidence = min(len(group_items) / 5, 1.0)  # Max at 5 sources

        # Evidence quality (if available)
        evidence_scores = []
        for item in group_items:
            evidence = item["evidence"]
            score = evidence.get("relevance_score", 0.5)
            evidence_scores.append(score)

        avg_evidence_quality = (
            sum(evidence_scores) / len(evidence_scores) if evidence_scores else 0.5
        )

        # Combine scores
        confidence = (source_confidence * 0.6) + (avg_evidence_quality * 0.4)

        return min(confidence, 0.95)  # Cap at 95%

    def _assess_evidence_quality(self, group_items: List[Dict[str, Any]]) -> float:
        """Assess the overall quality of evidence for a perspective."""
        quality_scores = []

        for item in group_items:
            evidence = item["evidence"]

            # Content length (reasonable length is higher quality)
            content_length = len(evidence["content"])
            length_score = min(content_length / 500, 1.0) if content_length < 2000 else 0.8

            # Relevance score
            relevance_score = evidence.get("relevance_score", 0.5)

            # Metadata richness
            metadata_score = min(len(evidence["metadata"]) / 5, 1.0)

            # Relationship connectivity
            relationship_score = min(len(evidence["relationships"]) / 3, 1.0)

            # Combined quality score
            quality = (
                length_score * 0.3
                + relevance_score * 0.4
                + metadata_score * 0.2
                + relationship_score * 0.1
            )

            quality_scores.append(quality)

        return sum(quality_scores) / len(quality_scores) if quality_scores else 0.5

    def _deduplicate_perspectives(self, perspectives: List[Perspective]) -> List[Perspective]:
        """Remove duplicate perspectives based on similarity."""
        if len(perspectives) <= 1:
            return perspectives

        unique_perspectives = []

        for perspective in perspectives:
            is_duplicate = False

            for existing in unique_perspectives:
                # Check similarity
                similarity = self._calculate_perspective_similarity(perspective, existing)

                if similarity > 0.8:  # High similarity threshold
                    is_duplicate = True

                    # Keep the one with higher confidence
                    if perspective.confidence_score > existing.confidence_score:
                        unique_perspectives.remove(existing)
                        unique_perspectives.append(perspective)

                    break

            if not is_duplicate:
                unique_perspectives.append(perspective)

        return unique_perspectives

    def _calculate_perspective_similarity(self, p1: Perspective, p2: Perspective) -> float:
        """Calculate similarity between two perspectives."""
        # Compare viewpoints
        viewpoint1_words = set(p1.viewpoint.lower().split())
        viewpoint2_words = set(p2.viewpoint.lower().split())

        if viewpoint1_words or viewpoint2_words:
            intersection = len(viewpoint1_words & viewpoint2_words)
            union = len(viewpoint1_words | viewpoint2_words)
            viewpoint_similarity = intersection / union if union > 0 else 0
        else:
            viewpoint_similarity = 0

        # Compare key claims
        claims1 = set(" ".join(p1.key_claims).lower().split())
        claims2 = set(" ".join(p2.key_claims).lower().split())

        if claims1 or claims2:
            claims_intersection = len(claims1 & claims2)
            claims_union = len(claims1 | claims2)
            claims_similarity = claims_intersection / claims_union if claims_union > 0 else 0
        else:
            claims_similarity = 0

        # Combined similarity
        return (viewpoint_similarity * 0.7) + (claims_similarity * 0.3)

    def _score_perspectives(self, perspectives: List[Perspective]) -> List[Perspective]:
        """Score and rank perspectives by quality and relevance."""
        for perspective in perspectives:
            # Calculate overall score based on multiple factors
            score = (
                perspective.confidence_score * 0.4
                + perspective.evidence_quality * 0.3
                + min(perspective.source_count / 5, 1.0) * 0.2
                + min(len(perspective.key_claims) / 3, 1.0) * 0.1
            )

            # Update confidence score with overall score
            perspective.confidence_score = min(score, 1.0)

        # Sort by confidence score
        perspectives.sort(key=lambda p: p.confidence_score, reverse=True)

        return perspectives


class PerspectiveComparator:
    """Compares and analyzes relationships between perspectives."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def compare_perspectives(
        self, topic: str, perspectives: List[Perspective]
    ) -> PerspectiveComparison:
        """
        Compare multiple perspectives on a topic.

        Args:
            topic: The topic being analyzed
            perspectives: List of perspectives to compare

        Returns:
            PerspectiveComparison with analysis results
        """
        try:
            if len(perspectives) < 2:
                return self._create_single_perspective_comparison(topic, perspectives)

            # Create comparison matrix
            comparison_matrix = self._create_comparison_matrix(perspectives)

            # Find consensus areas
            consensus_areas = self._find_consensus_areas(perspectives)

            # Find disagreement areas
            disagreement_areas = self._find_disagreement_areas(perspectives)

            # Determine overall consensus level
            consensus_level = self._determine_consensus_level(perspectives, comparison_matrix)

            # Generate synthesis summary
            synthesis_summary = self._generate_synthesis_summary(
                topic, perspectives, consensus_areas, disagreement_areas
            )

            # Calculate analysis confidence
            analysis_confidence = self._calculate_analysis_confidence(perspectives)

            return PerspectiveComparison(
                topic=topic,
                perspectives=perspectives,
                consensus_areas=consensus_areas,
                disagreement_areas=disagreement_areas,
                consensus_level=consensus_level,
                comparison_matrix=comparison_matrix,
                synthesis_summary=synthesis_summary,
                confidence_in_analysis=analysis_confidence,
            )

        except Exception as e:
            self.logger.error(f"Error comparing perspectives: {e}")
            return self._create_error_comparison(topic, perspectives, str(e))

    def _create_comparison_matrix(
        self, perspectives: List[Perspective]
    ) -> Dict[str, Dict[str, float]]:
        """Create a matrix showing similarity between all perspective pairs."""
        matrix = {}

        for i, p1 in enumerate(perspectives):
            matrix[p1.perspective_id] = {}

            for j, p2 in enumerate(perspectives):
                if i == j:
                    similarity = 1.0
                else:
                    similarity = self._calculate_perspective_similarity(p1, p2)

                matrix[p1.perspective_id][p2.perspective_id] = similarity

        return matrix

    def _calculate_perspective_similarity(self, p1: Perspective, p2: Perspective) -> float:
        """Calculate similarity between two perspectives."""
        # Different perspective types are inherently less similar
        if p1.perspective_type != p2.perspective_type:
            type_penalty = 0.2
        else:
            type_penalty = 0.0

        # Compare viewpoints
        viewpoint_similarity = self._calculate_text_similarity(p1.viewpoint, p2.viewpoint)

        # Compare key claims
        claims1 = " ".join(p1.key_claims)
        claims2 = " ".join(p2.key_claims)
        claims_similarity = self._calculate_text_similarity(claims1, claims2)

        # Check for overlapping evidence
        evidence_overlap = len(set(p1.supporting_evidence) & set(p2.supporting_evidence))
        max_evidence = max(len(p1.supporting_evidence), len(p2.supporting_evidence))
        evidence_similarity = evidence_overlap / max_evidence if max_evidence > 0 else 0

        # Combined similarity
        similarity = (
            viewpoint_similarity * 0.5 + claims_similarity * 0.3 + evidence_similarity * 0.2
        ) - type_penalty

        return max(0, similarity)

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        if not text1 or not text2:
            return 0.0

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _find_consensus_areas(self, perspectives: List[Perspective]) -> List[str]:
        """Find areas where perspectives agree."""
        consensus_areas = []

        if len(perspectives) < 2:
            return consensus_areas

        # Extract all claims
        all_claims = []
        for perspective in perspectives:
            all_claims.extend(perspective.key_claims)

        # Find similar claims across perspectives
        claim_groups = self._group_similar_claims(all_claims)

        # Identify consensus (claims supported by multiple perspectives)
        for group in claim_groups:
            if len(group) >= max(2, len(perspectives) * 0.5):  # At least half support
                # Create consensus statement
                consensus_statement = self._create_consensus_statement(group)
                consensus_areas.append(consensus_statement)

        return consensus_areas

    def _group_similar_claims(self, claims: List[str]) -> List[List[str]]:
        """Group similar claims together."""
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

                similarity = self._calculate_text_similarity(claim1, claim2)
                if similarity > 0.6:  # Similarity threshold for grouping
                    group.append(claim2)
                    processed.add(j)

            if len(group) > 1:  # Only keep groups with multiple claims
                groups.append(group)

        return groups

    def _create_consensus_statement(self, similar_claims: List[str]) -> str:
        """Create a consensus statement from similar claims."""
        if not similar_claims:
            return ""

        if len(similar_claims) == 1:
            return similar_claims[0]

        # Find common words/themes
        all_words = []
        for claim in similar_claims:
            all_words.extend(claim.lower().split())

        word_counts = Counter(all_words)
        common_words = [word for word, count in word_counts.items() if count >= 2]

        # Use the longest claim as base and incorporate common themes
        base_claim = max(similar_claims, key=len)

        return f"There is consensus that {base_claim.lower()}"

    def _find_disagreement_areas(self, perspectives: List[Perspective]) -> List[str]:
        """Find areas where perspectives disagree."""
        disagreement_areas = []

        if len(perspectives) < 2:
            return disagreement_areas

        # Look for opposing perspectives
        opposing_pairs = []

        for i, p1 in enumerate(perspectives):
            for p2 in perspectives[i + 1 :]:
                # Check if perspectives are opposing
                if self._are_perspectives_opposing(p1, p2):
                    opposing_pairs.append((p1, p2))

        # Create disagreement statements
        for p1, p2 in opposing_pairs:
            disagreement = self._create_disagreement_statement(p1, p2)
            disagreement_areas.append(disagreement)

        return disagreement_areas

    def _are_perspectives_opposing(self, p1: Perspective, p2: Perspective) -> bool:
        """Check if two perspectives are opposing."""
        # Check for opposing keywords
        opposing_keywords = [
            ("positive", "negative"),
            ("good", "bad"),
            ("effective", "ineffective"),
            ("beneficial", "harmful"),
            ("increase", "decrease"),
            ("support", "oppose"),
            ("agree", "disagree"),
            ("should", "should not"),
            ("yes", "no"),
        ]

        text1 = (p1.viewpoint + " " + " ".join(p1.key_claims)).lower()
        text2 = (p2.viewpoint + " " + " ".join(p2.key_claims)).lower()

        for pos_word, neg_word in opposing_keywords:
            if (pos_word in text1 and neg_word in text2) or (
                neg_word in text1 and pos_word in text2
            ):
                return True

        # Check for explicit opposition types
        if (
            p1.perspective_type == PerspectiveType.OPPOSING
            or p2.perspective_type == PerspectiveType.OPPOSING
        ):
            return True

        return False

    def _create_disagreement_statement(self, p1: Perspective, p2: Perspective) -> str:
        """Create a statement describing disagreement between perspectives."""
        return (
            f"There is disagreement between {p1.perspective_type.value} and "
            f"{p2.perspective_type.value} perspectives on key aspects"
        )

    def _determine_consensus_level(
        self, perspectives: List[Perspective], comparison_matrix: Dict[str, Dict[str, float]]
    ) -> ConsensusLevel:
        """Determine the overall level of consensus among perspectives."""
        if len(perspectives) < 2:
            return ConsensusLevel.STRONG_CONSENSUS

        # Calculate average similarity
        similarities = []
        for p1_id in comparison_matrix:
            for p2_id in comparison_matrix[p1_id]:
                if p1_id != p2_id:
                    similarities.append(comparison_matrix[p1_id][p2_id])

        if not similarities:
            return ConsensusLevel.NO_CONSENSUS

        avg_similarity = sum(similarities) / len(similarities)

        # Map similarity to consensus level
        if avg_similarity >= 0.8:
            return ConsensusLevel.STRONG_CONSENSUS
        elif avg_similarity >= 0.6:
            return ConsensusLevel.MODERATE_CONSENSUS
        elif avg_similarity >= 0.4:
            return ConsensusLevel.WEAK_CONSENSUS
        elif avg_similarity >= 0.2:
            return ConsensusLevel.NO_CONSENSUS
        else:
            return ConsensusLevel.STRONG_DISAGREEMENT

    def _generate_synthesis_summary(
        self,
        topic: str,
        perspectives: List[Perspective],
        consensus_areas: List[str],
        disagreement_areas: List[str],
    ) -> str:
        """Generate a synthesis summary of the perspective analysis."""
        summary_parts = []

        # Topic introduction
        summary_parts.append(f"Analysis of {len(perspectives)} perspectives on {topic}:")

        # Consensus summary
        if consensus_areas:
            summary_parts.append(f"Areas of consensus include: {'; '.join(consensus_areas[:2])}")
        else:
            summary_parts.append("No clear consensus areas identified")

        # Disagreement summary
        if disagreement_areas:
            summary_parts.append(f"Key disagreements involve: {'; '.join(disagreement_areas[:2])}")
        else:
            summary_parts.append("No significant disagreements found")

        # Perspective type distribution
        type_counts = Counter(p.perspective_type.value for p in perspectives)
        dominant_type = type_counts.most_common(1)[0]
        summary_parts.append(f"Perspectives are primarily {dominant_type[0]} in nature")

        return ". ".join(summary_parts) + "."

    def _calculate_analysis_confidence(self, perspectives: List[Perspective]) -> float:
        """Calculate confidence in the perspective analysis."""
        if not perspectives:
            return 0.0

        # Base confidence from number of perspectives
        count_confidence = min(len(perspectives) / 5, 1.0)

        # Average perspective confidence
        avg_perspective_confidence = sum(p.confidence_score for p in perspectives) / len(
            perspectives
        )

        # Evidence diversity (different types of perspectives)
        type_diversity = len(set(p.perspective_type for p in perspectives)) / len(PerspectiveType)

        # Combined confidence
        confidence = (
            count_confidence * 0.4 + avg_perspective_confidence * 0.4 + type_diversity * 0.2
        )

        return min(confidence, 0.95)

    def _create_single_perspective_comparison(
        self, topic: str, perspectives: List[Perspective]
    ) -> PerspectiveComparison:
        """Create comparison for single perspective (or empty list)."""
        if not perspectives:
            return PerspectiveComparison(
                topic=topic,
                perspectives=[],
                consensus_areas=[],
                disagreement_areas=[],
                consensus_level=ConsensusLevel.NO_CONSENSUS,
                comparison_matrix={},
                synthesis_summary=f"No perspectives found for {topic}",
                confidence_in_analysis=0.0,
            )

        perspective = perspectives[0]
        return PerspectiveComparison(
            topic=topic,
            perspectives=[perspective],
            consensus_areas=perspective.key_claims,
            disagreement_areas=[],
            consensus_level=ConsensusLevel.STRONG_CONSENSUS,
            comparison_matrix={perspective.perspective_id: {perspective.perspective_id: 1.0}},
            synthesis_summary=f"Single {perspective.perspective_type.value} perspective on {topic}: {perspective.viewpoint[:100]}...",
            confidence_in_analysis=perspective.confidence_score,
        )

    def _create_error_comparison(
        self, topic: str, perspectives: List[Perspective], error_msg: str
    ) -> PerspectiveComparison:
        """Create error comparison when analysis fails."""
        return PerspectiveComparison(
            topic=topic,
            perspectives=perspectives,
            consensus_areas=[],
            disagreement_areas=[],
            consensus_level=ConsensusLevel.NO_CONSENSUS,
            comparison_matrix={},
            synthesis_summary=f"Perspective comparison failed for {topic}: {error_msg}",
            confidence_in_analysis=0.0,
        )


class PerspectiveAnalysisEngine:
    """
    Complete Perspective-Based Analysis Engine.

    Integrates perspective extraction and comparison to provide comprehensive
    analysis of different viewpoints on topics, including consensus and disagreement
    identification.
    """

    def __init__(self, query_engine: AdvancedQueryEngine):
        """
        Initialize the Perspective Analysis Engine.

        Args:
            query_engine: Query engine for accessing knowledge graph
        """
        self.query_engine = query_engine
        self.perspective_extractor = PerspectiveExtractor(query_engine)
        self.perspective_comparator = PerspectiveComparator()
        self.logger = logging.getLogger(__name__)

        # Statistics
        self.stats = {
            "analyses_performed": 0,
            "perspectives_extracted": 0,
            "comparisons_made": 0,
            "avg_processing_time_ms": 0.0,
            "consensus_distribution": Counter(),
        }

    def analyze_perspectives(
        self,
        topic: str,
        perspective_types: List[PerspectiveType] = None,
        include_temporal_analysis: bool = True,
        include_stakeholder_analysis: bool = True,
    ) -> PerspectiveAnalysisReport:
        """
        Perform comprehensive perspective analysis on a topic.

        Args:
            topic: Topic to analyze perspectives for
            perspective_types: Types of perspectives to extract
            include_temporal_analysis: Whether to include temporal evolution
            include_stakeholder_analysis: Whether to include stakeholder analysis

        Returns:
            PerspectiveAnalysisReport with comprehensive analysis
        """
        start_time = time.time()

        try:
            self.logger.info(f"Starting perspective analysis for topic: {topic}")

            # Extract perspectives
            perspectives = self.perspective_extractor.extract_perspectives(
                topic=topic, perspective_types=perspective_types
            )

            # Compare perspectives
            comparisons = []
            if len(perspectives) >= 2:
                # Group perspectives by type for targeted comparisons
                type_groups = defaultdict(list)
                for perspective in perspectives:
                    type_groups[perspective.perspective_type].append(perspective)

                # Compare within types
                for ptype, group_perspectives in type_groups.items():
                    if len(group_perspectives) >= 2:
                        comparison = self.perspective_comparator.compare_perspectives(
                            f"{topic} ({ptype.value})", group_perspectives
                        )
                        comparisons.append(comparison)

                # Overall comparison
                if len(perspectives) >= 2:
                    overall_comparison = self.perspective_comparator.compare_perspectives(
                        topic, perspectives
                    )
                    comparisons.append(overall_comparison)

            # Stakeholder analysis
            stakeholder_analysis = []
            if include_stakeholder_analysis:
                stakeholder_analysis = self._perform_stakeholder_analysis(topic, perspectives)

            # Temporal evolution analysis
            temporal_evolution = None
            if include_temporal_analysis:
                temporal_evolution = self._analyze_temporal_evolution(topic, perspectives)

            # Determine overall consensus
            overall_consensus = self._determine_overall_consensus(comparisons)

            # Generate insights and recommendations
            insights = self._generate_insights(perspectives, comparisons)
            recommendations = self._generate_recommendations(perspectives, comparisons)

            # Calculate analysis confidence
            analysis_confidence = self._calculate_analysis_confidence(perspectives, comparisons)

            processing_time = (time.time() - start_time) * 1000

            # Create report
            report = PerspectiveAnalysisReport(
                topic=topic,
                perspectives=perspectives,
                comparisons=comparisons,
                stakeholder_analysis=stakeholder_analysis,
                temporal_evolution=temporal_evolution,
                overall_consensus=overall_consensus,
                key_insights=insights,
                recommendations=recommendations,
                analysis_confidence=analysis_confidence,
                processing_time_ms=processing_time,
            )

            # Update statistics
            self._update_statistics(report, start_time)

            self.logger.info(
                f"Perspective analysis completed: {len(perspectives)} perspectives, "
                f"{len(comparisons)} comparisons"
            )

            return report

        except Exception as e:
            self.logger.error(f"Error during perspective analysis: {e}")

            # Return error report
            return PerspectiveAnalysisReport(
                topic=topic,
                perspectives=[],
                comparisons=[],
                stakeholder_analysis=[],
                temporal_evolution=None,
                overall_consensus=ConsensusLevel.NO_CONSENSUS,
                key_insights=[f"Analysis failed: {str(e)}"],
                recommendations=[],
                analysis_confidence=0.0,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

    def _perform_stakeholder_analysis(
        self, topic: str, perspectives: List[Perspective]
    ) -> List[StakeholderPerspective]:
        """Perform stakeholder-specific analysis."""
        stakeholder_perspectives = []

        # Group perspectives by stakeholder type
        stakeholder_groups = defaultdict(list)

        for perspective in perspectives:
            if perspective.perspective_type == PerspectiveType.STAKEHOLDER:
                stakeholder_group = perspective.metadata.get("stakeholder_group", "unknown")
                stakeholder_groups[stakeholder_group].append(perspective)

        # Analyze each stakeholder group
        for group_name, group_perspectives in stakeholder_groups.items():
            if group_perspectives:
                # Use the strongest perspective for this stakeholder group
                main_perspective = max(group_perspectives, key=lambda p: p.confidence_score)

                # Extract stakeholder-specific information
                interests = self._extract_stakeholder_interests(group_name, main_perspective)
                concerns = self._extract_stakeholder_concerns(group_name, main_perspective)
                priorities = self._extract_stakeholder_priorities(group_name, main_perspective)
                influence_level = self._estimate_stakeholder_influence(group_name)

                stakeholder_perspective = StakeholderPerspective(
                    stakeholder_group=group_name,
                    perspective=main_perspective,
                    interests=interests,
                    concerns=concerns,
                    priorities=priorities,
                    influence_level=influence_level,
                )
                stakeholder_perspectives.append(stakeholder_perspective)

        return stakeholder_perspectives

    def _extract_stakeholder_interests(
        self, stakeholder_group: str, perspective: Perspective
    ) -> List[str]:
        """Extract interests for a stakeholder group."""
        interests = []

        # Define typical interests by stakeholder group
        interest_patterns = {
            "researchers": ["accuracy", "methodology", "validation", "evidence"],
            "practitioners": ["efficiency", "practicality", "implementation", "results"],
            "users": ["usability", "benefit", "cost", "accessibility"],
            "experts": ["quality", "standards", "best practices", "innovation"],
            "organizations": ["ROI", "scalability", "compliance", "competitive advantage"],
        }

        patterns = interest_patterns.get(stakeholder_group, [])
        viewpoint_lower = perspective.viewpoint.lower()

        for pattern in patterns:
            if pattern in viewpoint_lower:
                interests.append(pattern.replace("_", " ").title())

        # Extract from key claims
        for claim in perspective.key_claims:
            claim_lower = claim.lower()
            for pattern in patterns:
                if pattern in claim_lower:
                    interests.append(f"Claims about {pattern}")
                    break

        return interests[:3]  # Top 3 interests

    def _extract_stakeholder_concerns(
        self, stakeholder_group: str, perspective: Perspective
    ) -> List[str]:
        """Extract concerns for a stakeholder group."""
        concerns = []

        # Look for concern indicators in perspective
        concern_indicators = ["concern", "worry", "risk", "problem", "issue", "challenge"]

        viewpoint = perspective.viewpoint.lower()
        for indicator in concern_indicators:
            if indicator in viewpoint:
                # Extract context around the concern
                concern_context = self._extract_context_around_word(
                    perspective.viewpoint, indicator
                )
                if concern_context:
                    concerns.append(concern_context)

        return concerns[:3]  # Top 3 concerns

    def _extract_stakeholder_priorities(
        self, stakeholder_group: str, perspective: Perspective
    ) -> List[str]:
        """Extract priorities for a stakeholder group."""
        priorities = []

        # Look for priority indicators
        priority_indicators = ["important", "priority", "critical", "essential", "key", "primary"]

        for claim in perspective.key_claims:
            claim_lower = claim.lower()
            for indicator in priority_indicators:
                if indicator in claim_lower:
                    priorities.append(claim[:50] + "..." if len(claim) > 50 else claim)
                    break

        return priorities[:3]  # Top 3 priorities

    def _estimate_stakeholder_influence(self, stakeholder_group: str) -> float:
        """Estimate the influence level of a stakeholder group."""
        # Simple influence estimation based on stakeholder type
        influence_levels = {
            "experts": 0.9,
            "researchers": 0.8,
            "organizations": 0.7,
            "practitioners": 0.6,
            "users": 0.5,
            "unknown": 0.3,
        }

        return influence_levels.get(stakeholder_group, 0.3)

    def _extract_context_around_word(
        self, text: str, word: str, context_size: int = 50
    ) -> Optional[str]:
        """Extract context around a specific word."""
        word_pos = text.lower().find(word.lower())
        if word_pos == -1:
            return None

        start = max(0, word_pos - context_size)
        end = min(len(text), word_pos + len(word) + context_size)

        context = text[start:end].strip()

        return context if len(context) > 10 else None

    def _analyze_temporal_evolution(
        self, topic: str, perspectives: List[Perspective]
    ) -> Optional[TemporalPerspectiveEvolution]:
        """Analyze how perspectives have evolved over time."""
        # Filter temporal perspectives
        temporal_perspectives = [
            p for p in perspectives if p.perspective_type == PerspectiveType.TEMPORAL
        ]

        if len(temporal_perspectives) < 2:
            return None

        try:
            # Sort by time period
            temporal_perspectives.sort(key=lambda p: p.metadata.get("time_period", 0))

            # Analyze evolution pattern
            evolution_trend = self._determine_evolution_trend(temporal_perspectives)

            # Identify turning points
            turning_points = self._identify_turning_points(temporal_perspectives)

            # Determine current trajectory
            current_trajectory = self._determine_current_trajectory(temporal_perspectives)

            # Create time periods
            time_periods = []
            for perspective in temporal_perspectives:
                year = perspective.metadata.get("time_period")
                if year:
                    start = datetime(year, 1, 1)
                    end = datetime(year, 12, 31)
                    time_periods.append((start, end))

            # Create perspective changes
            perspective_changes = []
            for i, perspective in enumerate(temporal_perspectives):
                change = {
                    "period": perspective.metadata.get("time_period"),
                    "perspective_id": perspective.perspective_id,
                    "viewpoint": perspective.viewpoint,
                    "confidence": perspective.confidence_score,
                    "key_changes": self._identify_key_changes(
                        perspective, temporal_perspectives[:i]
                    ),
                }
                perspective_changes.append(change)

            return TemporalPerspectiveEvolution(
                topic=topic,
                time_periods=time_periods,
                perspective_changes=perspective_changes,
                evolution_trend=evolution_trend,
                key_turning_points=turning_points,
                current_trajectory=current_trajectory,
            )

        except Exception as e:
            self.logger.error(f"Error analyzing temporal evolution: {e}")
            return None

    def _determine_evolution_trend(self, temporal_perspectives: List[Perspective]) -> str:
        """Determine the overall evolution trend."""
        if len(temporal_perspectives) < 3:
            return "insufficient_data"

        # Calculate similarity between consecutive perspectives
        similarities = []
        for i in range(1, len(temporal_perspectives)):
            prev_perspective = temporal_perspectives[i - 1]
            curr_perspective = temporal_perspectives[i]

            similarity = self.perspective_comparator._calculate_perspective_similarity(
                prev_perspective, curr_perspective
            )
            similarities.append(similarity)

        avg_similarity = sum(similarities) / len(similarities)

        # Determine trend based on similarity patterns
        if avg_similarity > 0.7:
            return "stable"
        elif all(s1 > s2 for s1, s2 in zip(similarities[:-1], similarities[1:])):
            return "converging"
        elif all(s1 < s2 for s1, s2 in zip(similarities[:-1], similarities[1:])):
            return "diverging"
        else:
            return "cyclical"

    def _identify_turning_points(
        self, temporal_perspectives: List[Perspective]
    ) -> List[Dict[str, Any]]:
        """Identify key turning points in perspective evolution."""
        turning_points = []

        if len(temporal_perspectives) < 3:
            return turning_points

        # Look for significant changes between consecutive periods
        for i in range(1, len(temporal_perspectives) - 1):
            prev_perspective = temporal_perspectives[i - 1]
            curr_perspective = temporal_perspectives[i]
            next_perspective = temporal_perspectives[i + 1]

            # Calculate change magnitudes
            prev_similarity = self.perspective_comparator._calculate_perspective_similarity(
                prev_perspective, curr_perspective
            )
            next_similarity = self.perspective_comparator._calculate_perspective_similarity(
                curr_perspective, next_perspective
            )

            # Identify significant changes
            if abs(prev_similarity - next_similarity) > 0.3:  # Significant change threshold
                turning_point = {
                    "period": curr_perspective.metadata.get("time_period"),
                    "perspective_id": curr_perspective.perspective_id,
                    "change_magnitude": abs(prev_similarity - next_similarity),
                    "description": f"Significant perspective shift in {curr_perspective.metadata.get('time_period', 'unknown period')}",
                }
                turning_points.append(turning_point)

        return turning_points

    def _determine_current_trajectory(self, temporal_perspectives: List[Perspective]) -> str:
        """Determine the current trajectory of perspective evolution."""
        if len(temporal_perspectives) < 2:
            return "unknown"

        # Look at the most recent trend
        recent_perspectives = (
            temporal_perspectives[-3:]
            if len(temporal_perspectives) >= 3
            else temporal_perspectives[-2:]
        )

        if len(recent_perspectives) < 2:
            return "stable"

        # Calculate recent similarity trend
        similarities = []
        for i in range(1, len(recent_perspectives)):
            similarity = self.perspective_comparator._calculate_perspective_similarity(
                recent_perspectives[i - 1], recent_perspectives[i]
            )
            similarities.append(similarity)

        if len(similarities) == 1:
            return "stable" if similarities[0] > 0.6 else "changing"

        # Check if similarities are increasing (converging) or decreasing (diverging)
        if similarities[-1] > similarities[0]:
            return "converging"
        elif similarities[-1] < similarities[0]:
            return "diverging"
        else:
            return "stable"

    def _identify_key_changes(
        self, current_perspective: Perspective, previous_perspectives: List[Perspective]
    ) -> List[str]:
        """Identify key changes from previous perspectives."""
        changes = []

        if not previous_perspectives:
            return ["Initial perspective identified"]

        # Compare with most recent previous perspective
        prev_perspective = previous_perspectives[-1]

        # Compare viewpoints
        viewpoint_similarity = self.perspective_comparator._calculate_text_similarity(
            current_perspective.viewpoint, prev_perspective.viewpoint
        )

        if viewpoint_similarity < 0.5:
            changes.append("Significant viewpoint change")

        # Compare key claims
        current_claims = set(current_perspective.key_claims)
        prev_claims = set(prev_perspective.key_claims)

        new_claims = current_claims - prev_claims
        if new_claims:
            changes.append(f"New claims emerged: {len(new_claims)} new")

        lost_claims = prev_claims - current_claims
        if lost_claims:
            changes.append(f"Claims dropped: {len(lost_claims)} removed")

        # Compare confidence levels
        confidence_change = current_perspective.confidence_score - prev_perspective.confidence_score
        if abs(confidence_change) > 0.2:
            direction = "increased" if confidence_change > 0 else "decreased"
            changes.append(f"Confidence {direction} significantly")

        return changes if changes else ["Minor evolutionary changes"]

    def _determine_overall_consensus(
        self, comparisons: List[PerspectiveComparison]
    ) -> ConsensusLevel:
        """Determine overall consensus level across all comparisons."""
        if not comparisons:
            return ConsensusLevel.NO_CONSENSUS

        # Get consensus levels from all comparisons
        consensus_levels = [comp.consensus_level for comp in comparisons]

        # Calculate weighted average (overall comparison has more weight)
        if len(consensus_levels) == 1:
            return consensus_levels[0]

        # Map consensus levels to numeric values
        level_values = {
            ConsensusLevel.STRONG_CONSENSUS: 5,
            ConsensusLevel.MODERATE_CONSENSUS: 4,
            ConsensusLevel.WEAK_CONSENSUS: 3,
            ConsensusLevel.NO_CONSENSUS: 2,
            ConsensusLevel.STRONG_DISAGREEMENT: 1,
        }

        # Calculate average
        total_value = sum(level_values[level] for level in consensus_levels)
        avg_value = total_value / len(consensus_levels)

        # Map back to consensus level
        if avg_value >= 4.5:
            return ConsensusLevel.STRONG_CONSENSUS
        elif avg_value >= 3.5:
            return ConsensusLevel.MODERATE_CONSENSUS
        elif avg_value >= 2.5:
            return ConsensusLevel.WEAK_CONSENSUS
        elif avg_value >= 1.5:
            return ConsensusLevel.NO_CONSENSUS
        else:
            return ConsensusLevel.STRONG_DISAGREEMENT

    def _generate_insights(
        self, perspectives: List[Perspective], comparisons: List[PerspectiveComparison]
    ) -> List[str]:
        """Generate key insights from the perspective analysis."""
        insights = []

        # Perspective diversity insights
        if perspectives:
            type_counts = Counter(p.perspective_type.value for p in perspectives)
            most_common_type = type_counts.most_common(1)[0]

            insights.append(
                f"Perspectives are dominated by {most_common_type[0]} viewpoints ({most_common_type[1]} out of {len(perspectives)})"
            )

            # Quality insights
            high_confidence_count = len([p for p in perspectives if p.confidence_score > 0.7])
            if high_confidence_count > len(perspectives) * 0.5:
                insights.append(
                    f"High confidence in {high_confidence_count} perspectives indicates strong evidence base"
                )

            # Evidence diversity
            all_evidence = set()
            for perspective in perspectives:
                all_evidence.update(perspective.supporting_evidence)

            avg_evidence_per_perspective = len(all_evidence) / len(perspectives)
            if avg_evidence_per_perspective > 3:
                insights.append("Strong evidence diversity across perspectives")

        # Consensus insights
        if comparisons:
            consensus_areas_count = sum(len(comp.consensus_areas) for comp in comparisons)
            disagreement_areas_count = sum(len(comp.disagreement_areas) for comp in comparisons)

            if consensus_areas_count > disagreement_areas_count:
                insights.append("More consensus than disagreement identified across perspectives")
            elif disagreement_areas_count > consensus_areas_count:
                insights.append("Significant disagreements exist between perspectives")

        # Stakeholder insights
        stakeholder_perspectives = [
            p for p in perspectives if p.perspective_type == PerspectiveType.STAKEHOLDER
        ]
        if len(stakeholder_perspectives) >= 2:
            insights.append(
                f"Multiple stakeholder viewpoints ({len(stakeholder_perspectives)}) provide comprehensive coverage"
            )

        return insights[:5]  # Top 5 insights

    def _generate_recommendations(
        self, perspectives: List[Perspective], comparisons: List[PerspectiveComparison]
    ) -> List[str]:
        """Generate actionable recommendations based on the analysis."""
        recommendations = []

        # Evidence-based recommendations
        if perspectives:
            low_confidence_count = len([p for p in perspectives if p.confidence_score < 0.5])
            if low_confidence_count > len(perspectives) * 0.3:
                recommendations.append("Strengthen evidence base for low-confidence perspectives")

            # Type diversity recommendations
            type_counts = Counter(p.perspective_type.value for p in perspectives)
            if len(type_counts) < 3:
                recommendations.append(
                    "Seek additional perspective types for more comprehensive analysis"
                )

        # Consensus-based recommendations
        if comparisons:
            overall_comparison = next(
                (comp for comp in comparisons if len(comp.perspectives) == len(perspectives)), None
            )

            if overall_comparison:
                if overall_comparison.consensus_level in [
                    ConsensusLevel.STRONG_CONSENSUS,
                    ConsensusLevel.MODERATE_CONSENSUS,
                ]:
                    recommendations.append("Build on identified consensus areas for implementation")
                elif overall_comparison.consensus_level == ConsensusLevel.STRONG_DISAGREEMENT:
                    recommendations.append("Address fundamental disagreements before proceeding")
                else:
                    recommendations.append(
                        "Seek additional evidence to resolve areas of uncertainty"
                    )

        # Stakeholder recommendations
        stakeholder_perspectives = [
            p for p in perspectives if p.perspective_type == PerspectiveType.STAKEHOLDER
        ]
        if len(stakeholder_perspectives) < 2:
            recommendations.append(
                "Include additional stakeholder perspectives for balanced analysis"
            )

        # Temporal recommendations
        temporal_perspectives = [
            p for p in perspectives if p.perspective_type == PerspectiveType.TEMPORAL
        ]
        if temporal_perspectives:
            recommendations.append("Consider historical context when making decisions")

        return recommendations[:5]  # Top 5 recommendations

    def _calculate_analysis_confidence(
        self, perspectives: List[Perspective], comparisons: List[PerspectiveComparison]
    ) -> float:
        """Calculate overall confidence in the perspective analysis."""
        if not perspectives:
            return 0.0

        # Perspective quality confidence
        avg_perspective_confidence = sum(p.confidence_score for p in perspectives) / len(
            perspectives
        )

        # Perspective diversity confidence
        type_diversity = len(set(p.perspective_type for p in perspectives)) / len(PerspectiveType)

        # Comparison confidence
        comparison_confidence = 0.5
        if comparisons:
            comparison_confidence = sum(comp.confidence_in_analysis for comp in comparisons) / len(
                comparisons
            )

        # Evidence coverage confidence
        all_evidence = set()
        for perspective in perspectives:
            all_evidence.update(perspective.supporting_evidence)

        evidence_confidence = min(len(all_evidence) / 10, 1.0)  # Max at 10 unique sources

        # Combined confidence
        confidence = (
            avg_perspective_confidence * 0.4
            + type_diversity * 0.2
            + comparison_confidence * 0.2
            + evidence_confidence * 0.2
        )

        return min(confidence, 0.95)

    def _update_statistics(self, report: PerspectiveAnalysisReport, start_time: float):
        """Update engine statistics."""
        self.stats["analyses_performed"] += 1
        self.stats["perspectives_extracted"] += len(report.perspectives)
        self.stats["comparisons_made"] += len(report.comparisons)

        # Update consensus distribution
        self.stats["consensus_distribution"][report.overall_consensus.value] += 1

        # Update average processing time
        total_time = self.stats["avg_processing_time_ms"] * (self.stats["analyses_performed"] - 1)
        self.stats["avg_processing_time_ms"] = (
            total_time + report.processing_time_ms
        ) / self.stats["analyses_performed"]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get engine statistics.

        Returns:
            Dictionary with comprehensive statistics
        """
        return {
            "perspective_analysis": self.stats.copy(),
            "query_engine": self.query_engine.get_statistics(),
        }
