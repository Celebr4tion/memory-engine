"""
Result Ranking System for Advanced Query Engine

Ranks query results by combining relevance scores, quality metrics, and user preferences.
"""

import logging
import math
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from collections import defaultdict

from .query_types import QueryResult, QueryRequest, QueryType


@dataclass
class RankingCriteria:
    """Represents ranking criteria with weights."""

    relevance_weight: float = 0.4
    quality_weight: float = 0.3
    freshness_weight: float = 0.1
    popularity_weight: float = 0.1
    diversity_weight: float = 0.1
    custom_weights: Dict[str, float] = None

    def __post_init__(self):
        if self.custom_weights is None:
            self.custom_weights = {}

        # Normalize weights to sum to 1.0
        total_weight = (
            self.relevance_weight
            + self.quality_weight
            + self.freshness_weight
            + self.popularity_weight
            + self.diversity_weight
            + sum(self.custom_weights.values())
        )

        if total_weight > 0:
            self.relevance_weight /= total_weight
            self.quality_weight /= total_weight
            self.freshness_weight /= total_weight
            self.popularity_weight /= total_weight
            self.diversity_weight /= total_weight
            for key in self.custom_weights:
                self.custom_weights[key] /= total_weight


@dataclass
class RankingFeatures:
    """Features extracted from a query result for ranking."""

    relevance_score: float = 0.0
    quality_score: float = 0.0
    freshness_score: float = 0.0
    popularity_score: float = 0.0
    diversity_score: float = 0.0
    content_length_score: float = 0.0
    relationship_count_score: float = 0.0
    metadata_richness_score: float = 0.0
    custom_scores: Dict[str, float] = None

    def __post_init__(self):
        if self.custom_scores is None:
            self.custom_scores = {}


class ResultRanker:
    """
    Ranks query results using multiple scoring algorithms and criteria.

    Features:
    - Relevance-based ranking using semantic similarity
    - Quality-based ranking using node ratings
    - Freshness-based ranking using timestamps
    - Popularity-based ranking using access frequency
    - Diversity-based ranking to avoid redundant results
    - Customizable ranking criteria and weights
    """

    def __init__(self, quality_enhancement_engine=None):
        self.logger = logging.getLogger(__name__)

        # Optional quality enhancement engine for advanced quality scoring
        self.quality_enhancement_engine = quality_enhancement_engine

        # Statistics for adaptive ranking
        self.result_stats = defaultdict(dict)
        self.query_history = []
        self.ranking_feedback = defaultdict(list)

        # Caches for expensive computations
        self.similarity_cache = {}
        self.quality_cache = {}

        # Default ranking criteria
        self.default_criteria = RankingCriteria()

    def rank_results(
        self,
        results: List[QueryResult],
        request: QueryRequest,
        criteria: Optional[RankingCriteria] = None,
    ) -> List[QueryResult]:
        """
        Rank a list of query results based on multiple criteria.

        Args:
            results: List of query results to rank
            request: Original query request for context
            criteria: Ranking criteria (uses default if None)

        Returns:
            List of ranked query results
        """
        if not results:
            return results

        if criteria is None:
            criteria = self._get_adaptive_criteria(request)

        self.logger.info(f"Ranking {len(results)} results for query type: {request.query_type}")

        # Extract features for each result
        features_list = []
        for result in results:
            features = self._extract_ranking_features(result, request, results)
            features_list.append(features)

        # Calculate combined scores
        ranked_results = []
        for result, features in zip(results, features_list):
            combined_score = self._calculate_combined_score(features, criteria)

            # Update result with scores
            result.relevance_score = features.relevance_score
            result.quality_score = features.quality_score
            result.combined_score = combined_score

            ranked_results.append(result)

        # Sort by combined score (descending)
        ranked_results.sort(key=lambda r: r.combined_score, reverse=True)

        # Apply diversity if requested
        if criteria.diversity_weight > 0:
            ranked_results = self._apply_diversity_filtering(ranked_results, request)

        # Update statistics
        self._update_ranking_stats(request, ranked_results)

        self.logger.info(
            f"Ranking completed. Top result score: {ranked_results[0].combined_score:.3f}"
        )
        return ranked_results

    def _extract_ranking_features(
        self, result: QueryResult, request: QueryRequest, all_results: List[QueryResult]
    ) -> RankingFeatures:
        """
        Extract ranking features from a query result.

        Args:
            result: Query result to extract features from
            request: Query request for context
            all_results: All results for relative scoring

        Returns:
            RankingFeatures object
        """
        features = RankingFeatures()

        # Relevance score (from vector similarity or text matching)
        features.relevance_score = self._calculate_relevance_score(result, request)

        # Quality score (from node ratings)
        features.quality_score = self._calculate_quality_score(result)

        # Freshness score (from timestamps)
        features.freshness_score = self._calculate_freshness_score(result)

        # Popularity score (from access history)
        features.popularity_score = self._calculate_popularity_score(result)

        # Content length score (optimal length preference)
        features.content_length_score = self._calculate_content_length_score(result)

        # Relationship count score
        features.relationship_count_score = self._calculate_relationship_score(result)

        # Metadata richness score
        features.metadata_richness_score = self._calculate_metadata_richness_score(result)

        # Diversity score (relative to other results)
        features.diversity_score = self._calculate_diversity_score(result, all_results)

        return features

    def _calculate_relevance_score(self, result: QueryResult, request: QueryRequest) -> float:
        """
        Calculate relevance score based on query type and content.

        Args:
            result: Query result
            request: Query request

        Returns:
            Relevance score (0.0 to 1.0)
        """
        # Use existing relevance score if available
        if hasattr(result, "relevance_score") and result.relevance_score > 0:
            return min(result.relevance_score, 1.0)

        # Calculate based on query type
        if request.query_type == QueryType.SEMANTIC_SEARCH:
            # For semantic searches, use vector similarity
            return self._calculate_semantic_relevance(result, request)
        elif request.query_type == QueryType.NATURAL_LANGUAGE:
            # For natural language queries, use text matching
            return self._calculate_text_relevance(result, request)
        else:
            # For other query types, use a baseline score
            return 0.5

    def _calculate_semantic_relevance(self, result: QueryResult, request: QueryRequest) -> float:
        """Calculate semantic relevance using embeddings."""
        # This would typically use cosine similarity between query and result embeddings
        # For now, return a placeholder based on content matching
        query_lower = request.query.lower()
        content_lower = result.content.lower()

        # Simple keyword matching as a proxy for semantic similarity
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())

        if not query_words:
            return 0.0

        intersection = query_words.intersection(content_words)
        jaccard_similarity = len(intersection) / len(query_words.union(content_words))

        return min(jaccard_similarity * 2, 1.0)  # Scale up and cap at 1.0

    def _calculate_text_relevance(self, result: QueryResult, request: QueryRequest) -> float:
        """Calculate text-based relevance score."""
        query_lower = request.query.lower()
        content_lower = result.content.lower()

        # Exact phrase match bonus
        if query_lower in content_lower:
            return 1.0

        # Word overlap scoring
        query_words = query_lower.split()
        content_words = content_lower.split()

        if not query_words:
            return 0.0

        matches = sum(1 for word in query_words if word in content_words)
        return matches / len(query_words)

    def _calculate_quality_score(self, result: QueryResult) -> float:
        """
        Calculate quality score from node ratings or advanced quality assessment.

        Args:
            result: Query result

        Returns:
            Quality score (0.0 to 1.0)
        """
        # Use existing quality score if available
        if hasattr(result, "quality_score") and result.quality_score > 0:
            return min(result.quality_score, 1.0)

        # Check for enhanced quality score in metadata
        metadata = result.metadata or {}
        if "quality_score" in metadata:
            return min(metadata["quality_score"], 1.0)

        # Use quality enhancement engine if available
        if self.quality_enhancement_engine:
            try:
                from memory_core.model.knowledge_node import KnowledgeNode

                # Create temporary node for quality assessment using correct constructor
                temp_node = KnowledgeNode(
                    content=result.content,
                    source=metadata.get("source", "unknown"),
                    node_id=result.node_id,
                    rating_richness=metadata.get("rating_richness", 0.5),
                    rating_truthfulness=metadata.get("rating_truthfulness", 0.5),
                    rating_stability=metadata.get("rating_stability", 0.5),
                )

                # Add metadata and node_type as attributes
                temp_node.metadata = metadata
                temp_node.node_type = result.node_type or "document"

                # Get quality score from enhancement engine
                quality_score = self.quality_enhancement_engine.get_quality_score(temp_node)

                # Cache the result for future use
                self.quality_cache[result.node_id] = quality_score.overall_score

                return quality_score.overall_score

            except Exception as e:
                self.logger.warning(
                    f"Failed to get enhanced quality score for {result.node_id}: {e}"
                )
                # Fall through to basic quality calculation

        # Check quality cache
        if result.node_id in self.quality_cache:
            return self.quality_cache[result.node_id]

        # Extract quality metrics from metadata (fallback)
        richness = metadata.get("rating_richness", 0.5)
        truthfulness = metadata.get("rating_truthfulness", 0.5)
        stability = metadata.get("rating_stability", 0.5)

        # Weighted combination
        quality_score = richness * 0.4 + truthfulness * 0.4 + stability * 0.2
        quality_score = min(quality_score, 1.0)

        # Cache the result
        self.quality_cache[result.node_id] = quality_score

        return quality_score

    def _calculate_freshness_score(self, result: QueryResult) -> float:
        """
        Calculate freshness score based on creation/update timestamps.

        Args:
            result: Query result

        Returns:
            Freshness score (0.0 to 1.0)
        """
        metadata = result.metadata or {}

        # Get timestamps
        creation_timestamp = metadata.get("creation_timestamp")
        update_timestamp = metadata.get("update_timestamp")

        if not creation_timestamp and not update_timestamp:
            return 0.5  # Neutral score for unknown timestamps

        # Use the more recent timestamp
        timestamp = max(creation_timestamp or 0, update_timestamp or 0)

        # Calculate age in days
        import time

        current_time = time.time()
        age_days = (current_time - timestamp) / 86400  # 86400 seconds per day

        # Fresher content gets higher scores
        # Score decays exponentially with age
        if age_days <= 0:
            return 1.0
        elif age_days <= 7:  # Within a week
            return 0.9
        elif age_days <= 30:  # Within a month
            return 0.7
        elif age_days <= 90:  # Within 3 months
            return 0.5
        elif age_days <= 365:  # Within a year
            return 0.3
        else:
            return 0.1

    def _calculate_popularity_score(self, result: QueryResult) -> float:
        """
        Calculate popularity score based on access frequency.

        Args:
            result: Query result

        Returns:
            Popularity score (0.0 to 1.0)
        """
        node_id = result.node_id

        # Get access statistics
        stats = self.result_stats.get(node_id, {})
        access_count = stats.get("access_count", 0)
        recent_access_count = stats.get("recent_access_count", 0)

        # Calculate popularity based on access patterns
        if access_count == 0:
            return 0.1  # Minimum score for new content

        # Log scale for access count to prevent dominance by highly accessed items
        popularity_base = math.log(access_count + 1) / math.log(
            100
        )  # Normalize to common access ranges
        popularity_score = min(popularity_base, 1.0)

        # Boost for recent access
        if recent_access_count > 0:
            recent_boost = min(recent_access_count / 10, 0.2)  # Up to 20% boost
            popularity_score = min(popularity_score + recent_boost, 1.0)

        return popularity_score

    def _calculate_content_length_score(self, result: QueryResult) -> float:
        """
        Calculate score based on content length (optimal length preference).

        Args:
            result: Query result

        Returns:
            Content length score (0.0 to 1.0)
        """
        content_length = len(result.content)

        # Optimal range: 50-500 characters
        if 50 <= content_length <= 500:
            return 1.0
        elif content_length < 50:
            # Penalize very short content
            return content_length / 50
        else:
            # Gradually penalize very long content
            if content_length <= 1000:
                return 1.0 - (content_length - 500) / 1000
            else:
                return max(0.3, 1.0 - (content_length - 500) / 2000)

    def _calculate_relationship_score(self, result: QueryResult) -> float:
        """
        Calculate score based on relationship richness.

        Args:
            result: Query result

        Returns:
            Relationship score (0.0 to 1.0)
        """
        relationship_count = len(result.relationships) if result.relationships else 0

        # More relationships indicate higher connectivity and potential value
        if relationship_count == 0:
            return 0.2
        elif relationship_count <= 5:
            return 0.5 + relationship_count * 0.1
        else:
            # Diminishing returns for very high relationship counts
            return min(1.0, 0.8 + (relationship_count - 5) * 0.02)

    def _calculate_metadata_richness_score(self, result: QueryResult) -> float:
        """
        Calculate score based on metadata richness.

        Args:
            result: Query result

        Returns:
            Metadata richness score (0.0 to 1.0)
        """
        metadata = result.metadata or {}

        # Count meaningful metadata fields
        meaningful_fields = 0
        total_fields = len(metadata)

        # Check for specific valuable metadata
        if metadata.get("source"):
            meaningful_fields += 1
        if metadata.get("tags"):
            meaningful_fields += 1
        if metadata.get("domain"):
            meaningful_fields += 1
        if metadata.get("language"):
            meaningful_fields += 1
        if metadata.get("creation_timestamp"):
            meaningful_fields += 1

        # Add extra points for custom metadata
        custom_fields = max(0, total_fields - 5)  # Beyond the standard fields
        meaningful_fields += min(custom_fields, 3)  # Cap bonus from custom fields

        # Score based on richness
        max_meaningful = 8  # Maximum expected meaningful fields
        return min(meaningful_fields / max_meaningful, 1.0)

    def _calculate_diversity_score(
        self, result: QueryResult, all_results: List[QueryResult]
    ) -> float:
        """
        Calculate diversity score relative to other results.

        Args:
            result: Query result
            all_results: All results for comparison

        Returns:
            Diversity score (0.0 to 1.0)
        """
        if len(all_results) <= 1:
            return 1.0

        # Calculate diversity based on content uniqueness
        result_words = set(result.content.lower().split())

        similarities = []
        for other in all_results:
            if other.node_id == result.node_id:
                continue

            other_words = set(other.content.lower().split())

            # Jaccard similarity
            if not result_words and not other_words:
                similarity = 1.0
            elif not result_words or not other_words:
                similarity = 0.0
            else:
                intersection = result_words.intersection(other_words)
                union = result_words.union(other_words)
                similarity = len(intersection) / len(union)

            similarities.append(similarity)

        if not similarities:
            return 1.0

        # Higher diversity score for results that are less similar to others
        avg_similarity = sum(similarities) / len(similarities)
        diversity_score = 1.0 - avg_similarity

        return max(diversity_score, 0.0)

    def _calculate_combined_score(
        self, features: RankingFeatures, criteria: RankingCriteria
    ) -> float:
        """
        Calculate combined ranking score using weighted criteria.

        Args:
            features: Extracted ranking features
            criteria: Ranking criteria with weights

        Returns:
            Combined score (0.0 to 1.0)
        """
        score = (
            features.relevance_score * criteria.relevance_weight
            + features.quality_score * criteria.quality_weight
            + features.freshness_score * criteria.freshness_weight
            + features.popularity_score * criteria.popularity_weight
            + features.diversity_score * criteria.diversity_weight
        )

        # Add custom scoring
        for feature_name, weight in criteria.custom_weights.items():
            custom_score = features.custom_scores.get(feature_name, 0.0)
            score += custom_score * weight

        return min(score, 1.0)

    def _apply_diversity_filtering(
        self, results: List[QueryResult], request: QueryRequest
    ) -> List[QueryResult]:
        """
        Apply diversity filtering to avoid too similar results.

        Args:
            results: Ranked results
            request: Query request

        Returns:
            Diversity-filtered results
        """
        if len(results) <= 3:
            return results  # No filtering needed for small result sets

        filtered_results = [results[0]]  # Always include the top result

        similarity_threshold = 0.7  # Threshold for considering results too similar

        for result in results[1:]:
            is_diverse = True

            # Check similarity with already selected results
            for selected in filtered_results:
                if self._calculate_content_similarity(result, selected) > similarity_threshold:
                    is_diverse = False
                    break

            if is_diverse:
                filtered_results.append(result)

        return filtered_results

    def _calculate_content_similarity(self, result1: QueryResult, result2: QueryResult) -> float:
        """
        Calculate content similarity between two results.

        Args:
            result1: First result
            result2: Second result

        Returns:
            Similarity score (0.0 to 1.0)
        """
        words1 = set(result1.content.lower().split())
        words2 = set(result2.content.lower().split())

        if not words1 and not words2:
            return 1.0
        elif not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def _get_adaptive_criteria(self, request: QueryRequest) -> RankingCriteria:
        """
        Get adaptive ranking criteria based on query type and history.

        Args:
            request: Query request

        Returns:
            Adaptive ranking criteria
        """
        criteria = RankingCriteria()

        # Adjust weights based on query type
        if request.query_type == QueryType.SEMANTIC_SEARCH:
            criteria.relevance_weight = 0.6  # Higher relevance weight for semantic searches
            criteria.quality_weight = 0.2
            criteria.diversity_weight = 0.2
        elif request.query_type == QueryType.AGGREGATION:
            criteria.quality_weight = 0.5  # Higher quality weight for aggregations
            criteria.relevance_weight = 0.3
            criteria.freshness_weight = 0.2
        elif request.query_type == QueryType.RELATIONSHIP_SEARCH:
            criteria.relevance_weight = 0.4
            criteria.quality_weight = 0.3
            criteria.popularity_weight = 0.3  # Popular nodes likely have more relationships

        # Adjust based on user preferences (if available)
        if request.user_id:
            user_preferences = self._get_user_preferences(request.user_id)
            if user_preferences:
                criteria = self._apply_user_preferences(criteria, user_preferences)

        return criteria

    def _get_user_preferences(self, user_id: str) -> Optional[Dict[str, float]]:
        """
        Get user-specific ranking preferences.

        Args:
            user_id: User identifier

        Returns:
            User preferences or None if not available
        """
        # This would typically load from a user preferences database
        # For now, return None (use default criteria)
        return None

    def _apply_user_preferences(
        self, criteria: RankingCriteria, preferences: Dict[str, float]
    ) -> RankingCriteria:
        """
        Apply user preferences to ranking criteria.

        Args:
            criteria: Base criteria
            preferences: User preferences

        Returns:
            Modified criteria
        """
        # Apply preference modifiers
        if "relevance_preference" in preferences:
            criteria.relevance_weight *= preferences["relevance_preference"]
        if "quality_preference" in preferences:
            criteria.quality_weight *= preferences["quality_preference"]
        if "freshness_preference" in preferences:
            criteria.freshness_weight *= preferences["freshness_preference"]

        # Re-normalize weights
        total = (
            criteria.relevance_weight
            + criteria.quality_weight
            + criteria.freshness_weight
            + criteria.popularity_weight
            + criteria.diversity_weight
        )
        if total > 0:
            criteria.relevance_weight /= total
            criteria.quality_weight /= total
            criteria.freshness_weight /= total
            criteria.popularity_weight /= total
            criteria.diversity_weight /= total

        return criteria

    def _update_ranking_stats(self, request: QueryRequest, results: List[QueryResult]):
        """
        Update ranking statistics for learning and improvement.

        Args:
            request: Query request
            results: Ranked results
        """
        # Record query for pattern analysis
        self.query_history.append(
            {
                "query_type": request.query_type.value,
                "result_count": len(results),
                "top_score": results[0].combined_score if results else 0.0,
                "timestamp": time.time(),
            }
        )

        # Update result access statistics
        for i, result in enumerate(results[:10]):  # Only track top 10
            node_id = result.node_id
            if node_id not in self.result_stats:
                self.result_stats[node_id] = {
                    "access_count": 0,
                    "recent_access_count": 0,
                    "last_access": 0,
                    "ranking_positions": [],
                }

            stats = self.result_stats[node_id]
            stats["access_count"] += 1
            stats["ranking_positions"].append(i)
            stats["last_access"] = time.time()

            # Update recent access count (last 7 days)
            current_time = time.time()
            if current_time - stats["last_access"] < 7 * 24 * 3600:
                stats["recent_access_count"] += 1

        # Limit history size
        if len(self.query_history) > 1000:
            self.query_history = self.query_history[-1000:]

    def get_ranking_statistics(self) -> Dict[str, Any]:
        """
        Get ranking system statistics.

        Returns:
            Dictionary with ranking statistics
        """
        total_queries = len(self.query_history)

        if total_queries == 0:
            return {"total_queries": 0}

        # Calculate average scores
        avg_top_score = sum(q["top_score"] for q in self.query_history) / total_queries

        # Query type distribution
        query_types = defaultdict(int)
        for query in self.query_history:
            query_types[query["query_type"]] += 1

        return {
            "total_queries_ranked": total_queries,
            "average_top_score": avg_top_score,
            "query_type_distribution": dict(query_types),
            "total_results_tracked": len(self.result_stats),
            "cache_size": len(self.similarity_cache) + len(self.quality_cache),
        }
