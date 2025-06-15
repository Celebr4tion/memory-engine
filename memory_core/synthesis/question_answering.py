"""
Question Answering System for Knowledge Synthesis

Provides intelligent question answering capabilities that parse natural language
questions, identify relevant subgraphs, and synthesize comprehensive answers
from multiple knowledge sources.
"""

import logging
import re
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from memory_core.query.query_engine import AdvancedQueryEngine
from memory_core.query.query_types import QueryRequest, QueryType, QueryResponse
from memory_core.model.knowledge_node import KnowledgeNode
from memory_core.model.relationship import Relationship


class QuestionType(Enum):
    """Types of questions the system can handle."""

    FACTUAL = "factual"  # What is X? Who is Y?
    COMPARATIVE = "comparative"  # How does X compare to Y?
    CAUSAL = "causal"  # Why does X happen? What causes Y?
    PROCEDURAL = "procedural"  # How to do X? What steps for Y?
    TEMPORAL = "temporal"  # When did X happen? What happened after Y?
    DEFINITIONAL = "definitional"  # What does X mean? Define Y
    RELATIONAL = "relational"  # How is X related to Y?
    ANALYTICAL = "analytical"  # What patterns exist in X?


@dataclass
class QuestionContext:
    """Context information for question processing."""

    domain: Optional[str] = None
    time_frame: Optional[str] = None
    entities: List[str] = None
    constraints: Dict[str, Any] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class AnswerSource:
    """Source attribution for answer components."""

    node_id: str
    content_snippet: str
    relevance_score: float
    confidence_score: float
    node_type: str
    metadata: Dict[str, Any]


@dataclass
class SynthesizedAnswer:
    """Complete synthesized answer with attribution."""

    answer: str
    confidence_score: float
    sources: List[AnswerSource]
    reasoning: str
    question_type: QuestionType
    subgraphs_used: List[str]
    processing_time_ms: float
    alternative_perspectives: List[str] = None
    follow_up_questions: List[str] = None


@dataclass
class ParsedQuestion:
    """Parsed question structure."""

    original_question: str
    question_type: QuestionType
    entities: List[str]
    intent: str
    keywords: List[str]
    constraints: Dict[str, Any]
    confidence: float


class QuestionParser:
    """Natural language question parser."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Question type patterns
        self.type_patterns = {
            QuestionType.FACTUAL: [
                r"^what is|^who is|^where is|^when is|^which is",
                r"tell me about|information about|details about",
            ],
            QuestionType.COMPARATIVE: [
                r"compare|difference between|similar to|versus|vs",
                r"how does .* differ|how is .* different",
            ],
            QuestionType.CAUSAL: [
                r"why does|what causes|reason for|because of",
                r"how does .* affect|impact of|effect of",
            ],
            QuestionType.PROCEDURAL: [
                r"how to|steps to|process of|procedure for",
                r"how do I|how can I|method to",
            ],
            QuestionType.TEMPORAL: [
                r"when did|before|after|during|timeline",
                r"history of|chronology|sequence of events",
            ],
            QuestionType.DEFINITIONAL: [
                r"define|definition of|meaning of|what does .* mean",
                r"explain|explanation of",
            ],
            QuestionType.RELATIONAL: [
                r"relationship between|how is .* related|connection between",
                r"associated with|linked to|ties between",
            ],
            QuestionType.ANALYTICAL: [
                r"patterns in|trends in|analysis of|insights about",
                r"what can we learn|implications of",
            ],
        }

        # Entity extraction patterns
        self.entity_patterns = [
            r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*",  # Proper nouns
            r'"([^"]*)"',  # Quoted strings
            r"'([^']*)'",  # Single quoted strings
        ]

    def parse_question(self, question: str, context: QuestionContext = None) -> ParsedQuestion:
        """
        Parse a natural language question into structured components.

        Args:
            question: The question to parse
            context: Optional context information

        Returns:
            ParsedQuestion with extracted components
        """
        question_lower = question.lower().strip()

        # Detect question type
        question_type = self._detect_question_type(question_lower)

        # Extract entities
        entities = self._extract_entities(question)

        # Extract keywords
        keywords = self._extract_keywords(question_lower)

        # Determine intent
        intent = self._determine_intent(question_type, entities, keywords)

        # Extract constraints
        constraints = self._extract_constraints(question_lower, context)

        # Calculate confidence
        confidence = self._calculate_parsing_confidence(question_type, entities, keywords)

        return ParsedQuestion(
            original_question=question,
            question_type=question_type,
            entities=entities,
            intent=intent,
            keywords=keywords,
            constraints=constraints,
            confidence=confidence,
        )

    def _detect_question_type(self, question: str) -> QuestionType:
        """Detect the type of question based on patterns."""
        for q_type, patterns in self.type_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question, re.IGNORECASE):
                    return q_type

        # Default to factual if no pattern matches
        return QuestionType.FACTUAL

    def _extract_entities(self, question: str) -> List[str]:
        """Extract entities from the question."""
        entities = []

        for pattern in self.entity_patterns:
            matches = re.findall(pattern, question)
            if isinstance(matches[0], tuple) if matches else False:
                entities.extend([match[0] for match in matches])
            else:
                entities.extend(matches)

        # Remove duplicates and filter short entities
        entities = list(set([e for e in entities if len(e) > 2]))

        return entities

    def _extract_keywords(self, question: str) -> List[str]:
        """Extract important keywords from the question."""
        # Remove stop words and common question words
        stop_words = {
            "what",
            "who",
            "where",
            "when",
            "why",
            "how",
            "is",
            "are",
            "was",
            "were",
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "can",
            "could",
            "would",
            "should",
            "will",
            "do",
            "does",
        }

        words = re.findall(r"\b\w+\b", question.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]

        return keywords

    def _determine_intent(
        self, question_type: QuestionType, entities: List[str], keywords: List[str]
    ) -> str:
        """Determine the intent of the question."""
        if question_type == QuestionType.FACTUAL:
            if entities:
                return f"Get factual information about {', '.join(entities[:2])}"
            else:
                return "Get factual information"
        elif question_type == QuestionType.COMPARATIVE:
            return f"Compare entities: {', '.join(entities[:3])}"
        elif question_type == QuestionType.CAUSAL:
            return f"Explain causation involving {', '.join(entities[:2])}"
        elif question_type == QuestionType.PROCEDURAL:
            return f"Provide procedural information for {', '.join(keywords[:2])}"
        else:
            return f"Process {question_type.value} question about {', '.join(entities[:2])}"

    def _extract_constraints(
        self, question: str, context: QuestionContext = None
    ) -> Dict[str, Any]:
        """Extract constraints from the question and context."""
        constraints = {}

        # Time constraints
        time_patterns = [r"in (\d{4})", r"during ([^,]+)", r"before ([^,]+)", r"after ([^,]+)"]
        for pattern in time_patterns:
            match = re.search(pattern, question)
            if match:
                constraints["time_filter"] = match.group(1)
                break

        # Domain constraints from context
        if context and context.domain:
            constraints["domain"] = context.domain

        # Quantity constraints
        quantity_match = re.search(r"(\d+)\s+(most|top|best)", question)
        if quantity_match:
            constraints["limit"] = int(quantity_match.group(1))

        return constraints

    def _calculate_parsing_confidence(
        self, question_type: QuestionType, entities: List[str], keywords: List[str]
    ) -> float:
        """Calculate confidence in the parsing results."""
        confidence = 0.5  # Base confidence

        # Boost confidence based on entities found
        if entities:
            confidence += 0.2 * min(len(entities) / 3, 1)

        # Boost confidence based on keywords
        if keywords:
            confidence += 0.1 * min(len(keywords) / 5, 1)

        # Boost confidence for clear question types
        if question_type != QuestionType.FACTUAL:  # Non-default type
            confidence += 0.2

        return min(confidence, 1.0)


class SubgraphIdentifier:
    """Identifies relevant subgraphs for answering questions."""

    def __init__(self, query_engine: AdvancedQueryEngine):
        self.query_engine = query_engine
        self.logger = logging.getLogger(__name__)

    def identify_relevant_subgraphs(
        self, parsed_question: ParsedQuestion, max_depth: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Identify subgraphs relevant to answering the question.

        Args:
            parsed_question: Parsed question structure
            max_depth: Maximum depth for subgraph exploration

        Returns:
            List of relevant subgraph descriptors
        """
        subgraphs = []

        try:
            # For each entity, find its neighborhood
            for entity in parsed_question.entities:
                subgraph = self._explore_entity_neighborhood(entity, max_depth)
                if subgraph:
                    subgraphs.append(subgraph)

            # For keyword-based searches when no entities found
            if not subgraphs and parsed_question.keywords:
                keyword_subgraph = self._explore_keyword_space(parsed_question.keywords, max_depth)
                if keyword_subgraph:
                    subgraphs.append(keyword_subgraph)

            # Merge overlapping subgraphs
            subgraphs = self._merge_overlapping_subgraphs(subgraphs)

            return subgraphs

        except Exception as e:
            self.logger.error(f"Error identifying subgraphs: {e}")
            return []

    def _explore_entity_neighborhood(self, entity: str, max_depth: int) -> Optional[Dict[str, Any]]:
        """Explore the neighborhood around an entity."""
        try:
            # Search for nodes matching the entity
            request = QueryRequest(
                query=entity,
                query_type=QueryType.SEMANTIC_SEARCH,
                limit=10,
                include_relationships=True,
                max_depth=max_depth,
                similarity_threshold=0.7,
            )

            response = self.query_engine.query(request)

            if response.results:
                # Extract node IDs and relationships
                node_ids = [result.node_id for result in response.results]
                all_relationships = []

                for result in response.results:
                    if result.relationships:
                        all_relationships.extend(result.relationships)

                return {
                    "entity": entity,
                    "center_nodes": node_ids[:5],  # Top 5 most relevant
                    "all_nodes": node_ids,
                    "relationships": all_relationships,
                    "relevance_scores": [result.relevance_score for result in response.results],
                }

            return None

        except Exception as e:
            self.logger.error(f"Error exploring entity {entity}: {e}")
            return None

    def _explore_keyword_space(
        self, keywords: List[str], max_depth: int
    ) -> Optional[Dict[str, Any]]:
        """Explore the space around keywords when no entities are found."""
        try:
            # Combine keywords into a search query
            query = " ".join(keywords[:3])  # Use top 3 keywords

            request = QueryRequest(
                query=query,
                query_type=QueryType.SEMANTIC_SEARCH,
                limit=15,
                include_relationships=True,
                max_depth=max_depth,
                similarity_threshold=0.6,
            )

            response = self.query_engine.query(request)

            if response.results:
                node_ids = [result.node_id for result in response.results]
                all_relationships = []

                for result in response.results:
                    if result.relationships:
                        all_relationships.extend(result.relationships)

                return {
                    "keywords": keywords,
                    "center_nodes": node_ids[:8],
                    "all_nodes": node_ids,
                    "relationships": all_relationships,
                    "relevance_scores": [result.relevance_score for result in response.results],
                }

            return None

        except Exception as e:
            self.logger.error(f"Error exploring keyword space: {e}")
            return None

    def _merge_overlapping_subgraphs(self, subgraphs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge subgraphs that have significant overlap."""
        if len(subgraphs) <= 1:
            return subgraphs

        merged = []

        for subgraph in subgraphs:
            merged_with_existing = False

            for existing in merged:
                # Check overlap
                overlap = set(subgraph["all_nodes"]) & set(existing["all_nodes"])
                overlap_ratio = len(overlap) / min(
                    len(subgraph["all_nodes"]), len(existing["all_nodes"])
                )

                if overlap_ratio > 0.3:  # 30% overlap threshold
                    # Merge subgraphs
                    existing["all_nodes"] = list(set(existing["all_nodes"] + subgraph["all_nodes"]))
                    existing["center_nodes"] = list(
                        set(existing["center_nodes"] + subgraph["center_nodes"])
                    )
                    existing["relationships"].extend(subgraph["relationships"])

                    # Merge entity/keyword info
                    if "entity" in subgraph:
                        existing.setdefault("entities", []).append(subgraph["entity"])
                    if "keywords" in subgraph:
                        existing.setdefault("keywords", []).extend(subgraph["keywords"])

                    merged_with_existing = True
                    break

            if not merged_with_existing:
                merged.append(subgraph)

        return merged


class AnswerSynthesizer:
    """Synthesizes answers from multiple knowledge sources."""

    def __init__(self, query_engine: AdvancedQueryEngine):
        self.query_engine = query_engine
        self.logger = logging.getLogger(__name__)

    def synthesize_answer(
        self,
        parsed_question: ParsedQuestion,
        subgraphs: List[Dict[str, Any]],
        context: QuestionContext = None,
    ) -> SynthesizedAnswer:
        """
        Synthesize a comprehensive answer from identified subgraphs.

        Args:
            parsed_question: Parsed question structure
            subgraphs: Relevant subgraphs
            context: Optional context information

        Returns:
            SynthesizedAnswer with complete response
        """
        start_time = time.time()

        try:
            # Gather evidence from subgraphs
            evidence = self._gather_evidence(subgraphs, parsed_question)

            # Generate answer based on question type
            answer = self._generate_typed_answer(parsed_question, evidence, context)

            # Calculate confidence score
            confidence = self._calculate_answer_confidence(evidence, parsed_question)

            # Create source attributions
            sources = self._create_source_attributions(evidence)

            # Generate reasoning explanation
            reasoning = self._generate_reasoning(parsed_question, evidence, answer)

            # Generate alternative perspectives
            alternatives = self._generate_alternative_perspectives(parsed_question, evidence)

            # Generate follow-up questions
            follow_ups = self._generate_follow_up_questions(parsed_question, evidence)

            processing_time = (time.time() - start_time) * 1000

            return SynthesizedAnswer(
                answer=answer,
                confidence_score=confidence,
                sources=sources,
                reasoning=reasoning,
                question_type=parsed_question.question_type,
                subgraphs_used=[sg.get("entity", "keyword_space") for sg in subgraphs],
                processing_time_ms=processing_time,
                alternative_perspectives=alternatives,
                follow_up_questions=follow_ups,
            )

        except Exception as e:
            self.logger.error(f"Error synthesizing answer: {e}")
            return self._create_error_answer(parsed_question, str(e), start_time)

    def _gather_evidence(
        self, subgraphs: List[Dict[str, Any]], parsed_question: ParsedQuestion
    ) -> List[Dict[str, Any]]:
        """Gather evidence from subgraphs relevant to the question."""
        evidence = []

        for subgraph in subgraphs:
            for node_id in subgraph["center_nodes"]:
                try:
                    # Get node details
                    request = QueryRequest(
                        query=node_id, query_type=QueryType.GRAPH_PATTERN, limit=1
                    )

                    response = self.query_engine.query(request)

                    if response.results:
                        result = response.results[0]

                        # Calculate relevance to question
                        relevance = self._calculate_evidence_relevance(result, parsed_question)

                        evidence.append(
                            {
                                "node_id": node_id,
                                "content": result.content,
                                "node_type": result.node_type,
                                "metadata": result.metadata,
                                "relevance_score": relevance,
                                "relationships": result.relationships or [],
                            }
                        )

                except Exception as e:
                    self.logger.warning(f"Error gathering evidence for node {node_id}: {e}")
                    continue

        # Sort by relevance
        evidence.sort(key=lambda x: x["relevance_score"], reverse=True)

        return evidence[:20]  # Limit to top 20 pieces of evidence

    def _calculate_evidence_relevance(self, result, parsed_question: ParsedQuestion) -> float:
        """Calculate how relevant a piece of evidence is to the question."""
        relevance = 0.0
        content_lower = result.content.lower()

        # Check for entity mentions
        for entity in parsed_question.entities:
            if entity.lower() in content_lower:
                relevance += 0.3

        # Check for keyword matches
        for keyword in parsed_question.keywords:
            if keyword in content_lower:
                relevance += 0.1

        # Boost for certain node types based on question type
        if parsed_question.question_type == QuestionType.DEFINITIONAL:
            if result.node_type in ["definition", "concept", "term"]:
                relevance += 0.2
        elif parsed_question.question_type == QuestionType.PROCEDURAL:
            if result.node_type in ["process", "procedure", "step"]:
                relevance += 0.2

        # Use original relevance score if available
        if hasattr(result, "relevance_score") and result.relevance_score:
            relevance += result.relevance_score * 0.5

        return min(relevance, 1.0)

    def _generate_typed_answer(
        self,
        parsed_question: ParsedQuestion,
        evidence: List[Dict[str, Any]],
        context: QuestionContext = None,
    ) -> str:
        """Generate an answer based on the question type."""
        if not evidence:
            return "I don't have enough information to answer this question."

        if parsed_question.question_type == QuestionType.FACTUAL:
            return self._generate_factual_answer(evidence, parsed_question)
        elif parsed_question.question_type == QuestionType.COMPARATIVE:
            return self._generate_comparative_answer(evidence, parsed_question)
        elif parsed_question.question_type == QuestionType.CAUSAL:
            return self._generate_causal_answer(evidence, parsed_question)
        elif parsed_question.question_type == QuestionType.PROCEDURAL:
            return self._generate_procedural_answer(evidence, parsed_question)
        elif parsed_question.question_type == QuestionType.DEFINITIONAL:
            return self._generate_definitional_answer(evidence, parsed_question)
        else:
            # Default to factual approach
            return self._generate_factual_answer(evidence, parsed_question)

    def _generate_factual_answer(
        self, evidence: List[Dict[str, Any]], parsed_question: ParsedQuestion
    ) -> str:
        """Generate a factual answer."""
        # Take the most relevant pieces of evidence
        top_evidence = evidence[:3]

        answer_parts = []
        for item in top_evidence:
            # Extract relevant sentences
            content = item["content"]
            relevant_sentences = self._extract_relevant_sentences(
                content, parsed_question.entities + parsed_question.keywords
            )

            if relevant_sentences:
                answer_parts.extend(relevant_sentences[:2])  # Top 2 sentences per source

        if answer_parts:
            return " ".join(answer_parts)
        else:
            return f"Based on the available information: {evidence[0]['content'][:200]}..."

    def _generate_comparative_answer(
        self, evidence: List[Dict[str, Any]], parsed_question: ParsedQuestion
    ) -> str:
        """Generate a comparative answer."""
        entities = parsed_question.entities
        if len(entities) < 2:
            return self._generate_factual_answer(evidence, parsed_question)

        # Group evidence by entities
        entity_evidence = {}
        for item in evidence:
            content_lower = item["content"].lower()
            for entity in entities:
                if entity.lower() in content_lower:
                    entity_evidence.setdefault(entity, []).append(item)

        comparison_parts = []
        for entity, items in entity_evidence.items():
            if items:
                relevant_content = items[0]["content"][:150]
                comparison_parts.append(f"Regarding {entity}: {relevant_content}")

        if comparison_parts:
            return " ".join(comparison_parts)
        else:
            return self._generate_factual_answer(evidence, parsed_question)

    def _generate_causal_answer(
        self, evidence: List[Dict[str, Any]], parsed_question: ParsedQuestion
    ) -> str:
        """Generate a causal answer."""
        # Look for causal relationships in the evidence
        causal_evidence = []
        causal_keywords = ["because", "due to", "causes", "leads to", "results in", "reason"]

        for item in evidence:
            content_lower = item["content"].lower()
            if any(keyword in content_lower for keyword in causal_keywords):
                causal_evidence.append(item)

        if causal_evidence:
            # Use causal evidence preferentially
            relevant_content = causal_evidence[0]["content"]
            return self._extract_causal_explanation(relevant_content, parsed_question)
        else:
            return self._generate_factual_answer(evidence, parsed_question)

    def _generate_procedural_answer(
        self, evidence: List[Dict[str, Any]], parsed_question: ParsedQuestion
    ) -> str:
        """Generate a procedural answer."""
        # Look for step-by-step information
        procedural_evidence = []
        procedural_keywords = ["step", "first", "then", "next", "finally", "process", "method"]

        for item in evidence:
            content_lower = item["content"].lower()
            if any(keyword in content_lower for keyword in procedural_keywords):
                procedural_evidence.append(item)

        if procedural_evidence:
            # Extract and organize steps
            steps = self._extract_procedural_steps(procedural_evidence)
            if steps:
                return "Here's the process: " + " ".join(steps)

        return self._generate_factual_answer(evidence, parsed_question)

    def _generate_definitional_answer(
        self, evidence: List[Dict[str, Any]], parsed_question: ParsedQuestion
    ) -> str:
        """Generate a definitional answer."""
        # Look for definitional content
        for item in evidence:
            content = item["content"]

            # Look for definition patterns
            if re.search(r"is defined as|means|refers to|is a type of", content, re.IGNORECASE):
                return content[:300] + "..." if len(content) > 300 else content

        # Fall back to most relevant content
        if evidence:
            return (
                evidence[0]["content"][:300] + "..."
                if len(evidence[0]["content"]) > 300
                else evidence[0]["content"]
            )

        return "I don't have a clear definition for this term."

    def _extract_relevant_sentences(self, content: str, keywords: List[str]) -> List[str]:
        """Extract sentences most relevant to the keywords."""
        sentences = re.split(r"[.!?]+", content)
        relevant_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue

            relevance_score = 0
            sentence_lower = sentence.lower()

            for keyword in keywords:
                if keyword.lower() in sentence_lower:
                    relevance_score += 1

            if relevance_score > 0:
                relevant_sentences.append((sentence, relevance_score))

        # Sort by relevance and return top sentences
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        return [sentence for sentence, _ in relevant_sentences[:3]]

    def _extract_causal_explanation(self, content: str, parsed_question: ParsedQuestion) -> str:
        """Extract causal explanation from content."""
        # Look for sentences with causal indicators
        sentences = re.split(r"[.!?]+", content)
        causal_sentences = []

        causal_patterns = [
            r"because\s+",
            r"due to\s+",
            r"caused by\s+",
            r"results from\s+",
            r"leads to\s+",
            r"reason.*is\s+",
        ]

        for sentence in sentences:
            sentence = sentence.strip()
            for pattern in causal_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    causal_sentences.append(sentence)
                    break

        if causal_sentences:
            return causal_sentences[0]
        else:
            return content[:200] + "..."

    def _extract_procedural_steps(self, evidence: List[Dict[str, Any]]) -> List[str]:
        """Extract procedural steps from evidence."""
        steps = []

        for item in evidence:
            content = item["content"]

            # Look for numbered or ordered steps
            step_patterns = [
                r"(\d+[\.\)]\s*[^.]+)",
                r"(first[^.]+)",
                r"(then[^.]+)",
                r"(next[^.]+)",
                r"(finally[^.]+)",
            ]

            for pattern in step_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                steps.extend(matches)

        return steps[:5]  # Limit to 5 steps

    def _calculate_answer_confidence(
        self, evidence: List[Dict[str, Any]], parsed_question: ParsedQuestion
    ) -> float:
        """Calculate confidence in the synthesized answer."""
        if not evidence:
            return 0.1

        # Base confidence from evidence quality
        avg_relevance = sum(item["relevance_score"] for item in evidence) / len(evidence)
        confidence = avg_relevance * 0.6

        # Boost for multiple corroborating sources
        if len(evidence) >= 3:
            confidence += 0.2
        elif len(evidence) >= 2:
            confidence += 0.1

        # Boost for high-quality evidence
        high_quality_count = sum(1 for item in evidence if item["relevance_score"] > 0.7)
        confidence += (high_quality_count / len(evidence)) * 0.2

        return min(confidence, 0.95)  # Cap at 95%

    def _create_source_attributions(self, evidence: List[Dict[str, Any]]) -> List[AnswerSource]:
        """Create source attributions for the answer."""
        sources = []

        for item in evidence[:5]:  # Top 5 sources
            source = AnswerSource(
                node_id=item["node_id"],
                content_snippet=(
                    item["content"][:200] + "..." if len(item["content"]) > 200 else item["content"]
                ),
                relevance_score=item["relevance_score"],
                confidence_score=min(item["relevance_score"] + 0.2, 1.0),
                node_type=item["node_type"] or "unknown",
                metadata=item["metadata"],
            )
            sources.append(source)

        return sources

    def _generate_reasoning(
        self, parsed_question: ParsedQuestion, evidence: List[Dict[str, Any]], answer: str
    ) -> str:
        """Generate reasoning explanation for the answer."""
        reasoning_parts = [f"To answer this {parsed_question.question_type.value} question"]

        if parsed_question.entities:
            reasoning_parts.append(f"about {', '.join(parsed_question.entities[:2])}")

        reasoning_parts.append(f"I analyzed {len(evidence)} relevant knowledge sources")

        if evidence:
            high_confidence_count = sum(1 for item in evidence if item["relevance_score"] > 0.7)
            if high_confidence_count > 0:
                reasoning_parts.append(f"with {high_confidence_count} high-confidence matches")

        return ", ".join(reasoning_parts) + "."

    def _generate_alternative_perspectives(
        self, parsed_question: ParsedQuestion, evidence: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate alternative perspectives on the question."""
        if len(evidence) < 2:
            return []

        perspectives = []

        # Look for contrasting information
        for i, item in enumerate(evidence[:3]):
            content = item["content"]

            # Check for perspective indicators
            perspective_indicators = [
                "however",
                "on the other hand",
                "alternatively",
                "some argue",
                "others believe",
            ]

            for indicator in perspective_indicators:
                if indicator in content.lower():
                    # Extract the alternative perspective
                    sentences = re.split(r"[.!?]+", content)
                    for sentence in sentences:
                        if indicator in sentence.lower():
                            perspectives.append(sentence.strip())
                            break
                    break

        return perspectives[:2]  # Limit to 2 alternative perspectives

    def _generate_follow_up_questions(
        self, parsed_question: ParsedQuestion, evidence: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate relevant follow-up questions."""
        follow_ups = []

        # Generate follow-ups based on question type
        if parsed_question.question_type == QuestionType.FACTUAL:
            if parsed_question.entities:
                entity = parsed_question.entities[0]
                follow_ups.append(f"How does {entity} relate to other concepts?")
                follow_ups.append(f"What are the implications of {entity}?")

        elif parsed_question.question_type == QuestionType.CAUSAL:
            follow_ups.append("What are the broader implications of this relationship?")
            follow_ups.append("Are there other contributing factors?")

        elif parsed_question.question_type == QuestionType.COMPARATIVE:
            follow_ups.append("What are the key differences between these concepts?")
            follow_ups.append("Which approach is more effective?")

        # Look for questions in the evidence
        for item in evidence[:2]:
            content = item["content"]
            questions = re.findall(r"[^.!?]*\?", content)
            for question in questions[:1]:  # One question per source
                if len(question.strip()) > 10:
                    follow_ups.append(question.strip())

        return follow_ups[:3]  # Limit to 3 follow-up questions

    def _create_error_answer(
        self, parsed_question: ParsedQuestion, error_msg: str, start_time: float
    ) -> SynthesizedAnswer:
        """Create an error answer when synthesis fails."""
        return SynthesizedAnswer(
            answer=f"I encountered an error while trying to answer your question: {error_msg}",
            confidence_score=0.0,
            sources=[],
            reasoning="Answer synthesis failed due to an internal error.",
            question_type=parsed_question.question_type,
            subgraphs_used=[],
            processing_time_ms=(time.time() - start_time) * 1000,
            alternative_perspectives=[],
            follow_up_questions=[],
        )


class QuestionAnsweringSystem:
    """
    Complete Question Answering System.

    Integrates natural language parsing, subgraph identification,
    and answer synthesis to provide comprehensive answers to questions.
    """

    def __init__(self, query_engine: AdvancedQueryEngine):
        """
        Initialize the Question Answering System.

        Args:
            query_engine: Query engine for accessing knowledge graph
        """
        self.query_engine = query_engine
        self.question_parser = QuestionParser()
        self.subgraph_identifier = SubgraphIdentifier(query_engine)
        self.answer_synthesizer = AnswerSynthesizer(query_engine)
        self.logger = logging.getLogger(__name__)

        # Statistics
        self.stats = {
            "questions_processed": 0,
            "avg_processing_time_ms": 0.0,
            "avg_confidence_score": 0.0,
            "question_type_distribution": {},
            "error_count": 0,
        }

    def answer_question(self, question: str, context: QuestionContext = None) -> SynthesizedAnswer:
        """
        Answer a natural language question.

        Args:
            question: The question to answer
            context: Optional context information

        Returns:
            SynthesizedAnswer with complete response
        """
        start_time = time.time()

        try:
            self.logger.info(f"Processing question: {question[:100]}...")

            # Step 1: Parse the question
            parsed_question = self.question_parser.parse_question(question, context)

            self.logger.debug(
                f"Question type: {parsed_question.question_type}, "
                f"Entities: {parsed_question.entities}, "
                f"Confidence: {parsed_question.confidence}"
            )

            # Step 2: Identify relevant subgraphs
            subgraphs = self.subgraph_identifier.identify_relevant_subgraphs(
                parsed_question, max_depth=2
            )

            self.logger.debug(f"Identified {len(subgraphs)} relevant subgraphs")

            # Step 3: Synthesize answer
            answer = self.answer_synthesizer.synthesize_answer(parsed_question, subgraphs, context)

            # Update statistics
            self._update_statistics(answer, start_time)

            self.logger.info(f"Question answered with confidence {answer.confidence_score:.2f}")
            return answer

        except Exception as e:
            self.logger.error(f"Error answering question: {e}")
            self.stats["error_count"] += 1

            # Return error answer
            return SynthesizedAnswer(
                answer=f"I'm sorry, I encountered an error while processing your question: {str(e)}",
                confidence_score=0.0,
                sources=[],
                reasoning="Internal error during question processing.",
                question_type=QuestionType.FACTUAL,
                subgraphs_used=[],
                processing_time_ms=(time.time() - start_time) * 1000,
                alternative_perspectives=[],
                follow_up_questions=[],
            )

    def _update_statistics(self, answer: SynthesizedAnswer, start_time: float):
        """Update system statistics."""
        self.stats["questions_processed"] += 1

        # Update average processing time
        total_time = self.stats["avg_processing_time_ms"] * (self.stats["questions_processed"] - 1)
        self.stats["avg_processing_time_ms"] = (
            total_time + answer.processing_time_ms
        ) / self.stats["questions_processed"]

        # Update average confidence
        total_confidence = self.stats["avg_confidence_score"] * (
            self.stats["questions_processed"] - 1
        )
        self.stats["avg_confidence_score"] = (
            total_confidence + answer.confidence_score
        ) / self.stats["questions_processed"]

        # Update question type distribution
        q_type = answer.question_type.value
        self.stats["question_type_distribution"][q_type] = (
            self.stats["question_type_distribution"].get(q_type, 0) + 1
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get system statistics.

        Returns:
            Dictionary with comprehensive statistics
        """
        return {
            "question_answering": self.stats.copy(),
            "query_engine": self.query_engine.get_statistics(),
        }
