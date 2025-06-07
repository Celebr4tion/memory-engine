"""
Knowledge Synthesis Engine - Main Integration Module

Provides a unified interface for all knowledge synthesis capabilities,
integrating question answering, insight discovery, and perspective analysis
into a comprehensive synthesis system.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from memory_core.query.query_engine import AdvancedQueryEngine
from memory_core.synthesis.question_answering import (
    QuestionAnsweringSystem, QuestionContext, SynthesizedAnswer
)
from memory_core.synthesis.insight_discovery import (
    InsightDiscoveryEngine, PatternType, TrendType, AnomalyType, InsightReport
)
from memory_core.synthesis.perspective_analysis import (
    PerspectiveAnalysisEngine, PerspectiveType, PerspectiveAnalysisReport
)


class SynthesisTaskType(Enum):
    """Types of synthesis tasks the engine can perform."""
    QUESTION_ANSWERING = "question_answering"
    INSIGHT_DISCOVERY = "insight_discovery"
    PERSPECTIVE_ANALYSIS = "perspective_analysis"
    COMPREHENSIVE_SYNTHESIS = "comprehensive_synthesis"


class SynthesisMode(Enum):
    """Synthesis processing modes."""
    FAST = "fast"  # Quick synthesis with basic analysis
    BALANCED = "balanced"  # Balanced speed and depth
    COMPREHENSIVE = "comprehensive"  # Deep analysis with all features
    CUSTOM = "custom"  # Custom configuration


@dataclass
class SynthesisRequest:
    """Request for knowledge synthesis."""
    task_type: SynthesisTaskType
    query: str
    context: Optional[QuestionContext] = None
    mode: SynthesisMode = SynthesisMode.BALANCED
    
    # Specific configurations
    perspective_types: Optional[List[PerspectiveType]] = None
    pattern_types: Optional[List[PatternType]] = None
    trend_types: Optional[List[TrendType]] = None
    anomaly_types: Optional[List[AnomalyType]] = None
    
    # Processing options
    include_temporal_analysis: bool = True
    include_stakeholder_analysis: bool = True
    min_confidence_threshold: float = 0.3
    max_processing_time_ms: Optional[int] = None


@dataclass
class ComprehensiveSynthesisResult:
    """Result from comprehensive synthesis combining all engines."""
    request: SynthesisRequest
    
    # Individual results
    question_answer: Optional[SynthesizedAnswer] = None
    insights: Optional[InsightReport] = None
    perspectives: Optional[PerspectiveAnalysisReport] = None
    
    # Integrated synthesis
    executive_summary: str = ""
    key_findings: List[str] = None
    confidence_assessment: Dict[str, float] = None
    actionable_recommendations: List[str] = None
    
    # Meta information
    synthesis_confidence: float = 0.0
    processing_time_ms: float = 0.0
    sources_analyzed: int = 0
    cross_validation_score: float = 0.0
    
    # Follow-up suggestions
    suggested_follow_ups: List[str] = None
    related_topics: List[str] = None


class KnowledgeSynthesisEngine:
    """
    Main Knowledge Synthesis Engine.
    
    Integrates question answering, insight discovery, and perspective analysis
    to provide comprehensive knowledge synthesis capabilities.
    """
    
    def __init__(self, query_engine: AdvancedQueryEngine):
        """
        Initialize the Knowledge Synthesis Engine.
        
        Args:
            query_engine: Query engine for accessing knowledge graph
        """
        self.query_engine = query_engine
        
        # Initialize synthesis engines
        self.question_answering = QuestionAnsweringSystem(query_engine)
        self.insight_discovery = InsightDiscoveryEngine(query_engine)
        self.perspective_analysis = PerspectiveAnalysisEngine(query_engine)
        
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.stats = {
            'total_syntheses': 0,
            'synthesis_by_type': {},
            'synthesis_by_mode': {},
            'avg_processing_time_ms': 0.0,
            'avg_confidence_score': 0.0,
            'cross_validation_scores': []
        }
        
        # Mode configurations
        self.mode_configs = {
            SynthesisMode.FAST: {
                'question_answering': True,
                'insight_discovery': False,
                'perspective_analysis': False,
                'max_patterns': 3,
                'max_perspectives': 2,
                'include_temporal': False,
                'include_stakeholder': False
            },
            SynthesisMode.BALANCED: {
                'question_answering': True,
                'insight_discovery': True,
                'perspective_analysis': True,
                'max_patterns': 5,
                'max_perspectives': 4,
                'include_temporal': True,
                'include_stakeholder': False
            },
            SynthesisMode.COMPREHENSIVE: {
                'question_answering': True,
                'insight_discovery': True,
                'perspective_analysis': True,
                'max_patterns': 10,
                'max_perspectives': 8,
                'include_temporal': True,
                'include_stakeholder': True
            }
        }
    
    def synthesize(self, request: SynthesisRequest) -> Union[SynthesizedAnswer, InsightReport, 
                                                          PerspectiveAnalysisReport, 
                                                          ComprehensiveSynthesisResult]:
        """
        Perform knowledge synthesis based on request.
        
        Args:
            request: Synthesis request with task specification
            
        Returns:
            Synthesis result (type depends on task_type)
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting {request.task_type.value} synthesis: {request.query[:100]}...")
            
            # Route to specific synthesis type
            if request.task_type == SynthesisTaskType.QUESTION_ANSWERING:
                result = self._perform_question_answering(request)
            elif request.task_type == SynthesisTaskType.INSIGHT_DISCOVERY:
                result = self._perform_insight_discovery(request)
            elif request.task_type == SynthesisTaskType.PERSPECTIVE_ANALYSIS:
                result = self._perform_perspective_analysis(request)
            elif request.task_type == SynthesisTaskType.COMPREHENSIVE_SYNTHESIS:
                result = self._perform_comprehensive_synthesis(request)
            else:
                raise ValueError(f"Unsupported synthesis task type: {request.task_type}")
            
            # Update statistics
            processing_time = (time.time() - start_time) * 1000
            self._update_statistics(request, result, processing_time)
            
            self.logger.info(f"Synthesis completed in {processing_time:.2f}ms")
            return result
            
        except Exception as e:
            self.logger.error(f"Synthesis failed: {e}")
            return self._create_error_result(request, str(e), start_time)
    
    def _perform_question_answering(self, request: SynthesisRequest) -> SynthesizedAnswer:
        """Perform question answering synthesis."""
        try:
            return self.question_answering.answer_question(
                question=request.query,
                context=request.context
            )
        except Exception as e:
            self.logger.error(f"Question answering failed: {e}")
            # Return basic error answer
            from memory_core.synthesis.question_answering import SynthesizedAnswer, QuestionType
            return SynthesizedAnswer(
                answer=f"I'm sorry, I couldn't answer your question due to an error: {str(e)}",
                confidence_score=0.0,
                sources=[],
                reasoning="Error during question processing",
                question_type=QuestionType.FACTUAL,
                subgraphs_used=[],
                processing_time_ms=0.0
            )
    
    def _perform_insight_discovery(self, request: SynthesisRequest) -> InsightReport:
        """Perform insight discovery synthesis."""
        try:
            # Determine domain from query
            domain = self._extract_domain_from_query(request.query)
            
            # Get configuration
            config = self.mode_configs.get(request.mode, self.mode_configs[SynthesisMode.BALANCED])
            
            return self.insight_discovery.discover_insights(
                domain=domain,
                pattern_types=request.pattern_types,
                trend_types=request.trend_types,
                anomaly_types=request.anomaly_types
            )
        except Exception as e:
            self.logger.error(f"Insight discovery failed: {e}")
            # Return empty insight report
            from memory_core.synthesis.insight_discovery import InsightReport
            return InsightReport(
                patterns=[],
                trends=[],
                anomalies=[],
                summary=f"Insight discovery failed: {str(e)}",
                discovery_time_ms=0.0,
                total_entities_analyzed=0,
                confidence_distribution={},
                recommendations=[]
            )
    
    def _perform_perspective_analysis(self, request: SynthesisRequest) -> PerspectiveAnalysisReport:
        """Perform perspective analysis synthesis."""
        try:
            # Extract topic from query
            topic = self._extract_topic_from_query(request.query)
            
            # Get configuration
            config = self.mode_configs.get(request.mode, self.mode_configs[SynthesisMode.BALANCED])
            
            return self.perspective_analysis.analyze_perspectives(
                topic=topic,
                perspective_types=request.perspective_types,
                include_temporal_analysis=request.include_temporal_analysis and config['include_temporal'],
                include_stakeholder_analysis=request.include_stakeholder_analysis and config['include_stakeholder']
            )
        except Exception as e:
            self.logger.error(f"Perspective analysis failed: {e}")
            # Return empty perspective report
            from memory_core.synthesis.perspective_analysis import PerspectiveAnalysisReport, ConsensusLevel
            return PerspectiveAnalysisReport(
                topic=request.query,
                perspectives=[],
                comparisons=[],
                stakeholder_analysis=[],
                temporal_evolution=None,
                overall_consensus=ConsensusLevel.NO_CONSENSUS,
                key_insights=[f"Perspective analysis failed: {str(e)}"],
                recommendations=[],
                analysis_confidence=0.0,
                processing_time_ms=0.0
            )
    
    def _perform_comprehensive_synthesis(self, request: SynthesisRequest) -> ComprehensiveSynthesisResult:
        """Perform comprehensive synthesis using all engines."""
        start_time = time.time()
        
        try:
            # Get configuration
            config = self.mode_configs.get(request.mode, self.mode_configs[SynthesisMode.COMPREHENSIVE])
            
            # Initialize result
            result = ComprehensiveSynthesisResult(
                request=request,
                key_findings=[],
                confidence_assessment={},
                actionable_recommendations=[],
                suggested_follow_ups=[],
                related_topics=[]
            )
            
            # Perform individual syntheses
            if config['question_answering']:
                self.logger.debug("Performing question answering...")
                qa_request = SynthesisRequest(
                    task_type=SynthesisTaskType.QUESTION_ANSWERING,
                    query=request.query,
                    context=request.context,
                    mode=request.mode
                )
                result.question_answer = self._perform_question_answering(qa_request)
            
            if config['insight_discovery']:
                self.logger.debug("Performing insight discovery...")
                insight_request = SynthesisRequest(
                    task_type=SynthesisTaskType.INSIGHT_DISCOVERY,
                    query=request.query,
                    mode=request.mode,
                    pattern_types=request.pattern_types,
                    trend_types=request.trend_types,
                    anomaly_types=request.anomaly_types
                )
                result.insights = self._perform_insight_discovery(insight_request)
            
            if config['perspective_analysis']:
                self.logger.debug("Performing perspective analysis...")
                perspective_request = SynthesisRequest(
                    task_type=SynthesisTaskType.PERSPECTIVE_ANALYSIS,
                    query=request.query,
                    mode=request.mode,
                    perspective_types=request.perspective_types,
                    include_temporal_analysis=request.include_temporal_analysis and config['include_temporal'],
                    include_stakeholder_analysis=request.include_stakeholder_analysis and config['include_stakeholder']
                )
                result.perspectives = self._perform_perspective_analysis(perspective_request)
            
            # Integrate results
            self.logger.debug("Integrating synthesis results...")
            self._integrate_synthesis_results(result)
            
            # Calculate meta information
            result.processing_time_ms = (time.time() - start_time) * 1000
            result.synthesis_confidence = self._calculate_synthesis_confidence(result)
            result.sources_analyzed = self._count_unique_sources(result)
            result.cross_validation_score = self._calculate_cross_validation_score(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Comprehensive synthesis failed: {e}")
            return ComprehensiveSynthesisResult(
                request=request,
                executive_summary=f"Comprehensive synthesis failed: {str(e)}",
                key_findings=[],
                confidence_assessment={'error': 0.0},
                actionable_recommendations=[],
                synthesis_confidence=0.0,
                processing_time_ms=(time.time() - start_time) * 1000,
                sources_analyzed=0,
                cross_validation_score=0.0,
                suggested_follow_ups=[],
                related_topics=[]
            )
    
    def _integrate_synthesis_results(self, result: ComprehensiveSynthesisResult):
        """Integrate results from different synthesis engines."""
        # Generate executive summary
        result.executive_summary = self._generate_executive_summary(result)
        
        # Collect key findings
        result.key_findings = self._collect_key_findings(result)
        
        # Assess confidence across all results
        result.confidence_assessment = self._assess_comprehensive_confidence(result)
        
        # Generate actionable recommendations
        result.actionable_recommendations = self._generate_actionable_recommendations(result)
        
        # Generate follow-up suggestions
        result.suggested_follow_ups = self._generate_follow_up_suggestions(result)
        
        # Identify related topics
        result.related_topics = self._identify_related_topics(result)
    
    def _generate_executive_summary(self, result: ComprehensiveSynthesisResult) -> str:
        """Generate executive summary from all synthesis results."""
        summary_parts = []
        
        # Query context
        summary_parts.append(f"Comprehensive synthesis for: {result.request.query}")
        
        # Question answering summary
        if result.question_answer:
            qa_summary = f"Direct answer provided with {result.question_answer.confidence_score:.1%} confidence"
            if result.question_answer.sources:
                qa_summary += f" based on {len(result.question_answer.sources)} sources"
            summary_parts.append(qa_summary)
        
        # Insights summary
        if result.insights:
            insight_count = len(result.insights.patterns) + len(result.insights.trends) + len(result.insights.anomalies)
            if insight_count > 0:
                summary_parts.append(f"Discovered {insight_count} insights including patterns, trends, and anomalies")
            else:
                summary_parts.append("No significant patterns or trends identified")
        
        # Perspective summary
        if result.perspectives:
            if result.perspectives.perspectives:
                perspective_summary = f"Analyzed {len(result.perspectives.perspectives)} different perspectives"
                if result.perspectives.overall_consensus:
                    consensus_desc = result.perspectives.overall_consensus.value.replace('_', ' ')
                    perspective_summary += f" with {consensus_desc} among viewpoints"
                summary_parts.append(perspective_summary)
            else:
                summary_parts.append("Limited perspective diversity found")
        
        # Overall assessment
        if result.synthesis_confidence > 0.7:
            summary_parts.append("High confidence in synthesis results")
        elif result.synthesis_confidence > 0.4:
            summary_parts.append("Moderate confidence in synthesis results")
        else:
            summary_parts.append("Low confidence - additional research recommended")
        
        return ". ".join(summary_parts) + "."
    
    def _collect_key_findings(self, result: ComprehensiveSynthesisResult) -> List[str]:
        """Collect key findings from all synthesis results."""
        findings = []
        
        # Question answering findings
        if result.question_answer and result.question_answer.confidence_score > 0.5:
            findings.append(f"Answer: {result.question_answer.answer[:100]}...")
            
            # Add follow-up questions as findings
            if result.question_answer.follow_up_questions:
                findings.extend(f"Related question: {q}" for q in result.question_answer.follow_up_questions[:2])
        
        # Insight findings
        if result.insights:
            # Patterns
            high_conf_patterns = [p for p in result.insights.patterns if p.confidence > 0.6]
            for pattern in high_conf_patterns[:2]:
                findings.append(f"Pattern: {pattern.description}")
            
            # Trends
            strong_trends = [t for t in result.insights.trends if t.strength > 0.6]
            for trend in strong_trends[:2]:
                findings.append(f"Trend: {trend.description}")
            
            # High-severity anomalies
            severe_anomalies = [a for a in result.insights.anomalies if a.severity > 0.7]
            for anomaly in severe_anomalies[:1]:
                findings.append(f"Anomaly: {anomaly.description}")
        
        # Perspective findings
        if result.perspectives:
            # High-confidence perspectives
            strong_perspectives = [p for p in result.perspectives.perspectives if p.confidence_score > 0.7]
            for perspective in strong_perspectives[:2]:
                findings.append(f"Perspective ({perspective.perspective_type.value}): {perspective.viewpoint[:80]}...")
            
            # Consensus areas
            if result.perspectives.comparisons:
                for comparison in result.perspectives.comparisons[:1]:
                    for consensus in comparison.consensus_areas[:1]:
                        findings.append(f"Consensus: {consensus}")
        
        return findings[:8]  # Limit to top 8 findings
    
    def _assess_comprehensive_confidence(self, result: ComprehensiveSynthesisResult) -> Dict[str, float]:
        """Assess confidence across all synthesis components."""
        confidence_assessment = {}
        
        # Individual component confidences
        if result.question_answer:
            confidence_assessment['question_answering'] = result.question_answer.confidence_score
        
        if result.insights:
            # Calculate average confidence from insights
            all_confidences = []
            all_confidences.extend([p.confidence for p in result.insights.patterns])
            all_confidences.extend([t.confidence for t in result.insights.trends])
            all_confidences.extend([a.confidence for a in result.insights.anomalies])
            
            if all_confidences:
                confidence_assessment['insight_discovery'] = sum(all_confidences) / len(all_confidences)
            else:
                confidence_assessment['insight_discovery'] = 0.0
        
        if result.perspectives:
            confidence_assessment['perspective_analysis'] = result.perspectives.analysis_confidence
        
        # Cross-validation confidence
        confidence_assessment['cross_validation'] = self._calculate_cross_validation_confidence(result)
        
        # Data coverage confidence
        confidence_assessment['data_coverage'] = self._calculate_data_coverage_confidence(result)
        
        return confidence_assessment
    
    def _calculate_cross_validation_confidence(self, result: ComprehensiveSynthesisResult) -> float:
        """Calculate confidence based on cross-validation between components."""
        if not (result.question_answer and result.perspectives):
            return 0.5  # Default when cross-validation not possible
        
        # Check if question answer aligns with perspectives
        answer_text = result.question_answer.answer.lower()
        perspective_agreements = 0
        total_perspectives = len(result.perspectives.perspectives)
        
        if total_perspectives == 0:
            return 0.5
        
        for perspective in result.perspectives.perspectives:
            perspective_text = perspective.viewpoint.lower()
            
            # Simple word overlap check
            answer_words = set(answer_text.split())
            perspective_words = set(perspective_text.split())
            
            if answer_words and perspective_words:
                overlap = len(answer_words & perspective_words)
                union = len(answer_words | perspective_words)
                similarity = overlap / union if union > 0 else 0
                
                if similarity > 0.2:  # Some alignment
                    perspective_agreements += 1
        
        # Calculate alignment ratio
        alignment_ratio = perspective_agreements / total_perspectives
        
        return min(alignment_ratio + 0.3, 1.0)  # Boost base confidence
    
    def _calculate_data_coverage_confidence(self, result: ComprehensiveSynthesisResult) -> float:
        """Calculate confidence based on data coverage across components."""
        unique_sources = self._count_unique_sources(result)
        
        # Confidence based on source diversity
        if unique_sources >= 10:
            return 0.9
        elif unique_sources >= 5:
            return 0.7
        elif unique_sources >= 3:
            return 0.5
        elif unique_sources >= 1:
            return 0.3
        else:
            return 0.1
    
    def _generate_actionable_recommendations(self, result: ComprehensiveSynthesisResult) -> List[str]:
        """Generate actionable recommendations from synthesis results."""
        recommendations = []
        
        # Question answering recommendations
        if result.question_answer and result.question_answer.confidence_score < 0.6:
            recommendations.append("Seek additional sources to increase answer confidence")
        
        # Insight recommendations
        if result.insights:
            recommendations.extend(result.insights.recommendations[:2])
        
        # Perspective recommendations
        if result.perspectives:
            recommendations.extend(result.perspectives.recommendations[:2])
        
        # Cross-synthesis recommendations
        if result.synthesis_confidence < 0.5:
            recommendations.append("Consider expanding the scope of analysis for more comprehensive insights")
        
        if result.sources_analyzed < 5:
            recommendations.append("Increase data collection to improve synthesis quality")
        
        # Conflict resolution recommendations
        if (result.perspectives and 
            result.perspectives.overall_consensus.value in ['no_consensus', 'strong_disagreement']):
            recommendations.append("Address conflicting viewpoints before making decisions")
        
        return list(set(recommendations))[:5]  # Remove duplicates and limit to 5
    
    def _generate_follow_up_suggestions(self, result: ComprehensiveSynthesisResult) -> List[str]:
        """Generate follow-up suggestions for further exploration."""
        follow_ups = []
        
        # From question answering
        if result.question_answer and result.question_answer.follow_up_questions:
            follow_ups.extend(result.question_answer.follow_up_questions[:2])
        
        # From insights
        if result.insights:
            # Suggest exploring patterns
            high_conf_patterns = [p for p in result.insights.patterns if p.confidence > 0.7]
            for pattern in high_conf_patterns[:1]:
                follow_ups.append(f"Explore implications of {pattern.pattern_type.value} pattern")
            
            # Suggest trend analysis
            strong_trends = [t for t in result.insights.trends if t.strength > 0.6]
            for trend in strong_trends[:1]:
                follow_ups.append(f"Investigate {trend.trend_type.value} trend in more detail")
        
        # From perspectives
        if result.perspectives:
            # Suggest exploring specific perspective types
            diverse_types = set(p.perspective_type.value for p in result.perspectives.perspectives)
            if len(diverse_types) < 3:
                missing_types = set(['stakeholder', 'temporal', 'methodological']) - diverse_types
                for missing_type in list(missing_types)[:1]:
                    follow_ups.append(f"Consider {missing_type} perspectives for broader analysis")
        
        # General synthesis follow-ups
        if result.synthesis_confidence > 0.7:
            follow_ups.append("Apply insights to practical implementation scenarios")
        else:
            follow_ups.append("Gather additional evidence to strengthen conclusions")
        
        return follow_ups[:5]  # Limit to 5 suggestions
    
    def _identify_related_topics(self, result: ComprehensiveSynthesisResult) -> List[str]:
        """Identify related topics for further exploration."""
        related_topics = set()
        
        # From question answering sources
        if result.question_answer and result.question_answer.sources:
            for source in result.question_answer.sources[:3]:
                # Extract key terms from source metadata
                if 'topic' in source.metadata:
                    related_topics.add(source.metadata['topic'])
                if 'domain' in source.metadata:
                    related_topics.add(source.metadata['domain'])
        
        # From insight patterns
        if result.insights:
            for pattern in result.insights.patterns[:2]:
                if 'domain' in pattern.metadata:
                    related_topics.add(pattern.metadata['domain'])
        
        # From perspectives
        if result.perspectives:
            for perspective in result.perspectives.perspectives[:3]:
                # Extract entities from perspective metadata
                if 'entities' in perspective.metadata:
                    for entity in perspective.metadata['entities'][:2]:
                        related_topics.add(entity)
        
        # Remove the main query topic
        query_words = set(result.request.query.lower().split())
        filtered_topics = []
        
        for topic in related_topics:
            topic_words = set(topic.lower().split())
            # Only include if not too similar to main query
            overlap = len(query_words & topic_words)
            if overlap < len(query_words) * 0.8:  # Less than 80% overlap
                filtered_topics.append(topic)
        
        return filtered_topics[:5]  # Limit to 5 related topics
    
    def _calculate_synthesis_confidence(self, result: ComprehensiveSynthesisResult) -> float:
        """Calculate overall synthesis confidence."""
        confidence_scores = []
        
        if result.question_answer:
            confidence_scores.append(result.question_answer.confidence_score)
        
        if result.insights:
            # Average confidence from insights
            all_confidences = []
            all_confidences.extend([p.confidence for p in result.insights.patterns])
            all_confidences.extend([t.confidence for t in result.insights.trends])
            all_confidences.extend([a.confidence for a in result.insights.anomalies])
            
            if all_confidences:
                confidence_scores.append(sum(all_confidences) / len(all_confidences))
        
        if result.perspectives:
            confidence_scores.append(result.perspectives.analysis_confidence)
        
        if not confidence_scores:
            return 0.0
        
        # Calculate weighted average (equal weights for now)
        base_confidence = sum(confidence_scores) / len(confidence_scores)
        
        # Apply cross-validation boost
        cross_val_confidence = result.cross_validation_score if hasattr(result, 'cross_validation_score') else 0.5
        
        # Combine confidences
        synthesis_confidence = (base_confidence * 0.7) + (cross_val_confidence * 0.3)
        
        return min(synthesis_confidence, 0.95)
    
    def _count_unique_sources(self, result: ComprehensiveSynthesisResult) -> int:
        """Count unique sources across all synthesis results."""
        unique_sources = set()
        
        if result.question_answer and result.question_answer.sources:
            unique_sources.update(source.node_id for source in result.question_answer.sources)
        
        if result.insights:
            # Add pattern sources
            for pattern in result.insights.patterns:
                unique_sources.update(pattern.elements)
            
            # Add anomaly sources
            for anomaly in result.insights.anomalies:
                unique_sources.update(anomaly.affected_entities)
        
        if result.perspectives:
            for perspective in result.perspectives.perspectives:
                unique_sources.update(perspective.supporting_evidence)
        
        return len(unique_sources)
    
    def _calculate_cross_validation_score(self, result: ComprehensiveSynthesisResult) -> float:
        """Calculate cross-validation score between different synthesis components."""
        if not hasattr(result, 'confidence_assessment'):
            return 0.5
        
        # Use cross-validation confidence from assessment
        return result.confidence_assessment.get('cross_validation', 0.5)
    
    def _extract_domain_from_query(self, query: str) -> Optional[str]:
        """Extract domain/topic from query for insight discovery."""
        # Simple domain extraction - look for key domain indicators
        domain_keywords = {
            'technology': ['tech', 'software', 'computer', 'digital', 'ai', 'algorithm'],
            'business': ['business', 'market', 'company', 'finance', 'economy'],
            'science': ['research', 'study', 'analysis', 'data', 'experiment'],
            'health': ['health', 'medical', 'doctor', 'patient', 'treatment'],
            'education': ['education', 'learning', 'student', 'teach', 'academic']
        }
        
        query_lower = query.lower()
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return domain
        
        return None  # Let insight discovery analyze all domains
    
    def _extract_topic_from_query(self, query: str) -> str:
        """Extract main topic from query for perspective analysis."""
        # Simple topic extraction - use the query as topic
        # In a more sophisticated implementation, this could use NLP to extract entities
        return query
    
    def _create_error_result(self, request: SynthesisRequest, error_msg: str, start_time: float):
        """Create error result based on task type."""
        processing_time = (time.time() - start_time) * 1000
        
        if request.task_type == SynthesisTaskType.QUESTION_ANSWERING:
            from memory_core.synthesis.question_answering import SynthesizedAnswer, QuestionType
            return SynthesizedAnswer(
                answer=f"Synthesis failed: {error_msg}",
                confidence_score=0.0,
                sources=[],
                reasoning="Error during synthesis",
                question_type=QuestionType.FACTUAL,
                subgraphs_used=[],
                processing_time_ms=processing_time
            )
        
        elif request.task_type == SynthesisTaskType.INSIGHT_DISCOVERY:
            from memory_core.synthesis.insight_discovery import InsightReport
            return InsightReport(
                patterns=[],
                trends=[],
                anomalies=[],
                summary=f"Insight discovery failed: {error_msg}",
                discovery_time_ms=processing_time,
                total_entities_analyzed=0,
                confidence_distribution={},
                recommendations=[]
            )
        
        elif request.task_type == SynthesisTaskType.PERSPECTIVE_ANALYSIS:
            from memory_core.synthesis.perspective_analysis import PerspectiveAnalysisReport, ConsensusLevel
            return PerspectiveAnalysisReport(
                topic=request.query,
                perspectives=[],
                comparisons=[],
                stakeholder_analysis=[],
                temporal_evolution=None,
                overall_consensus=ConsensusLevel.NO_CONSENSUS,
                key_insights=[f"Perspective analysis failed: {error_msg}"],
                recommendations=[],
                analysis_confidence=0.0,
                processing_time_ms=processing_time
            )
        
        else:  # COMPREHENSIVE_SYNTHESIS
            return ComprehensiveSynthesisResult(
                request=request,
                executive_summary=f"Comprehensive synthesis failed: {error_msg}",
                key_findings=[],
                confidence_assessment={'error': 0.0},
                actionable_recommendations=[],
                synthesis_confidence=0.0,
                processing_time_ms=processing_time,
                sources_analyzed=0,
                cross_validation_score=0.0,
                suggested_follow_ups=[],
                related_topics=[]
            )
    
    def _update_statistics(self, request: SynthesisRequest, result: Any, processing_time: float):
        """Update engine statistics."""
        self.stats['total_syntheses'] += 1
        
        # Update by type
        task_type = request.task_type.value
        self.stats['synthesis_by_type'][task_type] = self.stats['synthesis_by_type'].get(task_type, 0) + 1
        
        # Update by mode
        mode = request.mode.value
        self.stats['synthesis_by_mode'][mode] = self.stats['synthesis_by_mode'].get(mode, 0) + 1
        
        # Update average processing time
        total_time = self.stats['avg_processing_time_ms'] * (self.stats['total_syntheses'] - 1)
        self.stats['avg_processing_time_ms'] = (total_time + processing_time) / self.stats['total_syntheses']
        
        # Update confidence scores
        confidence_score = 0.0
        if hasattr(result, 'confidence_score'):
            confidence_score = result.confidence_score
        elif hasattr(result, 'synthesis_confidence'):
            confidence_score = result.synthesis_confidence
        elif hasattr(result, 'analysis_confidence'):
            confidence_score = result.analysis_confidence
        
        total_confidence = self.stats['avg_confidence_score'] * (self.stats['total_syntheses'] - 1)
        self.stats['avg_confidence_score'] = (total_confidence + confidence_score) / self.stats['total_syntheses']
        
        # Update cross-validation scores
        if hasattr(result, 'cross_validation_score'):
            self.stats['cross_validation_scores'].append(result.cross_validation_score)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive synthesis engine statistics.
        
        Returns:
            Dictionary with statistics from all components
        """
        return {
            'synthesis_engine': self.stats.copy(),
            'question_answering': self.question_answering.get_statistics(),
            'insight_discovery': self.insight_discovery.get_statistics(),
            'perspective_analysis': self.perspective_analysis.get_statistics(),
            'query_engine': self.query_engine.get_statistics()
        }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get information about synthesis engine capabilities.
        
        Returns:
            Dictionary describing available capabilities
        """
        return {
            'synthesis_types': [task_type.value for task_type in SynthesisTaskType],
            'synthesis_modes': [mode.value for mode in SynthesisMode],
            'perspective_types': [ptype.value for ptype in PerspectiveType],
            'pattern_types': [ptype.value for ptype in PatternType],
            'trend_types': [ttype.value for ttype in TrendType],
            'anomaly_types': [atype.value for atype in AnomalyType],
            'mode_configurations': self.mode_configs,
            'features': {
                'question_answering': True,
                'insight_discovery': True,
                'perspective_analysis': True,
                'comprehensive_synthesis': True,
                'cross_validation': True,
                'temporal_analysis': True,
                'stakeholder_analysis': True
            }
        }