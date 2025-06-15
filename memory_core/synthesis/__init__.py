"""
Knowledge Synthesis Engine Module

Provides intelligent synthesis capabilities for generating insights,
answering complex questions, and discovering patterns in knowledge graphs.
"""

from .question_answering import QuestionAnsweringSystem
from .insight_discovery import InsightDiscoveryEngine
from .perspective_analysis import PerspectiveAnalysisEngine
from .knowledge_synthesis_engine import (
    KnowledgeSynthesisEngine,
    SynthesisRequest,
    SynthesisTaskType,
    SynthesisMode,
    ComprehensiveSynthesisResult,
)

__all__ = [
    "QuestionAnsweringSystem",
    "InsightDiscoveryEngine",
    "PerspectiveAnalysisEngine",
    "KnowledgeSynthesisEngine",
    "SynthesisRequest",
    "SynthesisTaskType",
    "SynthesisMode",
    "ComprehensiveSynthesisResult",
]
