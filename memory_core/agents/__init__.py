"""
Agents package for integrating with Google Agent Development Kit.

This package provides agent implementations that leverage the Memory Engine
for knowledge extraction and retrieval.
"""

from memory_core.agents.knowledge_agent import KnowledgeAgent, create_knowledge_agent

__all__ = ["KnowledgeAgent", "create_knowledge_agent"]
