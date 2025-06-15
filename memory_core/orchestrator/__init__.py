"""
Orchestrator Integration Module

This module provides orchestrator integration capabilities including:
- Enhanced MCP interface with streaming support
- Inter-module communication and event system
- Module registry with capability advertisement
- Standardized data formats and interfaces
"""

from .enhanced_mcp import EnhancedMCPServer, MCPStreaming, ProgressCallback
from .query_language import GraphQLQueryProcessor, QueryBuilder, QueryValidator
from .event_system import EventSystem, EventBus, EventSubscriber, EventPublisher
from .module_registry import ModuleRegistry, ModuleCapability, ModuleMetadata
from .data_formats import StandardizedKnowledge, CrossModuleEntity, UnifiedError

__all__ = [
    # Enhanced MCP
    'EnhancedMCPServer',
    'MCPStreaming', 
    'ProgressCallback',
    
    # Query Language
    'GraphQLQueryProcessor',
    'QueryBuilder',
    'QueryValidator',
    
    # Event System
    'EventSystem',
    'EventBus',
    'EventSubscriber',
    'EventPublisher',
    
    # Module Registry
    'ModuleRegistry',
    'ModuleCapability',
    'ModuleMetadata',
    
    # Data Formats
    'StandardizedKnowledge',
    'CrossModuleEntity',
    'UnifiedError'
]