"""
Orchestrator Integration Module

This module provides orchestrator integration capabilities including:
- Enhanced MCP interface with streaming support
- Inter-module communication and event system
- Module registry with capability advertisement
- Standardized data formats and interfaces
"""

from .enhanced_mcp import EnhancedMCPServer, MCPStreaming, ProgressCallback
from .query_language import (
    GraphQLQueryProcessor,
    QueryBuilder,
    QueryValidator,
    QueryType,
    FilterOperator,
    QuerySpec,
)
from .event_system import (
    EventSystem,
    EventBus,
    EventSubscriber,
    EventPublisher,
    Event,
    EventType,
    EventPriority,
    EventStatus,
    EventMetadata,
    KnowledgeChangeEvent,
    RelationshipChangeEvent,
    QueryEvent,
    SystemEvent,
    CustomEvent,
)
from .module_registry import (
    ModuleRegistry,
    ModuleCapability,
    ModuleMetadata,
    ModuleInterface,
    ModuleStatus,
    CapabilityType,
    Version,
    RegisteredModule,
)
from .data_formats import (
    StandardizedKnowledge,
    CrossModuleEntity,
    UnifiedError,
    OperationResult,
    StandardizedIdentifier,
    StandardizedRelationship,
    StandardizedQuery,
    EntityType,
    ErrorCode,
    StatusCode,
    DataFormat,
    create_knowledge_entity,
)

__all__ = [
    # Enhanced MCP
    "EnhancedMCPServer",
    "MCPStreaming",
    "ProgressCallback",
    # Query Language
    "GraphQLQueryProcessor",
    "QueryBuilder",
    "QueryValidator",
    "QueryType",
    "FilterOperator",
    "QuerySpec",
    # Event System
    "EventSystem",
    "EventBus",
    "EventSubscriber",
    "EventPublisher",
    "Event",
    "EventType",
    "EventPriority",
    "EventStatus",
    "EventMetadata",
    "KnowledgeChangeEvent",
    "RelationshipChangeEvent",
    "QueryEvent",
    "SystemEvent",
    "CustomEvent",
    # Module Registry
    "ModuleRegistry",
    "ModuleCapability",
    "ModuleMetadata",
    "ModuleInterface",
    "ModuleStatus",
    "CapabilityType",
    "Version",
    "RegisteredModule",
    # Data Formats
    "StandardizedKnowledge",
    "CrossModuleEntity",
    "UnifiedError",
    "OperationResult",
    "StandardizedIdentifier",
    "StandardizedRelationship",
    "StandardizedQuery",
    "EntityType",
    "ErrorCode",
    "StatusCode",
    "DataFormat",
    "create_knowledge_entity",
]
