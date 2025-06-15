"""
Standardized Data Formats for Memory Engine Inter-Module Communication

This module defines standardized data formats, interfaces, and protocols
for consistent communication between Memory Engine modules:
- Common knowledge representation formats
- Cross-module entity resolution
- Unified error responses and status codes
- Standard query and response formats
"""

import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Set, Tuple, TypeVar, Generic
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

T = TypeVar("T")


class EntityType(Enum):
    """Standard entity types across modules."""

    KNOWLEDGE_NODE = "knowledge_node"
    RELATIONSHIP = "relationship"
    CONCEPT = "concept"
    DOCUMENT = "document"
    USER = "user"
    QUERY = "query"
    RESULT = "result"
    CUSTOM = "custom"


class DataFormat(Enum):
    """Supported data formats for serialization."""

    JSON = "json"
    YAML = "yaml"
    XML = "xml"
    PROTOBUF = "protobuf"
    MSGPACK = "msgpack"


class ErrorCode(Enum):
    """Standardized error codes."""

    # Success
    SUCCESS = "SUCCESS"

    # Client errors (4xx equivalent)
    INVALID_REQUEST = "INVALID_REQUEST"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    RATE_LIMITED = "RATE_LIMITED"

    # Server errors (5xx equivalent)
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    TIMEOUT = "TIMEOUT"
    STORAGE_ERROR = "STORAGE_ERROR"
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"

    # Module-specific errors
    PROVIDER_ERROR = "PROVIDER_ERROR"
    CAPABILITY_NOT_SUPPORTED = "CAPABILITY_NOT_SUPPORTED"
    VERSION_INCOMPATIBLE = "VERSION_INCOMPATIBLE"
    DEPENDENCY_ERROR = "DEPENDENCY_ERROR"


class StatusCode(Enum):
    """Standard status codes for operations."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"
    PARTIAL = "partial"


@dataclass
class StandardizedIdentifier:
    """Standardized identifier for cross-module entity references."""

    entity_type: EntityType
    entity_id: str
    module_id: str
    version: Optional[str] = None
    namespace: Optional[str] = None

    def __str__(self) -> str:
        """String representation of identifier."""
        parts = [self.module_id, self.entity_type.value, self.entity_id]
        if self.namespace:
            parts.insert(0, self.namespace)
        if self.version:
            parts.append(f"v{self.version}")
        return ":".join(parts)

    @classmethod
    def parse(cls, identifier_str: str) -> "StandardizedIdentifier":
        """Parse identifier from string."""
        parts = identifier_str.split(":")

        if len(parts) < 3:
            raise ValueError(f"Invalid identifier format: {identifier_str}")

        # Check for version suffix
        version = None
        if parts[-1].startswith("v"):
            version = parts[-1][1:]
            parts = parts[:-1]

        # Check for namespace prefix
        namespace = None
        if len(parts) > 3:
            namespace = parts[0]
            parts = parts[1:]

        module_id, entity_type_str, entity_id = parts
        entity_type = EntityType(entity_type_str)

        return cls(
            entity_type=entity_type,
            entity_id=entity_id,
            module_id=module_id,
            version=version,
            namespace=namespace,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StandardizedIdentifier":
        """Create from dictionary."""
        data["entity_type"] = EntityType(data["entity_type"])
        return cls(**data)


@dataclass
class UnifiedError:
    """Unified error response format for all modules."""

    code: ErrorCode
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    correlation_id: Optional[str] = None
    module_id: Optional[str] = None
    request_id: Optional[str] = None
    stack_trace: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["code"] = self.code.value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedError":
        """Create from dictionary."""
        data["code"] = ErrorCode(data["code"])
        return cls(**data)

    @classmethod
    def from_exception(
        cls, exception: Exception, module_id: str = None, correlation_id: str = None
    ) -> "UnifiedError":
        """Create error from exception."""
        import traceback

        return cls(
            code=ErrorCode.INTERNAL_ERROR,
            message=str(exception),
            details={"exception_type": type(exception).__name__},
            module_id=module_id,
            correlation_id=correlation_id,
            stack_trace=traceback.format_exc(),
        )


@dataclass
class OperationResult(Generic[T]):
    """Standardized result wrapper for all operations."""

    success: bool
    data: Optional[T] = None
    error: Optional[UnifiedError] = None
    status: StatusCode = StatusCode.COMPLETED
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    operation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "success": self.success,
            "status": self.status.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "operation_id": self.operation_id,
        }

        if self.data is not None:
            if hasattr(self.data, "to_dict"):
                result["data"] = self.data.to_dict()
            else:
                result["data"] = self.data

        if self.error:
            result["error"] = self.error.to_dict()

        return result

    @classmethod
    def success_result(cls, data: T, metadata: Dict[str, Any] = None) -> "OperationResult[T]":
        """Create successful result."""
        return cls(success=True, data=data, metadata=metadata or {})

    @classmethod
    def error_result(
        cls, error: UnifiedError, metadata: Dict[str, Any] = None
    ) -> "OperationResult[T]":
        """Create error result."""
        return cls(success=False, error=error, status=StatusCode.FAILED, metadata=metadata or {})


@dataclass
class StandardizedKnowledge:
    """Standardized knowledge representation for cross-module compatibility."""

    identifier: StandardizedIdentifier
    content: str
    content_type: str = "text/plain"
    metadata: Dict[str, Any] = field(default_factory=dict)
    embeddings: Dict[str, List[float]] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    version: int = 1
    source: Optional[str] = None
    confidence_score: Optional[float] = None
    quality_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["identifier"] = self.identifier.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StandardizedKnowledge":
        """Create from dictionary."""
        data["identifier"] = StandardizedIdentifier.from_dict(data["identifier"])
        return cls(**data)

    def get_embedding(self, provider: str) -> Optional[List[float]]:
        """Get embedding for specific provider."""
        return self.embeddings.get(provider)

    def set_embedding(self, provider: str, embedding: List[float]):
        """Set embedding for provider."""
        self.embeddings[provider] = embedding
        self.updated_at = time.time()

    def update_content(self, content: str, content_type: str = None):
        """Update content and metadata."""
        self.content = content
        if content_type:
            self.content_type = content_type
        self.updated_at = time.time()
        self.version += 1


@dataclass
class StandardizedRelationship:
    """Standardized relationship representation."""

    identifier: StandardizedIdentifier
    source_id: StandardizedIdentifier
    target_id: StandardizedIdentifier
    relationship_type: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    version: int = 1
    weight: Optional[float] = None
    bidirectional: bool = False
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["identifier"] = self.identifier.to_dict()
        result["source_id"] = self.source_id.to_dict()
        result["target_id"] = self.target_id.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StandardizedRelationship":
        """Create from dictionary."""
        data["identifier"] = StandardizedIdentifier.from_dict(data["identifier"])
        data["source_id"] = StandardizedIdentifier.from_dict(data["source_id"])
        data["target_id"] = StandardizedIdentifier.from_dict(data["target_id"])
        return cls(**data)


@dataclass
class CrossModuleEntity:
    """Entity that can be referenced across multiple modules."""

    identifier: StandardizedIdentifier
    entity_data: Dict[str, Any]
    schema_version: str = "1.0"
    module_specific_data: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    synchronization_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["identifier"] = self.identifier.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CrossModuleEntity":
        """Create from dictionary."""
        data["identifier"] = StandardizedIdentifier.from_dict(data["identifier"])
        return cls(**data)

    def get_module_data(self, module_id: str) -> Dict[str, Any]:
        """Get module-specific data."""
        return self.module_specific_data.get(module_id, {})

    def set_module_data(self, module_id: str, data: Dict[str, Any]):
        """Set module-specific data."""
        self.module_specific_data[module_id] = data
        self.synchronization_metadata["last_updated"] = time.time()
        self.synchronization_metadata["updated_by"] = module_id


@dataclass
class StandardizedQuery:
    """Standardized query format for cross-module queries."""

    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query_type: str = "semantic_search"
    query_text: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    filters: Dict[str, Any] = field(default_factory=dict)
    limit: Optional[int] = None
    offset: Optional[int] = None
    sort_by: Optional[str] = None
    include_metadata: bool = True
    format_preferences: List[DataFormat] = field(default_factory=lambda: [DataFormat.JSON])
    target_modules: Optional[List[str]] = None
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["format_preferences"] = [fmt.value for fmt in self.format_preferences]
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StandardizedQuery":
        """Create from dictionary."""
        if "format_preferences" in data:
            data["format_preferences"] = [DataFormat(fmt) for fmt in data["format_preferences"]]
        return cls(**data)


@dataclass
class StandardizedQueryResult:
    """Standardized query result format."""

    query_id: str
    results: List[Union[StandardizedKnowledge, StandardizedRelationship, CrossModuleEntity]]
    total_count: int
    returned_count: int
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    next_page_token: Optional[str] = None
    facets: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["results"] = [
            item.to_dict() if hasattr(item, "to_dict") else item for item in self.results
        ]
        return result


class EntityResolver:
    """Resolves entity references across modules."""

    def __init__(self):
        self.entity_mappings: Dict[str, Set[StandardizedIdentifier]] = {}
        self.canonical_entities: Dict[str, StandardizedIdentifier] = {}

    def register_entity_mapping(self, canonical_id: str, entity_ids: List[StandardizedIdentifier]):
        """Register mapping between entities across modules."""
        if canonical_id not in self.entity_mappings:
            self.entity_mappings[canonical_id] = set()

        for entity_id in entity_ids:
            self.entity_mappings[canonical_id].add(entity_id)
            self.canonical_entities[str(entity_id)] = entity_ids[0]  # First as canonical

    def resolve_entity(self, entity_id: StandardizedIdentifier) -> Set[StandardizedIdentifier]:
        """Resolve entity to all its known references."""
        entity_str = str(entity_id)

        # Find canonical group
        for canonical_id, entities in self.entity_mappings.items():
            if entity_id in entities:
                return entities.copy()

        # If not found, return just the original
        return {entity_id}

    def get_canonical_entity(self, entity_id: StandardizedIdentifier) -> StandardizedIdentifier:
        """Get canonical entity for given ID."""
        return self.canonical_entities.get(str(entity_id), entity_id)


class DataFormatConverter:
    """Converts between different data formats."""

    @staticmethod
    def convert(data: Any, from_format: DataFormat, to_format: DataFormat) -> Any:
        """Convert data between formats."""
        if from_format == to_format:
            return data

        # Convert to intermediate dict representation
        if from_format == DataFormat.JSON:
            if isinstance(data, str):
                intermediate = json.loads(data)
            else:
                intermediate = data
        else:
            raise NotImplementedError(f"Conversion from {from_format} not implemented")

        # Convert from intermediate to target format
        if to_format == DataFormat.JSON:
            return intermediate
        elif to_format == DataFormat.YAML:
            import yaml

            return yaml.dump(intermediate)
        else:
            raise NotImplementedError(f"Conversion to {to_format} not implemented")

    @staticmethod
    def serialize_for_format(obj: Any, format_type: DataFormat) -> str:
        """Serialize object for specific format."""
        if hasattr(obj, "to_dict"):
            data = obj.to_dict()
        else:
            data = obj

        if format_type == DataFormat.JSON:
            return json.dumps(data, indent=2)
        elif format_type == DataFormat.YAML:
            import yaml

            return yaml.dump(data, default_flow_style=False)
        else:
            raise NotImplementedError(f"Serialization for {format_type} not implemented")


class SchemaValidator:
    """Validates data against standardized schemas."""

    def __init__(self):
        self.schemas: Dict[str, Dict[str, Any]] = {}
        self._load_default_schemas()

    def _load_default_schemas(self):
        """Load default schemas for standardized types."""
        # This would typically load from schema files
        # For now, we'll define basic schemas programmatically

        self.schemas["StandardizedKnowledge"] = {
            "type": "object",
            "required": ["identifier", "content"],
            "properties": {
                "identifier": {"type": "object"},
                "content": {"type": "string"},
                "content_type": {"type": "string"},
                "metadata": {"type": "object"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
        }

        self.schemas["StandardizedRelationship"] = {
            "type": "object",
            "required": ["identifier", "source_id", "target_id", "relationship_type"],
            "properties": {
                "identifier": {"type": "object"},
                "source_id": {"type": "object"},
                "target_id": {"type": "object"},
                "relationship_type": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
        }

    def validate(self, data: Dict[str, Any], schema_name: str) -> Tuple[bool, List[str]]:
        """Validate data against schema."""
        if schema_name not in self.schemas:
            return False, [f"Schema '{schema_name}' not found"]

        schema = self.schemas[schema_name]
        errors = []

        # Basic validation (in production, use jsonschema library)
        if schema.get("type") == "object":
            required_fields = schema.get("required", [])
            for field in required_fields:
                if field not in data:
                    errors.append(f"Required field '{field}' missing")

        return len(errors) == 0, errors

    def register_schema(self, name: str, schema: Dict[str, Any]):
        """Register custom schema."""
        self.schemas[name] = schema


class ProtocolHandler(ABC):
    """Abstract base class for protocol handlers."""

    @abstractmethod
    async def send_request(self, target_module: str, request: Dict[str, Any]) -> OperationResult:
        """Send request to target module."""
        pass

    @abstractmethod
    async def handle_request(self, request: Dict[str, Any]) -> OperationResult:
        """Handle incoming request."""
        pass


class StandardizedInterface(ABC):
    """Standard interface for inter-module communication."""

    @abstractmethod
    def get_supported_operations(self) -> List[str]:
        """Get list of supported operations."""
        pass

    @abstractmethod
    async def execute_operation(
        self, operation: str, parameters: Dict[str, Any]
    ) -> OperationResult:
        """Execute operation with parameters."""
        pass

    @abstractmethod
    def get_schema_for_operation(self, operation: str) -> Dict[str, Any]:
        """Get schema for operation parameters."""
        pass

    @abstractmethod
    async def health_check(self) -> OperationResult[Dict[str, Any]]:
        """Perform health check."""
        pass


# Convenience functions for common operations
def create_knowledge_entity(
    module_id: str, content: str, entity_id: str = None
) -> StandardizedKnowledge:
    """Create standardized knowledge entity."""
    if not entity_id:
        entity_id = str(uuid.uuid4())

    identifier = StandardizedIdentifier(
        entity_type=EntityType.KNOWLEDGE_NODE, entity_id=entity_id, module_id=module_id
    )

    return StandardizedKnowledge(identifier=identifier, content=content)


def create_relationship(
    source_id: StandardizedIdentifier,
    target_id: StandardizedIdentifier,
    relationship_type: str,
    module_id: str,
    relationship_id: str = None,
) -> StandardizedRelationship:
    """Create standardized relationship."""
    if not relationship_id:
        relationship_id = str(uuid.uuid4())

    identifier = StandardizedIdentifier(
        entity_type=EntityType.RELATIONSHIP, entity_id=relationship_id, module_id=module_id
    )

    return StandardizedRelationship(
        identifier=identifier,
        source_id=source_id,
        target_id=target_id,
        relationship_type=relationship_type,
    )


def create_success_response(data: Any, metadata: Dict[str, Any] = None) -> OperationResult:
    """Create successful operation result."""
    return OperationResult.success_result(data, metadata)


def create_error_response(
    error_code: ErrorCode, message: str, details: Dict[str, Any] = None
) -> OperationResult:
    """Create error operation result."""
    error = UnifiedError(code=error_code, message=message, details=details or {})
    return OperationResult.error_result(error)
