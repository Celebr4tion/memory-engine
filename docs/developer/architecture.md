# Memory Engine Architecture

This document provides a comprehensive overview of the Memory Engine's architecture, component interactions, and design principles.

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Diagrams](#architecture-diagrams)
3. [Component Descriptions](#component-descriptions)
4. [Data Flow](#data-flow)
5. [Technology Stack](#technology-stack)
6. [Design Patterns](#design-patterns)
7. [Scalability Considerations](#scalability-considerations)

## System Overview

The Memory Engine is a semantic knowledge management system that combines graph-based knowledge representation with modern vector embeddings. It provides a unified platform for knowledge extraction, storage, retrieval, and relationship discovery.

### Core Principles

- **Dual Storage Strategy**: Graph database for relationships, vector database for similarity
- **Semantic Understanding**: LLM-powered knowledge extraction and embedding generation
- **Automatic Relationship Discovery**: Intelligent detection of connections between knowledge
- **Version Control**: Complete change tracking and rollback capabilities
- **Modular Design**: Loosely coupled components with clear interfaces

## Architecture Diagrams

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Memory Engine System                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │   External      │    │       MCP       │    │   Python     │ │
│  │   Systems       │◄──►│   Interface     │◄──►│     API      │ │
│  │                 │    │                 │    │              │ │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
│                                 │                              │
│                                 ▼                              │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                Knowledge Engine Core                        │ │
│  ├─────────────────┬─────────────────┬─────────────────┬──────┤ │
│  │   Knowledge     │   Relationship  │    Versioning   │ Rate │ │
│  │   Processing    │   Extraction    │    Manager      │ Sys  │ │
│  └─────────────────┴─────────────────┴─────────────────┴──────┘ │
│                                 │                              │
│                                 ▼                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │   Graph Store   │    │   Vector Store  │    │  Embedding   │ │
│  │  (JanusGraph)   │    │   (Milvus)      │    │   Manager    │ │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
│                                                        │        │
│                                                        ▼        │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                 Google Gemini API                           │ │
│  │           (Embeddings + Knowledge Extraction)              │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Component Interaction Diagram

```
┌─────────────────┐
│      User       │
│   Application   │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐     ┌─────────────────┐
│  MCP Endpoint   │◄───►│  Knowledge      │
│                 │     │  Agent (ADK)    │
└─────────┬───────┘     └─────────────────┘
          │
          ▼
┌─────────────────┐
│  Knowledge      │
│  Engine         │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Graph Storage  │     │  Versioned      │     │  Embedding      │
│  Adapter        │◄───►│  Graph Adapter  │◄───►│  Manager        │
└─────────┬───────┘     └─────────┬───────┘     └─────────┬───────┘
          │                       │                       │
          ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  JanusGraph     │     │  Revision       │     │  Vector Store   │
│  Storage        │     │  Manager        │     │  (Milvus)       │
└─────────────────┘     └─────────────────┘     └─────────┬───────┘
                                                          │
                                                          ▼
                                                ┌─────────────────┐
                                                │  Gemini API     │
                                                │  (Embeddings)   │
                                                └─────────────────┘
```

### Knowledge Processing Pipeline

```
Raw Text Input
      │
      ▼
┌─────────────────┐
│  Advanced       │  ◄── Gemini API (Text Processing)
│  Extractor      │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Knowledge      │
│  Units          │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐     ┌─────────────────┐
│  Merging        │◄───►│  Similarity     │ ◄── Vector Embeddings
│  System         │     │  Detection      │
└─────────┬───────┘     └─────────────────┘
          │
          ▼
┌─────────────────┐
│  Knowledge      │
│  Nodes          │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐     ┌─────────────────┐
│  Relationship   │     │  Automatic      │
│  Extractor      │────►│  Relationship   │
└─────────────────┘     │  Detection      │
          │             └─────────────────┘
          ▼
┌─────────────────┐
│  Knowledge      │
│  Graph          │
└─────────────────┘
```

### Data Storage Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Storage Layer                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────┐    ┌─────────────────────────┐  │
│  │       Graph Database        │    │     Vector Database     │  │
│  │      (JanusGraph)           │    │       (Milvus)         │  │
│  │                             │    │                         │  │
│  │  ┌─────────────────────┐    │    │  ┌─────────────────────┐│  │
│  │  │   Knowledge Nodes   │    │    │  │    Embeddings      ││  │
│  │  │                     │    │    │  │                     ││  │
│  │  │ • ID               │    │    │  │ • Node ID          ││  │
│  │  │ • Content          │    │    │  │ • Vector (3072D)   ││  │
│  │  │ • Source           │    │    │  │ • Timestamp        ││  │
│  │  │ • Ratings          │    │    │  └─────────────────────┘│  │
│  │  │ • Metadata         │    │    │                         │  │
│  │  └─────────────────────┘    │    │  ┌─────────────────────┐│  │
│  │                             │    │  │    Search Index     ││  │
│  │  ┌─────────────────────┐    │    │  │                     ││  │
│  │  │   Relationships     │    │    │  │ • IVF_FLAT         ││  │
│  │  │                     │    │    │  │ • L2 Distance      ││  │
│  │  │ • Edge ID          │    │    │  │ • nlist: 1024      ││  │
│  │  │ • From/To Nodes    │    │    │  └─────────────────────┘│  │
│  │  │ • Relation Type    │    │    │                         │  │
│  │  │ • Confidence       │    │    └─────────────────────────┘  │
│  │  │ • Metadata         │    │                                │
│  │  └─────────────────────┘    │                                │
│  │                             │                                │
│  │  ┌─────────────────────┐    │                                │
│  │  │   Revision Log      │    │                                │
│  │  │                     │    │                                │
│  │  │ • Change Type      │    │                                │
│  │  │ • Object ID        │    │                                │
│  │  │ • Old/New Data     │    │                                │
│  │  │ • Timestamp        │    │                                │
│  │  └─────────────────────┘    │                                │
│  │                             │                                │
│  │  ┌─────────────────────┐    │                                │
│  │  │   Snapshots         │    │                                │
│  │  │                     │    │                                │
│  │  │ • Snapshot ID      │    │                                │
│  │  │ • Full Graph State │    │                                │
│  │  │ • Timestamp        │    │                                │
│  │  └─────────────────────┘    │                                │
│  └─────────────────────────────┘                                │
└─────────────────────────────────────────────────────────────────┘
```

## Component Descriptions

### Core Components

#### 1. Knowledge Engine
**Location**: `memory_core/core/knowledge_engine.py`

**Purpose**: Main orchestrator that provides a unified interface to all subsystems.

**Responsibilities**:
- Coordinate between storage, versioning, and embedding systems
- Provide high-level API for knowledge operations
- Manage database connections and lifecycle
- Handle configuration and initialization

**Key Methods**:
- `save_node()`, `get_node()`, `delete_node()`
- `save_relationship()`, `get_relationship()`, `delete_relationship()`
- `connect()`, `disconnect()`
- Version management operations

#### 2. Graph Storage Adapter
**Location**: `memory_core/db/graph_storage_adapter.py`

**Purpose**: Abstraction layer between domain models and JanusGraph storage.

**Responsibilities**:
- Convert KnowledgeNode/Relationship objects to/from graph format
- Handle data type conversions and validations
- Provide CRUD operations for domain objects
- Abstract away database-specific details

#### 3. Versioned Graph Adapter
**Location**: `memory_core/db/versioned_graph_adapter.py`

**Purpose**: Add versioning capabilities to graph operations.

**Responsibilities**:
- Wrap all graph operations with change logging
- Integrate with RevisionManager for tracking
- Provide rollback and recovery operations
- Maintain operation atomicity

#### 4. JanusGraph Storage
**Location**: `memory_core/db/janusgraph_storage.py`

**Purpose**: Low-level interface to JanusGraph database.

**Responsibilities**:
- Manage Gremlin connections and queries
- Handle async/sync operation conversion
- Provide basic CRUD operations
- Connection pooling and error handling

### Knowledge Processing Components

#### 5. Advanced Extractor
**Location**: `memory_core/ingestion/advanced_extractor.py`

**Purpose**: Extract structured knowledge from raw text using LLMs.

**Responsibilities**:
- Interface with Gemini API for text processing
- Parse and validate extracted knowledge units
- Handle API rate limiting and errors
- Convert unstructured text to structured knowledge

#### 6. Relationship Extractor
**Location**: `memory_core/ingestion/relationship_extractor.py`

**Purpose**: Automatically detect relationships between knowledge nodes.

**Responsibilities**:
- Tag-based relationship detection
- Domain-based relationship creation
- Semantic similarity relationship suggestion
- Relationship confidence scoring

#### 7. Merging System
**Location**: `memory_core/ingestion/merging.py`

**Purpose**: Detect and merge similar knowledge nodes.

**Responsibilities**:
- Embedding-based similarity detection
- Intelligent data merging strategies
- Duplicate prevention and consolidation
- Metadata preservation during merging

### Data Management Components

#### 8. Embedding Manager
**Location**: `memory_core/embeddings/embedding_manager.py`

**Purpose**: Manage embedding generation and storage.

**Responsibilities**:
- Interface with Gemini API for embeddings
- Handle different embedding task types
- Coordinate with vector storage
- Embedding lifecycle management

#### 9. Vector Store (Milvus)
**Location**: `memory_core/embeddings/vector_store.py`

**Purpose**: Vector database operations for similarity search.

**Responsibilities**:
- Milvus connection and collection management
- Vector indexing and search operations
- Handle embedding storage and retrieval
- Performance optimization for vector operations

#### 10. Revision Manager
**Location**: `memory_core/versioning/revision_manager.py`

**Purpose**: Track changes and manage graph versioning.

**Responsibilities**:
- Log all graph modifications
- Create and manage snapshots
- Provide rollback capabilities
- Maintain change history and metadata

### Interface Components

#### 11. MCP Endpoint
**Location**: `memory_core/mcp_integration/mcp_endpoint.py`

**Purpose**: Module Communication Protocol interface.

**Responsibilities**:
- Provide standardized API for external systems
- Handle command parsing and validation
- Coordinate complex workflows
- Error handling and response formatting

#### 12. Knowledge Agent
**Location**: `memory_core/agents/knowledge_agent.py`

**Purpose**: Google ADK integration for agent-based interactions.

**Responsibilities**:
- Provide agent-based knowledge operations
- Integration with Google Agent Development Kit
- Tool-based interaction patterns
- Conversational knowledge interfaces

## Data Flow

### Knowledge Ingestion Flow

1. **Input**: Raw text + source metadata
2. **Extraction**: Gemini API processes text → knowledge units
3. **Similarity Check**: Generate embeddings → search for similar nodes
4. **Merging Decision**: If similarity > threshold → merge, else create new
5. **Storage**: Save nodes to JanusGraph + embeddings to Milvus
6. **Relationship Detection**: Analyze tags, domains, semantics
7. **Relationship Creation**: Create edges in graph
8. **Versioning**: Log all changes for tracking
9. **Output**: Return node IDs and relationship statistics

### Knowledge Retrieval Flow

1. **Input**: Search query text
2. **Embedding Generation**: Convert query to vector
3. **Vector Search**: Find similar embeddings in Milvus
4. **Node Retrieval**: Get full node data from JanusGraph
5. **Relationship Expansion**: Fetch connected nodes if requested
6. **Ranking**: Score results by relevance
7. **Output**: Return ranked knowledge nodes

### Rating Update Flow

1. **Input**: Node ID + evidence data
2. **Current State**: Fetch existing node ratings
3. **Evidence Processing**: Apply rating formulas
4. **Validation**: Ensure ratings stay within bounds
5. **Update**: Modify node in JanusGraph
6. **Versioning**: Log rating change
7. **Output**: Return updated node state

## Technology Stack

### Core Technologies

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Graph Database | JanusGraph | 0.6.3 | Knowledge graph storage |
| Vector Database | Milvus | 2.2.11 | Embedding storage/search |
| Storage Backend | Berkeley DB | Built-in | JanusGraph persistence |
| Object Storage | MinIO | Latest | Milvus object storage |
| Metadata Store | etcd | 3.5.5 | Milvus metadata |
| LLM API | Google Gemini | Latest | Embeddings & extraction |

### Python Dependencies

| Library | Purpose | Version Range |
|---------|---------|---------------|
| gremlinpython | JanusGraph client | >=3.7.0 |
| pymilvus | Milvus client | Latest |
| google-genai | Gemini API client | Latest |
| google-adk | Agent framework | Latest |
| fastapi | Web framework | Latest |
| pydantic | Data validation | Latest |
| numpy | Numerical operations | Latest |
| torch | ML operations | Latest |

### Development Tools

| Tool | Purpose |
|------|---------|
| pytest | Testing framework |
| Docker Compose | Development environment |
| Black | Code formatting |
| isort | Import sorting |
| mypy | Type checking |

## Design Patterns

### 1. Adapter Pattern
- **GraphStorageAdapter**: Adapts domain models to storage format
- **VectorStoreMilvus**: Adapts to Milvus vector operations
- **VersionedGraphAdapter**: Adds versioning behavior

### 2. Strategy Pattern
- **Relationship Extraction**: Multiple strategies (tags, domain, semantic)
- **Merging Logic**: Different merging strategies for different data types
- **Rating Updates**: Various evidence processing strategies

### 3. Facade Pattern
- **KnowledgeEngine**: Provides simplified interface to complex subsystem
- **MemoryEngineMCP**: Unified interface for external systems

### 4. Observer Pattern
- **RevisionManager**: Observes graph changes for logging
- **Embedding Manager**: Observes node creation for embedding generation

### 5. Factory Pattern
- **Knowledge Unit Creation**: Factory methods for different sources
- **Relationship Creation**: Factory for different relationship types

### 6. Command Pattern
- **MCP Commands**: Encapsulate operations as command objects
- **Rating Updates**: Evidence as commands for rating changes

## Scalability Considerations

### Horizontal Scaling

#### Database Scaling
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   JanusGraph    │    │   JanusGraph    │    │   JanusGraph    │
│   Instance 1    │    │   Instance 2    │    │   Instance 3    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Distributed   │
                    │    Backend      │
                    │  (Cassandra)    │
                    └─────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Milvus      │    │     Milvus      │    │     Milvus      │
│   Instance 1    │    │   Instance 2    │    │   Instance 3    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

#### Application Scaling
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Memory Engine  │    │  Memory Engine  │    │  Memory Engine  │
│   Instance 1    │    │   Instance 2    │    │   Instance 3    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Load Balancer  │
                    │   (nginx/HAP)   │
                    └─────────────────┘
```

### Performance Optimization

#### Caching Strategy
```
┌─────────────────┐
│   Application   │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐     Cache Hit
│   Redis Cache   │◄────────────┐
└─────────┬───────┘             │
          │ Cache Miss          │
          ▼                     │
┌─────────────────┐             │
│   Database      │─────────────┘
└─────────────────┘
```

#### Batch Processing
```
┌─────────────────┐
│   Input Queue   │
│  (RabbitMQ/     │
│   Kafka)        │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Worker 1       │    │  Worker 2       │    │  Worker 3       │
│  (Batch Size:   │    │  (Batch Size:   │    │  (Batch Size:   │
│   100)          │    │   100)          │    │   100)          │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Partitioning

#### Graph Partitioning
- **By Domain**: Partition nodes by knowledge domain
- **By Source**: Partition by data source
- **By Time**: Partition by creation timestamp

#### Vector Partitioning
- **By Collection**: Separate collections for different domains
- **By Index**: Multiple indexes for different similarity metrics
- **By Dimension**: Different dimensions for different embedding models

### Monitoring and Observability

#### Metrics Collection
```
┌─────────────────┐
│   Application   │
│    Metrics      │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐     ┌─────────────────┐
│   Prometheus    │────►│    Grafana      │
│   (Metrics)     │     │  (Dashboards)   │
└─────────────────┘     └─────────────────┘

┌─────────────────┐     ┌─────────────────┐
│   Application   │────►│   ELK Stack     │
│     Logs        │     │   (Logging)     │
└─────────────────┘     └─────────────────┘

┌─────────────────┐     ┌─────────────────┐
│   Application   │────►│     Jaeger      │
│    Traces       │     │   (Tracing)     │
└─────────────────┘     └─────────────────┘
```

This architecture provides a solid foundation for building a scalable, maintainable, and extensible knowledge management system while maintaining clear separation of concerns and enabling future enhancements.