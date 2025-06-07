# Memory Engine Architecture

This document provides an overview of the Memory Engine project structure and organization.

## Project Structure

```
memory-engine/
├── LICENSE.md                      # Project license
├── README.md                       # Project overview and quick start
├── ARCHITECTURE.md                 # This file - project structure
├── MANIFEST.in                     # Package manifest for distribution
├── setup.py                        # Python package setup (legacy)
├── pyproject.toml                  # Modern Python package configuration
├── requirements.txt                # Python dependencies
├── pytest.ini                      # Test configuration
├── .gitignore                      # Git ignore rules
│
├── config/                         # Configuration management
│   ├── README.md                   # Configuration guide
│   ├── config.yaml                 # Base configuration
│   └── environments/               # Environment-specific configs
│       ├── config.development.yaml
│       ├── config.testing.yaml
│       ├── config.staging.yaml
│       └── config.production.yaml
│
├── memory_core/                    # Main package
│   ├── __init__.py
│   ├── config/                     # Configuration management system
│   │   ├── __init__.py
│   │   └── config_manager.py
│   ├── core/                       # Core engine components
│   │   └── knowledge_engine.py
│   ├── db/                         # Database adapters and storage
│   │   ├── __init__.py
│   │   ├── graph_storage_adapter.py
│   │   ├── janusgraph_storage.py
│   │   └── versioned_graph_adapter.py
│   ├── embeddings/                 # Embedding and vector operations
│   │   ├── embedding_manager.py
│   │   └── vector_store.py
│   ├── ingestion/                  # Data ingestion and processing
│   │   ├── advanced_extractor.py
│   │   ├── merging.py
│   │   └── relationship_extractor.py
│   ├── mcp_integration/            # MCP (Model Context Protocol) integration
│   │   ├── __init__.py
│   │   ├── enhanced_mcp_endpoint.py
│   │   └── mcp_endpoint.py
│   ├── model/                      # Data models and schemas
│   │   ├── knowledge_node.py
│   │   └── relationship.py
│   ├── rating/                     # Rating and quality assessment
│   │   ├── __init__.py
│   │   └── rating_system.py
│   ├── agents/                     # AI agents and automation
│   │   ├── __init__.py
│   │   └── knowledge_agent.py
│   └── versioning/                 # Version control and history
│       ├── __init__.py
│       └── revision_manager.py
│
├── tests/                          # Test suite
│   ├── README.md                   # Testing guide
│   ├── conftest.py                 # Shared test configuration
│   ├── unit/                       # Unit tests (fast, isolated)
│   │   ├── test_config_manager.py
│   │   ├── test_knowledge_node.py
│   │   ├── test_relationship.py
│   │   ├── test_rating_system.py
│   │   ├── test_embedding_manager.py
│   │   ├── test_vector_store.py
│   │   └── test_advanced_extractor.py
│   ├── integration/                # Integration tests (require services)
│   │   ├── test_janusgraph_connection.py
│   │   ├── test_janusgraph_minimal.py
│   │   ├── test_janusgraph_storage_simple.py
│   │   └── test_milvus_connection.py
│   └── [component tests]           # Component-level tests
│       ├── test_knowledge_engine.py
│       ├── test_janusgraph_storage.py
│       ├── test_mcp_endpoint.py
│       ├── test_enhanced_mcp_endpoint.py
│       ├── test_merging.py
│       ├── test_relationship_extractor.py
│       ├── test_revision_manager.py
│       ├── test_versioned_graph_adapter.py
│       ├── test_graph_storage_adapter.py
│       └── test_knowledge_agent.py
│
├── docs/                           # Documentation
│   ├── README.md                   # Documentation overview
│   ├── user/                       # User documentation
│   │   ├── README.md
│   │   ├── configuration.md
│   │   └── troubleshooting.md
│   ├── developer/                  # Developer documentation
│   │   ├── README.md
│   │   ├── setup_guide.md
│   │   ├── architecture.md
│   │   └── configuration_system.md
│   └── api/                        # API documentation
│       ├── README.md
│       └── api_reference.md
│
├── examples/                       # Usage examples
│   ├── README.md                   # Examples overview
│   ├── basic_usage.py
│   ├── config_example.py
│   ├── enhanced_mcp_example.py
│   ├── knowledge_extraction.py
│   └── mcp_client_example.py
│
├── scripts/                        # Development and utility scripts
│   ├── README.md                   # Scripts documentation
│   ├── setup.sh                    # Development environment setup
│   └── test.sh                     # Test runner script
│
└── docker/                        # Docker configuration
    └── docker-compose.yml          # Services (JanusGraph, Milvus)
```

## Architecture Overview

### Core Components

#### 1. Configuration Management (`memory_core/config/`)
- **Purpose**: Centralized configuration system with environment-specific overrides
- **Key Features**: Validation, file watching, type safety, multiple environments
- **Main Files**: `config_manager.py`

#### 2. Knowledge Engine (`memory_core/core/`)
- **Purpose**: Main orchestrator for knowledge management operations
- **Key Features**: High-level API, component coordination, workflow management
- **Main Files**: `knowledge_engine.py`

#### 3. Database Layer (`memory_core/db/`)
- **Purpose**: Storage abstractions and adapters for different databases
- **Key Features**: Graph storage, versioning, adapter pattern
- **Main Files**: `janusgraph_storage.py`, `graph_storage_adapter.py`, `versioned_graph_adapter.py`

#### 4. Embeddings & Vector Operations (`memory_core/embeddings/`)
- **Purpose**: Vector embeddings generation and similarity search
- **Key Features**: Multiple embedding models, vector databases, similarity search
- **Main Files**: `embedding_manager.py`, `vector_store.py`

#### 5. Data Ingestion (`memory_core/ingestion/`)
- **Purpose**: Processing raw data into structured knowledge
- **Key Features**: Text extraction, knowledge unit creation, relationship extraction
- **Main Files**: `advanced_extractor.py`, `relationship_extractor.py`, `merging.py`

#### 6. Data Models (`memory_core/model/`)
- **Purpose**: Core data structures and schemas
- **Key Features**: Knowledge nodes, relationships, type safety
- **Main Files**: `knowledge_node.py`, `relationship.py`

#### 7. MCP Integration (`memory_core/mcp_integration/`)
- **Purpose**: Model Context Protocol integration for AI systems
- **Key Features**: MCP endpoints, protocol compliance, AI agent communication
- **Main Files**: `mcp_endpoint.py`, `enhanced_mcp_endpoint.py`

### Supporting Components

#### 8. Rating System (`memory_core/rating/`)
- **Purpose**: Quality assessment and rating of knowledge
- **Key Features**: Multiple rating dimensions, quality metrics
- **Main Files**: `rating_system.py`

#### 9. AI Agents (`memory_core/agents/`)
- **Purpose**: Automated knowledge processing and management
- **Key Features**: Intelligent knowledge curation, automated workflows
- **Main Files**: `knowledge_agent.py`

#### 10. Versioning (`memory_core/versioning/`)
- **Purpose**: Version control and history management
- **Key Features**: Revision tracking, change management, rollback
- **Main Files**: `revision_manager.py`

## Design Principles

### 1. Modularity
- Each component has a single, well-defined responsibility
- Components communicate through well-defined interfaces
- Easy to swap implementations (e.g., different vector databases)

### 2. Configuration-Driven
- All behavior controlled through configuration
- Environment-specific settings without code changes
- Type-safe configuration with validation

### 3. Extensibility
- Plugin architecture for new storage backends
- Configurable processing pipelines
- Support for different AI models and APIs

### 4. Reliability
- Comprehensive testing at unit, integration, and component levels
- Error handling and graceful degradation
- Monitoring and observability built-in

### 5. Developer Experience
- Clear project structure and documentation
- Automated setup and testing scripts
- Comprehensive examples and guides

## Data Flow

```
Raw Text Input
      ↓
[Advanced Extractor] → Knowledge Units
      ↓
[Merging & Validation] → Validated Knowledge
      ↓
[Knowledge Engine] → Storage + Embeddings
      ↓
[JanusGraph Storage] ← → [Vector Store (Milvus)]
      ↓
[Rating System] → Quality Scores
      ↓
[Versioning] → Revision History
      ↓
[MCP Integration] → AI Agent Access
```

## Dependencies

### External Services
- **JanusGraph**: Graph database for knowledge storage
- **Milvus**: Vector database for embeddings and similarity search
- **Google Gemini**: AI model for text processing and embeddings

### Key Python Libraries
- **google-genai**: Google Gemini API client
- **gremlin-python**: JanusGraph/Gremlin client
- **pymilvus**: Milvus vector database client
- **pyyaml**: Configuration file parsing
- **sqlalchemy**: Database abstractions
- **networkx**: Graph algorithms and analysis

## Getting Started

1. **Setup**: Run `./scripts/setup.sh` for automated environment setup
2. **Configuration**: Update `config/.env` with your API keys
3. **Services**: Start with `docker-compose up -d` (optional)
4. **Testing**: Run `./scripts/test.sh unit` for quick tests
5. **Examples**: Try `python examples/basic_usage.py`

For detailed setup instructions, see [docs/developer/setup_guide.md](docs/developer/setup_guide.md).

## Contributing

- Follow the established project structure
- Add tests for new components
- Update documentation for changes
- Use the provided scripts for development tasks
- See [docs/developer/README.md](docs/developer/README.md) for development workflow