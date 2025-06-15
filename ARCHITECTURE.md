# Memory Engine Architecture

This document provides an overview of the Memory Engine project structure and organization.

## Current Version: 0.5.0 (Orchestrator Integration)

## Project Structure

```
memory-engine/
├── LICENSE.md                      # Project license
├── README.md                       # Project overview and quick start
├── ARCHITECTURE.md                 # This file - project structure
├── CHANGELOG.md                    # Version history and changes
├── CONTRIBUTING.md                 # Contribution guidelines
├── CODE_OF_CONDUCT.md              # Community code of conduct
├── SECURITY.md                     # Security policy and supported versions
├── CLAUDE.md                       # Development guidance for Claude Code
├── MANIFEST.in                     # Package manifest for distribution
├── setup.py                        # Python package setup (legacy)
├── pyproject.toml                  # Modern Python package configuration
├── requirements.txt                # Python dependencies
├── pytest.ini                      # Test configuration
├── .gitignore                      # Git ignore rules
├── memory_engine_cli.py            # CLI interface (v0.4.0+)
│
├── config/                         # Configuration management
│   ├── README.md                   # Configuration guide
│   ├── config.yaml                 # Base configuration
│   ├── environments/               # Environment-specific configs
│   │   ├── config.development.yaml
│   │   ├── config.testing.yaml
│   │   ├── config.staging.yaml
│   │   └── config.production.yaml
│   └── monitoring/                 # Monitoring configurations
│       └── grafana_dashboard.json
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
│   │   ├── sqlite_storage.py       # SQLite adapter (v0.4.0+)
│   │   ├── json_storage.py         # JSON adapter (v0.4.0+)
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
│   ├── versioning/                 # Version control and history
│   │   ├── __init__.py
│   │   └── revision_manager.py
│   ├── providers/                  # LLM and embedding providers (v0.4.0+)
│   │   ├── __init__.py
│   │   ├── llm/                    # LLM provider implementations
│   │   │   ├── __init__.py
│   │   │   ├── interface.py        # LLMProviderInterface
│   │   │   ├── google_llm.py      # Google Gemini provider
│   │   │   ├── openai_llm.py      # OpenAI provider
│   │   │   ├── anthropic_llm.py   # Anthropic Claude provider
│   │   │   ├── ollama_llm.py      # Ollama local provider
│   │   │   ├── huggingface_llm.py # HuggingFace provider
│   │   │   └── fallback.py         # Fallback chain implementation
│   │   └── embeddings/             # Embedding provider implementations
│   │       ├── __init__.py
│   │       ├── interface.py        # EmbeddingProviderInterface
│   │       └── [provider implementations]
│   ├── vector/                     # Vector store implementations (v0.4.0+)
│   │   ├── __init__.py
│   │   ├── interface.py            # VectorStoreInterface
│   │   ├── milvus/                 # Milvus implementation
│   │   ├── chromadb/               # ChromaDB implementation
│   │   ├── faiss/                  # FAISS implementation
│   │   ├── qdrant/                 # Qdrant implementation
│   │   └── numpy/                  # NumPy implementation
│   ├── performance/                # Performance optimization (v0.4.0+)
│   │   ├── __init__.py
│   │   ├── cache_manager.py        # Intelligent caching with TTL
│   │   ├── connection_pool.py      # Database connection pooling
│   │   ├── batch_optimizer.py      # Batch processing optimization
│   │   ├── performance_monitor.py  # Performance monitoring
│   │   └── metrics_collector.py    # Metrics collection
│   ├── health/                     # Health monitoring (v0.4.0+)
│   │   ├── __init__.py
│   │   ├── health_monitor.py       # System health checks
│   │   ├── component_health.py     # Component-level health
│   │   └── health_aggregator.py    # Health status aggregation
│   ├── migration/                  # Migration tools (v0.4.0+)
│   │   ├── __init__.py
│   │   ├── migration_manager.py    # Migration orchestration
│   │   ├── data_migrator.py        # Data migration logic
│   │   └── migration_strategies.py # Backend-specific strategies
│   ├── backup/                     # Backup and restore (v0.4.0+)
│   │   ├── __init__.py
│   │   ├── backup_manager.py       # Backup orchestration
│   │   ├── restore_manager.py      # Restore operations
│   │   └── backup_strategies.py    # Backup strategies
│   ├── plugins/                    # Plugin architecture (v0.4.0+)
│   │   ├── __init__.py
│   │   ├── plugin_interface.py     # PluginInterface
│   │   ├── plugin_manager.py       # Plugin discovery and loading
│   │   └── plugin_registry.py      # Plugin registration
│   ├── security/                   # Security framework (v0.5.0+)
│   │   ├── __init__.py
│   │   ├── authentication.py       # Authentication mechanisms
│   │   ├── authorization.py        # RBAC implementation
│   │   ├── encryption.py           # Data encryption
│   │   └── audit_logger.py         # Security audit logging
│   └── orchestrator/               # Orchestrator integration (v0.5.0+)
│       ├── __init__.py
│       ├── enhanced_mcp.py         # Enhanced MCP with streaming
│       ├── query_language.py       # GraphQL-like query language
│       ├── event_system.py         # Inter-module event system
│       ├── module_registry.py      # Module registry & capabilities
│       └── data_formats.py         # Standardized data formats
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
│   │   ├── test_advanced_extractor.py
│   │   ├── test_event_system.py    # v0.5.0 event system tests
│   │   └── [other unit tests]
│   ├── integration/                # Integration tests (require services)
│   │   ├── test_janusgraph_connection.py
│   │   ├── test_janusgraph_minimal.py
│   │   ├── test_janusgraph_storage_simple.py
│   │   ├── test_milvus_connection.py
│   │   └── [other integration tests]
│   └── tests/                      # Component-level tests
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
│   │   ├── troubleshooting.md
│   │   └── [provider guides]
│   ├── developer/                  # Developer documentation
│   │   ├── README.md
│   │   ├── setup_guide.md
│   │   ├── architecture.md
│   │   └── configuration_system.md
│   ├── api/                        # API documentation
│   │   ├── README.md
│   │   └── api_reference.md
│   └── security/                   # Security documentation
│       └── README.md
│
├── examples/                       # Usage examples
│   ├── README.md                   # Examples overview
│   ├── basic_usage.py
│   ├── config_example.py
│   ├── enhanced_mcp_example.py
│   ├── knowledge_extraction.py
│   ├── mcp_client_example.py
│   ├── orchestrator_integration_example.py  # v0.5.0
│   ├── security_example.py         # v0.5.0
│   ├── advanced_query_example.py   # v0.5.0
│   └── synthesis_example.py        # v0.5.0
│
├── scripts/                        # Development and utility scripts
│   ├── README.md                   # Scripts documentation
│   ├── setup.sh                    # Development environment setup
│   └── test.sh                     # Test runner script (unit/integration/all)
│
├── docker/                         # Docker configuration
│   └── docker-compose.yml          # Services (JanusGraph, Milvus)
│
├── data/                           # Data storage
│   └── events/                     # Event logs (v0.5.0+)
│
└── logs/                           # Application logs
    └── audit/                      # Audit logs (v0.5.0+)
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
- **Key Features**: Graph storage, versioning, adapter pattern, multiple backends
- **Supported Backends**: JanusGraph, SQLite, JSON
- **Main Files**: `janusgraph_storage.py`, `sqlite_storage.py`, `json_storage.py`, `graph_storage_adapter.py`, `versioned_graph_adapter.py`

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

### v0.4.0 Components (Production Readiness)

#### 8. Provider System (`memory_core/providers/`)
- **Purpose**: Modular LLM and embedding provider system
- **LLM Providers**: Google Gemini, OpenAI, Anthropic, Ollama, HuggingFace
- **Key Features**: Provider interfaces, fallback chains, circuit breakers
- **Main Files**: `llm/interface.py`, `embeddings/interface.py`, `llm/fallback.py`

#### 9. Performance Module (`memory_core/performance/`)
- **Purpose**: System-wide performance optimization
- **Key Features**: 
  - Intelligent caching with TTL
  - Connection pooling for databases
  - Batch processing optimization
  - Real-time performance monitoring
  - Metrics collection and analysis
- **Main Files**: `cache_manager.py`, `connection_pool.py`, `batch_optimizer.py`, `performance_monitor.py`

#### 10. Health Monitoring (`memory_core/health/`)
- **Purpose**: Comprehensive system health monitoring
- **Key Features**: Component health checks, dependency monitoring, health aggregation
- **Main Files**: `health_monitor.py`, `component_health.py`, `health_aggregator.py`

#### 11. Migration Tools (`memory_core/migration/`)
- **Purpose**: Backend migration and data portability
- **Key Features**: Zero-downtime migration, data validation, rollback support
- **Main Files**: `migration_manager.py`, `data_migrator.py`, `migration_strategies.py`

#### 12. Backup/Restore (`memory_core/backup/`)
- **Purpose**: Data backup and disaster recovery
- **Key Features**: Incremental backups, point-in-time recovery, multiple strategies
- **Main Files**: `backup_manager.py`, `restore_manager.py`, `backup_strategies.py`

#### 13. Plugin Architecture (`memory_core/plugins/`)
- **Purpose**: Extensibility through plugins
- **Key Features**: Dynamic loading, plugin interfaces, registry management
- **Main Files**: `plugin_interface.py`, `plugin_manager.py`, `plugin_registry.py`

### v0.5.0 Components (Orchestrator Integration)

#### 14. Security Framework (`memory_core/security/`)
- **Purpose**: Comprehensive security implementation
- **Key Features**: 
  - Authentication mechanisms
  - Role-based access control (RBAC)
  - Data encryption at rest and in transit
  - Security audit logging
- **Main Files**: `authentication.py`, `authorization.py`, `encryption.py`, `audit_logger.py`

#### 15. Orchestrator Module (`memory_core/orchestrator/`)
- **Purpose**: Inter-module communication and orchestration
- **Key Features**:
  - Enhanced MCP with streaming support
  - GraphQL-like query language
  - Event-driven architecture
  - Module registry with capability advertisement
  - Standardized data formats
- **Main Files**: `enhanced_mcp.py`, `query_language.py`, `event_system.py`, `module_registry.py`, `data_formats.py`

### Supporting Components

#### 16. Rating System (`memory_core/rating/`)
- **Purpose**: Quality assessment and rating of knowledge
- **Key Features**: Multiple rating dimensions, quality metrics
- **Main Files**: `rating_system.py`

#### 17. AI Agents (`memory_core/agents/`)
- **Purpose**: Automated knowledge processing and management
- **Key Features**: Intelligent knowledge curation, automated workflows
- **Main Files**: `knowledge_agent.py`

#### 18. Versioning (`memory_core/versioning/`)
- **Purpose**: Version control and history management
- **Key Features**: Revision tracking, change management, rollback
- **Main Files**: `revision_manager.py`

#### 19. CLI Interface (`memory_engine_cli.py`)
- **Purpose**: Command-line interface for system management
- **Key Features**: 
  - System initialization and configuration
  - Health monitoring and diagnostics
  - Backend migration commands
  - Backup and restore operations
  - Plugin management
  - Orchestrator commands (v0.5.0)

## Design Principles

### 1. Modularity
- Each component has a single, well-defined responsibility
- Components communicate through well-defined interfaces
- Easy to swap implementations (e.g., different vector databases)
- Plugin architecture for extensibility

### 2. Configuration-Driven
- All behavior controlled through configuration
- Environment-specific settings without code changes
- Type-safe configuration with validation
- Hot-reloading of configuration changes

### 3. Extensibility
- Plugin architecture for new storage backends
- Configurable processing pipelines
- Support for different AI models and APIs
- Provider interfaces for LLM and embedding flexibility

### 4. Reliability
- Comprehensive testing at unit, integration, and component levels
- Error handling and graceful degradation
- Monitoring and observability built-in
- Circuit breakers and fallback mechanisms

### 5. Performance
- Intelligent caching strategies
- Connection pooling and resource management
- Batch processing optimization
- Asynchronous operations where beneficial

### 6. Security
- Authentication and authorization
- Data encryption
- Audit logging
- Principle of least privilege

### 7. Developer Experience
- Clear project structure and documentation
- Automated setup and testing scripts
- Comprehensive examples and guides
- CLI tools for common operations

## Data Flow

```
Raw Text Input
      ↓
[Advanced Extractor] → Knowledge Units
      ↓
[Merging & Validation] → Validated Knowledge
      ↓
[Knowledge Engine] → Storage + Embeddings
      ↓                    ↓
[Storage Backend] ← → [Vector Store]
(JanusGraph/SQLite/JSON)  (Milvus/ChromaDB/etc)
      ↓
[Rating System] → Quality Scores
      ↓
[Versioning] → Revision History
      ↓
[Security Layer] → Access Control
      ↓
[Performance Layer] → Optimization
      ↓
[MCP/Orchestrator] → External Access
```

## Technology Stack

### Core Technologies
- **Python 3.8+**: Primary language
- **asyncio**: Asynchronous programming
- **Type hints**: Static type checking

### Storage Options
- **Graph Databases**: JanusGraph, SQLite (graph mode)
- **Vector Databases**: Milvus, ChromaDB, FAISS, Qdrant
- **Simple Storage**: JSON files (development)

### LLM Providers
- **Google Gemini**: Primary LLM and embeddings
- **OpenAI**: GPT models and embeddings
- **Anthropic**: Claude models
- **Ollama**: Local LLM support
- **HuggingFace**: Open-source models

### Key Python Libraries
- **google-genai**: Google Gemini API client
- **openai**: OpenAI API client
- **anthropic**: Anthropic API client
- **gremlin-python**: JanusGraph/Gremlin client
- **pymilvus**: Milvus vector database client
- **chromadb**: ChromaDB vector database
- **pyyaml**: Configuration file parsing
- **sqlalchemy**: Database abstractions
- **networkx**: Graph algorithms and analysis
- **fastapi**: REST API framework (optional)
- **pydantic**: Data validation
- **pytest**: Testing framework
- **black**: Code formatting
- **mypy**: Type checking

## Getting Started

1. **Setup**: Run `./scripts/setup.sh` for automated environment setup
2. **Configuration**: Update `config/.env` with your API keys
3. **Services**: Start with `docker-compose up -d` (optional for production backends)
4. **Initialize**: Run `memory-engine init --backend=sqlite` to initialize
5. **Testing**: Run `./scripts/test.sh unit` for quick tests
6. **Examples**: Try `python examples/basic_usage.py`

For detailed setup instructions, see [docs/developer/setup_guide.md](docs/developer/setup_guide.md).

## CLI Usage (v0.4.0+)

```bash
# System management
memory-engine init --backend=sqlite
memory-engine status
memory-engine health-check --detailed

# Migration operations
memory-engine migrate --from=sqlite --to=janusgraph

# Backup operations
memory-engine backup --strategy=full --output=/backups/

# v0.5.0 Orchestrator commands
memory-engine mcp stream-query --query="AI research" --batch-size=50
memory-engine events list --status=pending
memory-engine modules list --capabilities
memory-engine query build --type=nodes --filter="content contains 'AI'"
```

## Contributing

- Follow the established project structure
- Add tests for new components
- Update documentation for changes
- Use the provided scripts for development tasks
- Run `./scripts/test.sh check` before committing
- Update CHANGELOG.md for significant changes
- See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines

## Version History

See [CHANGELOG.md](CHANGELOG.md) for detailed version history and changes.

Current version: **0.5.0 (Orchestrator Integration)**