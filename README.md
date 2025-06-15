# Memory Engine

A semantic knowledge management system that combines graph-based knowledge representation with vector embeddings for information storage, retrieval, and synthesis.

## 🌟 Overview

Memory Engine is an experimental knowledge management system that transforms unstructured text into a structured, searchable knowledge graph. It combines graph databases with vector embeddings to create a foundation for applications that can understand, connect, and reason about information.

## ⚠️ Important Notice

**This is a personal open-source project developed for learning and research purposes. No guarantees are made regarding reliability, security, or suitability for production use. Use at your own risk.**

## 🚧 Project Status

**This project is currently in active development (v0.5.0 - Orchestrator Integration) and should be considered experimental.**

### Vision

Our goal is to create a truly open and accessible knowledge management system that works with:
- **Any AI model**: Commercial APIs (OpenAI, Anthropic, Google) and local models (Ollama, Hugging Face)
- **Any deployment**: From laptop development to distributed production systems
- **Any data**: Text, documents, structured data, and multimedia content

We aim to eliminate dependency on paid APIs by providing full support for local model execution, making advanced knowledge management accessible to everyone.

## 🎯 What Memory Engine Does

**Input**: Unstructured text, documents, or data

**Output**: Structured knowledge with automatic relationships and semantic search capabilities

### Core Functions

1. **Knowledge Ingestion**: Feed text/documents → Engine extracts entities, facts, and relationships → Stores in graph database
2. **Knowledge Retrieval**: Query in natural language → Engine searches semantically → Returns relevant information with context
3. **Automatic Processing**: The engine handles complexity internally - relationship discovery, quality assessment, versioning, and optimization

### Key Features

#### 🧠 **AI & Language Models**
- **Multi-LLM Support**: 5 different LLM providers (Gemini, OpenAI, Anthropic, Ollama, HuggingFace)
- **LLM Independence**: Fallback chains and circuit breaker pattern for resilience
- **Local Operation**: Complete offline capabilities with Ollama and HuggingFace Transformers
- **Automatic Relationship Discovery**: Detects and creates relationships between knowledge entities

#### ⚡ **Performance & Production**
- **Advanced Caching**: Multi-level caching with TTL, memory limits, and intelligent invalidation
- **Connection Pooling**: Health monitoring and configurable pool management
- **Query Optimization**: Prepared statements and batch processing for high throughput
- **Memory Management**: Garbage collection optimization and automatic resource cleanup

#### 🛠️ **Operations & Management**
- **Health Monitoring**: Comprehensive system health checks and service monitoring
- **CLI Tools**: Complete command-line interface for all management operations
- **Migration Tools**: Backend migration utilities with multiple strategies
- **Backup & Restore**: Automated backup with compression and retention policies

#### 🔌 **Extensibility**
- **Plugin Architecture**: Custom storage backends, LLM providers, and embedding providers
- **Data Export/Import**: Multiple formats (JSON, CSV, XML, GraphML, Cypher, Gremlin, RDF)
- **Metrics Collection**: Prometheus-compatible metrics with counters, gauges, histograms

#### 🔍 **Knowledge Management**
- **Semantic Search**: Multi-provider vector embeddings with modular vector stores
- **Modular Storage**: Choose from JanusGraph, SQLite, or JSON backends
- **Quality Enhancement**: Automated quality assessment and contradiction resolution
- **Version Control**: Complete change tracking and rollback capabilities

#### 🔐 **Security & Integration**
- **Basic Security Features**: Authentication, RBAC, encryption, and audit logging (educational purposes)
- **Privacy Controls**: Fine-grained knowledge privacy levels and access control
- **Flexible Integration**: MCP (Module Communication Protocol) interface for external systems
- **Agent Support**: Google ADK integration for conversational knowledge interactions

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Docker & Docker Compose (optional, for JanusGraph/Milvus)
- At least one LLM provider API key:
  - Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))
  - OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
  - Anthropic API key ([Get one here](https://console.anthropic.com/))
  - Or use local models with Ollama or HuggingFace (no API key needed)

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Celebr4tion/memory-engine.git
cd memory-engine

# Run automated setup
./scripts/setup.sh
```

The setup script will:
- Check Python version compatibility
- Create virtual environment
- Install dependencies
- Create configuration template
- Set up development tools

### 2. Environment Setup

```bash
# Edit the .env file created by setup
# Set your preferred LLM provider API keys (at least one required)
GOOGLE_API_KEY="your-gemini-api-key"           # For Gemini
OPENAI_API_KEY="your-openai-api-key"           # For OpenAI GPT
ANTHROPIC_API_KEY="your-anthropic-api-key"     # For Claude
HUGGINGFACE_API_KEY="your-hf-api-key"          # For HuggingFace API (optional)

# Optional: Set environment (defaults to development)
ENVIRONMENT="development"
```

### 3. Start Infrastructure (Optional)

For production storage backends:

```bash
# Start JanusGraph and Milvus (optional, for production storage)
cd docker
docker-compose up -d

# Wait for services to initialize (2-3 minutes)
docker-compose logs -f
```

For development, you can use lightweight storage backends (SQLite/JSON) that don't require external services.

### 4. Basic Usage

```python
from memory_core.core.knowledge_engine import KnowledgeEngine
from memory_core.model.knowledge_node import KnowledgeNode

# Initialize the system
engine = KnowledgeEngine()
engine.connect()

# Create knowledge from text
node = KnowledgeNode(
    content="Machine learning is a subset of artificial intelligence",
    source="AI Textbook",
    rating_truthfulness=0.9
)

# Save to knowledge graph
node_id = engine.save_node(node)
print(f"Created knowledge node: {node_id}")

# Retrieve and explore
retrieved = engine.get_node(node_id)
print(f"Content: {retrieved.content}")
```

### 5. CLI Management (v0.4.0+) & Orchestrator Features (v0.5.0+)

Memory Engine includes a comprehensive CLI for production management:

```bash
# Initialize a new Memory Engine instance
memory-engine init --backend=sqlite --embedding=sentence_transformers

# Check system health
memory-engine health-check --detailed

# Migrate between storage backends
memory-engine migrate --from=sqlite --to=janusgraph --verify

# Export knowledge graph data
memory-engine export --format=json --output=backup.json --include-metadata

# Import data from various formats
memory-engine import --file=data.json --merge-duplicates

# Create system backups
memory-engine backup --strategy=full --compression=gzip

# Restore from backup
memory-engine restore --backup=backup_12345 --clear-existing

# Manage plugins
memory-engine plugins list --type=storage
memory-engine plugins install custom-backend

# Configuration management
memory-engine config show --section=storage
memory-engine config set storage.backend janusgraph
memory-engine config validate

# System status
memory-engine status
memory-engine version

# Orchestrator Integration (v0.5.0+)
# Start streaming MCP operations
memory-engine mcp stream-query --query="knowledge about AI" --batch-size=50

# Manage event system
memory-engine events list --status=pending
memory-engine events replay --from-timestamp=1234567890

# Module registry management
memory-engine modules list --capabilities
memory-engine modules register my-custom-module

# Advanced GraphQL-like queries
memory-engine query build --type=nodes --filter="content contains 'AI'" --limit=10
memory-engine query execute --query-file=complex_query.json
```

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [📋 Setup Guide](docs/developer/setup_guide.md) | Complete installation and configuration instructions |
| [⚙️ Configuration](docs/user/configuration.md) | Basic configuration and environment setup |
| [🔧 Advanced Configuration](docs/developer/configuration_system.md) | Advanced configuration system |
| [🏗️ Architecture](docs/developer/architecture.md) | System architecture and component interactions |
| [🏗️ Project Structure](ARCHITECTURE.md) | Detailed project organization and structure |
| [📡 API Reference](docs/api/api_reference.md) | Complete API documentation including MCP interface |
| [🔐 Security Framework](docs/security/README.md) | Authentication, RBAC, encryption, and privacy controls |
| [🔧 Troubleshooting](docs/user/troubleshooting.md) | Common issues and solutions |

## 💻 Examples

Explore practical examples in the [`examples/`](examples/) directory:

- [**Basic Usage**](examples/basic_usage.py): Core operations and workflows
- [**Knowledge Extraction**](examples/knowledge_extraction.py): Text processing and knowledge extraction
- [**MCP Integration**](examples/mcp_client_example.py): Using the Module Communication Protocol
- [**Security Framework**](examples/security_example.py): Authentication, RBAC, encryption, and privacy controls
- [**Advanced Queries**](examples/advanced_query_example.py): Complex querying and analytics
- [**Knowledge Synthesis**](examples/synthesis_example.py): Question answering and insight discovery

### Run Examples

```bash
# Ensure infrastructure is running
cd docker && docker-compose up -d

# Run basic usage example
python examples/basic_usage.py

# Run knowledge extraction demo
python examples/knowledge_extraction.py

# Test MCP interface
python examples/mcp_client_example.py

# Try configuration system
python examples/config_example.py
```

## 🧪 Testing

Memory Engine includes a comprehensive test suite organized by type:

```bash
# Run all tests
./scripts/test.sh all

# Run only unit tests (fast, no external dependencies)
./scripts/test.sh unit

# Run integration tests (requires JanusGraph and Milvus)
./scripts/test.sh integration

# Run tests with coverage report
./scripts/test.sh coverage

# Run specific test file
./scripts/test.sh --file config_manager
```

Test organization:
- **Unit Tests** (`tests/unit/`): Fast, isolated tests
- **Integration Tests** (`tests/integration/`): Tests requiring external services
- **Component Tests** (`tests/`): End-to-end component testing

## 🏗️ Architecture

Memory Engine uses a sophisticated multi-layer architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Application Layer                            │
├─────────────────┬─────────────────┬─────────────────┬───────────┤
│   Python API    │   MCP Interface │  Knowledge Agent│ REST API  │
├─────────────────┴─────────────────┴─────────────────┴───────────┤
│                    Knowledge Engine Core                        │
├─────────────────┬─────────────────┬─────────────────┬───────────┤
│   Knowledge     │   Relationship  │    Versioning   │  Rating   │
│   Processing    │   Extraction    │    Manager      │  System   │
├─────────────────┼─────────────────┼─────────────────┼───────────┤
│   Graph Store   │   Vector Store  │   Embedding     │  LLM API  │
│  (JanusGraph)   │   (Milvus)      │   Manager       │ (Gemini)  │
└─────────────────┴─────────────────┴─────────────────┴───────────┘
```

### Core Components

- **Modular Graph Storage**: Multiple backend options (JanusGraph, SQLite, JSON file)
- **Vector Database (Milvus)**: Enables semantic similarity search
- **Embedding System**: Generates and manages vector representations
- **Processing Pipeline**: Extracts and structures knowledge from text
- **Versioning System**: Tracks changes and enables rollbacks
- **MCP Interface**: Standardized API for external integration

### Storage Backend Options

Choose the storage backend that fits your deployment needs:

- **🏢 JanusGraph**: Production-grade distributed graph database
- **💾 SQLite**: Single-user deployments with SQL capabilities  
- **📄 JSON File**: Development and testing with human-readable storage

## 🔧 Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Graph Storage | JanusGraph / SQLite / JSON | Knowledge relationships |
| Vector Database | Milvus / ChromaDB / NumPy | Similarity search |
| LLM Providers | Gemini / OpenAI / Anthropic / Ollama / HuggingFace | Knowledge extraction |
| Embedding Providers | Gemini / OpenAI / Sentence Transformers / Ollama | Vector generation |
| Agent Framework | Google ADK | Conversational interfaces |
| Web Framework | FastAPI | REST API endpoints |
| Language | Python 3.8+ | Core implementation |

## 🧪 Development

### Running Tests

```bash
# Unit tests only
pytest tests/ -k "not integration" -v

# All tests (requires infrastructure)
pytest tests/ -v

# With coverage
pytest tests/ --cov=memory_core --cov-report=html
```

### Development Setup

```bash
# Install development dependencies
pip install pytest pytest-cov black isort mypy

# Format code
black memory_core/ tests/
isort memory_core/ tests/

# Type checking
mypy memory_core/

# Pre-commit hooks
pip install pre-commit
pre-commit install
```

## 📊 Performance

Performance characteristics will vary depending on your hardware, data complexity, and configuration. We recommend testing with your specific use case and data to establish realistic benchmarks.

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Ensure all tests pass (`pytest`)
5. Format code (`black . && isort .`)
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Development Standards

- **Code Quality**: All code must pass linting and type checking
- **Testing**: Maintain >90% test coverage
- **Documentation**: Update docs for any API changes
- **Performance**: Benchmark performance-critical changes

## 📝 License

This project is licensed under the [Hippocratic License 3.0](LICENSE.md) - an ethical source license that promotes responsible use of software while protecting human rights and environmental sustainability.

## 🆘 Support

### Getting Help

- 📖 **Documentation**: Check the [`docs/`](docs/) directory
- 🐛 **Issues**: Report bugs or request features via [GitHub Issues](https://github.com/Celebr4tion/memory-engine/issues)
- 💬 **Discussions**: Join conversations in [GitHub Discussions](https://github.com/Celebr4tion/memory-engine/discussions)
- 🔧 **Troubleshooting**: See the [troubleshooting guide](docs/user/troubleshooting.md)

### Community

- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines
- **Code of Conduct**: Please read our [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- **Security**: Report security issues via [SECURITY.md](SECURITY.md)

### Status

- ⚠️ **Development Status**: Alpha version - breaking changes expected
- 📝 **Documentation**: Basic setup and usage guides available
- 🧪 **Testing**: Core functionality tested, expanding coverage
- 🔧 **Stability**: Experimental - not recommended for production use yet

---

**Memory Engine** - *Transforming information into intelligence* 🧠✨
