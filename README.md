# Memory Engine

A sophisticated semantic knowledge management system that combines graph-based knowledge representation with modern vector embeddings for intelligent information storage, retrieval, and synthesis.

## 🌟 Overview

Memory Engine is a comprehensive knowledge management platform that transforms unstructured text into a structured, searchable knowledge graph. By combining the relationship modeling power of graph databases with the semantic understanding of vector embeddings, it creates a foundation for building intelligent applications that can understand, connect, and reason about information.

## 🚧 Project Status

**This project is currently in active development (v0.1.0) and should be considered experimental.**

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

- 🧠 **Intelligent Knowledge Extraction**: Uses Google Gemini API to extract structured knowledge from raw text
- 🕸️ **Automatic Relationship Discovery**: Detects and creates relationships between knowledge entities
- 🔍 **Semantic Search**: Vector-based similarity search for contextual information retrieval
- 🔐 **Enterprise Security**: Comprehensive authentication, RBAC, encryption, and audit logging
- 🛡️ **Privacy Controls**: Fine-grained knowledge privacy levels and access control
- 📊 **Quality Enhancement**: Automated quality assessment and contradiction resolution
- 📚 **Version Control**: Complete change tracking and rollback capabilities
- 🔗 **Flexible Integration**: MCP (Module Communication Protocol) interface for external systems
- 🤖 **Agent Support**: Google ADK integration for conversational knowledge interactions
- ⚡ **Real-time Processing**: Concurrent processing of knowledge ingestion and retrieval
- 📈 **Monitoring**: Performance monitoring, health checks, and observability

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Docker & Docker Compose
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

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
# Set your Gemini API key
GEMINI_API_KEY="your-gemini-api-key"

# Optional: Set environment (defaults to development)
ENVIRONMENT="development"
```

### 3. Start Infrastructure

```bash
# Start JanusGraph and Milvus
cd docker
docker-compose up -d

# Wait for services to initialize (2-3 minutes)
docker-compose logs -f
```

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
| Vector Database | Milvus 2.2.11 | Similarity search |
| LLM API | Google Gemini | Knowledge extraction & embeddings |
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
- **Security**: Report security issues via [SECURITY.md](SECURITY.md)

### Status

- ⚠️ **Development Status**: Alpha version - breaking changes expected
- 📝 **Documentation**: Basic setup and usage guides available
- 🧪 **Testing**: Core functionality tested, expanding coverage
- 🔧 **Stability**: Experimental - not recommended for production use yet

---

**Memory Engine** - *Transforming information into intelligence* 🧠✨
