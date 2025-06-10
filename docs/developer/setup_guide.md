# Memory Engine Setup Guide

This guide will help you set up and configure the Memory Engine system for development and production use.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Infrastructure Setup](#infrastructure-setup)
6. [Verification](#verification)
7. [Development Setup](#development-setup)
8. [Production Deployment](#production-deployment)

## Prerequisites

### System Requirements

- **Python**: 3.8 or higher
- **Docker**: 20.10 or higher (for infrastructure components)
- **Docker Compose**: 1.29 or higher
- **Memory**: At least 4GB RAM (8GB recommended)
- **Storage**: At least 10GB free space

### External Services

- **Google Gemini API**: For embeddings and knowledge extraction
  - Sign up at [Google AI Studio](https://makersuite.google.com/app/apikey)
  - Obtain API key for `text-embedding-004` and `gemini-2.5-flash`

### Optional Dependencies

- **Google Agent Development Kit (ADK)**: For agent functionality
- **JanusGraph**: Graph database (can use Docker setup)
- **Milvus**: Vector database (can use Docker setup)

## Quick Start

### 1. Clone and Install

```bash
# Clone the repository
git clone https://github.com/Celebr4tion/memory-engine.git
cd memory-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
# Required: Google Gemini API key
export GEMINI_API_KEY="your-gemini-api-key-here"

# Optional: Custom database hosts
export JANUSGRAPH_HOST="localhost"
export JANUSGRAPH_PORT="8182"
export MILVUS_HOST="localhost"
export MILVUS_PORT="19530"
```

### 3. Start Infrastructure

```bash
# Start JanusGraph and Milvus using Docker
cd docker
docker-compose up -d

# Wait for services to be ready (may take 2-3 minutes)
docker-compose logs -f
```

### 4. Verify Installation

```bash
# Run basic tests
python -m pytest tests/test_knowledge_node.py -v

# Test with infrastructure (if running)
python -m pytest tests/test_janusgraph_storage.py::TestJanusGraphStorage::test_connect -v
```

### 5. Quick Usage Example

```python
from memory_core.core.knowledge_engine import KnowledgeEngine
from memory_core.model.knowledge_node import KnowledgeNode

# Initialize the engine
engine = KnowledgeEngine()
engine.connect()

# Create a knowledge node
node = KnowledgeNode(
    content="The capital of France is Paris",
    source="User Input",
    rating_truthfulness=0.9
)

# Save to graph
node_id = engine.save_node(node)
print(f"Created node: {node_id}")

# Retrieve the node
retrieved = engine.get_node(node_id)
print(f"Retrieved: {retrieved.content}")
```

## Installation

### Development Installation

```bash
# Clone repository
git clone https://github.com/Celebr4tion/memory-engine.git
cd memory-engine

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (for testing)
pip install pytest pytest-cov pytest-asyncio pytest-env coverage
```

### Production Installation

```bash
# For production, use a requirements.lock file or pip-tools
pip install -r requirements.txt

# Set up systemd service or container deployment
# See production deployment section
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Required Configuration
GEMINI_API_KEY=your-gemini-api-key-here

# Database Configuration (optional, defaults shown)
JANUSGRAPH_HOST=localhost
JANUSGRAPH_PORT=8182
MILVUS_HOST=localhost
MILVUS_PORT=19530

# Test Configuration (optional)
SKIP_INTEGRATION_TESTS=false

# Logging Configuration (optional)
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

### Configuration Options

#### Knowledge Engine Configuration

```python
from memory_core.core.knowledge_engine import KnowledgeEngine

# Configure the engine
engine = KnowledgeEngine(
    host="localhost",           # JanusGraph host
    port=8182,                 # JanusGraph port
    changes_threshold=100,      # Changes before snapshot
    enable_versioning=True,     # Enable change tracking
    enable_snapshots=True       # Enable periodic snapshots
)
```

#### Vector Store Configuration

```python
from memory_core.embeddings.vector_store import VectorStoreMilvus

# Configure vector store
vector_store = VectorStoreMilvus(
    host="localhost",
    port=19530,
    collection_name="memory_engine_embeddings",
    dimension=768  # For text-embedding-004
)
```

#### Embedding Configuration

```python
from memory_core.embeddings.embedding_manager import EmbeddingManager

# The embedding manager automatically uses:
# - Model: text-embedding-004
# - API Key: from GEMINI_API_KEY environment variable
```

## Infrastructure Setup

### Using Docker Compose (Recommended)

The project includes a complete Docker Compose setup for all infrastructure components.

```bash
# Navigate to docker directory
cd docker

# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Stop and remove volumes (clean reset)
docker-compose down -v
```

#### Services Included

- **JanusGraph**: Graph database on port 8182
- **Milvus**: Vector database on port 19530
- **etcd**: Milvus dependency for metadata
- **MinIO**: Milvus dependency for object storage

### Manual Infrastructure Setup

#### JanusGraph Setup

```bash
# Download JanusGraph
wget https://github.com/JanusGraph/janusgraph/releases/download/v0.6.3/janusgraph-0.6.3.zip
unzip janusgraph-0.6.3.zip
cd janusgraph-0.6.3

# Start with Berkeley DB backend
bin/janusgraph-server.sh start
```

#### Milvus Setup

```bash
# Using Docker (simplest method)
docker run -d --name milvus \
  -p 19530:19530 \
  -p 9091:9091 \
  milvusdb/milvus:v2.2.11 \
  milvus run standalone
```

## Verification

### Test Infrastructure Connectivity

```bash
# Test JanusGraph connection
python tests/integration/test_janusgraph_connection.py

# Test Milvus connection
python -c "
from memory_core.embeddings.vector_store import VectorStoreMilvus
vs = VectorStoreMilvus()
print('Milvus available:', vs.connect())
"

# Test Gemini API
python -c "
import os
from memory_core.embeddings.embedding_manager import EmbeddingManager
from memory_core.embeddings.vector_store import VectorStoreMilvus
if os.getenv('GEMINI_API_KEY'):
    vs = VectorStoreMilvus()
    if vs.connect():
        em = EmbeddingManager(vs)
        result = em.generate_embedding('test text')
        print('Gemini API working, embedding length:', len(result))
    else:
        print('Milvus not available')
else:
    print('GEMINI_API_KEY not set')
"
```

### Run Test Suite

```bash
# Run unit tests only
python -m pytest tests/ -k "not integration" -v

# Run all tests (requires infrastructure)
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ --cov=memory_core --cov-report=html
```

### Verify Core Functionality

```python
# Complete verification script
from memory_core.core.knowledge_engine import KnowledgeEngine
from memory_core.embeddings.vector_store import VectorStoreMilvus
from memory_core.embeddings.embedding_manager import EmbeddingManager

# Test knowledge engine
engine = KnowledgeEngine()
connected = engine.connect()
print(f"Knowledge Engine connected: {connected}")

# Test vector storage
vector_store = VectorStoreMilvus()
vs_connected = vector_store.connect()
print(f"Vector Store connected: {vs_connected}")

# Test embedding generation
if vs_connected:
    em = EmbeddingManager(vector_store)
    embedding = em.generate_embedding("test knowledge")
    print(f"Embedding generated: {len(embedding)} dimensions")

# Cleanup
if connected:
    engine.disconnect()
if vs_connected:
    vector_store.disconnect()
```

## Development Setup

### IDE Configuration

#### VS Code

Create `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"],
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true
}
```

#### PyCharm

1. Set Python interpreter to `./venv/bin/python`
2. Mark `memory_core` as source root
3. Configure pytest as test runner
4. Set environment variables in run configurations

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Create .pre-commit-config.yaml
cat > .pre-commit-config.yaml << EOF
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8
  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort
EOF

# Install hooks
pre-commit install
```

### Development Workflow

```bash
# Start development environment
docker-compose -f docker/docker-compose.yml up -d

# Run tests continuously during development
python -m pytest tests/ --watch

# Format code
black memory_core/ tests/
isort memory_core/ tests/

# Type checking (optional)
pip install mypy
mypy memory_core/
```

## Production Deployment

### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY memory_core/ ./memory_core/
COPY . .

CMD ["python", "-m", "memory_core.mcp_integration.mcp_endpoint"]
```

### Environment Configuration

Production `.env` file:

```bash
# API Configuration
GEMINI_API_KEY=your-production-api-key

# Database Configuration
JANUSGRAPH_HOST=janusgraph.your-domain.com
JANUSGRAPH_PORT=8182
MILVUS_HOST=milvus.your-domain.com
MILVUS_PORT=19530

# Logging
LOG_LEVEL=WARNING
PYTHONUNBUFFERED=1

# Performance
WORKERS=4
MAX_MEMORY=2G
```

### Health Checks

```python
# health_check.py
from memory_core.core.knowledge_engine import KnowledgeEngine

def health_check():
    try:
        engine = KnowledgeEngine()
        connected = engine.connect()
        engine.disconnect()
        return {"status": "healthy", "database": connected}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    print(health_check())
```

### Monitoring

```bash
# Example with systemd service
sudo tee /etc/systemd/system/memory-engine.service << EOF
[Unit]
Description=Memory Engine Service
After=network.target

[Service]
Type=simple
User=memory-engine
WorkingDirectory=/opt/memory-engine
Environment=PATH=/opt/memory-engine/venv/bin
EnvironmentFile=/opt/memory-engine/.env
ExecStart=/opt/memory-engine/venv/bin/python -m memory_core.mcp_integration.mcp_endpoint
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable memory-engine
sudo systemctl start memory-engine
```

## Next Steps

1. **Read the [Architecture Guide](architecture.md)** to understand system components
2. **Try the [Example Scripts](../examples/)** for common workflows
3. **Check the [API Documentation](api_reference.md)** for detailed interface information
4. **Review [Troubleshooting Guide](troubleshooting.md)** for common issues

## Support

- **Issues**: Create GitHub issues for bugs and feature requests
- **Documentation**: Check the `docs/` directory for detailed guides
- **Examples**: See `examples/` directory for usage patterns