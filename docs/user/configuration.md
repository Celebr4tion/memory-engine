# Memory Engine Configuration Guide

This document provides comprehensive information about all configuration options, environment variables, and settings for the Memory Engine system.

## Table of Contents

1. [Environment Variables](#environment-variables)
2. [Configuration Files](#configuration-files)
3. [Component Configuration](#component-configuration)
4. [Runtime Configuration](#runtime-configuration)
5. [Security Configuration](#security-configuration)
6. [Performance Tuning](#performance-tuning)
7. [Environment-Specific Settings](#environment-specific-settings)

## Environment Variables

### Required Variables

#### `GEMINI_API_KEY`
- **Required**: Yes
- **Type**: String
- **Description**: Google Gemini API key for embedding generation and knowledge extraction
- **Example**: `GEMINI_API_KEY=AIzaSyC-abc123def456ghi789jkl012mno345pqr`
- **How to obtain**: 
  1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
  2. Create a new API key
  3. Copy the key value

### Optional Database Variables

#### `JANUSGRAPH_HOST`
- **Required**: No
- **Type**: String
- **Default**: `localhost`
- **Description**: Hostname or IP address of the JanusGraph server
- **Example**: `JANUSGRAPH_HOST=janusgraph.example.com`

#### `JANUSGRAPH_PORT`
- **Required**: No
- **Type**: Integer
- **Default**: `8182`
- **Description**: Port number for JanusGraph Gremlin server
- **Example**: `JANUSGRAPH_PORT=8182`

#### `MILVUS_HOST`
- **Required**: No
- **Type**: String
- **Default**: `localhost`
- **Description**: Hostname or IP address of the Milvus vector database
- **Example**: `MILVUS_HOST=milvus.example.com`

#### `MILVUS_PORT`
- **Required**: No
- **Type**: Integer
- **Default**: `19530`
- **Description**: Port number for Milvus server
- **Example**: `MILVUS_PORT=19530`

### Testing Variables

#### `SKIP_INTEGRATION_TESTS`
- **Required**: No
- **Type**: Boolean (string)
- **Default**: `true`
- **Description**: Skip integration tests that require external services
- **Values**: `true`, `false`
- **Example**: `SKIP_INTEGRATION_TESTS=false`

### Logging Variables

#### `LOG_LEVEL`
- **Required**: No
- **Type**: String
- **Default**: `INFO`
- **Description**: Python logging level
- **Values**: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- **Example**: `LOG_LEVEL=DEBUG`

#### `LOG_FORMAT`
- **Required**: No
- **Type**: String
- **Default**: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`
- **Description**: Python logging format string
- **Example**: `LOG_FORMAT=%(levelname)s:%(name)s:%(message)s`

### Performance Variables

#### `PYTHONUNBUFFERED`
- **Required**: No (recommended for containers)
- **Type**: Boolean (string)
- **Default**: Not set
- **Description**: Force Python output to be unbuffered
- **Example**: `PYTHONUNBUFFERED=1`

#### `WORKERS`
- **Required**: No
- **Type**: Integer
- **Default**: `1`
- **Description**: Number of worker processes for production deployment
- **Example**: `WORKERS=4`

## Configuration Files

### `.env` File Structure

Create a `.env` file in the project root for local development:

```bash
# =============================================================================
# Memory Engine Configuration
# =============================================================================

# -----------------------------------------------------------------------------
# Required Configuration
# -----------------------------------------------------------------------------
GEMINI_API_KEY=your-api-key-here

# -----------------------------------------------------------------------------
# Database Configuration
# -----------------------------------------------------------------------------
JANUSGRAPH_HOST=localhost
JANUSGRAPH_PORT=8182
MILVUS_HOST=localhost
MILVUS_PORT=19530

# -----------------------------------------------------------------------------
# Development Configuration
# -----------------------------------------------------------------------------
LOG_LEVEL=DEBUG
SKIP_INTEGRATION_TESTS=false

# -----------------------------------------------------------------------------
# Production Configuration (uncomment for production)
# -----------------------------------------------------------------------------
# LOG_LEVEL=WARNING
# PYTHONUNBUFFERED=1
# WORKERS=4
```

### Environment-Specific Files

#### Development (`.env.development`)
```bash
GEMINI_API_KEY=your-dev-api-key
JANUSGRAPH_HOST=localhost
JANUSGRAPH_PORT=8182
MILVUS_HOST=localhost
MILVUS_PORT=19530
LOG_LEVEL=DEBUG
SKIP_INTEGRATION_TESTS=false
```

#### Testing (`.env.testing`)
```bash
GEMINI_API_KEY=your-test-api-key
JANUSGRAPH_HOST=localhost
JANUSGRAPH_PORT=8182
MILVUS_HOST=localhost
MILVUS_PORT=19530
LOG_LEVEL=INFO
SKIP_INTEGRATION_TESTS=true
```

#### Production (`.env.production`)
```bash
GEMINI_API_KEY=your-prod-api-key
JANUSGRAPH_HOST=janusgraph-prod.internal
JANUSGRAPH_PORT=8182
MILVUS_HOST=milvus-prod.internal
MILVUS_PORT=19530
LOG_LEVEL=WARNING
PYTHONUNBUFFERED=1
WORKERS=4
```

## Component Configuration

### Knowledge Engine Configuration

```python
from memory_core.core.knowledge_engine import KnowledgeEngine

# Basic configuration
engine = KnowledgeEngine(
    host="localhost",              # JanusGraph host
    port=8182,                    # JanusGraph port
    changes_threshold=100,         # Changes before auto-snapshot
    enable_versioning=True,        # Enable change tracking
    enable_snapshots=True          # Enable periodic snapshots
)
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | str | `"localhost"` | JanusGraph server hostname |
| `port` | int | `8182` | JanusGraph server port |
| `changes_threshold` | int | `100` | Number of changes before creating snapshot |
| `enable_versioning` | bool | `True` | Enable revision tracking |
| `enable_snapshots` | bool | `True` | Enable automatic snapshots |

### Vector Store Configuration

```python
from memory_core.embeddings.vector_store import VectorStoreMilvus

# Vector store configuration
vector_store = VectorStoreMilvus(
    host="localhost",              # Milvus server host
    port=19530,                   # Milvus server port
    collection_name="memory_engine_embeddings",  # Collection name
    dimension=768                # Embedding dimension
)
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | str | `"localhost"` | Milvus server hostname |
| `port` | int | `19530` | Milvus server port |
| `collection_name` | str | `"memory_engine_embeddings"` | Milvus collection name |
| `dimension` | int | `768` | Embedding vector dimension |

### Embedding Manager Configuration

```python
from memory_core.embeddings.embedding_manager import EmbeddingManager

# The embedding manager automatically configures:
# - API Key: from GEMINI_API_KEY environment variable
# - Model: text-embedding-004 (768 dimensions)
# - Task Types: SEMANTIC_SIMILARITY, RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY
```

#### Supported Models

| Model | Dimensions | Task Types | Description |
|-------|------------|------------|-------------|
| `text-embedding-004` | 768 | All | Current stable model |

### Knowledge Extraction Configuration

```python
from memory_core.ingestion.advanced_extractor import AdvancedExtractor

# Advanced extractor automatically configures:
# - Model: gemini-2.5-flash
# - API Key: from GEMINI_API_KEY environment variable
# - Temperature: 0.4 (for consistent extraction)
# - Output format: JSON
```

### MCP Endpoint Configuration

```python
from memory_core.mcp_integration.mcp_endpoint import MemoryEngineMCP

# MCP interface configuration
mcp = MemoryEngineMCP(
    host="localhost",              # JanusGraph host
    port=8182                     # JanusGraph port
)
```

## Runtime Configuration

### Logging Configuration

```python
import logging
import os

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format=os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('memory_engine.log')
    ]
)
```

### Custom Configuration Class

```python
import os
from typing import Optional

class MemoryEngineConfig:
    """Centralized configuration management."""
    
    def __init__(self):
        # Required configuration
        self.gemini_api_key = self._require_env('GEMINI_API_KEY')
        
        # Database configuration
        self.janusgraph_host = os.getenv('JANUSGRAPH_HOST', 'localhost')
        self.janusgraph_port = int(os.getenv('JANUSGRAPH_PORT', '8182'))
        self.milvus_host = os.getenv('MILVUS_HOST', 'localhost')
        self.milvus_port = int(os.getenv('MILVUS_PORT', '19530'))
        
        # Performance configuration
        self.changes_threshold = int(os.getenv('CHANGES_THRESHOLD', '100'))
        self.enable_versioning = os.getenv('ENABLE_VERSIONING', 'true').lower() == 'true'
        self.enable_snapshots = os.getenv('ENABLE_SNAPSHOTS', 'true').lower() == 'true'
        
        # Logging configuration
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        
    def _require_env(self, key: str) -> str:
        """Get required environment variable or raise error."""
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Required environment variable {key} is not set")
        return value
    
    def validate(self) -> bool:
        """Validate configuration."""
        try:
            self._require_env('GEMINI_API_KEY')
            return True
        except ValueError:
            return False

# Usage
config = MemoryEngineConfig()
if not config.validate():
    raise RuntimeError("Invalid configuration")
```

## Security Configuration

### API Key Management

#### Development
```bash
# Use a dedicated development API key
GEMINI_API_KEY=your-dev-api-key
```

#### Production
```bash
# Use environment variable injection or secrets management
# Never commit production keys to version control

# With Docker secrets:
docker run -e GEMINI_API_KEY_FILE=/run/secrets/gemini_key memory-engine

# With Kubernetes secrets:
kubectl create secret generic gemini-api-key --from-literal=key=your-api-key
```

### Network Security

```yaml
# docker-compose.yml security configuration
version: '3.8'
services:
  janusgraph:
    networks:
      - internal
    # Don't expose ports externally in production
    
  milvus:
    networks:
      - internal
    # Use internal network only

networks:
  internal:
    driver: bridge
    internal: true  # No external access
```

### Data Encryption

```python
# Example encryption configuration (not implemented in current version)
class SecureConfig:
    def __init__(self):
        self.encrypt_at_rest = os.getenv('ENCRYPT_AT_REST', 'false').lower() == 'true'
        self.encryption_key = os.getenv('ENCRYPTION_KEY')
        self.tls_enabled = os.getenv('TLS_ENABLED', 'false').lower() == 'true'
```

## Performance Tuning

### Memory Configuration

```bash
# JanusGraph memory settings
JAVA_OPTS="-Xms512m -Xmx2g"

# Python memory settings
PYTHONMALLOC=malloc
```

### Database Performance

```python
# JanusGraph connection tuning
janusgraph_config = {
    'connection_timeout': 30,
    'max_connections': 10,
    'retry_attempts': 3
}

# Milvus performance tuning
milvus_config = {
    'index_params': {
        'metric_type': 'L2',
        'index_type': 'IVF_FLAT',
        'params': {'nlist': 1024}
    },
    'search_params': {
        'nprobe': 10
    }
}
```

### Embedding Performance

```python
# Batch processing configuration
batch_config = {
    'embedding_batch_size': 100,
    'storage_batch_size': 50,
    'parallel_requests': 5
}
```

## Environment-Specific Settings

### Docker Compose Configuration

```yaml
# docker/docker-compose.yml
version: '3.8'
services:
  janusgraph:
    environment:
      - JAVA_OPTS=-Xms512m -Xmx2g
      - janusgraph.storage.backend=berkeleyje
    
  milvus:
    environment:
      - MILVUS_MODE=standalone
      - ETCD_ENDPOINTS=etcd:2379
    
  memory-engine:
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
```

### Kubernetes Configuration

```yaml
# k8s-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: memory-engine-config
data:
  LOG_LEVEL: "WARNING"
  JANUSGRAPH_HOST: "janusgraph-service"
  MILVUS_HOST: "milvus-service"
---
apiVersion: v1
kind: Secret
metadata:
  name: memory-engine-secrets
type: Opaque
stringData:
  GEMINI_API_KEY: "your-api-key"
```

### Systemd Configuration

```ini
# /etc/systemd/system/memory-engine.service
[Unit]
Description=Memory Engine Service
After=network.target

[Service]
Type=simple
User=memory-engine
WorkingDirectory=/opt/memory-engine
Environment="PYTHONPATH=/opt/memory-engine"
EnvironmentFile=/opt/memory-engine/.env
ExecStart=/opt/memory-engine/venv/bin/python -m memory_core.mcp_integration.mcp_endpoint
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

## Configuration Validation

### Startup Validation Script

```python
#!/usr/bin/env python3
"""Configuration validation script."""

import os
import sys
from typing import List, Tuple

def validate_config() -> List[Tuple[str, bool, str]]:
    """Validate all configuration settings."""
    checks = []
    
    # Required environment variables
    required_vars = ['GEMINI_API_KEY']
    for var in required_vars:
        value = os.getenv(var)
        checks.append((
            f"Environment variable {var}",
            bool(value and value.strip()),
            f"Required variable {var} is {'set' if value else 'missing'}"
        ))
    
    # Database connectivity
    try:
        from memory_core.core.knowledge_engine import KnowledgeEngine
        engine = KnowledgeEngine()
        connected = engine.connect()
        engine.disconnect()
        checks.append((
            "JanusGraph connectivity",
            connected,
            f"JanusGraph connection {'successful' if connected else 'failed'}"
        ))
    except Exception as e:
        checks.append(("JanusGraph connectivity", False, f"Error: {e}"))
    
    # Vector store connectivity
    try:
        from memory_core.embeddings.vector_store import VectorStoreMilvus
        vs = VectorStoreMilvus()
        connected = vs.connect()
        if connected:
            vs.disconnect()
        checks.append((
            "Milvus connectivity",
            connected,
            f"Milvus connection {'successful' if connected else 'failed'}"
        ))
    except Exception as e:
        checks.append(("Milvus connectivity", False, f"Error: {e}"))
    
    return checks

if __name__ == "__main__":
    print("Memory Engine Configuration Validation")
    print("=" * 50)
    
    checks = validate_config()
    all_passed = True
    
    for name, passed, message in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {name}: {message}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("✅ All configuration checks passed!")
        sys.exit(0)
    else:
        print("❌ Some configuration checks failed!")
        sys.exit(1)
```

## Troubleshooting Configuration Issues

### Common Problems

1. **Missing API Key**
   ```bash
   Error: Required environment variable GEMINI_API_KEY is not set
   Solution: Export GEMINI_API_KEY=your-api-key
   ```

2. **Database Connection Failed**
   ```bash
   Error: Could not connect to JanusGraph
   Solution: Check JANUSGRAPH_HOST and JANUSGRAPH_PORT, ensure service is running
   ```

3. **Permission Denied**
   ```bash
   Error: Permission denied accessing configuration file
   Solution: Check file permissions, ensure proper user ownership
   ```

### Configuration Debugging

```python
# Debug configuration script
import os

def debug_config():
    """Print all configuration for debugging."""
    print("Memory Engine Configuration Debug")
    print("=" * 40)
    
    # Environment variables
    env_vars = [
        'GEMINI_API_KEY', 'JANUSGRAPH_HOST', 'JANUSGRAPH_PORT',
        'MILVUS_HOST', 'MILVUS_PORT', 'LOG_LEVEL'
    ]
    
    for var in env_vars:
        value = os.getenv(var, 'NOT SET')
        # Mask sensitive values
        if 'key' in var.lower() or 'secret' in var.lower():
            display_value = f"{value[:8]}..." if value != 'NOT SET' else value
        else:
            display_value = value
        print(f"{var}: {display_value}")

if __name__ == "__main__":
    debug_config()
```

Save this script as `debug_config.py` and run it to troubleshoot configuration issues.