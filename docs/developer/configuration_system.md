# Configuration Management System

The Memory Engine uses a robust, centralized configuration management system that supports:
- Environment-specific configurations
- Configuration validation on startup
- Dynamic configuration updates
- Support for multiple deployment scenarios
- Comprehensive configuration documentation

## Overview

The configuration system is built around the `ConfigManager` class which implements a singleton pattern and provides type-safe access to all configuration values.

### Key Features

1. **Centralized Configuration**: All configuration is managed through a single `ConfigManager` instance
2. **Environment-Specific Overrides**: Support for development, testing, staging, and production configurations
3. **Configuration Validation**: Automatic validation of configuration values on startup
4. **Dynamic Updates**: File watching and configuration reload without restart (non-production environments)
5. **Multiple Sources**: Configuration can be loaded from YAML, JSON files, and environment variables
6. **Type Safety**: Strong typing with data classes and enums

## Configuration Sources (Priority Order)

Configuration values are loaded in the following priority order (highest priority first):

1. **Environment Variables** (highest priority)
2. **Environment-specific configuration files** (`config.{environment}.yaml`)
3. **Base configuration files** (`config.yaml`, `config.json`)
4. **Default values** (lowest priority)

## Quick Start

### Basic Usage

```python
from memory_core.config import get_config

# Get the global configuration instance
config = get_config()

# Access configuration values
api_key = config.config.api.gemini_api_key
db_url = config.config.database.url
vector_store_type = config.config.vector_store.type
```

### Initialize with Custom Directory

```python
from memory_core.config import init_config

# Initialize with custom configuration directory
config = init_config("/path/to/config/directory")
```

## Configuration Structure

### Main Configuration Sections

```python
config.config.environment          # Application environment
config.config.debug                # Debug mode flag
config.config.database             # Database configuration
config.config.janusgraph           # JanusGraph configuration
config.config.vector_store         # Vector store configuration
config.config.embedding            # Embedding model configuration
config.config.llm                  # LLM configuration
config.config.api                  # API keys and authentication
config.config.logging              # Logging configuration
config.config.versioning           # Versioning and revision management
config.config.security             # Security settings
config.config.performance          # Performance settings
```

## Environment Variables

### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key (required) | `your-gemini-api-key` |

### Optional Environment Variables

#### Application Settings
| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `ENVIRONMENT` | Application environment | `development` | `production` |
| `DEBUG` | Enable debug mode | `false` | `true` |

#### Database Configuration
| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | Database connection URL | `sqlite:///memory_engine.db` |
| `DATABASE_POOL_SIZE` | Connection pool size | `10` |

#### JanusGraph Configuration
| Variable | Description | Default |
|----------|-------------|---------|
| `JANUSGRAPH_HOST` | JanusGraph server host | `localhost` |
| `JANUSGRAPH_PORT` | JanusGraph server port | `8182` |
| `JANUSGRAPH_USE_SSL` | Use SSL for JanusGraph connection | `false` |
| `JANUSGRAPH_USERNAME` | JanusGraph username | `null` |
| `JANUSGRAPH_PASSWORD` | JanusGraph password | `null` |

#### Milvus Configuration
| Variable | Description | Default |
|----------|-------------|---------|
| `MILVUS_HOST` | Milvus server host | `localhost` |
| `MILVUS_PORT` | Milvus server port | `19530` |
| `MILVUS_USER` | Milvus username | `null` |
| `MILVUS_PASSWORD` | Milvus password | `null` |
| `COLLECTION_NAME` | Milvus collection name | `knowledge_vectors` |

#### Vector Store Configuration
| Variable | Description | Default |
|----------|-------------|---------|
| `VECTOR_STORE_TYPE` | Vector store type | `milvus` |
| `EMBEDDING_DIMENSION` | Embedding vector dimension | `768` |

#### Model Configuration
| Variable | Description | Default |
|----------|-------------|---------|
| `EMBEDDING_MODEL` | Embedding model name | `text-embedding-004` |
| `LLM_MODEL` | LLM model name | `gemini-2.0-flash-thinking-exp` |
| `FALLBACK_MODEL` | Fallback LLM model | `gemini-2.0-flash-exp` |

#### Logging Configuration
| Variable | Description | Default |
|----------|-------------|---------|
| `LOG_LEVEL` | Logging level | `INFO` |
| `LOG_FORMAT` | Log message format | Standard format |

#### Security Configuration
| Variable | Description | Default |
|----------|-------------|---------|
| `ENCRYPT_AT_REST` | Enable encryption at rest | `false` |
| `ENCRYPTION_KEY` | Encryption key | `null` |
| `TLS_ENABLED` | Enable TLS connections | `false` |

#### Performance Configuration
| Variable | Description | Default |
|----------|-------------|---------|
| `WORKERS` | Number of worker processes | `4` |

## Configuration Files

### Base Configuration (`config.yaml`)

The base configuration file contains default values for all environments:

```yaml
# Application Settings
environment: development
debug: false

# Database Configuration
database:
  url: "sqlite:///memory_engine.db"
  pool_size: 10
  max_overflow: 20

# JanusGraph Configuration
janusgraph:
  host: "localhost"
  port: 8182
  use_ssl: false

# Vector Store Configuration
vector_store:
  type: "milvus"
  dimension: 768
  milvus:
    host: "localhost"
    port: 19530
    collection_name: "knowledge_vectors"

# More configuration sections...
```

### Environment-Specific Configurations

Environment-specific files override base configuration values:

#### Development (`config.development.yaml`)
```yaml
environment: development
debug: true

database:
  url: "sqlite:///memory_engine_dev.db"

logging:
  level: "DEBUG"
  file_path: "logs/development.log"
```

#### Testing (`config.testing.yaml`)
```yaml
environment: testing
testing: true

database:
  url: "sqlite:///:memory:"

logging:
  level: "WARNING"
  enable_console: false

versioning:
  enable_versioning: false
```

#### Production (`config.production.yaml`)
```yaml
environment: production
debug: false

database:
  url: "postgresql://user:password@prod-db:5432/memory_engine"

security:
  encrypt_at_rest: true
  tls_enabled: true

performance:
  workers: 8
  max_memory_usage: 4294967296  # 4GB
```

## Configuration Management

### Accessing Configuration Values

```python
from memory_core.config import get_config

config = get_config()

# Access nested configuration using dot notation
host = config.get('janusgraph.host')
port = config.get('janusgraph.port', 8182)  # with default value

# Access using object attributes
api_key = config.config.api.gemini_api_key
dimension = config.config.vector_store.dimension
```

### Setting Configuration Values

```python
# Set configuration value (triggers validation)
config.set('janusgraph.host', 'new-host')

# Direct attribute access
config.config.janusgraph.port = 9999
```

### Converting to Dictionary

```python
# Get configuration as dictionary
config_dict = config.to_dict()
```

### Saving Configuration

```python
# Save current configuration to YAML file
config.save_to_file('current_config.yaml', 'yaml')

# Save to JSON file
config.save_to_file('current_config.json', 'json')
```

## Configuration Validation

The configuration system automatically validates settings on startup:

### Validation Rules

1. **Required API Keys**: `GEMINI_API_KEY` must be provided
2. **Database URL**: Must be a valid connection string
3. **Dimension Consistency**: Vector store dimension must match embedding dimension
4. **Port Ranges**: Ports must be between 1 and 65535
5. **Security Dependencies**: Encryption key required when encryption is enabled

### Handling Validation Errors

```python
from memory_core.config import ConfigManager, ConfigValidationError

try:
    config = ConfigManager()
except ConfigValidationError as e:
    print(f"Configuration validation failed: {e}")
    # Handle the error appropriately
```

## Dynamic Configuration Updates

In non-production environments, the configuration system can watch for file changes and automatically reload:

```python
# File watching is enabled by default in development/testing
config = ConfigManager()

# Manually reload configuration
config.reload_configuration()

# Stop file watching
config.stop_file_watching()
```

## Best Practices

### 1. Environment Variables for Secrets
Store sensitive information like API keys in environment variables, not in configuration files:

```bash
export GEMINI_API_KEY="your-secret-api-key"
export DATABASE_PASSWORD="your-database-password"
```

### 2. Environment-Specific Files
Use environment-specific configuration files for environment differences:

- `config/environments/config.development.yaml` - Local development settings
- `config/environments/config.testing.yaml` - Test environment settings
- `config/environments/config.staging.yaml` - Staging environment settings
- `config/environments/config.production.yaml` - Production environment settings

### 3. Configuration Directory Structure
```
project_root/
├── config/
│   ├── config.yaml                    # Base configuration
│   └── environments/                  # Environment-specific configurations
│       ├── config.development.yaml   # Development overrides
│       ├── config.testing.yaml       # Testing overrides
│       ├── config.staging.yaml       # Staging overrides
│       └── config.production.yaml    # Production overrides
└── .env                              # Environment variables (development only)
```

### 4. Validation First
Always ensure your configuration is valid before deploying:

```python
from memory_core.config import ConfigManager

try:
    config = ConfigManager()
    print("Configuration is valid!")
except Exception as e:
    print(f"Configuration error: {e}")
    exit(1)
```

## Migration from Environment Variables

If you're migrating from the old environment variable system:

### Before (Old System)
```python
import os
api_key = os.getenv('GEMINI_API_KEY')
host = os.getenv('JANUSGRAPH_HOST', 'localhost')
```

### After (New System)
```python
from memory_core.config import get_config

config = get_config()
api_key = config.config.api.gemini_api_key
host = config.config.janusgraph.host
```

### Environment Variable Compatibility

The new system is fully backward compatible. All existing environment variables will continue to work and will override configuration file values.

## Troubleshooting

### Common Issues

1. **Missing API Key Error**
   ```
   ConfigValidationError: GEMINI_API_KEY is required
   ```
   **Solution**: Set the `GEMINI_API_KEY` environment variable or in configuration file

2. **Dimension Mismatch Error**
   ```
   ConfigValidationError: Vector store dimension must match embedding dimension
   ```
   **Solution**: Ensure `vector_store.dimension` matches `embedding.dimension`

3. **File Not Found**
   **Solution**: Ensure configuration files are in the correct directory or use absolute paths

4. **Invalid YAML/JSON**
   **Solution**: Validate your configuration files using a YAML/JSON validator

### Debug Configuration Loading

Enable debug logging to see configuration loading process:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from memory_core.config import ConfigManager
config = ConfigManager()
```

## API Reference

### ConfigManager Class

```python
class ConfigManager:
    def __init__(self, config_dir: Optional[Union[str, Path]] = None)
    def get(self, path: str, default: Any = None) -> Any
    def set(self, path: str, value: Any) -> None
    def to_dict(self) -> Dict[str, Any]
    def save_to_file(self, filename: str, format: str = 'yaml') -> None
    def reload_configuration(self) -> None
    def stop_file_watching(self) -> None
```

### Global Functions

```python
def get_config() -> ConfigManager
def init_config(config_dir: Optional[Union[str, Path]] = None) -> ConfigManager
```

## Examples

### Example 1: Basic Setup
```python
from memory_core.config import get_config

# Get configuration
config = get_config()

# Use configuration in your application
if config.config.debug:
    print("Debug mode enabled")

# Access database URL
db_url = config.config.database.url
```

### Example 2: Custom Configuration Directory
```python
from memory_core.config import init_config

# Initialize with custom config directory
config = init_config("/app/config")

# Access configuration
embedding_model = config.config.embedding.model
```

### Example 3: Environment-Specific Behavior
```python
from memory_core.config import get_config, Environment

config = get_config()

if config.config.environment == Environment.PRODUCTION:
    # Production-specific logic
    setup_monitoring()
elif config.config.environment == Environment.DEVELOPMENT:
    # Development-specific logic
    enable_debug_tools()
```

This configuration system provides a solid foundation for managing all aspects of the Memory Engine's configuration in a type-safe, validated, and environment-aware manner.