# API Documentation

API reference documentation for the Memory Engine.

## Available Documentation

### [API Reference](api_reference.md)
Complete API reference with classes, methods, and examples.

## Quick API Overview

### Core Components

#### Configuration Management
```python
from memory_core.config import get_config

config = get_config()
api_key = config.config.api.gemini_api_key
```

#### Knowledge Engine
```python
from memory_core.core.knowledge_engine import KnowledgeEngine

engine = KnowledgeEngine()
engine.connect()
node_id = engine.add_knowledge("Sample knowledge")
```

#### Embedding Management
```python
from memory_core.embeddings.embedding_manager import EmbeddingManager
from memory_core.embeddings.vector_store import VectorStoreMilvus

vector_store = VectorStoreMilvus()
embedding_manager = EmbeddingManager(vector_store)
```

### Key Classes
- `KnowledgeEngine` - Main engine for knowledge management
- `ConfigManager` - Configuration management
- `EmbeddingManager` - Embedding generation and storage
- `VectorStoreMilvus` - Vector storage with Milvus
- `JanusGraphStorage` - Graph storage with JanusGraph
- `AdvancedExtractor` - Knowledge extraction from text

### Example Usage
See the `/examples` directory for complete usage examples:
- `basic_usage.py` - Basic knowledge engine usage
- `config_example.py` - Configuration system examples
- `knowledge_extraction.py` - Knowledge extraction examples

For complete API documentation, see [api_reference.md](api_reference.md).