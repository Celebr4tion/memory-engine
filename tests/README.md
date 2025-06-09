# Tests Directory

This directory contains all tests for the Memory Engine project.

## Structure

```
tests/
├── README.md                    # This file
├── conftest.py                  # Shared test configuration and fixtures
├── integration/                 # Integration tests (require external services)
│   ├── test_janusgraph_connection.py
│   ├── test_janusgraph_minimal.py
│   ├── test_janusgraph_storage_simple.py
│   └── test_milvus_connection.py
├── unit/                        # Unit tests (isolated, fast)
│   ├── test_config_manager.py
│   ├── test_knowledge_node.py
│   ├── test_relationship.py
│   ├── test_rating_system.py
│   ├── test_embedding_manager.py
│   ├── test_vector_store.py
│   └── test_advanced_extractor.py
└── [component tests]            # Component-level tests
    ├── test_knowledge_engine.py
    ├── test_janusgraph_storage.py
    ├── test_mcp_endpoint.py
    ├── test_enhanced_mcp_endpoint.py
    ├── test_merging.py
    ├── test_relationship_extractor.py
    ├── test_revision_manager.py
    ├── test_versioned_graph_adapter.py
    ├── test_graph_storage_adapter.py
    └── test_knowledge_agent.py
```

## Running Tests

### All Tests
```bash
pytest
```

### Unit Tests Only (Fast)
```bash
pytest tests/unit/
```

### Integration Tests Only (Requires External Services)
```bash
pytest tests/integration/
```

### Component Tests
```bash
pytest tests/test_*.py
```

### Specific Test File
```bash
pytest tests/unit/test_config_manager.py
```

## Test Categories

- **Unit Tests**: Test individual functions/classes in isolation
- **Integration Tests**: Test interaction with external services (JanusGraph, Milvus)
- **Component Tests**: Test larger components and their interactions within the system

## Test Requirements

Integration tests require:
- JanusGraph server running on localhost:8182
- Milvus server running on localhost:19530
- GEMINI_API_KEY environment variable set

Unit tests have no external dependencies.