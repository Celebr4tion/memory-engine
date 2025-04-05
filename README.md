# Memory Engine

A robust semantic knowledge system that stores, retrieves, and synthesizes information using a combination of graph and vector databases.

## Purpose

Memory Engine is designed to serve as a comprehensive knowledge management system that combines traditional graph-based knowledge representation with modern vector embeddings. The system provides efficient storage, retrieval, and contextual understanding of information, enabling applications to build upon a shared semantic foundation.

## Architecture

The Memory Engine is built on several key components:

1. **Knowledge Graph** - Using JanusGraph for storing structured relationships between entities
2. **Vector Store** - A separate database for managing and searching vector embeddings
3. **LLM-based Ingestion** - Intelligent processing of information from various sources
4. **MCP Interface** - Module Communication Protocol for external modules to interact with the system

## Directory Structure

- `memory_core/` - Core implementation of the memory engine
  - `db/` - Database connection and management
  - `embeddings/` - Vector embeddings storage and retrieval
  - `ingestion/` - Data ingestion pipelines
  - `mcp_integration/` - Module Communication Protocol
  - `model/` - Data models and schemas
  - `rating/` - Relevance scoring and ranking
  - `versioning/` - Data versioning support

- `tests/` - Unit and integration tests
- `docs/` - Documentation
- `docker/` - Container configuration

## License

This project is licensed under the [Hippocratic License](LICENSE.md), which is an ethical source license that enforces ethical use of the software.

## Getting Started

[Documentation to be added]