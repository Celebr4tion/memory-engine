# Memory Engine - Claude Code Guide

This file contains essential information for Claude Code to effectively work with the Memory Engine project.

## Project Overview

Memory Engine is a knowledge management system that combines:
- **Graph Database**: JanusGraph for structured knowledge storage
- **Vector Database**: Milvus for semantic search and embeddings
- **AI Integration**: Gemini API for knowledge extraction and reasoning
- **MCP Protocol**: Model Context Protocol for external integrations

## Environment Setup

### Prerequisites
```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies (if not already done)
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file with:
```
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here  # For backwards compatibility
LLM_MODEL=gemini-2.0-flash-thinking-exp
FALLBACK_MODEL=gemini-2.0-flash-exp
EMBEDDING_MODEL=text-embedding-004
EMBEDDING_DIMENSION=768
```

### Docker Services
Start required services:
```bash
# Start all services
sudo docker-compose -f docker/docker-compose.yml up -d

# Check service status
sudo docker ps

# Restart JanusGraph if connection issues occur
sudo docker restart janusgraph
```

## Testing

### Run All Tests
```bash
source .venv/bin/activate && python -m pytest tests/ -v
```

### Run Specific Test Categories
```bash
# Integration tests (requires API keys and Docker services)
python -m pytest tests/test_embedding_manager.py::TestEmbeddingManagerIntegration -v
python -m pytest tests/test_advanced_extractor.py::TestAdvancedExtractorIntegration -v

# Unit tests only
python -m pytest tests/ -v -k "not Integration"

# Test specific functionality
python -m pytest tests/test_janusgraph_storage.py -v
python -m pytest tests/test_milvus_connection.py -v
```

### Connectivity Tests
```bash
# Test JanusGraph connectivity
python test_janusgraph_connection.py

# Test basic usage example
python examples/basic_usage.py
```

## Development Workflow

### 1. Explore Phase
- Use `Read` tool to understand existing code
- Use `Grep` and `Glob` to find relevant files
- Check existing tests for expected behavior

### 2. Plan Phase
- Create todos using TodoWrite for complex tasks
- Identify files that need modification
- Plan test strategy

### 3. Code Phase
- Follow existing code patterns and style
- Use type hints where appropriate
- Handle async/sync properly (JanusGraph uses sync, avoid async conflicts)

### 4. Test Phase
- Run relevant tests after changes
- Test with Docker services running
- Verify integration functionality

### 5. Commit Phase
- Use conventional commit messages
- Include co-authoring with Claude Code
- Stage only relevant files

## Code Style Guidelines

### Python
- Follow PEP 8 conventions
- Use descriptive variable names
- Add docstrings for public methods
- Use type hints for function signatures
- Handle exceptions appropriately

### Testing
- Write unit tests for new functionality
- Use mocking for external dependencies
- Test both success and error cases
- Add integration tests for API endpoints

## Architecture Notes

### Key Components
- `memory_core/core/knowledge_engine.py`: Main engine interface
- `memory_core/db/janusgraph_storage.py`: Graph database operations
- `memory_core/embeddings/embedding_manager.py`: Vector operations
- `memory_core/mcp_integration/`: MCP protocol implementations

### Database Connections
- **JanusGraph**: Use synchronous methods only (async causes event loop conflicts)
- **Milvus**: Async-safe, use connection pooling
- **Connection Management**: Always check connection status before operations

### API Integration
- **Gemini API**: Used for embeddings and knowledge extraction
- **Rate Limiting**: Handle API limits gracefully
- **Error Handling**: Retry logic for transient failures

## Common Issues

### JanusGraph Connection Problems
- Restart Docker container: `sudo docker restart janusgraph`
- Wait for health check: Check `sudo docker ps` for healthy status
- Avoid async/await in JanusGraph operations

### Milvus Connection Issues
- Verify container is running on port 19530
- Check embedding dimensions match (768 for text-embedding-004)
- Ensure collection exists before operations

### API Key Issues
- Verify both GOOGLE_API_KEY and GEMINI_API_KEY are set
- Test with simple embedding generation first
- Check API quotas and limits

## Project Structure

```
memory_core/
├── agents/          # AI agents and reasoning
├── core/           # Main engine and orchestration
├── db/             # Database adapters and storage
├── embeddings/     # Vector operations and embeddings
├── ingestion/      # Knowledge extraction and processing
├── mcp_integration/# MCP protocol implementations
├── model/          # Data models and schemas
├── rating/         # Quality assessment systems
└── versioning/     # Version control for knowledge
```

## Useful Commands

### Development
```bash
# Run basic usage example
python examples/basic_usage.py

# Run MCP server
python -m memory_core.mcp_integration.mcp_endpoint

# Check Docker logs
sudo docker logs janusgraph --tail 20
sudo docker logs milvus --tail 20
```

### Debugging
```bash
# Test individual components
python -c "from memory_core.embeddings.embedding_manager import EmbeddingManager; print('Embedding manager imported successfully')"

# Test connections directly
python test_janusgraph_minimal.py
python -m pytest tests/test_milvus_connection.py -v
```

## Unexpected Behaviors

### Known Issues
1. **JanusGraph Async Conflicts**: Always use synchronous methods for JanusGraph operations
2. **Container Stability**: JanusGraph may need restarts during development
3. **API Rate Limits**: Gemini API has usage quotas that may affect integration tests
4. **Event Loop Conflicts**: Avoid mixing async/sync patterns in the same execution context

### Workarounds
- Use `_sync_` prefixed methods for JanusGraph operations
- Restart containers if connection errors persist
- Use mocking for tests that hit API limits frequently
- Run integration tests with sufficient wait times

## Recent Changes

- Fixed async event loop conflicts in JanusGraph storage
- Updated to Gemini 2.0 models for better performance  
- Simplified connection handling to prevent blocking issues
- Added comprehensive test coverage for integration scenarios

## Notes for Claude Code

- Always activate virtual environment before running Python commands
- Check Docker service status if connectivity tests fail
- Use TodoWrite for complex multi-step tasks
- Commit changes frequently with descriptive messages
- Test both unit and integration scenarios when making changes
- Ask for API keys or permissions if needed for testing