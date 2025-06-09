# Memory Engine - Claude Code Guide

## Project Overview
Knowledge management system: JanusGraph + Milvus + Gemini API + MCP Protocol

## Environment Setup
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables (.env)
```
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
LLM_MODEL=gemini-2.0-flash-thinking-exp
FALLBACK_MODEL=gemini-2.0-flash-exp
EMBEDDING_MODEL=text-embedding-004
EMBEDDING_DIMENSION=768
```

### Docker Services
```bash
sudo docker-compose -f docker/docker-compose.yml up -d
sudo docker ps  # Check status
sudo docker restart janusgraph  # If connection issues
```

## Testing
```bash
# All tests
source .venv/bin/activate && python -m pytest tests/ -v

# Unit tests only
python -m pytest tests/ -v -k "not Integration"

# Integration tests (requires API keys + Docker)
python -m pytest tests/test_embedding_manager.py::TestEmbeddingManagerIntegration -v
```

## Development Workflow
1. **Explore**: Use Read/Grep/Glob tools to understand code
2. **Plan**: Use TodoWrite for complex tasks
3. **Code**: Follow existing patterns, handle async/sync properly
4. **Test**: Run tests with Docker services
5. **Commit**: Frequent commits with descriptive messages

## Code Style & Rules
- Follow PEP 8, use type hints, add docstrings
- **JanusGraph**: Use SYNC methods only (no async)
- **Testing**: Unit tests + integration tests
- **Commits**: Always commit changes frequently
- **API Integration**: Handle rate limits, retry logic
- **Error Handling**: Comprehensive exception handling

## Key Architecture
- `memory_core/core/knowledge_engine.py`: Main engine
- `memory_core/db/janusgraph_storage.py`: Graph database (SYNC only)
- `memory_core/embeddings/embedding_manager.py`: Vector operations
- `memory_core/mcp_integration/`: MCP protocol implementations
- `memory_core/monitoring/`: Performance & observability

## Common Issues & Fixes
### JanusGraph
```bash
sudo docker restart janusgraph && sleep 15
```
- Use sync methods only (avoid async/await)
- Wait for healthy status

### Milvus
```bash
sudo docker restart milvus && sleep 10
```
- Check embedding dimensions (768)
- Ensure collection exists

### API Keys
- Verify GOOGLE_API_KEY and GEMINI_API_KEY
- Test with simple embedding generation

## Emergency Commands
```bash
# Full reset
sudo docker-compose -f docker/docker-compose.yml restart

# Quick test
python examples/basic_usage.py

# Component test
python -c "from memory_core.embeddings.embedding_manager import EmbeddingManager; print('OK')"
```

## Claude Code Rules
- Always activate venv first: `source .venv/bin/activate`
- Check Docker status if connectivity fails
- Use TodoWrite for complex multi-step tasks
- **Commit frequently** with descriptive messages
- Test both unit and integration scenarios
- Handle async/sync properly (JanusGraph=sync, Milvus=async-safe)
- Follow existing code patterns and style
- Add comprehensive error handling and logging