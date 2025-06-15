# Memory Engine Examples

This directory contains practical examples demonstrating how to use the Memory Engine system for various knowledge management tasks.

## Examples Overview

### 🔧 [basic_usage.py](basic_usage.py)
**Core operations and fundamental workflows**

Demonstrates:
- Setting up the Knowledge Engine
- Creating and storing knowledge nodes
- Building relationships between nodes  
- Embedding generation and similarity search
- Rating updates based on evidence
- Versioning and change tracking

**Run**: `python examples/basic_usage.py`

### 🧠 [knowledge_extraction.py](knowledge_extraction.py)
**Text processing and knowledge extraction**

Demonstrates:
- Extracting knowledge from various text types
- Processing scientific and technical content
- Automatic knowledge unit generation
- Batch processing multiple documents
- Knowledge merging and deduplication
- Cross-document relationship creation

**Run**: `python examples/knowledge_extraction.py`

### 📡 [mcp_client_example.py](mcp_client_example.py)
**Module Communication Protocol integration**

Demonstrates:
- Using the MCP interface for external integration
- Ingesting text via MCP commands
- Searching knowledge through MCP
- Retrieving detailed node information
- Updating ratings via MCP
- Error handling and batch operations

**Run**: `python examples/mcp_client_example.py`

### 🗄️ [storage_backends_example.py](storage_backends_example.py)
**Modular storage backend usage (Added in v0.2.0)**

Demonstrates:
- Using different storage backends (JanusGraph, SQLite, JSON file)
- Backend availability checking and selection
- Performance comparison across backends
- Data migration between storage systems
- Configuration for different deployment scenarios
- Backend-specific features and optimizations

**Run**: `python examples/storage_backends_example.py`

### 🤖 [anthropic_llm_example.py](anthropic_llm_example.py)
**Anthropic Claude LLM provider usage (New in v0.3.0)**

Demonstrates:
- Using Claude models for knowledge extraction
- Streaming completions and chat functionality
- Structured output with JSON mode
- Error handling and provider health checks

**Run**: `python examples/anthropic_llm_example.py`

### 🌐 [openai_llm_example.py](openai_llm_example.py)
**OpenAI GPT provider usage (New in v0.3.0)**

Demonstrates:
- Using GPT models for knowledge tasks
- JSON mode for structured responses
- Rate limit handling and retries
- Provider configuration and management

**Run**: `python examples/openai_llm_example.py`

### 🏠 [ollama_llm_example.py](ollama_llm_example.py)
**Local Ollama LLM provider usage (New in v0.3.0)**

Demonstrates:
- Running knowledge extraction completely offline
- Local model management and configuration
- Performance optimization for local inference
- Fallback strategies and error handling

**Run**: `python examples/ollama_llm_example.py`

## Prerequisites

Before running the examples:

1. **Set up environment** (at least one API key required):
   ```bash
   export GOOGLE_API_KEY="your-gemini-api-key"        # For Gemini
   export OPENAI_API_KEY="your-openai-api-key"        # For OpenAI GPT  
   export ANTHROPIC_API_KEY="your-anthropic-api-key"  # For Claude
   # Or use local models (Ollama/HuggingFace) - no API key needed
   ```

2. **Start infrastructure** (for JanusGraph/Milvus backends):
   ```bash
   cd docker
   docker-compose up -d
   # Wait 2-3 minutes for services to initialize
   ```
   
   **Note**: For JSON file or SQLite backends, no additional infrastructure is needed.

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   
   # Optional: For SQLite backend support
   pip install aiosqlite
   ```

## Running Examples

### Individual Examples
```bash
# Basic usage
python examples/basic_usage.py

# Knowledge extraction
python examples/knowledge_extraction.py

# MCP interface
python examples/mcp_client_example.py
```

### All Examples
```bash
# Run all examples in sequence
for script in examples/*.py; do
    echo "Running $script..."
    python "$script"
    echo "Completed $script"
    echo "---"
done
```

## Example Output

### Basic Usage Output
```
🌟 Memory Engine - Basic Usage Examples
============================================================
🔍 Checking prerequisites...
✅ GEMINI_API_KEY is set

🚀 Setting up Memory Engine components...
✅ Connected to JanusGraph
✅ Connected to Milvus and initialized embedding manager

============================================================
📝 Example 1: Basic Node Operations
============================================================

📌 Creating node 1...
   ✅ Created node with ID: abc123...
   📄 Content: Python is a high-level programming language...

🔍 Retrieving created nodes...
   Node 1 (ID: abc123):
   📄 Content: Python is a high-level programming language...
   📊 Truthfulness: 0.95
   🎯 Richness: 0.80
```

### Knowledge Extraction Output
```
🌟 Memory Engine - Knowledge Extraction Examples
======================================================================
🚀 Setting up Memory Engine for knowledge extraction...
✅ Connected to JanusGraph
✅ Connected to Milvus

======================================================================
📝 Example 1: Simple Text Knowledge Extraction
======================================================================
🧠 Extracting knowledge units...
✅ Extracted 4 knowledge units:

   Unit 1:
   📄 Content: Artificial Intelligence is a branch of computer science
   🏷️  Tags: ai, computer_science, technology
   📊 Confidence: 0.9, Domain: computer_science
```

## Customization

### Modifying Examples

Each example is self-contained and can be modified:

```python
# Customize knowledge extraction
def custom_extraction_example():
    # Your custom text
    custom_text = "Your domain-specific content here..."
    
    # Extract knowledge
    units = extract_knowledge_units(custom_text)
    
    # Process with custom source label
    node_ids = process_extracted_units(
        units=units,
        source_label="My Custom Source",
        storage=engine.storage,
        embedding_manager=embedding_manager
    )
    
    return node_ids
```

### Configuration Options

Examples use these configurable parameters:

```python
# Knowledge Engine configuration
engine = KnowledgeEngine(
    host="localhost",           # JanusGraph host
    port=8182,                 # JanusGraph port
    enable_versioning=True,     # Track changes
    enable_snapshots=True,      # Periodic snapshots
    changes_threshold=100       # Changes before snapshot
)

# Vector store configuration  
vector_store = VectorStoreMilvus(
    host="localhost",
    port=19530,
    collection_name="memory_engine_embeddings",
    dimension=3072
)

# Search parameters
search_params = {
    "top_k": 5,                # Number of results
    "similarity_threshold": 0.8 # Minimum similarity
}
```

## Error Handling

Examples include comprehensive error handling:

```python
try:
    # Memory Engine operations
    result = engine.save_node(node)
    
except ConnectionError as e:
    print(f"❌ Database connection failed: {e}")
    
except ValueError as e:
    print(f"❌ Invalid data: {e}")
    
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    
finally:
    # Always cleanup
    engine.disconnect()
```

## Performance Notes

- **Knowledge Extraction**: Processes ~500-1000 words in 2-5 seconds
- **Vector Search**: Returns results in 50-200ms for typical datasets
- **Batch Processing**: Optimal batch size is 10-50 items
- **API Rate Limits**: Examples include appropriate delays for Gemini API

## Next Steps

After running the examples:

1. **Explore the [Documentation](../docs/)** for detailed guides
2. **Check [API Reference](../docs/api_reference.md)** for complete API details  
3. **Read [Architecture Guide](../docs/architecture.md)** to understand system design
4. **Try [Troubleshooting Guide](../docs/troubleshooting.md)** if you encounter issues

## Support

If you encounter issues with the examples:

1. Check the [Troubleshooting Guide](../docs/troubleshooting.md)
2. Verify prerequisites are met
3. Ensure infrastructure services are running
4. Check logs for detailed error information

For additional help, create an issue in the GitHub repository with:
- Example script that failed
- Complete error output
- Environment information (OS, Python version, etc.)