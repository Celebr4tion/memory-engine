# Ollama Embedding Provider

The Ollama Embedding Provider enables fully local embedding generation using the Ollama server. This provider is ideal for privacy-sensitive applications or when you want to avoid external API calls.

## Prerequisites

1. **Install Ollama**: Download and install Ollama from [https://ollama.ai](https://ollama.ai)
2. **Start Ollama Server**: Run `ollama serve` to start the server on `http://localhost:11434`
3. **Pull Embedding Model**: Download an embedding model, e.g., `ollama pull nomic-embed-text`

## Supported Models

The provider supports any Ollama embedding model. Popular choices include:

- `nomic-embed-text` - 768 dimensions, good general-purpose embedding model
- `mxbai-embed-large` - 1024 dimensions, high-quality embeddings
- `all-minilm` - 384 dimensions, lightweight and fast
- `snowflake-arctic-embed` - 1024 dimensions, optimized for retrieval

## Configuration

```python
config = {
    'model_name': 'nomic-embed-text',         # Ollama model name
    'base_url': 'http://localhost:11434',     # Ollama server URL
    'max_batch_size': 32,                     # Maximum batch size
    'timeout': 60,                            # Request timeout (seconds)
    'keep_alive': '5m',                       # Model keep-alive duration
    'max_retries': 3,                         # Connection retry attempts
    'retry_delay': 1                          # Delay between retries (seconds)
}
```

## Basic Usage

```python
import asyncio
from memory_core.embeddings.providers.ollama import OllamaEmbeddingProvider
from memory_core.embeddings.interfaces import TaskType

async def main():
    # Initialize provider
    config = {
        'model_name': 'nomic-embed-text',
        'base_url': 'http://localhost:11434'
    }
    provider = OllamaEmbeddingProvider(config)
    
    # Check availability
    if not provider.is_available():
        print("Ollama server is not available")
        return
    
    # Test connection
    if not await provider.test_connection():
        print("Connection test failed")
        return
    
    # Generate single embedding
    embedding = await provider.generate_embedding(
        "The quick brown fox jumps over the lazy dog.",
        TaskType.SEMANTIC_SIMILARITY
    )
    print(f"Embedding shape: {embedding.shape}")
    
    # Generate batch embeddings
    texts = [
        "Machine learning is transforming technology.",
        "Natural language processing enables text understanding.",
        "Deep learning requires large datasets."
    ]
    embeddings = await provider.generate_embeddings(texts)
    print(f"Generated {len(embeddings)} embeddings")

if __name__ == "__main__":
    asyncio.run(main())
```

## Advanced Features

### Model Management

```python
# Get available models
models = await provider.get_available_models()
print(f"Available models: {models}")

# Get model information
info = await provider.get_model_info()
print(f"Model family: {info.get('details', {}).get('family')}")
```

### Error Handling

```python
from memory_core.embeddings.interfaces import EmbeddingProviderError

try:
    embedding = await provider.generate_embedding("test text")
except EmbeddingProviderError as e:
    print(f"Provider error: {e}")
    print(f"Details: {e.details}")
```

### Custom Server Configuration

```python
# For custom Ollama server
config = {
    'model_name': 'custom-model',
    'base_url': 'http://my-server:8080',
    'timeout': 120,
    'keep_alive': '10m'
}
provider = OllamaEmbeddingProvider(config)
```

## Integration with Memory Engine

```python
from memory_core.embeddings.embedding_manager import EmbeddingManager

# Configure embedding manager to use Ollama
embedding_config = {
    'provider': 'ollama',
    'config': {
        'model_name': 'nomic-embed-text',
        'base_url': 'http://localhost:11434'
    }
}

embedding_manager = EmbeddingManager(embedding_config)
```

## Performance Considerations

- **Local Processing**: All embeddings are generated locally, ensuring privacy
- **No Batch Processing**: Ollama API processes one text at a time
- **Model Loading**: First request may be slower due to model loading
- **Keep-Alive**: Configure appropriate keep-alive duration to balance memory and latency
- **Concurrent Requests**: Provider handles concurrent requests with retries

## Troubleshooting

### Common Issues

1. **Connection Refused**
   - Ensure Ollama server is running: `ollama serve`
   - Check server URL and port

2. **Model Not Found**
   - Pull the model: `ollama pull model-name`
   - Verify model name with: `ollama list`

3. **Timeout Errors**
   - Increase timeout in configuration
   - Check server resources and model size

4. **Memory Issues**
   - Adjust keep-alive duration
   - Use smaller models for resource-constrained environments

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Comparison with Other Providers

| Feature | Ollama | Gemini | OpenAI |
|---------|--------|---------|---------|
| Privacy | ✅ Fully Local | ❌ External API | ❌ External API |
| Cost | ✅ Free | 💰 Usage-based | 💰 Usage-based |
| Internet Required | ❌ No | ✅ Yes | ✅ Yes |
| Model Variety | 🔄 Growing | ✅ Multiple | ✅ Multiple |
| Performance | 🔄 Hardware-dependent | ✅ Fast | ✅ Fast |
| Setup Complexity | 🔄 Moderate | ✅ Easy | ✅ Easy |

The Ollama provider is ideal for applications requiring data privacy, offline operation, or cost control, while cloud providers offer easier setup and potentially better performance.