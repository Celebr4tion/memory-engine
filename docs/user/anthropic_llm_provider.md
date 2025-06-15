# Anthropic Claude LLM Provider

The Anthropic Claude LLM provider enables the Memory Engine to use Anthropic's Claude language models for various natural language processing tasks. This provider supports all major Claude models including Claude 3.5 Sonnet, Claude 3 Haiku, and Claude 3 Opus.

## Features

- **Complete LLM Integration**: Supports all Memory Engine LLM tasks
- **Multiple Claude Models**: Compatible with all Claude 3 and 3.5 models
- **Streaming Support**: Real-time response streaming
- **Chat Conversations**: Multi-turn conversations with system messages
- **Knowledge Extraction**: Extract structured knowledge from text
- **Relationship Detection**: Identify relationships between entities
- **Query Parsing**: Parse natural language queries for knowledge graphs
- **Content Validation**: Validate content against custom criteria
- **Error Handling**: Comprehensive error handling with proper exception types
- **Health Monitoring**: Built-in health checks and provider information

## Installation

1. Install the Anthropic library:
```bash
pip install anthropic
```

2. Set your API key as an environment variable:
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

## Configuration

### Environment Variables

The provider requires an Anthropic API key:

```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### Configuration File

Add the Anthropic provider configuration to your `config.yaml`:

```yaml
llm:
  provider: "anthropic"
  
  anthropic:
    api_key: null  # Set via ANTHROPIC_API_KEY environment variable
    model_name: "claude-3-5-sonnet-20241022"
    temperature: 0.7
    max_tokens: 4096
    timeout: 30
    top_p: 0.9
    top_k: null
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `api_key` | string | `null` | Anthropic API key (use environment variable) |
| `model_name` | string | `"claude-3-5-sonnet-20241022"` | Claude model to use |
| `temperature` | float | `0.7` | Sampling temperature (0.0-1.0) |
| `max_tokens` | int | `4096` | Maximum tokens in response |
| `timeout` | int | `30` | Request timeout in seconds |
| `top_p` | float | `0.9` | Top-p sampling parameter |
| `top_k` | int | `null` | Top-k sampling parameter (optional) |
| `base_url` | string | `null` | Custom API endpoint (optional) |

## Supported Models

The provider supports all Claude models:

### Claude 3.5 Models
- `claude-3-5-sonnet-20241022` (recommended)
- `claude-3-5-sonnet-20240620`

### Claude 3 Models
- `claude-3-opus-20240229`
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`

## Usage Examples

### Basic Setup

```python
from memory_core.llm.providers.anthropic import AnthropicLLMProvider
from memory_core.llm.interfaces.llm_provider_interface import LLMTask

# Configuration
config = {
    'api_key': 'your-anthropic-api-key',
    'model_name': 'claude-3-5-sonnet-20241022',
    'temperature': 0.7,
    'max_tokens': 4096
}

# Initialize provider
provider = AnthropicLLMProvider(config)
await provider.connect()
```

### Text Completion

```python
response = await provider.generate_completion(
    prompt="Explain artificial intelligence in simple terms.",
    task_type=LLMTask.GENERAL_COMPLETION
)

print(response.content)
print(f"Model: {response.model}")
print(f"Usage: {response.usage}")
```

### Chat Conversation

```python
from memory_core.llm.interfaces.llm_provider_interface import Message, MessageRole

messages = [
    Message(role=MessageRole.SYSTEM, content="You are a helpful AI assistant."),
    Message(role=MessageRole.USER, content="What is machine learning?"),
    Message(role=MessageRole.ASSISTANT, content="Machine learning is a subset of AI..."),
    Message(role=MessageRole.USER, content="Can you give me an example?")
]

response = await provider.generate_chat_completion(
    messages=messages,
    task_type=LLMTask.GENERAL_COMPLETION
)

print(response.content)
```

### Knowledge Extraction

```python
text = """
Python is a high-level programming language created by Guido van Rossum in 1991.
It emphasizes code readability and supports multiple programming paradigms.
"""

knowledge_units = await provider.extract_knowledge_units(
    text=text,
    source_info={"type": "documentation", "domain": "programming"}
)

for unit in knowledge_units:
    print(f"Content: {unit['content']}")
    print(f"Tags: {unit['tags']}")
    print(f"Confidence: {unit['metadata']['confidence_level']}")
```

### Relationship Detection

```python
entities = [
    "Python programming language",
    "Machine learning",
    "Data science",
    "TensorFlow"
]

relationships = await provider.detect_relationships(
    entities=entities,
    context="Programming and AI development context"
)

for rel in relationships:
    print(f"{rel['source']} -> {rel['target']}")
    print(f"Type: {rel['relationship_type']}")
    print(f"Confidence: {rel['confidence']}")
```

### Natural Language Query Parsing

```python
query = "Find all concepts related to machine learning"

parsed = await provider.parse_natural_language_query(
    query=query,
    context="Knowledge graph with AI and programming concepts"
)

print(f"Intent: {parsed['intent']}")
print(f"Entities: {parsed['entities']}")
print(f"Query type: {parsed['query_type']}")
```

### Content Validation

```python
content = "Python is a programming language created in 1991."

criteria = [
    "Content is factually accurate",
    "Content mentions creation date",
    "Content is written in clear English"
]

validation = await provider.validate_content(
    content=content,
    criteria=criteria
)

print(f"Valid: {validation['valid']}")
print(f"Score: {validation['overall_score']}")
for criterion, result in validation['criteria_results'].items():
    print(f"{criterion}: {'✓' if result else '✗'}")
```

### Streaming Responses

```python
async for chunk in provider.generate_streaming_completion(
    prompt="Write a short story about AI.",
    task_type=LLMTask.GENERAL_COMPLETION
):
    print(chunk, end='', flush=True)
```

### Health Check

```python
health = await provider.health_check()

print(f"Provider: {health['provider']}")
print(f"Connected: {health['connected']}")
print(f"Response time: {health['response_time']}s")
print(f"Test passed: {health['test_passed']}")
```

## Integration with Memory Engine

### Configuration Manager

```python
from memory_core.config.config_manager import ConfigManager
from memory_core.llm.providers.anthropic import AnthropicLLMProvider

# Load configuration
config_manager = ConfigManager()
llm_config = config_manager.get_llm_config()

# Initialize provider
provider = AnthropicLLMProvider(llm_config['anthropic'])
```

### Knowledge Engine

```python
from memory_core.core.knowledge_engine import KnowledgeEngine

# Initialize with Anthropic provider
engine = KnowledgeEngine(
    storage_backend="janusgraph",
    llm_provider="anthropic"
)

# Use for knowledge extraction
await engine.add_knowledge("Python is a programming language.")
```

## Error Handling

The provider includes comprehensive error handling:

```python
from memory_core.llm.interfaces.llm_provider_interface import (
    LLMError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMValidationError
)

try:
    response = await provider.generate_completion("Your prompt")
except LLMConnectionError as e:
    print(f"Connection error: {e}")
except LLMRateLimitError as e:
    print(f"Rate limit exceeded: {e}")
except LLMValidationError as e:
    print(f"Validation error: {e}")
except LLMError as e:
    print(f"General LLM error: {e}")
```

## Best Practices

### API Key Security

- Always use environment variables for API keys
- Never commit API keys to version control
- Use different API keys for development and production

### Model Selection

- **Claude 3.5 Sonnet**: Best balance of performance and cost (recommended)
- **Claude 3 Haiku**: Fastest and most cost-effective for simple tasks
- **Claude 3 Opus**: Most capable for complex reasoning tasks

### Performance Optimization

- Use lower temperatures (0.1-0.3) for consistent, structured outputs
- Set appropriate `max_tokens` limits to control costs
- Use streaming for long responses to improve user experience
- Implement retry logic for rate limit handling

### Temperature Guidelines

- **0.1-0.3**: Structured tasks (knowledge extraction, validation)
- **0.7**: General conversation and balanced creativity
- **0.8-1.0**: Creative writing and brainstorming

## Troubleshooting

### Common Issues

**API Key Error**
```
LLMConnectionError: Anthropic API key is required for Anthropic provider
```
- Ensure `ANTHROPIC_API_KEY` environment variable is set
- Verify the API key is valid and has sufficient credits

**Library Not Found**
```
LLMConnectionError: Anthropic library is not installed
```
- Install the library: `pip install anthropic`

**Rate Limiting**
```
LLMRateLimitError: Rate limit exceeded
```
- Implement exponential backoff retry logic
- Consider upgrading your Anthropic plan
- Reduce request frequency

**Timeout Errors**
- Increase the `timeout` configuration
- Check your network connection
- Try a different model (Haiku is faster)

### Debug Mode

Enable debug logging for detailed information:

```python
import logging
logging.getLogger('memory_core.llm.providers.anthropic').setLevel(logging.DEBUG)
```

## Security Considerations

- API keys provide full access to your Anthropic account
- Use separate API keys for different environments
- Monitor API usage and costs regularly
- Implement proper access controls in production
- Be aware of data privacy when sending content to external APIs

## Performance Characteristics

| Model | Speed | Cost | Capability | Best For |
|-------|-------|------|------------|----------|
| Claude 3.5 Sonnet | Medium | Medium | High | General purpose, reasoning |
| Claude 3 Haiku | Fast | Low | Good | Simple tasks, high volume |
| Claude 3 Opus | Slow | High | Highest | Complex reasoning, analysis |

## Limitations

- Requires internet connection for API access
- Subject to Anthropic's rate limits and pricing
- Response time depends on model and request complexity
- Token limits vary by model
- API availability depends on Anthropic's service status

## Support

For issues with the Anthropic provider:

1. Check the [Anthropic API documentation](https://docs.anthropic.com/)
2. Verify your API key and account status
3. Review the Memory Engine logs for detailed error information
4. Consult the Memory Engine documentation for integration issues

For Anthropic API specific issues, contact [Anthropic Support](https://support.anthropic.com/).