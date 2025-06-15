"""
Unit tests for Ollama LLM provider.

These tests mock the Ollama API responses and do not require a running Ollama server.
"""

import pytest
from unittest.mock import AsyncMock, patch, Mock
import json
import aiohttp

from memory_core.llm.providers.ollama.ollama_provider import OllamaLLMProvider
from memory_core.llm.interfaces.llm_provider_interface import (
    LLMTask, Message, MessageRole, LLMError, LLMConnectionError, LLMValidationError
)


@pytest.fixture
def ollama_config():
    """Ollama LLM configuration for testing."""
    return {
        'base_url': 'http://localhost:11434',
        'model_name': 'llama2',
        'temperature': 0.7,
        'max_tokens': 100,
        'timeout': 30,
        'top_p': 0.9,
        'top_k': 40,
        'keep_alive': '5m',
        'repeat_penalty': 1.1
    }


@pytest.fixture
def ollama_provider(ollama_config):
    """Create Ollama LLM provider for testing."""
    return OllamaLLMProvider(ollama_config)


class TestOllamaLLMProvider:
    """Unit tests for Ollama LLM provider."""

    def test_init_default_config(self):
        """Test provider initialization with default configuration."""
        config = {}
        provider = OllamaLLMProvider(config)
        
        assert provider.base_url == 'http://localhost:11434'
        assert provider.model_name == 'llama2'  # default model
        assert provider.temperature == 0.7
        assert provider.max_tokens == 4096
        assert provider.timeout == 30
        assert provider.top_p == 0.9
        assert provider.top_k == 40
        assert provider.keep_alive == '5m'
        assert provider.repeat_penalty == 1.1

    def test_init_custom_config(self, ollama_config):
        """Test provider initialization with custom configuration."""
        provider = OllamaLLMProvider(ollama_config)
        
        assert provider.base_url == ollama_config['base_url']
        assert provider.model_name == ollama_config['model_name']
        assert provider.temperature == ollama_config['temperature']
        assert provider.max_tokens == ollama_config['max_tokens']
        assert provider.timeout == ollama_config['timeout']
        assert provider.top_p == ollama_config['top_p']
        assert provider.top_k == ollama_config['top_k']
        assert provider.keep_alive == ollama_config['keep_alive']
        assert provider.repeat_penalty == ollama_config['repeat_penalty']

    def test_get_default_model(self, ollama_provider):
        """Test getting default model name."""
        assert ollama_provider.get_default_model() == "llama2"

    def test_provider_info(self, ollama_provider):
        """Test provider information methods."""
        info = ollama_provider.get_provider_info()
        
        assert info['name'] == 'OllamaLLMProvider'
        assert info['provider'] == 'ollama'
        assert info['type'] == 'local'
        assert info['model'] == ollama_provider.model_name
        assert info['base_url'] == ollama_provider.base_url
        assert isinstance(info['supported_tasks'], list)
        assert len(info['supported_tasks']) > 0
        assert isinstance(info['features'], list)
        assert 'local_inference' in info['features']
        assert 'streaming' in info['features']
        assert 'model_pulling' in info['features']

    def test_supported_tasks(self, ollama_provider):
        """Test supported tasks listing."""
        tasks = ollama_provider.get_supported_tasks()
        
        assert isinstance(tasks, list)
        assert len(tasks) > 0
        
        # Check that we support expected tasks
        task_values = [task.value for task in tasks]
        assert 'general_completion' in task_values
        assert 'knowledge_extraction' in task_values
        assert 'relationship_detection' in task_values
        assert 'natural_language_query' in task_values
        assert 'text_classification' in task_values
        assert 'summarization' in task_values
        assert 'content_validation' in task_values

    def test_token_estimation(self, ollama_provider):
        """Test token count estimation."""
        text = "This is a test sentence for token estimation."
        estimated_tokens = ollama_provider.estimate_tokens(text)
        
        assert isinstance(estimated_tokens, int)
        assert estimated_tokens > 0
        # Should be roughly reasonable (not exact due to simple estimation)
        assert len(text) // 6 <= estimated_tokens <= len(text) // 2

    def test_should_use_json_mode(self, ollama_provider):
        """Test JSON mode detection for different task types."""
        # Tasks that should use JSON mode
        json_tasks = [
            LLMTask.KNOWLEDGE_EXTRACTION,
            LLMTask.RELATIONSHIP_DETECTION,
            LLMTask.NATURAL_LANGUAGE_QUERY,
            LLMTask.TEXT_CLASSIFICATION,
            LLMTask.CONTENT_VALIDATION
        ]
        
        for task in json_tasks:
            assert ollama_provider._should_use_json_mode(task)
        
        # Tasks that should NOT use JSON mode
        non_json_tasks = [
            LLMTask.GENERAL_COMPLETION,
            LLMTask.SUMMARIZATION
        ]
        
        for task in non_json_tasks:
            assert not ollama_provider._should_use_json_mode(task)

    def test_clean_markdown_json(self, ollama_provider):
        """Test markdown cleaning for JSON responses."""
        # Test with markdown code blocks
        markdown_json = '''```json
{
    "test": "value"
}
```'''
        cleaned = ollama_provider._clean_markdown_json(markdown_json)
        expected = '''{\n    "test": "value"\n}'''
        assert cleaned == expected
        
        # Test with language-specific code blocks
        markdown_json2 = '''```
{
    "another": "test"
}
```'''
        cleaned2 = ollama_provider._clean_markdown_json(markdown_json2)
        expected2 = '''{\n    "another": "test"\n}'''
        assert cleaned2 == expected2
        
        # Test without markdown (should remain unchanged)
        plain_json = '{"plain": "json"}'
        cleaned3 = ollama_provider._clean_markdown_json(plain_json)
        assert cleaned3 == plain_json

    def test_convert_messages_to_ollama(self, ollama_provider):
        """Test message conversion to Ollama format."""
        messages = [
            Message(MessageRole.SYSTEM, "You are a helpful assistant."),
            Message(MessageRole.USER, "Hello!"),
            Message(MessageRole.ASSISTANT, "Hi there!")
        ]
        
        ollama_messages = ollama_provider._convert_messages_to_ollama(messages)
        
        assert len(ollama_messages) == 3
        assert ollama_messages[0] == {"role": "system", "content": "You are a helpful assistant."}
        assert ollama_messages[1] == {"role": "user", "content": "Hello!"}
        assert ollama_messages[2] == {"role": "assistant", "content": "Hi there!"}

    @pytest.mark.asyncio
    async def test_empty_prompt_validation(self, ollama_provider):
        """Test validation of empty prompts."""
        with pytest.raises(LLMValidationError) as exc_info:
            await ollama_provider.generate_completion("")
        
        assert "Prompt cannot be empty" in str(exc_info.value)
        
        with pytest.raises(LLMValidationError) as exc_info:
            await ollama_provider.generate_completion("   ")  # whitespace only
        
        assert "Prompt cannot be empty" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_empty_messages_validation(self, ollama_provider):
        """Test validation of empty message lists."""
        with pytest.raises(LLMValidationError) as exc_info:
            await ollama_provider.generate_chat_completion([])
        
        assert "Messages cannot be empty" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_empty_text_streaming_validation(self, ollama_provider):
        """Test validation of empty text for streaming."""
        with pytest.raises(LLMValidationError) as exc_info:
            async for _ in ollama_provider.generate_streaming_completion(""):
                pass
        
        assert "Prompt cannot be empty" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_empty_query_validation(self, ollama_provider):
        """Test validation of empty natural language queries."""
        with pytest.raises(LLMValidationError) as exc_info:
            await ollama_provider.parse_natural_language_query("")
        
        assert "Query cannot be empty" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_empty_content_validation(self, ollama_provider):
        """Test validation of empty content for validation."""
        with pytest.raises(LLMValidationError) as exc_info:
            await ollama_provider.validate_content("", ["criterion1"])
        
        assert "Content cannot be empty" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_empty_criteria_validation(self, ollama_provider):
        """Test validation of empty criteria list."""
        with pytest.raises(LLMValidationError) as exc_info:
            await ollama_provider.validate_content("content", [])
        
        assert "Validation criteria cannot be empty" in str(exc_info.value)

    def test_knowledge_extraction_prompt_creation(self, ollama_provider):
        """Test knowledge extraction prompt creation."""
        text = "Python is a programming language."
        source_info = {"type": "test", "domain": "programming"}
        
        prompt = ollama_provider._create_knowledge_extraction_prompt(text, source_info)
        
        assert "knowledge units" in prompt.lower()
        assert text in prompt
        assert json.dumps(source_info) in prompt
        assert "json" in prompt.lower()

    def test_relationship_detection_prompt_creation(self, ollama_provider):
        """Test relationship detection prompt creation."""
        entities = ["Python", "programming", "language"]
        context = "Programming context"
        
        prompt = ollama_provider._create_relationship_detection_prompt(entities, context)
        
        assert "relationships" in prompt.lower()
        assert all(entity in prompt for entity in entities)
        assert context in prompt
        assert "json" in prompt.lower()

    def test_query_parsing_prompt_creation(self, ollama_provider):
        """Test natural language query parsing prompt creation."""
        query = "Find Python tutorials"
        context = "Programming context"
        
        prompt = ollama_provider._create_query_parsing_prompt(query, context)
        
        assert "query" in prompt.lower()
        assert query in prompt
        assert context in prompt
        assert "json" in prompt.lower()

    def test_validation_prompt_creation(self, ollama_provider):
        """Test content validation prompt creation."""
        content = "This is test content."
        criteria = ["Has proper grammar", "Is meaningful"]
        
        prompt = ollama_provider._create_validation_prompt(content, criteria)
        
        assert "validate" in prompt.lower()
        assert content in prompt
        assert all(criterion in prompt for criterion in criteria)
        assert "json" in prompt.lower()

    @pytest.mark.asyncio
    async def test_disconnect_cleanup(self, ollama_provider):
        """Test disconnection cleanup."""
        # Mock session
        mock_session = AsyncMock()
        mock_session.closed = False  # Session appears open
        ollama_provider._session = mock_session
        ollama_provider._is_connected = True
        
        result = await ollama_provider.disconnect()
        
        assert result is True
        assert not ollama_provider._is_connected
        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_error_handling(self, ollama_provider):
        """Test disconnect handles errors gracefully."""
        # Mock session that raises error on close
        mock_session = AsyncMock()
        mock_session.close.side_effect = Exception("Close error")
        ollama_provider._session = mock_session
        ollama_provider._is_connected = True
        
        result = await ollama_provider.disconnect()
        
        # Should still return True and set disconnected state
        assert result is True
        assert not ollama_provider._is_connected

    def test_is_connected_property(self, ollama_provider):
        """Test the is_connected property."""
        # Initially not connected
        assert not ollama_provider.is_connected
        
        # Set connected state
        ollama_provider._is_connected = True
        assert ollama_provider.is_connected
        
        # Set disconnected state
        ollama_provider._is_connected = False
        assert not ollama_provider.is_connected