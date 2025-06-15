"""
Unit tests for the Anthropic Claude LLM provider.

These tests verify the functionality of the AnthropicLLMProvider class
including initialization, configuration validation, and core methods.
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

# Import the provider and related classes
from memory_core.llm.providers.anthropic.anthropic_provider import AnthropicLLMProvider
from memory_core.llm.interfaces.llm_provider_interface import (
    LLMTask,
    MessageRole,
    Message,
    LLMResponse,
    LLMError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMValidationError,
)


class MockAnthropicResponse:
    """Mock response object for Anthropic API."""

    def __init__(
        self,
        content: str,
        model: str = "claude-3-5-sonnet-20241022",
        stop_reason: str = "end_turn",
        usage: Dict = None,
    ):
        self.content = [Mock(text=content)]
        self.model = model
        self.stop_reason = stop_reason
        self.usage = (
            Mock(
                input_tokens=usage.get("input_tokens", 10) if usage else 10,
                output_tokens=usage.get("output_tokens", 20) if usage else 20,
            )
            if usage or True
            else None
        )


class MockAnthropicClient:
    """Mock Anthropic client for testing."""

    def __init__(self):
        self.messages = Mock()
        self.messages.create = AsyncMock()

    async def close(self):
        pass


@pytest.fixture
def mock_anthropic_available():
    """Mock the anthropic library availability."""
    with patch("memory_core.llm.providers.anthropic.anthropic_provider.ANTHROPIC_AVAILABLE", True):
        with patch(
            "memory_core.llm.providers.anthropic.anthropic_provider.AsyncAnthropic"
        ) as mock_client:
            mock_client.return_value = MockAnthropicClient()
            yield mock_client


@pytest.fixture
def basic_config():
    """Basic configuration for testing."""
    return {
        "api_key": "test-api-key",
        "model_name": "claude-3-5-sonnet-20241022",
        "temperature": 0.7,
        "max_tokens": 4096,
        "timeout": 30,
        "top_p": 0.9,
    }


@pytest.fixture
def provider(mock_anthropic_available, basic_config):
    """Create a provider instance for testing."""
    return AnthropicLLMProvider(basic_config)


class TestAnthropicLLMProviderInitialization:
    """Test provider initialization and configuration."""

    def test_anthropic_not_available(self):
        """Test error when anthropic library is not available."""
        with patch(
            "memory_core.llm.providers.anthropic.anthropic_provider.ANTHROPIC_AVAILABLE", False
        ):
            with pytest.raises(LLMConnectionError, match="Anthropic library is not installed"):
                AnthropicLLMProvider({"api_key": "test"})

    def test_missing_api_key(self, mock_anthropic_available):
        """Test error when API key is missing."""
        config = {"model_name": "claude-3-5-sonnet-20241022"}

        with pytest.raises(LLMConnectionError, match="Anthropic API key is required"):
            AnthropicLLMProvider(config)

    def test_successful_initialization(self, provider, basic_config):
        """Test successful provider initialization."""
        assert provider.api_key == basic_config["api_key"]
        assert provider.model_name == basic_config["model_name"]
        assert provider.temperature == basic_config["temperature"]
        assert provider.max_tokens == basic_config["max_tokens"]
        assert provider.timeout == basic_config["timeout"]
        assert provider.top_p == basic_config["top_p"]
        assert not provider.is_connected

    def test_default_model(self, provider):
        """Test default model name."""
        assert provider.get_default_model() == "claude-3-5-sonnet-20241022"

    def test_custom_configuration(self, mock_anthropic_available):
        """Test provider with custom configuration."""
        config = {
            "api_key": "custom-key",
            "model_name": "claude-3-haiku-20240307",
            "temperature": 0.5,
            "max_tokens": 2048,
            "timeout": 60,
            "top_p": 0.8,
            "top_k": 40,
            "base_url": "https://custom.api.url",
        }

        provider = AnthropicLLMProvider(config)

        assert provider.api_key == "custom-key"
        assert provider.model_name == "claude-3-haiku-20240307"
        assert provider.temperature == 0.5
        assert provider.max_tokens == 2048
        assert provider.timeout == 60
        assert provider.top_p == 0.8
        assert provider.top_k == 40
        assert provider.base_url == "https://custom.api.url"


class TestAnthropicLLMProviderConnection:
    """Test provider connection management."""

    @pytest.mark.asyncio
    async def test_connect_success(self, provider):
        """Test successful connection."""
        # Mock the test_connection method
        provider.test_connection = AsyncMock(return_value=True)

        result = await provider.connect()

        assert result is True
        assert provider.is_connected is True

    @pytest.mark.asyncio
    async def test_connect_failure(self, provider):
        """Test connection failure."""
        # Mock the test_connection method to raise an exception
        provider.test_connection = AsyncMock(side_effect=Exception("Connection failed"))

        with pytest.raises(LLMConnectionError, match="Failed to connect to Anthropic API"):
            await provider.connect()

        assert provider.is_connected is False

    @pytest.mark.asyncio
    async def test_disconnect(self, provider):
        """Test disconnection."""
        provider._is_connected = True

        result = await provider.disconnect()

        assert result is True
        assert provider.is_connected is False


class TestAnthropicLLMProviderCompletion:
    """Test completion generation methods."""

    @pytest.mark.asyncio
    async def test_generate_completion_success(self, provider):
        """Test successful completion generation."""
        # Mock the client response
        mock_response = MockAnthropicResponse("This is a test response.")
        provider.client.messages.create.return_value = mock_response

        response = await provider.generate_completion(
            prompt="Test prompt", task_type=LLMTask.GENERAL_COMPLETION
        )

        assert isinstance(response, LLMResponse)
        assert response.content == "This is a test response."
        assert response.model == "claude-3-5-sonnet-20241022"
        assert response.finish_reason == "end_turn"
        assert response.metadata["provider"] == "anthropic"
        assert response.usage["input_tokens"] == 10
        assert response.usage["output_tokens"] == 20

    @pytest.mark.asyncio
    async def test_generate_completion_empty_prompt(self, provider):
        """Test completion with empty prompt."""
        with pytest.raises(LLMValidationError, match="Prompt cannot be empty"):
            await provider.generate_completion("")

    @pytest.mark.asyncio
    async def test_generate_chat_completion_success(self, provider):
        """Test successful chat completion generation."""
        messages = [
            Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            Message(role=MessageRole.USER, content="Hello!"),
        ]

        # Mock the client response
        mock_response = MockAnthropicResponse("Hello! How can I help you?")
        provider.client.messages.create.return_value = mock_response

        response = await provider.generate_chat_completion(
            messages=messages, task_type=LLMTask.GENERAL_COMPLETION
        )

        assert isinstance(response, LLMResponse)
        assert response.content == "Hello! How can I help you?"
        assert response.metadata["message_count"] == 2

        # Verify the client was called with correct parameters
        call_args = provider.client.messages.create.call_args
        assert call_args[1]["system"] == "You are a helpful assistant."
        assert len(call_args[1]["messages"]) == 1
        assert call_args[1]["messages"][0]["role"] == "user"
        assert call_args[1]["messages"][0]["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_generate_chat_completion_empty_messages(self, provider):
        """Test chat completion with empty messages."""
        with pytest.raises(LLMValidationError, match="Messages cannot be empty"):
            await provider.generate_chat_completion([])

    @pytest.mark.asyncio
    async def test_structured_output_request(self, provider):
        """Test that structured tasks request JSON format."""
        mock_response = MockAnthropicResponse('{"result": "test"}')
        provider.client.messages.create.return_value = mock_response

        await provider.generate_completion(
            prompt="Extract knowledge from this text.", task_type=LLMTask.KNOWLEDGE_EXTRACTION
        )

        # Verify that JSON format was requested
        call_args = provider.client.messages.create.call_args
        messages = call_args[1]["messages"]
        assert "json" in messages[0]["content"].lower()


class TestAnthropicLLMProviderTasks:
    """Test specialized LLM tasks."""

    @pytest.mark.asyncio
    async def test_extract_knowledge_units_success(self, provider):
        """Test successful knowledge extraction."""
        # Mock JSON response
        knowledge_data = [
            {
                "content": "Python is a programming language",
                "tags": ["programming", "python"],
                "metadata": {"confidence_level": 0.9, "domain": "technology"},
            }
        ]

        mock_response = MockAnthropicResponse(json.dumps(knowledge_data))
        provider.client.messages.create.return_value = mock_response

        result = await provider.extract_knowledge_units(
            text="Python is a high-level programming language.",
            source_info={"type": "text", "domain": "programming"},
        )

        assert len(result) == 1
        assert result[0]["content"] == "Python is a programming language"
        assert "programming" in result[0]["tags"]

    @pytest.mark.asyncio
    async def test_extract_knowledge_units_empty_text(self, provider):
        """Test knowledge extraction with empty text."""
        result = await provider.extract_knowledge_units("")
        assert result == []

    @pytest.mark.asyncio
    async def test_extract_knowledge_units_invalid_json(self, provider):
        """Test knowledge extraction with invalid JSON response."""
        mock_response = MockAnthropicResponse("Invalid JSON response")
        provider.client.messages.create.return_value = mock_response

        result = await provider.extract_knowledge_units("Some text")
        assert result == []

    @pytest.mark.asyncio
    async def test_detect_relationships_success(self, provider):
        """Test successful relationship detection."""
        # Mock JSON response
        relationships_data = [
            {
                "source": "Python",
                "target": "Machine Learning",
                "relationship_type": "used_for",
                "confidence": 0.8,
            }
        ]

        mock_response = MockAnthropicResponse(json.dumps(relationships_data))
        provider.client.messages.create.return_value = mock_response

        result = await provider.detect_relationships(
            entities=["Python", "Machine Learning", "Data Science"], context="Programming context"
        )

        assert len(result) == 1
        assert result[0]["source"] == "Python"
        assert result[0]["target"] == "Machine Learning"
        assert result[0]["relationship_type"] == "used_for"

    @pytest.mark.asyncio
    async def test_detect_relationships_insufficient_entities(self, provider):
        """Test relationship detection with insufficient entities."""
        result = await provider.detect_relationships(["Python"])
        assert result == []

    @pytest.mark.asyncio
    async def test_parse_natural_language_query_success(self, provider):
        """Test successful query parsing."""
        # Mock JSON response
        query_data = {
            "intent": "search",
            "entities": ["machine learning"],
            "query_type": "semantic_search",
            "confidence": 0.9,
        }

        mock_response = MockAnthropicResponse(json.dumps(query_data))
        provider.client.messages.create.return_value = mock_response

        result = await provider.parse_natural_language_query(
            query="Find information about machine learning"
        )

        assert result["intent"] == "search"
        assert "machine learning" in result["entities"]
        assert result["query_type"] == "semantic_search"
        assert result["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_parse_natural_language_query_empty(self, provider):
        """Test query parsing with empty query."""
        with pytest.raises(LLMValidationError, match="Query cannot be empty"):
            await provider.parse_natural_language_query("")

    @pytest.mark.asyncio
    async def test_validate_content_success(self, provider):
        """Test successful content validation."""
        # Mock JSON response
        validation_data = {
            "valid": True,
            "criteria_results": {"Has proper grammar": True, "Contains factual information": True},
            "overall_score": 0.95,
        }

        mock_response = MockAnthropicResponse(json.dumps(validation_data))
        provider.client.messages.create.return_value = mock_response

        result = await provider.validate_content(
            content="This is well-written content with facts.",
            criteria=["Has proper grammar", "Contains factual information"],
        )

        assert result["valid"] is True
        assert result["overall_score"] == 0.95
        assert result["criteria_results"]["Has proper grammar"] is True

    @pytest.mark.asyncio
    async def test_validate_content_empty_content(self, provider):
        """Test content validation with empty content."""
        with pytest.raises(LLMValidationError, match="Content cannot be empty"):
            await provider.validate_content("", ["criterion"])

    @pytest.mark.asyncio
    async def test_validate_content_empty_criteria(self, provider):
        """Test content validation with empty criteria."""
        with pytest.raises(LLMValidationError, match="Validation criteria cannot be empty"):
            await provider.validate_content("content", [])


class TestAnthropicLLMProviderUtilities:
    """Test utility methods."""

    def test_clean_markdown_json(self, provider):
        """Test markdown cleaning utility."""
        markdown_text = '```json\n{"key": "value"}\n```'
        cleaned = provider._clean_markdown_json(markdown_text)
        assert cleaned == '{"key": "value"}'

    def test_convert_messages_to_anthropic(self, provider):
        """Test message conversion utility."""
        messages = [
            Message(role=MessageRole.SYSTEM, content="System message"),
            Message(role=MessageRole.USER, content="User message"),
            Message(role=MessageRole.ASSISTANT, content="Assistant message"),
        ]

        anthropic_messages, system_message = provider._convert_messages_to_anthropic(messages)

        assert system_message == "System message"
        assert len(anthropic_messages) == 2
        assert anthropic_messages[0]["role"] == "user"
        assert anthropic_messages[0]["content"] == "User message"
        assert anthropic_messages[1]["role"] == "assistant"
        assert anthropic_messages[1]["content"] == "Assistant message"

    def test_should_use_structured_output(self, provider):
        """Test structured output detection."""
        assert provider._should_use_structured_output(LLMTask.KNOWLEDGE_EXTRACTION)
        assert provider._should_use_structured_output(LLMTask.RELATIONSHIP_DETECTION)
        assert not provider._should_use_structured_output(LLMTask.GENERAL_COMPLETION)

    @pytest.mark.asyncio
    async def test_health_check_success(self, provider):
        """Test successful health check."""
        # Mock successful completion
        provider.generate_completion = AsyncMock(
            return_value=LLMResponse(content="OK", metadata={}, model="claude-3-5-sonnet-20241022")
        )

        health = await provider.health_check()

        assert health["provider"] == "anthropic"
        assert health["connected"] is True
        assert health["test_passed"] is True
        assert health["api_key_valid"] is True
        assert health["response_time"] is not None

    @pytest.mark.asyncio
    async def test_health_check_failure(self, provider):
        """Test health check failure."""
        # Mock failed completion
        provider.generate_completion = AsyncMock(side_effect=Exception("API error"))

        health = await provider.health_check()

        assert health["provider"] == "anthropic"
        assert health["connected"] is False
        assert health["test_passed"] is False
        assert "API error" in health["error"]

    def test_get_provider_info(self, provider):
        """Test provider information."""
        info = provider.get_provider_info()

        assert info["name"] == "AnthropicLLMProvider"
        assert info["provider"] == "anthropic"
        assert info["type"] == "api"
        assert "streaming" in info["features"]
        assert "knowledge_extraction" in info["features"]
        assert len(info["supported_tasks"]) > 0

    def test_get_supported_tasks(self, provider):
        """Test supported tasks list."""
        tasks = provider.get_supported_tasks()

        assert LLMTask.GENERAL_COMPLETION in tasks
        assert LLMTask.KNOWLEDGE_EXTRACTION in tasks
        assert LLMTask.RELATIONSHIP_DETECTION in tasks
        assert LLMTask.NATURAL_LANGUAGE_QUERY in tasks
        assert LLMTask.CONTENT_VALIDATION in tasks

    def test_estimate_tokens(self, provider):
        """Test token estimation."""
        text = "This is a test string with multiple words."
        tokens = provider.estimate_tokens(text)

        # Should be approximately len(text) / 3.5
        expected = int(len(text) / 3.5)
        assert abs(tokens - expected) <= 1


class TestAnthropicLLMProviderErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_anthropic_rate_limit_error(self, provider):
        """Test handling of rate limit errors."""

        # Create a mock exception that looks like an Anthropic error
        class MockRateLimitError(Exception):
            pass

        with patch(
            "memory_core.llm.providers.anthropic.anthropic_provider.anthropic"
        ) as mock_anthropic:
            mock_anthropic.RateLimitError = MockRateLimitError
            mock_anthropic.APIConnectionError = Exception
            mock_anthropic.APIError = Exception

            # Make the client raise this error
            provider.client.messages.create.side_effect = MockRateLimitError("Rate limited")

            with pytest.raises(LLMRateLimitError, match="Rate limit exceeded"):
                await provider.generate_completion("test prompt")

    @pytest.mark.asyncio
    async def test_anthropic_api_error(self, provider):
        """Test handling of API errors."""

        # Create a mock exception that looks like an Anthropic error
        class MockAPIError(Exception):
            pass

        with patch(
            "memory_core.llm.providers.anthropic.anthropic_provider.anthropic"
        ) as mock_anthropic:
            mock_anthropic.RateLimitError = Exception
            mock_anthropic.APIConnectionError = Exception
            mock_anthropic.APIError = MockAPIError

            # Make the client raise this error
            provider.client.messages.create.side_effect = MockAPIError("API error")

            with pytest.raises(LLMError, match="Anthropic API error"):
                await provider.generate_completion("test prompt")

    @pytest.mark.asyncio
    async def test_anthropic_connection_error(self, provider):
        """Test handling of connection errors."""

        # Create a mock exception that looks like an Anthropic error
        class MockConnectionError(Exception):
            pass

        with patch(
            "memory_core.llm.providers.anthropic.anthropic_provider.anthropic"
        ) as mock_anthropic:
            mock_anthropic.RateLimitError = Exception
            mock_anthropic.APIConnectionError = MockConnectionError
            mock_anthropic.APIError = Exception

            # Make the client raise this error
            provider.client.messages.create.side_effect = MockConnectionError("Connection error")

            with pytest.raises(LLMConnectionError, match="Failed to connect to Anthropic API"):
                await provider.generate_completion("test prompt")


if __name__ == "__main__":
    pytest.main([__file__])
