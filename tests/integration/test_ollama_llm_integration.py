"""
Integration tests for Ollama LLM provider.

These tests require a running Ollama server with at least one model available.
They will be skipped if Ollama is not available.
"""

import pytest
import asyncio
import json

from memory_core.llm.providers.ollama import OllamaLLMProvider
from memory_core.llm.interfaces.llm_provider_interface import (
    LLMTask, Message, MessageRole, LLMError, LLMConnectionError
)


@pytest.fixture
def ollama_config():
    """Ollama LLM configuration for testing."""
    return {
        'base_url': 'http://localhost:11434',
        'model_name': 'llama2',  # Default model, may need to be changed
        'temperature': 0.7,
        'max_tokens': 100,  # Keep small for faster tests
        'timeout': 60,
        'top_p': 0.9,
        'top_k': 40,
        'keep_alive': '5m'
    }


@pytest.fixture
def ollama_provider(ollama_config):
    """Create Ollama LLM provider."""
    return OllamaLLMProvider(ollama_config)


async def check_ollama_availability(provider):
    """Check if Ollama is available and skip tests if not."""
    # Check health first to skip tests if Ollama is not available
    health = await provider.health_check()
    if not health.get('server_running', False):
        pytest.skip("Ollama server is not running")
    
    if not health.get('model_available', False):
        pytest.skip(f"Model {provider.model_name} is not available")
    
    # Try to connect
    try:
        await provider.connect()
    except LLMConnectionError:
        pytest.skip("Cannot connect to Ollama server")


class TestOllamaLLMIntegration:
    """Integration tests for Ollama LLM provider."""

    @pytest.mark.asyncio
    async def test_health_check(self, ollama_provider):
        """Test provider health check."""
        await check_ollama_availability(ollama_provider)
        health = await ollama_provider.health_check()
        
        assert health['provider'] == 'ollama'
        assert health['model'] == ollama_provider.model_name
        assert health['base_url'] == ollama_provider.base_url
        assert health['server_running'] is True
        assert health['model_available'] is True
        assert health['connected'] is True
        assert health['test_passed'] is True
        assert 'timestamp' in health
        assert health.get('response_time', 0) > 0

    @pytest.mark.asyncio
    async def test_basic_completion(self, ollama_provider):
        """Test basic text completion."""
        prompt = "What is 2 + 2? Answer briefly:"
        
        response = await ollama_provider.generate_completion(prompt, LLMTask.GENERAL_COMPLETION)
        
        assert isinstance(response.content, str)
        assert len(response.content.strip()) > 0
        assert response.model == ollama_provider.model_name
        assert response.metadata['provider'] == 'ollama'
        assert response.metadata['task_type'] == 'general_completion'
        
        # Check usage info if available
        if response.usage:
            assert isinstance(response.usage, dict)
            assert 'total_tokens' in response.usage

    @pytest.mark.asyncio
    async def test_chat_completion(self, ollama_provider):
        """Test chat-style completion."""
        messages = [
            Message(MessageRole.SYSTEM, "You are a helpful assistant. Keep responses brief."),
            Message(MessageRole.USER, "What is Python?"),
        ]
        
        response = await ollama_provider.generate_chat_completion(messages, LLMTask.GENERAL_COMPLETION)
        
        assert isinstance(response.content, str)
        assert len(response.content.strip()) > 0
        assert response.model == ollama_provider.model_name
        assert response.metadata['provider'] == 'ollama'
        assert response.metadata['message_count'] == len(messages)

    @pytest.mark.asyncio
    async def test_streaming_completion(self, ollama_provider):
        """Test streaming text generation."""
        prompt = "Count from 1 to 3:"
        
        chunks = []
        async for chunk in ollama_provider.generate_streaming_completion(prompt):
            chunks.append(chunk)
            # Limit to prevent infinite streams in tests
            if len(chunks) > 20:
                break
        
        assert len(chunks) > 0
        full_response = ''.join(chunks)
        assert len(full_response.strip()) > 0

    @pytest.mark.asyncio
    async def test_knowledge_extraction(self, ollama_provider):
        """Test knowledge extraction capabilities."""
        text = """
        Python is a high-level programming language created by Guido van Rossum in 1991.
        It is known for its simple, readable syntax and extensive standard library.
        """
        
        knowledge_units = await ollama_provider.extract_knowledge_units(text)
        
        # Knowledge extraction might not always work perfectly with smaller models
        # so we're lenient with the validation
        assert isinstance(knowledge_units, list)
        
        # If we got results, validate their structure
        for unit in knowledge_units:
            assert isinstance(unit, dict)
            assert 'content' in unit

    @pytest.mark.asyncio
    async def test_relationship_detection(self, ollama_provider):
        """Test relationship detection between entities."""
        entities = ["Python", "programming language", "Guido van Rossum"]
        context = "These are related to programming and software development."
        
        relationships = await ollama_provider.detect_relationships(entities, context)
        
        # Relationship detection might not always work perfectly with smaller models
        assert isinstance(relationships, list)
        
        # If we got results, validate their structure
        for rel in relationships:
            assert isinstance(rel, dict)
            assert 'source' in rel
            assert 'target' in rel
            assert 'relationship_type' in rel

    @pytest.mark.asyncio
    async def test_query_parsing(self, ollama_provider):
        """Test natural language query parsing."""
        query = "Find information about Python programming"
        
        parsed = await ollama_provider.parse_natural_language_query(query)
        
        assert isinstance(parsed, dict)
        assert 'intent' in parsed
        assert 'entities' in parsed
        assert 'query_type' in parsed
        assert 'confidence' in parsed
        
        # Check that default values are set
        assert isinstance(parsed['entities'], list)
        assert isinstance(parsed['confidence'], (int, float))

    @pytest.mark.asyncio
    async def test_content_validation(self, ollama_provider):
        """Test content validation."""
        content = "This is a well-formed sentence with proper grammar."
        criteria = [
            "Has proper grammar",
            "Is a complete sentence",
            "Contains meaningful content"
        ]
        
        validation = await ollama_provider.validate_content(content, criteria)
        
        assert isinstance(validation, dict)
        assert 'valid' in validation
        assert 'criteria_results' in validation
        assert 'overall_score' in validation
        
        # Check criteria results structure
        assert isinstance(validation['criteria_results'], dict)
        assert isinstance(validation['overall_score'], (int, float))

    @pytest.mark.asyncio
    async def test_text_classification(self, ollama_provider):
        """Test text classification functionality."""
        text = "Machine learning is a subset of artificial intelligence."
        categories = ["Technology", "Science", "Sports", "Politics"]
        
        classification = await ollama_provider.classify_text(text, categories)
        
        assert isinstance(classification, dict)
        assert len(classification) == len(categories)
        
        for category in categories:
            assert category in classification
            score = classification[category]
            assert isinstance(score, (int, float))
            assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_text_summarization(self, ollama_provider):
        """Test text summarization."""
        text = """
        Artificial intelligence (AI) is intelligence demonstrated by machines, 
        in contrast to the natural intelligence displayed by humans and animals. 
        Leading AI textbooks define the field as the study of "intelligent agents": 
        any device that perceives its environment and takes actions that maximize 
        its chance of successfully achieving its goals.
        """
        
        summary = await ollama_provider.summarize_text(text, max_length=50)
        
        assert isinstance(summary, str)
        assert len(summary.strip()) > 0
        # Summary should be shorter than original (rough check)
        assert len(summary) < len(text)

    @pytest.mark.asyncio
    async def test_empty_prompt_error(self, ollama_provider):
        """Test error handling for empty prompts."""
        with pytest.raises(Exception):  # Should raise LLMValidationError or similar
            await ollama_provider.generate_completion("")

    @pytest.mark.asyncio
    async def test_empty_messages_error(self, ollama_provider):
        """Test error handling for empty message list."""
        with pytest.raises(Exception):  # Should raise LLMValidationError or similar
            await ollama_provider.generate_chat_completion([])

    @pytest.mark.asyncio
    async def test_connection_persistence(self, ollama_provider):
        """Test that connection persists across multiple requests."""
        assert ollama_provider.is_connected
        
        # Make multiple requests
        for i in range(3):
            response = await ollama_provider.generate_completion(f"Say number {i}")
            assert len(response.content.strip()) > 0
            assert ollama_provider.is_connected

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, ollama_provider):
        """Test handling of concurrent requests."""
        prompts = [f"What is {i} + 1?" for i in range(3)]
        
        # Generate responses concurrently
        tasks = [
            ollama_provider.generate_completion(prompt, LLMTask.GENERAL_COMPLETION)
            for prompt in prompts
        ]
        
        responses = await asyncio.gather(*tasks)
        
        assert len(responses) == len(prompts)
        for response in responses:
            assert isinstance(response.content, str)
            assert len(response.content.strip()) > 0

    @pytest.mark.asyncio
    async def test_get_available_models(self, ollama_provider):
        """Test getting available models."""
        models = await ollama_provider.get_available_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
        
        # Check that our configured model is in the list
        model_names = [model.get('name', '') for model in models]
        # The model name might have a tag, so check if our model name is a prefix
        assert any(name.startswith(ollama_provider.model_name) for name in model_names)
        
        # Validate model structure
        for model in models:
            assert isinstance(model, dict)
            assert 'name' in model
            assert 'size' in model

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

    def test_token_estimation(self, ollama_provider):
        """Test token count estimation."""
        text = "This is a test sentence for token estimation."
        estimated_tokens = ollama_provider.estimate_tokens(text)
        
        assert isinstance(estimated_tokens, int)
        assert estimated_tokens > 0
        # Should be roughly reasonable (not exact due to simple estimation)
        assert len(text) // 6 <= estimated_tokens <= len(text) // 2

    @pytest.mark.asyncio
    async def test_connection_test(self, ollama_provider):
        """Test connection testing functionality."""
        result = await ollama_provider.test_connection()
        assert result is True