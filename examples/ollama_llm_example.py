#!/usr/bin/env python3
"""
Example of using the Ollama LLM provider for local model inference.

This example demonstrates how to use the Ollama LLM provider for various tasks
including completion generation, knowledge extraction, relationship detection,
and natural language query parsing.

Prerequisites:
1. Install and run Ollama locally: https://ollama.ai/
2. Pull a model: `ollama pull llama2`
3. Ensure Ollama is running on localhost:11434

Usage:
    python examples/ollama_llm_example.py
"""

import asyncio
import json
import logging
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_core.llm.providers.ollama.ollama_provider import OllamaLLMProvider
from memory_core.llm.interfaces.llm_provider_interface import (
    LLMTask, Message, MessageRole, LLMError, LLMConnectionError
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_basic_completion():
    """Test basic text completion."""
    print("\n=== Testing Basic Completion ===")
    
    config = {
        'base_url': 'http://localhost:11434',
        'model_name': 'llama2',  # Change to your preferred model
        'temperature': 0.7,
        'max_tokens': 200,
        'timeout': 60
    }
    
    provider = OllamaLLMProvider(config)
    
    try:
        # Connect to Ollama
        await provider.connect()
        print(f"✓ Connected to Ollama server")
        print(f"✓ Using model: {provider.model_name}")
        
        # Test basic completion
        prompt = "Explain what artificial intelligence is in simple terms."
        response = await provider.generate_completion(prompt)
        
        print(f"Prompt: {prompt}")
        print(f"Response: {response.content}")
        print(f"Usage: {response.usage}")
        print(f"Metadata: {response.metadata}")
        
    except LLMConnectionError as e:
        print(f"✗ Connection failed: {e}")
        print("Make sure Ollama is running and the model is available")
    except Exception as e:
        print(f"✗ Error: {e}")
    finally:
        await provider.disconnect()


async def test_chat_completion():
    """Test chat-style conversation."""
    print("\n=== Testing Chat Completion ===")
    
    config = {
        'base_url': 'http://localhost:11434',
        'model_name': 'llama2',
        'temperature': 0.8,
        'max_tokens': 150
    }
    
    provider = OllamaLLMProvider(config)
    
    try:
        await provider.connect()
        
        # Create a conversation
        messages = [
            Message(MessageRole.SYSTEM, "You are a helpful programming assistant."),
            Message(MessageRole.USER, "What is Python?"),
            Message(MessageRole.ASSISTANT, "Python is a high-level programming language known for its simplicity and readability."),
            Message(MessageRole.USER, "Can you give me a simple Python example?")
        ]
        
        response = await provider.generate_chat_completion(messages)
        
        print(f"Chat response: {response.content}")
        print(f"Finish reason: {response.finish_reason}")
        print(f"Token usage: {response.usage}")
        
    except Exception as e:
        print(f"✗ Chat completion error: {e}")
    finally:
        await provider.disconnect()


async def test_streaming_completion():
    """Test streaming text generation."""
    print("\n=== Testing Streaming Completion ===")
    
    config = {
        'base_url': 'http://localhost:11434',
        'model_name': 'llama2',
        'temperature': 0.7,
        'max_tokens': 100
    }
    
    provider = OllamaLLMProvider(config)
    
    try:
        await provider.connect()
        
        prompt = "Write a short story about a robot learning to paint:"
        print(f"Prompt: {prompt}")
        print("Streaming response:")
        
        async for chunk in provider.generate_streaming_completion(prompt):
            print(chunk, end='', flush=True)
        
        print("\n✓ Streaming completed")
        
    except Exception as e:
        print(f"✗ Streaming error: {e}")
    finally:
        await provider.disconnect()


async def test_knowledge_extraction():
    """Test knowledge extraction capabilities."""
    print("\n=== Testing Knowledge Extraction ===")
    
    config = {
        'base_url': 'http://localhost:11434',
        'model_name': 'llama2',
        'temperature': 0.1,  # Lower temperature for more consistent extraction
        'timeout': 90
    }
    
    provider = OllamaLLMProvider(config)
    
    try:
        await provider.connect()
        
        text = """
        Machine learning is a subset of artificial intelligence that focuses on algorithms
        that can learn and improve from data without being explicitly programmed.
        Deep learning, which uses neural networks with multiple layers, is particularly
        effective for image recognition and natural language processing tasks.
        """
        
        knowledge_units = await provider.extract_knowledge_units(
            text,
            source_info={"type": "example_text", "domain": "machine_learning"}
        )
        
        print(f"Extracted {len(knowledge_units)} knowledge units:")
        for i, unit in enumerate(knowledge_units, 1):
            print(f"{i}. {unit.get('content', 'N/A')}")
            print(f"   Tags: {unit.get('tags', [])}")
            print(f"   Domain: {unit.get('metadata', {}).get('domain', 'N/A')}")
            print()
        
    except Exception as e:
        print(f"✗ Knowledge extraction error: {e}")
    finally:
        await provider.disconnect()


async def test_relationship_detection():
    """Test relationship detection between entities."""
    print("\n=== Testing Relationship Detection ===")
    
    config = {
        'base_url': 'http://localhost:11434',
        'model_name': 'llama2',
        'temperature': 0.1
    }
    
    provider = OllamaLLMProvider(config)
    
    try:
        await provider.connect()
        
        entities = [
            "Python programming language",
            "Machine learning",
            "TensorFlow library",
            "Data science",
            "Neural networks"
        ]
        
        context = "These are all related to artificial intelligence and data processing."
        
        relationships = await provider.detect_relationships(entities, context)
        
        print(f"Detected {len(relationships)} relationships:")
        for rel in relationships:
            print(f"- {rel.get('source')} → {rel.get('relationship_type')} → {rel.get('target')}")
            print(f"  Confidence: {rel.get('confidence', 0):.2f}")
            print(f"  Description: {rel.get('description', 'N/A')}")
            print()
        
    except Exception as e:
        print(f"✗ Relationship detection error: {e}")
    finally:
        await provider.disconnect()


async def test_query_parsing():
    """Test natural language query parsing."""
    print("\n=== Testing Query Parsing ===")
    
    config = {
        'base_url': 'http://localhost:11434',
        'model_name': 'llama2',
        'temperature': 0.1
    }
    
    provider = OllamaLLMProvider(config)
    
    try:
        await provider.connect()
        
        queries = [
            "Find all concepts related to machine learning",
            "Show me relationships between Python and data science",
            "Count how many programming languages are mentioned in the database"
        ]
        
        for query in queries:
            parsed = await provider.parse_natural_language_query(query)
            
            print(f"Query: {query}")
            print(f"Intent: {parsed.get('intent')}")
            print(f"Entities: {parsed.get('entities')}")
            print(f"Query type: {parsed.get('query_type')}")
            print(f"Confidence: {parsed.get('confidence', 0):.2f}")
            print()
        
    except Exception as e:
        print(f"✗ Query parsing error: {e}")
    finally:
        await provider.disconnect()


async def test_health_check():
    """Test provider health check."""
    print("\n=== Testing Health Check ===")
    
    config = {
        'base_url': 'http://localhost:11434',
        'model_name': 'llama2'
    }
    
    provider = OllamaLLMProvider(config)
    
    try:
        health = await provider.health_check()
        
        print("Health Check Results:")
        print(f"✓ Provider: {health['provider']}")
        print(f"✓ Model: {health['model']}")
        print(f"✓ Base URL: {health['base_url']}")
        print(f"✓ Server running: {health['server_running']}")
        print(f"✓ Model available: {health['model_available']}")
        print(f"✓ Connected: {health['connected']}")
        print(f"✓ Test passed: {health['test_passed']}")
        
        if health.get('error'):
            print(f"✗ Error: {health['error']}")
        
        if health.get('response_time'):
            print(f"✓ Response time: {health['response_time']:.2f}s")
        
    except Exception as e:
        print(f"✗ Health check error: {e}")


async def test_available_models():
    """Test getting available models."""
    print("\n=== Testing Available Models ===")
    
    config = {
        'base_url': 'http://localhost:11434',
        'model_name': 'llama2'
    }
    
    provider = OllamaLLMProvider(config)
    
    try:
        await provider.connect()
        
        models = await provider.get_available_models()
        
        print(f"Found {len(models)} available models:")
        for model in models:
            size_mb = model.get('size', 0) / (1024 * 1024)  # Convert to MB
            print(f"- {model.get('name')} ({size_mb:.1f} MB)")
            print(f"  Modified: {model.get('modified_at', 'Unknown')}")
        
    except Exception as e:
        print(f"✗ Error getting models: {e}")
    finally:
        await provider.disconnect()


async def main():
    """Run all examples."""
    print("Ollama LLM Provider Examples")
    print("=" * 40)
    
    # Test provider info
    config = {'base_url': 'http://localhost:11434', 'model_name': 'llama2'}
    provider = OllamaLLMProvider(config)
    info = provider.get_provider_info()
    
    print(f"Provider: {info['name']}")
    print(f"Type: {info['type']}")
    print(f"Features: {', '.join(info['features'])}")
    print(f"Supported tasks: {len(info['supported_tasks'])} tasks")
    
    # Run tests
    test_functions = [
        test_health_check,
        test_available_models,
        test_basic_completion,
        test_chat_completion,
        test_streaming_completion,
        test_knowledge_extraction,
        test_relationship_detection,
        test_query_parsing
    ]
    
    for test_func in test_functions:
        try:
            await test_func()
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            break
        except Exception as e:
            print(f"✗ Test {test_func.__name__} failed: {e}")
    
    print("\n=== Examples completed ===")


if __name__ == "__main__":
    asyncio.run(main())