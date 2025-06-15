#!/usr/bin/env python3
"""
Anthropic Claude LLM Provider Example

This example demonstrates how to use the Anthropic Claude LLM provider
with the Memory Engine for various language model tasks.

Prerequisites:
    1. Install the anthropic library: pip install anthropic
    2. Set your API key: export ANTHROPIC_API_KEY="your-api-key-here"
    3. Or pass the API key directly in the configuration

Features demonstrated:
    - Provider initialization and connection
    - Text completion and chat completion
    - Knowledge extraction from text
    - Relationship detection between entities
    - Natural language query parsing
    - Content validation
    - Streaming responses
    - Health checks and provider information
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from memory_core.llm.providers.anthropic import AnthropicLLMProvider
from memory_core.llm.interfaces.llm_provider_interface import (
    LLMTask,
    MessageRole,
    Message,
    LLMError,
)


async def basic_completion_example(provider: AnthropicLLMProvider):
    """Demonstrate basic text completion."""
    print("=" * 60)
    print("BASIC COMPLETION EXAMPLE")
    print("=" * 60)

    try:
        prompt = "Explain the concept of artificial intelligence in 2-3 sentences."

        print(f"Prompt: {prompt}")
        print("\nGenerating response...")

        response = await provider.generate_completion(
            prompt=prompt, task_type=LLMTask.GENERAL_COMPLETION, temperature=0.7
        )

        print(f"Response: {response.content}")
        print(f"Model: {response.model}")
        print(f"Finish reason: {response.finish_reason}")

        if response.usage:
            print(f"Token usage: {response.usage}")

        print("✓ Basic completion successful")

    except Exception as e:
        print(f"✗ Basic completion failed: {str(e)}")


async def chat_completion_example(provider: AnthropicLLMProvider):
    """Demonstrate chat-based completion with conversation context."""
    print("\n" + "=" * 60)
    print("CHAT COMPLETION EXAMPLE")
    print("=" * 60)

    try:
        # Create a conversation
        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content="You are a helpful AI assistant specializing in technology.",
            ),
            Message(role=MessageRole.USER, content="What is machine learning?"),
            Message(
                role=MessageRole.ASSISTANT,
                content="Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task.",
            ),
            Message(
                role=MessageRole.USER,
                content="Can you give me an example of how it's used in practice?",
            ),
        ]

        print("Conversation context:")
        for msg in messages:
            print(f"  {msg.role.value}: {msg.content}")

        print("\nGenerating response...")

        response = await provider.generate_chat_completion(
            messages=messages, task_type=LLMTask.GENERAL_COMPLETION, temperature=0.7
        )

        print(f"\nAssistant: {response.content}")

        if response.usage:
            print(f"Token usage: {response.usage}")

        print("✓ Chat completion successful")

    except Exception as e:
        print(f"✗ Chat completion failed: {str(e)}")


async def knowledge_extraction_example(provider: AnthropicLLMProvider):
    """Demonstrate knowledge extraction from text."""
    print("\n" + "=" * 60)
    print("KNOWLEDGE EXTRACTION EXAMPLE")
    print("=" * 60)

    try:
        text = """
        Python is a high-level programming language created by Guido van Rossum in 1991.
        It emphasizes code readability and supports multiple programming paradigms including
        procedural, object-oriented, and functional programming. Python is widely used in
        web development, data science, artificial intelligence, and automation. The language
        has a large standard library and an active community that contributes to its ecosystem.
        """

        source_info = {"type": "example_text", "domain": "programming", "language": "en"}

        print(f"Text to analyze: {text.strip()}")
        print(f"Source info: {json.dumps(source_info, indent=2)}")
        print("\nExtracting knowledge units...")

        knowledge_units = await provider.extract_knowledge_units(text=text, source_info=source_info)

        print(f"\nExtracted {len(knowledge_units)} knowledge units:")
        for i, unit in enumerate(knowledge_units, 1):
            print(f"\n{i}. {unit.get('content', 'N/A')}")
            print(f"   Tags: {unit.get('tags', [])}")
            print(f"   Domain: {unit.get('metadata', {}).get('domain', 'N/A')}")
            print(f"   Confidence: {unit.get('metadata', {}).get('confidence_level', 'N/A')}")

        print("✓ Knowledge extraction successful")

    except Exception as e:
        print(f"✗ Knowledge extraction failed: {str(e)}")


async def relationship_detection_example(provider: AnthropicLLMProvider):
    """Demonstrate relationship detection between entities."""
    print("\n" + "=" * 60)
    print("RELATIONSHIP DETECTION EXAMPLE")
    print("=" * 60)

    try:
        entities = [
            "Python programming language",
            "Machine learning",
            "Data science",
            "TensorFlow",
            "Neural networks",
            "Artificial intelligence",
        ]

        context = "These entities are related to the field of computer science and AI development."

        print(f"Entities: {entities}")
        print(f"Context: {context}")
        print("\nDetecting relationships...")

        relationships = await provider.detect_relationships(entities=entities, context=context)

        print(f"\nDetected {len(relationships)} relationships:")
        for i, rel in enumerate(relationships, 1):
            print(f"\n{i}. {rel.get('source', 'N/A')} -> {rel.get('target', 'N/A')}")
            print(f"   Type: {rel.get('relationship_type', 'N/A')}")
            print(f"   Direction: {rel.get('direction', 'N/A')}")
            print(f"   Confidence: {rel.get('confidence', 'N/A')}")
            print(f"   Description: {rel.get('description', 'N/A')}")

        print("✓ Relationship detection successful")

    except Exception as e:
        print(f"✗ Relationship detection failed: {str(e)}")


async def query_parsing_example(provider: AnthropicLLMProvider):
    """Demonstrate natural language query parsing."""
    print("\n" + "=" * 60)
    print("QUERY PARSING EXAMPLE")
    print("=" * 60)

    try:
        queries = [
            "Find all concepts related to machine learning",
            "Show me the relationships between Python and data science",
            "What programming languages are used for AI development?",
            "Count how many neural network types are mentioned in the database",
        ]

        for query in queries:
            print(f"\nQuery: '{query}'")
            print("Parsing...")

            parsed = await provider.parse_natural_language_query(
                query=query, context="Knowledge graph containing programming and AI concepts"
            )

            print(f"  Intent: {parsed.get('intent', 'N/A')}")
            print(f"  Query type: {parsed.get('query_type', 'N/A')}")
            print(f"  Entities: {parsed.get('entities', [])}")
            print(f"  Keywords: {parsed.get('semantic_keywords', [])}")
            print(f"  Confidence: {parsed.get('confidence', 'N/A')}")

        print("\n✓ Query parsing successful")

    except Exception as e:
        print(f"✗ Query parsing failed: {str(e)}")


async def content_validation_example(provider: AnthropicLLMProvider):
    """Demonstrate content validation against criteria."""
    print("\n" + "=" * 60)
    print("CONTENT VALIDATION EXAMPLE")
    print("=" * 60)

    try:
        content = """
        Python is a programming language that was created in 1991. It is widely used
        for web development and data analysis. The language supports object-oriented
        programming and has a large community of developers.
        """

        criteria = [
            "Content mentions the creation year of Python",
            "Content describes at least two use cases for Python",
            "Content mentions programming paradigms supported by Python",
            "Content is factually accurate",
            "Content is written in clear, understandable English",
        ]

        print(f"Content: {content.strip()}")
        print(f"\nValidation criteria:")
        for i, criterion in enumerate(criteria, 1):
            print(f"  {i}. {criterion}")

        print("\nValidating content...")

        validation = await provider.validate_content(content=content, criteria=criteria)

        print(f"\nValidation results:")
        print(f"  Overall valid: {validation.get('valid', False)}")
        print(f"  Overall score: {validation.get('overall_score', 0.0)}")

        print(f"\nCriteria results:")
        for criterion, result in validation.get("criteria_results", {}).items():
            status = "✓" if result else "✗"
            print(f"  {status} {criterion}")

        if validation.get("errors"):
            print(f"\nErrors: {validation['errors']}")

        if validation.get("suggestions"):
            print(f"\nSuggestions: {validation['suggestions']}")

        print("✓ Content validation successful")

    except Exception as e:
        print(f"✗ Content validation failed: {str(e)}")


async def streaming_example(provider: AnthropicLLMProvider):
    """Demonstrate streaming text generation."""
    print("\n" + "=" * 60)
    print("STREAMING COMPLETION EXAMPLE")
    print("=" * 60)

    try:
        prompt = "Write a short story about a robot learning to understand human emotions."

        print(f"Prompt: {prompt}")
        print("\nStreaming response:")
        print("-" * 40)

        # Collect chunks for complete response
        complete_response = ""

        async for chunk in provider.generate_streaming_completion(
            prompt=prompt, task_type=LLMTask.GENERAL_COMPLETION, temperature=0.8
        ):
            print(chunk, end="", flush=True)
            complete_response += chunk

        print("\n" + "-" * 40)
        print(f"Complete response length: {len(complete_response)} characters")
        print("✓ Streaming completion successful")

    except Exception as e:
        print(f"✗ Streaming completion failed: {str(e)}")


async def provider_info_example(provider: AnthropicLLMProvider):
    """Demonstrate provider information and health check."""
    print("\n" + "=" * 60)
    print("PROVIDER INFORMATION EXAMPLE")
    print("=" * 60)

    try:
        # Get provider information
        info = provider.get_provider_info()

        print("Provider Information:")
        print(f"  Name: {info['name']}")
        print(f"  Provider: {info['provider']}")
        print(f"  Model: {info['model']}")
        print(f"  Type: {info['type']}")
        print(f"  Connected: {info['connected']}")

        print(f"\n  Configuration:")
        for key, value in info["config"].items():
            print(f"    {key}: {value}")

        print(f"\n  Features: {info['features']}")
        print(f"  Supported tasks: {info['supported_tasks']}")

        # Health check
        print("\nPerforming health check...")
        health = await provider.health_check()

        print("Health Status:")
        for key, value in health.items():
            if key == "timestamp" and value:
                import datetime

                dt = datetime.datetime.fromtimestamp(value)
                print(f"  {key}: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                print(f"  {key}: {value}")

        print("✓ Provider information retrieved successfully")

    except Exception as e:
        print(f"✗ Provider information failed: {str(e)}")


async def main():
    """Main example function."""
    print("Anthropic Claude LLM Provider Example")
    print("=" * 60)

    # Configuration - you can also load this from config files
    config = {
        "api_key": os.getenv("ANTHROPIC_API_KEY"),  # Set via environment variable
        "model_name": "claude-3-5-sonnet-20241022",
        "temperature": 0.7,
        "max_tokens": 4096,
        "timeout": 30,
        "top_p": 0.9,
    }

    # Check if API key is available
    if not config["api_key"]:
        print("⚠ ANTHROPIC_API_KEY environment variable not set.")
        print("Set your API key with: export ANTHROPIC_API_KEY='your-api-key-here'")
        print("Or pass it directly in the config dictionary.")
        print("\nRunning provider info example only (no API calls)...")

        # Create provider without API key for basic testing
        config["api_key"] = "dummy-key-for-testing"

        try:
            provider = AnthropicLLMProvider(config)
            await provider_info_example(provider)
        except Exception as e:
            print(f"Error: {str(e)}")

        return

    try:
        # Initialize provider
        print("Initializing Anthropic provider...")
        provider = AnthropicLLMProvider(config)

        # Connect to the API
        print("Connecting to Anthropic API...")
        await provider.connect()
        print("✓ Connected successfully")

        # Run all examples
        await basic_completion_example(provider)
        await chat_completion_example(provider)
        await knowledge_extraction_example(provider)
        await relationship_detection_example(provider)
        await query_parsing_example(provider)
        await content_validation_example(provider)
        await streaming_example(provider)
        await provider_info_example(provider)

        # Disconnect
        await provider.disconnect()
        print("\n✓ All examples completed successfully!")

    except LLMError as e:
        print(f"LLM Error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
