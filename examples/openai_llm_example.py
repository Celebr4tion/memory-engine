#!/usr/bin/env python3
"""
Example demonstrating the OpenAI LLM provider usage.

This example shows how to use the OpenAI GPT LLM provider for various tasks
including knowledge extraction, relationship detection, and natural language query parsing.
"""

import asyncio
import os
import json
from typing import Dict, Any

from memory_core.llm.providers.openai import OpenAILLMProvider
from memory_core.llm.interfaces.llm_provider_interface import LLMTask, Message, MessageRole


async def main():
    """Main example function demonstrating OpenAI LLM provider usage."""

    # Configuration for OpenAI LLM provider
    # Note: Set OPENAI_API_KEY environment variable with your API key
    config: Dict[str, Any] = {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "model_name": "gpt-4o-mini",  # or 'gpt-4', 'gpt-3.5-turbo'
        "temperature": 0.7,
        "max_tokens": 4096,
        "timeout": 30,
        "organization": None,  # Optional organization ID
        "base_url": None,  # Optional custom endpoint
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    }

    if not config["api_key"]:
        print("❌ Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        return

    try:
        # Initialize the OpenAI LLM provider
        print("🔧 Initializing OpenAI LLM Provider...")
        provider = OpenAILLMProvider(config)

        # Connect to the provider
        print("🔌 Connecting to OpenAI API...")
        await provider.connect()

        # Display provider information
        print("📋 Provider Information:")
        info = provider.get_provider_info()
        print(f"   Provider: {info['provider']}")
        print(f"   Model: {info['model']}")
        print(f"   Connected: {info['connected']}")
        print(f"   Features: {', '.join(info['features'])}")

        # Example 1: Basic completion
        print("\n1️⃣ Basic Completion Example:")
        print("-" * 40)

        response = await provider.generate_completion(
            "Explain the concept of machine learning in simple terms.", LLMTask.GENERAL_COMPLETION
        )
        print(f"Response: {response.content[:200]}...")
        print(f"Model: {response.model}")
        print(f"Usage: {response.usage}")

        # Example 2: Knowledge extraction
        print("\n2️⃣ Knowledge Extraction Example:")
        print("-" * 40)

        sample_text = """
        Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines 
        that can work and react like humans. Machine learning is a subset of AI that provides systems the 
        ability to automatically learn and improve from experience without being explicitly programmed. 
        Deep learning is a subset of machine learning that uses neural networks with three or more layers.
        """

        knowledge_units = await provider.extract_knowledge_units(sample_text)
        print(f"Extracted {len(knowledge_units)} knowledge units:")
        for i, unit in enumerate(knowledge_units[:2], 1):  # Show first 2 units
            print(f"   Unit {i}: {unit.get('content', 'N/A')}")
            print(f"   Tags: {unit.get('tags', [])}")
            print(f"   Domain: {unit.get('metadata', {}).get('domain', 'N/A')}")
            print()

        # Example 3: Relationship detection
        print("\n3️⃣ Relationship Detection Example:")
        print("-" * 40)

        entities = [
            "Artificial Intelligence",
            "Machine Learning",
            "Deep Learning",
            "Neural Networks",
        ]
        relationships = await provider.detect_relationships(entities, sample_text)
        print(f"Detected {len(relationships)} relationships:")
        for i, rel in enumerate(relationships[:3], 1):  # Show first 3 relationships
            print(
                f"   Relationship {i}: {rel.get('source')} --[{rel.get('relationship_type')}]--> {rel.get('target')}"
            )
            print(f"   Confidence: {rel.get('confidence', 0):.2f}")
            print()

        # Example 4: Natural language query parsing
        print("\n4️⃣ Natural Language Query Parsing Example:")
        print("-" * 40)

        query = "Find all concepts related to machine learning algorithms"
        parsed_query = await provider.parse_natural_language_query(query)
        print(f"Original query: '{query}'")
        print(f"Parsed intent: {parsed_query.get('intent')}")
        print(f"Entities: {parsed_query.get('entities', [])}")
        print(f"Keywords: {parsed_query.get('semantic_keywords', [])}")
        print(f"Query type: {parsed_query.get('query_type')}")

        # Example 5: Chat completion
        print("\n5️⃣ Chat Completion Example:")
        print("-" * 40)

        messages = [
            Message(
                MessageRole.SYSTEM, "You are a helpful AI assistant specializing in technology."
            ),
            Message(
                MessageRole.USER,
                "What are the main differences between supervised and unsupervised learning?",
            ),
        ]

        chat_response = await provider.generate_chat_completion(
            messages, LLMTask.GENERAL_COMPLETION
        )
        print(f"Chat response: {chat_response.content[:200]}...")

        # Example 6: Content validation
        print("\n6️⃣ Content Validation Example:")
        print("-" * 40)

        content_to_validate = "Machine learning is a type of artificial intelligence."
        criteria = [
            "Contains accurate information",
            "Is grammatically correct",
            "Uses appropriate terminology",
        ]

        validation_result = await provider.validate_content(content_to_validate, criteria)
        print(f"Content: '{content_to_validate}'")
        print(f"Overall valid: {validation_result.get('valid')}")
        print(f"Score: {validation_result.get('overall_score', 0):.2f}")
        print("Criteria results:")
        for criterion, result in validation_result.get("criteria_results", {}).items():
            print(f"   ✅ {criterion}: {result}")

        # Example 7: Health check
        print("\n7️⃣ Health Check Example:")
        print("-" * 40)

        health = await provider.health_check()
        print(f"Provider health: {'✅ Healthy' if health.get('test_passed') else '❌ Unhealthy'}")
        print(f"Response time: {health.get('response_time', 0):.2f}s")
        if health.get("error"):
            print(f"Error: {health['error']}")

        # Disconnect
        await provider.disconnect()
        print("\n🔌 Disconnected from OpenAI API")
        print("✅ Example completed successfully!")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("🤖 OpenAI LLM Provider Example")
    print("=" * 50)
    asyncio.run(main())
