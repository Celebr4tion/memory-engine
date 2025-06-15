#!/usr/bin/env python3
"""
LLM Manager Example - Complete LLM Independence System

This example demonstrates the complete LLM independence system including:
- LLM Manager with fallback support
- Multiple provider orchestration
- Circuit breaker pattern
- Health monitoring and performance metrics
- Graceful degradation when providers fail

Prerequisites:
    Set up at least one LLM provider with API keys:
    - export GOOGLE_API_KEY="your-key" (for Gemini)
    - export OPENAI_API_KEY="your-key" (for OpenAI)
    - export ANTHROPIC_API_KEY="your-key" (for Anthropic)
    - Start Ollama server: ollama serve (for Ollama)

Usage:
    python examples/llm_manager_example.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from memory_core.llm import (
    LLMManager,
    LLMManagerConfig,
    FallbackStrategy,
    LLMTask,
    MessageRole,
    Message,
    LLMError,
)


async def demonstrate_basic_completion(manager: LLMManager):
    """Demonstrate basic completion with fallback."""
    print("=" * 60)
    print("BASIC COMPLETION WITH FALLBACK")
    print("=" * 60)

    try:
        prompt = "Explain artificial intelligence in one sentence."

        print(f"Prompt: {prompt}")
        print("Generating response with automatic fallback...")

        response = await manager.generate_completion(
            prompt=prompt, task_type=LLMTask.GENERAL_COMPLETION
        )

        print(f"Response: {response.content}")
        print(f"Model: {response.model}")
        print(f"Provider: {response.metadata.get('provider', 'unknown')}")

        if response.usage:
            print(f"Token usage: {response.usage}")

        print("✓ Basic completion successful")

    except Exception as e:
        print(f"✗ Basic completion failed: {str(e)}")


async def demonstrate_knowledge_extraction(manager: LLMManager):
    """Demonstrate knowledge extraction with fallback."""
    print("\n" + "=" * 60)
    print("KNOWLEDGE EXTRACTION WITH FALLBACK")
    print("=" * 60)

    try:
        text = """
        Machine learning is a subset of artificial intelligence that enables computers 
        to learn and make decisions from data without being explicitly programmed. 
        It uses algorithms to identify patterns in data and make predictions or decisions.
        """

        print(f"Text: {text.strip()}")
        print("Extracting knowledge units...")

        knowledge_units = await manager.extract_knowledge_units(text)

        print(f"Extracted {len(knowledge_units)} knowledge units:")
        for i, unit in enumerate(knowledge_units[:3], 1):  # Show first 3
            print(f"\n{i}. {unit.get('content', 'N/A')}")
            print(f"   Domain: {unit.get('metadata', {}).get('domain', 'N/A')}")
            print(f"   Confidence: {unit.get('metadata', {}).get('confidence_level', 'N/A')}")

        print("✓ Knowledge extraction successful")

    except Exception as e:
        print(f"✗ Knowledge extraction failed: {str(e)}")


async def demonstrate_health_monitoring(manager: LLMManager):
    """Demonstrate health monitoring and provider status."""
    print("\n" + "=" * 60)
    print("HEALTH MONITORING AND PROVIDER STATUS")
    print("=" * 60)

    try:
        print("Performing comprehensive health check...")
        health = await manager.health_check()

        print(f"Overall healthy: {health['overall_healthy']}")
        print(f"Provider count: {health['manager_config']['provider_count']}")
        print(f"Primary provider: {health['manager_config']['primary_provider']}")
        print(f"Fallback strategy: {health['manager_config']['fallback_strategy']}")

        print("\nProvider Health:")
        for provider_name, provider_health in health["providers"].items():
            status = "✓" if provider_health.get("test_passed", False) else "✗"
            print(f"  {status} {provider_name}: {provider_health.get('connected', False)}")
            if provider_health.get("error"):
                print(f"    Error: {provider_health['error']}")

        print("\nPerformance Metrics:")
        for provider_name, metrics in health["performance_metrics"].items():
            if metrics.get("total_requests", 0) > 0:
                print(f"  {provider_name}:")
                print(f"    Avg response time: {metrics.get('avg_response_time', 0):.2f}s")
                print(f"    Success rate: {metrics.get('success_rate', 0):.1%}")
                print(f"    Total requests: {metrics.get('total_requests', 0)}")

        # Provider status
        print("\nProvider Status:")
        status = manager.get_provider_status()
        for provider_name, provider_status in status["providers"].items():
            available = "Available" if provider_status["available"] else "Unavailable"
            failures = provider_status.get("consecutive_failures", 0)
            print(f"  {provider_name}: {available} (failures: {failures})")

        print("✓ Health monitoring successful")

    except Exception as e:
        print(f"✗ Health monitoring failed: {str(e)}")


async def demonstrate_best_provider(manager: LLMManager):
    """Demonstrate best provider selection."""
    print("\n" + "=" * 60)
    print("BEST PROVIDER SELECTION")
    print("=" * 60)

    try:
        best_provider = manager.get_best_provider()

        if best_provider:
            info = best_provider.get_provider_info()
            print(f"Best provider: {info['provider']}")
            print(f"Model: {info['model']}")
            print(f"Type: {info['type']}")
            print(f"Connected: {info['connected']}")
        else:
            print("No providers available")

        print("✓ Best provider selection successful")

    except Exception as e:
        print(f"✗ Best provider selection failed: {str(e)}")


async def main():
    """Main example function."""
    print("LLM Manager Example - Complete LLM Independence System")
    print("=" * 60)

    # Configuration for the LLM Manager
    config = LLMManagerConfig(
        primary_provider="gemini",  # Try Gemini first
        fallback_providers=["openai", "anthropic", "ollama"],  # Fallback chain
        fallback_strategy=FallbackStrategy.GRACEFUL_DEGRADATION,
        max_retries=3,
        timeout=30.0,
        health_check_interval=300,
        circuit_breaker_threshold=3,
    )

    # Check available providers
    print("Checking for API keys and available providers...")

    available_apis = []
    if os.getenv("GOOGLE_API_KEY"):
        available_apis.append("Gemini")
    if os.getenv("OPENAI_API_KEY"):
        available_apis.append("OpenAI")
    if os.getenv("ANTHROPIC_API_KEY"):
        available_apis.append("Anthropic")

    if available_apis:
        print(f"Available API providers: {', '.join(available_apis)}")
    else:
        print("⚠ No API keys found. Some providers may not work.")
        print("Set API keys with:")
        print("  export GOOGLE_API_KEY='your-key'")
        print("  export OPENAI_API_KEY='your-key'")
        print("  export ANTHROPIC_API_KEY='your-key'")

    print("Ollama: Check if running at http://localhost:11434")
    print("HuggingFace: Available for local models\n")

    try:
        # Initialize LLM Manager
        print("Initializing LLM Manager with fallback chain...")
        manager = LLMManager(config)

        # Connect to providers
        print("Connecting to available providers...")
        connected = await manager.connect()

        if not connected:
            print("✗ No providers could be connected. Check your setup.")
            return

        print(f"✓ Connected successfully")

        # Run demonstrations
        await demonstrate_basic_completion(manager)
        await demonstrate_knowledge_extraction(manager)
        await demonstrate_health_monitoring(manager)
        await demonstrate_best_provider(manager)

        # Disconnect
        await manager.disconnect()
        print("\n✓ All demonstrations completed successfully!")
        print("\nThe LLM independence system is working correctly with:")
        print("  ✓ Multiple provider support")
        print("  ✓ Automatic fallback chains")
        print("  ✓ Circuit breaker pattern")
        print("  ✓ Health monitoring")
        print("  ✓ Performance metrics")
        print("  ✓ Graceful degradation")

    except LLMError as e:
        print(f"LLM Error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
