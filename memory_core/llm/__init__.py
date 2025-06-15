"""
LLM Provider System for Memory Engine

This module provides a comprehensive LLM provider system with support for multiple
backends, fallback chains, and provider orchestration. It enables the Memory Engine
to work with different LLM providers (API-based and local) with graceful degradation
when providers are unavailable.

Features:
- Multiple LLM provider support (Gemini, OpenAI, Anthropic, Ollama, HuggingFace)
- Provider factory system for dynamic instantiation
- LLM manager with fallback chains and circuit breaker pattern
- Comprehensive error handling and health monitoring
- Support for all Memory Engine LLM tasks
"""

from .factory import (
    create_provider,
    create_fallback_chain,
    list_available_providers,
    is_provider_available,
    get_provider_requirements,
    LLMProviderFactory
)

from .manager import (
    LLMManager,
    LLMManagerConfig,
    FallbackStrategy
)

from .interfaces.llm_provider_interface import (
    LLMProviderInterface,
    LLMTask,
    MessageRole,
    Message,
    LLMResponse,
    LLMError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMValidationError
)

__all__ = [
    # Factory functions
    'create_provider',
    'create_fallback_chain',
    'list_available_providers',
    'is_provider_available',
    'get_provider_requirements',
    'LLMProviderFactory',
    
    # Manager classes
    'LLMManager',
    'LLMManagerConfig',
    'FallbackStrategy',
    
    # Interfaces and types
    'LLMProviderInterface',
    'LLMTask',
    'MessageRole',
    'Message',
    'LLMResponse',
    'LLMError',
    'LLMConnectionError',
    'LLMRateLimitError',
    'LLMValidationError'
]