"""
Anthropic Claude LLM provider for the Memory Engine.

This module provides the AnthropicLLMProvider class which implements the LLMProviderInterface
for Anthropic's Claude API. The provider supports various language model tasks including
knowledge extraction, relationship detection, natural language query processing, and more.

The provider supports Claude models including:
- claude-3-5-sonnet-20241022 (default)
- claude-3-haiku-20240307
- claude-3-opus-20240229
- claude-3-sonnet-20240229

Configuration:
    The provider requires an Anthropic API key and supports the following configuration options:
    - api_key: Anthropic API key (required)
    - model_name: Claude model name (default: 'claude-3-5-sonnet-20241022')
    - temperature: Sampling temperature (default: 0.7)
    - max_tokens: Maximum output tokens (default: 4096)
    - timeout: Request timeout in seconds (default: 30)
    - top_p: Top-p sampling (default: 0.9)
    - top_k: Top-k sampling (optional)

Example:
    >>> from memory_core.llm.providers.anthropic import AnthropicLLMProvider
    >>> from memory_core.llm.interfaces.llm_provider_interface import LLMTask
    >>> 
    >>> config = {
    ...     'api_key': 'your-anthropic-api-key',
    ...     'model_name': 'claude-3-5-sonnet-20241022',
    ...     'temperature': 0.7,
    ...     'max_tokens': 4096
    ... }
    >>> 
    >>> provider = AnthropicLLMProvider(config)
    >>> await provider.connect()
    >>> 
    >>> response = await provider.generate_completion(
    ...     "Explain artificial intelligence in simple terms.",
    ...     LLMTask.GENERAL_COMPLETION
    ... )
    >>> print(response.content)

Dependencies:
    This module requires the 'anthropic' package to be installed:
    pip install anthropic
"""

from .anthropic_provider import AnthropicLLMProvider

__all__ = ['AnthropicLLMProvider']

# Version info
__version__ = '1.0.0'
__author__ = 'Memory Engine Team'

# Provider metadata
PROVIDER_NAME = 'anthropic'
PROVIDER_CLASS = AnthropicLLMProvider
SUPPORTED_MODELS = [
    'claude-3-5-sonnet-20241022',
    'claude-3-haiku-20240307',
    'claude-3-opus-20240229',
    'claude-3-sonnet-20240229'
]
DEFAULT_MODEL = 'claude-3-5-sonnet-20241022'

# Required configuration keys
REQUIRED_CONFIG = ['api_key']

# Optional configuration keys with defaults
OPTIONAL_CONFIG = {
    'model_name': DEFAULT_MODEL,
    'temperature': 0.7,
    'max_tokens': 4096,
    'timeout': 30,
    'top_p': 0.9,
    'top_k': None
}