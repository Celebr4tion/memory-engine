"""
LLM provider factory for creating LLM provider instances.

This module provides a factory function to instantiate the appropriate
LLM provider based on configuration settings, with support for fallback chains.
"""

import logging
import os
from typing import Dict, Any, Optional, List

from memory_core.llm.interfaces.llm_provider_interface import LLMProviderInterface
from memory_core.config import get_config


class LLMProviderFactory:
    """
    Factory class for creating LLM provider instances.

    This factory creates and configures LLM providers based on the
    application configuration, providing a unified way to instantiate
    different LLM implementations with proper fallback support.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._providers = {}
        self._register_providers()

    def _register_providers(self):
        """Register available LLM providers."""
        # Gemini provider
        try:
            from memory_core.llm.providers.gemini.gemini_provider import GeminiLLMProvider

            self._providers["gemini"] = GeminiLLMProvider
        except ImportError:
            self.logger.warning("Gemini LLM provider not available (missing google-genai)")

        # OpenAI provider
        try:
            from memory_core.llm.providers.openai.openai_provider import OpenAILLMProvider

            self._providers["openai"] = OpenAILLMProvider
        except ImportError:
            self.logger.warning("OpenAI LLM provider not available (missing openai)")

        # Anthropic provider
        try:
            from memory_core.llm.providers.anthropic.anthropic_provider import AnthropicLLMProvider

            self._providers["anthropic"] = AnthropicLLMProvider
        except ImportError:
            self.logger.warning("Anthropic LLM provider not available (missing anthropic)")

        # Ollama provider
        try:
            from memory_core.llm.providers.ollama.ollama_provider import OllamaLLMProvider

            self._providers["ollama"] = OllamaLLMProvider
        except ImportError:
            self.logger.warning("Ollama LLM provider not available (missing aiohttp)")

        # HuggingFace provider
        try:
            from memory_core.llm.providers.huggingface.huggingface_provider import (
                HuggingFaceLLMProvider,
            )

            self._providers["huggingface"] = HuggingFaceLLMProvider
        except ImportError:
            self.logger.warning("HuggingFace LLM provider not available (missing transformers)")

    def create_provider(
        self, provider_type: Optional[str] = None, config_override: Optional[Dict[str, Any]] = None
    ) -> LLMProviderInterface:
        """
        Create an LLM provider instance.

        Args:
            provider_type: Type of provider to create ('gemini', 'openai', 'anthropic',
                          'ollama', 'huggingface'). If None, uses configuration setting.
            config_override: Optional configuration override for the provider.

        Returns:
            Configured LLM provider instance

        Raises:
            ValueError: If the provider type is not supported
            ImportError: If the provider dependencies are not available
        """
        config = get_config()

        # Determine provider type
        if provider_type is None:
            provider_type = getattr(config.config.llm, "provider", "gemini")

        if provider_type not in self._providers:
            available_providers = list(self._providers.keys())
            raise ValueError(
                f"Unsupported provider type '{provider_type}'. "
                f"Available providers: {available_providers}"
            )

        provider_class = self._providers[provider_type]

        # Get provider-specific configuration
        provider_config = self._get_provider_config(provider_type, config, config_override)

        # Create provider instance
        try:
            self.logger.info(f"Creating {provider_type} LLM provider")
            return provider_class(provider_config)

        except Exception as e:
            self.logger.error(f"Failed to create {provider_type} LLM provider: {e}")
            raise

    def _get_provider_config(
        self, provider_type: str, config: Any, config_override: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get configuration for a specific provider."""
        provider_config = {}

        # Get base configuration from config object
        llm_config = getattr(config.config, "llm", None)
        if llm_config:
            provider_specific_config = getattr(llm_config, provider_type, None)
            if provider_specific_config:
                # Convert dataclass to dict
                if hasattr(provider_specific_config, "__dict__"):
                    provider_config = dict(provider_specific_config.__dict__)
                else:
                    provider_config = dict(provider_specific_config)

        # Add API keys from environment variables
        provider_config = self._add_api_keys(provider_type, provider_config)

        # Apply override configuration
        if config_override:
            provider_config.update(config_override)

        return provider_config

    def _add_api_keys(self, provider_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Add API keys from environment variables."""
        # API key environment variable mapping
        api_key_mapping = {
            "gemini": "GOOGLE_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "huggingface": "HUGGINGFACE_API_KEY",
        }

        if provider_type in api_key_mapping:
            env_var = api_key_mapping[provider_type]
            api_key = os.getenv(env_var)
            if api_key:
                config["api_key"] = api_key
                self.logger.debug(f"API key loaded from {env_var}")
            elif not config.get("api_key"):
                # Only warn if no API key is configured at all
                if provider_type != "ollama":  # Ollama doesn't need API keys
                    self.logger.warning(
                        f"No API key found for {provider_type} provider. "
                        f"Set {env_var} environment variable."
                    )

        return config

    def create_fallback_chain(
        self,
        primary_provider: str,
        fallback_providers: Optional[List[str]] = None,
        config_override: Optional[Dict[str, Any]] = None,
    ) -> List[LLMProviderInterface]:
        """
        Create a chain of LLM providers for fallback support.

        Args:
            primary_provider: Primary provider to use
            fallback_providers: List of fallback providers (in order of preference)
            config_override: Optional configuration override

        Returns:
            List of configured provider instances
        """
        if fallback_providers is None:
            # Default fallback chain
            all_providers = ["gemini", "openai", "anthropic", "ollama", "huggingface"]
            fallback_providers = [p for p in all_providers if p != primary_provider]

        providers = []

        # Add primary provider
        try:
            primary = self.create_provider(primary_provider, config_override)
            providers.append(primary)
            self.logger.info(f"Primary provider {primary_provider} created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create primary provider {primary_provider}: {e}")

        # Add fallback providers
        for fallback_provider in fallback_providers:
            try:
                if self.is_provider_available(fallback_provider):
                    fallback = self.create_provider(fallback_provider, config_override)
                    providers.append(fallback)
                    self.logger.info(f"Fallback provider {fallback_provider} created successfully")
            except Exception as e:
                self.logger.warning(f"Failed to create fallback provider {fallback_provider}: {e}")

        if not providers:
            raise RuntimeError(
                "No LLM providers could be created. Check your configuration and dependencies."
            )

        self.logger.info(f"Created fallback chain with {len(providers)} providers")
        return providers

    def list_available_providers(self) -> List[str]:
        """
        List all available LLM providers.

        Returns:
            List of provider type names
        """
        return list(self._providers.keys())

    def is_provider_available(self, provider_type: str) -> bool:
        """
        Check if a specific provider is available.

        Args:
            provider_type: Type of provider to check

        Returns:
            True if provider is available, False otherwise
        """
        return provider_type in self._providers

    def get_provider_requirements(self, provider_type: str) -> Dict[str, Any]:
        """
        Get requirements for a specific provider.

        Args:
            provider_type: Type of provider

        Returns:
            Dictionary with provider requirements
        """
        requirements = {
            "gemini": {
                "dependencies": ["google-genai"],
                "api_key": "GOOGLE_API_KEY",
                "type": "api",
            },
            "openai": {"dependencies": ["openai"], "api_key": "OPENAI_API_KEY", "type": "api"},
            "anthropic": {
                "dependencies": ["anthropic"],
                "api_key": "ANTHROPIC_API_KEY",
                "type": "api",
            },
            "ollama": {
                "dependencies": ["aiohttp"],
                "api_key": None,
                "type": "local",
                "server_required": True,
            },
            "huggingface": {
                "dependencies": ["transformers", "torch"],
                "api_key": "HUGGINGFACE_API_KEY",
                "type": "local/api",
            },
        }

        return requirements.get(provider_type, {})


# Global factory instance
_llm_factory = LLMProviderFactory()


def create_provider(
    provider_type: Optional[str] = None, config_override: Optional[Dict[str, Any]] = None
) -> LLMProviderInterface:
    """
    Create an LLM provider instance using the global factory.

    Args:
        provider_type: Type of provider to create. If None, uses configuration setting.
        config_override: Optional configuration override for the provider.

    Returns:
        Configured LLM provider instance

    Raises:
        ValueError: If the provider type is not supported
        ImportError: If the provider dependencies are not available
    """
    return _llm_factory.create_provider(provider_type, config_override)


def create_fallback_chain(
    primary_provider: str,
    fallback_providers: Optional[List[str]] = None,
    config_override: Optional[Dict[str, Any]] = None,
) -> List[LLMProviderInterface]:
    """
    Create a chain of LLM providers for fallback support.

    Args:
        primary_provider: Primary provider to use
        fallback_providers: List of fallback providers (in order of preference)
        config_override: Optional configuration override

    Returns:
        List of configured provider instances
    """
    return _llm_factory.create_fallback_chain(primary_provider, fallback_providers, config_override)


def list_available_providers() -> List[str]:
    """
    List all available LLM providers.

    Returns:
        List of provider type names
    """
    return _llm_factory.list_available_providers()


def is_provider_available(provider_type: str) -> bool:
    """
    Check if a specific provider is available.

    Args:
        provider_type: Type of provider to check

    Returns:
        True if provider is available, False otherwise
    """
    return _llm_factory.is_provider_available(provider_type)


def get_provider_requirements(provider_type: str) -> Dict[str, Any]:
    """
    Get requirements for a specific provider.

    Args:
        provider_type: Type of provider

    Returns:
        Dictionary with provider requirements
    """
    return _llm_factory.get_provider_requirements(provider_type)
