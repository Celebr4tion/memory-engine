"""
LLM Manager with fallback support and provider orchestration.

This module provides the LLMManager class that handles multiple LLM providers,
implements fallback chains, and provides a unified interface for LLM operations
with graceful degradation when providers are unavailable.
"""
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
from dataclasses import dataclass
from enum import Enum

from memory_core.llm.interfaces.llm_provider_interface import (
    LLMProviderInterface,
    LLMTask,
    Message,
    LLMResponse,
    LLMError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMValidationError
)
from memory_core.llm.factory import create_provider, create_fallback_chain


class FallbackStrategy(Enum):
    """Strategies for handling provider failures."""
    FAIL_FAST = "fail_fast"  # Stop on first failure
    GRACEFUL_DEGRADATION = "graceful_degradation"  # Try all providers in chain
    BEST_EFFORT = "best_effort"  # Return partial results if some providers fail


@dataclass
class ProviderAttempt:
    """Record of a provider attempt."""
    provider_name: str
    success: bool
    error: Optional[str] = None
    response_time: Optional[float] = None
    response: Optional[LLMResponse] = None


@dataclass
class LLMManagerConfig:
    """Configuration for LLM Manager."""
    primary_provider: str = "gemini"
    fallback_providers: Optional[List[str]] = None
    fallback_strategy: FallbackStrategy = FallbackStrategy.GRACEFUL_DEGRADATION
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0
    health_check_interval: int = 300  # seconds
    circuit_breaker_threshold: int = 5  # failures before circuit opens
    circuit_breaker_timeout: int = 60  # seconds before circuit resets


class LLMManager:
    """
    LLM Manager with fallback support and provider orchestration.
    
    This class manages multiple LLM providers, implements fallback chains,
    and provides circuit breaker functionality for resilient LLM operations.
    """
    
    def __init__(self, config: Optional[LLMManagerConfig] = None):
        """
        Initialize the LLM Manager.
        
        Args:
            config: Manager configuration. If None, uses default configuration.
        """
        self.config = config or LLMManagerConfig()
        self.logger = logging.getLogger(__name__)
        
        # Provider management
        self.providers: List[LLMProviderInterface] = []
        self.provider_status: Dict[str, Dict[str, Any]] = {}
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.performance_metrics: Dict[str, List[float]] = {}
        self.failure_counts: Dict[str, int] = {}
        
        # Initialize providers
        self._initialize_providers()
        
        # Start health check task
        self._health_check_task = None
        
    def _initialize_providers(self):
        """Initialize LLM providers based on configuration."""
        try:
            self.providers = create_fallback_chain(
                primary_provider=self.config.primary_provider,
                fallback_providers=self.config.fallback_providers
            )
            
            # Initialize provider status and circuit breakers
            for provider in self.providers:
                provider_name = provider.get_provider_info()['provider']
                self.provider_status[provider_name] = {
                    'available': True,
                    'last_success': None,
                    'last_failure': None,
                    'consecutive_failures': 0
                }
                self.circuit_breakers[provider_name] = {
                    'state': 'closed',  # closed, open, half_open
                    'failure_count': 0,
                    'last_failure_time': None,
                    'last_success_time': None
                }
                self.performance_metrics[provider_name] = []
                self.failure_counts[provider_name] = 0
            
            self.logger.info(f"Initialized LLM Manager with {len(self.providers)} providers")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM providers: {e}")
            raise
    
    async def connect(self) -> bool:
        """
        Connect to all available providers.
        
        Returns:
            True if at least one provider connected successfully
        """
        connected_count = 0
        
        for provider in self.providers:
            try:
                success = await provider.connect()
                provider_name = provider.get_provider_info()['provider']
                
                if success:
                    connected_count += 1
                    self.provider_status[provider_name]['available'] = True
                    self.logger.info(f"Connected to {provider_name} provider")
                else:
                    self.provider_status[provider_name]['available'] = False
                    self.logger.warning(f"Failed to connect to {provider_name} provider")
                    
            except Exception as e:
                provider_name = provider.get_provider_info()['provider']
                self.provider_status[provider_name]['available'] = False
                self.logger.error(f"Error connecting to {provider_name}: {e}")
        
        if connected_count > 0:
            self.logger.info(f"LLM Manager connected with {connected_count}/{len(self.providers)} providers")
            # Start health check task
            if self._health_check_task is None:
                self._health_check_task = asyncio.create_task(self._health_check_loop())
            return True
        else:
            self.logger.error("No LLM providers could be connected")
            return False
    
    async def disconnect(self) -> bool:
        """
        Disconnect from all providers.
        
        Returns:
            True if all providers disconnected successfully
        """
        # Stop health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            self._health_check_task = None
        
        success_count = 0
        
        for provider in self.providers:
            try:
                success = await provider.disconnect()
                if success:
                    success_count += 1
            except Exception as e:
                provider_name = provider.get_provider_info()['provider']
                self.logger.error(f"Error disconnecting from {provider_name}: {e}")
        
        self.logger.info(f"Disconnected from {success_count}/{len(self.providers)} providers")
        return success_count == len(self.providers)
    
    def _is_circuit_breaker_open(self, provider_name: str) -> bool:
        """Check if circuit breaker is open for a provider."""
        breaker = self.circuit_breakers[provider_name]
        
        if breaker['state'] == 'open':
            # Check if enough time has passed to try again
            if (time.time() - breaker['last_failure_time']) > self.config.circuit_breaker_timeout:
                breaker['state'] = 'half_open'
                self.logger.info(f"Circuit breaker for {provider_name} moved to half-open state")
                return False
            return True
        
        return False
    
    def _update_circuit_breaker(self, provider_name: str, success: bool):
        """Update circuit breaker state based on operation result."""
        breaker = self.circuit_breakers[provider_name]
        
        if success:
            breaker['failure_count'] = 0
            breaker['last_success_time'] = time.time()
            if breaker['state'] in ['open', 'half_open']:
                breaker['state'] = 'closed'
                self.logger.info(f"Circuit breaker for {provider_name} closed")
        else:
            breaker['failure_count'] += 1
            breaker['last_failure_time'] = time.time()
            
            if breaker['failure_count'] >= self.config.circuit_breaker_threshold:
                breaker['state'] = 'open'
                self.logger.warning(f"Circuit breaker for {provider_name} opened after {breaker['failure_count']} failures")
    
    async def _try_provider(self, provider: LLMProviderInterface, operation: str, *args, **kwargs) -> ProviderAttempt:
        """
        Try an operation with a specific provider.
        
        Args:
            provider: Provider to use
            operation: Operation name (method name on provider)
            *args: Operation arguments
            **kwargs: Operation keyword arguments
            
        Returns:
            ProviderAttempt with result information
        """
        provider_info = provider.get_provider_info()
        provider_name = provider_info['provider']
        
        # Check circuit breaker
        if self._is_circuit_breaker_open(provider_name):
            return ProviderAttempt(
                provider_name=provider_name,
                success=False,
                error="Circuit breaker open"
            )
        
        # Check if provider is available
        if not self.provider_status[provider_name]['available']:
            return ProviderAttempt(
                provider_name=provider_name,
                success=False,
                error="Provider marked as unavailable"
            )
        
        start_time = time.time()
        
        try:
            # Get the method to call
            method = getattr(provider, operation)
            
            # Execute with timeout
            response = await asyncio.wait_for(
                method(*args, **kwargs),
                timeout=self.config.timeout
            )
            
            response_time = time.time() - start_time
            
            # Update metrics
            self.performance_metrics[provider_name].append(response_time)
            # Keep only last 100 measurements
            if len(self.performance_metrics[provider_name]) > 100:
                self.performance_metrics[provider_name].pop(0)
            
            # Update circuit breaker
            self._update_circuit_breaker(provider_name, True)
            
            # Update provider status
            self.provider_status[provider_name]['last_success'] = time.time()
            self.provider_status[provider_name]['consecutive_failures'] = 0
            
            self.logger.debug(f"Operation {operation} succeeded with {provider_name} in {response_time:.2f}s")
            
            return ProviderAttempt(
                provider_name=provider_name,
                success=True,
                response_time=response_time,
                response=response
            )
            
        except asyncio.TimeoutError:
            error_msg = f"Operation {operation} timed out after {self.config.timeout}s"
            self.logger.warning(f"{provider_name}: {error_msg}")
            self._handle_provider_failure(provider_name, error_msg)
            return ProviderAttempt(provider_name=provider_name, success=False, error=error_msg)
            
        except (LLMConnectionError, LLMRateLimitError) as e:
            error_msg = f"Provider error: {str(e)}"
            self.logger.warning(f"{provider_name}: {error_msg}")
            self._handle_provider_failure(provider_name, error_msg)
            return ProviderAttempt(provider_name=provider_name, success=False, error=error_msg)
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self.logger.error(f"{provider_name}: {error_msg}")
            self._handle_provider_failure(provider_name, error_msg)
            return ProviderAttempt(provider_name=provider_name, success=False, error=error_msg)
    
    def _handle_provider_failure(self, provider_name: str, error_msg: str):
        """Handle provider failure by updating status and metrics."""
        self.failure_counts[provider_name] += 1
        self.provider_status[provider_name]['last_failure'] = time.time()
        self.provider_status[provider_name]['consecutive_failures'] += 1
        
        # Update circuit breaker
        self._update_circuit_breaker(provider_name, False)
        
        # Mark provider as unavailable after too many consecutive failures
        if self.provider_status[provider_name]['consecutive_failures'] >= self.config.circuit_breaker_threshold:
            self.provider_status[provider_name]['available'] = False
            self.logger.warning(f"Provider {provider_name} marked as unavailable after {self.provider_status[provider_name]['consecutive_failures']} consecutive failures")
    
    async def generate_completion(
        self,
        prompt: str,
        task_type: LLMTask = LLMTask.GENERAL_COMPLETION,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a completion using the best available provider.
        
        Args:
            prompt: Input prompt text
            task_type: Type of task being performed
            **kwargs: Additional provider-specific parameters
            
        Returns:
            LLM response with generated content
            
        Raises:
            LLMError: If all providers fail
        """
        attempts = []
        
        for provider in self.providers:
            attempt = await self._try_provider(
                provider, 'generate_completion', prompt, task_type, **kwargs
            )
            attempts.append(attempt)
            
            if attempt.success:
                return attempt.response
            
            # Check fallback strategy
            if self.config.fallback_strategy == FallbackStrategy.FAIL_FAST:
                break
        
        # All providers failed
        error_details = [f"{a.provider_name}: {a.error}" for a in attempts]
        raise LLMError(f"All providers failed for completion generation: {'; '.join(error_details)}")
    
    async def generate_chat_completion(
        self,
        messages: List[Message],
        task_type: LLMTask = LLMTask.GENERAL_COMPLETION,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a chat completion using the best available provider.
        
        Args:
            messages: List of conversation messages
            task_type: Type of task being performed
            **kwargs: Additional provider-specific parameters
            
        Returns:
            LLM response with generated content
            
        Raises:
            LLMError: If all providers fail
        """
        attempts = []
        
        for provider in self.providers:
            attempt = await self._try_provider(
                provider, 'generate_chat_completion', messages, task_type, **kwargs
            )
            attempts.append(attempt)
            
            if attempt.success:
                return attempt.response
            
            # Check fallback strategy
            if self.config.fallback_strategy == FallbackStrategy.FAIL_FAST:
                break
        
        # All providers failed
        error_details = [f"{a.provider_name}: {a.error}" for a in attempts]
        raise LLMError(f"All providers failed for chat completion generation: {'; '.join(error_details)}")
    
    async def extract_knowledge_units(
        self,
        text: str,
        source_info: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract knowledge units using the best available provider.
        
        Args:
            text: Input text to analyze
            source_info: Optional source information
            
        Returns:
            List of knowledge unit dictionaries
            
        Raises:
            LLMError: If all providers fail
        """
        attempts = []
        
        for provider in self.providers:
            attempt = await self._try_provider(
                provider, 'extract_knowledge_units', text, source_info
            )
            attempts.append(attempt)
            
            if attempt.success:
                return attempt.response
            
            # Check fallback strategy
            if self.config.fallback_strategy == FallbackStrategy.FAIL_FAST:
                break
        
        # All providers failed - return empty list for graceful degradation
        if self.config.fallback_strategy == FallbackStrategy.BEST_EFFORT:
            self.logger.warning("All providers failed for knowledge extraction, returning empty list")
            return []
        
        error_details = [f"{a.provider_name}: {a.error}" for a in attempts]
        raise LLMError(f"All providers failed for knowledge extraction: {'; '.join(error_details)}")
    
    async def detect_relationships(
        self,
        entities: List[str],
        context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect relationships using the best available provider.
        
        Args:
            entities: List of entity names/descriptions
            context: Optional context text
            
        Returns:
            List of relationship dictionaries
            
        Raises:
            LLMError: If all providers fail
        """
        attempts = []
        
        for provider in self.providers:
            attempt = await self._try_provider(
                provider, 'detect_relationships', entities, context
            )
            attempts.append(attempt)
            
            if attempt.success:
                return attempt.response
            
            # Check fallback strategy
            if self.config.fallback_strategy == FallbackStrategy.FAIL_FAST:
                break
        
        # All providers failed - return empty list for graceful degradation
        if self.config.fallback_strategy == FallbackStrategy.BEST_EFFORT:
            self.logger.warning("All providers failed for relationship detection, returning empty list")
            return []
        
        error_details = [f"{a.provider_name}: {a.error}" for a in attempts]
        raise LLMError(f"All providers failed for relationship detection: {'; '.join(error_details)}")
    
    async def parse_natural_language_query(
        self,
        query: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Parse natural language query using the best available provider.
        
        Args:
            query: Natural language query
            context: Optional context
            
        Returns:
            Parsed query structure
            
        Raises:
            LLMError: If all providers fail
        """
        attempts = []
        
        for provider in self.providers:
            attempt = await self._try_provider(
                provider, 'parse_natural_language_query', query, context
            )
            attempts.append(attempt)
            
            if attempt.success:
                return attempt.response
            
            # Check fallback strategy
            if self.config.fallback_strategy == FallbackStrategy.FAIL_FAST:
                break
        
        # All providers failed
        error_details = [f"{a.provider_name}: {a.error}" for a in attempts]
        raise LLMError(f"All providers failed for query parsing: {'; '.join(error_details)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all providers.
        
        Returns:
            Overall health status
        """
        provider_health = {}
        overall_healthy = False
        
        for provider in self.providers:
            provider_name = provider.get_provider_info()['provider']
            try:
                health = await provider.health_check()
                provider_health[provider_name] = health
                
                if health.get('test_passed', False):
                    overall_healthy = True
                    
            except Exception as e:
                provider_health[provider_name] = {
                    'provider': provider_name,
                    'connected': False,
                    'test_passed': False,
                    'error': str(e)
                }
        
        return {
            'overall_healthy': overall_healthy,
            'providers': provider_health,
            'manager_config': {
                'primary_provider': self.config.primary_provider,
                'fallback_strategy': self.config.fallback_strategy.value,
                'provider_count': len(self.providers)
            },
            'performance_metrics': self._get_performance_summary()
        }
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all providers."""
        summary = {}
        
        for provider_name, metrics in self.performance_metrics.items():
            if metrics:
                summary[provider_name] = {
                    'avg_response_time': sum(metrics) / len(metrics),
                    'min_response_time': min(metrics),
                    'max_response_time': max(metrics),
                    'total_requests': len(metrics),
                    'failure_count': self.failure_counts[provider_name],
                    'success_rate': 1 - (self.failure_counts[provider_name] / max(len(metrics) + self.failure_counts[provider_name], 1))
                }
            else:
                summary[provider_name] = {
                    'avg_response_time': 0,
                    'total_requests': 0,
                    'failure_count': self.failure_counts[provider_name],
                    'success_rate': 0
                }
        
        return summary
    
    async def _health_check_loop(self):
        """Background task for periodic health checks."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                # Perform health checks
                for provider in self.providers:
                    provider_name = provider.get_provider_info()['provider']
                    
                    try:
                        health = await provider.health_check()
                        
                        if health.get('test_passed', False):
                            # Provider is healthy, mark as available
                            if not self.provider_status[provider_name]['available']:
                                self.provider_status[provider_name]['available'] = True
                                self.logger.info(f"Provider {provider_name} is now available")
                        else:
                            # Provider is not healthy
                            if self.provider_status[provider_name]['available']:
                                self.provider_status[provider_name]['available'] = False
                                self.logger.warning(f"Provider {provider_name} is now unavailable")
                                
                    except Exception as e:
                        self.logger.debug(f"Health check failed for {provider_name}: {e}")
                        if self.provider_status[provider_name]['available']:
                            self.provider_status[provider_name]['available'] = False
                            self.logger.warning(f"Provider {provider_name} marked as unavailable due to health check failure")
                
                self.logger.debug("Health check completed")
                
            except asyncio.CancelledError:
                self.logger.info("Health check task cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
    
    def get_provider_status(self) -> Dict[str, Any]:
        """
        Get current status of all providers.
        
        Returns:
            Dictionary with provider status information
        """
        return {
            'providers': dict(self.provider_status),
            'circuit_breakers': dict(self.circuit_breakers),
            'performance_metrics': self._get_performance_summary()
        }
    
    def get_best_provider(self) -> Optional[LLMProviderInterface]:
        """
        Get the best available provider based on performance and availability.
        
        Returns:
            Best provider instance or None if none available
        """
        available_providers = []
        
        for provider in self.providers:
            provider_name = provider.get_provider_info()['provider']
            
            if (self.provider_status[provider_name]['available'] and 
                not self._is_circuit_breaker_open(provider_name)):
                available_providers.append((provider, provider_name))
        
        if not available_providers:
            return None
        
        # Return provider with best performance (lowest average response time)
        best_provider = None
        best_performance = float('inf')
        
        for provider, provider_name in available_providers:
            metrics = self.performance_metrics[provider_name]
            if metrics:
                avg_time = sum(metrics) / len(metrics)
                if avg_time < best_performance:
                    best_performance = avg_time
                    best_provider = provider
            else:
                # No metrics yet, consider this provider
                if best_provider is None:
                    best_provider = provider
        
        return best_provider or available_providers[0][0]