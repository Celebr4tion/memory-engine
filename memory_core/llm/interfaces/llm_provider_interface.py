"""
Abstract interface for Large Language Model (LLM) providers.

This module defines the common interface that all LLM providers must implement,
enabling the Memory Engine to work with different LLM backends (API-based and local).
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
import logging


class LLMTask(Enum):
    """Different types of tasks that LLMs can perform in the Memory Engine."""
    KNOWLEDGE_EXTRACTION = "knowledge_extraction"
    RELATIONSHIP_DETECTION = "relationship_detection"
    NATURAL_LANGUAGE_QUERY = "natural_language_query"
    TEXT_CLASSIFICATION = "text_classification"
    SUMMARIZATION = "summarization"
    CONTENT_VALIDATION = "content_validation"
    GENERAL_COMPLETION = "general_completion"


class MessageRole(Enum):
    """Roles for conversation messages."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Message:
    """A message in a conversation with an LLM."""
    role: MessageRole
    content: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str
    metadata: Dict[str, Any]
    usage: Optional[Dict[str, int]] = None  # Token usage info
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    confidence: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            'content': self.content,
            'metadata': self.metadata,
            'usage': self.usage,
            'model': self.model,
            'finish_reason': self.finish_reason,
            'confidence': self.confidence
        }


class LLMError(Exception):
    """Base exception for LLM provider errors."""
    pass


class LLMConnectionError(LLMError):
    """Raised when connection to LLM provider fails."""
    pass


class LLMRateLimitError(LLMError):
    """Raised when rate limit is exceeded."""
    pass


class LLMValidationError(LLMError):
    """Raised when input validation fails."""
    pass


class LLMProviderInterface(ABC):
    """
    Abstract interface for LLM providers.
    
    This interface standardizes how the Memory Engine interacts with different
    LLM backends, whether they are API-based (OpenAI, Anthropic, Gemini) or
    local (Ollama, Hugging Face Transformers).
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LLM provider.
        
        Args:
            config: Provider-specific configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._is_connected = False
        
        # Extract common configuration
        self.model_name = config.get('model_name', self.get_default_model())
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 4096)
        self.timeout = config.get('timeout', 30)
    
    @property
    def is_connected(self) -> bool:
        """Check if the provider is connected and ready."""
        return self._is_connected
    
    @abstractmethod
    def get_default_model(self) -> str:
        """
        Get the default model name for this provider.
        
        Returns:
            Default model name
        """
        pass
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to the LLM provider.
        
        Returns:
            True if connection successful, False otherwise
            
        Raises:
            LLMConnectionError: If connection fails
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Disconnect from the LLM provider.
        
        Returns:
            True if disconnection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def generate_completion(
        self,
        prompt: str,
        task_type: LLMTask = LLMTask.GENERAL_COMPLETION,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a completion for a single prompt.
        
        Args:
            prompt: Input prompt text
            task_type: Type of task being performed
            **kwargs: Additional provider-specific parameters
            
        Returns:
            LLM response with generated content
            
        Raises:
            LLMError: If generation fails
        """
        pass
    
    @abstractmethod
    async def generate_chat_completion(
        self,
        messages: List[Message],
        task_type: LLMTask = LLMTask.GENERAL_COMPLETION,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a completion for a conversation.
        
        Args:
            messages: List of conversation messages
            task_type: Type of task being performed
            **kwargs: Additional provider-specific parameters
            
        Returns:
            LLM response with generated content
            
        Raises:
            LLMError: If generation fails
        """
        pass
    
    async def generate_streaming_completion(
        self,
        prompt: str,
        task_type: LLMTask = LLMTask.GENERAL_COMPLETION,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming completion for a prompt.
        
        Args:
            prompt: Input prompt text
            task_type: Type of task being performed
            **kwargs: Additional provider-specific parameters
            
        Yields:
            Chunks of generated text
            
        Raises:
            LLMError: If generation fails
        """
        # Default implementation: return full response as single chunk
        response = await self.generate_completion(prompt, task_type, **kwargs)
        yield response.content
    
    @abstractmethod
    async def extract_knowledge_units(
        self,
        text: str,
        source_info: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract structured knowledge units from text.
        
        Args:
            text: Input text to analyze
            source_info: Optional source information
            
        Returns:
            List of knowledge unit dictionaries
            
        Raises:
            LLMError: If extraction fails
        """
        pass
    
    @abstractmethod
    async def detect_relationships(
        self,
        entities: List[str],
        context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect relationships between entities.
        
        Args:
            entities: List of entity names/descriptions
            context: Optional context text
            
        Returns:
            List of relationship dictionaries
            
        Raises:
            LLMError: If relationship detection fails
        """
        pass
    
    @abstractmethod
    async def parse_natural_language_query(
        self,
        query: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Parse a natural language query into structured format.
        
        Args:
            query: Natural language query
            context: Optional context
            
        Returns:
            Parsed query structure
            
        Raises:
            LLMError: If parsing fails
        """
        pass
    
    async def classify_text(
        self,
        text: str,
        categories: List[str],
        **kwargs
    ) -> Dict[str, float]:
        """
        Classify text into given categories.
        
        Args:
            text: Text to classify
            categories: List of possible categories
            **kwargs: Additional parameters
            
        Returns:
            Dictionary mapping categories to confidence scores
            
        Raises:
            LLMError: If classification fails
        """
        # Default implementation using general completion
        prompt = f"""Classify the following text into one or more of these categories: {', '.join(categories)}

Text: {text}

Provide confidence scores (0.0-1.0) for each category in JSON format:
{{"category_name": confidence_score, ...}}"""
        
        response = await self.generate_completion(
            prompt, 
            LLMTask.TEXT_CLASSIFICATION,
            **kwargs
        )
        
        try:
            import json
            return json.loads(response.content)
        except json.JSONDecodeError:
            # Fallback: return empty scores
            return {cat: 0.0 for cat in categories}
    
    async def summarize_text(
        self,
        text: str,
        max_length: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate a summary of the given text.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            **kwargs: Additional parameters
            
        Returns:
            Summary text
            
        Raises:
            LLMError: If summarization fails
        """
        # Default implementation using general completion
        length_constraint = f" in {max_length} words or less" if max_length else ""
        prompt = f"Summarize the following text{length_constraint}:\n\n{text}"
        
        response = await self.generate_completion(
            prompt,
            LLMTask.SUMMARIZATION,
            **kwargs
        )
        
        return response.content
    
    @abstractmethod
    async def validate_content(
        self,
        content: str,
        criteria: List[str]
    ) -> Dict[str, Any]:
        """
        Validate content against given criteria.
        
        Args:
            content: Content to validate
            criteria: List of validation criteria
            
        Returns:
            Validation results
            
        Raises:
            LLMError: If validation fails
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the LLM provider.
        
        Returns:
            Health status information
        """
        pass
    
    def get_supported_tasks(self) -> List[LLMTask]:
        """
        Get list of tasks supported by this provider.
        
        Returns:
            List of supported LLM tasks
        """
        # Default: support all tasks
        return list(LLMTask)
    
    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about this provider.
        
        Returns:
            Provider information dictionary
        """
        return {
            'name': self.__class__.__name__,
            'model': self.model_name,
            'type': 'api' if hasattr(self, 'api_key') else 'local',
            'supported_tasks': [task.value for task in self.get_supported_tasks()],
            'connected': self.is_connected,
            'config': {
                'temperature': self.temperature,
                'max_tokens': self.max_tokens,
                'timeout': self.timeout
            }
        }
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for given text.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Simple estimation: ~4 characters per token
        return len(text) // 4
    
    async def test_connection(self) -> bool:
        """
        Test if the provider connection is working.
        
        Returns:
            True if connection test succeeds, False otherwise
        """
        try:
            response = await self.generate_completion(
                "Test connection. Respond with 'OK'.",
                LLMTask.GENERAL_COMPLETION
            )
            return 'ok' in response.content.lower()
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False