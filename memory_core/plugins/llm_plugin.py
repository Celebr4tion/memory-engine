"""
LLM provider plugin interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, AsyncGenerator
from .plugin_manager import PluginInterface


class LLMPluginInterface(PluginInterface):
    """Interface for LLM provider plugins."""
    
    @property
    @abstractmethod
    def plugin_type(self) -> str:
        """Return 'llm'."""
        return "llm"
    
    @abstractmethod
    async def generate_completion(self, prompt: str, max_tokens: int = 1000, 
                                 temperature: float = 0.7, **kwargs) -> str:
        """Generate text completion."""
        pass
    
    @abstractmethod
    async def generate_chat_completion(self, messages: List[Dict[str, str]], 
                                      max_tokens: int = 1000, 
                                      temperature: float = 0.7, **kwargs) -> str:
        """Generate chat completion."""
        pass
    
    async def generate_streaming_completion(self, prompt: str, max_tokens: int = 1000,
                                          temperature: float = 0.7, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming completion."""
        # Default implementation - return full completion at once
        result = await self.generate_completion(prompt, max_tokens, temperature, **kwargs)
        yield result
    
    async def extract_knowledge_units(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        """Extract knowledge units from text."""
        # Default implementation using completion
        prompt = f"""Extract key knowledge units from the following text. 
        Return as JSON array with objects containing 'content', 'type', and 'confidence'.
        
        Text: {text}"""
        
        response = await self.generate_completion(prompt, max_tokens=2000, temperature=0.3)
        
        # Basic parsing - should be overridden by specific implementations
        try:
            import json
            return json.loads(response)
        except:
            return [{'content': text, 'type': 'general', 'confidence': 0.5}]
    
    async def detect_relationships(self, entities: List[str], 
                                  context: str = "") -> List[Dict[str, Any]]:
        """Detect relationships between entities."""
        # Default implementation
        prompt = f"""Analyze the relationships between these entities: {', '.join(entities)}
        Context: {context}
        
        Return as JSON array with objects containing 'source', 'target', 'type', and 'confidence'."""
        
        response = await self.generate_completion(prompt, max_tokens=1000, temperature=0.3)
        
        try:
            import json
            return json.loads(response)
        except:
            return []
    
    async def parse_natural_language_query(self, query: str, 
                                          context: str = "") -> Dict[str, Any]:
        """Parse natural language query into structured format."""
        prompt = f"""Parse this natural language query into structured format:
        Query: {query}
        Context: {context}
        
        Return JSON with 'intent', 'entities', 'query_type', and 'parameters'."""
        
        response = await self.generate_completion(prompt, max_tokens=500, temperature=0.3)
        
        try:
            import json
            return json.loads(response)
        except:
            return {'intent': 'search', 'entities': [query], 'query_type': 'general', 'parameters': {}}
    
    async def validate_content(self, content: str, 
                              criteria: List[str]) -> Dict[str, Any]:
        """Validate content against criteria."""
        criteria_text = '\n'.join(f"- {criterion}" for criterion in criteria)
        
        prompt = f"""Validate this content against the criteria:
        
        Content: {content}
        
        Criteria:
        {criteria_text}
        
        Return JSON with 'valid' (boolean), 'score' (0-1), and 'criteria_results' (dict)."""
        
        response = await self.generate_completion(prompt, max_tokens=500, temperature=0.1)
        
        try:
            import json
            return json.loads(response)
        except:
            return {'valid': True, 'score': 0.5, 'criteria_results': {}}
    
    async def health_check(self) -> Dict[str, Any]:
        """Check LLM provider health."""
        try:
            response = await self.generate_completion("Hello", max_tokens=5)
            return {
                'healthy': True,
                'response_time': 0.0,  # Should be measured by implementation
                'provider': self.name
            }
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'provider': self.name
            }
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            'provider': self.name,
            'model': getattr(self, 'model_name', 'unknown'),
            'version': self.version
        }


class LLMPlugin(LLMPluginInterface):
    """Base class for LLM plugins."""
    
    def __init__(self):
        self._config = {}
        self._api_key = None
        self._model_name = None
        self._initialized = False
    
    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name or "unknown"
    
    @property
    def is_initialized(self) -> bool:
        """Check if plugin is initialized."""
        return self._initialized
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the LLM plugin."""
        self._config = config
        self._api_key = config.get('api_key')
        self._model_name = config.get('model_name', 'default')
        
        # Test connection
        try:
            health = await self.health_check()
            self._initialized = health.get('healthy', False)
            return self._initialized
        except Exception:
            self._initialized = False
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the LLM plugin."""
        self._initialized = False
        return True
    
    async def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate LLM configuration."""
        required_keys = ['api_key', 'model_name']
        
        for key in required_keys:
            if key not in config:
                return False
        
        return True