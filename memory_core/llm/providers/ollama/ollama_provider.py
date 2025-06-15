"""
Ollama LLM provider implementation.

This module implements the LLMProviderInterface for Ollama's local model inference API,
enabling the use of local language models including llama2, codellama, mistral, and others
without requiring external API services.
"""

import json
import logging
import re
import time
from typing import List, Dict, Any, Optional, AsyncGenerator
import aiohttp

from memory_core.llm.interfaces.llm_provider_interface import (
    LLMProviderInterface,
    LLMTask,
    MessageRole,
    Message,
    LLMResponse,
    LLMError,
    LLMConnectionError,
    LLMValidationError
)


class OllamaLLMProvider(LLMProviderInterface):
    """
    Ollama LLM provider for local model inference.
    
    Supports Ollama models for various LLM tasks including text generation,
    knowledge extraction, relationship detection, and natural language query processing.
    Uses HTTP requests to communicate with the Ollama API running locally.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Ollama LLM provider.
        
        Args:
            config: Configuration dictionary with keys:
                - base_url: Ollama server URL (default: "http://localhost:11434")
                - model_name: Model name (default: "llama2")
                - temperature: Sampling temperature (default: 0.7)
                - max_tokens: Maximum output tokens (default: 4096)
                - timeout: Request timeout in seconds (default: 60)
                - top_p: Top-p sampling (default: 0.9)
                - top_k: Top-k sampling (default: 40)
                - keep_alive: Keep model loaded (default: "5m")
                - num_predict: Number of tokens to predict (-1 for unlimited)
                - repeat_penalty: Penalty for repeating tokens (default: 1.1)
        """
        super().__init__(config)
        
        self.logger = logging.getLogger(__name__)
        
        # Extract configuration
        self.base_url = config.get('base_url', 'http://localhost:11434').rstrip('/')
        self.top_p = config.get('top_p', 0.9)
        self.top_k = config.get('top_k', 40)
        self.keep_alive = config.get('keep_alive', '5m')
        self.num_predict = config.get('num_predict', self.max_tokens)
        self.repeat_penalty = config.get('repeat_penalty', 1.1)
        
        # HTTP session for connection pooling
        self._session: Optional[aiohttp.ClientSession] = None

    def get_default_model(self) -> str:
        """Get the default model name for Ollama provider."""
        return "llama2"

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def connect(self) -> bool:
        """
        Connect to the Ollama server.
        
        Returns:
            True if connection successful, False otherwise
            
        Raises:
            LLMConnectionError: If connection fails
        """
        try:
            session = await self._get_session()
            
            # Test connection by checking server status
            async with session.get(f"{self.base_url}/api/version") as response:
                if response.status == 200:
                    version_info = await response.json()
                    version = version_info.get('version', 'unknown')
                    self.logger.info(f"Connected to Ollama server version: {version}")
                    
                    # Check if the model is available
                    await self._ensure_model_available()
                    
                    self._is_connected = True
                    return True
                else:
                    raise LLMConnectionError(f"Ollama server returned status {response.status}")
                    
        except aiohttp.ClientError as e:
            self.logger.error(f"Failed to connect to Ollama server: {str(e)}")
            raise LLMConnectionError(f"Failed to connect to Ollama server at {self.base_url}: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error connecting to Ollama: {str(e)}")
            raise LLMConnectionError(f"Unexpected error connecting to Ollama: {str(e)}")

    async def _ensure_model_available(self) -> bool:
        """
        Ensure the specified model is available, pull if necessary.
        
        Returns:
            True if model is available
            
        Raises:
            LLMConnectionError: If model cannot be made available
        """
        try:
            # Check if model exists
            if await self._check_model_exists():
                return True
            
            # Model doesn't exist, try to pull it
            self.logger.info(f"Model {self.model_name} not found, attempting to pull...")
            success = await self._pull_model()
            
            if not success:
                raise LLMConnectionError(f"Failed to pull model {self.model_name}")
                
            return True
            
        except Exception as e:
            raise LLMConnectionError(f"Failed to ensure model availability: {str(e)}")

    async def _check_model_exists(self) -> bool:
        """Check if the specified model exists locally."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    models_data = await response.json()
                    models = models_data.get('models', [])
                    
                    for model in models:
                        if model.get('name', '').startswith(self.model_name):
                            return True
                    return False
                else:
                    self.logger.warning(f"Failed to check model existence: status {response.status}")
                    return False
                    
        except Exception as e:
            self.logger.warning(f"Error checking model existence: {str(e)}")
            return False

    async def _pull_model(self) -> bool:
        """Pull the specified model."""
        try:
            session = await self._get_session()
            pull_data = {"name": self.model_name}
            
            self.logger.info(f"Pulling model {self.model_name}...")
            
            async with session.post(
                f"{self.base_url}/api/pull",
                json=pull_data
            ) as response:
                if response.status == 200:
                    # Stream the pull progress
                    async for line in response.content:
                        if line:
                            try:
                                pull_status = json.loads(line.decode().strip())
                                if 'status' in pull_status:
                                    self.logger.debug(f"Pull status: {pull_status['status']}")
                                if pull_status.get('status') == 'success':
                                    self.logger.info(f"Successfully pulled model {self.model_name}")
                                    return True
                            except json.JSONDecodeError:
                                continue
                    return True
                else:
                    self.logger.error(f"Failed to pull model: status {response.status}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error pulling model: {str(e)}")
            return False

    async def disconnect(self) -> bool:
        """
        Disconnect from the Ollama server.
        
        Returns:
            True if disconnection successful, False otherwise
        """
        try:
            if self._session and not self._session.closed:
                await self._session.close()
            
            self._is_connected = False
            self.logger.info("Disconnected from Ollama server")
            return True
            
        except Exception as e:
            self.logger.warning(f"Error during disconnect: {str(e)}")
            return False

    def _clean_markdown_json(self, text: str) -> str:
        """
        Clean markdown code blocks from response text.
        
        Args:
            text: Raw response text that might contain markdown
            
        Returns:
            Cleaned JSON text
        """
        # Remove markdown code blocks (```json ... ```)
        text = re.sub(r'^```(?:json)?\s*\n', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n```\s*$', '', text, flags=re.MULTILINE)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text

    def _convert_messages_to_ollama(self, messages: List[Message]) -> List[Dict[str, str]]:
        """
        Convert internal Message objects to Ollama API format.
        
        Args:
            messages: List of internal Message objects
            
        Returns:
            List of Ollama message dictionaries
        """
        ollama_messages = []
        for msg in messages:
            role_mapping = {
                MessageRole.USER: "user",
                MessageRole.ASSISTANT: "assistant",
                MessageRole.SYSTEM: "system"
            }
            
            ollama_messages.append({
                "role": role_mapping[msg.role],
                "content": msg.content
            })
        
        return ollama_messages

    def _should_use_json_mode(self, task_type: LLMTask) -> bool:
        """
        Determine if JSON mode should be used for the given task type.
        
        Args:
            task_type: The LLM task type
            
        Returns:
            True if JSON mode should be used
        """
        return task_type in [
            LLMTask.KNOWLEDGE_EXTRACTION,
            LLMTask.RELATIONSHIP_DETECTION,
            LLMTask.NATURAL_LANGUAGE_QUERY,
            LLMTask.TEXT_CLASSIFICATION,
            LLMTask.CONTENT_VALIDATION
        ]

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
        if not prompt or not prompt.strip():
            raise LLMValidationError("Prompt cannot be empty")

        try:
            self.logger.debug(f"Generating completion for task: {task_type.value}")
            
            # Prepare request data
            request_data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get('temperature', self.temperature),
                    "top_p": kwargs.get('top_p', self.top_p),
                    "top_k": kwargs.get('top_k', self.top_k),
                    "repeat_penalty": kwargs.get('repeat_penalty', self.repeat_penalty),
                    "num_predict": kwargs.get('max_tokens', self.num_predict)
                },
                "keep_alive": kwargs.get('keep_alive', self.keep_alive)
            }
            
            # Add JSON instruction for structured tasks
            if self._should_use_json_mode(task_type):
                if "json" not in prompt.lower():
                    request_data["prompt"] += "\n\nPlease respond with valid JSON format."
            
            # Make request to Ollama
            session = await self._get_session()
            async with session.post(
                f"{self.base_url}/api/generate",
                json=request_data
            ) as response:
                if response.status == 200:
                    response_data = await response.json()
                    
                    response_content = response_data.get('response', '')
                    
                    # Clean markdown if JSON response expected
                    if self._should_use_json_mode(task_type):
                        response_content = self._clean_markdown_json(response_content)
                    
                    # Create LLM response
                    llm_response = LLMResponse(
                        content=response_content,
                        metadata={
                            'task_type': task_type.value,
                            'model': self.model_name,
                            'provider': 'ollama',
                            'eval_count': response_data.get('eval_count'),
                            'eval_duration': response_data.get('eval_duration'),
                            'prompt_eval_count': response_data.get('prompt_eval_count'),
                            'prompt_eval_duration': response_data.get('prompt_eval_duration')
                        },
                        model=self.model_name,
                        finish_reason=response_data.get('done_reason', 'stop')
                    )
                    
                    # Add usage info
                    if response_data.get('eval_count') or response_data.get('prompt_eval_count'):
                        llm_response.usage = {
                            'prompt_tokens': response_data.get('prompt_eval_count', 0),
                            'completion_tokens': response_data.get('eval_count', 0),
                            'total_tokens': (response_data.get('prompt_eval_count', 0) + 
                                           response_data.get('eval_count', 0))
                        }
                    
                    return llm_response
                    
                else:
                    error_text = await response.text()
                    raise LLMError(f"Ollama API returned status {response.status}: {error_text}")
            
        except aiohttp.ClientError as e:
            self.logger.error(f"HTTP error generating completion: {str(e)}")
            raise LLMConnectionError(f"Failed to connect to Ollama API: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error generating completion: {str(e)}")
            raise LLMError(f"Failed to generate completion: {str(e)}")

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
        if not messages:
            raise LLMValidationError("Messages cannot be empty")

        try:
            # Convert messages to Ollama format
            ollama_messages = self._convert_messages_to_ollama(messages)
            
            # Prepare request data
            request_data = {
                "model": self.model_name,
                "messages": ollama_messages,
                "stream": False,
                "options": {
                    "temperature": kwargs.get('temperature', self.temperature),
                    "top_p": kwargs.get('top_p', self.top_p),
                    "top_k": kwargs.get('top_k', self.top_k),
                    "repeat_penalty": kwargs.get('repeat_penalty', self.repeat_penalty),
                    "num_predict": kwargs.get('max_tokens', self.num_predict)
                },
                "keep_alive": kwargs.get('keep_alive', self.keep_alive)
            }
            
            # Add JSON instruction for structured tasks
            if self._should_use_json_mode(task_type):
                if ollama_messages and "json" not in ollama_messages[-1]["content"].lower():
                    ollama_messages[-1]["content"] += "\n\nPlease respond with valid JSON format."
            
            # Make request to Ollama
            session = await self._get_session()
            async with session.post(
                f"{self.base_url}/api/chat",
                json=request_data
            ) as response:
                if response.status == 200:
                    response_data = await response.json()
                    
                    message_data = response_data.get('message', {})
                    response_content = message_data.get('content', '')
                    
                    # Clean markdown if JSON response expected
                    if self._should_use_json_mode(task_type):
                        response_content = self._clean_markdown_json(response_content)
                    
                    # Create LLM response
                    llm_response = LLMResponse(
                        content=response_content,
                        metadata={
                            'task_type': task_type.value,
                            'model': self.model_name,
                            'provider': 'ollama',
                            'message_count': len(messages),
                            'eval_count': response_data.get('eval_count'),
                            'eval_duration': response_data.get('eval_duration'),
                            'prompt_eval_count': response_data.get('prompt_eval_count'),
                            'prompt_eval_duration': response_data.get('prompt_eval_duration')
                        },
                        model=self.model_name,
                        finish_reason=response_data.get('done_reason', 'stop')
                    )
                    
                    # Add usage info
                    if response_data.get('eval_count') or response_data.get('prompt_eval_count'):
                        llm_response.usage = {
                            'prompt_tokens': response_data.get('prompt_eval_count', 0),
                            'completion_tokens': response_data.get('eval_count', 0),
                            'total_tokens': (response_data.get('prompt_eval_count', 0) + 
                                           response_data.get('eval_count', 0))
                        }
                    
                    return llm_response
                    
                else:
                    error_text = await response.text()
                    raise LLMError(f"Ollama API returned status {response.status}: {error_text}")
            
        except aiohttp.ClientError as e:
            self.logger.error(f"HTTP error generating chat completion: {str(e)}")
            raise LLMConnectionError(f"Failed to connect to Ollama API: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error generating chat completion: {str(e)}")
            raise LLMError(f"Failed to generate chat completion: {str(e)}")

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
        if not prompt or not prompt.strip():
            raise LLMValidationError("Prompt cannot be empty")

        try:
            self.logger.debug(f"Generating streaming completion for task: {task_type.value}")
            
            # Prepare request data
            request_data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": kwargs.get('temperature', self.temperature),
                    "top_p": kwargs.get('top_p', self.top_p),
                    "top_k": kwargs.get('top_k', self.top_k),
                    "repeat_penalty": kwargs.get('repeat_penalty', self.repeat_penalty),
                    "num_predict": kwargs.get('max_tokens', self.num_predict)
                },
                "keep_alive": kwargs.get('keep_alive', self.keep_alive)
            }
            
            # Make request to Ollama
            session = await self._get_session()
            async with session.post(
                f"{self.base_url}/api/generate",
                json=request_data
            ) as response:
                if response.status == 200:
                    # Process streaming response
                    async for line in response.content:
                        if line:
                            try:
                                chunk_data = json.loads(line.decode().strip())
                                if 'response' in chunk_data:
                                    yield chunk_data['response']
                                if chunk_data.get('done', False):
                                    break
                            except json.JSONDecodeError:
                                continue
                else:
                    error_text = await response.text()
                    raise LLMError(f"Ollama API returned status {response.status}: {error_text}")
                    
        except aiohttp.ClientError as e:
            self.logger.error(f"HTTP error generating streaming completion: {str(e)}")
            raise LLMConnectionError(f"Failed to connect to Ollama API: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error generating streaming completion: {str(e)}")
            raise LLMError(f"Failed to generate streaming completion: {str(e)}")

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
        if not text or not text.strip():
            return []

        try:
            prompt = self._create_knowledge_extraction_prompt(text, source_info)
            
            response = await self.generate_completion(
                prompt,
                LLMTask.KNOWLEDGE_EXTRACTION,
                temperature=0.1  # Lower temperature for more consistent extraction
            )
            
            # Parse JSON response
            try:
                knowledge_units = json.loads(response.content)
                
                # Validate the response structure
                if not isinstance(knowledge_units, list):
                    return []
                
                # Filter out any malformed units
                valid_units = []
                for unit in knowledge_units:
                    if isinstance(unit, dict) and "content" in unit:
                        valid_units.append(unit)
                
                self.logger.info(f"Extracted {len(valid_units)} knowledge units from text")
                return valid_units
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse knowledge extraction response as JSON: {str(e)}")
                self.logger.debug(f"Raw response: {response.content}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error extracting knowledge units: {str(e)}")
            raise LLMError(f"Failed to extract knowledge units: {str(e)}")

    def _create_knowledge_extraction_prompt(self, raw_text: str, source_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a prompt for knowledge extraction.
        
        Args:
            raw_text: The raw text to extract knowledge from
            source_info: Optional source information
            
        Returns:
            A formatted prompt string
        """
        source_context = ""
        if source_info:
            source_context = f"\nSource context: {json.dumps(source_info)}"
        
        return f"""You are an expert at transforming raw text into structured knowledge units.
For each distinct piece of knowledge in the text, return a JSON object with this format:
{{
"content": "<short statement capturing a single knowledge unit>",
"tags": ["<tag1>", "<tag2>", ...],
"metadata": {{
    "confidence_level": "<float between 0 and 1>",
    "domain": "<primary knowledge domain>",
    "language": "<language of the content>",
    "importance": "<float between 0 and 1>"
}},
"source": {{
    "type": "<source type: webpage, book, scientific_paper, video, user_input, etc.>",
    "url": "<url if applicable>",
    "reference": "<citation or reference information>",
    "page": "<page number if applicable>"
}}
}}

Extract as many meaningful knowledge units as possible from the input. If the text is too short, 
ambiguous, or doesn't contain meaningful information, return an empty list: [].

Format your entire response as a valid JSON array of these objects.{source_context}

Text input:
{raw_text}
"""

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
        if not entities or len(entities) < 2:
            return []

        try:
            prompt = self._create_relationship_detection_prompt(entities, context)
            
            response = await self.generate_completion(
                prompt,
                LLMTask.RELATIONSHIP_DETECTION,
                temperature=0.1  # Lower temperature for more consistent detection
            )
            
            # Parse JSON response
            try:
                relationships = json.loads(response.content)
                
                # Validate the response structure
                if not isinstance(relationships, list):
                    return []
                
                # Filter out any malformed relationships
                valid_relationships = []
                for rel in relationships:
                    if (isinstance(rel, dict) and 
                        "source" in rel and "target" in rel and "relationship_type" in rel):
                        valid_relationships.append(rel)
                
                self.logger.info(f"Detected {len(valid_relationships)} relationships between entities")
                return valid_relationships
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse relationship detection response as JSON: {str(e)}")
                self.logger.debug(f"Raw response: {response.content}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error detecting relationships: {str(e)}")
            raise LLMError(f"Failed to detect relationships: {str(e)}")

    def _create_relationship_detection_prompt(self, entities: List[str], context: Optional[str] = None) -> str:
        """
        Create a prompt for relationship detection.
        
        Args:
            entities: List of entities to analyze
            context: Optional context text
            
        Returns:
            A formatted prompt string
        """
        entities_str = "\n".join([f"- {entity}" for entity in entities])
        context_part = f"\nContext: {context}" if context else ""
        
        return f"""You are an expert at detecting relationships between entities.
Analyze the following entities and identify meaningful relationships between them.

Entities:
{entities_str}{context_part}

For each relationship you identify, return a JSON object with this format:
{{
"source": "<source entity>",
"target": "<target entity>",
"relationship_type": "<type of relationship: related_to, part_of, causes, enables, etc.>",
"direction": "<bidirectional, source_to_target, or target_to_source>",
"confidence": "<float between 0 and 1>",
"description": "<brief description of the relationship>",
"metadata": {{
    "strength": "<float between 0 and 1>",
    "domain": "<relationship domain>",
    "evidence": "<evidence for this relationship>"
}}
}}

Format your entire response as a valid JSON array of these relationship objects.
Only include relationships with high confidence (>0.7).
"""

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
        if not query or not query.strip():
            raise LLMValidationError("Query cannot be empty")

        try:
            prompt = self._create_query_parsing_prompt(query, context)
            
            response = await self.generate_completion(
                prompt,
                LLMTask.NATURAL_LANGUAGE_QUERY,
                temperature=0.1  # Lower temperature for more consistent parsing
            )
            
            # Parse JSON response
            try:
                parsed_query = json.loads(response.content)
                
                # Validate the response structure
                if not isinstance(parsed_query, dict):
                    raise LLMValidationError("Invalid query parsing response structure")
                
                # Set default values for required fields
                parsed_query.setdefault('intent', 'search')
                parsed_query.setdefault('entities', [])
                parsed_query.setdefault('relationships', [])
                parsed_query.setdefault('constraints', [])
                parsed_query.setdefault('query_type', 'natural_language')
                parsed_query.setdefault('confidence', 0.5)
                
                self.logger.info(f"Parsed query with intent: {parsed_query.get('intent')}")
                return parsed_query
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse query parsing response as JSON: {str(e)}")
                self.logger.debug(f"Raw response: {response.content}")
                # Return default structure
                return {
                    'intent': 'search',
                    'entities': [],
                    'relationships': [],
                    'constraints': [],
                    'query_type': 'natural_language',
                    'confidence': 0.0
                }
                
        except Exception as e:
            self.logger.error(f"Error parsing natural language query: {str(e)}")
            raise LLMError(f"Failed to parse natural language query: {str(e)}")

    def _create_query_parsing_prompt(self, query: str, context: Optional[str] = None) -> str:
        """
        Create a prompt for natural language query parsing.
        
        Args:
            query: Natural language query
            context: Optional context
            
        Returns:
            Formatted prompt string
        """
        context_part = f"\nContext: {context}" if context else ""
        
        return f"""You are an expert at parsing natural language queries for a knowledge graph database.
Analyze the following query and extract structured information.

Query: "{query}"{context_part}

Return a JSON object with the following structure:
{{
    "intent": "<primary intent: search, find_relationships, aggregate, compare, etc.>",
    "entities": ["<list of entities/concepts mentioned>"],
    "relationships": ["<list of relationships mentioned>"],
    "constraints": ["<list of constraints/filters mentioned>"],
    "query_type": "<one of: natural_language, graph_pattern, semantic_search, relationship_search, aggregation, hybrid>",
    "semantic_keywords": ["<key terms for semantic search>"],
    "confidence": <float between 0 and 1>,
    "graph_pattern": {{
        "nodes": [
            {{"type": "<node type>", "properties": {{"key": "value"}}, "variable": "<variable name>"}}
        ],
        "edges": [
            {{"type": "<relationship type>", "from": "<source variable>", "to": "<target variable>", "properties": {{}}}}
        ],
        "constraints": ["<gremlin-style constraints>"]
    }},
    "filters": [
        {{"field": "<field name>", "operator": "<eq|ne|gt|lt|contains|regex>", "value": "<filter value>"}}
    ],
    "sort_criteria": [
        {{"field": "<field name>", "order": "<asc|desc>"}}
    ],
    "aggregations": [
        {{"type": "<count|sum|avg|min|max>", "field": "<field name>"}}
    ]
}}

Guidelines:
- For similarity/semantic queries, focus on semantic_keywords
- For relationship queries, identify source and target entities
- For aggregation queries, identify what to count/sum/etc
- Extract any mentioned filters, sorting, or constraints
- Be conservative with confidence scores
- Use graph_pattern for structured queries that can be represented as node-edge patterns

Examples:
- "Find nodes similar to 'artificial intelligence'" → semantic_search
- "Show relationships between Python and machine learning" → relationship_search  
- "Count how many programming languages are mentioned" → aggregation
- "Find all concepts related to databases created after 2020" → graph_pattern with constraints
"""

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
        if not content or not content.strip():
            raise LLMValidationError("Content cannot be empty")
        
        if not criteria:
            raise LLMValidationError("Validation criteria cannot be empty")

        try:
            prompt = self._create_validation_prompt(content, criteria)
            
            response = await self.generate_completion(
                prompt,
                LLMTask.CONTENT_VALIDATION,
                temperature=0.1  # Lower temperature for more consistent validation
            )
            
            # Parse JSON response
            try:
                validation_result = json.loads(response.content)
                
                # Validate the response structure
                if not isinstance(validation_result, dict):
                    raise LLMValidationError("Invalid validation response structure")
                
                self.logger.info(f"Content validation completed with {len(criteria)} criteria")
                return validation_result
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse validation response as JSON: {str(e)}")
                self.logger.debug(f"Raw response: {response.content}")
                # Return default validation failure
                return {
                    'valid': False,
                    'criteria_results': {criterion: False for criterion in criteria},
                    'overall_score': 0.0,
                    'errors': [f"Failed to parse validation response: {str(e)}"]
                }
                
        except Exception as e:
            self.logger.error(f"Error validating content: {str(e)}")
            raise LLMError(f"Failed to validate content: {str(e)}")

    def _create_validation_prompt(self, content: str, criteria: List[str]) -> str:
        """
        Create a prompt for content validation.
        
        Args:
            content: Content to validate
            criteria: List of validation criteria
            
        Returns:
            Formatted prompt string
        """
        criteria_str = "\n".join([f"- {criterion}" for criterion in criteria])
        
        return f"""You are an expert content validator. Analyze the following content against the given criteria.

Content to validate:
{content}

Validation criteria:
{criteria_str}

Return a JSON object with the following structure:
{{
    "valid": <true/false - overall validation result>,
    "criteria_results": {{
        "<criterion_1>": <true/false>,
        "<criterion_2>": <true/false>,
        ...
    }},
    "overall_score": <float between 0 and 1>,
    "errors": ["<list of validation errors>"],
    "warnings": ["<list of validation warnings>"],
    "suggestions": ["<list of improvement suggestions>"],
    "metadata": {{
        "validation_timestamp": "<current timestamp>",
        "criteria_count": <number of criteria>,
        "passed_criteria": <number of passed criteria>
    }}
}}

For each criterion, provide a true/false result based on whether the content meets that requirement.
Include detailed errors, warnings, and suggestions for improvement.
"""

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the Ollama provider.
        
        Returns:
            Health status information
        """
        health_status = {
            'provider': 'ollama',
            'model': self.model_name,
            'base_url': self.base_url,
            'connected': False,
            'server_running': False,
            'model_available': False,
            'test_passed': False,
            'error': None,
            'timestamp': None,
            'response_time': None
        }
        
        try:
            start_time = time.time()
            
            # Check if server is running
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/version") as response:
                if response.status == 200:
                    health_status['server_running'] = True
                else:
                    health_status['error'] = f"Server returned status {response.status}"
                    return health_status
            
            # Check if model is available
            health_status['model_available'] = await self._check_model_exists()
            
            if not health_status['model_available']:
                health_status['error'] = f"Model {self.model_name} not available"
                return health_status
            
            # Test with a simple completion
            test_response = await self.generate_completion(
                "Test connection. Respond with 'OK'.",
                LLMTask.GENERAL_COMPLETION
            )
            
            response_time = time.time() - start_time
            
            if test_response and test_response.content:
                health_status.update({
                    'connected': True,
                    'test_passed': True,
                    'response_time': response_time,
                    'timestamp': time.time()
                })
            else:
                health_status['error'] = "Empty response from test request"
                
        except Exception as e:
            health_status['error'] = str(e)
            self.logger.error(f"Ollama health check failed: {str(e)}")
        
        return health_status

    def get_supported_tasks(self) -> List[LLMTask]:
        """
        Get list of tasks supported by Ollama provider.
        
        Returns:
            List of supported LLM tasks
        """
        return [
            LLMTask.GENERAL_COMPLETION,
            LLMTask.KNOWLEDGE_EXTRACTION,
            LLMTask.RELATIONSHIP_DETECTION,
            LLMTask.NATURAL_LANGUAGE_QUERY,
            LLMTask.TEXT_CLASSIFICATION,
            LLMTask.SUMMARIZATION,
            LLMTask.CONTENT_VALIDATION
        ]

    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about this provider.
        
        Returns:
            Provider information dictionary
        """
        return {
            'name': 'OllamaLLMProvider',
            'provider': 'ollama',
            'model': self.model_name,
            'type': 'local',
            'base_url': self.base_url,
            'supported_tasks': [task.value for task in self.get_supported_tasks()],
            'connected': self.is_connected,
            'config': {
                'temperature': self.temperature,
                'max_tokens': self.max_tokens,
                'timeout': self.timeout,
                'top_p': self.top_p,
                'top_k': self.top_k,
                'keep_alive': self.keep_alive
            },
            'features': [
                'streaming',
                'chat_completion',
                'knowledge_extraction',
                'relationship_detection',
                'query_parsing',
                'local_inference',
                'model_pulling'
            ]
        }

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for given text.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Simple estimation: roughly 4 characters per token
        # This varies by model and tokenizer, but provides a reasonable approximation
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

    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models from Ollama server.
        
        Returns:
            List of model information dictionaries
            
        Raises:
            LLMConnectionError: If unable to retrieve models
        """
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    models_data = await response.json()
                    models = models_data.get('models', [])
                    
                    model_list = []
                    for model in models:
                        model_info = {
                            'name': model.get('name', ''),
                            'size': model.get('size', 0),
                            'modified_at': model.get('modified_at', ''),
                            'digest': model.get('digest', ''),
                            'details': model.get('details', {})
                        }
                        model_list.append(model_info)
                    
                    return model_list
                else:
                    raise LLMConnectionError(f"Failed to get models: status {response.status}")
                    
        except aiohttp.ClientError as e:
            raise LLMConnectionError(f"Failed to connect to Ollama API: {str(e)}")
        except Exception as e:
            raise LLMError(f"Error getting available models: {str(e)}")
