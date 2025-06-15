"""
HuggingFace Transformers LLM provider implementation.

This module implements the LLMProviderInterface for HuggingFace Transformers,
supporting both local model execution and HuggingFace Inference API for various
LLM tasks including knowledge extraction, relationship detection, and natural language
query processing.
"""

import json
import logging
import re
import time
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
import asyncio

try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, pipeline,
        GenerationConfig, StoppingCriteria, StoppingCriteriaList
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    pipeline = None
    GenerationConfig = None
    StoppingCriteria = None
    StoppingCriteriaList = None

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None

from memory_core.llm.interfaces.llm_provider_interface import (
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


class StopOnTokens(StoppingCriteria):
    """Custom stopping criteria for generation."""
    
    def __init__(self, stop_token_ids: List[int]):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


class HuggingFaceLLMProvider(LLMProviderInterface):
    """
    HuggingFace Transformers LLM provider.
    
    Supports both local model execution and HuggingFace Inference API for various
    LLM tasks including text generation, knowledge extraction, relationship detection,
    and natural language query processing.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the HuggingFace LLM provider.
        
        Args:
            config: Configuration dictionary with keys:
                - model_name: Model name/path (default: 'microsoft/DialoGPT-medium')
                - use_api: Whether to use HuggingFace Inference API (default: False)
                - api_key: HuggingFace API token (required if use_api=True)
                - device: Device for local execution ('cpu', 'cuda', 'auto')
                - torch_dtype: Torch dtype ('auto', 'float16', 'float32')
                - temperature: Sampling temperature (default: 0.7)
                - max_tokens: Maximum output tokens (default: 4096)
                - timeout: Request timeout in seconds (default: 30)
                - trust_remote_code: Whether to trust remote code (default: False)
                - load_in_8bit: Whether to load model in 8-bit (default: False)
                - load_in_4bit: Whether to load model in 4-bit (default: False)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise LLMConnectionError(
                "HuggingFace Transformers library is not installed. "
                "Please install with: pip install transformers torch"
            )
        
        super().__init__(config)
        
        self.logger = logging.getLogger(__name__)
        
        # Extract configuration
        self.use_api = config.get('use_api', False)
        self.api_key = config.get('api_key')
        self.device = config.get('device', 'auto')
        self.torch_dtype = config.get('torch_dtype', 'auto')
        self.trust_remote_code = config.get('trust_remote_code', False)
        self.load_in_8bit = config.get('load_in_8bit', False)
        self.load_in_4bit = config.get('load_in_4bit', False)
        
        # Model components (for local execution)
        self.tokenizer = None
        self.model = None
        self.generation_pipeline = None
        
        # HTTP session (for API usage)
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Determine execution mode
        if self.use_api:
            if not AIOHTTP_AVAILABLE:
                raise LLMConnectionError(
                    "aiohttp library is required for HuggingFace API usage. "
                    "Please install with: pip install aiohttp"
                )
            if not self.api_key:
                raise LLMConnectionError(
                    "HuggingFace API key is required when use_api=True"
                )
        
        # Initialize based on mode
        if not self.use_api:
            self._initialize_local_model()

    def get_default_model(self) -> str:
        """Get the default model name for HuggingFace provider."""
        return "microsoft/DialoGPT-medium"

    def _initialize_local_model(self):
        """Initialize local model and tokenizer."""
        try:
            self.logger.info(f"Loading HuggingFace model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Prepare model loading kwargs
            model_kwargs = {
                'trust_remote_code': self.trust_remote_code
            }
            
            # Set torch dtype
            if self.torch_dtype != 'auto':
                if self.torch_dtype == 'float16':
                    model_kwargs['torch_dtype'] = torch.float16
                elif self.torch_dtype == 'float32':
                    model_kwargs['torch_dtype'] = torch.float32
            
            # Set quantization options
            if self.load_in_8bit:
                model_kwargs['load_in_8bit'] = True
            elif self.load_in_4bit:
                model_kwargs['load_in_4bit'] = True
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Move to device
            if self.device == 'auto':
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            if not (self.load_in_8bit or self.load_in_4bit):
                self.model = self.model.to(self.device)
            
            # Create generation pipeline
            self.generation_pipeline = pipeline(
                'text-generation',
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == 'cuda' else -1
            )
            
            self.logger.info(f"Model loaded successfully on device: {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize local model: {str(e)}")
            raise LLMConnectionError(f"Failed to initialize HuggingFace model: {str(e)}")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session for API calls."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            headers = {'Authorization': f'Bearer {self.api_key}'}
            self._session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self._session

    async def connect(self) -> bool:
        """
        Connect to the HuggingFace service.
        
        Returns:
            True if connection successful, False otherwise
            
        Raises:
            LLMConnectionError: If connection fails
        """
        try:
            if self.use_api:
                # Test API connection
                await self.test_connection()
            else:
                # For local models, check if model is loaded
                if self.model is None or self.tokenizer is None:
                    raise LLMConnectionError("Local model not properly initialized")
            
            self._is_connected = True
            self.logger.info("Successfully connected to HuggingFace service")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to HuggingFace service: {str(e)}")
            raise LLMConnectionError(f"Failed to connect to HuggingFace service: {str(e)}")

    async def disconnect(self) -> bool:
        """
        Disconnect from the HuggingFace service.
        
        Returns:
            True if disconnection successful, False otherwise
        """
        try:
            if self._session and not self._session.closed:
                await self._session.close()
            
            # Note: We don't unload local models as that would be expensive
            # Models remain in memory for reuse
            
            self._is_connected = False
            self.logger.info("Disconnected from HuggingFace service")
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

    def _should_use_structured_output(self, task_type: LLMTask) -> bool:
        """
        Determine if structured output should be requested for the given task type.
        
        Args:
            task_type: The LLM task type
            
        Returns:
            True if structured output should be requested
        """
        return task_type in [
            LLMTask.KNOWLEDGE_EXTRACTION,
            LLMTask.RELATIONSHIP_DETECTION,
            LLMTask.NATURAL_LANGUAGE_QUERY,
            LLMTask.TEXT_CLASSIFICATION,
            LLMTask.CONTENT_VALIDATION
        ]

    async def _generate_with_api(self, prompt: str, **kwargs) -> str:
        """Generate text using HuggingFace Inference API."""
        session = await self._get_session()
        
        api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": kwargs.get('temperature', self.temperature),
                "max_new_tokens": kwargs.get('max_tokens', self.max_tokens),
                "return_full_text": False
            }
        }
        
        async with session.post(api_url, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('generated_text', '')
                else:
                    return str(result)
                    
            elif response.status == 503:
                # Model is loading
                error_text = await response.text()
                raise LLMConnectionError(f"Model is loading. Please try again later: {error_text}")
            else:
                error_text = await response.text()
                raise LLMError(f"HuggingFace API error {response.status}: {error_text}")

    def _generate_with_local(self, prompt: str, **kwargs) -> str:
        """Generate text using local model."""
        generation_kwargs = {
            'max_new_tokens': kwargs.get('max_tokens', self.max_tokens),
            'temperature': kwargs.get('temperature', self.temperature),
            'do_sample': True,
            'pad_token_id': self.tokenizer.eos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id
        }
        
        # Add stopping criteria
        stop_tokens = [self.tokenizer.eos_token_id]
        if hasattr(self.tokenizer, 'convert_tokens_to_ids'):
            # Add common stop tokens
            for stop_word in ['\n\n', '<|endoftext|>', '</s>']:
                try:
                    stop_id = self.tokenizer.convert_tokens_to_ids(stop_word)
                    if stop_id != self.tokenizer.unk_token_id:
                        stop_tokens.append(stop_id)
                except:
                    pass
        
        stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_tokens)])
        generation_kwargs['stopping_criteria'] = stopping_criteria
        
        # Generate
        try:
            outputs = self.generation_pipeline(
                prompt,
                **generation_kwargs
            )
            
            if outputs and len(outputs) > 0:
                generated_text = outputs[0]['generated_text']
                # Remove the input prompt from the output
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
                return generated_text
            else:
                return ""
                
        except Exception as e:
            raise LLMError(f"Local generation failed: {str(e)}")

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
            
            # Add JSON instruction for structured tasks
            if self._should_use_structured_output(task_type):
                if "json" not in prompt.lower():
                    prompt += "\n\nPlease respond with valid JSON format."
            
            start_time = time.time()
            
            # Generate based on mode
            if self.use_api:
                response_content = await self._generate_with_api(prompt, **kwargs)
            else:
                # Run local generation in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                response_content = await loop.run_in_executor(
                    None, self._generate_with_local, prompt, kwargs
                )
            
            response_time = time.time() - start_time
            
            # Clean markdown if JSON response expected
            if self._should_use_structured_output(task_type):
                response_content = self._clean_markdown_json(response_content)
            
            # Create LLM response
            llm_response = LLMResponse(
                content=response_content,
                metadata={
                    'task_type': task_type.value,
                    'model': self.model_name,
                    'provider': 'huggingface',
                    'mode': 'api' if self.use_api else 'local',
                    'response_time': response_time
                },
                model=self.model_name
            )
            
            # Estimate token usage (rough approximation)
            input_tokens = len(prompt.split()) * 1.3  # Rough token count
            output_tokens = len(response_content.split()) * 1.3
            
            llm_response.usage = {
                'prompt_tokens': int(input_tokens),
                'completion_tokens': int(output_tokens),
                'total_tokens': int(input_tokens + output_tokens)
            }
            
            return llm_response
            
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
            # Convert messages to a single prompt
            prompt_parts = []
            
            for message in messages:
                role_prefix = {
                    MessageRole.SYSTEM: "System:",
                    MessageRole.USER: "Human:",
                    MessageRole.ASSISTANT: "Assistant:"
                }.get(message.role, "")
                
                prompt_parts.append(f"{role_prefix} {message.content}")
            
            # Add final assistant prompt
            prompt_parts.append("Assistant:")
            prompt = "\n".join(prompt_parts)
            
            # Generate completion
            return await self.generate_completion(prompt, task_type, **kwargs)
            
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
            # For simplicity, generate full response and yield in chunks
            # Real streaming would require more complex implementation
            response = await self.generate_completion(prompt, task_type, **kwargs)
            
            # Yield response in chunks of ~50 characters
            content = response.content
            chunk_size = 50
            
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i + chunk_size]
                yield chunk
                await asyncio.sleep(0.01)  # Small delay to simulate streaming
                
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
        """Create a prompt for knowledge extraction."""
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
{raw_text}"""

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
                temperature=0.1
            )
            
            # Parse JSON response
            try:
                relationships = json.loads(response.content)
                
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
                return []
                
        except Exception as e:
            self.logger.error(f"Error detecting relationships: {str(e)}")
            raise LLMError(f"Failed to detect relationships: {str(e)}")

    def _create_relationship_detection_prompt(self, entities: List[str], context: Optional[str] = None) -> str:
        """Create a prompt for relationship detection."""
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
Only include relationships with high confidence (>0.7)."""

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
                temperature=0.1
            )
            
            # Parse JSON response
            try:
                parsed_query = json.loads(response.content)
                
                if not isinstance(parsed_query, dict):
                    raise LLMValidationError("Invalid query parsing response structure")
                
                # Set default values for required fields
                parsed_query.setdefault('intent', 'search')
                parsed_query.setdefault('entities', [])
                parsed_query.setdefault('relationships', [])
                parsed_query.setdefault('constraints', [])
                parsed_query.setdefault('query_type', 'natural_language')
                parsed_query.setdefault('confidence', 0.5)
                
                return parsed_query
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse query parsing response as JSON: {str(e)}")
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
        """Create a prompt for natural language query parsing."""
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
    "confidence": <float between 0 and 1>
}}

Guidelines:
- For similarity/semantic queries, focus on semantic_keywords
- For relationship queries, identify source and target entities
- For aggregation queries, identify what to count/sum/etc
- Extract any mentioned filters, sorting, or constraints
- Be conservative with confidence scores"""

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
                temperature=0.1
            )
            
            # Parse JSON response
            try:
                validation_result = json.loads(response.content)
                
                if not isinstance(validation_result, dict):
                    raise LLMValidationError("Invalid validation response structure")
                
                return validation_result
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse validation response as JSON: {str(e)}")
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
        """Create a prompt for content validation."""
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
    "suggestions": ["<list of improvement suggestions>"]
}}

For each criterion, provide a true/false result based on whether the content meets that requirement."""

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the HuggingFace provider.
        
        Returns:
            Health status information
        """
        health_status = {
            'provider': 'huggingface',
            'model': self.model_name,
            'mode': 'api' if self.use_api else 'local',
            'connected': False,
            'test_passed': False,
            'error': None,
            'timestamp': None,
            'response_time': None
        }
        
        try:
            start_time = time.time()
            
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
            self.logger.error(f"HuggingFace health check failed: {str(e)}")
        
        return health_status

    def get_supported_tasks(self) -> List[LLMTask]:
        """
        Get list of tasks supported by HuggingFace provider.
        
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
        info = {
            'name': 'HuggingFaceLLMProvider',
            'provider': 'huggingface',
            'model': self.model_name,
            'type': 'api' if self.use_api else 'local',
            'supported_tasks': [task.value for task in self.get_supported_tasks()],
            'connected': self.is_connected,
            'config': {
                'temperature': self.temperature,
                'max_tokens': self.max_tokens,
                'timeout': self.timeout,
                'device': self.device,
                'torch_dtype': self.torch_dtype
            },
            'features': [
                'local_execution',
                'api_inference',
                'knowledge_extraction',
                'relationship_detection',
                'query_parsing'
            ]
        }
        
        if not self.use_api:
            info['config'].update({
                'load_in_8bit': self.load_in_8bit,
                'load_in_4bit': self.load_in_4bit,
                'trust_remote_code': self.trust_remote_code
            })
        
        return info

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for given text.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        if self.tokenizer:
            # Use actual tokenizer if available
            try:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                return len(tokens)
            except:
                pass
        
        # Fallback to simple estimation
        return len(text.split()) * 1.3  # Rough approximation