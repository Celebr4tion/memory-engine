"""
Anthropic Claude LLM provider implementation.

This module implements the LLMProviderInterface for Anthropic's Claude API,
supporting various LLM tasks including knowledge extraction, relationship detection,
and natural language query processing.
"""

import json
import logging
import re
import time
from typing import List, Dict, Any, Optional, AsyncGenerator

try:
    import anthropic
    from anthropic import AsyncAnthropic
    from anthropic.types import Message as AnthropicMessage
    from anthropic.types import MessageParam

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None
    AsyncAnthropic = None
    AnthropicMessage = None
    MessageParam = None

from memory_core.llm.interfaces.llm_provider_interface import (
    LLMProviderInterface,
    LLMTask,
    MessageRole,
    Message,
    LLMResponse,
    LLMError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMValidationError,
)


class AnthropicLLMProvider(LLMProviderInterface):
    """
    Anthropic Claude LLM provider.

    Supports Anthropic Claude models for various LLM tasks including text generation,
    knowledge extraction, relationship detection, and natural language query processing.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Anthropic LLM provider.

        Args:
            config: Configuration dictionary with keys:
                - api_key: Anthropic API key (required)
                - model_name: Claude model name (default: 'claude-3-5-sonnet-20241022')
                - temperature: Sampling temperature (default: 0.7)
                - max_tokens: Maximum output tokens (default: 4096)
                - timeout: Request timeout in seconds (default: 30)
                - base_url: Optional custom base URL
                - top_p: Top-p sampling (default: 0.9)
                - top_k: Top-k sampling (default: None)
        """
        if not ANTHROPIC_AVAILABLE:
            raise LLMConnectionError(
                "Anthropic library is not installed. Please install with: pip install anthropic"
            )

        super().__init__(config)

        self.logger = logging.getLogger(__name__)

        # Extract configuration
        self.api_key = config.get("api_key")
        if not self.api_key:
            raise LLMConnectionError("Anthropic API key is required for Anthropic provider")

        self.base_url = config.get("base_url")
        self.top_p = config.get("top_p", 0.9)
        self.top_k = config.get("top_k")

        # Initialize Anthropic client
        try:
            client_kwargs = {"api_key": self.api_key, "timeout": self.timeout}

            if self.base_url:
                client_kwargs["base_url"] = self.base_url

            self.client = AsyncAnthropic(**client_kwargs)
        except Exception as e:
            raise LLMConnectionError(f"Failed to initialize Anthropic client: {str(e)}")

    def get_default_model(self) -> str:
        """Get the default model name for Anthropic provider."""
        return "claude-3-5-sonnet-20241022"

    async def connect(self) -> bool:
        """
        Connect to the Anthropic API.

        Returns:
            True if connection successful, False otherwise

        Raises:
            LLMConnectionError: If connection fails
        """
        try:
            # Test connection with a simple request
            await self.test_connection()
            self._is_connected = True
            self.logger.info("Successfully connected to Anthropic API")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Anthropic API: {str(e)}")
            raise LLMConnectionError(f"Failed to connect to Anthropic API: {str(e)}")

    async def disconnect(self) -> bool:
        """
        Disconnect from the Anthropic API.

        Returns:
            True if disconnection successful, False otherwise
        """
        try:
            await self.client.close()
        except Exception:
            pass  # Ignore errors during cleanup

        self._is_connected = False
        self.logger.info("Disconnected from Anthropic API")
        return True

    def _clean_markdown_json(self, text: str) -> str:
        """
        Clean markdown code blocks from response text.

        Args:
            text: Raw response text that might contain markdown

        Returns:
            Cleaned JSON text
        """
        # Remove markdown code blocks (```json ... ```)
        text = re.sub(r"^```(?:json)?\s*\n", "", text, flags=re.MULTILINE)
        text = re.sub(r"\n```\s*$", "", text, flags=re.MULTILINE)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def _convert_messages_to_anthropic(
        self, messages: List[Message]
    ) -> tuple[List[MessageParam], Optional[str]]:
        """
        Convert internal Message objects to Anthropic API format.

        Args:
            messages: List of internal Message objects

        Returns:
            Tuple of (anthropic_messages, system_message)
        """
        anthropic_messages = []
        system_message = None

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                # Claude handles system messages separately
                system_message = msg.content
            else:
                role_mapping = {MessageRole.USER: "user", MessageRole.ASSISTANT: "assistant"}

                if msg.role in role_mapping:
                    anthropic_messages.append(
                        {"role": role_mapping[msg.role], "content": msg.content}
                    )

        return anthropic_messages, system_message

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
            LLMTask.CONTENT_VALIDATION,
        ]

    async def generate_completion(
        self, prompt: str, task_type: LLMTask = LLMTask.GENERAL_COMPLETION, **kwargs
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

            # Convert prompt to messages format
            messages = [Message(role=MessageRole.USER, content=prompt)]

            return await self.generate_chat_completion(messages, task_type, **kwargs)

        except Exception as e:
            self.logger.error(f"Error generating completion: {str(e)}")
            raise LLMError(f"Failed to generate completion: {str(e)}")

    async def generate_chat_completion(
        self, messages: List[Message], task_type: LLMTask = LLMTask.GENERAL_COMPLETION, **kwargs
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
            # Convert messages to Anthropic format
            anthropic_messages, system_message = self._convert_messages_to_anthropic(messages)

            if not anthropic_messages:
                raise LLMValidationError("At least one user or assistant message is required")

            # Prepare request parameters
            request_params = {
                "model": self.model_name,
                "messages": anthropic_messages,
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
            }

            # Add system message if present
            if system_message:
                request_params["system"] = system_message

            # Add optional parameters
            if self.top_p is not None:
                request_params["top_p"] = kwargs.get("top_p", self.top_p)
            if self.top_k is not None:
                request_params["top_k"] = kwargs.get("top_k", self.top_k)

            # Add JSON format request for structured tasks
            if self._should_use_structured_output(task_type):
                # Ensure the last message requests JSON format
                if anthropic_messages and "json" not in anthropic_messages[-1]["content"].lower():
                    anthropic_messages[-1][
                        "content"
                    ] += "\n\nPlease respond with valid JSON format."

            # Generate completion
            response = await self.client.messages.create(**request_params)

            # Extract response content
            if not response.content or not response.content:
                raise LLMError("Empty response from Anthropic API")

            # Claude returns content as a list of content blocks
            response_content = ""
            for content_block in response.content:
                if hasattr(content_block, "text"):
                    response_content += content_block.text
                elif hasattr(content_block, "content"):
                    response_content += str(content_block.content)
                else:
                    response_content += str(content_block)

            # Clean markdown if JSON response expected
            if self._should_use_structured_output(task_type):
                response_content = self._clean_markdown_json(response_content)

            # Create LLM response
            llm_response = LLMResponse(
                content=response_content,
                metadata={
                    "task_type": task_type.value,
                    "model": self.model_name,
                    "provider": "anthropic",
                    "message_count": len(messages),
                },
                model=self.model_name,
                finish_reason=response.stop_reason,
            )

            # Add usage info
            if hasattr(response, "usage") and response.usage:
                llm_response.usage = {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                }

            return llm_response

        except Exception as e:
            # Handle Anthropic-specific errors if the library is available
            if ANTHROPIC_AVAILABLE and anthropic:
                if isinstance(e, anthropic.RateLimitError):
                    self.logger.error(f"Rate limit error: {str(e)}")
                    raise LLMRateLimitError(f"Rate limit exceeded: {str(e)}")
                elif isinstance(e, anthropic.APIConnectionError):
                    self.logger.error(f"API connection error: {str(e)}")
                    raise LLMConnectionError(f"Failed to connect to Anthropic API: {str(e)}")
                elif isinstance(e, anthropic.APIError):
                    self.logger.error(f"Anthropic API error: {str(e)}")
                    raise LLMError(f"Anthropic API error: {str(e)}")

            # Generic error handling
            self.logger.error(f"Error generating chat completion: {str(e)}")
            raise LLMError(f"Failed to generate chat completion: {str(e)}")

    async def generate_streaming_completion(
        self, prompt: str, task_type: LLMTask = LLMTask.GENERAL_COMPLETION, **kwargs
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

            # Convert prompt to messages format
            messages = [{"role": "user", "content": prompt}]

            # Prepare request parameters
            request_params = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
                "stream": True,
            }

            # Add optional parameters
            if self.top_p is not None:
                request_params["top_p"] = kwargs.get("top_p", self.top_p)
            if self.top_k is not None:
                request_params["top_k"] = kwargs.get("top_k", self.top_k)

            # Note: Structured output is not typically used with streaming
            # as it's harder to ensure valid JSON in chunks

            # Generate streaming completion
            stream = await self.client.messages.create(**request_params)

            # Yield chunks
            async for chunk in stream:
                if hasattr(chunk, "delta") and hasattr(chunk.delta, "text"):
                    yield chunk.delta.text
                elif hasattr(chunk, "content_block") and hasattr(chunk.content_block, "text"):
                    yield chunk.content_block.text

        except Exception as e:
            # Handle Anthropic-specific errors if the library is available
            if ANTHROPIC_AVAILABLE and anthropic:
                if isinstance(e, anthropic.RateLimitError):
                    self.logger.error(f"Rate limit error: {str(e)}")
                    raise LLMRateLimitError(f"Rate limit exceeded: {str(e)}")
                elif isinstance(e, anthropic.APIConnectionError):
                    self.logger.error(f"API connection error: {str(e)}")
                    raise LLMConnectionError(f"Failed to connect to Anthropic API: {str(e)}")
                elif isinstance(e, anthropic.APIError):
                    self.logger.error(f"Anthropic API error: {str(e)}")
                    raise LLMError(f"Anthropic API error: {str(e)}")

            # Generic error handling
            self.logger.error(f"Error generating streaming completion: {str(e)}")
            raise LLMError(f"Failed to generate streaming completion: {str(e)}")

    async def extract_knowledge_units(
        self, text: str, source_info: Optional[Dict[str, Any]] = None
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
                temperature=0.1,  # Lower temperature for more consistent extraction
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
                self.logger.error(
                    f"Failed to parse knowledge extraction response as JSON: {str(e)}"
                )
                self.logger.debug(f"Raw response: {response.content}")
                return []

        except Exception as e:
            self.logger.error(f"Error extracting knowledge units: {str(e)}")
            raise LLMError(f"Failed to extract knowledge units: {str(e)}")

    def _create_knowledge_extraction_prompt(
        self, raw_text: str, source_info: Optional[Dict[str, Any]] = None
    ) -> str:
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
        self, entities: List[str], context: Optional[str] = None
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
                temperature=0.1,  # Lower temperature for more consistent detection
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
                    if (
                        isinstance(rel, dict)
                        and "source" in rel
                        and "target" in rel
                        and "relationship_type" in rel
                    ):
                        valid_relationships.append(rel)

                self.logger.info(
                    f"Detected {len(valid_relationships)} relationships between entities"
                )
                return valid_relationships

            except json.JSONDecodeError as e:
                self.logger.error(
                    f"Failed to parse relationship detection response as JSON: {str(e)}"
                )
                self.logger.debug(f"Raw response: {response.content}")
                return []

        except Exception as e:
            self.logger.error(f"Error detecting relationships: {str(e)}")
            raise LLMError(f"Failed to detect relationships: {str(e)}")

    def _create_relationship_detection_prompt(
        self, entities: List[str], context: Optional[str] = None
    ) -> str:
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
        self, query: str, context: Optional[str] = None
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
                temperature=0.1,  # Lower temperature for more consistent parsing
            )

            # Parse JSON response
            try:
                parsed_query = json.loads(response.content)

                # Validate the response structure
                if not isinstance(parsed_query, dict):
                    raise LLMValidationError("Invalid query parsing response structure")

                # Set default values for required fields
                parsed_query.setdefault("intent", "search")
                parsed_query.setdefault("entities", [])
                parsed_query.setdefault("relationships", [])
                parsed_query.setdefault("constraints", [])
                parsed_query.setdefault("query_type", "natural_language")
                parsed_query.setdefault("confidence", 0.5)

                self.logger.info(f"Parsed query with intent: {parsed_query.get('intent')}")
                return parsed_query

            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse query parsing response as JSON: {str(e)}")
                self.logger.debug(f"Raw response: {response.content}")
                # Return default structure
                return {
                    "intent": "search",
                    "entities": [],
                    "relationships": [],
                    "constraints": [],
                    "query_type": "natural_language",
                    "confidence": 0.0,
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

    async def validate_content(self, content: str, criteria: List[str]) -> Dict[str, Any]:
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
                temperature=0.1,  # Lower temperature for more consistent validation
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
                    "valid": False,
                    "criteria_results": {criterion: False for criterion in criteria},
                    "overall_score": 0.0,
                    "errors": [f"Failed to parse validation response: {str(e)}"],
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
        Perform a health check on the Anthropic provider.

        Returns:
            Health status information
        """
        health_status = {
            "provider": "anthropic",
            "model": self.model_name,
            "connected": False,
            "api_key_valid": bool(self.api_key),
            "test_passed": False,
            "error": None,
            "timestamp": None,
            "response_time": None,
        }

        try:
            start_time = time.time()

            # Test with a simple completion
            test_response = await self.generate_completion(
                "Test connection. Respond with 'OK'.", LLMTask.GENERAL_COMPLETION
            )

            response_time = time.time() - start_time

            if test_response and test_response.content:
                health_status.update(
                    {
                        "connected": True,
                        "test_passed": True,
                        "response_time": response_time,
                        "timestamp": time.time(),
                    }
                )
            else:
                health_status["error"] = "Empty response from test request"

        except Exception as e:
            health_status["error"] = str(e)
            self.logger.error(f"Anthropic health check failed: {str(e)}")

        return health_status

    def get_supported_tasks(self) -> List[LLMTask]:
        """
        Get list of tasks supported by Anthropic provider.

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
            LLMTask.CONTENT_VALIDATION,
        ]

    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about this provider.

        Returns:
            Provider information dictionary
        """
        return {
            "name": "AnthropicLLMProvider",
            "provider": "anthropic",
            "model": self.model_name,
            "type": "api",
            "supported_tasks": [task.value for task in self.get_supported_tasks()],
            "connected": self.is_connected,
            "config": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "timeout": self.timeout,
                "top_p": self.top_p,
                "top_k": self.top_k,
            },
            "features": [
                "streaming",
                "chat_completion",
                "system_messages",
                "knowledge_extraction",
                "relationship_detection",
                "query_parsing",
            ],
        }

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for given text using Anthropic's approximation.

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        # Anthropic uses roughly 3.5 characters per token for English text
        return int(len(text) / 3.5)
