"""
Advanced extractor for transforming raw text into structured knowledge units.

This module processes raw text using LLMs to extract structured knowledge units
with content, tags, metadata, and source information.
"""

import json
import logging
import os
import time
from typing import List, Dict, Any, Optional

from google import genai

class AdvancedExtractor:
    """
    Extracts structured knowledge units from raw text using LLMs.
    
    This class leverages the Gemini API to parse raw text into discrete
    knowledge units, each containing structured information about the content.
    """
    
    def __init__(self):
        """
        Initialize the advanced extractor.
        
        Sets up the Gemini client for LLM interactions.
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize Gemini client
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-1.5-pro"
        
    def _create_prompt(self, raw_text: str) -> str:
        """
        Create a prompt for the LLM to extract knowledge units.
        
        Args:
            raw_text: The raw text to extract knowledge from
            
        Returns:
            A formatted prompt string
        """
        return f"""
You are an expert at transforming raw text into structured knowledge units.
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

Format your entire response as a valid JSON array of these objects.

Text input:
{raw_text}
"""

    def extract_knowledge_units(self, raw_text: str) -> List[Dict[str, Any]]:
        """
        Generate and extract knowledge units from raw text.
        
        Args:
            raw_text: The text to extract knowledge units from
            
        Returns:
            A list of dictionaries, each representing a knowledge unit with
            content, tags, and metadata fields
            
        Raises:
            RuntimeError: If extraction fails due to API or parsing errors
        """
        if not raw_text or not raw_text.strip():
            return []
            
        try:
            # Prepare prompt
            prompt = self._create_prompt(raw_text)
            
            # Call LLM API
            generation_config = {
                "temperature": 0.2,  # Low temperature for more deterministic responses
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,  # Allow long responses for complex texts
            }
            
            response = self.client.generate_content(
                model=self.model,
                contents=prompt,
                generation_config=generation_config
            )
            
            # Extract JSON from response
            try:
                response_text = response.text
                # Clean up the response to ensure it's valid JSON
                # Sometimes LLMs add markdown code block markers
                json_text = response_text.strip()
                if json_text.startswith("```json"):
                    json_text = json_text[7:]
                if json_text.startswith("```"):
                    json_text = json_text[3:]
                if json_text.endswith("```"):
                    json_text = json_text[:-3]
                json_text = json_text.strip()
                
                # Parse the JSON
                knowledge_units = json.loads(json_text)
                
                # Validate the response structure
                if not isinstance(knowledge_units, list):
                    return []
                
                # Filter out any malformed units
                valid_units = []
                for unit in knowledge_units:
                    if isinstance(unit, dict) and "content" in unit:
                        valid_units.append(unit)
                
                return valid_units
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
                self.logger.debug(f"Raw response: {response_text}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error extracting knowledge units: {str(e)}")
            raise RuntimeError(f"Failed to extract knowledge units: {str(e)}")


def process_extracted_units(units: List[Dict[str, Any]], source_label: str, 
                           storage=None, embedding_manager=None) -> List[str]:
    """
    Process extracted knowledge units and store them in the graph database.
    
    This function:
    1. Converts knowledge units into node data structures
    2. Uses merge_or_create_node to check for similar existing nodes
    3. Either merges with existing nodes or creates new ones
    4. Generates and stores embeddings for the nodes
    
    Args:
        units: List of knowledge units from extract_knowledge_units()
        source_label: Source identifier for the knowledge (e.g., "Wikipedia", "User Input")
        storage: Optional JanusGraphStorage instance (if None, one will be created)
        embedding_manager: Optional EmbeddingManager instance (if None, no embeddings will be stored)
        
    Returns:
        List of node IDs for the created or merged nodes
        
    Raises:
        RuntimeError: If node creation or embedding storage fails
    """
    from memory_core.db.janusgraph_storage import JanusGraphStorage
    from memory_core.ingestion.merging import merge_or_create_node
    
    logger = logging.getLogger(__name__)
    
    # Create storage if not provided
    if storage is None:
        from memory_core.core.knowledge_engine import KnowledgeEngine
        engine = KnowledgeEngine(enable_versioning=True)
        engine.connect()
        storage = engine.storage
    
    # Process each unit
    node_ids = []
    
    for unit in units:
        try:
            # Construct node data
            node_data = {
                'content': unit['content'],
                'source': source_label,  # Use the provided source_label
                'creation_timestamp': time.time(),
                'rating_richness': 0.5,  # Default values
                'rating_truthfulness': 0.5,
                'rating_stability': 0.5
            }
            
            # Add tags if present
            if 'tags' in unit and isinstance(unit['tags'], list):
                node_data['tags'] = ','.join(unit['tags'])
            
            # Add metadata if present
            if 'metadata' in unit and isinstance(unit['metadata'], dict):
                # Convert metadata to string for storage
                node_data['extra_metadata'] = json.dumps(unit['metadata'])
                
                # Use metadata to set ratings if available
                if 'importance' in unit['metadata']:
                    try:
                        node_data['rating_richness'] = float(unit['metadata']['importance'])
                    except (ValueError, TypeError):
                        pass
                        
                if 'confidence_level' in unit['metadata']:
                    try:
                        node_data['rating_truthfulness'] = float(unit['metadata']['confidence_level'])
                    except (ValueError, TypeError):
                        pass
            
            # Add source info if present
            if 'source' in unit and isinstance(unit['source'], dict):
                source_info = unit['source']
                source_details = []
                
                if source_info.get('type'):
                    source_details.append(f"Type: {source_info['type']}")
                if source_info.get('url'):
                    source_details.append(f"URL: {source_info['url']}")
                if source_info.get('reference'):
                    source_details.append(f"Ref: {source_info['reference']}")
                if source_info.get('page'):
                    source_details.append(f"Page: {source_info['page']}")
                
                if source_details:
                    node_data['source_details'] = '; '.join(source_details)
            
            # Create or merge the node using the merging module
            node_id = merge_or_create_node(
                content=unit['content'],
                node_data=node_data,
                storage=storage,
                embedding_manager=embedding_manager,
                similarity_threshold=0.92  # Slightly higher threshold for stricter matching
            )
            
            logger.info(f"Created or merged knowledge node with ID: {node_id}")
            node_ids.append(node_id)
            
        except Exception as e:
            logger.error(f"Error processing knowledge unit: {str(e)}")
            logger.debug(f"Problematic unit: {unit}")
    
    return node_ids


# Function to directly use the extractor
def extract_knowledge_units(raw_text: str) -> List[Dict[str, Any]]:
    """
    Extract structured knowledge units from raw text using LLM.
    
    This is a convenience function that creates an AdvancedExtractor instance
    and uses it to process the input text.
    
    Args:
        raw_text: The text to extract knowledge from
        
    Returns:
        A list of dictionaries, each containing 'content', 'tags', and 'metadata'
        
    Raises:
        RuntimeError: If extraction fails due to API or parsing errors
    """
    extractor = AdvancedExtractor()
    return extractor.extract_knowledge_units(raw_text)