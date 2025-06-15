"""
Advanced extractor for transforming raw text into structured knowledge units.

This module processes raw text using LLMs to extract structured knowledge units
with content, tags, metadata, and source information.
"""

import json
import logging
import time
from typing import List, Dict, Any, Optional

from memory_core.config import get_config
from memory_core.llm.providers.gemini import GeminiLLMProvider

class AdvancedExtractor:
    """
    Extracts structured knowledge units from raw text using LLMs.
    
    This class leverages the modular LLM provider system to parse raw text into discrete
    knowledge units, each containing structured information about the content.
    """
    
    def __init__(self):
        """
        Initialize the advanced extractor.
        
        Sets up the LLM provider for knowledge extraction.
        """
        self.logger = logging.getLogger(__name__)
        self.config = get_config()
        
        # Initialize LLM provider
        api_key = self.config.config.api.google_api_key
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not configured. Set it via environment variable or configuration file.")
        
        llm_config = {
            'api_key': api_key,
            'model_name': self.config.config.llm.model,
            'temperature': self.config.config.llm.temperature,
            'max_tokens': self.config.config.llm.max_tokens,
            'timeout': self.config.config.llm.timeout,
        }
        
        self.llm_provider = GeminiLLMProvider(llm_config)
    

    async def extract_knowledge_units(self, raw_text: str) -> List[Dict[str, Any]]:
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
            # Ensure LLM provider is connected
            if not self.llm_provider.is_connected:
                await self.llm_provider.connect()
            
            # Use the LLM provider's knowledge extraction method
            knowledge_units = await self.llm_provider.extract_knowledge_units(raw_text)
            
            self.logger.info(f"Extracted {len(knowledge_units)} knowledge units from text")
            return knowledge_units
            
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
async def extract_knowledge_units(raw_text: str) -> List[Dict[str, Any]]:
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
    return await extractor.extract_knowledge_units(raw_text)