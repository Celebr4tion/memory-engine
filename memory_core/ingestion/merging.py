"""
Node merging module for detecting and merging similar knowledge nodes.

This module provides functionality for checking similarity between knowledge nodes
and intelligently merging similar content to avoid duplication in the knowledge graph.
"""

import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Union

from memory_core.embeddings.embedding_manager import EmbeddingManager
from memory_core.db.janusgraph_storage import JanusGraphStorage


def merge_or_create_node(content: str, node_data: Dict[str, Any], 
                        storage=None, embedding_manager=None,
                        similarity_threshold: float = 0.9,
                        top_k: int = 1) -> str:
    """
    Check if similar node exists and merge if needed, otherwise create new node.
    
    This function:
    1. Generates an embedding for the content
    2. Searches for similar nodes using vector similarity
    3. If a very similar node exists, merges data with it
    4. If no similar node exists, creates a new one
    
    Args:
        content: The text content of the node
        node_data: Dictionary with node properties
        storage: Optional JanusGraphStorage instance (if None, one will be created)
        embedding_manager: Optional EmbeddingManager instance (if None, one will be created)
        similarity_threshold: Threshold for considering nodes as similar (0.0-1.0)
        top_k: Number of similar nodes to retrieve
        
    Returns:
        The ID of the node (either existing merged node or newly created)
        
    Raises:
        RuntimeError: If node creation or merging fails
    """
    logger = logging.getLogger(__name__)
    
    # Create storage if not provided
    if storage is None:
        from memory_core.core.knowledge_engine import KnowledgeEngine
        engine = KnowledgeEngine(enable_versioning=True)
        engine.connect()
        storage = engine.storage
    
    # If embedding manager isn't available, just create a new node without similarity check
    if embedding_manager is None:
        logger.info("No embedding manager available, skipping similarity check")
        node_id = storage.create_node(node_data)
        return node_id
    
    try:
        # Generate embedding for the content
        embedding = embedding_manager.generate_embedding(content)
        
        # Search for similar nodes
        similar_nodes = embedding_manager.vector_store.search_embedding(embedding, top_k=top_k)
        
        # Check if we have a very similar node
        if similar_nodes and similar_nodes[0]["score"] >= similarity_threshold:
            # Found a similar node, merge with it
            similar_node_id = similar_nodes[0]["node_id"]
            logger.info(f"Found similar node {similar_node_id} with score {similar_nodes[0]['score']}")
            
            # Get the existing node data
            try:
                existing_node = storage.get_node(similar_node_id)
                merged_data = _merge_node_data(existing_node, node_data)
                
                # Update the existing node with merged data
                storage.update_node(similar_node_id, merged_data)
                logger.info(f"Updated existing node {similar_node_id} with merged data")
                
                return similar_node_id
            except Exception as e:
                logger.error(f"Error retrieving or updating similar node: {str(e)}")
                # Fall back to creating a new node in case of error
        
        # No similar node or error occurred, create a new node
        node_id = storage.create_node(node_data)
        
        # Store the embedding for the new node
        embedding_manager.vector_store.add_embedding(node_id, embedding)
        logger.info(f"Created new node {node_id} with embedding")
        
        return node_id
        
    except Exception as e:
        logger.error(f"Error in merge_or_create_node: {str(e)}")
        # As a fallback, try to create a new node without any similarity checks
        try:
            node_id = storage.create_node(node_data)
            logger.info(f"Created new node {node_id} as fallback (without embedding)")
            return node_id
        except Exception as inner_e:
            logger.error(f"Critical error creating fallback node: {str(inner_e)}")
            raise RuntimeError(f"Failed to create node: {str(inner_e)}")


def _merge_node_data(existing_data: Dict[str, Any], new_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Intelligently merge two node data dictionaries.
    
    Args:
        existing_data: Data from the existing node
        new_data: Data from the new node to be merged
        
    Returns:
        A dictionary with the merged data
    """
    merged = existing_data.copy()
    
    # For ratings, take the higher value as it might indicate more confidence
    if 'rating_richness' in new_data:
        merged['rating_richness'] = max(
            merged.get('rating_richness', 0.5),
            new_data['rating_richness']
        )
    
    if 'rating_truthfulness' in new_data:
        merged['rating_truthfulness'] = max(
            merged.get('rating_truthfulness', 0.5),
            new_data['rating_truthfulness']
        )
    
    if 'rating_stability' in new_data:
        merged['rating_stability'] = max(
            merged.get('rating_stability', 0.5),
            new_data['rating_stability']
        )
    
    # For tags, combine both sets
    if 'tags' in new_data and new_data['tags']:
        existing_tags = set(merged.get('tags', '').split(',')) if merged.get('tags') else set()
        new_tags = set(new_data['tags'].split(','))
        # Remove empty strings
        existing_tags = {t for t in existing_tags if t}
        new_tags = {t for t in new_tags if t}
        
        # Combine tags and create a comma-separated string
        merged['tags'] = ','.join(sorted(existing_tags.union(new_tags)))
    
    # For metadata, merge dictionaries
    if 'extra_metadata' in new_data and new_data['extra_metadata']:
        try:
            existing_metadata = json.loads(merged.get('extra_metadata', '{}'))
            new_metadata = json.loads(new_data['extra_metadata'])
            
            # Deep merge metadata
            merged_metadata = _deep_merge_dicts(existing_metadata, new_metadata)
            merged['extra_metadata'] = json.dumps(merged_metadata)
        except json.JSONDecodeError:
            # If there's an error parsing JSON, keep the existing metadata
            pass
    
    # For source details, append new information if not already present
    if 'source_details' in new_data and new_data['source_details']:
        if 'source_details' not in merged or not merged['source_details']:
            merged['source_details'] = new_data['source_details']
        else:
            # Combine source details
            existing_details = set(merged['source_details'].split('; '))
            new_details = set(new_data['source_details'].split('; '))
            merged['source_details'] = '; '.join(sorted(existing_details.union(new_details)))
    
    # Update the timestamp to indicate this is a merged/updated node
    merged['last_updated_timestamp'] = time.time()
    
    return merged


def _deep_merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """
    Deep merge two dictionaries, with values from dict2 taking precedence.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (its values override dict1 on conflict)
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            # For conflicting values, take the one from dict2
            result[key] = value
    
    return result