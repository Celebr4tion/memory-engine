"""
Embedding manager for handling text-to-vector transformations and storage.

This module provides functions to generate embeddings from text and store them
in the vector database, with optional integration with the knowledge graph.
"""
import logging
import os
from typing import List, Optional

from sentence_transformers import SentenceTransformer
from memory_core.embeddings.vector_store import VectorStoreMilvus

# Configure logging
logger = logging.getLogger(__name__)

# Default model for generating embeddings
DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Singleton for model instance to avoid loading it multiple times
_model_instance = None


def get_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """
    Get or initialize the embedding model.
    
    Args:
        model_name: Name of the sentence-transformers model to use
        
    Returns:
        The sentence transformer model instance
    """
    global _model_instance
    
    if _model_instance is None:
        try:
            logger.info(f"Loading embedding model: {model_name}")
            _model_instance = SentenceTransformer(model_name)
            logger.info(f"Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise
    
    return _model_instance


def generate_embedding(text: str, model_name: str = DEFAULT_MODEL) -> List[float]:
    """
    Generate an embedding vector from text.
    
    Args:
        text: The input text to convert to an embedding
        model_name: Name of the sentence-transformers model to use
        
    Returns:
        A list of floats representing the embedding vector
        
    Raises:
        Exception: If there's an error generating the embedding
    """
    try:
        # Get the model
        model = get_model(model_name)
        
        # Generate embedding
        embedding = model.encode(text)
        
        # Convert from numpy to list for consistent serialization
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise


def store_node_embedding(
    node_id: str, 
    text: str, 
    vector_store: Optional[VectorStoreMilvus] = None,
    graph_storage = None,
    model_name: str = DEFAULT_MODEL
) -> None:
    """
    Generate and store an embedding for a knowledge node.
    
    Args:
        node_id: ID of the knowledge node
        text: Text content to generate embedding from
        vector_store: Vector store instance, created if None
        graph_storage: Optional JanusGraph storage for updating node properties
        model_name: Name of the sentence-transformers model to use
        
    Raises:
        ConnectionError: If connecting to the vector store fails
        Exception: If there's an error storing the embedding
    """
    try:
        # Generate embedding
        embedding = generate_embedding(text, model_name)
        
        # Handle vector store: create new one or use the provided one
        connection_required = False
        
        if vector_store is None:
            # Create a new vector store with default settings
            host = os.environ.get("MILVUS_HOST", "localhost")
            port = int(os.environ.get("MILVUS_PORT", "19530"))
            logger.info(f"Creating vector store with host={host}, port={port}")
            
            vector_store = VectorStoreMilvus(host=host, port=port)
            connection_required = True
        elif not hasattr(vector_store, 'collection') or vector_store.collection is None:
            # Existing vector store that doesn't have an active connection
            connection_required = True
        
        # Connect if needed
        if connection_required and not vector_store.connect():
            logger.error("Failed to connect to Milvus vector store")
            raise ConnectionError("Failed to connect to Milvus vector store")
        
        # Store embedding in vector store
        vector_store.add_embedding(node_id, embedding)
        logger.info(f"Stored embedding for node {node_id} in vector store")
        
        # Optionally update the node in JanusGraph to indicate embedding exists
        if graph_storage is not None:
            try:
                # Update node property
                graph_storage.update_node(node_id, {"embedding_exists": True})
                logger.info(f"Updated node {node_id} in graph to mark embedding_exists=True")
            except Exception as e:
                logger.warning(f"Failed to update node in graph: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error storing node embedding: {str(e)}")
        raise


def search_similar_nodes(
    query_text: str,
    top_k: int = 5,
    vector_store: Optional[VectorStoreMilvus] = None,
    model_name: str = DEFAULT_MODEL
) -> List[str]:
    """
    Search for nodes with similar embeddings to the query text.
    
    Args:
        query_text: The text to search for similar nodes
        top_k: Number of similar nodes to return
        vector_store: Vector store instance, created if None
        model_name: Name of the sentence-transformers model to use
        
    Returns:
        List of node IDs of the most similar vectors
        
    Raises:
        ConnectionError: If connecting to the vector store fails
        Exception: If there's an error searching for similar nodes
    """
    try:
        # Generate embedding for query text
        query_embedding = generate_embedding(query_text, model_name)
        
        # Handle vector store: create new one or use the provided one
        connection_required = False
        
        if vector_store is None:
            # Create a new vector store with default settings
            host = os.environ.get("MILVUS_HOST", "localhost")
            port = int(os.environ.get("MILVUS_PORT", "19530"))
            logger.info(f"Creating vector store with host={host}, port={port}")
            
            vector_store = VectorStoreMilvus(host=host, port=port)
            connection_required = True
        elif not hasattr(vector_store, 'collection') or vector_store.collection is None:
            # Existing vector store that doesn't have an active connection
            connection_required = True
        
        # Connect if needed
        if connection_required and not vector_store.connect():
            logger.error("Failed to connect to Milvus vector store")
            raise ConnectionError("Failed to connect to Milvus vector store")
        
        # Search for similar nodes
        node_ids = vector_store.get_node_ids(query_embedding, top_k)
        logger.info(f"Found {len(node_ids)} similar nodes for query text")
        
        return node_ids
        
    except Exception as e:
        logger.error(f"Error searching for similar nodes: {str(e)}")
        raise 