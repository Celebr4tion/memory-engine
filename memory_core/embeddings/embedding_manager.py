"""
Embedding manager for generating and storing embeddings using Gemini API.
"""
import os
import logging
from typing import List

from google import genai
from memory_core.embeddings.vector_store import VectorStoreMilvus

class EmbeddingManager:
    """Manager for generating and storing embeddings using Gemini API."""
    
    def __init__(self, vector_store: VectorStoreMilvus):
        """
        Initialize the embedding manager.
        
        Args:
            vector_store: VectorStoreMilvus instance for storing embeddings
        """
        self.vector_store = vector_store
        self.logger = logging.getLogger(__name__)
        
        # Initialize Gemini client
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        self.client = genai.Client(api_key=api_key)
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding vector for the given text using Gemini API.
        
        Args:
            text: The text to generate embedding for
            
        Returns:
            List[float]: The embedding vector
            
        Raises:
            ValueError: If text is empty or None
            RuntimeError: If embedding generation fails
        """
        if not text:
            raise ValueError("Text cannot be empty or None")
        
        try:
            self.logger.info(f"Generating embedding for text: {text[:50]}...")
            result = self.client.models.embed_content(
                model='gemini-embedding-exp-03-07',
                contents=text,
                config={'task_type': 'SEMANTIC_SIMILARITY'}
            )
            
            if not result.embeddings:
                raise RuntimeError("No embeddings returned from Gemini API")
                
            embedding = result.embeddings[0]
            self.logger.info(f"Generated embedding of length {len(embedding)}")
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error generating embedding: {str(e)}")
            raise RuntimeError(f"Failed to generate embedding: {str(e)}")
    
    def store_node_embedding(self, node_id: str, text: str) -> None:
        """
        Generate and store an embedding for a node.
        
        Args:
            node_id: Unique identifier for the node
            text: The text content to generate embedding for
            
        Raises:
            ValueError: If node_id or text is empty/None
            RuntimeError: If embedding generation or storage fails
        """
        if not node_id:
            raise ValueError("node_id cannot be empty or None")
        if not text:
            raise ValueError("text cannot be empty or None")
            
        try:
            # Generate embedding
            embedding = self.generate_embedding(text)
            
            # Store in vector store
            self.vector_store.add_embedding(node_id, embedding)
            self.logger.info(f"Stored embedding for node {node_id}")
            
        except Exception as e:
            self.logger.error(f"Error storing embedding for node {node_id}: {str(e)}")
            raise RuntimeError(f"Failed to store embedding: {str(e)}")

    def search_similar_nodes(
        self,
        query_text: str,
        top_k: int = 5
    ) -> List[str]:
        """
        Search for nodes with similar embeddings to the query text.
        
        Args:
            query_text: The text to search for similar nodes
            top_k: Number of similar nodes to return
            
        Returns:
            List of node IDs of the most similar vectors
            
        Raises:
            RuntimeError: If there's an error searching for similar nodes
        """
        try:
            self.logger.info(f"Searching for similar nodes to query: {query_text[:50]}...")
            query_embedding = self.generate_embedding(query_text)
            
            node_ids = self.vector_store.get_node_ids(query_embedding, top_k)
            self.logger.info(f"Found {len(node_ids)} similar nodes for query text")
            
            return node_ids
            
        except Exception as e:
            self.logger.error(f"Error searching for similar nodes: {str(e)}")
            raise RuntimeError(f"Failed to search for similar nodes: {str(e)}") 