"""
Embedding manager for generating and storing embeddings using Gemini API.
"""
import logging
from typing import List

from google import genai
from google.genai import types
from memory_core.embeddings.vector_store import VectorStoreMilvus
from memory_core.config import get_config

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
        self.config = get_config()
        
        # Initialize Gemini client using genai.Client
        api_key = self.config.config.api.gemini_api_key
        if not api_key:
            raise ValueError("GEMINI_API_KEY not configured. Set it via environment variable or configuration file.")
        self.client = genai.Client(api_key=api_key)
        self.embedding_model = self.config.config.embedding.model
    
    def generate_embedding(self, text: str, task_type: str = "SEMANTIC_SIMILARITY") -> List[float]:
        """
        Generate an embedding vector for the given text using Gemini API.
        
        Args:
            text: The text to generate embedding for
            task_type: The task type for embedding generation (e.g., SEMANTIC_SIMILARITY, RETRIEVAL_QUERY)
            
        Returns:
            List[float]: The embedding vector
            
        Raises:
            ValueError: If text is empty or None
            RuntimeError: If embedding generation fails
        """
        if not text:
            raise ValueError("Text cannot be empty or None")
        
        try:
            self.logger.info(f"Generating embedding for text (task: {task_type}): {text[:50]}...")
            # Generate embedding using typed EmbedContentConfig to match API expectations and tests
            result = self.client.models.embed_content(
                model=self.embedding_model,
                contents=text,
                config=types.EmbedContentConfig(task_type=task_type)
            )
            # Support different response structures
            if hasattr(result, 'embeddings'):
                # result.embeddings expected to be a list of embed outputs
                if isinstance(result.embeddings, list) and result.embeddings:
                    first_item = result.embeddings[0]
                    # list of floats
                    if isinstance(first_item, list):
                        embedding = first_item
                    # object with .values attribute
                    elif hasattr(first_item, 'values'):
                        embedding = first_item.values
                    else:
                        self.logger.error(f"Unexpected embeddings element type: {type(first_item)}")
                        raise RuntimeError("Unexpected embedding result structure received from API.")
                else:
                    self.logger.error(f"Unexpected embeddings structure: {result.embeddings}")
                    raise RuntimeError("Unexpected embedding result structure received from API.")
            elif hasattr(result, 'embedding') and hasattr(result.embedding, 'values'):
                embedding = result.embedding.values
            else:
                self.logger.error(f"Unexpected embedding result structure: {result}")
                raise RuntimeError("Unexpected embedding result structure received from API.")
            # Validate embedding format
            if not isinstance(embedding, list):
                self.logger.error(f"Final embedding format is not a list: {type(embedding)}")
                raise RuntimeError(f"Final embedding format is not a list: {type(embedding)}")
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
            # Generate embedding with RETRIEVAL_DOCUMENT task type
            embedding = self.generate_embedding(text, task_type="RETRIEVAL_DOCUMENT")
            
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
            # Generate query embedding with RETRIEVAL_QUERY task type
            query_embedding = self.generate_embedding(query_text, task_type="RETRIEVAL_QUERY")

            node_ids = self.vector_store.get_node_ids(query_embedding, top_k)
            self.logger.info(f"Found {len(node_ids)} similar nodes for query text")
            
            return node_ids
            
        except Exception as e:
            self.logger.error(f"Error searching for similar nodes: {str(e)}")
            raise RuntimeError(f"Failed to search for similar nodes: {str(e)}")