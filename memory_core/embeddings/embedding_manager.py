"""
Embedding manager for generating and storing embeddings using Gemini API.
"""
import hashlib
import logging
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import OrderedDict
import threading

from google import genai
from google.genai import types
from memory_core.embeddings.vector_store import VectorStoreMilvus
from memory_core.config import get_config


@dataclass
class EmbeddingCacheEntry:
    """Represents a cached embedding."""
    text_hash: str
    embedding: List[float]
    task_type: str
    created_at: float
    last_accessed: float
    access_count: int


class EmbeddingCache:
    """
    LRU cache for embeddings to avoid regenerating identical embeddings.
    """
    
    def __init__(self, max_entries: int = 1000, ttl_seconds: int = 3600):
        """Initialize embedding cache."""
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, EmbeddingCacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def get(self, text: str, task_type: str) -> Optional[List[float]]:
        """Get cached embedding if available and not expired."""
        cache_key = self._generate_key(text, task_type)
        
        with self._lock:
            if cache_key not in self._cache:
                self.misses += 1
                return None
            
            entry = self._cache[cache_key]
            
            # Check if expired
            if time.time() - entry.created_at > self.ttl_seconds:
                del self._cache[cache_key]
                self.misses += 1
                return None
            
            # Update access info and move to end
            entry.last_accessed = time.time()
            entry.access_count += 1
            self._cache.move_to_end(cache_key)
            
            self.hits += 1
            return entry.embedding.copy()
    
    def put(self, text: str, task_type: str, embedding: List[float]):
        """Store embedding in cache."""
        cache_key = self._generate_key(text, task_type)
        
        with self._lock:
            # Remove if exists
            if cache_key in self._cache:
                del self._cache[cache_key]
            
            # Evict LRU if at capacity
            while len(self._cache) >= self.max_entries:
                self._cache.popitem(last=False)
            
            # Add new entry
            entry = EmbeddingCacheEntry(
                text_hash=cache_key,
                embedding=embedding.copy(),
                task_type=task_type,
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=0
            )
            self._cache[cache_key] = entry
    
    def _generate_key(self, text: str, task_type: str) -> str:
        """Generate cache key from text and task type."""
        combined = f"{text.strip().lower()}||{task_type}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        total = self.hits + self.misses
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / total if total > 0 else 0.0,
            'entries': len(self._cache)
        }

class EmbeddingManager:
    """Manager for generating and storing embeddings using Gemini API."""
    
    def __init__(self, vector_store: VectorStoreMilvus, cache_size: int = 1000, cache_ttl: int = 3600):
        """
        Initialize the embedding manager.
        
        Args:
            vector_store: VectorStoreMilvus instance for storing embeddings
            cache_size: Maximum number of embeddings to cache
            cache_ttl: Cache time-to-live in seconds
        """
        self.vector_store = vector_store
        self.logger = logging.getLogger(__name__)
        self.config = get_config()
        
        # Initialize embedding cache
        self.cache = EmbeddingCache(max_entries=cache_size, ttl_seconds=cache_ttl)
        
        # Initialize Gemini client using genai.Client
        api_key = self.config.config.api.google_api_key
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not configured. Set it via environment variable or configuration file.")
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
        
        # Check cache first
        cached_embedding = self.cache.get(text, task_type)
        if cached_embedding is not None:
            self.logger.debug(f"Using cached embedding for text: {text[:50]}...")
            return cached_embedding
        
        try:
            self.logger.info(f"Generating embedding for text (task: {task_type}): {text[:50]}...")
            # Generate embedding using typed EmbedContentConfig to match API expectations and tests
            embedding_config = types.EmbedContentConfig(task_type=task_type)
            
            # Add dimension specification for gemini-embedding-exp models
            if "gemini-embedding-exp" in self.embedding_model:
                embedding_config.output_dimensionality = self.config.config.embedding.dimension
            
            result = self.client.models.embed_content(
                model=self.embedding_model,
                contents=text,
                config=embedding_config
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
            
            # Cache the embedding
            self.cache.put(text, task_type, embedding)
            
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
    
    def generate_embeddings(self, texts: List[str], task_type: str = "SEMANTIC_SIMILARITY") -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of texts to generate embeddings for
            task_type: The task type for embedding generation
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        cache_hits = 0
        
        for text in texts:
            try:
                embedding = self.generate_embedding(text, task_type)
                embeddings.append(embedding)
                
                # Check if this was a cache hit
                cached = self.cache.get(text, task_type)
                if cached is not None:
                    cache_hits += 1
                    
            except Exception as e:
                self.logger.error(f"Failed to generate embedding for text: {text[:50]}... Error: {e}")
                # Use zero vector as fallback with configured dimension
                embeddings.append([0.0] * self.config.config.embedding.dimension)
        
        self.logger.info(f"Generated {len(embeddings)} embeddings with {cache_hits} cache hits")
        return embeddings
    
    def find_similar_nodes(self, query_embedding: List[float], top_k: int = 5, similarity_threshold: float = 0.7) -> List[Tuple[str, float]]:
        """
        Find similar nodes using the vector store.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of similar nodes to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of (node_id, similarity_score) tuples
        """
        try:
            node_ids = self.vector_store.get_node_ids(query_embedding, top_k)
            # Return with mock similarity scores for now - would need vector store enhancement
            return [(node_id, 0.8) for node_id in node_ids]
        except Exception as e:
            self.logger.error(f"Error finding similar nodes: {e}")
            return []
    
    def get_cache_statistics(self) -> Dict[str, float]:
        """Get embedding cache performance statistics."""
        return self.cache.get_stats()
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self.cache = EmbeddingCache(
            max_entries=self.cache.max_entries, 
            ttl_seconds=self.cache.ttl_seconds
        )
        self.logger.info("Embedding cache cleared")