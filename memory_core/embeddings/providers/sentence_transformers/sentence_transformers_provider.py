"""
Sentence Transformers embedding provider implementation.

This module implements the EmbeddingProviderInterface for the sentence-transformers library,
providing local embedding generation with various pre-trained models.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None
    torch = None

from memory_core.embeddings.interfaces import (
    EmbeddingProviderInterface, 
    TaskType, 
    EmbeddingProviderError,
    EmbeddingConfigError
)


class SentenceTransformersProvider(EmbeddingProviderInterface):
    """
    Sentence Transformers embedding provider.
    
    Provides local embedding generation using pre-trained sentence transformer models.
    Supports both CPU and GPU execution with efficient batch processing.
    """

    # Popular model configurations with their dimensions
    MODEL_CONFIGS = {
        'all-MiniLM-L6-v2': {'dimension': 384, 'max_seq_length': 256},
        'all-mpnet-base-v2': {'dimension': 768, 'max_seq_length': 384},
        'all-MiniLM-L12-v2': {'dimension': 384, 'max_seq_length': 256},
        'all-distilroberta-v1': {'dimension': 768, 'max_seq_length': 512},
        'all-roberta-large-v1': {'dimension': 1024, 'max_seq_length': 512},
        'paraphrase-MiniLM-L6-v2': {'dimension': 384, 'max_seq_length': 128},
        'paraphrase-mpnet-base-v2': {'dimension': 768, 'max_seq_length': 384},
        'multi-qa-MiniLM-L6-cos-v1': {'dimension': 384, 'max_seq_length': 512},
        'multi-qa-mpnet-base-cos-v1': {'dimension': 768, 'max_seq_length': 512},
        'msmarco-distilbert-base-v4': {'dimension': 768, 'max_seq_length': 512}
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Sentence Transformers embedding provider.
        
        Args:
            config: Configuration dictionary with keys:
                - model_name: Model name (default: 'all-MiniLM-L6-v2')
                - device: Device to use ('cpu', 'cuda', 'auto') (default: 'auto')
                - max_batch_size: Maximum batch size (default: 64)
                - trust_remote_code: Whether to trust remote code (default: False)
                - normalize_embeddings: Whether to normalize embeddings (default: True)
                - cache_folder: Custom cache folder for models (optional)
        """
        super().__init__(config)
        
        self.logger = logging.getLogger(__name__)
        
        # Check if sentence-transformers is available
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise EmbeddingProviderError(
                "sentence-transformers library is not installed. "
                "Install it with: pip install sentence-transformers",
                provider="sentence_transformers",
                details={"missing_dependency": "sentence-transformers"}
            )
        
        # Extract configuration
        self._model_name = config.get('model_name', 'all-MiniLM-L6-v2')
        self.device = self._determine_device(config.get('device', 'auto'))
        self._max_batch_size = config.get('max_batch_size', 64)
        self.trust_remote_code = config.get('trust_remote_code', False)
        self.normalize_embeddings_flag = config.get('normalize_embeddings', True)
        self.cache_folder = config.get('cache_folder', None)
        
        # Model instance (lazy loaded)
        self._model: Optional[SentenceTransformer] = None
        self._dimension: Optional[int] = None
        
        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sentence_transformers")
        
        self.logger.info(f"Initialized SentenceTransformers provider with model: {self._model_name}, device: {self.device}")

    def _determine_device(self, device_config: str) -> str:
        """Determine the best device to use."""
        if device_config == 'auto':
            if torch and torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        elif device_config == 'cuda':
            if not torch or not torch.cuda.is_available():
                self.logger.warning("CUDA requested but not available, falling back to CPU")
                return 'cpu'
            return 'cuda'
        else:
            return 'cpu'

    def _load_model(self) -> SentenceTransformer:
        """Load the sentence transformer model (lazy loading)."""
        if self._model is None:
            try:
                self.logger.info(f"Loading SentenceTransformer model: {self._model_name}")
                
                # Model loading parameters
                model_kwargs = {
                    'device': self.device,
                    'trust_remote_code': self.trust_remote_code
                }
                
                if self.cache_folder:
                    model_kwargs['cache_folder'] = self.cache_folder
                
                # Load the model
                self._model = SentenceTransformer(
                    self._model_name,
                    **model_kwargs
                )
                
                # Auto-detect dimension
                self._dimension = self._detect_dimension()
                
                self.logger.info(f"Loaded model with dimension: {self._dimension}")
                
            except Exception as e:
                raise EmbeddingProviderError(
                    f"Failed to load SentenceTransformer model '{self._model_name}': {str(e)}",
                    provider="sentence_transformers",
                    details={
                        "model_name": self._model_name,
                        "device": self.device,
                        "error": str(e)
                    }
                )
        
        return self._model

    def _detect_dimension(self) -> int:
        """Auto-detect the embedding dimension of the loaded model."""
        if self._model is None:
            raise EmbeddingProviderError(
                "Model must be loaded before detecting dimension",
                provider="sentence_transformers"
            )
        
        # Check if we have a known configuration
        if self._model_name in self.MODEL_CONFIGS:
            return self.MODEL_CONFIGS[self._model_name]['dimension']
        
        # Try to get dimension from model
        try:
            # Get dimension from the model's pooling layer
            pooling_layer = self._model._modules.get('1')  # Pooling is usually the second module
            if hasattr(pooling_layer, 'pooling_output_dimension'):
                return pooling_layer.pooling_output_dimension
            
            # Fallback: generate a test embedding
            test_embedding = self._model.encode("test", convert_to_numpy=True)
            return len(test_embedding)
            
        except Exception as e:
            self.logger.warning(f"Could not auto-detect dimension, using default 384: {str(e)}")
            return 384

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        if self._dimension is None:
            # Load model to detect dimension
            self._load_model()
        return self._dimension

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    async def generate_embedding(
        self, 
        text: str, 
        task_type: TaskType = TaskType.SEMANTIC_SIMILARITY
    ) -> np.ndarray:
        """
        Generate embedding for a single text using SentenceTransformers.
        
        Args:
            text: Input text to embed
            task_type: Type of embedding task (ignored for sentence transformers)
            
        Returns:
            Embedding vector as numpy array
            
        Raises:
            EmbeddingProviderError: If embedding generation fails
        """
        if not text or not text.strip():
            raise EmbeddingProviderError(
                "Text cannot be empty or None",
                provider="sentence_transformers",
                details={"text_length": len(text) if text else 0}
            )
        
        try:
            self.logger.debug(f"Generating SentenceTransformers embedding for text: {text[:50]}...")
            
            # Run in thread pool to avoid blocking
            embedding = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                self._encode_single,
                text
            )
            
            self.logger.debug(f"Generated SentenceTransformers embedding of length {len(embedding)}")
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error generating SentenceTransformers embedding: {str(e)}")
            raise EmbeddingProviderError(
                f"Failed to generate SentenceTransformers embedding: {str(e)}",
                provider="sentence_transformers",
                details={
                    "model": self._model_name,
                    "device": self.device,
                    "text_preview": text[:100] if text else "",
                    "error": str(e)
                }
            )

    def _encode_single(self, text: str) -> np.ndarray:
        """Encode a single text (runs in thread pool)."""
        model = self._load_model()
        
        embedding = model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings_flag,
            show_progress_bar=False
        )
        
        return embedding.astype(np.float32)

    async def generate_embeddings(
        self, 
        texts: List[str], 
        task_type: TaskType = TaskType.SEMANTIC_SIMILARITY
    ) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts to embed
            task_type: Type of embedding task (ignored for sentence transformers)
            
        Returns:
            List of embedding vectors as numpy arrays
            
        Raises:
            EmbeddingProviderError: If embedding generation fails
        """
        if not texts:
            return []
        
        try:
            # Filter out empty texts and keep track of indices
            valid_texts = []
            valid_indices = []
            
            for i, text in enumerate(texts):
                if text and text.strip():
                    valid_texts.append(text)
                    valid_indices.append(i)
            
            if not valid_texts:
                # All texts were empty, return zero vectors
                zero_embedding = np.zeros(self.dimension, dtype=np.float32)
                return [zero_embedding.copy() for _ in texts]
            
            self.logger.debug(f"Generating SentenceTransformers embeddings for {len(valid_texts)} texts")
            
            # Process in batches
            all_embeddings = []
            for i in range(0, len(valid_texts), self.max_batch_size):
                batch = valid_texts[i:i + self.max_batch_size]
                
                # Run in thread pool to avoid blocking
                batch_embeddings = await asyncio.get_event_loop().run_in_executor(
                    self._executor,
                    self._encode_batch,
                    batch
                )
                
                all_embeddings.extend(batch_embeddings)
            
            # Create result list with proper ordering
            result = []
            valid_embedding_iter = iter(all_embeddings)
            zero_embedding = np.zeros(self.dimension, dtype=np.float32)
            
            for i in range(len(texts)):
                if i in valid_indices:
                    result.append(next(valid_embedding_iter))
                else:
                    # Use zero vector for empty texts
                    result.append(zero_embedding.copy())
            
            self.logger.debug(f"Generated {len(result)} SentenceTransformers embeddings")
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating SentenceTransformers embeddings: {str(e)}")
            raise EmbeddingProviderError(
                f"Failed to generate SentenceTransformers embeddings: {str(e)}",
                provider="sentence_transformers",
                details={
                    "model": self._model_name,
                    "device": self.device,
                    "num_texts": len(texts),
                    "error": str(e)
                }
            )

    def _encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Encode a batch of texts (runs in thread pool)."""
        model = self._load_model()
        
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings_flag,
            show_progress_bar=False,
            batch_size=min(len(texts), self.max_batch_size)
        )
        
        # Convert to list of numpy arrays
        if len(embeddings.shape) == 1:
            # Single embedding
            return [embeddings.astype(np.float32)]
        else:
            # Multiple embeddings
            return [emb.astype(np.float32) for emb in embeddings]

    def is_available(self) -> bool:
        """Check if the SentenceTransformers provider is available."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return False
        
        try:
            # Try to load the model
            self._load_model()
            return True
        except Exception as e:
            self.logger.warning(f"SentenceTransformers provider not available: {str(e)}")
            return False

    async def test_connection(self) -> bool:
        """Test the SentenceTransformers provider."""
        try:
            # Test with a simple embedding request
            test_embedding = await self.generate_embedding(
                "Test connection", 
                TaskType.SEMANTIC_SIMILARITY
            )
            
            expected_dim = self.dimension
            if len(test_embedding) != expected_dim:
                self.logger.error(f"Dimension mismatch: expected {expected_dim}, got {len(test_embedding)}")
                return False
            
            self.logger.info("SentenceTransformers connection test successful")
            return True
            
        except Exception as e:
            self.logger.error(f"SentenceTransformers connection test failed: {str(e)}")
            return False

    def get_supported_task_types(self) -> List[TaskType]:
        """Get supported task types for SentenceTransformers."""
        # SentenceTransformers models are generally trained for similarity tasks
        # but can be used for various purposes
        return [
            TaskType.SEMANTIC_SIMILARITY,
            TaskType.RETRIEVAL_DOCUMENT,
            TaskType.RETRIEVAL_QUERY,
            TaskType.CLUSTERING
        ]

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        info = {
            "model_name": self._model_name,
            "device": self.device,
            "dimension": self.dimension,
            "max_batch_size": self.max_batch_size,
            "normalize_embeddings": self.normalize_embeddings_flag,
            "available": self.is_available()
        }
        
        # Add known model configuration if available
        if self._model_name in self.MODEL_CONFIGS:
            info["model_config"] = self.MODEL_CONFIGS[self._model_name]
        
        return info

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, '_executor') and self._executor:
            self._executor.shutdown(wait=False)