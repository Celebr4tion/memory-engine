#!/usr/bin/env python3
"""
Example demonstrating the Ollama embedding provider.

This example shows how to use the OllamaEmbeddingProvider for local embedding generation.
"""

import asyncio
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_core.embeddings.providers.ollama import OllamaEmbeddingProvider
from memory_core.embeddings.interfaces import TaskType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    """Main example function."""
    
    # Configuration for Ollama provider
    config = {
        'model_name': 'nomic-embed-text',  # Use your preferred Ollama embedding model
        'base_url': 'http://localhost:11434',
        'max_batch_size': 32,
        'timeout': 60,
        'keep_alive': '5m'
    }
    
    print("🦙 Ollama Embedding Provider Example")
    print("=" * 50)
    
    try:
        # Initialize the provider
        print("1. Initializing Ollama embedding provider...")
        provider = OllamaEmbeddingProvider(config)
        print(f"   Provider: {provider}")
        
        # Check if Ollama server is available
        print("\n2. Checking Ollama server availability...")
        if not provider.is_available():
            print("   ❌ Ollama server is not available!")
            print("   Please ensure:")
            print("   - Ollama is installed and running")
            print("   - Server is accessible at", config['base_url'])
            print("   - The model is available (run: ollama pull nomic-embed-text)")
            return
        print("   ✅ Ollama server is available")
        
        # Test connection
        print("\n3. Testing connection...")
        connection_ok = await provider.test_connection()
        if not connection_ok:
            print("   ❌ Connection test failed!")
            print("   Please check:")
            print("   - Model is available (run: ollama list)")
            print("   - Model name matches configuration")
            return
        print("   ✅ Connection test successful")
        
        # Get available models
        print("\n4. Getting available models...")
        available_models = await provider.get_available_models()
        if available_models:
            print(f"   Available models: {', '.join(available_models)}")
        else:
            print("   No models found or unable to retrieve list")
        
        # Get model info
        print(f"\n5. Getting info for model '{provider.model_name}'...")
        model_info = await provider.get_model_info()
        if model_info:
            print(f"   Model size: {model_info.get('size', 'Unknown')}")
            print(f"   Model family: {model_info.get('details', {}).get('family', 'Unknown')}")
        
        # Generate single embedding
        print(f"\n6. Generating single embedding...")
        print(f"   Model: {provider.model_name}")
        print(f"   Dimension: {provider.dimension}")
        
        test_text = "The quick brown fox jumps over the lazy dog."
        embedding = await provider.generate_embedding(test_text, TaskType.SEMANTIC_SIMILARITY)
        
        print(f"   Text: '{test_text}'")
        print(f"   Embedding shape: {embedding.shape}")
        print(f"   Embedding type: {embedding.dtype}")
        print(f"   Sample values: {embedding[:5]}")
        
        # Generate multiple embeddings
        print(f"\n7. Generating batch embeddings...")
        test_texts = [
            "Machine learning is transforming technology.",
            "Natural language processing enables computers to understand text.",
            "Deep learning models require large amounts of data.",
            "Artificial intelligence is revolutionizing various industries."
        ]
        
        embeddings = await provider.generate_embeddings(test_texts, TaskType.RETRIEVAL_DOCUMENT)
        
        print(f"   Number of texts: {len(test_texts)}")
        print(f"   Number of embeddings: {len(embeddings)}")
        print(f"   Embedding dimensions: {[emb.shape for emb in embeddings]}")
        
        # Calculate similarity between first two embeddings
        if len(embeddings) >= 2:
            import numpy as np
            
            # Normalize embeddings
            norm_emb1 = embeddings[0] / np.linalg.norm(embeddings[0])
            norm_emb2 = embeddings[1] / np.linalg.norm(embeddings[1])
            
            # Calculate cosine similarity
            similarity = np.dot(norm_emb1, norm_emb2)
            
            print(f"\n8. Similarity analysis:")
            print(f"   Text 1: '{test_texts[0]}'")
            print(f"   Text 2: '{test_texts[1]}'")
            print(f"   Cosine similarity: {similarity:.4f}")
        
        # Test different task types
        print(f"\n9. Testing different task types...")
        supported_types = provider.get_supported_task_types()
        print(f"   Supported task types: {[t.value for t in supported_types]}")
        
        for task_type in [TaskType.SEMANTIC_SIMILARITY, TaskType.RETRIEVAL_QUERY]:
            emb = await provider.generate_embedding("Test text", task_type)
            print(f"   {task_type.value}: shape {emb.shape}")
        
        print(f"\n✅ Ollama embedding provider example completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())