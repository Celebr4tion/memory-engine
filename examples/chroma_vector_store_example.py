#!/usr/bin/env python3
"""
ChromaDB Vector Store Example

This example demonstrates how to use the ChromaDB vector store implementation
with the Memory Engine's modular embedding system.

Requirements:
    pip install chromadb

Usage:
    python examples/chroma_vector_store_example.py
"""

import asyncio
import numpy as np
from typing import List, Dict, Any

# Import the ChromaDB vector store
from memory_core.embeddings.vector_stores.chroma import ChromaVectorStore
from memory_core.embeddings.vector_stores.interfaces import MetricType


async def main():
    """Demonstrate ChromaDB vector store functionality."""

    print("ChromaDB Vector Store Example")
    print("=" * 50)

    # Configuration for ChromaDB
    config = {
        "path": None,  # Use in-memory database
        "collection_name": "example_collection",
        "dimension": 384,  # Smaller dimension for example
        "metric_type": "COSINE",
        "batch_size": 100,
    }

    print(f"Configuration: {config}")

    try:
        # Initialize ChromaDB vector store
        vector_store = ChromaVectorStore(config)
        print(f"✓ Initialized ChromaVectorStore")
        print(f"  - Collection: {vector_store.collection_name}")
        print(f"  - Dimension: {vector_store.dimension}")
        print(f"  - Metric: {vector_store.metric_type}")
        print(f"  - ChromaDB distance: {vector_store._chroma_distance}")

        # Connect to ChromaDB
        print("\n1. Connecting to ChromaDB...")
        await vector_store.connect()
        print(f"✓ Connected: {vector_store.is_connected}")

        # Create collection
        print("\n2. Creating collection...")
        await vector_store.create_collection(
            collection_name=config["collection_name"],
            dimension=config["dimension"],
            metric_type=MetricType.COSINE,
        )
        print("✓ Collection created/verified")

        # Generate sample vectors
        print("\n3. Adding sample vectors...")
        sample_vectors = [np.random.rand(config["dimension"]).astype(np.float32) for _ in range(10)]
        sample_ids = [f"doc_{i}" for i in range(10)]
        sample_metadata = [
            {"category": "science" if i % 2 == 0 else "technology", "index": i} for i in range(10)
        ]

        # Add vectors to store
        added_ids = await vector_store.add_vectors(
            vectors=sample_vectors, ids=sample_ids, metadata=sample_metadata
        )
        print(f"✓ Added {len(added_ids)} vectors")

        # Count vectors
        print("\n4. Counting vectors in collection...")
        count = await vector_store.count_vectors()
        print(f"✓ Collection contains {count} vectors")

        # Search for similar vectors
        print("\n5. Searching for similar vectors...")
        query_vector = np.random.rand(config["dimension"]).astype(np.float32)

        # Basic search
        search_results = await vector_store.search_vectors(query_vector=query_vector, top_k=3)
        print(f"✓ Found {len(search_results)} similar vectors:")
        for i, result in enumerate(search_results):
            print(f"  {i+1}. ID: {result['id']}, Score: {result['score']:.4f}")
            if "metadata" in result:
                print(f"      Metadata: {result['metadata']}")

        # Search with filter
        print("\n6. Searching with metadata filter...")
        filter_expr = {"category": "science"}
        filtered_results = await vector_store.search_vectors(
            query_vector=query_vector, top_k=5, filter_expr=filter_expr
        )
        print(f"✓ Found {len(filtered_results)} science vectors:")
        for result in filtered_results:
            print(f"  - ID: {result['id']}, Score: {result['score']:.4f}")

        # Get specific vector
        print("\n7. Retrieving specific vector...")
        vector_data = await vector_store.get_vector("doc_0")
        if vector_data:
            print(f"✓ Retrieved vector: {vector_data['id']}")
            print(f"  Vector shape: {vector_data['vector'].shape}")
            print(f"  Metadata: {vector_data.get('metadata', 'None')}")

        # Update vector metadata
        print("\n8. Updating vector metadata...")
        update_success = await vector_store.update_vector(
            vector_id="doc_0", metadata={"category": "updated", "status": "modified"}
        )
        print(f"✓ Update successful: {update_success}")

        # Verify update
        updated_vector = await vector_store.get_vector("doc_0")
        if updated_vector:
            print(f"  Updated metadata: {updated_vector.get('metadata', 'None')}")

        # Get collection info
        print("\n9. Getting collection information...")
        collection_info = await vector_store.get_collection_info()
        print("✓ Collection info:")
        for key, value in collection_info.items():
            print(f"  {key}: {value}")

        # Delete some vectors
        print("\n10. Deleting vectors...")
        delete_ids = ["doc_8", "doc_9"]
        deleted_count = await vector_store.delete_vectors(delete_ids)
        print(f"✓ Deleted {deleted_count} vectors")

        # Final count
        final_count = await vector_store.count_vectors()
        print(f"✓ Final count: {final_count} vectors")

        # Health check
        print("\n11. Performing health check...")
        health_status = await vector_store.health_check()
        print(f"✓ Health status: {health_status['status']}")

        # Disconnect
        print("\n12. Disconnecting...")
        await vector_store.disconnect()
        print(f"✓ Disconnected: {not vector_store.is_connected}")

        print("\n" + "=" * 50)
        print("ChromaDB Vector Store Example Completed Successfully!")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
