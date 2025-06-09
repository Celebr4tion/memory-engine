#!/usr/bin/env python3
"""
Basic usage examples for the Memory Engine.

This script demonstrates the fundamental operations:
- Creating and storing knowledge nodes
- Creating relationships between nodes
- Searching for similar content
- Updating node ratings

Prerequisites:
- GEMINI_API_KEY environment variable set
- JanusGraph and Milvus running (use docker-compose)
"""

import os
import sys
import time
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_core.core.knowledge_engine import KnowledgeEngine
from memory_core.model.knowledge_node import KnowledgeNode
from memory_core.model.relationship import Relationship
from memory_core.embeddings.vector_store import VectorStoreMilvus
from memory_core.embeddings.embedding_manager import EmbeddingManager


def check_prerequisites() -> bool:
    """Check if prerequisites are met."""
    print("🔍 Checking prerequisites...")
    
    # Check API key
    if not os.getenv('GEMINI_API_KEY'):
        print("❌ GEMINI_API_KEY environment variable not set")
        print("   Please set it: export GEMINI_API_KEY=your-api-key")
        return False
    print("✅ GEMINI_API_KEY is set")
    
    return True


def setup_components():
    """Set up the Memory Engine components."""
    print("\n🚀 Setting up Memory Engine components...")
    
    # Initialize Knowledge Engine
    engine = KnowledgeEngine(
        host="localhost",
        port=8182,
        enable_versioning=True,
        enable_snapshots=True
    )
    
    # Connect to JanusGraph
    if not engine.connect():
        print("❌ Failed to connect to JanusGraph")
        print("   Make sure JanusGraph is running on localhost:8182")
        return None, None
    print("✅ Connected to JanusGraph")
    
    # Setup vector store and embedding manager
    vector_store = VectorStoreMilvus(host="localhost", port=19530)
    embedding_manager = None
    
    if vector_store.connect():
        embedding_manager = EmbeddingManager(vector_store)
        print("✅ Connected to Milvus and initialized embedding manager")
    else:
        print("⚠️  Could not connect to Milvus - similarity search won't work")
        print("   Make sure Milvus is running on localhost:19530")
    
    return engine, embedding_manager


def example_1_basic_node_operations(engine: KnowledgeEngine):
    """Example 1: Basic node creation and retrieval."""
    print("\n" + "="*60)
    print("📝 Example 1: Basic Node Operations")
    print("="*60)
    
    # Create knowledge nodes
    nodes_data = [
        {
            "content": "Python is a high-level programming language known for its simplicity and readability.",
            "source": "Programming Knowledge Base",
            "rating_truthfulness": 0.95,
            "rating_richness": 0.8
        },
        {
            "content": "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
            "source": "AI Reference",
            "rating_truthfulness": 0.9,
            "rating_richness": 0.85
        },
        {
            "content": "Graph databases store data in nodes and edges, making them ideal for representing relationships.",
            "source": "Database Guide",
            "rating_truthfulness": 0.9,
            "rating_richness": 0.75
        }
    ]
    
    created_nodes = []
    
    for i, data in enumerate(nodes_data, 1):
        print(f"\n📌 Creating node {i}...")
        
        # Create knowledge node
        node = KnowledgeNode(
            content=data["content"],
            source=data["source"],
            rating_truthfulness=data["rating_truthfulness"],
            rating_richness=data["rating_richness"],
            rating_stability=0.7
        )
        
        # Save to graph
        node_id = engine.save_node(node)
        created_nodes.append(node_id)
        
        print(f"   ✅ Created node with ID: {node_id}")
        print(f"   📄 Content: {data['content'][:50]}...")
    
    # Retrieve and display nodes
    print(f"\n🔍 Retrieving created nodes...")
    for i, node_id in enumerate(created_nodes, 1):
        node = engine.get_node(node_id)
        print(f"\n   Node {i} (ID: {node_id}):")
        print(f"   📄 Content: {node.content}")
        print(f"   📊 Truthfulness: {node.rating_truthfulness:.2f}")
        print(f"   🎯 Richness: {node.rating_richness:.2f}")
        print(f"   📍 Source: {node.source}")
    
    return created_nodes


def example_2_relationships(engine: KnowledgeEngine, node_ids: List[str]):
    """Example 2: Creating relationships between nodes."""
    print("\n" + "="*60)
    print("🔗 Example 2: Creating Relationships")
    print("="*60)
    
    if len(node_ids) < 2:
        print("❌ Need at least 2 nodes to create relationships")
        return []
    
    # Create relationships
    relationships_data = [
        {
            "from_id": node_ids[0],  # Python
            "to_id": node_ids[1],    # Machine Learning
            "relation_type": "USED_IN",
            "confidence_score": 0.8
        },
        {
            "from_id": node_ids[1],  # Machine Learning
            "to_id": node_ids[2],    # Graph Databases
            "relation_type": "BENEFITS_FROM",
            "confidence_score": 0.7
        }
    ]
    
    created_relationships = []
    
    for i, rel_data in enumerate(relationships_data, 1):
        print(f"\n🔗 Creating relationship {i}...")
        
        # Create relationship
        relationship = Relationship(
            from_id=rel_data["from_id"],
            to_id=rel_data["to_id"],
            relation_type=rel_data["relation_type"],
            confidence_score=rel_data["confidence_score"],
            timestamp=time.time()
        )
        
        # Save to graph
        edge_id = engine.save_relationship(relationship)
        created_relationships.append(edge_id)
        
        print(f"   ✅ Created relationship with ID: {edge_id}")
        print(f"   🏷️  Type: {rel_data['relation_type']}")
        print(f"   📊 Confidence: {rel_data['confidence_score']}")
    
    # Display relationship network
    print(f"\n🕸️  Relationship Network:")
    for node_id in node_ids:
        node = engine.get_node(node_id)
        print(f"\n   📍 Node: {node.content[:40]}...")
        
        # Get outgoing relationships
        outgoing = engine.get_outgoing_relationships(node_id)
        for rel in outgoing:
            target_node = engine.get_node(rel.to_id)
            print(f"      ➡️  {rel.relation_type} → {target_node.content[:30]}...")
        
        # Get incoming relationships
        incoming = engine.get_incoming_relationships(node_id)
        for rel in incoming:
            source_node = engine.get_node(rel.from_id)
            print(f"      ⬅️  {source_node.content[:30]}... → {rel.relation_type}")
    
    return created_relationships


def example_3_embeddings_and_search(engine: KnowledgeEngine, embedding_manager: EmbeddingManager, node_ids: List[str]):
    """Example 3: Working with embeddings and similarity search."""
    print("\n" + "="*60)
    print("🔍 Example 3: Embeddings and Similarity Search")
    print("="*60)
    
    if not embedding_manager:
        print("❌ Embedding manager not available - skipping this example")
        return
    
    # Generate and store embeddings for existing nodes
    print("📊 Generating embeddings for existing nodes...")
    for i, node_id in enumerate(node_ids, 1):
        node = engine.get_node(node_id)
        print(f"\n   Processing node {i}...")
        
        try:
            # Generate and store embedding
            embedding_manager.store_node_embedding(node_id, node.content)
            print(f"   ✅ Generated embedding for: {node.content[:40]}...")
        except Exception as e:
            print(f"   ❌ Failed to generate embedding: {e}")
    
    # Search for similar content
    search_queries = [
        "programming languages for beginners",
        "artificial intelligence and deep learning",
        "database systems for storing relationships"
    ]
    
    print(f"\n🔍 Performing similarity searches...")
    for i, query in enumerate(search_queries, 1):
        print(f"\n   Query {i}: '{query}'")
        
        try:
            # Search for similar nodes
            similar_node_ids = embedding_manager.search_similar_nodes(query, top_k=3)
            
            if similar_node_ids:
                print(f"   📋 Found {len(similar_node_ids)} similar nodes:")
                for j, similar_id in enumerate(similar_node_ids, 1):
                    try:
                        similar_node = engine.get_node(similar_id)
                        print(f"      {j}. {similar_node.content[:50]}...")
                    except Exception as e:
                        print(f"      {j}. Error retrieving node {similar_id}: {e}")
            else:
                print("   📭 No similar nodes found")
        except Exception as e:
            print(f"   ❌ Search failed: {e}")


def example_4_rating_updates(engine: KnowledgeEngine, node_ids: List[str]):
    """Example 4: Updating node ratings based on evidence."""
    print("\n" + "="*60)
    print("📊 Example 4: Rating Updates")
    print("="*60)
    
    if not node_ids:
        print("❌ No nodes available for rating updates")
        return
    
    # Select first node for rating updates
    node_id = node_ids[0]
    node = engine.get_node(node_id)
    
    print(f"📍 Updating ratings for node: {node.content[:50]}...")
    print(f"   📊 Current ratings:")
    print(f"      Truthfulness: {node.rating_truthfulness:.2f}")
    print(f"      Richness: {node.rating_richness:.2f}")
    print(f"      Stability: {node.rating_stability:.2f}")
    
    # Simulate evidence updates
    evidence_scenarios = [
        {
            "name": "Positive confirmation",
            "evidence": {"confirmation": 0.3},
            "description": "Found supporting evidence for this knowledge"
        },
        {
            "name": "Richness improvement", 
            "evidence": {"richness": 0.2},
            "description": "Added more detailed information"
        },
        {
            "name": "Stability confirmation",
            "evidence": {"stability": 0.1},
            "description": "Information verified to be stable over time"
        }
    ]
    
    for i, scenario in enumerate(evidence_scenarios, 1):
        print(f"\n   📈 Scenario {i}: {scenario['name']}")
        print(f"      Description: {scenario['description']}")
        
        # Get current node state
        current_node = engine.get_node(node_id)
        old_truthfulness = current_node.rating_truthfulness
        old_richness = current_node.rating_richness
        old_stability = current_node.rating_stability
        
        # Apply rating update using the rating system
        from memory_core.rating.rating_system import update_rating
        
        try:
            result = update_rating(node_id, scenario["evidence"], engine.storage)
            
            if result["status"] == "success":
                print(f"      ✅ Update successful")
                
                # Show changes
                updated_node = engine.get_node(node_id)
                if "rating_truthfulness" in result["updates"]:
                    print(f"      📊 Truthfulness: {old_truthfulness:.2f} → {updated_node.rating_truthfulness:.2f}")
                if "rating_richness" in result["updates"]:
                    print(f"      📊 Richness: {old_richness:.2f} → {updated_node.rating_richness:.2f}")
                if "rating_stability" in result["updates"]:
                    print(f"      📊 Stability: {old_stability:.2f} → {updated_node.rating_stability:.2f}")
            else:
                print(f"      ⚠️  {result.get('message', 'No changes needed')}")
                
        except Exception as e:
            print(f"      ❌ Update failed: {e}")


def example_5_versioning(engine: KnowledgeEngine, node_ids: List[str]):
    """Example 5: Working with versioning and change tracking."""
    print("\n" + "="*60)
    print("🕐 Example 5: Versioning and Change Tracking")
    print("="*60)
    
    if not node_ids or not engine.revision_manager:
        print("❌ Versioning not available or no nodes to work with")
        return
    
    node_id = node_ids[0]
    node = engine.get_node(node_id)
    
    print(f"📍 Working with node: {node.content[:50]}...")
    
    # Get revision history
    print(f"\n📚 Getting revision history...")
    try:
        history = engine.get_revision_history("node", node_id)
        print(f"   📋 Found {len(history)} revision entries:")
        
        for i, revision in enumerate(history[:3], 1):  # Show first 3
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(revision['timestamp']))
            print(f"      {i}. {revision['change_type']} at {timestamp}")
    except Exception as e:
        print(f"   ❌ Failed to get revision history: {e}")
    
    # Create a snapshot
    print(f"\n📸 Creating snapshot...")
    try:
        snapshot_id = engine.create_snapshot()
        print(f"   ✅ Created snapshot with ID: {snapshot_id}")
    except Exception as e:
        print(f"   ❌ Failed to create snapshot: {e}")


def cleanup(engine: KnowledgeEngine, embedding_manager: EmbeddingManager):
    """Clean up connections."""
    print("\n🧹 Cleaning up...")
    
    if embedding_manager and embedding_manager.vector_store:
        try:
            embedding_manager.vector_store.disconnect()
            print("✅ Disconnected from Milvus")
        except Exception as e:
            print(f"⚠️  Error disconnecting from Milvus: {e}")
    
    if engine:
        try:
            engine.disconnect()
            print("✅ Disconnected from JanusGraph")
        except Exception as e:
            print(f"⚠️  Error disconnecting from JanusGraph: {e}")


def main():
    """Main function demonstrating all examples."""
    print("🌟 Memory Engine - Basic Usage Examples")
    print("="*60)
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Setup components
    engine, embedding_manager = setup_components()
    if not engine:
        sys.exit(1)
    
    try:
        # Run examples
        node_ids = example_1_basic_node_operations(engine)
        
        if node_ids:
            relationship_ids = example_2_relationships(engine, node_ids)
            example_3_embeddings_and_search(engine, embedding_manager, node_ids)
            example_4_rating_updates(engine, node_ids)
            example_5_versioning(engine, node_ids)
        
        print("\n" + "="*60)
        print("🎉 All examples completed successfully!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup(engine, embedding_manager)


if __name__ == "__main__":
    main()