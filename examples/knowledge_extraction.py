#!/usr/bin/env python3
"""
Knowledge extraction examples for the Memory Engine.

This script demonstrates:
- Extracting knowledge from various text sources
- Processing and storing extracted knowledge
- Creating automatic relationships
- Merging similar knowledge nodes

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
from memory_core.ingestion.advanced_extractor import extract_knowledge_units, process_extracted_units
from memory_core.ingestion.relationship_extractor import analyze_and_create_relationships
from memory_core.embeddings.vector_store import VectorStoreMilvus
from memory_core.embeddings.embedding_manager import EmbeddingManager


def setup_system():
    """Set up the Memory Engine system."""
    print("🚀 Setting up Memory Engine for knowledge extraction...")
    
    # Check API key
    if not os.getenv('GEMINI_API_KEY'):
        print("❌ GEMINI_API_KEY environment variable not set")
        return None, None
    
    # Initialize Knowledge Engine
    engine = KnowledgeEngine(enable_versioning=True, enable_snapshots=True)
    if not engine.connect():
        print("❌ Failed to connect to JanusGraph")
        return None, None
    print("✅ Connected to JanusGraph")
    
    # Setup vector store and embedding manager
    vector_store = VectorStoreMilvus()
    embedding_manager = None
    
    if vector_store.connect():
        embedding_manager = EmbeddingManager(vector_store)
        print("✅ Connected to Milvus")
    else:
        print("⚠️  Could not connect to Milvus - continuing without similarity features")
    
    return engine, embedding_manager


def example_1_simple_text_extraction():
    """Example 1: Extract knowledge from simple text."""
    print("\n" + "="*70)
    print("📝 Example 1: Simple Text Knowledge Extraction")
    print("="*70)
    
    # Sample text about artificial intelligence
    text = """
    Artificial Intelligence (AI) is a branch of computer science that aims to create 
    intelligent machines capable of performing tasks that typically require human intelligence.
    Machine learning is a subset of AI that uses statistical techniques to enable machines 
    to improve their performance on a specific task through experience. Deep learning is 
    a subset of machine learning that uses neural networks with multiple layers to model 
    and understand complex patterns in data. Natural language processing (NLP) is another 
    important area of AI that focuses on the interaction between computers and human language.
    """
    
    print(f"📄 Input text:")
    print(f"   {text.strip()}")
    
    print(f"\n🧠 Extracting knowledge units...")
    try:
        # Extract knowledge units
        units = extract_knowledge_units(text)
        
        print(f"✅ Extracted {len(units)} knowledge units:")
        for i, unit in enumerate(units, 1):
            print(f"\n   Unit {i}:")
            print(f"   📄 Content: {unit['content']}")
            if 'tags' in unit:
                print(f"   🏷️  Tags: {', '.join(unit['tags'])}")
            if 'metadata' in unit:
                confidence = unit['metadata'].get('confidence_level', 'N/A')
                domain = unit['metadata'].get('domain', 'N/A')
                print(f"   📊 Confidence: {confidence}, Domain: {domain}")
        
        return units
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return []


def example_2_scientific_text_extraction():
    """Example 2: Extract knowledge from scientific text."""
    print("\n" + "="*70)
    print("🔬 Example 2: Scientific Text Knowledge Extraction")
    print("="*70)
    
    # Sample scientific text about quantum computing
    text = """
    Quantum computing is a type of computation that harnesses the collective properties 
    of quantum states, such as superposition, interference, and entanglement, to perform 
    calculations. A quantum computer uses quantum bits or qubits, which can exist in 
    multiple states simultaneously, unlike classical bits that are either 0 or 1. 
    Quantum algorithms, such as Shor's algorithm for factoring large numbers and Grover's 
    algorithm for searching unsorted databases, demonstrate potential advantages over 
    classical algorithms. However, quantum computers are highly sensitive to environmental 
    noise and require extremely low temperatures to maintain quantum coherence.
    """
    
    print(f"📄 Input scientific text:")
    print(f"   {text.strip()}")
    
    print(f"\n🧠 Extracting knowledge units...")
    try:
        units = extract_knowledge_units(text)
        
        print(f"✅ Extracted {len(units)} knowledge units:")
        for i, unit in enumerate(units, 1):
            print(f"\n   Unit {i}:")
            print(f"   📄 Content: {unit['content']}")
            
            # Show metadata details for scientific content
            if 'metadata' in unit:
                metadata = unit['metadata']
                print(f"   📊 Metadata:")
                for key, value in metadata.items():
                    print(f"      {key}: {value}")
        
        return units
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return []


def example_3_process_and_store_knowledge(engine: KnowledgeEngine, embedding_manager: EmbeddingManager):
    """Example 3: Process and store extracted knowledge."""
    print("\n" + "="*70)
    print("💾 Example 3: Process and Store Knowledge")
    print("="*70)
    
    # Sample text about climate change
    text = """
    Climate change refers to long-term shifts in global or regional climate patterns. 
    The primary cause of recent climate change is human activities, particularly the 
    emission of greenhouse gases like carbon dioxide from burning fossil fuels. 
    The greenhouse effect occurs when these gases trap heat in Earth's atmosphere, 
    leading to global warming. Rising temperatures cause sea level rise, melting 
    ice caps, and more frequent extreme weather events. Renewable energy sources 
    like solar and wind power are crucial for reducing greenhouse gas emissions.
    """
    
    print(f"📄 Processing text about climate change...")
    
    try:
        # Extract knowledge units
        print("🧠 Step 1: Extracting knowledge units...")
        units = extract_knowledge_units(text)
        print(f"   ✅ Extracted {len(units)} units")
        
        # Process and store units
        print("💾 Step 2: Processing and storing units...")
        node_ids = process_extracted_units(
            units=units,
            source_label="Climate Science Article",
            storage=engine.storage,
            embedding_manager=embedding_manager
        )
        print(f"   ✅ Created/merged {len(node_ids)} nodes")
        
        # Show created nodes
        print("📋 Created knowledge nodes:")
        for i, node_id in enumerate(node_ids, 1):
            node = engine.get_node(node_id)
            print(f"   {i}. ID: {node_id}")
            print(f"      Content: {node.content[:60]}...")
            print(f"      Truthfulness: {node.rating_truthfulness:.2f}")
            print(f"      Source: {node.source}")
        
        return node_ids
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        return []


def example_4_automatic_relationships(engine: KnowledgeEngine, embedding_manager: EmbeddingManager, node_ids: List[str]):
    """Example 4: Create automatic relationships between knowledge nodes."""
    print("\n" + "="*70)
    print("🔗 Example 4: Automatic Relationship Creation")
    print("="*70)
    
    if len(node_ids) < 2:
        print("❌ Need at least 2 nodes to create relationships")
        return
    
    print(f"🔍 Analyzing relationships between {len(node_ids)} nodes...")
    
    try:
        # Analyze and create relationships
        relationships = analyze_and_create_relationships(
            node_ids,
            engine.storage,
            embedding_manager
        )
        
        print(f"✅ Relationship analysis complete:")
        total_relationships = 0
        
        for rel_type, edge_ids in relationships.items():
            count = len(edge_ids)
            total_relationships += count
            print(f"   🏷️  {rel_type}: {count} relationships")
        
        print(f"   📊 Total relationships created: {total_relationships}")
        
        # Show some example relationships
        if total_relationships > 0:
            print(f"\n🔗 Example relationships:")
            shown = 0
            for rel_type, edge_ids in relationships.items():
                if shown >= 3:  # Limit to 3 examples
                    break
                for edge_id in edge_ids[:2]:  # Show up to 2 per type
                    if shown >= 3:
                        break
                    try:
                        rel = engine.get_relationship(edge_id)
                        from_node = engine.get_node(rel.from_id)
                        to_node = engine.get_node(rel.to_id)
                        
                        print(f"   {shown + 1}. {from_node.content[:30]}...")
                        print(f"      --[{rel.relation_type}]--> {to_node.content[:30]}...")
                        print(f"      Confidence: {rel.confidence_score:.2f}")
                        shown += 1
                    except Exception as e:
                        print(f"   Error displaying relationship {edge_id}: {e}")
        
    except Exception as e:
        print(f"❌ Relationship creation failed: {e}")


def example_5_knowledge_merging(engine: KnowledgeEngine, embedding_manager: EmbeddingManager):
    """Example 5: Demonstrate knowledge merging with similar content."""
    print("\n" + "="*70)
    print("🔀 Example 5: Knowledge Merging")
    print("="*70)
    
    # Create similar content that should be merged
    similar_texts = [
        "Machine learning is a method of data analysis that automates analytical model building.",
        "Machine learning is an automated data analysis method that builds analytical models.",
        "Python is a programming language that emphasizes code readability and simplicity."
    ]
    
    print("📝 Processing similar texts to demonstrate merging...")
    
    all_node_ids = []
    
    for i, text in enumerate(similar_texts, 1):
        print(f"\n📄 Processing text {i}: {text}")
        
        try:
            # Extract and process each text
            units = extract_knowledge_units(text)
            node_ids = process_extracted_units(
                units=units,
                source_label=f"Text Source {i}",
                storage=engine.storage,
                embedding_manager=embedding_manager
            )
            
            all_node_ids.extend(node_ids)
            print(f"   ✅ Created/merged {len(node_ids)} nodes")
            
        except Exception as e:
            print(f"   ❌ Processing failed: {e}")
    
    print(f"\n📊 Merging summary:")
    print(f"   📄 Processed {len(similar_texts)} texts")
    print(f"   📦 Total nodes created/merged: {len(all_node_ids)}")
    
    # Show final nodes
    print(f"\n📋 Final knowledge nodes:")
    unique_node_ids = list(set(all_node_ids))
    for i, node_id in enumerate(unique_node_ids, 1):
        try:
            node = engine.get_node(node_id)
            print(f"   {i}. {node.content}")
            print(f"      Source: {node.source}")
            print(f"      Truthfulness: {node.rating_truthfulness:.2f}")
        except Exception as e:
            print(f"   Error retrieving node {node_id}: {e}")


def example_6_batch_processing(engine: KnowledgeEngine, embedding_manager: EmbeddingManager):
    """Example 6: Batch processing of multiple documents."""
    print("\n" + "="*70)
    print("📚 Example 6: Batch Processing Multiple Documents")
    print("="*70)
    
    # Sample documents from different domains
    documents = [
        {
            "title": "Computer Science Basics",
            "content": """
            Computer science is the study of computational systems and algorithms. 
            Programming languages like Python, Java, and C++ are used to create software. 
            Data structures such as arrays, linked lists, and trees help organize information. 
            Algorithms are step-by-step procedures for solving problems efficiently.
            """,
            "source": "CS Textbook"
        },
        {
            "title": "Renewable Energy Overview", 
            "content": """
            Renewable energy comes from natural sources that are constantly replenished. 
            Solar energy harnesses sunlight through photovoltaic cells to generate electricity. 
            Wind energy uses turbines to convert wind motion into electrical power. 
            Hydroelectric power generates electricity from flowing water.
            """,
            "source": "Environmental Science Journal"
        },
        {
            "title": "Human Biology Fundamentals",
            "content": """
            The human body consists of multiple organ systems working together. 
            The circulatory system transports blood, nutrients, and oxygen throughout the body. 
            The nervous system controls body functions through electrical signals. 
            DNA contains genetic information that determines physical characteristics.
            """,
            "source": "Biology Reference"
        }
    ]
    
    print(f"📚 Processing {len(documents)} documents...")
    
    all_node_ids = []
    processing_stats = {
        "total_units": 0,
        "total_nodes": 0,
        "processing_time": 0
    }
    
    start_time = time.time()
    
    for i, doc in enumerate(documents, 1):
        print(f"\n📄 Processing document {i}: {doc['title']}")
        
        try:
            # Extract knowledge units
            units = extract_knowledge_units(doc['content'])
            processing_stats["total_units"] += len(units)
            
            # Process and store
            node_ids = process_extracted_units(
                units=units,
                source_label=doc['source'],
                storage=engine.storage,
                embedding_manager=embedding_manager
            )
            
            all_node_ids.extend(node_ids)
            processing_stats["total_nodes"] += len(node_ids)
            
            print(f"   ✅ Extracted {len(units)} units, created {len(node_ids)} nodes")
            
        except Exception as e:
            print(f"   ❌ Processing failed: {e}")
    
    processing_stats["processing_time"] = time.time() - start_time
    
    # Create relationships across all nodes
    print(f"\n🔗 Creating cross-document relationships...")
    try:
        relationships = analyze_and_create_relationships(
            all_node_ids,
            engine.storage,
            embedding_manager
        )
        
        total_relationships = sum(len(edges) for edges in relationships.values())
        print(f"   ✅ Created {total_relationships} relationships")
        
    except Exception as e:
        print(f"   ❌ Relationship creation failed: {e}")
    
    # Show processing statistics
    print(f"\n📊 Batch Processing Statistics:")
    print(f"   📚 Documents processed: {len(documents)}")
    print(f"   🧠 Knowledge units extracted: {processing_stats['total_units']}")
    print(f"   📦 Nodes created: {processing_stats['total_nodes']}")
    print(f"   ⏱️  Processing time: {processing_stats['processing_time']:.2f} seconds")
    print(f"   📈 Units per second: {processing_stats['total_units'] / processing_stats['processing_time']:.2f}")


def main():
    """Main function running all knowledge extraction examples."""
    print("🌟 Memory Engine - Knowledge Extraction Examples")
    print("="*70)
    
    # Setup system
    engine, embedding_manager = setup_system()
    if not engine:
        sys.exit(1)
    
    try:
        # Run examples
        print("\n🎯 Running knowledge extraction examples...")
        
        # Basic examples
        units1 = example_1_simple_text_extraction()
        units2 = example_2_scientific_text_extraction()
        
        # Processing and storage examples
        node_ids = example_3_process_and_store_knowledge(engine, embedding_manager)
        
        if node_ids:
            example_4_automatic_relationships(engine, embedding_manager, node_ids)
        
        # Advanced examples
        example_5_knowledge_merging(engine, embedding_manager)
        example_6_batch_processing(engine, embedding_manager)
        
        print("\n" + "="*70)
        print("🎉 All knowledge extraction examples completed!")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if embedding_manager and embedding_manager.vector_store:
            embedding_manager.vector_store.disconnect()
        if engine:
            engine.disconnect()
        print("🧹 Cleanup complete")


if __name__ == "__main__":
    main()