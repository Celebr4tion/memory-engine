#!/usr/bin/env python3
"""
MCP (Module Communication Protocol) client examples for the Memory Engine.

This script demonstrates how to interact with the Memory Engine through its MCP interface:
- Ingesting text through MCP commands
- Searching for knowledge via MCP
- Retrieving node details
- Updating node ratings
- Handling MCP responses

Prerequisites:
- GEMINI_API_KEY environment variable set
- JanusGraph and Milvus running (use docker-compose)
- Memory Engine MCP endpoint running
"""

import os
import sys
import json
import time
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_core.mcp_integration.mcp_endpoint import MemoryEngineMCP


def setup_mcp_client():
    """Set up the MCP client interface."""
    print("🚀 Setting up Memory Engine MCP interface...")
    
    # Check prerequisites
    if not os.getenv('GEMINI_API_KEY'):
        print("❌ GEMINI_API_KEY environment variable not set")
        return None
    
    # Initialize MCP interface
    try:
        mcp = MemoryEngineMCP(host="localhost", port=8182)
        print("✅ MCP interface initialized")
        return mcp
    except Exception as e:
        print(f"❌ Failed to initialize MCP interface: {e}")
        return None


def example_1_ingest_text_via_mcp(mcp: MemoryEngineMCP):
    """Example 1: Ingest text content via MCP commands."""
    print("\n" + "="*70)
    print("📝 Example 1: Text Ingestion via MCP")
    print("="*70)
    
    # Sample texts to ingest
    sample_texts = [
        {
            "text": """
            Blockchain technology is a distributed ledger that maintains a continuously growing 
            list of records, called blocks, which are linked and secured using cryptography. 
            Each block contains a cryptographic hash of the previous block, a timestamp, and 
            transaction data. Bitcoin was the first successful implementation of blockchain 
            technology, introduced in 2008 by Satoshi Nakamoto.
            """,
            "source": "Cryptocurrency Guide"
        },
        {
            "text": """
            Artificial Neural Networks are computing systems inspired by biological neural networks. 
            They consist of interconnected nodes (neurons) that process information using a 
            connectionist approach. Deep learning uses neural networks with multiple hidden layers 
            to learn complex patterns in data. Convolutional Neural Networks (CNNs) are particularly 
            effective for image recognition tasks.
            """,
            "source": "Machine Learning Handbook"
        },
        {
            "text": """
            Quantum entanglement is a physical phenomenon where quantum states of two or more 
            particles become correlated. When particles are entangled, measuring one particle 
            instantly affects the state of its entangled partner, regardless of the distance 
            between them. This property is fundamental to quantum computing and quantum 
            communication protocols.
            """,
            "source": "Physics Research Paper"
        }
    ]
    
    ingested_node_ids = []
    
    for i, sample in enumerate(sample_texts, 1):
        print(f"\n📄 Ingesting text {i} via MCP...")
        print(f"   Source: {sample['source']}")
        print(f"   Content preview: {sample['text'].strip()[:60]}...")
        
        # Create MCP command for text ingestion
        command = {
            "action": "ingest_text",
            "text": sample["text"],
            "source": sample["source"]
        }
        
        try:
            # Execute MCP command
            response = mcp.execute_mcp_command(command)
            
            if response["status"] == "success":
                node_ids = response["created_or_merged_node_ids"]
                ingested_node_ids.extend(node_ids)
                
                print(f"   ✅ Success! Created/merged {len(node_ids)} nodes")
                
                # Show relationship statistics
                if "relationship_counts" in response:
                    rel_stats = response["relationship_counts"]
                    total_rels = sum(rel_stats.values())
                    if total_rels > 0:
                        print(f"   🔗 Created {total_rels} relationships:")
                        for rel_type, count in rel_stats.items():
                            if count > 0:
                                print(f"      {rel_type}: {count}")
                
            else:
                print(f"   ❌ Ingestion failed: {response.get('message', 'Unknown error')}")
                
        except Exception as e:
            print(f"   ❌ MCP command failed: {e}")
    
    print(f"\n📊 Ingestion Summary:")
    print(f"   📄 Texts processed: {len(sample_texts)}")
    print(f"   📦 Total nodes created/merged: {len(ingested_node_ids)}")
    
    return ingested_node_ids


def example_2_search_via_mcp(mcp: MemoryEngineMCP):
    """Example 2: Search for knowledge via MCP commands."""
    print("\n" + "="*70)
    print("🔍 Example 2: Knowledge Search via MCP")
    print("="*70)
    
    # Search queries
    search_queries = [
        "blockchain and cryptocurrency technology",
        "neural networks and deep learning",
        "quantum physics and entanglement",
        "distributed systems and security"
    ]
    
    for i, query in enumerate(search_queries, 1):
        print(f"\n🔍 Search {i}: '{query}'")
        
        # Create MCP search command
        command = {
            "action": "search",
            "query": query,
            "top_k": 3
        }
        
        try:
            # Execute search command
            response = mcp.execute_mcp_command(command)
            
            if response["status"] == "success":
                results = response["results"]
                print(f"   ✅ Found {len(results)} results:")
                
                for j, result in enumerate(results, 1):
                    print(f"      {j}. Node ID: {result['node_id']}")
                    print(f"         Content: {result['content_preview']}")
                    print(f"         Source: {result['source']}")
                    print(f"         Truthfulness: {result['rating_truthfulness']:.2f}")
                    
            elif response["status"] == "error":
                print(f"   ❌ Search failed: {response['message']}")
            else:
                print(f"   ⚠️  {response.get('message', 'No results found')}")
                
        except Exception as e:
            print(f"   ❌ Search command failed: {e}")


def example_3_node_details_via_mcp(mcp: MemoryEngineMCP, node_ids: List[str]):
    """Example 3: Get detailed node information via MCP."""
    print("\n" + "="*70)
    print("📋 Example 3: Node Details via MCP")
    print("="*70)
    
    if not node_ids:
        print("❌ No node IDs available for detail retrieval")
        return
    
    # Get details for first few nodes
    sample_nodes = node_ids[:3]
    
    for i, node_id in enumerate(sample_nodes, 1):
        print(f"\n📍 Getting details for node {i} (ID: {node_id})...")
        
        # Create MCP command for node details
        command = {
            "action": "get_node",
            "node_id": node_id
        }
        
        try:
            # Execute command
            response = mcp.execute_mcp_command(command)
            
            if response["status"] == "success":
                node = response["node"]
                outgoing = response["outgoing_relationships"]
                incoming = response["incoming_relationships"]
                
                print(f"   ✅ Node details retrieved:")
                print(f"      📄 Content: {node['content']}")
                print(f"      📍 Source: {node['source']}")
                print(f"      📊 Ratings:")
                print(f"         Truthfulness: {node['rating_truthfulness']:.2f}")
                print(f"         Richness: {node['rating_richness']:.2f}")
                print(f"         Stability: {node['rating_stability']:.2f}")
                
                # Show relationships
                if outgoing:
                    print(f"      ➡️  Outgoing relationships: {len(outgoing)}")
                    for rel in outgoing[:2]:  # Show first 2
                        print(f"         {rel['relation_type']} → {rel['target_id']}")
                
                if incoming:
                    print(f"      ⬅️  Incoming relationships: {len(incoming)}")
                    for rel in incoming[:2]:  # Show first 2
                        print(f"         {rel['source_id']} → {rel['relation_type']}")
                        
            else:
                print(f"   ❌ Failed to get node details: {response.get('message', 'Unknown error')}")
                
        except Exception as e:
            print(f"   ❌ Command failed: {e}")


def example_4_update_ratings_via_mcp(mcp: MemoryEngineMCP, node_ids: List[str]):
    """Example 4: Update node ratings via MCP commands."""
    print("\n" + "="*70)
    print("📊 Example 4: Rating Updates via MCP")
    print("="*70)
    
    if not node_ids:
        print("❌ No node IDs available for rating updates")
        return
    
    # Select first node for rating updates
    node_id = node_ids[0]
    
    # Rating update scenarios
    update_scenarios = [
        {
            "name": "Positive confirmation",
            "rating": {"confirmation": 0.3},
            "description": "Found additional supporting evidence"
        },
        {
            "name": "Add richness",
            "rating": {"richness": 0.2},
            "description": "Enhanced with more detailed information"
        },
        {
            "name": "Stability boost",
            "rating": {"stability": 0.1},
            "description": "Verified information remains consistent"
        }
    ]
    
    print(f"📍 Updating ratings for node: {node_id}")
    
    # Get initial state
    get_command = {"action": "get_node", "node_id": node_id}
    initial_response = mcp.execute_mcp_command(get_command)
    
    if initial_response["status"] == "success":
        initial_node = initial_response["node"]
        print(f"   📊 Initial ratings:")
        print(f"      Truthfulness: {initial_node['rating_truthfulness']:.2f}")
        print(f"      Richness: {initial_node['rating_richness']:.2f}")
        print(f"      Stability: {initial_node['rating_stability']:.2f}")
    
    # Apply rating updates
    for i, scenario in enumerate(update_scenarios, 1):
        print(f"\n   📈 Scenario {i}: {scenario['name']}")
        print(f"      Description: {scenario['description']}")
        
        # Create rating update command
        command = {
            "action": "update_rating",
            "node_id": node_id,
            "rating": scenario["rating"]
        }
        
        try:
            # Execute update command
            response = mcp.execute_mcp_command(command)
            
            if response["status"] == "success":
                updated_node = response["node"]
                print(f"      ✅ Update successful!")
                print(f"         Truthfulness: {updated_node['rating_truthfulness']:.2f}")
                print(f"         Richness: {updated_node['rating_richness']:.2f}")
                print(f"         Stability: {updated_node['rating_stability']:.2f}")
            else:
                print(f"      ❌ Update failed: {response.get('message', 'Unknown error')}")
                
        except Exception as e:
            print(f"      ❌ Command failed: {e}")


def example_5_mcp_error_handling(mcp: MemoryEngineMCP):
    """Example 5: Demonstrate MCP error handling."""
    print("\n" + "="*70)
    print("⚠️  Example 5: MCP Error Handling")
    print("="*70)
    
    # Test various error scenarios
    error_scenarios = [
        {
            "name": "Invalid command format",
            "command": {"invalid": "command"},
            "expected": "Missing action field"
        },
        {
            "name": "Unknown action",
            "command": {"action": "unknown_action"},
            "expected": "Unknown action"
        },
        {
            "name": "Missing required field",
            "command": {"action": "ingest_text"},
            "expected": "Missing required field"
        },
        {
            "name": "Invalid node ID",
            "command": {"action": "get_node", "node_id": "nonexistent-id"},
            "expected": "Node not found"
        }
    ]
    
    for i, scenario in enumerate(error_scenarios, 1):
        print(f"\n🧪 Error Test {i}: {scenario['name']}")
        print(f"   Command: {scenario['command']}")
        
        try:
            response = mcp.execute_mcp_command(scenario["command"])
            
            if response["status"] == "error":
                print(f"   ✅ Expected error handled correctly:")
                print(f"      Message: {response['message']}")
            else:
                print(f"   ⚠️  Unexpected success: {response}")
                
        except Exception as e:
            print(f"   ✅ Exception handled: {e}")


def example_6_batch_mcp_operations(mcp: MemoryEngineMCP):
    """Example 6: Batch operations via MCP commands."""
    print("\n" + "="*70)
    print("📦 Example 6: Batch MCP Operations")
    print("="*70)
    
    # Batch of texts to process
    batch_texts = [
        "Photosynthesis is the process by which plants convert sunlight into chemical energy.",
        "The Internet is a global network of interconnected computers.",
        "DNA sequencing determines the precise order of nucleotides in a DNA molecule.",
        "Renewable energy includes solar, wind, and hydroelectric power sources.",
        "Machine learning algorithms learn patterns from data to make predictions."
    ]
    
    print(f"📦 Processing batch of {len(batch_texts)} texts...")
    
    start_time = time.time()
    batch_results = []
    
    for i, text in enumerate(batch_texts, 1):
        print(f"\n   📄 Processing text {i}/{len(batch_texts)}...")
        
        command = {
            "action": "ingest_text",
            "text": text,
            "source": f"Batch Source {i}"
        }
        
        try:
            response = mcp.execute_mcp_command(command)
            batch_results.append({
                "text_number": i,
                "success": response["status"] == "success",
                "node_count": len(response.get("created_or_merged_node_ids", [])),
                "response": response
            })
            
            if response["status"] == "success":
                node_count = len(response["created_or_merged_node_ids"])
                print(f"      ✅ Success: {node_count} nodes")
            else:
                print(f"      ❌ Failed: {response.get('message', 'Unknown error')}")
                
        except Exception as e:
            print(f"      ❌ Error: {e}")
            batch_results.append({
                "text_number": i,
                "success": False,
                "node_count": 0,
                "error": str(e)
            })
    
    processing_time = time.time() - start_time
    
    # Show batch statistics
    print(f"\n📊 Batch Processing Results:")
    successful = sum(1 for r in batch_results if r["success"])
    total_nodes = sum(r["node_count"] for r in batch_results)
    
    print(f"   📄 Texts processed: {len(batch_texts)}")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {len(batch_texts) - successful}")
    print(f"   📦 Total nodes created: {total_nodes}")
    print(f"   ⏱️  Processing time: {processing_time:.2f} seconds")
    print(f"   📈 Texts per second: {len(batch_texts) / processing_time:.2f}")


def main():
    """Main function running all MCP examples."""
    print("🌟 Memory Engine - MCP Client Examples")
    print("="*70)
    
    # Setup MCP client
    mcp = setup_mcp_client()
    if not mcp:
        sys.exit(1)
    
    try:
        # Run examples
        print("\n🎯 Running MCP interaction examples...")
        
        # Basic MCP operations
        node_ids = example_1_ingest_text_via_mcp(mcp)
        example_2_search_via_mcp(mcp)
        
        if node_ids:
            example_3_node_details_via_mcp(mcp, node_ids)
            example_4_update_ratings_via_mcp(mcp, node_ids)
        
        # Advanced MCP features
        example_5_mcp_error_handling(mcp)
        example_6_batch_mcp_operations(mcp)
        
        print("\n" + "="*70)
        print("🎉 All MCP examples completed successfully!")
        print("="*70)
        
        # Show usage tips
        print("\n💡 MCP Usage Tips:")
        print("   • Always check response status before processing results")
        print("   • Handle errors gracefully with try/catch blocks")
        print("   • Use batch operations for better performance")
        print("   • Monitor node IDs for tracking knowledge assets")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup MCP interface
        if mcp:
            try:
                mcp.close()
                print("🧹 MCP interface cleanup complete")
            except Exception as e:
                print(f"⚠️  Error during cleanup: {e}")


if __name__ == "__main__":
    main()