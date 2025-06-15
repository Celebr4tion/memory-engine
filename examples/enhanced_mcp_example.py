#!/usr/bin/env python3
"""
Enhanced MCP Interface Example

This example demonstrates the advanced capabilities of the Enhanced MCP interface
including graph queries, knowledge synthesis, bulk operations, and analytics.
"""

import os
import sys
import time
import json
from typing import Dict, List, Any

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_core.mcp_integration.enhanced_mcp_endpoint import EnhancedMemoryEngineMCP


def print_header(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"📊 {title}")
    print(f"{'='*60}")


def print_subheader(title: str):
    """Print a formatted subsection header."""
    print(f"\n{'─'*40}")
    print(f"🔹 {title}")
    print(f"{'─'*40}")


def print_result(result: Dict[str, Any], title: str = "Result"):
    """Print formatted result."""
    print(f"\n✅ {title}:")
    print(json.dumps(result, indent=2, default=str))


def demonstrate_advanced_graph_queries(mcp: EnhancedMemoryEngineMCP):
    """Demonstrate advanced graph query capabilities."""
    print_header("Advanced Graph Queries")

    # First, let's create some test data
    print_subheader("Setting up test data")

    # Create sample nodes for demonstration
    sample_texts = [
        {
            "text": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming.",
            "source": "AI Textbook",
        },
        {
            "text": "Neural networks are computing systems inspired by biological neural networks. They consist of layers of interconnected nodes.",
            "source": "Deep Learning Guide",
        },
        {
            "text": "Deep learning uses neural networks with multiple hidden layers to model complex patterns in data.",
            "source": "Research Paper",
        },
    ]

    # Ingest sample data first (using basic MCP)
    node_ids = []
    for sample in sample_texts:
        command = {"action": "ingest_text", "text": sample["text"], "source": sample["source"]}
        # Note: This would use the basic MCP endpoint in a real scenario
        print(f"📝 Would ingest: {sample['text'][:50]}...")

    # For demonstration, we'll use mock node IDs
    mock_node_ids = ["node_ml_001", "node_nn_002", "node_dl_003"]

    # 1. Multi-hop Traversal
    print_subheader("Multi-hop Traversal")

    traversal_command = {
        "action": "multi_hop_traversal",
        "start_node_id": "node_ml_001",
        "max_hops": 2,
        "relation_filter": ["relates_to", "contains"],
        "min_confidence": 0.6,
    }

    print(f"🔍 Command: {json.dumps(traversal_command, indent=2)}")

    try:
        result = mcp.execute_mcp_command(traversal_command)
        print_result(result, "Multi-hop Traversal")
    except Exception as e:
        print(f"⚠️ Demo mode - would perform traversal: {e}")

    # 2. Subgraph Extraction
    print_subheader("Subgraph Extraction")

    subgraph_command = {
        "action": "extract_subgraph",
        "topic_keywords": ["machine learning", "neural networks"],
        "max_nodes": 20,
        "min_relevance": 0.7,
    }

    print(f"🔍 Command: {json.dumps(subgraph_command, indent=2)}")

    try:
        result = mcp.execute_mcp_command(subgraph_command)
        print_result(result, "Subgraph Extraction")
    except Exception as e:
        print(f"⚠️ Demo mode - would extract subgraph: {e}")

    # 3. Pattern Matching
    print_subheader("Pattern Matching")

    pattern_command = {
        "action": "pattern_matching",
        "pattern": {
            "nodes": {"content_contains": "neural networks", "min_truthfulness": 0.8},
            "relationships": {"outgoing_relation_type": "defines"},
            "max_results": 10,
        },
    }

    print(f"🔍 Command: {json.dumps(pattern_command, indent=2)}")

    try:
        result = mcp.execute_mcp_command(pattern_command)
        print_result(result, "Pattern Matching")
    except Exception as e:
        print(f"⚠️ Demo mode - would perform pattern matching: {e}")

    # 4. Temporal Query
    print_subheader("Temporal Query")

    # Query for nodes created in the last 24 hours
    end_time = time.time()
    start_time = end_time - (24 * 3600)  # 24 hours ago

    temporal_command = {
        "action": "temporal_query",
        "start_time": start_time,
        "end_time": end_time,
        "operation_type": "nodes_created",
    }

    print(f"🔍 Command: {json.dumps(temporal_command, indent=2)}")

    try:
        result = mcp.execute_mcp_command(temporal_command)
        print_result(result, "Temporal Query")
    except Exception as e:
        print(f"⚠️ Demo mode - would perform temporal query: {e}")


def demonstrate_knowledge_synthesis(mcp: EnhancedMemoryEngineMCP):
    """Demonstrate knowledge synthesis capabilities."""
    print_header("Knowledge Synthesis")

    # Mock node IDs for demonstration
    mock_node_ids = ["node_ai_001", "node_ai_002", "node_ai_003"]

    # 1. Knowledge Synthesis - Summary
    print_subheader("Knowledge Summary")

    synthesis_command = {
        "action": "synthesize_knowledge",
        "node_ids": mock_node_ids,
        "synthesis_type": "summary",
    }

    print(f"🔍 Command: {json.dumps(synthesis_command, indent=2)}")

    try:
        result = mcp.execute_mcp_command(synthesis_command)
        print_result(result, "Knowledge Synthesis - Summary")
    except Exception as e:
        print(f"⚠️ Demo mode - would synthesize knowledge: {e}")

    # 2. Knowledge Synthesis - Comparison
    print_subheader("Knowledge Comparison")

    comparison_command = {
        "action": "synthesize_knowledge",
        "node_ids": mock_node_ids[:2],  # Compare just 2 nodes
        "synthesis_type": "comparison",
    }

    print(f"🔍 Command: {json.dumps(comparison_command, indent=2)}")

    try:
        result = mcp.execute_mcp_command(comparison_command)
        print_result(result, "Knowledge Synthesis - Comparison")
    except Exception as e:
        print(f"⚠️ Demo mode - would compare knowledge: {e}")

    # 3. Question Answering
    print_subheader("Question Answering")

    question_command = {
        "action": "answer_question",
        "question": "What are the key components of machine learning systems?",
        "max_hops": 2,
        "top_k_nodes": 10,
    }

    print(f"🔍 Command: {json.dumps(question_command, indent=2)}")

    try:
        result = mcp.execute_mcp_command(question_command)
        print_result(result, "Question Answering")
    except Exception as e:
        print(f"⚠️ Demo mode - would answer question: {e}")

    # 4. Contradiction Detection
    print_subheader("Contradiction Detection")

    contradiction_command = {
        "action": "find_contradictions",
        "topic_keywords": ["artificial intelligence", "machine learning"],
        "confidence_threshold": 0.8,
    }

    print(f"🔍 Command: {json.dumps(contradiction_command, indent=2)}")

    try:
        result = mcp.execute_mcp_command(contradiction_command)
        print_result(result, "Contradiction Detection")
    except Exception as e:
        print(f"⚠️ Demo mode - would detect contradictions: {e}")


def demonstrate_bulk_operations(mcp: EnhancedMemoryEngineMCP):
    """Demonstrate bulk operations with progress tracking."""
    print_header("Bulk Operations")

    # 1. Start Bulk Ingestion
    print_subheader("Starting Bulk Ingestion")

    start_command = {"action": "start_bulk_ingestion"}

    try:
        result = mcp.execute_mcp_command(start_command)
        print_result(result, "Bulk Ingestion Started")

        if result["status"] == "success":
            operation_id = result["operation_id"]

            # 2. Add texts to bulk ingestion
            print_subheader("Adding Texts to Bulk Ingestion")

            sample_texts = [
                {
                    "text": "Quantum computing leverages quantum mechanical phenomena to process information.",
                    "source": "Quantum Physics Journal",
                },
                {
                    "text": "Blockchain technology provides a decentralized ledger for secure transactions.",
                    "source": "Cryptocurrency Guide",
                },
                {
                    "text": "Cloud computing delivers computing services over the internet on-demand.",
                    "source": "Technology Magazine",
                },
            ]

            add_command = {
                "action": "add_to_bulk_ingestion",
                "operation_id": operation_id,
                "texts": sample_texts,
            }

            print(f"🔍 Command: {json.dumps(add_command, indent=2)}")

            try:
                result = mcp.execute_mcp_command(add_command)
                print_result(result, "Texts Added to Bulk Ingestion")
            except Exception as e:
                print(f"⚠️ Demo mode - would add texts: {e}")

            # 3. Check bulk operation status
            print_subheader("Checking Bulk Operation Status")

            status_command = {"action": "get_bulk_operation_status", "operation_id": operation_id}

            try:
                result = mcp.execute_mcp_command(status_command)
                print_result(result, "Bulk Operation Status")
            except Exception as e:
                print(f"⚠️ Demo mode - would check status: {e}")

    except Exception as e:
        print(f"⚠️ Demo mode - would start bulk operation: {e}")

    # 4. Export Subgraph
    print_subheader("Exporting Subgraph")

    export_command = {
        "action": "export_subgraph",
        "node_ids": ["node_001", "node_002", "node_003"],
        "format": "json",
        "include_relationships": True,
    }

    print(f"🔍 Command: {json.dumps(export_command, indent=2)}")

    try:
        result = mcp.execute_mcp_command(export_command)
        print_result(result, "Subgraph Export")
    except Exception as e:
        print(f"⚠️ Demo mode - would export subgraph: {e}")

    # 5. Bulk Create Relationships
    print_subheader("Bulk Creating Relationships")

    relationships = [
        {
            "from_id": "node_001",
            "to_id": "node_002",
            "relation_type": "relates_to",
            "confidence_score": 0.85,
        },
        {
            "from_id": "node_002",
            "to_id": "node_003",
            "relation_type": "contains",
            "confidence_score": 0.75,
        },
    ]

    bulk_rel_command = {"action": "bulk_create_relationships", "relationships": relationships}

    print(f"🔍 Command: {json.dumps(bulk_rel_command, indent=2)}")

    try:
        result = mcp.execute_mcp_command(bulk_rel_command)
        print_result(result, "Bulk Relationship Creation")
    except Exception as e:
        print(f"⚠️ Demo mode - would create relationships: {e}")


def demonstrate_analytics_endpoints(mcp: EnhancedMemoryEngineMCP):
    """Demonstrate analytics and metrics capabilities."""
    print_header("Analytics and Metrics")

    # 1. Knowledge Coverage Analysis
    print_subheader("Knowledge Coverage Analysis")

    coverage_command = {
        "action": "analyze_knowledge_coverage",
        "domains": ["technology", "science", "artificial_intelligence"],
    }

    print(f"🔍 Command: {json.dumps(coverage_command, indent=2)}")

    try:
        result = mcp.execute_mcp_command(coverage_command)
        print_result(result, "Knowledge Coverage Analysis")
    except Exception as e:
        print(f"⚠️ Demo mode - would analyze coverage: {e}")

    # 2. Relationship Metrics
    print_subheader("Relationship Metrics")

    metrics_command = {"action": "calculate_relationship_metrics"}

    try:
        result = mcp.execute_mcp_command(metrics_command)
        print_result(result, "Relationship Metrics")
    except Exception as e:
        print(f"⚠️ Demo mode - would calculate metrics: {e}")

    # 3. Quality Score Analysis
    print_subheader("Quality Score Analysis")

    quality_command = {"action": "analyze_quality_scores"}

    try:
        result = mcp.execute_mcp_command(quality_command)
        print_result(result, "Quality Score Analysis")
    except Exception as e:
        print(f"⚠️ Demo mode - would analyze quality: {e}")

    # 4. Knowledge Evolution Analysis
    print_subheader("Knowledge Evolution Analysis")

    evolution_command = {
        "action": "analyze_knowledge_evolution",
        "time_periods": 6,  # Last 6 months
    }

    print(f"🔍 Command: {json.dumps(evolution_command, indent=2)}")

    try:
        result = mcp.execute_mcp_command(evolution_command)
        print_result(result, "Knowledge Evolution Analysis")
    except Exception as e:
        print(f"⚠️ Demo mode - would analyze evolution: {e}")


def demonstrate_error_handling(mcp: EnhancedMemoryEngineMCP):
    """Demonstrate error handling in enhanced MCP commands."""
    print_header("Error Handling Examples")

    # 1. Invalid command
    print_subheader("Invalid Command")

    invalid_command = {"action": "nonexistent_action", "parameter": "value"}

    result = mcp.execute_mcp_command(invalid_command)
    print_result(result, "Invalid Command Response")

    # 2. Missing required parameters
    print_subheader("Missing Required Parameters")

    incomplete_command = {
        "action": "multi_hop_traversal"
        # Missing start_node_id
    }

    result = mcp.execute_mcp_command(incomplete_command)
    print_result(result, "Incomplete Command Response")

    # 3. Non-existent node
    print_subheader("Non-existent Node")

    nonexistent_command = {
        "action": "multi_hop_traversal",
        "start_node_id": "nonexistent_node_12345",
    }

    result = mcp.execute_mcp_command(nonexistent_command)
    print_result(result, "Non-existent Node Response")


def main():
    """Main demonstration function."""
    print("🌟 Memory Engine - Enhanced MCP Interface Examples")
    print("=" * 60)

    # Check prerequisites
    print("🔍 Checking prerequisites...")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("⚠️ Warning: GEMINI_API_KEY not set - some features may not work")
    else:
        print("✅ GEMINI_API_KEY is set")

    try:
        # Initialize Enhanced MCP endpoint
        print("\n🚀 Initializing Enhanced MCP endpoint...")

        # In demo mode, we'll handle connection errors gracefully
        try:
            mcp = EnhancedMemoryEngineMCP()
            print("✅ Enhanced MCP endpoint initialized")
        except Exception as e:
            print(f"⚠️ Running in demo mode due to: {e}")
            # Create a mock MCP for demonstration
            mcp = EnhancedMemoryEngineMCP.__new__(EnhancedMemoryEngineMCP)
            mcp.engine = None
            mcp.vector_store = None
            mcp.embedding_manager = None
            mcp.bulk_operations = {}

        # Run demonstrations
        demonstrate_advanced_graph_queries(mcp)
        demonstrate_knowledge_synthesis(mcp)
        demonstrate_bulk_operations(mcp)
        demonstrate_analytics_endpoints(mcp)
        demonstrate_error_handling(mcp)

        print_header("Summary")
        print("✅ Enhanced MCP Interface demonstration completed!")
        print("\n📋 Demonstrated capabilities:")
        print("   • Multi-hop graph traversal")
        print("   • Subgraph extraction and pattern matching")
        print("   • Temporal queries using version history")
        print("   • Knowledge synthesis and question answering")
        print("   • Contradiction detection")
        print("   • Bulk operations with progress tracking")
        print("   • Subgraph export in multiple formats")
        print("   • Analytics and metrics calculation")
        print("   • Comprehensive error handling")

        print("\n🔗 Next steps:")
        print("   • Explore the enhanced_mcp_endpoint.py implementation")
        print("   • Check the test_enhanced_mcp_endpoint.py test suite")
        print("   • Review the updated API documentation")
        print("   • Integrate with your applications using the MCP interface")

    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Cleanup
        try:
            if "mcp" in locals() and hasattr(mcp, "close"):
                mcp.close()
        except:
            pass


if __name__ == "__main__":
    main()
