"""
Tests for the Enhanced MCP endpoint implementation.

This test suite validates the advanced MCP interface capabilities including
graph queries, knowledge synthesis, bulk operations, and analytics.
"""

import pytest
import json
import time
import uuid
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

from memory_core.mcp_integration.enhanced_mcp_endpoint import EnhancedMemoryEngineMCP
from memory_core.model.knowledge_node import KnowledgeNode
from memory_core.model.relationship import Relationship


# Global fixtures available to all test classes
@pytest.fixture
def mock_engine():
    """Mock KnowledgeEngine for testing."""
    engine = Mock()
    engine.storage = Mock()
    engine.revision_manager = Mock()
    return engine


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    vector_store = Mock()
    vector_store.connect.return_value = True
    return vector_store


@pytest.fixture
def mock_embedding_manager():
    """Mock embedding manager for testing."""
    manager = Mock()
    manager.search_similar_nodes.return_value = ["node1", "node2", "node3"]
    return manager


@pytest.fixture
def enhanced_mcp(mock_engine, mock_vector_store, mock_embedding_manager):
    """Create EnhancedMemoryEngineMCP instance with mocked dependencies."""
    with patch(
        "memory_core.mcp_integration.enhanced_mcp_endpoint.KnowledgeEngine"
    ) as mock_ke, patch(
        "memory_core.mcp_integration.enhanced_mcp_endpoint.VectorStoreMilvus"
    ) as mock_vs, patch(
        "memory_core.mcp_integration.enhanced_mcp_endpoint.EmbeddingManager"
    ) as mock_em:

        mock_ke.return_value = mock_engine
        mock_vs.return_value = mock_vector_store
        mock_em.return_value = mock_embedding_manager

        mcp = EnhancedMemoryEngineMCP()
        mcp.engine = mock_engine
        mcp.vector_store = mock_vector_store
        mcp.embedding_manager = mock_embedding_manager

        return mcp


class TestEnhancedMCPEndpoint:
    """Test suite for Enhanced MCP endpoint functionality."""


class TestAdvancedGraphQueries:
    """Test advanced graph query capabilities."""

    def test_multi_hop_traversal_success(self, enhanced_mcp):
        """Test successful multi-hop traversal."""
        # Mock node and relationships
        start_node = KnowledgeNode("Test content", "test_source")
        start_node.node_id = "start_node"

        relationships = [
            Mock(to_id="node2", relation_type="related_to", confidence_score=0.8),
            Mock(to_id="node3", relation_type="contains", confidence_score=0.9),
        ]

        enhanced_mcp.engine.get_node.side_effect = [
            start_node,  # Initial node retrieval
            start_node,  # Node details in traversal
            KnowledgeNode("Node 2", "source2"),  # node2 details
            KnowledgeNode("Node 3", "source3"),  # node3 details
        ]

        enhanced_mcp.engine.get_outgoing_relationships.return_value = relationships

        # Execute traversal
        result = enhanced_mcp.multi_hop_traversal("start_node", max_hops=2)

        # Verify result
        assert result["status"] == "success"
        assert result["start_node"] == "start_node"
        assert result["max_hops"] == 2
        assert "nodes_by_distance" in result
        assert "paths" in result
        assert "node_details" in result

    def test_multi_hop_traversal_node_not_found(self, enhanced_mcp):
        """Test multi-hop traversal with non-existent starting node."""
        enhanced_mcp.engine.get_node.side_effect = ValueError("Node not found")

        result = enhanced_mcp.multi_hop_traversal("nonexistent_node")

        assert result["status"] == "error"
        assert "not found" in result["message"]

    def test_extract_subgraph_success(self, enhanced_mcp):
        """Test successful subgraph extraction."""
        # Mock similar nodes
        enhanced_mcp.embedding_manager.search_similar_nodes.return_value = ["node1", "node2"]

        # Mock nodes with keyword matches
        node1 = KnowledgeNode("This is about machine learning algorithms", "source1")
        node1.node_id = "node1"
        node2 = KnowledgeNode("Machine learning models are powerful", "source2")
        node2.node_id = "node2"

        enhanced_mcp.engine.get_node.side_effect = [node1, node2]
        enhanced_mcp.engine.get_outgoing_relationships.return_value = [
            Mock(from_id="node1", to_id="node2", relation_type="relates_to", confidence_score=0.7)
        ]

        result = enhanced_mcp.extract_subgraph(["machine", "learning"], max_nodes=10)

        assert result["status"] == "success"
        assert "nodes" in result
        assert "relationships" in result
        assert result["total_nodes"] > 0

    def test_extract_subgraph_no_embedding_manager(self, enhanced_mcp):
        """Test subgraph extraction without embedding manager."""
        enhanced_mcp.embedding_manager = None

        result = enhanced_mcp.extract_subgraph(["test"])

        assert result["status"] == "error"
        assert "Embedding manager not available" in result["message"]

    def test_pattern_matching_basic(self, enhanced_mcp):
        """Test basic pattern matching functionality."""
        pattern = {"nodes": {"content_contains": "python programming"}, "max_results": 5}

        # Mock search results
        enhanced_mcp.embedding_manager.search_similar_nodes.return_value = ["node1"]
        node1 = KnowledgeNode("Python programming is fun", "source1")
        node1.node_id = "node1"
        enhanced_mcp.engine.get_node.return_value = node1

        result = enhanced_mcp.pattern_matching(pattern)

        assert result["status"] == "success"
        assert "matches" in result
        assert result["total_matches"] >= 0

    def test_temporal_query_no_versioning(self, enhanced_mcp):
        """Test temporal query without versioning enabled."""
        enhanced_mcp.engine.revision_manager = None

        result = enhanced_mcp.temporal_query()

        assert result["status"] == "error"
        assert "Versioning not enabled" in result["message"]

    def test_temporal_query_with_versioning(self, enhanced_mcp):
        """Test temporal query with versioning enabled."""
        # Mock revision log data
        enhanced_mcp.engine.storage.query_vertices.return_value = [
            {
                "object_id": "node1",
                "timestamp": time.time() - 3600,  # 1 hour ago
                "change_type": "create",
                "object_type": "node",
            }
        ]

        result = enhanced_mcp.temporal_query(operation_type="nodes_created")

        assert result["status"] == "success"
        assert "results" in result
        assert result["operation_type"] == "nodes_created"


class TestKnowledgeSynthesis:
    """Test knowledge synthesis capabilities."""

    def test_synthesize_knowledge_summary(self, enhanced_mcp):
        """Test knowledge synthesis with summary type."""
        node1 = KnowledgeNode("First piece of knowledge.", "source1")
        node1.node_id = "node1"
        node2 = KnowledgeNode("Second piece of knowledge.", "source2")
        node2.node_id = "node2"

        enhanced_mcp.engine.get_node.side_effect = [node1, node2]

        result = enhanced_mcp.synthesize_knowledge(["node1", "node2"], "summary")

        assert result["status"] == "success"
        assert result["synthesis_type"] == "summary"
        assert "nodes_processed" in result
        assert "summary_points" in result

    def test_synthesize_knowledge_comparison(self, enhanced_mcp):
        """Test knowledge synthesis with comparison type."""
        node1 = KnowledgeNode("Content 1", "source1")
        node1.node_id = "node1"
        node1.rating_truthfulness = 0.8
        node2 = KnowledgeNode("Content 2", "source2")
        node2.node_id = "node2"
        node2.rating_truthfulness = 0.9

        enhanced_mcp.engine.get_node.side_effect = [node1, node2]

        result = enhanced_mcp.synthesize_knowledge(["node1", "node2"], "comparison")

        assert result["status"] == "success"
        assert result["synthesis_type"] == "comparison"
        assert "comparisons" in result

    def test_synthesize_knowledge_timeline(self, enhanced_mcp):
        """Test knowledge synthesis with timeline type."""
        node1 = KnowledgeNode("Event 1", "source1")
        node1.node_id = "node1"
        node1.creation_timestamp = time.time() - 3600
        node2 = KnowledgeNode("Event 2", "source2")
        node2.node_id = "node2"
        node2.creation_timestamp = time.time()

        enhanced_mcp.engine.get_node.side_effect = [node1, node2]

        result = enhanced_mcp.synthesize_knowledge(["node1", "node2"], "timeline")

        assert result["status"] == "success"
        assert result["synthesis_type"] == "timeline"
        assert "timeline_events" in result

    def test_synthesize_knowledge_no_nodes(self, enhanced_mcp):
        """Test knowledge synthesis with no node IDs."""
        result = enhanced_mcp.synthesize_knowledge([], "summary")

        assert result["status"] == "error"
        assert "No node IDs provided" in result["message"]

    def test_answer_question_success(self, enhanced_mcp):
        """Test successful question answering."""
        # Mock relevant nodes
        enhanced_mcp.embedding_manager.search_similar_nodes.return_value = ["node1", "node2"]

        # Mock traversal results
        enhanced_mcp.multi_hop_traversal = Mock(
            return_value={
                "status": "success",
                "nodes_by_distance": {0: ["node1"], 1: ["node2", "node3"]},
            }
        )

        # Mock nodes for evidence
        node1 = KnowledgeNode("Python is a programming language.", "source1")
        node1.node_id = "node1"
        node1.rating_truthfulness = 0.9
        enhanced_mcp.engine.get_node.return_value = node1

        result = enhanced_mcp.answer_question("What is Python?")

        assert result["status"] == "success"
        assert "answer" in result
        assert "evidence" in result
        assert "confidence_score" in result

    def test_answer_question_no_embedding_manager(self, enhanced_mcp):
        """Test question answering without embedding manager."""
        enhanced_mcp.embedding_manager = None

        result = enhanced_mcp.answer_question("What is Python?")

        assert result["status"] == "error"
        assert "Embedding manager not available" in result["message"]

    def test_find_contradictions_success(self, enhanced_mcp):
        """Test successful contradiction detection."""
        # Mock high confidence nodes
        enhanced_mcp._get_high_confidence_nodes = Mock(return_value=["node1", "node2"])

        # Mock contradictory nodes
        node1 = KnowledgeNode("Python is easy to learn.", "source1")
        node1.node_id = "node1"
        node1.rating_truthfulness = 0.9
        node2 = KnowledgeNode("Python is not easy to learn.", "source2")
        node2.node_id = "node2"
        node2.rating_truthfulness = 0.9

        enhanced_mcp.engine.get_node.side_effect = [node1, node2]

        result = enhanced_mcp.find_contradictions()

        assert result["status"] == "success"
        assert "contradictions" in result
        assert "nodes_analyzed" in result


class TestBulkOperations:
    """Test bulk operations functionality."""

    def test_start_bulk_ingestion(self, enhanced_mcp):
        """Test starting a bulk ingestion operation."""
        result = enhanced_mcp.start_bulk_ingestion()

        assert result["status"] == "success"
        assert "operation_id" in result
        operation_id = result["operation_id"]
        assert operation_id in enhanced_mcp.bulk_operations

    def test_add_to_bulk_ingestion_success(self, enhanced_mcp):
        """Test adding texts to bulk ingestion."""
        # Start operation first
        start_result = enhanced_mcp.start_bulk_ingestion()
        operation_id = start_result["operation_id"]

        # Mock processing functions
        with patch(
            "memory_core.mcp_integration.enhanced_mcp_endpoint.extract_knowledge_units"
        ) as mock_extract, patch(
            "memory_core.mcp_integration.enhanced_mcp_endpoint.process_extracted_units"
        ) as mock_process:

            mock_extract.return_value = [{"content": "test"}]
            mock_process.return_value = ["node1", "node2"]

            texts = [
                {"text": "Test content 1", "source": "source1"},
                {"text": "Test content 2", "source": "source2"},
            ]

            result = enhanced_mcp.add_to_bulk_ingestion(operation_id, texts)

            assert result["status"] == "success"
            assert result["processed"] == 2
            assert result["failed"] == 0

    def test_add_to_bulk_ingestion_invalid_operation(self, enhanced_mcp):
        """Test adding to non-existent bulk operation."""
        result = enhanced_mcp.add_to_bulk_ingestion("invalid_id", [])

        assert result["status"] == "error"
        assert "not found" in result["message"]

    def test_get_bulk_operation_status(self, enhanced_mcp):
        """Test getting bulk operation status."""
        # Start operation
        start_result = enhanced_mcp.start_bulk_ingestion()
        operation_id = start_result["operation_id"]

        result = enhanced_mcp.get_bulk_operation_status(operation_id)

        assert result["status"] == "success"
        assert result["operation_id"] == operation_id
        assert "operation_status" in result
        assert "progress" in result

    def test_export_subgraph_json(self, enhanced_mcp):
        """Test subgraph export in JSON format."""
        # Mock nodes
        node1 = KnowledgeNode("Content 1", "source1")
        node1.node_id = "node1"
        node2 = KnowledgeNode("Content 2", "source2")
        node2.node_id = "node2"

        enhanced_mcp.engine.get_node.side_effect = [node1, node2]
        enhanced_mcp.engine.get_outgoing_relationships.return_value = [
            Mock(
                from_id="node1",
                to_id="node2",
                relation_type="relates_to",
                confidence_score=0.8,
                timestamp=time.time(),
            )
        ]

        result = enhanced_mcp.export_subgraph(["node1", "node2"], "json")

        assert result["status"] == "success"
        assert result["format"] == "json"
        assert "data" in result
        assert "nodes" in result["data"]
        assert "relationships" in result["data"]

    def test_export_subgraph_cypher(self, enhanced_mcp):
        """Test subgraph export in Cypher format."""
        node1 = KnowledgeNode("Content 1", "source1")
        node1.node_id = "node1"
        enhanced_mcp.engine.get_node.return_value = node1
        enhanced_mcp.engine.get_outgoing_relationships.return_value = []

        result = enhanced_mcp.export_subgraph(["node1"], "cypher")

        assert result["status"] == "success"
        assert result["format"] == "cypher"
        assert isinstance(result["data"], list)

    def test_bulk_create_relationships(self, enhanced_mcp):
        """Test bulk relationship creation."""
        enhanced_mcp.engine.save_relationship.return_value = "edge_123"

        relationships = [
            {
                "from_id": "node1",
                "to_id": "node2",
                "relation_type": "relates_to",
                "confidence_score": 0.8,
            },
            {
                "from_id": "node2",
                "to_id": "node3",
                "relation_type": "contains",
                # confidence_score will use default
            },
        ]

        result = enhanced_mcp.bulk_create_relationships(relationships)

        assert result["status"] == "success"
        assert result["created_count"] == 2
        assert result["failed_count"] == 0


class TestAnalyticsEndpoints:
    """Test analytics capabilities."""

    def test_analyze_knowledge_coverage(self, enhanced_mcp):
        """Test knowledge coverage analysis."""
        # Mock sample nodes
        enhanced_mcp._get_sample_nodes = Mock(return_value=["node1", "node2"])

        node1 = KnowledgeNode("Content 1", "source1")
        node1.node_id = "node1"
        node1.rating_truthfulness = 0.8
        node1.rating_richness = 0.7
        node1.rating_stability = 0.9

        enhanced_mcp.engine.get_node.return_value = node1

        result = enhanced_mcp.analyze_knowledge_coverage()

        assert result["status"] == "success"
        assert "analysis" in result
        assert "analyzed_nodes" in result

    def test_calculate_relationship_metrics(self, enhanced_mcp):
        """Test relationship metrics calculation."""
        enhanced_mcp._get_sample_nodes = Mock(return_value=["node1", "node2"])

        relationships = [
            Mock(relation_type="relates_to", confidence_score=0.8),
            Mock(relation_type="contains", confidence_score=0.9),
        ]
        enhanced_mcp.engine.get_outgoing_relationships.return_value = relationships

        result = enhanced_mcp.calculate_relationship_metrics()

        assert result["status"] == "success"
        assert "metrics" in result
        assert "analyzed_nodes" in result

    def test_analyze_quality_scores(self, enhanced_mcp):
        """Test quality score analysis."""
        enhanced_mcp._get_sample_nodes = Mock(return_value=["node1"])

        node1 = KnowledgeNode("Content", "source")
        node1.rating_truthfulness = 0.8
        node1.rating_richness = 0.7
        node1.rating_stability = 0.9
        enhanced_mcp.engine.get_node.return_value = node1

        result = enhanced_mcp.analyze_quality_scores()

        assert result["status"] == "success"
        assert "quality_analysis" in result
        assert "truthfulness" in result["quality_analysis"]
        assert "richness" in result["quality_analysis"]
        assert "stability" in result["quality_analysis"]

    def test_analyze_knowledge_evolution_no_versioning(self, enhanced_mcp):
        """Test evolution analysis without versioning."""
        enhanced_mcp.engine.revision_manager = None

        result = enhanced_mcp.analyze_knowledge_evolution()

        assert result["status"] == "error"
        assert "Versioning not enabled" in result["message"]


class TestCommandRouting:
    """Test the enhanced MCP command routing."""

    def test_execute_mcp_command_multi_hop_traversal(self, enhanced_mcp):
        """Test routing for multi-hop traversal command."""
        enhanced_mcp.multi_hop_traversal = Mock(return_value={"status": "success"})

        command = {"action": "multi_hop_traversal", "start_node_id": "node1", "max_hops": 3}

        result = enhanced_mcp.execute_mcp_command(command)

        enhanced_mcp.multi_hop_traversal.assert_called_once_with(
            start_node_id="node1", max_hops=3, relation_filter=None, min_confidence=0.0
        )
        assert result["status"] == "success"

    def test_execute_mcp_command_extract_subgraph(self, enhanced_mcp):
        """Test routing for subgraph extraction command."""
        enhanced_mcp.extract_subgraph = Mock(return_value={"status": "success"})

        command = {
            "action": "extract_subgraph",
            "topic_keywords": ["machine", "learning"],
            "max_nodes": 20,
        }

        result = enhanced_mcp.execute_mcp_command(command)

        enhanced_mcp.extract_subgraph.assert_called_once_with(
            topic_keywords=["machine", "learning"], max_nodes=20, min_relevance=0.7
        )
        assert result["status"] == "success"

    def test_execute_mcp_command_answer_question(self, enhanced_mcp):
        """Test routing for question answering command."""
        enhanced_mcp.answer_question = Mock(
            return_value={"status": "success", "answer": "Test answer"}
        )

        command = {
            "action": "answer_question",
            "question": "What is machine learning?",
            "max_hops": 3,
        }

        result = enhanced_mcp.execute_mcp_command(command)

        enhanced_mcp.answer_question.assert_called_once_with(
            question="What is machine learning?", max_hops=3, top_k_nodes=10
        )
        assert result["status"] == "success"

    def test_execute_mcp_command_bulk_operations(self, enhanced_mcp):
        """Test routing for bulk operation commands."""
        enhanced_mcp.start_bulk_ingestion = Mock(
            return_value={"status": "success", "operation_id": "123"}
        )

        command = {"action": "start_bulk_ingestion"}
        result = enhanced_mcp.execute_mcp_command(command)

        enhanced_mcp.start_bulk_ingestion.assert_called_once_with(None)
        assert result["status"] == "success"

    def test_execute_mcp_command_analytics(self, enhanced_mcp):
        """Test routing for analytics commands."""
        enhanced_mcp.analyze_knowledge_coverage = Mock(return_value={"status": "success"})

        command = {"action": "analyze_knowledge_coverage"}
        result = enhanced_mcp.execute_mcp_command(command)

        enhanced_mcp.analyze_knowledge_coverage.assert_called_once_with(None)
        assert result["status"] == "success"

    def test_execute_mcp_command_invalid_action(self, enhanced_mcp):
        """Test routing with invalid action."""
        command = {"action": "invalid_action"}
        result = enhanced_mcp.execute_mcp_command(command)

        assert result["status"] == "error"
        assert "Unknown action" in result["message"]

    def test_execute_mcp_command_missing_action(self, enhanced_mcp):
        """Test routing with missing action field."""
        command = {"data": "test"}
        result = enhanced_mcp.execute_mcp_command(command)

        assert result["status"] == "error"
        assert "'action' field is required" in result["message"]


class TestUtilityMethods:
    """Test utility and helper methods."""

    def test_convert_to_csv(self, enhanced_mcp):
        """Test CSV conversion utility."""
        data = [
            {"id": "1", "name": "test1", "value": 100},
            {"id": "2", "name": "test2", "value": 200},
        ]

        csv_output = enhanced_mcp._convert_to_csv(data)

        assert "id,name,value" in csv_output
        assert "1,test1,100" in csv_output
        assert "2,test2,200" in csv_output

    def test_convert_to_csv_empty(self, enhanced_mcp):
        """Test CSV conversion with empty data."""
        result = enhanced_mcp._convert_to_csv([])
        assert result == ""

    def test_calculate_trend(self, enhanced_mcp):
        """Test trend calculation utility."""
        # Increasing trend
        increasing_values = [1, 2, 3, 4, 5, 6]
        assert enhanced_mcp._calculate_trend(increasing_values) == "increasing"

        # Decreasing trend
        decreasing_values = [6, 5, 4, 3, 2, 1]
        assert enhanced_mcp._calculate_trend(decreasing_values) == "decreasing"

        # Stable trend
        stable_values = [5, 5, 5, 5, 5, 5]
        assert enhanced_mcp._calculate_trend(stable_values) == "stable"

        # Insufficient data
        assert enhanced_mcp._calculate_trend([1]) == "insufficient_data"

    def test_calculate_content_similarity(self, enhanced_mcp):
        """Test content similarity calculation."""
        content1 = "machine learning algorithms"
        content2 = "learning algorithms for machines"

        similarity = enhanced_mcp._calculate_content_similarity(content1, content2)

        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
        assert similarity > 0  # Should have some overlap

    def test_detect_contradiction(self, enhanced_mcp):
        """Test contradiction detection."""
        content1 = "Python is easy to learn"
        content2 = "Python is not easy to learn"

        score = enhanced_mcp._detect_contradiction(content1, content2)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score > 0.3  # Should detect contradiction


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_close_connections(self, enhanced_mcp):
        """Test closing database connections."""
        # Should not raise exceptions even if connections fail
        enhanced_mcp.vector_store.disconnect = Mock(side_effect=Exception("Connection error"))
        enhanced_mcp.engine.disconnect = Mock(side_effect=Exception("Engine error"))

        # Should complete without exceptions
        enhanced_mcp.close()

    def test_command_execution_exception_handling(self, enhanced_mcp):
        """Test exception handling in command execution."""
        enhanced_mcp.multi_hop_traversal = Mock(side_effect=Exception("Test error"))

        command = {"action": "multi_hop_traversal", "start_node_id": "node1"}
        result = enhanced_mcp.execute_mcp_command(command)

        assert result["status"] == "error"
        assert "Command execution error" in result["message"]

    def test_bulk_operation_error_handling(self, enhanced_mcp):
        """Test error handling in bulk operations."""
        # Start operation
        start_result = enhanced_mcp.start_bulk_ingestion()
        operation_id = start_result["operation_id"]

        # Mock extraction failure
        with patch(
            "memory_core.mcp_integration.enhanced_mcp_endpoint.extract_knowledge_units"
        ) as mock_extract:
            mock_extract.side_effect = Exception("Extraction failed")

            texts = [{"text": "Test content", "source": "source1"}]
            result = enhanced_mcp.add_to_bulk_ingestion(operation_id, texts)

            assert result["status"] == "success"  # Operation continues
            assert result["failed"] == 1
            assert result["processed"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
