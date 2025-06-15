"""
Tests for the MCP endpoint functionality.

This module tests the MCP interface that allows external systems to interact
with the Memory Engine.
"""

import unittest
from unittest.mock import MagicMock, patch
import json
import time

from memory_core.mcp_integration.mcp_endpoint import MemoryEngineMCP


class TestMemoryEngineMCP(unittest.TestCase):
    """Tests for the MemoryEngineMCP class."""

    def setUp(self):
        """Set up test dependencies."""
        # Patch KnowledgeEngine
        self.engine_patcher = patch("memory_core.mcp_integration.mcp_endpoint.KnowledgeEngine")
        self.mock_engine_class = self.engine_patcher.start()

        # Set up mock engine instance
        self.mock_engine = MagicMock()
        self.mock_engine_class.return_value = self.mock_engine

        # Set up mock storage
        self.mock_storage = MagicMock()
        self.mock_engine.storage = self.mock_storage

        # Patch VectorStoreMilvus
        self.vector_store_patcher = patch(
            "memory_core.mcp_integration.mcp_endpoint.VectorStoreMilvus"
        )
        self.mock_vector_store_class = self.vector_store_patcher.start()

        # Set up mock vector store instance
        self.mock_vector_store = MagicMock()
        self.mock_vector_store_class.return_value = self.mock_vector_store
        self.mock_vector_store.connect.return_value = True

        # Patch EmbeddingManager
        self.embedding_manager_patcher = patch(
            "memory_core.mcp_integration.mcp_endpoint.EmbeddingManager"
        )
        self.mock_embedding_manager_class = self.embedding_manager_patcher.start()

        # Set up mock embedding manager instance
        self.mock_embedding_manager = MagicMock()
        self.mock_embedding_manager_class.return_value = self.mock_embedding_manager

        # Patch knowledge extraction function
        self.extract_knowledge_patcher = patch(
            "memory_core.mcp_integration.mcp_endpoint.extract_knowledge_units"
        )
        self.mock_extract_knowledge = self.extract_knowledge_patcher.start()
        self.mock_extract_knowledge.return_value = [
            {"content": "Knowledge unit 1", "tags": "ai,test"},
            {"content": "Knowledge unit 2", "tags": "memory,test"},
        ]

        # Patch process_extracted_units function
        self.process_units_patcher = patch(
            "memory_core.mcp_integration.mcp_endpoint.process_extracted_units"
        )
        self.mock_process_units = self.process_units_patcher.start()
        self.mock_process_units.return_value = ["node1", "node2"]

        # Patch analyze_and_create_relationships function
        self.create_relationships_patcher = patch(
            "memory_core.mcp_integration.mcp_endpoint.analyze_and_create_relationships"
        )
        self.mock_create_relationships = self.create_relationships_patcher.start()
        self.mock_create_relationships.return_value = {
            "tag_relationships": ["rel1"],
            "domain_relationships": ["rel2"],
            "semantic_relationships": ["rel3"],
        }

        # Create MCP instance
        self.mcp = MemoryEngineMCP()

    def tearDown(self):
        """Clean up patches."""
        self.engine_patcher.stop()
        self.vector_store_patcher.stop()
        self.embedding_manager_patcher.stop()
        self.extract_knowledge_patcher.stop()
        self.process_units_patcher.stop()
        self.create_relationships_patcher.stop()

    def test_initialization(self):
        """Test that MCP is initialized correctly."""
        # Verify KnowledgeEngine was created and connected
        self.mock_engine_class.assert_called_once()
        self.mock_engine.connect.assert_called_once()

        # Verify vector store was created and connected
        self.mock_vector_store_class.assert_called_once()
        self.mock_vector_store.connect.assert_called_once()

        # Verify embedding manager was created
        self.mock_embedding_manager_class.assert_called_once_with(self.mock_vector_store)

    def test_ingest_raw_text(self):
        """Test ingesting raw text through the MCP endpoint."""
        # Call the method
        result = self.mcp.ingest_raw_text(
            "This is some test text to ingest.", source_label="Test Source"
        )

        # Verify knowledge extraction was called with raw text
        self.mock_extract_knowledge.assert_called_once_with("This is some test text to ingest.")

        # Verify extracted units were processed
        self.mock_process_units.assert_called_once()
        process_args, process_kwargs = self.mock_process_units.call_args
        self.assertEqual(process_kwargs["source_label"], "Test Source")

        # Verify relationships were analyzed
        self.mock_create_relationships.assert_called_once_with(
            ["node1", "node2"], self.mock_storage, self.mock_embedding_manager
        )

        # Verify result structure
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["created_or_merged_node_ids"], ["node1", "node2"])
        self.assertIn("relationship_counts", result)

    def test_ingest_raw_text_with_no_knowledge(self):
        """Test ingesting text that doesn't yield any knowledge units."""
        # Configure mock to return empty list
        self.mock_extract_knowledge.return_value = []

        # Call the method
        result = self.mcp.ingest_raw_text("No knowledge here.", source_label="Empty Source")

        # Verify knowledge extraction was called
        self.mock_extract_knowledge.assert_called_once()

        # Verify processing was not called
        self.mock_process_units.assert_not_called()

        # Verify relationships were not analyzed
        self.mock_create_relationships.assert_not_called()

        # Verify result structure
        self.assertEqual(result["status"], "no_knowledge_extracted")
        self.assertEqual(result["created_or_merged_node_ids"], [])

    def test_search_text(self):
        """Test semantic search through the MCP endpoint."""
        # Configure mock embedding manager search
        self.mock_embedding_manager.search_similar_nodes.return_value = ["node1", "node2"]

        # Configure mock engine.get_node to return proper node objects with content attribute
        node1 = MagicMock()
        node1.content = "Node 1 content"
        node1.source = "Source 1"
        node1.rating_truthfulness = 0.8

        node2 = MagicMock()
        # Make sure the content is long enough to be truncated (> 100 chars)
        node2.content = "Node 2 with longer content that should be truncated in the preview. This text needs to be over 100 characters to ensure the truncation logic works properly in the test case."
        node2.source = "Source 2"
        node2.rating_truthfulness = 0.6

        def mock_get_node(node_id):
            if node_id == "node1":
                return node1
            elif node_id == "node2":
                return node2
            return {}

        self.mock_engine.get_node.side_effect = mock_get_node

        # Call the method
        result = self.mcp.search_text("AI concepts", top_k=5)

        # Verify embedding manager search was called
        self.mock_embedding_manager.search_similar_nodes.assert_called_once_with("AI concepts", 5)

        # Verify node details were retrieved
        self.assertEqual(self.mock_engine.get_node.call_count, 2)

        # Verify result structure
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["query"], "AI concepts")
        self.assertEqual(len(result["results"]), 2)

        # Check first result
        self.assertEqual(result["results"][0]["node_id"], "node1")
        self.assertEqual(result["results"][0]["content_preview"], "Node 1 content")

        # Check second result (should have truncated content)
        self.assertEqual(result["results"][1]["node_id"], "node2")
        self.assertTrue(len(result["results"][1]["content_preview"]) < len(node2.content))
        self.assertTrue(result["results"][1]["content_preview"].endswith("..."))

    def test_search_text_with_no_embedding_manager(self):
        """Test search behavior when no embedding manager is available."""
        # Create new MCP instance with no embedding manager
        with patch(
            "memory_core.mcp_integration.mcp_endpoint.VectorStoreMilvus"
        ) as mock_vector_store_class:
            # Make vector store connection fail
            mock_vector_store = MagicMock()
            mock_vector_store_class.return_value = mock_vector_store
            mock_vector_store.connect.return_value = False

            # Create MCP instance (should have no embedding manager)
            mcp_no_embeddings = MemoryEngineMCP()

            # Call the method
            result = mcp_no_embeddings.search_text("AI concepts")

            # Verify error result
            self.assertEqual(result["status"], "error")
            self.assertIn("message", result)
            self.assertEqual(result["results"], [])

    def test_get_node_details(self):
        """Test retrieving node details through the MCP endpoint."""
        # Create a proper mock node with to_dict method
        mock_node = MagicMock()
        mock_node.to_dict.return_value = {
            "node_id": "test123",
            "content": "Test content",
            "source": "Test Source",
            "rating_truthfulness": 0.8,
        }

        # Configure mock engine get_node
        self.mock_engine.get_node.return_value = mock_node

        # Configure mock engine get_relationships
        self.mock_engine.get_outgoing_relationships.return_value = [
            MagicMock(
                edge_id="edge1", to_id="related1", relation_type="RELATED", confidence_score=0.7
            )
        ]

        self.mock_engine.get_incoming_relationships.return_value = [
            MagicMock(
                edge_id="edge2",
                from_id="related2",
                relation_type="SEMANTICALLY_SIMILAR",
                confidence_score=0.8,
            )
        ]

        # Call the method
        result = self.mcp.get_node_details("test123")

        # Verify engine methods were called
        self.mock_engine.get_node.assert_called_once_with("test123")
        self.mock_engine.get_outgoing_relationships.assert_called_once_with("test123")
        self.mock_engine.get_incoming_relationships.assert_called_once_with("test123")

        # Verify result structure
        self.assertEqual(result["status"], "success")
        self.assertIn("node", result)
        self.assertEqual(result["node"]["node_id"], "test123")

        # Verify relationships
        self.assertEqual(len(result["outgoing_relationships"]), 1)
        self.assertEqual(result["outgoing_relationships"][0]["target_id"], "related1")

        self.assertEqual(len(result["incoming_relationships"]), 1)
        self.assertEqual(result["incoming_relationships"][0]["source_id"], "related2")

    def test_update_node_rating(self):
        """Test updating node rating through the MCP endpoint."""
        # Create a proper mock node with rating attributes
        mock_node = MagicMock()
        mock_node.node_id = "test123"
        mock_node.content = "Test content"
        mock_node.rating_truthfulness = 0.5
        mock_node.rating_richness = 0.5
        mock_node.rating_stability = 0.5
        mock_node.to_dict.return_value = {
            "node_id": "test123",
            "content": "Test content",
            "rating_truthfulness": 0.7,  # Updated value
            "rating_richness": 0.7,  # Updated value
            "rating_stability": 0.5,  # Unchanged
        }

        # Configure mock engine get_node
        self.mock_engine.get_node.return_value = mock_node

        # Mock the get_node_details method to return a successful result
        with patch.object(self.mcp, "get_node_details") as mock_get_details:
            mock_get_details.return_value = {
                "status": "success",
                "node": mock_node.to_dict(),
                "outgoing_relationships": [],
                "incoming_relationships": [],
            }

            # Call the method
            result = self.mcp.update_node_rating("test123", {"confirmation": 1, "richness": 0.5})

            # Verify node was retrieved
            self.mock_engine.get_node.assert_called_once_with("test123")

            # Verify node was saved with updated ratings
            self.mock_engine.save_node.assert_called_once_with(mock_node)

            # Verify get_node_details was called
            mock_get_details.assert_called_once_with("test123")

            # Verify result was returned
            self.assertEqual(result, mock_get_details.return_value)

    def test_execute_mcp_command_ingest_text(self):
        """Test executing ingest_text MCP command."""
        # Patch ingest_raw_text
        with patch.object(self.mcp, "ingest_raw_text") as mock_ingest:
            mock_ingest.return_value = {
                "status": "success",
                "created_or_merged_node_ids": ["node1"],
            }

            # Call execute_mcp_command with ingest_text action
            result = self.mcp.execute_mcp_command(
                {"action": "ingest_text", "text": "Text to ingest", "source": "Command Source"}
            )

            # Verify ingest_raw_text was called
            mock_ingest.assert_called_once_with("Text to ingest", "Command Source")

            # Verify result is passed through
            self.assertEqual(result, mock_ingest.return_value)

    def test_execute_mcp_command_search(self):
        """Test executing search MCP command."""
        # Patch search_text
        with patch.object(self.mcp, "search_text") as mock_search:
            mock_search.return_value = {"status": "success", "results": [{"node_id": "node1"}]}

            # Call execute_mcp_command with search action
            result = self.mcp.execute_mcp_command(
                {"action": "search", "query": "Search query", "top_k": 10}
            )

            # Verify search_text was called
            mock_search.assert_called_once_with("Search query", 10)

            # Verify result is passed through
            self.assertEqual(result, mock_search.return_value)

    def test_execute_mcp_command_missing_fields(self):
        """Test executing commands with missing required fields."""
        # Test ingest_text without text
        result = self.mcp.execute_mcp_command(
            {
                "action": "ingest_text"
                # Missing 'text' field
            }
        )

        # Verify error result
        self.assertEqual(result["status"], "error")
        self.assertIn("message", result)

        # Test search without query
        result = self.mcp.execute_mcp_command(
            {
                "action": "search"
                # Missing 'query' field
            }
        )

        # Verify error result
        self.assertEqual(result["status"], "error")
        self.assertIn("message", result)

    def test_execute_mcp_command_unknown_action(self):
        """Test executing command with unknown action."""
        result = self.mcp.execute_mcp_command({"action": "unknown_action", "data": "some data"})

        # Verify error result
        self.assertEqual(result["status"], "error")
        self.assertIn("message", result)
        self.assertIn("unknown_action", result["message"])


if __name__ == "__main__":
    unittest.main()
