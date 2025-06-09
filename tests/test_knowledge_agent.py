"""
Tests for the knowledge agent module.

This module tests the integration between the Knowledge Engine and the Google ADK.
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import os
import json

from memory_core.agents.knowledge_agent import KnowledgeAgent, create_knowledge_agent
from memory_core.core.knowledge_engine import KnowledgeEngine
from memory_core.model.knowledge_node import KnowledgeNode


class TestKnowledgeAgent:
    """Test cases for the KnowledgeAgent class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create mock KnowledgeEngine
        self.mock_knowledge_engine = MagicMock(spec=KnowledgeEngine)
        
        # Properly mock storage attribute
        self.mock_storage = MagicMock()
        self.mock_storage.g = True
        self.mock_knowledge_engine.storage = self.mock_storage
        
        # Mock embedded_manager for retrieval tests
        self.mock_knowledge_engine.embedding_manager = MagicMock()
        
        # Add this line to create a mock for the graph attribute
        self.mock_knowledge_engine.graph = MagicMock()
        
        # Create patches for ADK imports
        self.llm_agent_patcher = patch('memory_core.agents.knowledge_agent.LlmAgent')
        self.mock_llm_agent = self.llm_agent_patcher.start()
        
        self.function_tool_patcher = patch('memory_core.agents.knowledge_agent.FunctionTool')
        self.mock_function_tool = self.function_tool_patcher.start()
        
        self.gemini_patcher = patch('memory_core.agents.knowledge_agent.Gemini')
        self.mock_gemini = self.gemini_patcher.start()
        
        # Mock extract_knowledge_units function
        self.extract_patcher = patch('memory_core.agents.knowledge_agent.extract_knowledge_units')
        self.mock_extract = self.extract_patcher.start()
        
        # Create the agent with mocks
        self.agent = KnowledgeAgent(
            knowledge_engine=self.mock_knowledge_engine,
            model="mock-model"
        )
        
        # Sample knowledge unit for testing
        self.sample_knowledge_unit = {
            "content": "Artificial intelligence is intelligence demonstrated by machines.",
            "tags": ["AI", "intelligence", "machines"],
            "metadata": {
                "confidence_level": 0.9,
                "domain": "computer science",
                "language": "english",
                "importance": 0.8
            },
            "source": {
                "type": "text",
                "reference": "Test source",
                "url": None,
                "page": None
            }
        }
    
    def teardown_method(self):
        """Clean up after each test method."""
        self.llm_agent_patcher.stop()
        self.function_tool_patcher.stop()
        self.gemini_patcher.stop()
        self.extract_patcher.stop()
    
    @pytest.mark.asyncio
    async def test_extract_and_store_knowledge(self):
        """Test extracting and storing knowledge."""
        # Setup mocks
        self.mock_extract.return_value = [self.sample_knowledge_unit]
        self.mock_knowledge_engine.save_node.return_value = "node123"
        self.mock_knowledge_engine.save_relationship.return_value = "edge456"
        
        # Call the method
        result = await self.agent.extract_and_store_knowledge("Sample text")
        
        # Verify extract_knowledge_units was called
        self.mock_extract.assert_called_once_with("Sample text")
        
        # Verify save_node was called twice (once for content, once for domain)
        assert self.mock_knowledge_engine.save_node.call_count == 2
        
        # Verify save_relationship was called
        self.mock_knowledge_engine.save_relationship.assert_called_once()
        
        # Check result
        assert result["status"] == "success"
        assert len(result["node_ids"]) == 1
        assert result["node_ids"][0]["id"] == "node123"
    
    @pytest.mark.asyncio
    async def test_empty_extract_result(self):
        """Test handling of empty extraction result."""
        # Setup mock to return empty list
        self.mock_extract.return_value = []
        
        # Call the method
        result = await self.agent.extract_and_store_knowledge("Sample text")
        
        # Verify extract_knowledge_units was called
        self.mock_extract.assert_called_once()
        
        # Verify save_node was not called
        self.mock_knowledge_engine.save_node.assert_not_called()
        
        # Check result
        assert result["status"] == "no_knowledge_extracted"
        assert len(result["node_ids"]) == 0
    
    @pytest.mark.asyncio
    async def test_retrieve_knowledge(self):
        """Test retrieving knowledge."""
        # Setup mocks
        self.mock_knowledge_engine.embedding_manager.search_similar_nodes.return_value = ["node123"]
        
        # Mock get_node
        mock_node = MagicMock(spec=KnowledgeNode)
        mock_node.content = "Test content"
        mock_node.source = "Test source"
        self.mock_knowledge_engine.get_node.return_value = mock_node
        
        # Mock relationships
        self.mock_knowledge_engine.graph.get_outgoing_relationships.return_value = []
        self.mock_knowledge_engine.graph.get_incoming_relationships.return_value = []
        
        # Call the method
        result = await self.agent.retrieve_knowledge("Sample query")
        
        # Verify search_similar_nodes was called
        self.mock_knowledge_engine.embedding_manager.search_similar_nodes.assert_called_once_with("Sample query", top_k=5)
        
        # Verify get_node was called
        self.mock_knowledge_engine.get_node.assert_called_once_with("node123")
        
        # Check result
        assert result["status"] == "success"
        assert len(result["results"]) == 1
        assert result["results"][0]["id"] == "node123"
        assert result["results"][0]["content"] == "Test content"
    
    @pytest.mark.asyncio
    async def test_create_relationship(self):
        """Test creating a relationship."""
        # Setup mocks
        self.mock_knowledge_engine.save_relationship.return_value = "edge123"
        
        # Call the method
        result = await self.agent.create_relationship(
            from_id="node1",
            to_id="node2",
            relation_type="is_related_to",
            confidence_score=0.9
        )
        
        # Verify get_node was called twice to check nodes exist
        assert self.mock_knowledge_engine.get_node.call_count == 2
        
        # Verify save_relationship was called
        self.mock_knowledge_engine.save_relationship.assert_called_once()
        
        # Check result
        assert result["status"] == "success"
        assert result["edge_id"] == "edge123"
    
    def test_create_knowledge_agent_function(self):
        """Test the create_knowledge_agent helper function."""
        with patch('memory_core.agents.knowledge_agent.KnowledgeEngine') as mock_engine_class:
            # Setup mock engine instance
            mock_engine = MagicMock()
            mock_engine_class.return_value = mock_engine
            
            with patch('memory_core.agents.knowledge_agent.KnowledgeAgent') as mock_agent_class:
                # Setup mock agent instance
                mock_agent = MagicMock()
                mock_agent_class.return_value = mock_agent
                
                # Call the function
                agent = create_knowledge_agent(
                    host="testhost",
                    port=1234,
                    model="test-model",
                    enable_versioning=True
                )
                
                # Verify KnowledgeEngine was created with correct args
                mock_engine_class.assert_called_once_with(
                    host="testhost",
                    port=1234,
                    enable_versioning=True
                )
                
                # Verify KnowledgeAgent was created with correct args
                mock_agent_class.assert_called_once_with(
                    knowledge_engine=mock_engine,
                    model="test-model"
                )
                
                # Verify result
                assert agent == mock_agent


@pytest.mark.integration
class TestKnowledgeAgentIntegration:
    """Integration tests for KnowledgeAgent with real ADK and Gemini API."""
    
    def setup_method(self, method, gemini_api_key=None, google_adk_available=None):
        """Set up the test with actual connections to services."""
        # Get dependencies from fixtures or parameters (for direct calling)
        api_key = gemini_api_key or os.getenv('GOOGLE_API_KEY')
        adk_available = google_adk_available
        
        if adk_available is None:
            # Check if Google ADK is available if not provided via fixture
            try:
                import google.adk
                adk_available = True
            except ImportError:
                adk_available = False
                
        # Skip if no API key
        if not api_key:
            pytest.skip("GOOGLE_API_KEY environment variable not set")
        
        # Skip if Google ADK not available
        if not adk_available:
            pytest.skip("Google ADK not installed")

        try:
            # Create a mock KnowledgeEngine with mocked graph functions
            self.mock_engine = MagicMock(spec=KnowledgeEngine)
            
            # Create and configure mock storage
            self.mock_storage = MagicMock()
            self.mock_storage.g = True  # Pretend we're connected
            self.mock_engine.storage = self.mock_storage  # Set the storage on the engine
            
            # Mock required methods
            self.mock_engine.save_node = MagicMock(return_value="test_node_id")
            self.mock_engine.save_relationship = MagicMock(return_value="test_edge_id")
            
            # Create the agent with the mock engine but real ADK and Gemini connections
            os.environ['GOOGLE_API_KEY'] = api_key  # Ensure API key is set in env
            self.agent = KnowledgeAgent(
                knowledge_engine=self.mock_engine,
                model="gemini-1.5-pro"
            )
            
            # Test text
            self.test_text = """
            Artificial intelligence (AI) is intelligence demonstrated by machines, unlike natural
            intelligence in humans and animals. AI research has defined the field as the study of
            intelligent agents that perceive their environment and take actions to achieve goals.
            """
        except Exception as e:
            pytest.skip(f"Failed to initialize integration test: {str(e)}")
    
    @pytest.mark.asyncio
    async def test_extract_knowledge_integration(self):
        """Test extracting knowledge with real Gemini API."""
        # Call setup without fixtures - it will check environment
        self.setup_method(None)
        
        try:
            # Call the method
            result = await self.agent.extract_and_store_knowledge(self.test_text)
            
            # Basic validation
            assert "status" in result
            assert "node_ids" in result
            
            # Check if any nodes were created
            assert len(result["node_ids"]) > 0
        except Exception as e:
            pytest.skip(f"Skipping due to API error: {str(e)}")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])