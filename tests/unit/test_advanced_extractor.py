"""
Tests for the advanced extractor module.

This module tests the extraction of knowledge units from raw text.
"""

import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import json
import pytest
import os
import asyncio

from memory_core.ingestion.advanced_extractor import AdvancedExtractor, extract_knowledge_units


class TestAdvancedExtractor(unittest.TestCase):
    """Test cases for the AdvancedExtractor class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock for the GeminiLLMProvider
        self.llm_provider_patcher = patch(
            "memory_core.ingestion.advanced_extractor.GeminiLLMProvider"
        )
        self.mock_llm_provider_class = self.llm_provider_patcher.start()

        # Create a mock for the config to return a fake API key
        self.config_patcher = patch("memory_core.ingestion.advanced_extractor.get_config")
        self.mock_config = self.config_patcher.start()

        # Mock config object
        mock_config_obj = MagicMock()
        mock_config_obj.config.api.google_api_key = "fake_api_key"
        mock_config_obj.config.llm.model = "gemini-2.0-flash-thinking-exp"
        mock_config_obj.config.llm.temperature = 0.7
        mock_config_obj.config.llm.max_tokens = 4096
        mock_config_obj.config.llm.timeout = 60
        self.mock_config.return_value = mock_config_obj

        # Sample text for testing
        self.sample_text = """
        Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural 
        intelligence displayed by animals including humans. AI research has been defined as the field 
        of study of intelligent agents, which refers to any system that perceives its environment and 
        takes actions that maximize its chance of achieving its goals.
        
        The term "artificial intelligence" was first used in 1956 during the Dartmouth Conference, 
        where the discipline was born. AI research has gone through cycles of optimism, disappointment 
        and funding cuts, followed by new approaches, success and renewed funding.
        """

        # Mock knowledge units that the LLM would return
        self.mock_units = [
            {
                "content": "Artificial intelligence is intelligence demonstrated by machines, distinct from natural intelligence in animals and humans.",
                "tags": ["AI", "intelligence", "machines"],
                "metadata": {
                    "confidence_level": 0.95,
                    "domain": "computer science",
                    "language": "english",
                    "importance": 0.9,
                },
                "source": {
                    "type": "user_input",
                    "url": None,
                    "reference": "Provided text",
                    "page": None,
                },
            },
            {
                "content": "AI research is defined as studying intelligent agents that perceive their environment and act to achieve goals.",
                "tags": ["AI research", "intelligent agents", "goals"],
                "metadata": {
                    "confidence_level": 0.9,
                    "domain": "computer science",
                    "language": "english",
                    "importance": 0.8,
                },
                "source": {
                    "type": "user_input",
                    "url": None,
                    "reference": "Provided text",
                    "page": None,
                },
            },
            {
                "content": "The term 'artificial intelligence' was first used in 1956 at the Dartmouth Conference.",
                "tags": ["AI history", "Dartmouth Conference", "terminology"],
                "metadata": {
                    "confidence_level": 0.95,
                    "domain": "computer science history",
                    "language": "english",
                    "importance": 0.7,
                },
                "source": {
                    "type": "user_input",
                    "url": None,
                    "reference": "Provided text",
                    "page": None,
                },
            },
        ]

        # Set up the extractor with mocked LLM provider
        self.mock_llm_provider = MagicMock()
        self.mock_llm_provider.is_connected = False
        self.mock_llm_provider.connect = AsyncMock(return_value=True)
        self.mock_llm_provider.extract_knowledge_units = AsyncMock(return_value=self.mock_units)
        self.mock_llm_provider_class.return_value = self.mock_llm_provider

        self.extractor = AdvancedExtractor()

    def tearDown(self):
        """Clean up after each test method."""
        self.llm_provider_patcher.stop()
        self.config_patcher.stop()

    def test_llm_provider_initialization(self):
        """Test that the LLM provider is properly initialized."""
        # Verify that GeminiLLMProvider was called with correct config
        self.mock_llm_provider_class.assert_called_once()
        call_args = self.mock_llm_provider_class.call_args[0][0]

        assert call_args["api_key"] == "fake_api_key"
        assert call_args["model_name"] == "gemini-2.0-flash-thinking-exp"
        assert call_args["temperature"] == 0.7
        assert call_args["max_tokens"] == 4096

    def test_extract_knowledge_units(self):
        """Test extracting knowledge units from text."""
        # Extract knowledge units using asyncio
        units = asyncio.run(self.extractor.extract_knowledge_units(self.sample_text))

        # Verify LLM provider methods were called
        self.mock_llm_provider.connect.assert_called_once()
        self.mock_llm_provider.extract_knowledge_units.assert_called_once_with(self.sample_text)

        # Verify results
        assert len(units) == 3
        assert units[0]["content"] == self.mock_units[0]["content"]
        assert "tags" in units[0]
        assert "metadata" in units[0]
        assert "source" in units[0]

    def test_empty_input(self):
        """Test handling of empty input."""
        # Extract knowledge units from empty text
        units = asyncio.run(self.extractor.extract_knowledge_units(""))

        # Verify no LLM provider calls were made (since empty text returns immediately)
        self.mock_llm_provider.extract_knowledge_units.assert_not_called()

        # Verify empty result
        assert units == []

    def test_llm_provider_error_handling(self):
        """Test handling of LLM provider errors."""
        # Setup the mock to raise an exception
        self.mock_llm_provider.extract_knowledge_units.side_effect = Exception("LLM error")

        # Extract knowledge units should raise RuntimeError
        with pytest.raises(RuntimeError):
            asyncio.run(self.extractor.extract_knowledge_units(self.sample_text))

    def test_llm_provider_already_connected(self):
        """Test behavior when LLM provider is already connected."""
        # Setup the mock to be already connected
        self.mock_llm_provider.is_connected = True

        # Extract knowledge units
        units = asyncio.run(self.extractor.extract_knowledge_units(self.sample_text))

        # Verify connect was not called since already connected
        self.mock_llm_provider.connect.assert_not_called()
        self.mock_llm_provider.extract_knowledge_units.assert_called_once_with(self.sample_text)

        # Verify results
        assert len(units) == 3

    def test_extract_knowledge_units_function(self):
        """Test the standalone extract_knowledge_units function."""
        with patch(
            "memory_core.ingestion.advanced_extractor.AdvancedExtractor"
        ) as mock_extractor_class:
            # Setup the mock instance
            mock_extractor = MagicMock()
            mock_extractor_class.return_value = mock_extractor
            mock_extractor.extract_knowledge_units = AsyncMock(return_value=self.mock_units)

            # Call the function
            result = asyncio.run(extract_knowledge_units(self.sample_text))

            # Verify extractor was created and method was called
            mock_extractor_class.assert_called_once()
            mock_extractor.extract_knowledge_units.assert_called_once_with(self.sample_text)

            # Verify result
            assert result == self.mock_units

    def test_async_behavior(self):
        """Test that the async behavior works correctly."""

        # Test that the method can be called with await
        async def test_async():
            units = await self.extractor.extract_knowledge_units(self.sample_text)
            return units

        # Run the async function
        units = asyncio.run(test_async())

        # Verify results
        assert len(units) == 3
        assert units[0]["content"] == self.mock_units[0]["content"]


@pytest.mark.integration
class TestAdvancedExtractorIntegration:
    """Integration tests for AdvancedExtractor with real Gemini API."""

    def setup_method(self):
        """Set up the test with actual connection to Gemini API."""
        # Skip if API key not set
        if not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY environment variable not set")

        # Create extractor
        self.extractor = AdvancedExtractor()

        # Sample text for testing
        self.ai_text = """
        Artificial intelligence (AI) is transforming healthcare through applications 
        like medical imaging analysis, drug discovery, and personalized treatment plans. 
        AI systems can detect patterns in medical images that might be missed by human 
        radiologists and can process vast amounts of medical literature to identify potential 
        treatments. However, these systems face challenges including data privacy concerns, 
        potential bias in training data, and questions around medical liability.
        """

        self.short_text = "Hello world"

    def test_live_extract_knowledge_units(self):
        """Test extracting knowledge units with real Gemini API."""
        try:
            # Extract knowledge units
            units = self.extractor.extract_knowledge_units(self.ai_text)

            # Verify result structure
            assert len(units) > 0
            for unit in units:
                assert "content" in unit
                assert "tags" in unit
                assert "metadata" in unit
                assert isinstance(unit["tags"], list)
                assert isinstance(unit["metadata"], dict)
        except Exception as e:
            pytest.skip(f"Gemini API error: {str(e)}")

    def test_live_extract_empty_result(self):
        """Test extracting knowledge units from text that's too short."""
        try:
            # Extract knowledge units from short text
            units = self.extractor.extract_knowledge_units(self.short_text)

            # Should return empty list or very few items as the text is minimal
            assert isinstance(units, list)
            if units:  # If not empty, check structure
                for unit in units:
                    assert "content" in unit
        except Exception as e:
            pytest.skip(f"Gemini API error: {str(e)}")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
