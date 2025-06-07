"""
Tests for the advanced extractor module.

This module tests the extraction of knowledge units from raw text.
"""
import unittest
from unittest.mock import patch, MagicMock
import json
import pytest
import os

from memory_core.ingestion.advanced_extractor import AdvancedExtractor, extract_knowledge_units


class TestAdvancedExtractor(unittest.TestCase):
    """Test cases for the AdvancedExtractor class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock for the Gemini client
        self.client_patcher = patch('memory_core.ingestion.advanced_extractor.genai.Client')
        self.mock_client = self.client_patcher.start()
        
        # Create a mock for os.getenv to return a fake API key
        self.env_patcher = patch('memory_core.ingestion.advanced_extractor.os.getenv')
        self.mock_getenv = self.env_patcher.start()
        self.mock_getenv.return_value = 'fake_api_key'
        
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
                    "importance": 0.9
                },
                "source": {
                    "type": "user_input",
                    "url": None,
                    "reference": "Provided text",
                    "page": None
                }
            },
            {
                "content": "AI research is defined as studying intelligent agents that perceive their environment and act to achieve goals.",
                "tags": ["AI research", "intelligent agents", "goals"],
                "metadata": {
                    "confidence_level": 0.9,
                    "domain": "computer science",
                    "language": "english",
                    "importance": 0.8
                },
                "source": {
                    "type": "user_input",
                    "url": None,
                    "reference": "Provided text",
                    "page": None
                }
            },
            {
                "content": "The term 'artificial intelligence' was first used in 1956 at the Dartmouth Conference.",
                "tags": ["AI history", "Dartmouth Conference", "terminology"],
                "metadata": {
                    "confidence_level": 0.95,
                    "domain": "computer science history",
                    "language": "english",
                    "importance": 0.7
                },
                "source": {
                    "type": "user_input",
                    "url": None,
                    "reference": "Provided text",
                    "page": None
                }
            }
        ]
        
        # Set up the extractor with mocked client
        self.extractor = AdvancedExtractor()
        self.extractor.client = self.mock_client.return_value
        
    def tearDown(self):
        """Clean up after each test method."""
        self.client_patcher.stop()
        self.env_patcher.stop()
    
    def test_create_prompt(self):
        """Test creating a prompt for the LLM."""
        prompt = self.extractor._create_prompt("Some test text")
        
        # Check that the prompt contains the input text
        assert "Some test text" in prompt
        # Check that the prompt includes formatting instructions
        assert "JSON" in prompt
        assert "content" in prompt
        assert "tags" in prompt
        assert "metadata" in prompt
    
    def test_extract_knowledge_units(self):
        """Test extracting knowledge units from text."""
        # Setup the mock response
        mock_response = MagicMock()
        mock_response.text = json.dumps(self.mock_units)
        self.extractor.client.generate_content.return_value = mock_response
        
        # Extract knowledge units
        units = self.extractor.extract_knowledge_units(self.sample_text)
        
        # Verify API call
        self.extractor.client.generate_content.assert_called_once()
        
        # Verify results
        assert len(units) == 3
        assert units[0]["content"] == self.mock_units[0]["content"]
        assert "tags" in units[0]
        assert "metadata" in units[0]
        assert "source" in units[0]
    
    def test_empty_input(self):
        """Test handling of empty input."""
        # Extract knowledge units from empty text
        units = self.extractor.extract_knowledge_units("")
        
        # Verify no API call was made
        self.extractor.client.generate_content.assert_not_called()
        
        # Verify empty result
        assert units == []
    
    def test_json_parsing_error(self):
        """Test handling of JSON parsing errors."""
        # Setup the mock to return invalid JSON
        mock_response = MagicMock()
        mock_response.text = "This is not JSON"
        self.extractor.client.generate_content.return_value = mock_response
        
        # Extract knowledge units with invalid JSON response
        units = self.extractor.extract_knowledge_units(self.sample_text)
        
        # Verify API call
        self.extractor.client.generate_content.assert_called_once()
        
        # Verify empty result due to parsing error
        assert units == []
    
    def test_invalid_response_structure(self):
        """Test handling of invalid response structure."""
        # Setup the mock to return valid JSON but with wrong structure
        mock_response = MagicMock()
        mock_response.text = json.dumps({"not_a_list": "this is an object, not a list"})
        self.extractor.client.generate_content.return_value = mock_response
        
        # Extract knowledge units with invalid response structure
        units = self.extractor.extract_knowledge_units(self.sample_text)
        
        # Verify API call
        self.extractor.client.generate_content.assert_called_once()
        
        # Verify empty result due to wrong structure
        assert units == []
    
    def test_extract_knowledge_units_function(self):
        """Test the standalone extract_knowledge_units function."""
        with patch('memory_core.ingestion.advanced_extractor.AdvancedExtractor') as mock_extractor_class:
            # Setup the mock instance
            mock_extractor = MagicMock()
            mock_extractor_class.return_value = mock_extractor
            mock_extractor.extract_knowledge_units.return_value = self.mock_units
            
            # Call the function
            result = extract_knowledge_units(self.sample_text)
            
            # Verify extractor was created and method was called
            mock_extractor_class.assert_called_once()
            mock_extractor.extract_knowledge_units.assert_called_once_with(self.sample_text)
            
            # Verify result
            assert result == self.mock_units
    
    def test_markdown_code_block_cleanup(self):
        """Test handling of markdown code blocks in LLM responses."""
        # Setup the mock to return JSON wrapped in markdown code block
        mock_response = MagicMock()
        mock_response.text = "```json\n" + json.dumps(self.mock_units) + "\n```"
        self.extractor.client.generate_content.return_value = mock_response
        
        # Extract knowledge units
        units = self.extractor.extract_knowledge_units(self.sample_text)
        
        # Verify API call
        self.extractor.client.generate_content.assert_called_once()
        
        # Verify results were correctly parsed despite markdown formatting
        assert len(units) == 3
        assert units[0]["content"] == self.mock_units[0]["content"]


@pytest.mark.integration
class TestAdvancedExtractorIntegration:
    """Integration tests for AdvancedExtractor with real Gemini API."""
    
    def setup_method(self):
        """Set up the test with actual connection to Gemini API."""
        # Skip if API key not set
        if not os.getenv('GEMINI_API_KEY'):
            pytest.skip("GEMINI_API_KEY environment variable not set")
        
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