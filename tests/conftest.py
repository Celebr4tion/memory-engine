"""
Configuration file for pytest.

This file configures pytest to properly load environment variables
and provides shared fixtures for tests.
"""

import os
import pytest
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()


# Add fixtures to check if external services are available
@pytest.fixture(scope="session")
def milvus_available():
    """Check if Milvus is available."""
    try:
        from pymilvus import connections

        connections.connect(
            alias="default",
            host="localhost",
            port="19530",
            timeout=5.0,  # Short timeout for quick detection
        )
        connections.disconnect("default")
        return True
    except Exception as e:
        print(f"Milvus not available: {str(e)}")
        return False


@pytest.fixture(scope="session")
def janusgraph_available():
    """Check if JanusGraph is available."""
    try:
        from memory_core.db.janusgraph_storage import JanusGraphStorage

        return JanusGraphStorage.is_available(timeout=5)
    except Exception as e:
        print(f"JanusGraph not available: {str(e)}")
        return False


@pytest.fixture(scope="session")
def gemini_api_key():
    """Check if Gemini API key is available."""
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key and api_key.strip() and api_key != "None":
        return api_key
    return None


@pytest.fixture(scope="session")
def google_adk_available():
    """Check if Google ADK is available."""
    try:
        import google.adk

        return True
    except ImportError:
        return False


@pytest.fixture(scope="session")
def embedding_model_info():
    """Provide information about the embedding model."""
    return {"model_name": "gemini-embedding-exp-03-07", "dimension": 3072}


@pytest.fixture(scope="session")
def llm_model_info():
    """Provide information about the LLM model."""
    return {"model_name": "gemini-2.5-pro-exp-03-25"}
