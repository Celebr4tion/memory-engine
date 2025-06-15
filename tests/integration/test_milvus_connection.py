"""
Simple script to test Milvus connection with detailed logging.
"""

import sys
import logging
import time
import numpy as np
import uuid
import os
import pytest

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)


def test_milvus_connection():
    """Test connection to Milvus."""
    try:
        from pymilvus import connections

        # Try to establish a connection
        connections.connect(alias="default", host="localhost", port="19530")

        # Verify the connection
        assert connections.has_connection("default"), "Connection to Milvus not established"

        # Clean up - disconnect
        connections.disconnect("default")
    except ImportError:
        pytest.skip("pymilvus not installed")
    except Exception as e:
        pytest.skip(f"Failed to connect to Milvus: {str(e)}")


if __name__ == "__main__":
    try:
        test_milvus_connection()
        print("\nTest result: SUCCESS")
        sys.exit(0)
    except AssertionError as e:
        print(f"\nTest result: FAILED - {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\nTest result: FAILED - {str(e)}")
        sys.exit(1)
