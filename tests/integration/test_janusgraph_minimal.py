#!/usr/bin/env python
"""
Minimal test for JanusGraph connection.
Just verifies that the connection object can be created without errors.
"""
import sys
import socket
from memory_core.db.janusgraph_storage import JanusGraphStorage


def test_janusgraph_minimal():
    """Minimal test for JanusGraph connectivity."""
    host = "localhost"
    port = 8182

    print(f"Testing if JanusGraph port is open at {host}:{port}...")

    # First check if the port is open with a socket test
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5)

    try:
        s.connect((host, port))
        print(f"Socket connection to {host}:{port} succeeded.")
        s.close()
    except socket.error as e:
        print(f"Socket connection to {host}:{port} failed: {e}")
        s.close()
        assert False, f"Socket connection to {host}:{port} failed: {e}"
    finally:
        s.close()

    # Now verify that we can create a JanusGraphStorage instance without errors
    storage = JanusGraphStorage(host=host, port=port)
    print(f"Created JanusGraphStorage instance: {storage}")

    # Success if we got here
    print("Test successful: Connection object can be created.")
    assert True


if __name__ == "__main__":
    try:
        test_janusgraph_minimal()
        sys.exit(0)
    except AssertionError:
        sys.exit(1)
