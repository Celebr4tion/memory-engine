#!/usr/bin/env python
"""
Simple test script to verify connection to JanusGraph.
This script only checks if the server is listening on the expected port.
"""
import sys
import socket

def test_janusgraph_port():
    """Simple test to verify JanusGraph is listening on expected port."""
    host = 'localhost'
    port = 8182
    timeout = 5  # seconds
    
    print(f"Testing connection to JanusGraph at {host}:{port}...")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    
    try:
        s.connect((host, port))
        print(f"SUCCESS: JanusGraph is running at {host}:{port}")
        s.close()
        return True
    except socket.error as e:
        print(f"FAILED: Cannot connect to JanusGraph at {host}:{port}: {e}")
        s.close()
        return False

if __name__ == "__main__":
    result = test_janusgraph_port()
    sys.exit(0 if result else 1) 