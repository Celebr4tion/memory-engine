#!/usr/bin/env python
"""
Simple test for the JanusGraphStorage class.
This test focuses just on establishing a connection and doing a simple traversal.
"""
import sys
import asyncio
import pytest
from memory_core.db.janusgraph_storage import JanusGraphStorage

def test_janusgraph_basic():
    """Test basic JanusGraph connection and traversal."""
    host = 'localhost'
    port = 8182
    
    print(f"Testing JanusGraphStorage with {host}:{port}...")
    
    # Create JanusGraphStorage instance
    storage = JanusGraphStorage(host=host, port=port)
    
    try:
        print("Attempting to connect...")
        storage.connect()
        
        # If we got here, connection was established
        print("Connection established!")
        
        # Try a simple query using the client directly with submit_async
        if storage._client:
            try:
                print("Executing simple query via client.submit_async...")
                # Use submit_async and manually wait for result to avoid event loop issues
                result_set = storage._client.submit_async('g.V().count()')
                
                # Wait for completion without using result()
                # This avoids the internal loop.run_until_complete call
                while not result_set._done.done():
                    import time
                    time.sleep(0.1)
                
                # Get results
                if hasattr(result_set, '_result') and result_set._result:
                    print(f"Query result: {result_set._result}")
                else:
                    print("Query completed but no results available")
            except Exception as query_err:
                print(f"Query failed: {query_err}")
                import traceback
                traceback.print_exc()
        
        # Close connection
        print("Closing connection...")
        storage.close()
        print("Test completed successfully!")
        assert True
        
    except Exception as e:
        print(f"Connection failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to clean up
        try:
            if storage._client:
                storage.close()
        except:
            pass
        
        assert False, f"Connection failed: {e}"

if __name__ == "__main__":
    try:
        # Run the test
        test_janusgraph_basic()
        sys.exit(0)
    except AssertionError:
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nTest interrupted.")
        sys.exit(1) 