"""
Simple script to test Milvus connection with detailed logging.
"""
import sys
import logging
import time
import numpy as np
import uuid
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def test_milvus_connection():
    """Test connection to Milvus with detailed logging."""
    try:
        # Import pymilvus and check if it's available
        from pymilvus import connections, utility, Collection
        logger.info("PyMilvus is available, testing connection...")
        
        # Connection parameters
        host = "localhost"
        port = 19530
        alias = "default"
        
        # Try to connect
        logger.info(f"Connecting to Milvus at {host}:{port}...")
        connections.connect(alias=alias, host=host, port=port, timeout=10)
        
        # Connection info
        if connections.has_connection(alias):
            addr = connections.get_connection_addr(alias)
            logger.info(f"Connection successful! Details: {addr}")
        else:
            logger.error("Connection failed: has_connection() returned False")
            return False
        
        # List all collections
        logger.info("Listing all collections...")
        collections = utility.list_collections()
        logger.info(f"Found collections: {collections}")
        
        # Create a temporary collection for testing
        collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"
        dimension = 128
        
        logger.info(f"Creating temporary collection '{collection_name}' with dimension {dimension}...")
        
        try:
            from pymilvus import FieldSchema, CollectionSchema, DataType
            
            # Define fields
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension)
            ]
            
            # Create schema and collection
            schema = CollectionSchema(fields=fields, description="Test collection")
            collection = Collection(name=collection_name, schema=schema)
            
            logger.info(f"Collection '{collection_name}' created successfully")
            
            # Create index
            logger.info("Creating index...")
            index_params = {
                "metric_type": "L2",
                "index_type": "HNSW",
                "params": {"M": 8, "efConstruction": 64}
            }
            collection.create_index(field_name="vector", index_params=index_params)
            logger.info("Index created successfully")
            
            # Insert some data
            logger.info("Inserting test data...")
            # Generate random vectors
            vectors = [np.random.rand(dimension).tolist() for _ in range(5)]
            ids = [f"id_{i}" for i in range(5)]
            
            # Insert data
            collection.insert([ids, vectors])
            collection.flush()
            logger.info(f"Inserted {len(ids)} records")
            
            # Load collection for search
            logger.info("Loading collection...")
            collection.load()
            
            # Search
            logger.info("Performing search...")
            search_params = {"metric_type": "L2", "params": {"ef": 64}}
            results = collection.search(
                data=[vectors[0]],
                anns_field="vector",
                param=search_params,
                limit=3,
                output_fields=["id"]
            )
            
            logger.info(f"Search results: {results}")
            
            # Clean up
            logger.info(f"Dropping collection '{collection_name}'...")
            utility.drop_collection(collection_name)
            
        except Exception as e:
            logger.error(f"Error during collection operations: {str(e)}")
            return False
        finally:
            # Disconnect
            logger.info("Disconnecting from Milvus...")
            connections.disconnect(alias)
        
        logger.info("Milvus connection test completed successfully!")
        return True
    
    except ImportError as e:
        logger.error(f"ImportError: {str(e)}")
        logger.error("Make sure pymilvus is installed: pip install pymilvus")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return False

if __name__ == "__main__":
    result = test_milvus_connection()
    print(f"\nTest result: {'SUCCESS' if result else 'FAILED'}")
    sys.exit(0 if result else 1) 