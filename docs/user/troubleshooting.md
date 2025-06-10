# Memory Engine Troubleshooting Guide

This guide provides solutions to common issues encountered when setting up, configuring, and using the Memory Engine system.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Configuration Problems](#configuration-problems)
3. [Database Connection Issues](#database-connection-issues)
4. [API and Authentication Problems](#api-and-authentication-problems)
5. [Performance Issues](#performance-issues)
6. [Runtime Errors](#runtime-errors)
7. [Development and Testing Issues](#development-and-testing-issues)
8. [Logs and Diagnostics](#logs-and-diagnostics)

## Installation Issues

### Python Dependency Conflicts

**Problem**: Conflicting package versions during installation
```bash
ERROR: pip's dependency resolver does not currently consider all the packages that are installed
```

**Solutions**:
1. **Use Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Clear pip cache**:
   ```bash
   pip cache purge
   pip install --no-cache-dir -r requirements.txt
   ```

3. **Install with constraints**:
   ```bash
   pip install --constraint requirements.txt memory-engine
   ```

### Docker Compose Issues

**Problem**: Services fail to start with Docker Compose
```bash
ERROR: Service 'janusgraph' failed to build
```

**Solutions**:
1. **Check Docker daemon**:
   ```bash
   docker --version
   docker-compose --version
   sudo systemctl start docker  # Linux
   ```

2. **Clean Docker resources**:
   ```bash
   docker-compose down -v
   docker system prune -f
   docker-compose up -d --build
   ```

3. **Check port conflicts**:
   ```bash
   netstat -tulpn | grep :8182  # JanusGraph
   netstat -tulpn | grep :19530 # Milvus
   ```

### Missing System Dependencies

**Problem**: System-level dependencies missing
```bash
ImportError: No module named '_ctypes'
```

**Solutions**:
1. **Ubuntu/Debian**:
   ```bash
   sudo apt-get update
   sudo apt-get install python3-dev build-essential libffi-dev
   ```

2. **CentOS/RHEL**:
   ```bash
   sudo yum install python3-devel gcc libffi-devel
   ```

3. **macOS**:
   ```bash
   xcode-select --install
   brew install python@3.9
   ```

## Configuration Problems

### Environment Variables Not Set

**Problem**: Missing required environment variables
```bash
ValueError: Required environment variable GEMINI_API_KEY is not set
```

**Solutions**:
1. **Check current environment**:
   ```bash
   env | grep GEMINI
   echo $GEMINI_API_KEY
   ```

2. **Set in current session**:
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```

3. **Set permanently** (add to `.bashrc` or `.zshrc`):
   ```bash
   echo 'export GEMINI_API_KEY="your-api-key-here"' >> ~/.bashrc
   source ~/.bashrc
   ```

4. **Use .env file**:
   ```bash
   # Create .env file in project root
   echo "GEMINI_API_KEY=your-api-key-here" > .env
   ```

### Invalid Configuration Values

**Problem**: Configuration validation errors
```bash
ValueError: Invalid port number: 'localhost'
```

**Solutions**:
1. **Check data types**:
   ```bash
   # Correct
   export JANUSGRAPH_PORT=8182
   
   # Incorrect
   export JANUSGRAPH_PORT="localhost"
   ```

2. **Validate configuration**:
   ```python
   from memory_core.core.knowledge_engine import KnowledgeEngine
   
   try:
       engine = KnowledgeEngine()
       print("Configuration valid")
   except ValueError as e:
       print(f"Configuration error: {e}")
   ```

### File Permission Issues

**Problem**: Cannot read/write configuration files
```bash
PermissionError: [Errno 13] Permission denied: '.env'
```

**Solutions**:
1. **Fix file permissions**:
   ```bash
   chmod 644 .env
   chmod 755 examples/
   ```

2. **Check ownership**:
   ```bash
   ls -la .env
   sudo chown $USER:$USER .env
   ```

## Database Connection Issues

### JanusGraph Connection Failed

**Problem**: Cannot connect to JanusGraph
```bash
ConnectionError: Could not connect to JanusGraph: Connection refused
```

**Diagnostic Steps**:
1. **Check if JanusGraph is running**:
   ```bash
   docker-compose ps
   curl http://localhost:8182/status
   ```

2. **Check Docker logs**:
   ```bash
   docker-compose logs janusgraph
   ```

3. **Test network connectivity**:
   ```bash
   telnet localhost 8182
   netstat -tulpn | grep 8182
   ```

**Solutions**:
1. **Start JanusGraph**:
   ```bash
   cd docker
   docker-compose up -d janusgraph
   ```

2. **Check configuration**:
   ```bash
   # Verify environment variables
   echo $JANUSGRAPH_HOST
   echo $JANUSGRAPH_PORT
   ```

3. **Wait for startup** (JanusGraph can take 1-2 minutes):
   ```bash
   # Wait for health check
   docker-compose logs -f janusgraph
   # Look for: "Channel started at port 8182"
   ```

4. **Reset JanusGraph data**:
   ```bash
   docker-compose down -v
   docker-compose up -d
   ```

### Milvus Connection Issues

**Problem**: Cannot connect to Milvus vector database
```bash
MilvusException: Milvus Proxy is not ready yet. please wait
```

**Diagnostic Steps**:
1. **Check Milvus status**:
   ```bash
   docker-compose ps milvus
   docker-compose logs milvus
   ```

2. **Test Milvus connection**:
   ```python
   from pymilvus import connections
   connections.connect("default", host="localhost", port="19530")
   ```

**Solutions**:
1. **Wait for Milvus initialization** (can take 2-3 minutes):
   ```bash
   docker-compose logs -f milvus
   # Wait for: "Proxy successfully started"
   ```

2. **Check dependencies**:
   ```bash
   docker-compose ps etcd minio
   ```

3. **Restart Milvus stack**:
   ```bash
   docker-compose restart etcd minio milvus
   ```

### Database Storage Issues

**Problem**: Database runs out of space
```bash
No space left on device
```

**Solutions**:
1. **Check disk usage**:
   ```bash
   df -h
   docker system df
   ```

2. **Clean Docker volumes**:
   ```bash
   docker volume prune
   docker-compose down -v  # WARNING: This deletes all data
   ```

3. **Move data directory**:
   ```bash
   # Edit docker-compose.yml volumes section
   volumes:
     - /new/path/janusgraph_data:/var/lib/janusgraph
   ```

## API and Authentication Problems

### Gemini API Key Issues

**Problem**: Invalid or expired API key
```bash
google.api_core.exceptions.Unauthenticated: 401 API key not valid
```

**Solutions**:
1. **Verify API key**:
   ```bash
   curl -H "Authorization: Bearer $GEMINI_API_KEY" \
        https://generativelanguage.googleapis.com/v1/models
   ```

2. **Check API key format**:
   ```python
   import os
   api_key = os.getenv('GEMINI_API_KEY')
   print(f"API key length: {len(api_key) if api_key else 'Not set'}")
   print(f"Starts with 'AIza': {api_key.startswith('AIza') if api_key else False}")
   ```

3. **Regenerate API key**:
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create new API key
   - Update environment variable

### API Rate Limiting

**Problem**: Too many API requests
```bash
google.api_core.exceptions.ResourceExhausted: 429 Quota exceeded
```

**Solutions**:
1. **Add delays between requests**:
   ```python
   import time
   time.sleep(0.1)  # 100ms delay
   ```

2. **Implement retry logic**:
   ```python
   import time
   from functools import wraps
   
   def retry_on_rate_limit(max_retries=3):
       def decorator(func):
           @wraps(func)
           def wrapper(*args, **kwargs):
               for attempt in range(max_retries):
                   try:
                       return func(*args, **kwargs)
                   except Exception as e:
                       if "429" in str(e) and attempt < max_retries - 1:
                           time.sleep(2 ** attempt)  # Exponential backoff
                           continue
                       raise
               return wrapper
           return decorator
   ```

3. **Monitor API usage**:
   ```python
   # Track API calls
   api_calls = 0
   start_time = time.time()
   
   def track_api_call():
       global api_calls
       api_calls += 1
       if api_calls % 100 == 0:
           elapsed = time.time() - start_time
           print(f"Made {api_calls} API calls in {elapsed:.1f}s")
   ```

### Network and Firewall Issues

**Problem**: Network connectivity problems
```bash
requests.exceptions.ConnectionError: HTTPSConnectionPool
```

**Solutions**:
1. **Check internet connectivity**:
   ```bash
   ping google.com
   curl -I https://generativelanguage.googleapis.com
   ```

2. **Check firewall settings**:
   ```bash
   # Linux
   sudo ufw status
   
   # Check corporate proxy
   echo $http_proxy
   echo $https_proxy
   ```

3. **Configure proxy** (if needed):
   ```python
   import os
   os.environ['HTTP_PROXY'] = 'http://proxy.company.com:8080'
   os.environ['HTTPS_PROXY'] = 'http://proxy.company.com:8080'
   ```

## Performance Issues

### Slow Knowledge Extraction

**Problem**: Knowledge extraction takes too long
```bash
# Extraction takes > 30 seconds per text
```

**Diagnostic Steps**:
1. **Profile the extraction**:
   ```python
   import time
   
   start = time.time()
   units = extract_knowledge_units(text)
   extraction_time = time.time() - start
   print(f"Extraction took {extraction_time:.2f} seconds")
   ```

**Solutions**:
1. **Reduce text size**:
   ```python
   # Chunk large texts
   def chunk_text(text, max_words=500):
       words = text.split()
       chunks = []
       for i in range(0, len(words), max_words):
           chunks.append(' '.join(words[i:i + max_words]))
       return chunks
   ```

2. **Batch processing**:
   ```python
   def batch_extract(texts, batch_size=10):
       results = []
       for i in range(0, len(texts), batch_size):
           batch = texts[i:i + batch_size]
           # Process batch
           for text in batch:
               results.append(extract_knowledge_units(text))
           time.sleep(1)  # Rate limiting
       return results
   ```

### Slow Vector Search

**Problem**: Vector similarity search is slow
```bash
# Search takes > 5 seconds
```

**Solutions**:
1. **Optimize Milvus index**:
   ```python
   index_params = {
       "metric_type": "IP",  # Inner Product (faster than L2)
       "index_type": "IVF_SQ8",  # Quantized index
       "params": {"nlist": 2048}  # More clusters
   }
   ```

2. **Reduce search parameters**:
   ```python
   search_params = {
       "nprobe": 8  # Reduce from default 10
   }
   ```

3. **Monitor Milvus performance**:
   ```bash
   # Check Milvus metrics
   curl http://localhost:9091/metrics
   ```

### Memory Usage Issues

**Problem**: High memory consumption
```bash
MemoryError: Unable to allocate array
```

**Solutions**:
1. **Monitor memory usage**:
   ```python
   import psutil
   import os
   
   process = psutil.Process(os.getpid())
   memory_mb = process.memory_info().rss / 1024 / 1024
   print(f"Memory usage: {memory_mb:.1f} MB")
   ```

2. **Optimize batch sizes**:
   ```python
   # Reduce batch sizes
   EMBEDDING_BATCH_SIZE = 50  # Instead of 100
   STORAGE_BATCH_SIZE = 25    # Instead of 50
   ```

3. **Clear memory periodically**:
   ```python
   import gc
   
   def cleanup_memory():
       gc.collect()
       # Force garbage collection
   ```

## Runtime Errors

### Knowledge Extraction Failures

**Problem**: LLM extraction returns invalid JSON
```bash
json.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

**Solutions**:
1. **Add JSON validation**:
   ```python
   def safe_json_parse(response_text):
       try:
           return json.loads(response_text)
       except json.JSONDecodeError:
           # Try to extract JSON from text
           import re
           json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
           if json_match:
               return json.loads(json_match.group())
           return []
   ```

2. **Fallback processing**:
   ```python
   def extract_with_fallback(text):
       try:
           return extract_knowledge_units(text)
       except Exception as e:
           logger.warning(f"Extraction failed: {e}")
           # Create simple fallback unit
           return [{
               "content": text[:200] + "..." if len(text) > 200 else text,
               "tags": ["fallback"],
               "metadata": {"confidence_level": 0.5}
           }]
   ```

### Node/Relationship Not Found

**Problem**: Referenced nodes don't exist
```bash
ValueError: Node with ID 'abc123' not found
```

**Solutions**:
1. **Add existence checks**:
   ```python
   def safe_get_node(engine, node_id):
       try:
           return engine.get_node(node_id)
       except ValueError:
           logger.warning(f"Node {node_id} not found")
           return None
   ```

2. **Cleanup orphaned references**:
   ```python
   def cleanup_orphaned_relationships(engine):
       # Get all relationships
       # Check if both nodes exist
       # Remove relationships with missing nodes
       pass
   ```

### Version Conflicts

**Problem**: Versioning conflicts during updates
```bash
ConflictError: Node was modified by another process
```

**Solutions**:
1. **Implement optimistic locking**:
   ```python
   def safe_update_node(engine, node_id, updates, max_retries=3):
       for attempt in range(max_retries):
           try:
               node = engine.get_node(node_id)
               # Apply updates
               engine.save_node(node)
               return True
           except ConflictError:
               if attempt == max_retries - 1:
                   raise
               time.sleep(0.1 * (2 ** attempt))
       return False
   ```

## Development and Testing Issues

### Test Failures

**Problem**: Tests fail due to missing services
```bash
pytest.skip: JanusGraph not available
```

**Solutions**:
1. **Skip integration tests**:
   ```bash
   export SKIP_INTEGRATION_TESTS=true
   pytest tests/ -k "not integration"
   ```

2. **Start test infrastructure**:
   ```bash
   docker-compose up -d
   pytest tests/
   ```

3. **Mock external services**:
   ```python
   @pytest.fixture
   def mock_gemini_api():
       with patch('memory_core.embeddings.embedding_manager.genai.Client') as mock:
           yield mock
   ```

### Import Errors

**Problem**: Module import failures
```bash
ModuleNotFoundError: No module named 'memory_core'
```

**Solutions**:
1. **Add project root to path**:
   ```python
   import sys
   import os
   sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
   ```

2. **Install in development mode**:
   ```bash
   pip install -e .
   ```

3. **Check PYTHONPATH**:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:/path/to/memory-engine"
   ```

## Logs and Diagnostics

### Enable Debug Logging

```python
import logging

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Specific logger for Memory Engine
logger = logging.getLogger('memory_core')
logger.setLevel(logging.DEBUG)
```

### System Diagnostics Script

Create `diagnose.py`:
```python
#!/usr/bin/env python3
"""System diagnostics for Memory Engine."""

import os
import sys
import subprocess

def check_python():
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")

def check_environment():
    env_vars = ['GEMINI_API_KEY', 'JANUSGRAPH_HOST', 'MILVUS_HOST']
    for var in env_vars:
        value = os.getenv(var, 'NOT SET')
        if 'key' in var.lower():
            display = f"{value[:8]}..." if value != 'NOT SET' else value
        else:
            display = value
        print(f"{var}: {display}")

def check_docker():
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True)
        print(f"Docker: {result.stdout.strip()}")
        
        result = subprocess.run(['docker-compose', '--version'], 
                              capture_output=True, text=True)
        print(f"Docker Compose: {result.stdout.strip()}")
    except FileNotFoundError:
        print("Docker not found")

def check_services():
    services = [
        ('JanusGraph', 'localhost', 8182),
        ('Milvus', 'localhost', 19530)
    ]
    
    for name, host, port in services:
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()
            
            status = "✅ Running" if result == 0 else "❌ Not available"
            print(f"{name} ({host}:{port}): {status}")
        except Exception as e:
            print(f"{name}: ❌ Error checking: {e}")

def check_packages():
    packages = [
        'gremlinpython', 'pymilvus', 'google-genai', 
        'fastapi', 'pydantic', 'numpy'
    ]
    
    for package in packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"{package}: ✅ Installed")
        except ImportError:
            print(f"{package}: ❌ Not installed")

if __name__ == "__main__":
    print("Memory Engine System Diagnostics")
    print("=" * 40)
    
    print("\n🐍 Python Environment:")
    check_python()
    
    print("\n🔧 Environment Variables:")
    check_environment()
    
    print("\n🐳 Docker:")
    check_docker()
    
    print("\n🔌 Services:")
    check_services()
    
    print("\n📦 Python Packages:")
    check_packages()
```

Run diagnostics:
```bash
python diagnose.py
```

### Log Analysis

Common log patterns to look for:

#### JanusGraph Logs
```bash
# Good patterns
"Channel started at port 8182"
"Opened storage backend"

# Problem patterns
"Failed to connect"
"Connection refused"
"Out of memory"
```

#### Milvus Logs
```bash
# Good patterns
"Proxy successfully started"
"Server listening on"

# Problem patterns
"etcd is not ready"
"failed to connect to etcd"
"no space left"
```

#### Application Logs
```bash
# Good patterns
"Connected to JanusGraph"
"Generated embedding of length"
"Created relationship"

# Problem patterns
"Failed to generate embedding"
"Connection error"
"API key not valid"
```

### Performance Monitoring

Create `monitor.py`:
```python
#!/usr/bin/env python3
"""Performance monitoring for Memory Engine."""

import time
import psutil
import requests

def monitor_system():
    """Monitor system resources."""
    cpu = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    print(f"CPU: {cpu}%")
    print(f"Memory: {memory.percent}% ({memory.used // 1024 // 1024} MB)")
    print(f"Disk: {disk.percent}% ({disk.free // 1024 // 1024 // 1024} GB free)")

def monitor_services():
    """Monitor service endpoints."""
    services = [
        ("JanusGraph", "http://localhost:8182/status"),
        ("Milvus", "http://localhost:9091/metrics")
    ]
    
    for name, url in services:
        try:
            response = requests.get(url, timeout=5)
            status = "✅ Healthy" if response.status_code == 200 else f"⚠️ {response.status_code}"
            print(f"{name}: {status}")
        except Exception as e:
            print(f"{name}: ❌ {e}")

if __name__ == "__main__":
    while True:
        print(f"\n=== {time.strftime('%H:%M:%S')} ===")
        monitor_system()
        monitor_services()
        time.sleep(30)
```

This troubleshooting guide should help you identify and resolve most common issues with the Memory Engine system. For additional support, check the GitHub issues page or create a new issue with detailed error logs and system information.