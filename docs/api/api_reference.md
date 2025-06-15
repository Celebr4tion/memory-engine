# Memory Engine API Reference

This document provides comprehensive API documentation for the Memory Engine, including the MCP (Module Communication Protocol) interface, Python APIs, and integration patterns.

## Table of Contents

1. [MCP API Interface](#mcp-api-interface)
2. [Python API Reference](#python-api-reference)
3. [Error Handling](#error-handling)
4. [Response Formats](#response-formats)
5. [Integration Examples](#integration-examples)
6. [Rate Limits and Performance](#rate-limits-and-performance)

## MCP API Interface

The Memory Engine provides a Module Communication Protocol (MCP) interface for external systems to interact with the knowledge graph. All MCP commands follow a consistent request/response pattern.

### Base Command Structure

```json
{
    "action": "command_name",
    "parameter1": "value1",
    "parameter2": "value2"
}
```

### Response Structure

```json
{
    "status": "success|error|no_results",
    "message": "Optional descriptive message",
    "data": "Command-specific response data"
}
```

## Basic MCP Commands

### 1. Text Ingestion

Ingest raw text and extract knowledge nodes with automatic relationship detection.

#### `ingest_text`

**Request:**
```json
{
    "action": "ingest_text",
    "text": "Raw text to process and extract knowledge from",
    "source": "Optional source identifier (default: 'MCP Input')"
}
```

**Response:**
```json
{
    "status": "success",
    "created_or_merged_node_ids": ["node_id1", "node_id2", "..."],
    "relationship_counts": {
        "tag_relationships": 2,
        "domain_relationships": 1,
        "semantic_relationships": 0
    }
}
```

**Example:**
```python
command = {
    "action": "ingest_text",
    "text": "Machine learning is a subset of artificial intelligence that enables computers to learn patterns from data.",
    "source": "AI Textbook"
}

response = mcp.execute_mcp_command(command)
# Returns: {"status": "success", "created_or_merged_node_ids": ["node_123"], ...}
```

### 2. Knowledge Search

Search for knowledge nodes using semantic similarity.

#### `search`

**Request:**
```json
{
    "action": "search",
    "query": "Search query text",
    "top_k": 5
}
```

**Response:**
```json
{
    "status": "success",
    "query": "Original search query",
    "results": [
        {
            "node_id": "node_123",
            "content_preview": "Preview of node content (truncated to 100 chars)...",
            "source": "Source identifier",
            "rating_truthfulness": 0.85
        }
    ]
}
```

**Example:**
```python
command = {
    "action": "search",
    "query": "artificial intelligence machine learning",
    "top_k": 3
}

response = mcp.execute_mcp_command(command)
# Returns: {"status": "success", "results": [...]}
```

### 3. Node Details

Retrieve comprehensive information about a specific knowledge node.

#### `get_node`

**Request:**
```json
{
    "action": "get_node",
    "node_id": "target_node_id"
}
```

**Response:**
```json
{
    "status": "success",
    "node": {
        "node_id": "node_123",
        "content": "Full node content",
        "source": "Source identifier",
        "creation_timestamp": 1640995200.0,
        "rating_richness": 0.8,
        "rating_truthfulness": 0.9,
        "rating_stability": 0.7
    },
    "outgoing_relationships": [
        {
            "edge_id": "edge_456",
            "target_id": "node_789",
            "relation_type": "RELATED_TO",
            "confidence_score": 0.8
        }
    ],
    "incoming_relationships": [
        {
            "edge_id": "edge_789",
            "source_id": "node_456",
            "relation_type": "CONTAINS",
            "confidence_score": 0.9
        }
    ]
}
```

### 4. Rating Updates

Update knowledge node ratings based on evidence.

#### `update_rating`

**Request:**
```json
{
    "action": "update_rating",
    "node_id": "target_node_id",
    "rating": {
        "confirmation": 0.3,
        "contradiction": 0.0,
        "richness": 0.2,
        "stability": 0.1
    }
}
```

**Response:**
```json
{
    "status": "success",
    "node": {
        "node_id": "node_123",
        "content": "Updated node content",
        "rating_richness": 0.85,
        "rating_truthfulness": 0.92,
        "rating_stability": 0.75
    },
    "outgoing_relationships": [...],
    "incoming_relationships": [...]
}
```

**Rating Parameters:**
- `confirmation` (0.0-1.0): Positive evidence supporting truthfulness
- `contradiction` (0.0-1.0): Negative evidence challenging truthfulness  
- `richness` (-1.0 to 1.0): Evidence affecting information richness
- `stability` (-1.0 to 1.0): Evidence affecting information stability

### 5. Node Listing

List knowledge nodes with optional filtering.

#### `list_nodes`

**Request:**
```json
{
    "action": "list_nodes",
    "filters": {
        "rating_truthfulness": 0.8,
        "source": "specific_source"
    }
}
```

**Response:**
```json
{
    "status": "not_implemented",
    "message": "Filtering nodes is not yet implemented",
    "nodes": []
}
```

*Note: This command is planned for future implementation.*

## Enhanced MCP Commands

The Memory Engine provides advanced MCP commands for sophisticated knowledge operations including graph traversal, knowledge synthesis, bulk operations, and analytics.

### Advanced Graph Queries

#### `multi_hop_traversal`

Perform multi-hop traversal from a starting node to explore relationship networks.

**Request:**
```json
{
    "action": "multi_hop_traversal",
    "start_node_id": "starting_node_id",
    "max_hops": 3,
    "relation_filter": ["specific_relation_type"],
    "min_confidence": 0.7
}
```

**Response:**
```json
{
    "status": "success",
    "start_node": "starting_node_id",
    "max_hops": 3,
    "total_nodes_found": 15,
    "nodes_by_distance": {
        "0": ["starting_node_id"],
        "1": ["node1", "node2", "node3"],
        "2": ["node4", "node5"]
    },
    "paths": [
        ["starting_node_id", "--relates_to-->", "node1"],
        ["starting_node_id", "--contains-->", "node2", "--relates_to-->", "node4"]
    ],
    "node_details": {
        "node1": {
            "content": "Content preview...",
            "source": "source_name",
            "rating_truthfulness": 0.85
        }
    }
}
```

#### `extract_subgraph`

Extract a subgraph focused on specific topics using semantic similarity.

**Request:**
```json
{
    "action": "extract_subgraph",
    "topic_keywords": ["machine learning", "neural networks"],
    "max_nodes": 50,
    "min_relevance": 0.7
}
```

**Response:**
```json
{
    "status": "success",
    "topic_keywords": ["machine learning", "neural networks"],
    "total_nodes": 23,
    "total_relationships": 45,
    "nodes": [
        {
            "node_id": "node_123",
            "relevance": 0.85,
            "content": "Full node content",
            "source": "AI Research Paper"
        }
    ],
    "relationships": [
        {
            "from_id": "node_123",
            "to_id": "node_456",
            "relation_type": "relates_to",
            "confidence_score": 0.8
        }
    ],
    "subgraph_density": 1.96
}
```

#### `pattern_matching`

Find nodes and relationships matching specific structural patterns.

**Request:**
```json
{
    "action": "pattern_matching",
    "pattern": {
        "nodes": {
            "content_contains": "artificial intelligence",
            "min_truthfulness": 0.8,
            "source_contains": "research"
        },
        "relationships": {
            "outgoing_relation_type": "defines",
            "incoming_relation_type": "part_of"
        },
        "max_results": 20
    }
}
```

**Response:**
```json
{
    "status": "success",
    "pattern": {...},
    "total_matches": 12,
    "matches": [
        {
            "root_node": {
                "node_id": "node_789",
                "content": "Artificial intelligence definition...",
                "source": "Research Paper"
            },
            "pattern_score": 0.92
        }
    ]
}
```

#### `temporal_query`

Query knowledge based on temporal criteria using version history.

**Request:**
```json
{
    "action": "temporal_query",
    "start_time": 1699123200,
    "end_time": 1699209600,
    "operation_type": "nodes_created"
}
```

**Response:**
```json
{
    "status": "success",
    "operation_type": "nodes_created",
    "start_time": 1699123200,
    "end_time": 1699209600,
    "time_range_days": 1.0,
    "total_results": 15,
    "results": [
        {
            "node_id": "node_abc",
            "timestamp": 1699150000,
            "change_type": "create",
            "formatted_time": "2023-11-05T12:00:00"
        }
    ]
}
```

### Knowledge Synthesis

#### `synthesize_knowledge`

Combine multiple knowledge nodes into coherent responses.

**Request:**
```json
{
    "action": "synthesize_knowledge",
    "node_ids": ["node1", "node2", "node3"],
    "synthesis_type": "summary"
}
```

**Response:**
```json
{
    "status": "success",
    "synthesis_type": "summary",
    "nodes_processed": 3,
    "sources": ["source1", "source2"],
    "average_confidence": 0.84,
    "summary_points": [
        "Key insight from combined knowledge...",
        "Another important synthesis point..."
    ],
    "total_content_length": 1250
}
```

**Synthesis Types:**
- `summary`: Generate key points and insights
- `comparison`: Compare and contrast nodes
- `timeline`: Chronological organization of events

#### `answer_question`

Answer questions using graph traversal and knowledge synthesis.

**Request:**
```json
{
    "action": "answer_question",
    "question": "What are the key components of neural networks?",
    "max_hops": 2,
    "top_k_nodes": 10
}
```

**Response:**
```json
{
    "status": "success",
    "question": "What are the key components of neural networks?",
    "answer": "Neural networks consist of layers of interconnected nodes...",
    "evidence_count": 8,
    "evidence": [
        {
            "node_id": "node_123",
            "content": "Supporting evidence content",
            "source": "AI Textbook",
            "confidence": 0.92
        }
    ],
    "confidence_score": 0.87
}
```

#### `find_contradictions`

Identify potential contradictions in the knowledge base.

**Request:**
```json
{
    "action": "find_contradictions",
    "topic_keywords": ["climate change"],
    "confidence_threshold": 0.8
}
```

**Response:**
```json
{
    "status": "success",
    "topic_keywords": ["climate change"],
    "nodes_analyzed": 150,
    "contradictions_found": 3,
    "contradictions": [
        {
            "node1": {
                "id": "node_abc",
                "content": "Statement A...",
                "source": "Source 1",
                "confidence": 0.85
            },
            "node2": {
                "id": "node_def",
                "content": "Contradictory statement...",
                "source": "Source 2", 
                "confidence": 0.88
            },
            "contradiction_score": 0.94,
            "detected_patterns": ["Negation vs Affirmation"]
        }
    ]
}
```

### Bulk Operations

#### `start_bulk_ingestion`

Initialize a bulk ingestion operation with progress tracking.

**Request:**
```json
{
    "action": "start_bulk_ingestion",
    "operation_id": "optional_custom_id"
}
```

**Response:**
```json
{
    "status": "success",
    "operation_id": "bulk_op_12345",
    "message": "Bulk ingestion operation initialized"
}
```

#### `add_to_bulk_ingestion`

Add texts to an ongoing bulk ingestion operation.

**Request:**
```json
{
    "action": "add_to_bulk_ingestion",
    "operation_id": "bulk_op_12345",
    "texts": [
        {
            "text": "Text content to process",
            "source": "Document 1"
        },
        {
            "text": "Another text to process",
            "source": "Document 2"
        }
    ]
}
```

**Response:**
```json
{
    "status": "success",
    "operation_id": "bulk_op_12345",
    "progress": 0.67,
    "processed": 8,
    "failed": 1,
    "total": 12,
    "nodes_created": 23
}
```

#### `get_bulk_operation_status`

Check the status of a bulk operation.

**Request:**
```json
{
    "action": "get_bulk_operation_status",
    "operation_id": "bulk_op_12345"
}
```

**Response:**
```json
{
    "status": "success",
    "operation_id": "bulk_op_12345",
    "operation_status": "completed",
    "progress": 1.0,
    "processed_items": 12,
    "failed_items": 1,
    "total_items": 13,
    "nodes_created": 28,
    "elapsed_time_seconds": 45.2,
    "total_duration": 45.2,
    "errors": ["Error processing item 5: Invalid JSON"]
}
```

#### `export_subgraph`

Export a subgraph in various formats.

**Request:**
```json
{
    "action": "export_subgraph",
    "node_ids": ["node1", "node2", "node3"],
    "format": "json",
    "include_relationships": true
}
```

**Response:**
```json
{
    "status": "success",
    "format": "json",
    "data": {
        "nodes": [...],
        "relationships": [...],
        "export_metadata": {
            "timestamp": 1699123456,
            "node_count": 3,
            "relationship_count": 5
        }
    }
}
```

**Supported Formats:**
- `json`: Complete JSON export with metadata
- `cypher`: Cypher CREATE statements for Neo4j
- `csv`: Separate CSV files for nodes and relationships

#### `bulk_create_relationships`

Create multiple relationships in a single operation.

**Request:**
```json
{
    "action": "bulk_create_relationships",
    "relationships": [
        {
            "from_id": "node1",
            "to_id": "node2",
            "relation_type": "relates_to",
            "confidence_score": 0.8,
            "timestamp": 1699123456
        }
    ]
}
```

**Response:**
```json
{
    "status": "success",
    "created_count": 15,
    "failed_count": 2,
    "created_relationships": [
        {
            "edge_id": "edge_123",
            "from_id": "node1",
            "to_id": "node2",
            "relation_type": "relates_to"
        }
    ],
    "failed_relationships": [
        {
            "spec": {...},
            "error": "Source node not found"
        }
    ]
}
```

### Analytics Endpoints

#### `analyze_knowledge_coverage`

Analyze knowledge coverage across domains and sources.

**Request:**
```json
{
    "action": "analyze_knowledge_coverage",
    "domains": ["technology", "science"]
}
```

**Response:**
```json
{
    "status": "success",
    "analysis": {
        "total_nodes": 1500,
        "domains": {
            "technology": 450,
            "science": 680,
            "other": 370
        },
        "sources": {
            "Research Papers": 800,
            "Wikipedia": 400,
            "Textbooks": 300
        },
        "quality_distribution": {
            "high": 750,
            "medium": 500,
            "low": 250
        },
        "content_length_stats": {
            "min": 50,
            "max": 2000,
            "avg": 425.3
        },
        "temporal_distribution": {
            "2023-10": 200,
            "2023-11": 350
        }
    },
    "analyzed_nodes": 1500
}
```

#### `calculate_relationship_metrics`

Calculate relationship density and network metrics.

**Request:**
```json
{
    "action": "calculate_relationship_metrics"
}
```

**Response:**
```json
{
    "status": "success",
    "metrics": {
        "total_relationships": 3200,
        "relationship_types": {
            "relates_to": 1200,
            "contains": 800,
            "defines": 600,
            "part_of": 400,
            "similar_to": 200
        },
        "density": 0.043,
        "avg_confidence": 0.78,
        "confidence_distribution": {
            "high": 1600,
            "medium": 1200,
            "low": 400
        }
    },
    "analyzed_nodes": 500
}
```

#### `analyze_quality_scores`

Analyze quality score distributions across the knowledge base.

**Request:**
```json
{
    "action": "analyze_quality_scores"
}
```

**Response:**
```json
{
    "status": "success",
    "quality_analysis": {
        "truthfulness": {
            "avg": 0.82,
            "distribution": {
                "0.8-1.0": 650,
                "0.6-0.8": 300,
                "0.4-0.6": 40,
                "0.2-0.4": 8,
                "0.0-0.2": 2
            }
        },
        "richness": {
            "avg": 0.75,
            "distribution": {...}
        },
        "stability": {
            "avg": 0.88,
            "distribution": {...}
        }
    },
    "analyzed_nodes": 1000
}
```

#### `analyze_knowledge_evolution`

Analyze how knowledge has evolved over time.

**Request:**
```json
{
    "action": "analyze_knowledge_evolution",
    "time_periods": 12
}
```

**Response:**
```json
{
    "status": "success",
    "time_periods": 12,
    "period_duration_days": 30,
    "evolution_data": [
        {
            "period": 1,
            "start_date": "2023-01-01T00:00:00",
            "end_date": "2023-01-31T00:00:00",
            "nodes_created": 45,
            "nodes_updated": 12,
            "relationships_created": 78,
            "avg_quality": 0.83
        }
    ],
    "trends": {
        "nodes_created_trend": "increasing",
        "nodes_updated_trend": "stable",
        "relationships_created_trend": "increasing"
    }
}
```

## Python API Reference

### Core Components

#### KnowledgeEngine

Main interface for interacting with the knowledge graph.

```python
from memory_core.core.knowledge_engine import KnowledgeEngine

# Initialize
engine = KnowledgeEngine(
    host="localhost",           # JanusGraph host
    port=8182,                 # JanusGraph port
    changes_threshold=100,      # Changes before snapshot
    enable_versioning=True,     # Enable change tracking
    enable_snapshots=True       # Enable periodic snapshots
)

# Connect to database
success = engine.connect()

# Node operations
node_id = engine.save_node(knowledge_node)
node = engine.get_node(node_id)
success = engine.delete_node(node_id)

# Relationship operations
edge_id = engine.save_relationship(relationship)
relationship = engine.get_relationship(edge_id)
success = engine.delete_relationship(edge_id)

# Relationship queries
outgoing = engine.get_outgoing_relationships(node_id)
incoming = engine.get_incoming_relationships(node_id)

# Versioning operations (if enabled)
success = engine.revert_node(node_id)
success = engine.revert_relationship(edge_id)
snapshot_id = engine.create_snapshot()
history = engine.get_revision_history("node", node_id)

# Cleanup
engine.disconnect()
```

#### KnowledgeNode

Data model for knowledge nodes.

```python
from memory_core.model.knowledge_node import KnowledgeNode
import time

# Create node
node = KnowledgeNode(
    content="Knowledge content text",
    source="Source identifier", 
    creation_timestamp=time.time(),  # Optional, defaults to now
    rating_richness=0.8,            # 0.0-1.0, default 0.5
    rating_truthfulness=0.9,        # 0.0-1.0, default 0.5
    rating_stability=0.7,           # 0.0-1.0, default 0.5
    node_id=None                    # Set after saving
)

# Serialization
node_dict = node.to_dict()
node_copy = KnowledgeNode.from_dict(node_dict)

# Comparison
is_equal = node1 == node2
```

#### Relationship

Data model for relationships between nodes.

```python
from memory_core.model.relationship import Relationship
import time

# Create relationship
relationship = Relationship(
    from_id="source_node_id",
    to_id="target_node_id",
    relation_type="RELATED_TO",     # Relationship type
    timestamp=time.time(),          # Optional, defaults to now
    confidence_score=0.8,           # 0.0-1.0, default 0.5
    version=1,                      # Version number, default 1
    edge_id=None                    # Set after saving
)

# Serialization
rel_dict = relationship.to_dict()
rel_copy = Relationship.from_dict(rel_dict)
```

### Knowledge Processing

#### Advanced Extractor

Extract knowledge units from raw text using LLMs.

```python
from memory_core.ingestion.advanced_extractor import extract_knowledge_units, process_extracted_units

# Extract knowledge units
text = "Raw text to process..."
units = extract_knowledge_units(text)

# Process and store units
node_ids = process_extracted_units(
    units=units,
    source_label="Source identifier",
    storage=engine.storage,             # Optional
    embedding_manager=embedding_manager  # Optional
)
```

**Knowledge Unit Structure:**
```python
{
    "content": "Extracted knowledge statement",
    "tags": ["tag1", "tag2"],
    "metadata": {
        "confidence_level": 0.85,
        "domain": "computer_science",
        "language": "en",
        "importance": 0.75
    },
    "source": {
        "type": "webpage",
        "url": "https://example.com",
        "reference": "Citation information",
        "page": "123"
    }
}
```

#### Relationship Extractor

Automatically detect and create relationships between nodes.

```python
from memory_core.ingestion.relationship_extractor import analyze_and_create_relationships

# Analyze and create relationships
relationships = analyze_and_create_relationships(
    node_ids=["node1", "node2", "node3"],
    storage=engine.storage,
    embedding_manager=embedding_manager  # Optional
)

# Returns:
{
    "tag_relationships": ["edge1", "edge2"],
    "domain_relationships": ["edge3"], 
    "semantic_relationships": ["edge4"]
}
```

### Embedding System

#### EmbeddingManager

Generate and manage embeddings using Google Gemini API.

```python
from memory_core.embeddings.embedding_manager import EmbeddingManager
from memory_core.embeddings.vector_store import VectorStoreMilvus

# Setup
vector_store = VectorStoreMilvus()
vector_store.connect()
embedding_manager = EmbeddingManager(vector_store)

# Generate embedding
embedding = embedding_manager.generate_embedding(
    text="Text to embed",
    task_type="SEMANTIC_SIMILARITY"  # or "RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"
)

# Store node embedding
embedding_manager.store_node_embedding(node_id, text)

# Search similar nodes
similar_node_ids = embedding_manager.search_similar_nodes(
    query_text="Search query",
    top_k=5
)
```

#### VectorStoreMilvus

Milvus vector database integration.

```python
from memory_core.embeddings.vector_store import VectorStoreMilvus

# Initialize
vector_store = VectorStoreMilvus(
    host="localhost",
    port=19530,
    collection_name="memory_engine_embeddings",
    dimension=768  # For gemini-embedding-exp-03-07
)

# Connect
success = vector_store.connect()

# Store embedding
vector_store.add_embedding(node_id, embedding_vector)

# Search
matches = vector_store.search_embedding(
    query_vector=query_embedding,
    top_k=5
)

# Returns: [{"node_id": "node1", "score": 0.95}, ...]
```

### Rating System

#### Rating Updates

Update node ratings based on evidence.

```python
from memory_core.rating.rating_system import update_rating, RatingUpdater

# Function-based API
result = update_rating(
    node_id="target_node",
    evidence={
        "confirmation": 0.3,    # Positive evidence
        "contradiction": 0.1,   # Negative evidence
        "richness": 0.2,        # Richness change
        "stability": 0.1        # Stability change
    },
    storage=engine.storage
)

# Class-based API
updater = RatingUpdater(storage=engine.storage)

# Specific evidence types
updater.record_confirmation("node_id", strength=0.5)
updater.record_contradiction("node_id", strength=0.3)

# Batch updates
updater.update_all_ratings(
    "node_id",
    truthfulness_change=0.2,
    richness_change=0.1,
    stability_change=0.0
)
```

## Error Handling

### MCP Error Responses

All MCP commands return standardized error responses:

```json
{
    "status": "error",
    "message": "Descriptive error message"
}
```

### Common Error Types

#### Invalid Command Format
```json
{
    "status": "error",
    "message": "Invalid command format: 'action' field is required"
}
```

#### Missing Required Fields
```json
{
    "status": "error", 
    "message": "Missing required field: 'text'"
}
```

#### Unknown Action
```json
{
    "status": "error",
    "message": "Unknown action: invalid_action"
}
```

#### Node Not Found
```json
{
    "status": "error",
    "message": "Node with ID 'invalid_id' not found"
}
```

### Python Exception Handling

```python
try:
    # Knowledge Engine operations
    engine.connect()
    node_id = engine.save_node(node)
    
except ConnectionError as e:
    print(f"Database connection failed: {e}")
    
except ValueError as e:
    print(f"Invalid data provided: {e}")
    
except RuntimeError as e:
    print(f"Operation failed: {e}")
    
finally:
    engine.disconnect()
```

## Response Formats

### Success Responses

#### Text Ingestion Success
```json
{
    "status": "success",
    "created_or_merged_node_ids": ["uuid1", "uuid2"],
    "relationship_counts": {
        "tag_relationships": 2,
        "domain_relationships": 1,
        "semantic_relationships": 0
    }
}
```

#### Search Success
```json
{
    "status": "success",
    "query": "search terms",
    "results": [
        {
            "node_id": "uuid",
            "content_preview": "Content preview...",
            "source": "Source name",
            "rating_truthfulness": 0.85
        }
    ]
}
```

#### Node Details Success
```json
{
    "status": "success",
    "node": {
        "node_id": "uuid",
        "content": "Full content",
        "source": "Source",
        "creation_timestamp": 1640995200.0,
        "rating_richness": 0.8,
        "rating_truthfulness": 0.9,
        "rating_stability": 0.7
    },
    "outgoing_relationships": [...],
    "incoming_relationships": [...]
}
```

### No Results Responses

```json
{
    "status": "no_results",
    "message": "No relevant knowledge found",
    "results": []
}
```

### Partial Success

```json
{
    "status": "success",
    "message": "2 of 3 operations completed successfully",
    "created_or_merged_node_ids": ["uuid1", "uuid2"],
    "errors": ["Failed to process unit 3: API timeout"]
}
```

## Integration Examples

### REST API Wrapper

```python
from flask import Flask, request, jsonify
from memory_core.mcp_integration.mcp_endpoint import MemoryEngineMCP

app = Flask(__name__)
mcp = MemoryEngineMCP()

@app.route('/api/ingest', methods=['POST'])
def ingest_text():
    data = request.json
    command = {
        "action": "ingest_text",
        "text": data.get('text'),
        "source": data.get('source', 'API')
    }
    response = mcp.execute_mcp_command(command)
    return jsonify(response)

@app.route('/api/search', methods=['GET'])
def search_knowledge():
    query = request.args.get('q')
    top_k = int(request.args.get('top_k', 5))
    
    command = {
        "action": "search",
        "query": query,
        "top_k": top_k
    }
    response = mcp.execute_mcp_command(command)
    return jsonify(response)
```

### Async Integration

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncMemoryEngine:
    def __init__(self):
        self.mcp = MemoryEngineMCP()
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def ingest_text_async(self, text, source):
        """Asynchronously ingest text."""
        loop = asyncio.get_event_loop()
        command = {
            "action": "ingest_text",
            "text": text,
            "source": source
        }
        
        return await loop.run_in_executor(
            self.executor,
            self.mcp.execute_mcp_command,
            command
        )
    
    async def search_async(self, query, top_k=5):
        """Asynchronously search for knowledge."""
        loop = asyncio.get_event_loop()
        command = {
            "action": "search",
            "query": query,
            "top_k": top_k
        }
        
        return await loop.run_in_executor(
            self.executor,
            self.mcp.execute_mcp_command,
            command
        )

# Usage
async def main():
    engine = AsyncMemoryEngine()
    
    # Process multiple texts concurrently
    tasks = [
        engine.ingest_text_async("Text 1", "Source 1"),
        engine.ingest_text_async("Text 2", "Source 2"),
        engine.ingest_text_async("Text 3", "Source 3")
    ]
    
    results = await asyncio.gather(*tasks)
    print(f"Processed {len(results)} texts")
```

### Batch Processing

```python
def batch_process_documents(mcp, documents, batch_size=10):
    """Process documents in batches."""
    results = []
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batch_results = []
        
        for doc in batch:
            command = {
                "action": "ingest_text",
                "text": doc['content'],
                "source": doc['source']
            }
            
            try:
                response = mcp.execute_mcp_command(command)
                batch_results.append(response)
            except Exception as e:
                batch_results.append({
                    "status": "error",
                    "message": str(e)
                })
        
        results.extend(batch_results)
        
        # Optional: Add delay between batches
        time.sleep(0.1)
    
    return results
```

## Rate Limits and Performance

### Gemini API Limits

The Memory Engine uses Google Gemini API which has rate limits:

- **Requests per minute**: 1,500 (may vary by API key tier)
- **Requests per day**: 100,000 (may vary by API key tier)
- **Concurrent requests**: 10

### Performance Optimization

#### Batch Operations
```python
# Good: Process multiple items in batch
texts = ["text1", "text2", "text3"]
for text in texts:
    response = mcp.execute_mcp_command({
        "action": "ingest_text",
        "text": text
    })

# Better: Use batch processing with delays
def batch_ingest(texts, delay=0.1):
    for text in texts:
        response = mcp.execute_mcp_command({
            "action": "ingest_text", 
            "text": text
        })
        time.sleep(delay)  # Respect rate limits
```

#### Connection Pooling
```python
# Reuse MCP instances
mcp = MemoryEngineMCP()  # Create once
# Use mcp for multiple operations

# Don't create new instances for each operation
```

#### Embedding Optimization
```python
# Store embeddings to avoid regeneration
embedding_manager.store_node_embedding(node_id, text)

# Use appropriate task types
embedding = embedding_manager.generate_embedding(
    text="query text",
    task_type="RETRIEVAL_QUERY"  # More specific than SEMANTIC_SIMILARITY
)
```

### Monitoring and Metrics

```python
import time
import logging

class PerformanceMonitor:
    def __init__(self):
        self.stats = {
            "requests": 0,
            "errors": 0,
            "total_time": 0
        }
    
    def track_request(self, func):
        """Decorator to track request performance."""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                self.stats["requests"] += 1
                return result
            except Exception as e:
                self.stats["errors"] += 1
                raise
            finally:
                self.stats["total_time"] += time.time() - start_time
        return wrapper
    
    def get_stats(self):
        avg_time = self.stats["total_time"] / max(1, self.stats["requests"])
        return {
            "requests": self.stats["requests"],
            "errors": self.stats["errors"],
            "average_time": avg_time,
            "error_rate": self.stats["errors"] / max(1, self.stats["requests"])
        }

# Usage
monitor = PerformanceMonitor()

@monitor.track_request
def monitored_ingest(mcp, text, source):
    return mcp.execute_mcp_command({
        "action": "ingest_text",
        "text": text,
        "source": source
    })
```

## Orchestrator Integration API (v0.5.0+)

The Memory Engine v0.5.0 introduces orchestrator integration with streaming MCP, GraphQL-like queries, event system, and module registry. These features provide advanced capabilities for high-performance and scalable knowledge management.

### Streaming MCP Operations

Streaming MCP operations allow for real-time processing of large datasets with progress callbacks and cancellation support.

#### Starting a Streaming Query

```python
from memory_core.orchestrator.enhanced_mcp import EnhancedMCPServer
from memory_core.orchestrator.query_language import QueryBuilder, QueryType

# Initialize enhanced MCP server
mcp_server = EnhancedMCPServer(knowledge_engine)

# Start streaming query
parameters = {
    "query": "knowledge about artificial intelligence",
    "limit": 1000,
    "batch_size": 50
}

# Start the operation
response = await mcp_server.handle_request("stream_query", parameters)
operation_id = response["operation_id"]

# Stream results
async for result in mcp_server.stream_results(operation_id):
    print(f"Progress: {result['progress']['current']}/{result['progress']['total']}")
    if result['data']:
        # Process batch of results
        batch = result['data']['batch']
        for item in batch:
            print(f"Found: {item['content']}")
```

#### Streaming Bulk Import

```python
# Prepare data for import
import_data = [
    {"content": "Python is a programming language", "source": "Tech Book"},
    {"content": "Machine learning uses algorithms", "source": "AI Book"},
    # ... more data
]

# Start streaming import
parameters = {
    "data": import_data,
    "batch_size": 10
}

response = await mcp_server.handle_request("stream_import", parameters)
operation_id = response["operation_id"]

# Monitor import progress
async for result in mcp_server.stream_results(operation_id):
    if result['state'] == 'streaming':
        processed = result['data']['processed']
        errors = result['data']['errors']
        print(f"Imported: {processed}, Errors: {errors}")
    elif result['state'] == 'completed':
        print(f"Import completed! Success rate: {result['data']['success_rate']}")
```

### GraphQL-like Query Language

The new query language provides flexible, powerful querying capabilities with filtering, aggregation, and projection.

#### Basic Query Building

```python
from memory_core.orchestrator.query_language import (
    QueryBuilder, QueryType, FilterOperator
)

# Build a query for nodes containing "AI" with high ratings
query = (QueryBuilder()
    .query_type(QueryType.NODES)
    .where_contains("content", "AI")
    .where_greater_than("rating_truthfulness", 0.8)
    .select("id", "content", "rating_truthfulness", "source")
    .order_by("rating_truthfulness", ascending=False)
    .limit(10)
    .build()
)

# Execute the query
processor = GraphQLQueryProcessor(knowledge_engine)
result = await processor.execute_query(query)

if result["success"]:
    for node in result["data"]:
        print(f"Node: {node['content'][:50]}... (Rating: {node['rating_truthfulness']})")
```

#### Advanced Queries with Aggregation

```python
# Query with aggregation - count nodes by source
aggregation_query = (QueryBuilder()
    .query_type(QueryType.AGGREGATION)
    .where_greater_than("rating_truthfulness", 0.7)
    .group_by("source")
    .count("*", "node_count")
    .avg("rating_truthfulness", "avg_rating")
    .build()
)

result = await processor.execute_query(aggregation_query)
if result["success"]:
    for source, stats in result["data"].items():
        print(f"Source: {source}, Nodes: {stats['node_count']}, Avg Rating: {stats['avg_rating']:.2f}")
```

#### Complex Filtering and Search

```python
# Complex query with multiple filters and semantic search
search_query = (QueryBuilder()
    .query_type(QueryType.SEARCH)
    .where_contains("content", "machine learning")
    .where_in("source", ["AI Book", "ML Papers", "Research"])
    .where("rating_importance", FilterOperator.GTE, 0.8)
    .select("id", "content", "source", "rating_truthfulness", "rating_importance")
    .order_by("rating_importance", ascending=False)
    .limit(20)
    .build()
)

result = await processor.execute_query(search_query)
if result["success"]:
    print(f"Found {result['returned_count']} of {result['total_count']} results")
    for node in result["data"]:
        print(f"- {node['content'][:80]}... ({node['source']})")
```

### Event System

The event system enables inter-module communication with publish/subscribe patterns, persistence, and replay capabilities.

#### Setting up Event System

```python
from memory_core.orchestrator.event_system import EventSystem
from memory_core.orchestrator.event_system import EventType

# Initialize event system
event_system = EventSystem({
    "event_bus": {
        "enable_persistence": True,
        "enable_batching": True,
        "storage_path": "./data/events"
    }
})

await event_system.initialize()

# Create publisher and subscriber
publisher = event_system.create_publisher("knowledge_processor")
subscriber = event_system.create_subscriber("knowledge_monitor")
```

#### Publishing Events

```python
# Publish knowledge change events
await publisher.publish_knowledge_change(
    action="created",
    node_id="node_123",
    node_data={"content": "New knowledge", "source": "API"}
)

# Publish custom events
await publisher.publish_custom_event(
    "knowledge_validation",
    {"node_id": "node_123", "validation_score": 0.95}
)

# Publish system events
await publisher.publish_system_event(
    "warning",
    "storage_backend",
    {"message": "High memory usage", "usage": "85%"}
)
```

#### Subscribing to Events

```python
# Subscribe to knowledge events
async def handle_knowledge_change(event):
    print(f"Knowledge {event.data['action']}: {event.data['node_id']}")
    # Update caches, trigger workflows, etc.

subscriber.subscribe(EventType.KNOWLEDGE_NODE_CREATED, handle_knowledge_change)
subscriber.subscribe(EventType.KNOWLEDGE_NODE_UPDATED, handle_knowledge_change)

# Subscribe to custom events with filtering
def high_score_filter(event):
    return event.data.get("custom_data", {}).get("validation_score", 0) > 0.9

async def handle_high_score_validation(event):
    print(f"High-quality knowledge detected: {event.data}")

subscriber.subscribe(EventType.CUSTOM, handle_high_score_validation, high_score_filter)
```

#### Event Replay and Recovery

```python
# Replay events from specific timestamp
from_timestamp = 1640995200  # Unix timestamp

replayed_count = await event_system.replay_events(from_timestamp)
print(f"Replayed {replayed_count} events")

# Get system metrics
metrics = event_system.get_system_metrics()
print(f"Events published: {metrics['event_bus']['events_published']}")
print(f"Events processed: {metrics['event_bus']['events_processed']}")
print(f"Success rate: {metrics['event_bus']['success_rate']:.2%}")
```

### Module Registry

The module registry provides centralized module management with capability advertisement and dependency resolution.

#### Registering Modules

```python
from memory_core.orchestrator.module_registry import (
    ModuleRegistry, ModuleCapability, CapabilityType, Version
)

# Initialize module registry
registry = ModuleRegistry()
await registry.initialize()

# Register a custom module
capabilities = [
    ModuleCapability(
        capability_type=CapabilityType.DATA_PROCESSING,
        name="text_extraction",
        version=Version("1.2.0"),
        description="Extract text from various document formats"
    ),
    ModuleCapability(
        capability_type=CapabilityType.STORAGE,
        name="custom_storage",
        version=Version("2.0.1"),
        description="Custom high-performance storage backend"
    )
]

module_info = await registry.register_module(
    module_id="document_processor",
    name="Document Processing Module",
    version=Version("1.0.0"),
    capabilities=capabilities,
    dependencies=["storage_backend"],
    health_check_url="http://localhost:8080/health"
)

print(f"Registered module: {module_info.module_id}")
```

#### Discovering and Using Modules

```python
# Find modules with specific capabilities
storage_modules = await registry.find_modules_by_capability(
    CapabilityType.STORAGE,
    min_version=Version("1.0.0")
)

for module in storage_modules:
    print(f"Storage module: {module.name} v{module.version}")

# Check module health
health_status = await registry.check_module_health("document_processor")
print(f"Module health: {health_status}")

# Get dependency order for initialization
modules = await registry.get_all_modules()
init_order = registry.dependency_resolver.resolve_dependencies(modules)
print(f"Initialization order: {[m.module_id for m in init_order]}")
```

### Standardized Data Formats

The orchestrator provides standardized interfaces for cross-module communication and entity resolution.

#### Using Standardized Knowledge Entities

```python
from memory_core.orchestrator.data_formats import (
    create_knowledge_entity, StandardizedIdentifier, EntityType
)

# Create standardized knowledge entity
entity = create_knowledge_entity(
    content="Quantum computing uses quantum bits for processing",
    source="Quantum Physics Textbook",
    entity_type=EntityType.CONCEPT,
    metadata={
        "domain": "quantum_computing",
        "complexity": "advanced",
        "references": ["textbook_ch5", "paper_123"]
    }
)

print(f"Entity ID: {entity.identifier.id}")
print(f"Entity Type: {entity.identifier.entity_type}")
print(f"Source Module: {entity.identifier.source_module}")
```

#### Cross-Module Entity Resolution

```python
from memory_core.orchestrator.data_formats import (
    EntityResolver, StandardizedQuery, OperationResult
)

# Initialize entity resolver
resolver = EntityResolver()

# Add entity from different modules
entity1 = create_knowledge_entity("Machine learning", "AI_Module")
entity2 = create_knowledge_entity("Machine learning algorithms", "ML_Module")

# Resolve duplicates and conflicts
resolution_result = await resolver.resolve_entities([entity1, entity2])

if resolution_result.success:
    canonical_entity = resolution_result.data["canonical_entity"]
    print(f"Resolved to: {canonical_entity.content}")
    print(f"Confidence: {resolution_result.data['confidence']}")
else:
    print(f"Resolution failed: {resolution_result.error}")
```

#### Unified Error Handling

```python
from memory_core.orchestrator.data_formats import UnifiedError, ErrorCode

try:
    # Some operation that might fail
    result = await some_orchestrator_operation()
except Exception as e:
    # Create standardized error response
    error = UnifiedError(
        code=ErrorCode.PROCESSING_ERROR,
        message="Failed to process knowledge entity",
        details={"original_error": str(e), "entity_id": "ent_123"},
        source_module="knowledge_processor",
        timestamp=time.time()
    )
    
    # Log or handle the standardized error
    print(f"Error {error.code}: {error.message}")
    print(f"Source: {error.source_module}")
```

### CLI Integration

The v0.5.0 orchestrator features are also accessible via the enhanced CLI:

```bash
# Start streaming operations
memory-engine mcp stream-query --query="AI research" --batch-size=50 --progress

# Execute GraphQL-like queries
memory-engine query build \
    --type=nodes \
    --filter="content contains 'machine learning'" \
    --filter="rating_truthfulness > 0.8" \
    --select="id,content,source" \
    --limit=10 \
    --execute

# Manage event system
memory-engine events list --status=pending --limit=20
memory-engine events replay --from-timestamp=1640995200
memory-engine events metrics --detailed

# Module registry operations
memory-engine modules list --capabilities --health-status
memory-engine modules register my-module --capabilities=storage,processing
memory-engine modules health-check --module=document_processor

# Export orchestrator data
memory-engine export --format=json --include-events --include-modules
```

### Error Handling for Orchestrator Features

```python
from memory_core.orchestrator.data_formats import ErrorCode, UnifiedError

async def handle_orchestrator_operations():
    try:
        # Streaming operation
        async for result in mcp_server.stream_results(operation_id):
            if result['state'] == 'failed':
                error = UnifiedError.from_dict(result['error'])
                if error.code == ErrorCode.RESOURCE_EXHAUSTED:
                    # Handle resource exhaustion
                    await scale_resources()
                elif error.code == ErrorCode.TIMEOUT:
                    # Handle timeout
                    await retry_operation(operation_id)
                    
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Orchestrator operation failed: {e}")
        
        # Create standardized error response
        error_response = OperationResult(
            success=False,
            error=UnifiedError(
                code=ErrorCode.INTERNAL_ERROR,
                message="Orchestrator operation failed",
                details={"exception": str(e)}
            )
        )
        return error_response
```

This completes the API reference for Memory Engine v0.5.0 orchestrator integration features. The new capabilities provide powerful tools for building scalable, event-driven knowledge management systems with advanced querying and real-time processing capabilities.