"""Unit tests for enhanced MCP with streaming support."""

import pytest
import asyncio
import json
from typing import AsyncIterator, List, Dict, Any
from datetime import datetime

from memory_core.orchestrator.enhanced_mcp import (
    EnhancedMCPServer,
    MCPStreaming,
    StreamingBatch,
    ProgressCallback,
    StreamingError,
    MCPProtocolError,
    StreamingConfig
)
from memory_core.orchestrator.data_formats import (
    StandardizedKnowledge,
    EntityType,
    StandardizedIdentifier
)


class MockKnowledgeEngine:
    """Mock knowledge engine for testing."""
    
    def __init__(self, total_items: int = 100):
        self.total_items = total_items
        self.items = []
        
        # Generate mock data
        for i in range(total_items):
            self.items.append({
                'id': f'node-{i}',
                'content': f'Test content {i}',
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'relevance_score': 0.5 + (i % 10) * 0.05
                }
            })
    
    async def search_nodes(self, query: str = None, offset: int = 0, 
                          limit: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """Mock search implementation."""
        # Simulate some processing time
        await asyncio.sleep(0.01)
        
        # Return slice of items
        return self.items[offset:offset + limit]
    
    async def get_total_count(self, query: str = None) -> int:
        """Get total count of items."""
        return self.total_items


class TestMCPStreaming:
    """Test MCP streaming functionality."""
    
    @pytest.fixture
    def streaming_config(self):
        """Create test streaming configuration."""
        return StreamingConfig(
            batch_size=10,
            max_concurrent_batches=3,
            timeout_seconds=30,
            retry_attempts=2
        )
    
    @pytest.fixture
    def mcp_streaming(self, streaming_config):
        """Create MCP streaming instance."""
        return MCPStreaming(config=streaming_config)
    
    @pytest.mark.asyncio
    async def test_stream_basic_query(self, mcp_streaming):
        """Test basic streaming query."""
        engine = MockKnowledgeEngine(total_items=25)
        
        batches = []
        async for batch in mcp_streaming.stream_query(
            "test query", 
            engine=engine,
            batch_size=10
        ):
            batches.append(batch)
        
        # Should have 3 batches (10, 10, 5)
        assert len(batches) == 3
        assert len(batches[0].results) == 10
        assert len(batches[1].results) == 10
        assert len(batches[2].results) == 5
        
        # Check batch metadata
        assert batches[0].batch_id == 0
        assert batches[0].has_more is True
        assert batches[2].has_more is False
    
    @pytest.mark.asyncio
    async def test_stream_with_progress_callback(self, mcp_streaming):
        """Test streaming with progress callbacks."""
        engine = MockKnowledgeEngine(total_items=50)
        
        progress_updates = []
        
        def progress_callback(current: int, total: int, batch_id: int):
            progress_updates.append({
                'current': current,
                'total': total,
                'batch_id': batch_id
            })
        
        batches = []
        async for batch in mcp_streaming.stream_query(
            "test query",
            engine=engine,
            batch_size=20,
            progress_callback=progress_callback
        ):
            batches.append(batch)
        
        # Check progress updates
        assert len(progress_updates) == 3  # One per batch
        assert progress_updates[0]['current'] == 20
        assert progress_updates[1]['current'] == 40
        assert progress_updates[2]['current'] == 50
        assert all(p['total'] == 50 for p in progress_updates)
    
    @pytest.mark.asyncio
    async def test_stream_cancellation(self, mcp_streaming):
        """Test cancelling a stream mid-operation."""
        engine = MockKnowledgeEngine(total_items=100)
        
        batches_received = 0
        
        async for batch in mcp_streaming.stream_query(
            "test query",
            engine=engine,
            batch_size=10
        ):
            batches_received += 1
            
            # Cancel after 3 batches
            if batches_received >= 3:
                break
        
        # Should only have received 3 batches
        assert batches_received == 3
    
    @pytest.mark.asyncio
    async def test_stream_error_handling(self, mcp_streaming):
        """Test error handling during streaming."""
        # Create engine that fails after some items
        class FailingEngine(MockKnowledgeEngine):
            def __init__(self):
                super().__init__(50)
                self.call_count = 0
            
            async def search_nodes(self, **kwargs):
                self.call_count += 1
                if self.call_count > 2:
                    raise RuntimeError("Simulated failure")
                return await super().search_nodes(**kwargs)
        
        engine = FailingEngine()
        
        with pytest.raises(StreamingError):
            async for batch in mcp_streaming.stream_query(
                "test query",
                engine=engine,
                batch_size=10
            ):
                pass  # Should fail on third batch
    
    @pytest.mark.asyncio
    async def test_concurrent_streams(self, mcp_streaming):
        """Test multiple concurrent streams."""
        engine = MockKnowledgeEngine(total_items=30)
        
        async def consume_stream(query: str) -> List[StreamingBatch]:
            batches = []
            async for batch in mcp_streaming.stream_query(
                query,
                engine=engine,
                batch_size=10
            ):
                batches.append(batch)
            return batches
        
        # Run multiple streams concurrently
        results = await asyncio.gather(
            consume_stream("query1"),
            consume_stream("query2"),
            consume_stream("query3")
        )
        
        # Each should get all batches
        for batches in results:
            assert len(batches) == 3
            total_items = sum(len(b.results) for b in batches)
            assert total_items == 30


class TestEnhancedMCPServer:
    """Test enhanced MCP server functionality."""
    
    @pytest.fixture
    def mcp_server(self):
        """Create MCP server instance."""
        engine = MockKnowledgeEngine(total_items=50)
        config = {'mcp': {'streaming': {'batch_size': 10}}}
        return EnhancedMCPServer(engine, config)
    
    @pytest.mark.asyncio
    async def test_handle_stream_request(self, mcp_server):
        """Test handling streaming requests."""
        request = {
            'method': 'stream_query',
            'params': {
                'query': 'test search',
                'batch_size': 5,
                'fields': ['id', 'content']
            }
        }
        
        # Collect all responses
        responses = []
        async for response in mcp_server.handle_stream_request(request):
            responses.append(response)
        
        # Check responses
        assert len(responses) == 10  # 50 items / 5 per batch
        
        # First response should have batch info
        first = responses[0]
        assert 'batch_id' in first
        assert 'results' in first
        assert 'has_more' in first
        assert len(first['results']) == 5
    
    @pytest.mark.asyncio
    async def test_handle_standard_request(self, mcp_server):
        """Test handling non-streaming requests."""
        request = {
            'method': 'search',
            'params': {
                'query': 'test search',
                'limit': 20
            }
        }
        
        response = await mcp_server.handle_request(request)
        
        assert 'results' in response
        assert len(response['results']) == 20
        assert 'total' in response
    
    @pytest.mark.asyncio
    async def test_protocol_validation(self, mcp_server):
        """Test MCP protocol validation."""
        # Invalid request - missing method
        invalid_request = {
            'params': {'query': 'test'}
        }
        
        with pytest.raises(MCPProtocolError):
            await mcp_server.handle_request(invalid_request)
        
        # Invalid method
        invalid_method = {
            'method': 'invalid_method',
            'params': {}
        }
        
        with pytest.raises(MCPProtocolError):
            await mcp_server.handle_request(invalid_method)
    
    @pytest.mark.asyncio
    async def test_data_transformation(self, mcp_server):
        """Test data transformation to standardized format."""
        request = {
            'method': 'get_knowledge',
            'params': {
                'id': 'node-1'
            }
        }
        
        response = await mcp_server.handle_request(request)
        
        # Should return standardized knowledge format
        assert 'entity_type' in response
        assert 'identifier' in response
        assert 'content' in response
        assert 'metadata' in response
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, mcp_server):
        """Test batch request processing."""
        batch_request = {
            'method': 'batch_process',
            'params': {
                'operations': [
                    {'method': 'search', 'params': {'query': 'test1', 'limit': 5}},
                    {'method': 'search', 'params': {'query': 'test2', 'limit': 5}},
                    {'method': 'get_knowledge', 'params': {'id': 'node-1'}}
                ]
            }
        }
        
        response = await mcp_server.handle_request(batch_request)
        
        assert 'results' in response
        assert len(response['results']) == 3
        
        # Check individual results
        assert len(response['results'][0]['results']) == 5
        assert len(response['results'][1]['results']) == 5
        assert response['results'][2]['entity_type'] is not None
    
    @pytest.mark.asyncio
    async def test_streaming_with_filters(self, mcp_server):
        """Test streaming with filter parameters."""
        request = {
            'method': 'stream_query',
            'params': {
                'query': 'machine learning',
                'batch_size': 10,
                'filters': {
                    'min_relevance': 0.7,
                    'created_after': '2024-01-01'
                },
                'sort': 'relevance_desc'
            }
        }
        
        batches = []
        async for response in mcp_server.handle_stream_request(request):
            batches.append(response)
        
        # Should still receive results (mock doesn't filter)
        assert len(batches) > 0
        
        # Check filter params were passed
        assert 'metadata' in batches[0]
        assert batches[0]['metadata'].get('filters') is not None
    
    @pytest.mark.asyncio
    async def test_connection_pooling(self, mcp_server):
        """Test connection pooling for concurrent requests."""
        # Simulate multiple concurrent connections
        async def make_request(i: int):
            request = {
                'method': 'search',
                'params': {
                    'query': f'concurrent test {i}',
                    'limit': 10
                }
            }
            return await mcp_server.handle_request(request)
        
        # Make 10 concurrent requests
        results = await asyncio.gather(
            *[make_request(i) for i in range(10)]
        )
        
        # All should succeed
        assert len(results) == 10
        assert all('results' in r for r in results)
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, mcp_server):
        """Test graceful shutdown during streaming."""
        request = {
            'method': 'stream_query',
            'params': {
                'query': 'test',
                'batch_size': 5
            }
        }
        
        # Start streaming
        stream_iter = mcp_server.handle_stream_request(request).__aiter__()
        
        # Get first batch
        first_batch = await stream_iter.__anext__()
        assert first_batch is not None
        
        # Simulate shutdown
        await mcp_server.shutdown()
        
        # Further iterations should complete gracefully
        with pytest.raises(StopAsyncIteration):
            await stream_iter.__anext__()