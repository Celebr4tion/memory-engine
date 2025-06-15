"""
Enhanced MCP (Model Context Protocol) Interface with Streaming Support

This module extends the standard MCP interface to support:
- Streaming responses for large operations
- Progress callbacks for long-running tasks
- Partial result delivery
- Advanced error handling and recovery
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Union, AsyncGenerator, Callable, Protocol
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class StreamingState(Enum):
    """Streaming operation states."""
    INITIALIZED = "initialized"
    STREAMING = "streaming"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProgressType(Enum):
    """Types of progress indicators."""
    PERCENTAGE = "percentage"
    COUNT = "count"
    SIZE = "size"
    STAGE = "stage"


@dataclass
class ProgressInfo:
    """Progress information for streaming operations."""
    operation_id: str
    progress_type: ProgressType
    current: Union[int, float]
    total: Union[int, float, None]
    message: str
    timestamp: float
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class StreamingResult:
    """Result from streaming operation."""
    operation_id: str
    state: StreamingState
    data: Any
    progress: Optional[ProgressInfo] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        if self.progress:
            result['progress'] = self.progress.to_dict()
        return result


class ProgressCallback(Protocol):
    """Protocol for progress callback functions."""
    
    def __call__(self, progress: ProgressInfo) -> None:
        """Called when progress is updated."""
        ...


class StreamingOperation(ABC):
    """Abstract base class for streaming operations."""
    
    def __init__(self, operation_id: str, parameters: Dict[str, Any]):
        self.operation_id = operation_id
        self.parameters = parameters
        self.state = StreamingState.INITIALIZED
        self.start_time = time.time()
        self.progress_callback: Optional[ProgressCallback] = None
        self._cancelled = False
    
    @abstractmethod
    async def execute(self) -> AsyncGenerator[StreamingResult, None]:
        """Execute the streaming operation."""
        pass
    
    def set_progress_callback(self, callback: ProgressCallback) -> None:
        """Set progress callback."""
        self.progress_callback = callback
    
    def cancel(self) -> None:
        """Cancel the operation."""
        self._cancelled = True
        self.state = StreamingState.CANCELLED
    
    def is_cancelled(self) -> bool:
        """Check if operation is cancelled."""
        return self._cancelled
    
    def _emit_progress(self, progress_type: ProgressType, current: Union[int, float], 
                      total: Union[int, float, None], message: str, 
                      details: Optional[Dict[str, Any]] = None) -> None:
        """Emit progress update."""
        progress = ProgressInfo(
            operation_id=self.operation_id,
            progress_type=progress_type,
            current=current,
            total=total,
            message=message,
            timestamp=time.time(),
            details=details
        )
        
        if self.progress_callback:
            self.progress_callback(progress)


class KnowledgeQueryOperation(StreamingOperation):
    """Streaming operation for knowledge queries."""
    
    def __init__(self, operation_id: str, parameters: Dict[str, Any], 
                 knowledge_engine: Any):
        super().__init__(operation_id, parameters)
        self.knowledge_engine = knowledge_engine
    
    async def execute(self) -> AsyncGenerator[StreamingResult, None]:
        """Execute knowledge query with streaming results."""
        try:
            self.state = StreamingState.STREAMING
            query = self.parameters.get('query', '')
            limit = self.parameters.get('limit', 100)
            batch_size = self.parameters.get('batch_size', 10)
            
            self._emit_progress(ProgressType.STAGE, 0, 3, "Starting knowledge query")
            
            # Execute query
            results = await self.knowledge_engine.query(query, limit=limit)
            total_results = len(results)
            
            self._emit_progress(ProgressType.COUNT, 0, total_results, 
                              f"Found {total_results} results, streaming...")
            
            # Stream results in batches
            for i in range(0, total_results, batch_size):
                if self.is_cancelled():
                    break
                
                batch = results[i:i + batch_size]
                progress = min(i + batch_size, total_results)
                
                self._emit_progress(ProgressType.COUNT, progress, total_results,
                                  f"Streaming batch {i//batch_size + 1}")
                
                yield StreamingResult(
                    operation_id=self.operation_id,
                    state=StreamingState.STREAMING,
                    data={'batch': batch, 'batch_index': i // batch_size},
                    progress=ProgressInfo(
                        operation_id=self.operation_id,
                        progress_type=ProgressType.COUNT,
                        current=progress,
                        total=total_results,
                        message=f"Streamed {progress}/{total_results} results",
                        timestamp=time.time()
                    )
                )
                
                # Small delay to allow for cancellation
                await asyncio.sleep(0.01)
            
            if not self.is_cancelled():
                self.state = StreamingState.COMPLETED
                self._emit_progress(ProgressType.STAGE, 3, 3, "Query completed")
                
                yield StreamingResult(
                    operation_id=self.operation_id,
                    state=StreamingState.COMPLETED,
                    data={'total_results': total_results, 'completed': True}
                )
        
        except Exception as e:
            self.state = StreamingState.FAILED
            logger.error(f"Knowledge query operation failed: {e}")
            
            yield StreamingResult(
                operation_id=self.operation_id,
                state=StreamingState.FAILED,
                data=None,
                error=str(e)
            )


class BulkImportOperation(StreamingOperation):
    """Streaming operation for bulk data import."""
    
    def __init__(self, operation_id: str, parameters: Dict[str, Any], 
                 knowledge_engine: Any):
        super().__init__(operation_id, parameters)
        self.knowledge_engine = knowledge_engine
    
    async def execute(self) -> AsyncGenerator[StreamingResult, None]:
        """Execute bulk import with streaming progress."""
        try:
            self.state = StreamingState.STREAMING
            data = self.parameters.get('data', [])
            batch_size = self.parameters.get('batch_size', 10)
            
            total_items = len(data)
            processed = 0
            errors = []
            
            self._emit_progress(ProgressType.COUNT, 0, total_items, 
                              f"Starting bulk import of {total_items} items")
            
            # Process in batches
            for i in range(0, total_items, batch_size):
                if self.is_cancelled():
                    break
                
                batch = data[i:i + batch_size]
                batch_results = []
                
                for item in batch:
                    try:
                        result = await self.knowledge_engine.save_node(item)
                        batch_results.append({'success': True, 'id': result})
                        processed += 1
                    except Exception as e:
                        batch_results.append({'success': False, 'error': str(e)})
                        errors.append(str(e))
                
                progress = min(i + batch_size, total_items)
                self._emit_progress(ProgressType.COUNT, progress, total_items,
                                  f"Processed {processed}/{total_items} items")
                
                yield StreamingResult(
                    operation_id=self.operation_id,
                    state=StreamingState.STREAMING,
                    data={
                        'batch_results': batch_results,
                        'processed': processed,
                        'errors': len(errors)
                    },
                    progress=ProgressInfo(
                        operation_id=self.operation_id,
                        progress_type=ProgressType.COUNT,
                        current=progress,
                        total=total_items,
                        message=f"Imported {processed}/{total_items} items",
                        timestamp=time.time()
                    )
                )
                
                await asyncio.sleep(0.01)
            
            if not self.is_cancelled():
                self.state = StreamingState.COMPLETED
                self._emit_progress(ProgressType.COUNT, total_items, total_items, 
                                  "Bulk import completed")
                
                yield StreamingResult(
                    operation_id=self.operation_id,
                    state=StreamingState.COMPLETED,
                    data={
                        'total_processed': processed,
                        'total_errors': len(errors),
                        'success_rate': processed / total_items if total_items > 0 else 0,
                        'errors': errors[:10]  # Limit error list
                    }
                )
        
        except Exception as e:
            self.state = StreamingState.FAILED
            logger.error(f"Bulk import operation failed: {e}")
            
            yield StreamingResult(
                operation_id=self.operation_id,
                state=StreamingState.FAILED,
                data=None,
                error=str(e)
            )


class MCPStreaming:
    """MCP streaming manager for handling streaming operations."""
    
    def __init__(self):
        self.active_operations: Dict[str, StreamingOperation] = {}
        self.operation_history: Dict[str, StreamingResult] = {}
        self._operation_counter = 0
    
    def generate_operation_id(self) -> str:
        """Generate unique operation ID."""
        self._operation_counter += 1
        return f"op_{int(time.time())}_{self._operation_counter}"
    
    async def start_knowledge_query(self, parameters: Dict[str, Any], 
                                   knowledge_engine: Any,
                                   progress_callback: Optional[ProgressCallback] = None) -> str:
        """Start streaming knowledge query operation."""
        operation_id = self.generate_operation_id()
        operation = KnowledgeQueryOperation(operation_id, parameters, knowledge_engine)
        
        if progress_callback:
            operation.set_progress_callback(progress_callback)
        
        self.active_operations[operation_id] = operation
        return operation_id
    
    async def start_bulk_import(self, parameters: Dict[str, Any], 
                               knowledge_engine: Any,
                               progress_callback: Optional[ProgressCallback] = None) -> str:
        """Start streaming bulk import operation."""
        operation_id = self.generate_operation_id()
        operation = BulkImportOperation(operation_id, parameters, knowledge_engine)
        
        if progress_callback:
            operation.set_progress_callback(progress_callback)
        
        self.active_operations[operation_id] = operation
        return operation_id
    
    async def stream_operation(self, operation_id: str) -> AsyncGenerator[StreamingResult, None]:
        """Stream results from operation."""
        if operation_id not in self.active_operations:
            yield StreamingResult(
                operation_id=operation_id,
                state=StreamingState.FAILED,
                data=None,
                error="Operation not found"
            )
            return
        
        operation = self.active_operations[operation_id]
        
        try:
            async for result in operation.execute():
                self.operation_history[operation_id] = result
                yield result
                
                if result.state in [StreamingState.COMPLETED, StreamingState.FAILED, 
                                   StreamingState.CANCELLED]:
                    # Remove from active operations
                    self.active_operations.pop(operation_id, None)
                    break
                    
        except Exception as e:
            logger.error(f"Error streaming operation {operation_id}: {e}")
            error_result = StreamingResult(
                operation_id=operation_id,
                state=StreamingState.FAILED,
                data=None,
                error=str(e)
            )
            self.operation_history[operation_id] = error_result
            self.active_operations.pop(operation_id, None)
            yield error_result
    
    def cancel_operation(self, operation_id: str) -> bool:
        """Cancel streaming operation."""
        if operation_id in self.active_operations:
            self.active_operations[operation_id].cancel()
            return True
        return False
    
    def get_operation_status(self, operation_id: str) -> Optional[StreamingResult]:
        """Get current status of operation."""
        if operation_id in self.active_operations:
            op = self.active_operations[operation_id]
            return StreamingResult(
                operation_id=operation_id,
                state=op.state,
                data={'active': True}
            )
        
        return self.operation_history.get(operation_id)
    
    def list_active_operations(self) -> List[str]:
        """List all active operation IDs."""
        return list(self.active_operations.keys())


class EnhancedMCPServer:
    """Enhanced MCP server with streaming capabilities."""
    
    def __init__(self, knowledge_engine: Any):
        self.knowledge_engine = knowledge_engine
        self.streaming = MCPStreaming()
        self.request_handlers = {
            'stream_query': self._handle_stream_query,
            'stream_import': self._handle_stream_import,
            'cancel_operation': self._handle_cancel_operation,
            'operation_status': self._handle_operation_status,
            'list_operations': self._handle_list_operations
        }
    
    async def _handle_stream_query(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle streaming query request."""
        try:
            operation_id = await self.streaming.start_knowledge_query(
                parameters, self.knowledge_engine
            )
            
            return {
                'success': True,
                'operation_id': operation_id,
                'message': 'Streaming query started'
            }
        
        except Exception as e:
            logger.error(f"Failed to start streaming query: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _handle_stream_import(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle streaming import request."""
        try:
            operation_id = await self.streaming.start_bulk_import(
                parameters, self.knowledge_engine
            )
            
            return {
                'success': True,
                'operation_id': operation_id,
                'message': 'Streaming import started'
            }
        
        except Exception as e:
            logger.error(f"Failed to start streaming import: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _handle_cancel_operation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle operation cancellation request."""
        operation_id = parameters.get('operation_id')
        if not operation_id:
            return {'success': False, 'error': 'operation_id required'}
        
        cancelled = self.streaming.cancel_operation(operation_id)
        return {
            'success': cancelled,
            'message': 'Operation cancelled' if cancelled else 'Operation not found'
        }
    
    async def _handle_operation_status(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle operation status request."""
        operation_id = parameters.get('operation_id')
        if not operation_id:
            return {'success': False, 'error': 'operation_id required'}
        
        status = self.streaming.get_operation_status(operation_id)
        if status:
            return {
                'success': True,
                'status': status.to_dict()
            }
        else:
            return {
                'success': False,
                'error': 'Operation not found'
            }
    
    async def _handle_list_operations(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle list operations request."""
        active_ops = self.streaming.list_active_operations()
        return {
            'success': True,
            'active_operations': active_ops,
            'count': len(active_ops)
        }
    
    async def handle_request(self, method: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP request."""
        if method in self.request_handlers:
            return await self.request_handlers[method](parameters)
        else:
            return {
                'success': False,
                'error': f'Unknown method: {method}'
            }
    
    async def stream_results(self, operation_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream operation results."""
        async for result in self.streaming.stream_operation(operation_id):
            yield result.to_dict()