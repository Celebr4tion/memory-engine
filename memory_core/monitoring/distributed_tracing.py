"""
Distributed tracing system for Memory Engine using OpenTelemetry.

This module provides distributed tracing capabilities for complex operations
across multiple components and services in the Memory Engine system.
"""

import logging
import time
import functools
from typing import Dict, Any, Optional, Callable, Union
from contextlib import contextmanager

from opentelemetry import trace, baggage
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes
from opentelemetry.trace import Status, StatusCode
from opentelemetry.util.http import get_excluded_urls

from .structured_logger import get_logger, CorrelationIdManager


class MemoryEngineTracer:
    """
    Distributed tracing system for Memory Engine.

    Provides instrumentation for tracking operations across components
    with OpenTelemetry-compatible tracing.
    """

    def __init__(
        self,
        service_name: str = "memory-engine",
        jaeger_endpoint: Optional[str] = None,
        enable_console_export: bool = False,
    ):
        """
        Initialize the distributed tracing system.

        Args:
            service_name: Name of the service for tracing
            jaeger_endpoint: Jaeger collector endpoint
            enable_console_export: Whether to export traces to console
        """
        self.service_name = service_name
        self.logger = get_logger(__name__, "tracing")

        # Initialize tracing
        self._setup_tracing(jaeger_endpoint, enable_console_export)

        # Get tracer instance
        self.tracer = trace.get_tracer(__name__)

        self.logger.info("Distributed tracing initialized", service_name=service_name)

    def _setup_tracing(self, jaeger_endpoint: Optional[str], enable_console_export: bool):
        """Set up OpenTelemetry tracing configuration."""

        # Create resource
        resource = Resource.create(
            {
                ResourceAttributes.SERVICE_NAME: self.service_name,
                ResourceAttributes.SERVICE_VERSION: "1.0.0",
                ResourceAttributes.SERVICE_NAMESPACE: "memory-engine",
            }
        )

        # Create tracer provider
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)

        # Add span processors
        if jaeger_endpoint:
            jaeger_exporter = JaegerExporter(
                endpoint=jaeger_endpoint,
            )
            jaeger_processor = BatchSpanProcessor(jaeger_exporter)
            tracer_provider.add_span_processor(jaeger_processor)
            self.logger.info("Jaeger exporter configured", endpoint=jaeger_endpoint)

        if enable_console_export:
            console_exporter = ConsoleSpanExporter()
            console_processor = BatchSpanProcessor(console_exporter)
            tracer_provider.add_span_processor(console_processor)
            self.logger.info("Console exporter enabled")

        # Auto-instrument common libraries
        self._setup_auto_instrumentation()

    def _setup_auto_instrumentation(self):
        """Set up automatic instrumentation for common libraries."""
        try:
            # Instrument requests library
            RequestsInstrumentor().instrument()
            self.logger.debug("Requests instrumentation enabled")
        except Exception as e:
            self.logger.warning("Failed to instrument requests", error=e)

        # FastAPI instrumentation would be done at application level
        # RequestsInstrumentor().instrument() covers HTTP client calls

    def start_span(
        self, operation_name: str, parent_context: Optional[trace.Context] = None, **attributes
    ) -> trace.Span:
        """
        Start a new trace span.

        Args:
            operation_name: Name of the operation being traced
            parent_context: Parent trace context
            **attributes: Additional span attributes

        Returns:
            Started span
        """
        # Add correlation ID as baggage
        correlation_id = CorrelationIdManager.get_correlation_id()
        if correlation_id:
            baggage.set_baggage("correlation_id", correlation_id)
            attributes["correlation_id"] = correlation_id

        request_id = CorrelationIdManager.get_request_id()
        if request_id:
            baggage.set_baggage("request_id", request_id)
            attributes["request_id"] = request_id

        # Start span
        span = self.tracer.start_span(operation_name, context=parent_context, attributes=attributes)

        return span

    @contextmanager
    def trace_operation(self, operation_name: str, **attributes):
        """
        Context manager for tracing an operation.

        Args:
            operation_name: Name of the operation
            **attributes: Additional span attributes
        """
        span = self.start_span(operation_name, **attributes)

        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
        finally:
            span.end()

    def trace_query_operation(self, query_type: str, query_id: str):
        """
        Context manager for tracing query operations.

        Args:
            query_type: Type of query being executed
            query_id: Unique identifier for the query
        """
        return self.trace_operation(
            f"query.{query_type}",
            query_type=query_type,
            query_id=query_id,
            component="query_engine",
        )

    def trace_ingestion_operation(self, operation_type: str, items_count: int):
        """
        Context manager for tracing ingestion operations.

        Args:
            operation_type: Type of ingestion operation
            items_count: Number of items being processed
        """
        return self.trace_operation(
            f"ingestion.{operation_type}",
            operation_type=operation_type,
            items_count=items_count,
            component="ingestion",
        )

    def trace_storage_operation(self, operation: str, storage_type: str):
        """
        Context manager for tracing storage operations.

        Args:
            operation: Storage operation (read, write, delete, etc.)
            storage_type: Type of storage (janusgraph, milvus, etc.)
        """
        return self.trace_operation(
            f"storage.{storage_type}.{operation}",
            operation=operation,
            storage_type=storage_type,
            component="storage",
        )

    def trace_embedding_operation(self, operation: str, text_length: Optional[int] = None):
        """
        Context manager for tracing embedding operations.

        Args:
            operation: Embedding operation (generate, search, etc.)
            text_length: Length of text being processed
        """
        attributes = {"operation": operation, "component": "embeddings"}
        if text_length:
            attributes["text_length"] = text_length

        return self.trace_operation(f"embedding.{operation}", **attributes)

    def trace_api_call(self, api_name: str, endpoint: str):
        """
        Context manager for tracing external API calls.

        Args:
            api_name: Name of the API being called
            endpoint: API endpoint
        """
        return self.trace_operation(
            f"api.{api_name}", api_name=api_name, endpoint=endpoint, component="external_api"
        )


class TracingDecorators:
    """Decorators for automatic tracing of functions and methods."""

    def __init__(self, tracer: MemoryEngineTracer):
        self.tracer = tracer

    def trace_function(self, operation_name: Optional[str] = None, component: Optional[str] = None):
        """
        Decorator to automatically trace function execution.

        Args:
            operation_name: Custom operation name (defaults to function name)
            component: Component name for the operation
        """

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                op_name = operation_name or f"{func.__module__}.{func.__name__}"

                attributes = {
                    "function_name": func.__name__,
                    "module": func.__module__,
                }
                if component:
                    attributes["component"] = component

                with self.tracer.trace_operation(op_name, **attributes) as span:
                    # Add function arguments to span (be careful with sensitive data)
                    if args:
                        span.set_attribute("args_count", len(args))
                    if kwargs:
                        span.set_attribute("kwargs_count", len(kwargs))

                    result = func(*args, **kwargs)

                    # Add result metadata if it's a measurable result
                    if hasattr(result, "__len__"):
                        span.set_attribute("result_count", len(result))

                    return result

            return wrapper

        return decorator

    def trace_async_function(
        self, operation_name: Optional[str] = None, component: Optional[str] = None
    ):
        """
        Decorator to automatically trace async function execution.

        Args:
            operation_name: Custom operation name (defaults to function name)
            component: Component name for the operation
        """

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                op_name = operation_name or f"{func.__module__}.{func.__name__}"

                attributes = {
                    "function_name": func.__name__,
                    "module": func.__module__,
                    "is_async": True,
                }
                if component:
                    attributes["component"] = component

                with self.tracer.trace_operation(op_name, **attributes) as span:
                    # Add function arguments to span
                    if args:
                        span.set_attribute("args_count", len(args))
                    if kwargs:
                        span.set_attribute("kwargs_count", len(kwargs))

                    result = await func(*args, **kwargs)

                    # Add result metadata
                    if hasattr(result, "__len__"):
                        span.set_attribute("result_count", len(result))

                    return result

            return wrapper

        return decorator

    def trace_method(self, operation_name: Optional[str] = None, component: Optional[str] = None):
        """
        Decorator to automatically trace method execution.

        Args:
            operation_name: Custom operation name (defaults to class.method)
            component: Component name for the operation
        """

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                class_name = self.__class__.__name__
                op_name = operation_name or f"{class_name}.{func.__name__}"

                attributes = {
                    "method_name": func.__name__,
                    "class_name": class_name,
                    "module": func.__module__,
                }
                if component:
                    attributes["component"] = component

                with self.tracer.tracer.start_as_current_span(op_name) as span:
                    # Set attributes
                    for key, value in attributes.items():
                        span.set_attribute(key, value)

                    # Add correlation ID
                    correlation_id = CorrelationIdManager.get_correlation_id()
                    if correlation_id:
                        span.set_attribute("correlation_id", correlation_id)

                    try:
                        result = func(self, *args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise

            return wrapper

        return decorator


class TracingIntegration:
    """
    Integration class for setting up tracing across Memory Engine components.
    """

    def __init__(
        self,
        service_name: str = "memory-engine",
        jaeger_endpoint: Optional[str] = None,
        enable_console_export: bool = False,
    ):
        """
        Initialize tracing integration.

        Args:
            service_name: Name of the service
            jaeger_endpoint: Jaeger collector endpoint
            enable_console_export: Whether to export to console
        """
        self.tracer = MemoryEngineTracer(
            service_name=service_name,
            jaeger_endpoint=jaeger_endpoint,
            enable_console_export=enable_console_export,
        )
        self.decorators = TracingDecorators(self.tracer)
        self.logger = get_logger(__name__, "tracing_integration")

    def instrument_query_engine(self, query_engine):
        """Instrument query engine with tracing."""
        # Add tracing to query methods
        original_execute = query_engine.execute_query

        @functools.wraps(original_execute)
        def traced_execute_query(query, **kwargs):
            query_type = getattr(query, "query_type", "unknown")
            query_id = getattr(query, "query_id", "unknown")

            with self.tracer.trace_query_operation(query_type, query_id) as span:
                span.set_attribute("query_text", str(query)[:1000])  # Truncate large queries
                result = original_execute(query, **kwargs)

                if hasattr(result, "__len__"):
                    span.set_attribute("result_count", len(result))

                return result

        query_engine.execute_query = traced_execute_query
        self.logger.info("Query engine instrumented with tracing")

    def instrument_storage_layer(self, storage):
        """Instrument storage layer with tracing."""
        # Add tracing to storage operations
        for method_name in ["store", "retrieve", "update", "delete"]:
            if hasattr(storage, method_name):
                original_method = getattr(storage, method_name)

                @functools.wraps(original_method)
                def traced_method(*args, **kwargs):
                    storage_type = storage.__class__.__name__
                    with self.tracer.trace_storage_operation(method_name, storage_type) as span:
                        return original_method(*args, **kwargs)

                setattr(storage, method_name, traced_method)

        self.logger.info(
            "Storage layer instrumented with tracing", storage_type=storage.__class__.__name__
        )

    def instrument_embedding_manager(self, embedding_manager):
        """Instrument embedding manager with tracing."""
        # Trace embedding generation
        if hasattr(embedding_manager, "generate_embedding"):
            original_generate = embedding_manager.generate_embedding

            @functools.wraps(original_generate)
            def traced_generate_embedding(text, **kwargs):
                text_length = len(text) if text else 0
                with self.tracer.trace_embedding_operation("generate", text_length) as span:
                    span.set_attribute("text_preview", text[:100] if text else "")
                    return original_generate(text, **kwargs)

            embedding_manager.generate_embedding = traced_generate_embedding

        # Trace embedding search
        if hasattr(embedding_manager, "search_similar"):
            original_search = embedding_manager.search_similar

            @functools.wraps(original_search)
            def traced_search_similar(query_embedding, **kwargs):
                with self.tracer.trace_embedding_operation("search") as span:
                    limit = kwargs.get("limit", 10)
                    span.set_attribute("search_limit", limit)
                    result = original_search(query_embedding, **kwargs)

                    if hasattr(result, "__len__"):
                        span.set_attribute("results_found", len(result))

                    return result

            embedding_manager.search_similar = traced_search_similar

        self.logger.info("Embedding manager instrumented with tracing")

    def get_tracer(self) -> MemoryEngineTracer:
        """Get the tracer instance."""
        return self.tracer

    def get_decorators(self) -> TracingDecorators:
        """Get tracing decorators."""
        return self.decorators


# Global tracer instance
_global_tracer: Optional[MemoryEngineTracer] = None


def initialize_tracing(
    service_name: str = "memory-engine",
    jaeger_endpoint: Optional[str] = None,
    enable_console_export: bool = False,
) -> MemoryEngineTracer:
    """
    Initialize global tracing.

    Args:
        service_name: Name of the service
        jaeger_endpoint: Jaeger collector endpoint
        enable_console_export: Whether to export to console

    Returns:
        Configured tracer instance
    """
    global _global_tracer
    _global_tracer = MemoryEngineTracer(
        service_name=service_name,
        jaeger_endpoint=jaeger_endpoint,
        enable_console_export=enable_console_export,
    )
    return _global_tracer


def get_tracer() -> MemoryEngineTracer:
    """
    Get the global tracer instance.

    Returns:
        Global tracer instance

    Raises:
        RuntimeError: If tracing has not been initialized
    """
    if _global_tracer is None:
        raise RuntimeError("Tracing not initialized. Call initialize_tracing() first.")
    return _global_tracer


# Convenience decorators using global tracer
def trace_function(operation_name: Optional[str] = None, component: Optional[str] = None):
    """Decorator for tracing functions with global tracer."""

    def decorator(func):
        tracer = get_tracer()
        decorators = TracingDecorators(tracer)
        return decorators.trace_function(operation_name, component)(func)

    return decorator


def trace_method(operation_name: Optional[str] = None, component: Optional[str] = None):
    """Decorator for tracing methods with global tracer."""

    def decorator(func):
        tracer = get_tracer()
        decorators = TracingDecorators(tracer)
        return decorators.trace_method(operation_name, component)(func)

    return decorator
