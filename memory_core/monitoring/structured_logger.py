"""
Structured logging system with correlation IDs for Memory Engine.

This module provides structured logging capabilities with correlation ID tracking
for distributed request tracing and enhanced observability.
"""

import logging
import uuid
import time
import threading
import contextvars
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
import json


# Context variable for correlation ID
correlation_id_context: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "correlation_id", default=None
)

# Context variable for request ID
request_id_context: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "request_id", default=None
)

# Context variable for user ID
user_id_context: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "user_id", default=None
)


class LogLevel(Enum):
    """Log levels for structured logging."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class LogContext:
    """Context information for structured logs."""

    correlation_id: Optional[str] = None
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None


@dataclass
class LogEvent:
    """Structured log event with metadata."""

    timestamp: float
    level: str
    message: str
    logger_name: str
    context: LogContext
    extra_fields: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert log event to dictionary."""
        result = asdict(self)
        # Flatten context into top level
        context_dict = result.pop("context")
        result.update({k: v for k, v in context_dict.items() if v is not None})
        return result


class CorrelationIdProcessor:
    """Processor to add correlation ID to log records."""

    def __call__(self, logger, method_name, event_dict):
        """Add correlation and request IDs to log event."""
        correlation_id = correlation_id_context.get()
        if correlation_id:
            event_dict["correlation_id"] = correlation_id

        request_id = request_id_context.get()
        if request_id:
            event_dict["request_id"] = request_id

        user_id = user_id_context.get()
        if user_id:
            event_dict["user_id"] = user_id

        return event_dict


class TimestampProcessor:
    """Processor to add consistent timestamps."""

    def __call__(self, logger, method_name, event_dict):
        """Add timestamp to log event."""
        event_dict["timestamp"] = time.time()
        event_dict["timestamp_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime())
        return event_dict


class ComponentProcessor:
    """Processor to add component information."""

    def __init__(self, component: str):
        self.component = component

    def __call__(self, logger, method_name, event_dict):
        """Add component name to log event."""
        event_dict["component"] = self.component
        return event_dict


class MemoryEngineFormatter:
    """Custom formatter for Memory Engine logs."""

    def __call__(self, logger, method_name, event_dict):
        """Format log event for Memory Engine."""
        # Ensure required fields
        event_dict.setdefault("timestamp", time.time())
        event_dict.setdefault("level", method_name)
        event_dict.setdefault("logger", logger.name)

        # Add thread information
        event_dict["thread_id"] = threading.current_thread().ident
        event_dict["thread_name"] = threading.current_thread().name

        return event_dict


class StructuredLogger:
    """
    Structured logger for Memory Engine with correlation ID support.

    Provides structured logging with automatic correlation ID injection,
    request tracing, and JSON formatting for observability.
    """

    def __init__(
        self, name: str, component: Optional[str] = None, log_level: LogLevel = LogLevel.INFO
    ):
        """
        Initialize structured logger.

        Args:
            name: Logger name
            component: Component name for logging context
            log_level: Minimum log level to emit
        """
        self.name = name
        self.component = component or name
        self.log_level = log_level

        # Configure structlog
        self._configure_structlog()

        # Get structured logger
        self.logger = structlog.get_logger(name)

    def _configure_structlog(self):
        """Configure structlog processors and formatters."""
        processors = [
            structlog.stdlib.filter_by_level,
            TimestampProcessor(),
            CorrelationIdProcessor(),
            ComponentProcessor(self.component),
            MemoryEngineFormatter(),
            structlog.dev.ConsoleRenderer(),  # Human-readable for development
        ]

        # Configure structlog
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

    def _log(self, level: LogLevel, message: str, **kwargs):
        """Internal logging method."""
        getattr(self.logger, level.value)(message, **kwargs)

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log error message with optional exception."""
        if error:
            kwargs.update(
                {
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "error_traceback": repr(error.__traceback__) if error.__traceback__ else None,
                }
            )
        self._log(LogLevel.ERROR, message, **kwargs)

    def critical(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log critical message with optional exception."""
        if error:
            kwargs.update(
                {
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "error_traceback": repr(error.__traceback__) if error.__traceback__ else None,
                }
            )
        self._log(LogLevel.CRITICAL, message, **kwargs)

    def with_context(self, **context) -> "StructuredLogger":
        """Create a new logger with additional context."""
        bound_logger = self.logger.bind(**context)
        new_logger = StructuredLogger(self.name, self.component, self.log_level)
        new_logger.logger = bound_logger
        return new_logger


class CorrelationIdManager:
    """Manager for correlation ID lifecycle."""

    @staticmethod
    def generate_correlation_id() -> str:
        """Generate a new correlation ID."""
        return str(uuid.uuid4())

    @staticmethod
    def generate_request_id() -> str:
        """Generate a new request ID."""
        return str(uuid.uuid4())

    @staticmethod
    def set_correlation_id(correlation_id: str):
        """Set correlation ID for current context."""
        correlation_id_context.set(correlation_id)

    @staticmethod
    def set_request_id(request_id: str):
        """Set request ID for current context."""
        request_id_context.set(request_id)

    @staticmethod
    def set_user_id(user_id: str):
        """Set user ID for current context."""
        user_id_context.set(user_id)

    @staticmethod
    def get_correlation_id() -> Optional[str]:
        """Get current correlation ID."""
        return correlation_id_context.get()

    @staticmethod
    def get_request_id() -> Optional[str]:
        """Get current request ID."""
        return request_id_context.get()

    @staticmethod
    def get_user_id() -> Optional[str]:
        """Get current user ID."""
        return user_id_context.get()

    @staticmethod
    def clear_context():
        """Clear all context variables."""
        correlation_id_context.set(None)
        request_id_context.set(None)
        user_id_context.set(None)


class LoggingContext:
    """Context manager for logging with correlation IDs."""

    def __init__(
        self,
        correlation_id: Optional[str] = None,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        """
        Initialize logging context.

        Args:
            correlation_id: Correlation ID (generated if not provided)
            request_id: Request ID (generated if not provided)
            user_id: User ID for request
        """
        self.correlation_id = correlation_id or CorrelationIdManager.generate_correlation_id()
        self.request_id = request_id or CorrelationIdManager.generate_request_id()
        self.user_id = user_id

        # Store previous values for restoration
        self.prev_correlation_id = None
        self.prev_request_id = None
        self.prev_user_id = None

    def __enter__(self):
        """Enter logging context."""
        # Store previous values
        self.prev_correlation_id = CorrelationIdManager.get_correlation_id()
        self.prev_request_id = CorrelationIdManager.get_request_id()
        self.prev_user_id = CorrelationIdManager.get_user_id()

        # Set new values
        CorrelationIdManager.set_correlation_id(self.correlation_id)
        CorrelationIdManager.set_request_id(self.request_id)
        if self.user_id:
            CorrelationIdManager.set_user_id(self.user_id)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit logging context."""
        # Restore previous values
        if self.prev_correlation_id:
            CorrelationIdManager.set_correlation_id(self.prev_correlation_id)
        else:
            correlation_id_context.set(None)

        if self.prev_request_id:
            CorrelationIdManager.set_request_id(self.prev_request_id)
        else:
            request_id_context.set(None)

        if self.prev_user_id:
            CorrelationIdManager.set_user_id(self.prev_user_id)
        else:
            user_id_context.set(None)


class OperationLogger:
    """Logger for tracking operations with timing and metrics."""

    def __init__(self, logger: StructuredLogger, operation: str):
        """
        Initialize operation logger.

        Args:
            logger: Structured logger instance
            operation: Operation name
        """
        self.logger = logger
        self.operation = operation
        self.start_time = None
        self.context = {}

    def start(self, **context):
        """Start operation logging."""
        self.start_time = time.time()
        self.context = context
        self.logger.info(
            f"Starting operation: {self.operation}",
            operation=self.operation,
            operation_status="started",
            **context,
        )

    def success(self, **additional_context):
        """Log successful operation completion."""
        if self.start_time:
            duration_ms = (time.time() - self.start_time) * 1000
            self.logger.info(
                f"Operation completed successfully: {self.operation}",
                operation=self.operation,
                operation_status="success",
                duration_ms=duration_ms,
                **self.context,
                **additional_context,
            )

    def error(self, error: Exception, **additional_context):
        """Log operation error."""
        if self.start_time:
            duration_ms = (time.time() - self.start_time) * 1000
            self.logger.error(
                f"Operation failed: {self.operation}",
                error=error,
                operation=self.operation,
                operation_status="error",
                duration_ms=duration_ms,
                **self.context,
                **additional_context,
            )

    def __enter__(self):
        """Enter operation context."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit operation context."""
        if exc_type:
            self.error(exc_val)
        else:
            self.success()


class JSONFormatter(logging.Formatter):
    """JSON formatter for standard library logging integration."""

    def format(self, record):
        """Format log record as JSON."""
        log_entry = {
            "timestamp": record.created,
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime(record.created)),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread_id": record.thread,
            "thread_name": record.threadName,
        }

        # Add correlation ID if available
        correlation_id = CorrelationIdManager.get_correlation_id()
        if correlation_id:
            log_entry["correlation_id"] = correlation_id

        request_id = CorrelationIdManager.get_request_id()
        if request_id:
            log_entry["request_id"] = request_id

        user_id = CorrelationIdManager.get_user_id()
        if user_id:
            log_entry["user_id"] = user_id

        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


def get_logger(name: str, component: Optional[str] = None) -> StructuredLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name
        component: Component name for context

    Returns:
        Configured structured logger
    """
    return StructuredLogger(name, component)


def configure_logging(log_level: str = "INFO", json_format: bool = False):
    """
    Configure global logging settings.

    Args:
        log_level: Minimum log level
        json_format: Whether to use JSON formatting
    """
    level = getattr(logging, log_level.upper())

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)


# Convenience decorators
def logged_operation(operation_name: str, logger: Optional[StructuredLogger] = None):
    """Decorator for automatically logging operations."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            actual_logger = logger or get_logger(func.__module__)
            with OperationLogger(actual_logger, operation_name) as op_logger:
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    op_logger.error(e)
                    raise

        return wrapper

    return decorator


def with_correlation_id(correlation_id: Optional[str] = None):
    """Decorator for setting correlation ID context."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            with LoggingContext(correlation_id=correlation_id):
                return func(*args, **kwargs)

        return wrapper

    return decorator
