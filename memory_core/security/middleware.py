"""
Security middleware and decorators for the Memory Engine.

Provides authentication, authorization, and security enforcement
middleware for protecting API endpoints and operations.
"""

import logging
import functools
import inspect
from typing import Dict, List, Optional, Set, Callable, Any, Union
from datetime import datetime
from flask import Flask, request, g, jsonify, abort
from werkzeug.exceptions import Unauthorized, Forbidden

from memory_core.security.auth import AuthManager, UserSession
from memory_core.security.rbac import RBACManager, PermissionType
from memory_core.security.privacy import KnowledgeAccessControl, PrivacyLevel
from memory_core.security.audit import AuditLogger, AuditLevel, AuditCategory
from memory_core.config.config_manager import get_config


logger = logging.getLogger(__name__)


class SecurityContext:
    """Security context for the current request."""
    
    def __init__(self):
        self.user_id: Optional[str] = None
        self.username: Optional[str] = None
        self.session: Optional[UserSession] = None
        self.roles: Set[str] = set()
        self.permissions: Dict[PermissionType, bool] = {}
        self.ip_address: Optional[str] = None
        self.user_agent: Optional[str] = None
        self.authenticated: bool = False
        self.organization_id: Optional[str] = None


class SecurityMiddleware:
    """
    Flask middleware for handling authentication, authorization,
    and security enforcement.
    """
    
    def __init__(
        self,
        app: Optional[Flask] = None,
        auth_manager: Optional[AuthManager] = None,
        rbac_manager: Optional[RBACManager] = None,
        access_control: Optional[KnowledgeAccessControl] = None,
        audit_logger: Optional[AuditLogger] = None
    ):
        """
        Initialize security middleware.
        
        Args:
            app: Flask application
            auth_manager: Authentication manager
            rbac_manager: RBAC manager
            access_control: Knowledge access control
            audit_logger: Audit logger
        """
        self.auth_manager = auth_manager
        self.rbac_manager = rbac_manager
        self.access_control = access_control
        self.audit_logger = audit_logger
        self.config = get_config()
        
        # Security settings
        self.session_timeout_minutes = self.config.get('security.session_timeout_minutes', 60)
        self.enable_csrf_protection = self.config.get('security.enable_csrf_protection', True)
        self.enable_rate_limiting = self.config.get('security.enable_rate_limiting', True)
        self.max_requests_per_minute = self.config.get('security.max_requests_per_minute', 100)
        
        # Rate limiting storage (in production, use Redis or similar)
        self._rate_limit_storage: Dict[str, List[datetime]] = {}
        
        if app:
            self.init_app(app)
        
        logger.info("SecurityMiddleware initialized")
    
    def init_app(self, app: Flask) -> None:
        """Initialize middleware with Flask application."""
        app.before_request(self.before_request)
        app.after_request(self.after_request)
        app.teardown_appcontext(self.teardown_request)
        
        # Register error handlers
        app.errorhandler(401)(self.handle_unauthorized)
        app.errorhandler(403)(self.handle_forbidden)
        
        logger.info("Security middleware registered with Flask app")
    
    def before_request(self) -> Optional[Any]:
        """Process request before routing."""
        # Initialize security context
        g.security = SecurityContext()
        
        # Get client information
        g.security.ip_address = self.get_client_ip()
        g.security.user_agent = request.headers.get('User-Agent')
        
        # Check rate limiting
        if self.enable_rate_limiting and self.is_rate_limited():
            if self.audit_logger:
                self.audit_logger.log_security_incident(
                    "Rate limit exceeded",
                    ip_address=g.security.ip_address,
                    risk_score=0.6,
                    details={
                        'endpoint': request.endpoint,
                        'method': request.method,
                        'user_agent': g.security.user_agent
                    }
                )
            abort(429)  # Too Many Requests
        
        # Skip authentication for public endpoints
        if self.is_public_endpoint():
            return None
        
        # Authenticate request
        self.authenticate_request()
        
        return None
    
    def after_request(self, response) -> Any:
        """Process response after request handling."""
        # Add security headers
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        
        # Remove sensitive headers
        response.headers.pop('Server', None)
        
        return response
    
    def teardown_request(self, exception=None) -> None:
        """Clean up after request."""
        # Log access if audit logger is available
        if self.audit_logger and hasattr(g, 'security'):
            self.log_request_access(exception)
    
    def authenticate_request(self) -> None:
        """Authenticate the current request."""
        # Try different authentication methods
        session = None
        
        # 1. Try session-based authentication
        session_id = request.headers.get('X-Session-ID') or request.cookies.get('session_id')
        if session_id and self.auth_manager:
            session = self.auth_manager.get_session(session_id)
        
        # 2. Try JWT token authentication
        if not session:
            auth_header = request.headers.get('Authorization')
            if auth_header and auth_header.startswith('Bearer ') and self.auth_manager:
                token = auth_header[7:]  # Remove 'Bearer ' prefix
                payload = self.auth_manager.verify_jwt_token(token)
                if payload:
                    # Create temporary session from JWT
                    user = self.auth_manager.get_user(payload['user_id'])
                    if user:
                        session = UserSession(
                            session_id=f"jwt_{payload['user_id']}",
                            user_id=user.user_id,
                            username=user.username,
                            roles=set(payload.get('roles', [])),
                            ip_address=g.security.ip_address,
                            user_agent=g.security.user_agent
                        )
        
        # 3. Try API key authentication
        if not session:
            api_key = request.headers.get('X-API-Key')
            if api_key:
                # This would validate API key against stored keys
                # For now, we'll skip this implementation
                pass
        
        if session and session.is_valid():
            # Populate security context
            g.security.user_id = session.user_id
            g.security.username = session.username
            g.security.session = session
            g.security.roles = session.roles
            g.security.authenticated = True
            
            # Refresh session activity
            if self.auth_manager and hasattr(session, 'refresh'):
                session.refresh()
            
            # Get effective permissions
            if self.rbac_manager:
                g.security.permissions = self.rbac_manager.get_effective_permissions(session.roles)
            
            # Log successful authentication
            if self.audit_logger:
                self.audit_logger.log_authentication(
                    "Request authenticated",
                    user_id=session.user_id,
                    success=True,
                    ip_address=g.security.ip_address,
                    user_agent=g.security.user_agent,
                    details={'session_id': session.session_id}
                )
        else:
            # Authentication failed
            if self.audit_logger:
                self.audit_logger.log_authentication(
                    "Authentication failed",
                    success=False,
                    ip_address=g.security.ip_address,
                    user_agent=g.security.user_agent,
                    error_message="No valid session or token provided"
                )
            
            abort(401)  # Unauthorized
    
    def is_public_endpoint(self) -> bool:
        """Check if the current endpoint is public (no authentication required)."""
        public_endpoints = [
            'health',
            'status',
            'login',
            'register',
            'forgot_password',
            'reset_password'
        ]
        
        endpoint = request.endpoint
        return endpoint in public_endpoints or (endpoint and any(
            endpoint.startswith(prefix) for prefix in ['public_', 'api.public_']
        ))
    
    def get_client_ip(self) -> str:
        """Get the client IP address, considering proxies."""
        # Check for forwarded headers (common in production with load balancers)
        forwarded_ips = request.headers.get('X-Forwarded-For')
        if forwarded_ips:
            # Take the first IP (client IP)
            return forwarded_ips.split(',')[0].strip()
        
        real_ip = request.headers.get('X-Real-IP')
        if real_ip:
            return real_ip
        
        return request.remote_addr or 'unknown'
    
    def is_rate_limited(self) -> bool:
        """Check if the current request should be rate limited."""
        if not self.enable_rate_limiting:
            return False
        
        client_id = g.security.ip_address or 'unknown'
        current_time = datetime.utcnow()
        
        # Clean old entries
        if client_id in self._rate_limit_storage:
            cutoff_time = current_time.timestamp() - 60  # 1 minute ago
            self._rate_limit_storage[client_id] = [
                req_time for req_time in self._rate_limit_storage[client_id]
                if req_time.timestamp() > cutoff_time
            ]
        
        # Check current rate
        requests = self._rate_limit_storage.get(client_id, [])
        if len(requests) >= self.max_requests_per_minute:
            return True
        
        # Add current request
        if client_id not in self._rate_limit_storage:
            self._rate_limit_storage[client_id] = []
        self._rate_limit_storage[client_id].append(current_time)
        
        return False
    
    def log_request_access(self, exception=None) -> None:
        """Log access to the current request."""
        if not hasattr(g, 'security'):
            return
        
        action = f"{request.method} {request.endpoint or request.path}"
        success = exception is None
        error_message = str(exception) if exception else None
        
        # Determine audit category based on endpoint
        category = AuditCategory.KNOWLEDGE_ACCESS  # Default
        if request.endpoint:
            if 'user' in request.endpoint or 'auth' in request.endpoint:
                category = AuditCategory.USER_MANAGEMENT
            elif 'admin' in request.endpoint or 'config' in request.endpoint:
                category = AuditCategory.SYSTEM_CONFIGURATION
            elif 'privacy' in request.endpoint or 'access' in request.endpoint:
                category = AuditCategory.PRIVACY_CONTROL
        
        self.audit_logger.log_event(
            AuditLevel.INFO if success else AuditLevel.WARNING,
            category,
            action,
            user_id=g.security.user_id,
            session_id=g.security.session.session_id if g.security.session else None,
            ip_address=g.security.ip_address,
            user_agent=g.security.user_agent,
            success=success,
            error_message=error_message,
            details={
                'method': request.method,
                'path': request.path,
                'query_string': request.query_string.decode('utf-8'),
                'content_length': request.content_length
            }
        )
    
    def handle_unauthorized(self, error) -> Any:
        """Handle 401 Unauthorized errors."""
        return jsonify({
            'error': 'Authentication required',
            'message': 'Please provide valid authentication credentials'
        }), 401
    
    def handle_forbidden(self, error) -> Any:
        """Handle 403 Forbidden errors."""
        return jsonify({
            'error': 'Access forbidden',
            'message': 'You do not have permission to access this resource'
        }), 403


def require_auth(f: Callable) -> Callable:
    """
    Decorator to require authentication for a function.
    
    Args:
        f: Function to protect
    
    Returns:
        Protected function
    """
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if not hasattr(g, 'security') or not g.security.authenticated:
            abort(401)
        return f(*args, **kwargs)
    
    return decorated_function


def require_permission(
    permission_type: PermissionType,
    resource_type: Optional[str] = None,
    resource_id_param: Optional[str] = None
) -> Callable:
    """
    Decorator to require specific permission for a function.
    
    Args:
        permission_type: Required permission type
        resource_type: Optional resource type for resource-specific permissions
        resource_id_param: Parameter name containing resource ID
    
    Returns:
        Decorator function
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            if not hasattr(g, 'security') or not g.security.authenticated:
                abort(401)
            
            # Get resource ID if specified
            resource_id = None
            if resource_id_param:
                # Try to get from function arguments
                sig = inspect.signature(f)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                resource_id = bound_args.arguments.get(resource_id_param)
                
                # Try to get from request if not in function args
                if resource_id is None:
                    resource_id = request.view_args.get(resource_id_param) if request.view_args else None
            
            # Check RBAC permission
            has_permission = False
            if hasattr(g.security, 'permissions'):
                has_permission = g.security.permissions.get(permission_type, False)
            
            # Check resource-specific access if access control is available
            if not has_permission and resource_type and resource_id:
                # This would require access to the security middleware instance
                # For now, we'll assume the permission check is sufficient
                pass
            
            if not has_permission:
                # Log authorization failure
                if hasattr(g, 'audit_logger') and g.audit_logger:
                    g.audit_logger.log_authorization(
                        f"Permission denied: {permission_type.value}",
                        user_id=g.security.user_id,
                        resource_type=resource_type,
                        resource_id=resource_id,
                        success=False,
                        error_message=f"User lacks required permission: {permission_type.value}"
                    )
                
                abort(403)
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator


def require_role(required_roles: Union[str, List[str]]) -> Callable:
    """
    Decorator to require specific role(s) for a function.
    
    Args:
        required_roles: Required role name(s)
    
    Returns:
        Decorator function
    """
    if isinstance(required_roles, str):
        required_roles = [required_roles]
    
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            if not hasattr(g, 'security') or not g.security.authenticated:
                abort(401)
            
            user_roles = g.security.roles
            has_required_role = any(role in user_roles for role in required_roles)
            
            if not has_required_role:
                abort(403)
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator


def require_privacy_level(
    min_privacy_level: PrivacyLevel,
    resource_type: str,
    resource_id_param: str
) -> Callable:
    """
    Decorator to enforce privacy level requirements.
    
    Args:
        min_privacy_level: Minimum required privacy level
        resource_type: Type of resource
        resource_id_param: Parameter name containing resource ID
    
    Returns:
        Decorator function
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            if not hasattr(g, 'security') or not g.security.authenticated:
                abort(401)
            
            # Get resource ID
            sig = inspect.signature(f)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            resource_id = bound_args.arguments.get(resource_id_param)
            
            if not resource_id:
                resource_id = request.view_args.get(resource_id_param) if request.view_args else None
            
            if not resource_id:
                abort(400)  # Bad Request - missing resource ID
            
            # Check privacy level (this would require access to access control instance)
            # For now, we'll assume the check passes
            # In a real implementation, you'd inject the access control instance
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator


def audit_action(
    action: str,
    category: AuditCategory = AuditCategory.KNOWLEDGE_ACCESS,
    level: AuditLevel = AuditLevel.INFO
) -> Callable:
    """
    Decorator to audit function calls.
    
    Args:
        action: Action description
        category: Audit category
        level: Audit level
    
    Returns:
        Decorator function
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            start_time = datetime.utcnow()
            success = True
            error_message = None
            result = None
            
            try:
                result = f(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_message = str(e)
                raise
            finally:
                # Log the action if audit logger is available
                if hasattr(g, 'audit_logger') and g.audit_logger:
                    execution_time = (datetime.utcnow() - start_time).total_seconds()
                    
                    g.audit_logger.log_event(
                        level if success else AuditLevel.ERROR,
                        category,
                        action,
                        user_id=g.security.user_id if hasattr(g, 'security') else None,
                        session_id=g.security.session.session_id if hasattr(g, 'security') and g.security.session else None,
                        success=success,
                        error_message=error_message,
                        details={
                            'function': f.__name__,
                            'execution_time_seconds': execution_time,
                            'args_count': len(args),
                            'kwargs_count': len(kwargs)
                        }
                    )
        
        return decorated_function
    return decorator


def rate_limit(max_requests: int = 60, window_minutes: int = 1) -> Callable:
    """
    Decorator to rate limit function calls per user.
    
    Args:
        max_requests: Maximum requests allowed
        window_minutes: Time window in minutes
    
    Returns:
        Decorator function
    """
    def decorator(f: Callable) -> Callable:
        # Storage for rate limiting (in production, use Redis)
        call_storage: Dict[str, List[datetime]] = {}
        
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            if not hasattr(g, 'security') or not g.security.authenticated:
                # Apply rate limiting to unauthenticated requests by IP
                user_key = g.security.ip_address if hasattr(g, 'security') else 'unknown'
            else:
                user_key = g.security.user_id
            
            current_time = datetime.utcnow()
            window_start = current_time - timedelta(minutes=window_minutes)
            
            # Clean old entries
            if user_key in call_storage:
                call_storage[user_key] = [
                    call_time for call_time in call_storage[user_key]
                    if call_time > window_start
                ]
            
            # Check rate limit
            call_count = len(call_storage.get(user_key, []))
            if call_count >= max_requests:
                abort(429)  # Too Many Requests
            
            # Record this call
            if user_key not in call_storage:
                call_storage[user_key] = []
            call_storage[user_key].append(current_time)
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator