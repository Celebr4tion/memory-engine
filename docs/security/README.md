# Security Framework

This document describes the security components available in the Memory Engine project. This is an open-source project developed and maintained by one person as a learning and research tool.

## ⚠️ Important Disclaimers

- **No Security Guarantees**: This project provides security features for educational and development purposes. No guarantees are made regarding the security, reliability, or suitability for production use.
- **Use at Your Own Risk**: Users are responsible for evaluating the security requirements of their specific use case.
- **Community Project**: This is a personal open-source project, not a commercial product. No professional support or liability is provided.
- **Security Review Required**: Before using in any sensitive environment, conduct your own security review and testing.

## Table of Contents

- [Authentication System](#authentication-system)
- [Role-Based Access Control (RBAC)](#role-based-access-control-rbac)
- [Knowledge Privacy Controls](#knowledge-privacy-controls)
- [Data Encryption](#data-encryption)
- [Audit Logging](#audit-logging)
- [Security Middleware](#security-middleware)
- [Storage Backend Security](#storage-backend-security)
- [Configuration](#configuration)
- [Development Guidelines](#development-guidelines)

## Authentication System

### Overview

The authentication system provides basic user management and session handling capabilities.

### Available Features

- Password hashing using bcrypt
- JWT token support for API access
- Session management with expiration
- Account lockout after failed attempts
- Basic user management operations

### User Management

```python
from memory_core.security.auth import AuthManager

# Initialize authentication manager
auth_manager = AuthManager(secret_key="your-secret-key")

# Create a new user
user = auth_manager.create_user(
    username="admin",
    email="admin@example.com",
    password="SecurePass123!",
    roles={"admin"}
)

# Authenticate user
authenticated_user = auth_manager.authenticate("admin", "SecurePass123!")

# Create session
session = auth_manager.create_session(authenticated_user, ip_address="192.168.1.1")
```

### Password Requirements

Default password validation includes:
- Minimum 8 characters
- At least one uppercase letter
- At least one lowercase letter  
- At least one digit
- At least one special character

Note: These are basic requirements and may not be sufficient for all security contexts.

### Session Security

- Configurable expiration times
- IP address tracking
- Basic session refresh
- Manual logout capabilities

## Role-Based Access Control (RBAC)

### Overview

The RBAC system provides basic access control with predefined roles and permissions.

### Available Roles

#### Administrator Roles
- **super_admin**: Full system access
- **knowledge_admin**: Knowledge management operations
- **user_manager**: User account management

#### User Roles  
- **knowledge_editor**: Create and edit knowledge
- **knowledge_reader**: Read-only access to permitted knowledge
- **system_monitor**: View system metrics and logs

### Permission System

The system includes basic permissions for:
- Knowledge operations (create, read, update, delete, search)
- User management (create, update, delete users)
- System administration (configuration, monitoring)

```python
from memory_core.security.rbac import RBACManager, PermissionType

rbac_manager = RBACManager()

# Create custom role
custom_role = rbac_manager.create_role(
    name="Data Analyst",
    description="Can analyze knowledge data",
    permissions={
        "knowledge:read",
        "analytics:view",
        "query:execute"
    }
)

# Check permissions
has_permission = rbac_manager.check_permission(
    user_roles={"data_analyst"}, 
    permission_type=PermissionType.KNOWLEDGE_READ
)
```

## Knowledge Privacy Controls

### Privacy Levels

The system supports five privacy levels:

- **PUBLIC**: Accessible to all authenticated users
- **INTERNAL**: Accessible to organization members
- **CONFIDENTIAL**: Restricted access with explicit permissions
- **RESTRICTED**: Highly restricted access
- **PRIVATE**: Owner-only access

### Access Control

```python
from memory_core.security.privacy import KnowledgeAccessControl, PrivacyLevel

access_control = KnowledgeAccessControl(rbac_manager)

# Set privacy level
access_control.set_privacy_level(
    resource_type="node",
    resource_id="node_123",
    privacy_level=PrivacyLevel.CONFIDENTIAL,
    set_by_user_id="admin_user_id"
)

# Check access
can_access = access_control.check_access(
    user_id="user_id",
    user_roles={"knowledge_editor"},
    resource_type="node",
    resource_id="node_123",
    permission_type=PermissionType.KNOWLEDGE_READ
)
```

## Data Encryption

### Overview

The encryption system provides basic data protection capabilities using standard cryptographic libraries.

### Available Algorithms

- **AES-256-GCM**: For most data encryption needs
- **Fernet**: For session data encryption
- **RSA-2048/4096**: For key exchange and small data encryption

### Encryption Scopes

Different data types can use different encryption approaches:

```python
from memory_core.security.encryption import EncryptionManager, EncryptionScope

encryption_manager = EncryptionManager()

# Encrypt user data
encrypted_user_data = encryption_manager.encrypt_string(
    "user information",
    EncryptionScope.USER_DATA
)

# Encrypt knowledge content
encrypted_knowledge = encryption_manager.encrypt_string(
    "knowledge content",
    EncryptionScope.KNOWLEDGE_CONTENT
)
```

### Key Management

- Automatic key generation
- Configurable key rotation (default: 90 days)
- Key versioning support
- Basic key backup capabilities

**Note**: Key management is basic and may require additional security measures for sensitive use cases.

## Audit Logging

### Overview

The audit logging system tracks security-related events for monitoring and compliance purposes.

### Audit Categories

- Authentication events (login attempts, session management)
- Authorization events (permission checks, access denials)
- Knowledge access and modification events
- Privacy control changes
- User management operations
- System configuration changes
- Security incidents

### Implementation

```python
from memory_core.security.audit import AuditLogger, AuditLevel, AuditCategory

audit_logger = AuditLogger()

# Log authentication event
audit_logger.log_authentication(
    "User login attempt",
    user_id="user_123",
    success=True,
    ip_address="192.168.1.100",
    details={"method": "password"}
)

# Log knowledge access
audit_logger.log_knowledge_access(
    "Knowledge node accessed",
    user_id="user_123",
    resource_id="node_456",
    session_id="session_789"
)
```

## Security Middleware

### Flask Integration

Basic security middleware for Flask applications:

```python
from memory_core.security.middleware import SecurityMiddleware
from flask import Flask

app = Flask(__name__)
security_middleware = SecurityMiddleware(auth_manager, rbac_manager)

# Authentication decorator
@app.route('/protected')
@security_middleware.require_authentication
def protected_endpoint():
    return "Protected content"

# Permission-based decorator
@app.route('/admin')
@security_middleware.require_permission(PermissionType.SYSTEM_ADMIN)
def admin_endpoint():
    return "Admin content"
```

### Security Headers

Basic security headers are applied:
- Content Security Policy (CSP)
- X-Frame-Options
- X-Content-Type-Options
- Strict-Transport-Security
- X-XSS-Protection

**Note**: Default configurations may not be suitable for all deployment scenarios.

## Storage Backend Security

### Overview (New in v0.2.0)

The modular storage backend system introduces different security considerations for each storage type:

### JanusGraph Backend
- Network-based database requiring secure connections
- Consider using TLS for database connections
- Network access controls recommended
- Database authentication and authorization

### SQLite Backend
- File-based storage with local file permissions
- File encryption at OS level may be needed
- Backup security considerations
- Single-user focused security model

### JSON File Backend
- Human-readable storage format
- File system permissions crucial
- Consider encryption for sensitive data
- Easy to inspect but also easy to modify

### Security Recommendations by Backend

```python
# Configuration example for secure backends
storage_config = {
    "janusgraph": {
        "host": "localhost",  # Use secure network
        "port": 8182,
        # "tls_enabled": True,  # Configure TLS if available
        # "auth_username": "user",  # Database authentication
        # "auth_password": "pass"
    },
    "sqlite": {
        "database_path": "./secure/knowledge.db",  # Secure directory
        # Consider OS-level encryption
    },
    "json_file": {
        "directory": "./secure/graph",  # Secure directory with proper permissions
        "pretty_print": False  # Less readable for security
    }
}
```

### Backend-Specific Considerations

- **JanusGraph**: Shared database requires network security and database-level access controls
- **SQLite**: Single-user database relies on file system security
- **JSON File**: Plaintext storage requires careful file system security and encryption

## Configuration

### Environment Variables

```bash
# Authentication
AUTH_SECRET_KEY=your-secret-key-here
AUTH_TOKEN_EXPIRY=3600
AUTH_SESSION_TIMEOUT=7200
AUTH_MAX_FAILED_ATTEMPTS=5

# Encryption
ENCRYPTION_DEFAULT_ALGORITHM=AES_256_GCM
ENCRYPTION_KEY_ROTATION_DAYS=90

# Audit Logging
AUDIT_LOG_LEVEL=INFO
AUDIT_RETENTION_DAYS=365

# Storage Security (New in v0.2.0)
STORAGE_BACKEND=janusgraph  # or sqlite, json_file
```

### Configuration File

```yaml
# Storage configuration affects security model
storage:
  graph:
    backend: "janusgraph"  # Choose based on security requirements
    janusgraph:
      host: "localhost"
      port: 8182
    sqlite:
      database_path: "./data/knowledge.db"  # Secure path
    json_file:
      directory: "./data/graph"  # Secure directory
      pretty_print: false  # Less readable

security:
  authentication:
    secret_key: "${AUTH_SECRET_KEY}"
    token_expiry: 3600
    session_timeout: 7200
    max_failed_attempts: 5
  
  encryption:
    default_algorithm: "AES_256_GCM"
    key_rotation_days: 90
    enable_compression: true
  
  audit:
    log_level: "INFO"
    retention_days: 365
```

## Development Guidelines

### Security Considerations

1. **Code Review**: Review security-related code changes
2. **Testing**: Test security features thoroughly
3. **Dependencies**: Keep security dependencies updated
4. **Secrets**: Never commit secrets to version control
5. **Validation**: Validate all user inputs
6. **Error Handling**: Avoid revealing sensitive information in errors

### Deployment Security

1. **Environment**: Use appropriate security configurations for deployment environment
2. **Network**: Implement network security controls
3. **File Permissions**: Set appropriate file and directory permissions
4. **Monitoring**: Monitor security events and logs
5. **Updates**: Keep the system and dependencies updated

### Limitations

- This is educational/research software
- Security features are basic implementations
- Not audited by security professionals
- May contain vulnerabilities
- Not suitable for high-security environments without additional hardening

## Examples

For implementation examples, see:
- `examples/security_example.py` - Basic security feature demonstration
- `tests/test_security_*.py` - Security test suites

## Support

This is a community project with no professional support. For questions or issues:

1. Check the documentation
2. Review the source code
3. Open an issue on GitHub for bugs or feature requests
4. Contribute improvements via pull requests

**Remember**: Evaluate all security features for your specific use case and conduct your own security testing before any production use.