# Security Framework

The Memory Engine includes a comprehensive security framework designed to protect sensitive knowledge data and ensure secure access control. This documentation covers all security features and best practices.

## Table of Contents

- [Authentication System](#authentication-system)
- [Role-Based Access Control (RBAC)](#role-based-access-control-rbac)
- [Knowledge Privacy Controls](#knowledge-privacy-controls)
- [Data Encryption](#data-encryption)
- [Audit Logging](#audit-logging)
- [Security Middleware](#security-middleware)
- [Configuration](#configuration)
- [Best Practices](#best-practices)

## Authentication System

### Overview

The Memory Engine provides a robust authentication system with multi-factor capabilities, session management, and secure password handling.

### Key Features

- **Secure Password Hashing**: Uses bcrypt with configurable rounds
- **JWT Token Support**: Stateless authentication for API access
- **Session Management**: Secure session handling with expiration
- **Account Security**: Automatic lockout after failed attempts
- **Multi-user Support**: Comprehensive user management

### User Management

```python
from memory_core.security.auth import AuthManager

# Initialize authentication manager
auth_manager = AuthManager(secret_key="your-secret-key")

# Create a new user
user = auth_manager.create_user(
    username="admin",
    email="admin@company.com",
    password="SecurePass123!",
    roles={"super_admin"}
)

# Authenticate user
authenticated_user = auth_manager.authenticate("admin", "SecurePass123!")

# Create session
session = auth_manager.create_session(authenticated_user, ip_address="192.168.1.1")
```

### Password Requirements

- Minimum 8 characters
- At least one uppercase letter
- At least one lowercase letter
- At least one digit
- At least one special character
- Configurable complexity rules

### Session Security

- Automatic expiration (configurable)
- IP address tracking
- User agent validation
- Secure session refresh
- Force logout capabilities

## Role-Based Access Control (RBAC)

### Overview

The RBAC system provides fine-grained access control with hierarchical roles and extensive permission management.

### System Roles

#### Super Administrator
- **Full system access**
- User management
- System configuration
- All knowledge operations

#### Knowledge Administrator
- Knowledge management
- Privacy level control
- Quality management
- Bulk operations

#### Knowledge Editor
- Create and edit knowledge
- Relationship management
- Basic analytics access

#### Knowledge Reader
- Read-only access to permitted knowledge
- Basic search and query capabilities

#### User Manager
- User account management
- Role assignment
- Session management

#### System Monitor
- System health monitoring
- Performance metrics access
- Audit log viewing

### Custom Roles

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

### Permission System

The system includes 26+ built-in permissions across categories:
- **Knowledge Operations**: Create, read, update, delete, search
- **User Management**: Create, update, delete users
- **System Administration**: Configuration, monitoring, security
- **Analytics**: View metrics, generate reports
- **Quality Control**: Assess quality, resolve contradictions

## Knowledge Privacy Controls

### Privacy Levels

The system supports five privacy levels for knowledge resources:

#### PUBLIC
- Accessible to all authenticated users
- No special permissions required
- Default level for general knowledge

#### INTERNAL
- Accessible to organization members
- Requires internal access role
- Company-wide information

#### CONFIDENTIAL
- Restricted access with explicit permissions
- Requires confidential access role
- Sensitive business information

#### RESTRICTED
- Highly restricted access
- Admin approval required
- Critical security information

#### PRIVATE
- Owner-only access
- Personal or highly sensitive data
- Strictest access controls

### Access Control

```python
from memory_core.security.privacy import KnowledgeAccessControl, PrivacyLevel

access_control = KnowledgeAccessControl(rbac_manager)

# Set privacy level
access_control.set_privacy_level(
    resource_type="node",
    resource_id="sensitive_node_123",
    privacy_level=PrivacyLevel.CONFIDENTIAL,
    set_by_user_id="admin_user_id"
)

# Create specific access rule
access_control.create_access_rule(
    resource_type="node",
    resource_id="sensitive_node_123",
    permissions={"knowledge:read", "knowledge:update"},
    user_id="editor_user_id",
    created_by="admin_user_id"
)

# Check access
can_access = access_control.check_access(
    user_id="editor_user_id",
    user_roles={"knowledge_editor"},
    resource_type="node",
    resource_id="sensitive_node_123",
    permission_type=PermissionType.KNOWLEDGE_READ
)
```

## Data Encryption

### Overview

The Memory Engine provides comprehensive encryption for data at rest and in transit using industry-standard algorithms.

### Encryption Algorithms

- **AES-256-GCM**: Default for most data (authenticated encryption)
- **Fernet**: Symmetric encryption for session data
- **RSA-2048/4096**: Asymmetric encryption for key exchange
- **Hybrid Encryption**: RSA + AES for large data

### Encryption Scopes

Different data types use appropriate encryption:

```python
from memory_core.security.encryption import EncryptionManager, EncryptionScope

encryption_manager = EncryptionManager()

# Encrypt user data
encrypted_user_data = encryption_manager.encrypt_string(
    "sensitive user information",
    EncryptionScope.USER_DATA
)

# Encrypt knowledge content
encrypted_knowledge = encryption_manager.encrypt_string(
    "confidential knowledge content",
    EncryptionScope.KNOWLEDGE_CONTENT
)

# Encrypt API keys
encrypted_api_key = encryption_manager.encrypt_string(
    "sk-1234567890abcdef",
    EncryptionScope.API_KEYS
)
```

### Key Management

- **Automatic Key Generation**: Secure random key generation
- **Key Rotation**: Configurable automatic rotation (default: 90 days)
- **Key Versioning**: Multiple key versions for data migration
- **Backup Keys**: Secure key backup and recovery
- **Hardware Security**: Support for HSM integration

### Key Rotation

```python
# Manual key rotation
new_key = encryption_manager.rotate_key(EncryptionScope.USER_DATA)

# Automatic rotation for expired keys
rotated_keys = encryption_manager.auto_rotate_expired_keys()

# Check expiring keys
expiring_keys = encryption_manager.check_key_expiration()
```

## Audit Logging

### Overview

Comprehensive audit logging tracks all security-relevant events for compliance and security monitoring.

### Audit Categories

- **Authentication**: Login attempts, session management
- **Authorization**: Permission checks, access denials
- **Knowledge Access**: Data reading, querying
- **Knowledge Modification**: Data creation, updates, deletion
- **Privacy Control**: Privacy level changes, access rule modifications
- **User Management**: User creation, role changes
- **System Events**: Configuration changes, system operations
- **Security Incidents**: Failed attempts, suspicious activity
- **Compliance**: Regulatory compliance events
- **Error Events**: Security-related errors
- **Performance**: Security performance metrics
- **Integration**: External system interactions

### Audit Implementation

```python
from memory_core.security.audit import AuditLogger, AuditLevel, AuditCategory

audit_logger = AuditLogger()

# Log authentication event
audit_logger.log_authentication(
    "User login successful",
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
    session_id="session_789",
    details={"query": "search algorithms"}
)

# Log security incident
audit_logger.log_security_incident(
    "Multiple failed login attempts",
    user_id="user_123",
    severity="high",
    ip_address="192.168.1.100",
    details={"attempts": 5, "timeframe": "5 minutes"}
)
```

### Security Monitoring

- **Threat Detection**: Automatic detection of suspicious patterns
- **Risk Scoring**: Dynamic risk assessment for events
- **Alerting**: Real-time security alerts
- **Compliance Reporting**: Automated compliance reports
- **Correlation**: Event correlation for threat analysis

## Security Middleware

### Flask Integration

The security framework includes Flask middleware for web applications:

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

# Role-based decorator
@app.route('/editor')
@security_middleware.require_role("knowledge_editor")
def editor_endpoint():
    return "Editor content"
```

### Security Headers

Automatic security headers:
- Content Security Policy (CSP)
- X-Frame-Options
- X-Content-Type-Options
- Strict-Transport-Security
- X-XSS-Protection

### Rate Limiting

```python
# Configure rate limiting
@security_middleware.rate_limit(requests=100, window=3600)  # 100 req/hour
def api_endpoint():
    return "Rate limited endpoint"
```

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
ENCRYPTION_ENABLE_COMPRESSION=true

# Audit Logging
AUDIT_LOG_LEVEL=INFO
AUDIT_ENABLE_COMPLIANCE=true
AUDIT_RETENTION_DAYS=365

# Security
SECURITY_ENABLE_RATE_LIMITING=true
SECURITY_ENABLE_CSRF_PROTECTION=true
SECURITY_ENABLE_SECURITY_HEADERS=true
```

### Configuration File

```yaml
security:
  authentication:
    secret_key: "${AUTH_SECRET_KEY}"
    token_expiry: 3600
    session_timeout: 7200
    max_failed_attempts: 5
    password_requirements:
      min_length: 8
      require_uppercase: true
      require_lowercase: true
      require_digits: true
      require_special: true
  
  encryption:
    default_algorithm: "AES_256_GCM"
    key_rotation_days: 90
    enable_compression: true
    scopes:
      user_data:
        algorithm: "AES_256_GCM"
        required: true
      knowledge_content:
        algorithm: "AES_256_GCM"
        required: false
        compress: true
  
  audit:
    log_level: "INFO"
    enable_compliance: true
    retention_days: 365
    categories:
      - "authentication"
      - "authorization"
      - "knowledge_access"
      - "security_incident"
```

## Best Practices

### Authentication

1. **Strong Passwords**: Enforce password complexity requirements
2. **Regular Rotation**: Implement password rotation policies
3. **Multi-Factor**: Consider implementing MFA for high-privilege accounts
4. **Session Management**: Use secure session handling
5. **Token Security**: Protect JWT tokens and implement proper expiration

### Authorization

1. **Principle of Least Privilege**: Grant minimum required permissions
2. **Role Hierarchy**: Use role inheritance effectively
3. **Regular Reviews**: Audit permissions regularly
4. **Separation of Duties**: Implement proper role separation

### Data Protection

1. **Encryption**: Encrypt sensitive data at rest and in transit
2. **Key Management**: Implement proper key rotation and backup
3. **Privacy Levels**: Use appropriate privacy levels for data classification
4. **Access Controls**: Implement fine-grained access controls

### Monitoring

1. **Audit Everything**: Log all security-relevant events
2. **Real-time Monitoring**: Implement real-time security monitoring
3. **Incident Response**: Have incident response procedures
4. **Compliance**: Maintain compliance with regulatory requirements

### Development

1. **Security Reviews**: Conduct security code reviews
2. **Testing**: Implement comprehensive security testing
3. **Dependencies**: Keep security dependencies updated
4. **Secrets Management**: Never hardcode secrets in code

## Security Examples

For complete implementation examples, see:
- `examples/security_example.py` - Comprehensive security framework demonstration
- `tests/test_security_*.py` - Security test suites for reference

## Troubleshooting

### Common Issues

#### Authentication Failures
- Check password requirements
- Verify user account status
- Check for account lockouts

#### Permission Denied
- Verify user roles
- Check permission assignments
- Review privacy level settings

#### Encryption Errors
- Check encryption keys
- Verify algorithm support
- Review key rotation status

#### Audit Log Issues
- Check log file permissions
- Verify audit configuration
- Review log retention settings

For additional troubleshooting, see the main troubleshooting guide in `docs/user/troubleshooting.md`.