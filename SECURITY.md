# Security Policy

## ⚠️ Important Notice

**This is a personal open-source project developed for educational and research purposes. No security guarantees are provided. Users are responsible for evaluating security requirements for their specific use case.**

## Overview

This document describes the security aspects of the Memory Engine project. As an experimental open-source project, users should conduct their own security evaluation before use in any sensitive environment.

## Supported Versions

Security updates are provided on a best-effort basis:

| Version | Status                |
| ------- | -------------------- |
| 0.3.x   | Current development  |
| 0.2.x   | Maintenance only     |
| 0.1.x   | Legacy               |
| < 0.1   | No support           |

**Note**: This is a personal project with no formal support commitments.

## Available Security Features

Memory Engine includes basic security components for educational purposes:

### 🔐 Authentication & Authorization
- Basic user authentication with password hashing (bcrypt)
- Role-Based Access Control (RBAC) with predefined permissions
- JWT token support for API authentication
- Session management with configurable expiration
- Account lockout after failed login attempts

### 🛡️ Data Protection
- Basic encryption using standard libraries (AES-256-GCM, RSA-2048/4096)
- Privacy levels for knowledge classification
- Configurable key rotation
- Basic data integrity checks

### 📊 Monitoring & Logging
- Audit logging for security events
- Basic monitoring capabilities
- Configurable log retention

### 🌐 Web Security
- Basic security middleware for Flask applications
- Standard security headers
- Basic rate limiting capabilities
- Input validation helpers

### 🗄️ Storage Backend Security (Added in v0.2.0)
- **JanusGraph**: Network-based storage requiring external security measures
- **SQLite**: File-based storage relying on OS file permissions
- **JSON File**: Plaintext storage requiring careful file system security

## Reporting Security Issues

### For Educational/Research Use

If you discover security issues while studying or experimenting with the code:

1. **Feel free to open a public GitHub issue** for educational discussion
2. **Contact @Celebr4tion** for private discussion if needed
3. **Contribute fixes** via pull requests

### For Production Use (Not Recommended)

If you're using this in a sensitive environment despite our recommendations:

1. **Do not** use public GitHub issues for security vulnerabilities
2. **Contact @Celebr4tion** privately on GitHub
3. **Include**: Description, reproduction steps, affected versions

### Response Expectations

As a personal project:
- **Response time**: Best effort, no guarantees
- **Fix timeline**: Depends on maintainer availability
- **No formal SLA**: This is volunteer work

## Security Limitations

### Known Limitations

- **No professional security audit**: Code has not been audited by security professionals
- **Basic implementations**: Security features are educational implementations
- **Single maintainer**: Limited resources for security response
- **Experimental code**: May contain unknown vulnerabilities
- **No warranty**: No guarantees about security effectiveness

### Not Suitable For

- Production systems with sensitive data
- High-security environments
- Compliance-critical applications
- Commercial deployments without additional hardening

## Security Guidelines for Users

### Development/Learning Use

1. **Test Environment**: Use only in test/development environments
2. **No Real Data**: Don't use with real sensitive information
3. **Learning**: Great for understanding security concepts
4. **Experimentation**: Safe for security research and learning

### If You Must Use in Production (Not Recommended)

1. **Security Review**: Conduct thorough security review
2. **Additional Hardening**: Implement additional security measures
3. **Monitoring**: Implement comprehensive monitoring
4. **Backup Strategy**: Secure backup and recovery procedures
5. **Regular Updates**: Keep dependencies updated
6. **Network Security**: Implement proper network controls

### Configuration Security

```bash
# Use strong, unique secrets
AUTH_SECRET_KEY=<generate-strong-random-key>
ENCRYPTION_KEY=<generate-encryption-key>

# Choose storage backend based on security needs
STORAGE_BACKEND=janusgraph  # Network-based, shared
# STORAGE_BACKEND=sqlite    # File-based, single-user
# STORAGE_BACKEND=json_file # Plaintext, development only
```

### Storage Backend Considerations

- **JanusGraph**: Requires external database security (network, authentication, TLS)
- **SQLite**: Relies on file system permissions and OS-level encryption
- **JSON File**: Stores data in plaintext, suitable only for development/testing

## Security Testing

The project includes basic security tests for educational purposes:

```bash
# Run security tests
pytest tests/test_security_*.py -v

# Specific modules
pytest tests/test_security_auth.py -v      # Authentication (24 tests)
pytest tests/test_security_rbac.py -v     # Access control (22 tests)  
pytest tests/test_security_encryption.py -v # Encryption (27 tests)
```

**Note**: Passing tests do not guarantee security - they only verify basic functionality.

## Documentation

Security-related documentation:

- **Security Framework**: [`docs/security/README.md`](docs/security/README.md)
- **Configuration**: [`docs/user/configuration.md`](docs/user/configuration.md)
- **Examples**: [`examples/security_example.py`](examples/security_example.py)

## Contributing Security Improvements

Contributions to improve security are welcome:

1. **Open Issues**: Discuss security improvements
2. **Pull Requests**: Submit security enhancements
3. **Documentation**: Improve security documentation
4. **Testing**: Add security test cases

## Disclaimers

### No Warranty

This software is provided "as is" without any warranty of any kind. The author makes no representations about the suitability of this software for any purpose.

### No Liability

The author shall not be liable for any damages arising from the use of this software, including but not limited to data loss, security breaches, or system compromise.

### Educational Purpose

This project is intended for educational and research purposes. Users are solely responsible for evaluating its security and suitability for their use case.

### Personal Project

This is a personal learning project by one individual. It does not have the resources, expertise, or formal processes of a commercial security product.

## Contact

For questions about security aspects:

- **General Discussion**: [GitHub Discussions](https://github.com/Celebr4tion/memory-engine/discussions)
- **Issues**: [GitHub Issues](https://github.com/Celebr4tion/memory-engine/issues)
- **Private Contact**: @Celebr4tion on GitHub

**Remember**: This is experimental educational software. Evaluate carefully before any serious use.