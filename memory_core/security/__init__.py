"""
Security module for the Memory Engine.

This module provides authentication, authorization, encryption, and audit logging
capabilities for secure multi-user knowledge management.
"""

from .auth import AuthManager, User, UserSession
from .rbac import RBACManager, Role, Permission, PermissionType
from .privacy import PrivacyLevel, KnowledgeAccessControl
from .audit import AuditLogger, AuditEvent, AuditLevel
from .encryption import EncryptionManager, EncryptionConfig
from .middleware import SecurityMiddleware, require_auth, require_permission

__all__ = [
    'AuthManager', 'User', 'UserSession',
    'RBACManager', 'Role', 'Permission', 'PermissionType',
    'PrivacyLevel', 'KnowledgeAccessControl',
    'AuditLogger', 'AuditEvent', 'AuditLevel',
    'EncryptionManager', 'EncryptionConfig',
    'SecurityMiddleware', 'require_auth', 'require_permission'
]