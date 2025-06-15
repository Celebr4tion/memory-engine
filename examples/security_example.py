#!/usr/bin/env python3
"""
Security System Integration Example

This example demonstrates the comprehensive security framework for the Memory Engine,
showing authentication, authorization, privacy controls, audit logging, and encryption.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta, UTC
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set up environment
os.environ["GOOGLE_API_KEY"] = "demo-key-not-real"

from memory_core.security.auth import AuthManager, UserStatus
from memory_core.security.rbac import RBACManager, PermissionType
from memory_core.security.privacy import KnowledgeAccessControl, PrivacyLevel
from memory_core.security.audit import AuditLogger, AuditLevel, AuditCategory
from memory_core.security.encryption import EncryptionManager, EncryptionScope


async def main():
    """Demonstrate the comprehensive security system."""

    print("=== Memory Engine Security Framework Demo ===\n")

    # Initialize security components
    print("1. Initializing security components...")

    auth_manager = AuthManager(secret_key="demo-secret-key")
    rbac_manager = RBACManager()
    audit_logger = AuditLogger()
    encryption_manager = EncryptionManager()
    access_control = KnowledgeAccessControl(rbac_manager)

    print("   ✓ Authentication Manager")
    print("   ✓ RBAC Manager")
    print("   ✓ Audit Logger")
    print("   ✓ Encryption Manager")
    print("   ✓ Knowledge Access Control")

    # User Management Demo
    print("\n2. User Management Demo...")

    # Create users with different roles
    admin_user = auth_manager.create_user(
        username="admin", email="admin@company.com", password="AdminPass123!", roles={"super_admin"}
    )

    editor_user = auth_manager.create_user(
        username="editor",
        email="editor@company.com",
        password="EditorPass123!",
        roles={"knowledge_editor"},
    )

    reader_user = auth_manager.create_user(
        username="reader",
        email="reader@company.com",
        password="ReaderPass123!",
        roles={"knowledge_reader"},
    )

    print(f"   ✓ Created admin user: {admin_user.username}")
    print(f"   ✓ Created editor user: {editor_user.username}")
    print(f"   ✓ Created reader user: {reader_user.username}")

    # Authentication Demo
    print("\n3. Authentication Demo...")

    # Authenticate users
    auth_admin = auth_manager.authenticate("admin", "AdminPass123!")
    auth_editor = auth_manager.authenticate("editor", "EditorPass123!")

    # Create sessions
    admin_session = auth_manager.create_session(auth_admin, ip_address="192.168.1.100")
    editor_session = auth_manager.create_session(auth_editor, ip_address="192.168.1.101")

    print(f"   ✓ Admin authenticated, session: {admin_session.session_id[:8]}...")
    print(f"   ✓ Editor authenticated, session: {editor_session.session_id[:8]}...")

    # Log authentication events
    audit_logger.log_authentication(
        "User login successful",
        user_id=admin_user.user_id,
        success=True,
        ip_address="192.168.1.100",
    )

    # RBAC Demo
    print("\n4. Role-Based Access Control Demo...")

    # Check permissions
    admin_roles = auth_admin.roles
    editor_roles = auth_editor.roles

    admin_can_delete = rbac_manager.check_permission(admin_roles, PermissionType.KNOWLEDGE_DELETE)
    editor_can_delete = rbac_manager.check_permission(editor_roles, PermissionType.KNOWLEDGE_DELETE)
    editor_can_create = rbac_manager.check_permission(editor_roles, PermissionType.KNOWLEDGE_CREATE)

    print(f"   ✓ Admin can delete knowledge: {admin_can_delete}")
    print(f"   ✓ Editor can delete knowledge: {editor_can_delete}")
    print(f"   ✓ Editor can create knowledge: {editor_can_create}")

    # Privacy and Access Control Demo
    print("\n5. Privacy and Access Control Demo...")

    # Set up knowledge privacy levels
    knowledge_node_1 = "node_public_123"
    knowledge_node_2 = "node_confidential_456"
    knowledge_node_3 = "node_private_789"

    access_control.set_privacy_level(
        "node", knowledge_node_1, PrivacyLevel.PUBLIC, admin_user.user_id
    )

    access_control.set_privacy_level(
        "node", knowledge_node_2, PrivacyLevel.CONFIDENTIAL, admin_user.user_id
    )

    access_control.set_privacy_level(
        "node", knowledge_node_3, PrivacyLevel.PRIVATE, admin_user.user_id
    )

    print(f"   ✓ Set node {knowledge_node_1} as PUBLIC")
    print(f"   ✓ Set node {knowledge_node_2} as CONFIDENTIAL")
    print(f"   ✓ Set node {knowledge_node_3} as PRIVATE")

    # Check access permissions
    editor_can_access_public = access_control.check_access(
        editor_user.user_id, editor_roles, "node", knowledge_node_1, PermissionType.KNOWLEDGE_READ
    )

    editor_can_access_confidential = access_control.check_access(
        editor_user.user_id, editor_roles, "node", knowledge_node_2, PermissionType.KNOWLEDGE_READ
    )

    editor_can_access_private = access_control.check_access(
        editor_user.user_id, editor_roles, "node", knowledge_node_3, PermissionType.KNOWLEDGE_READ
    )

    print(f"   ✓ Editor can access public node: {editor_can_access_public}")
    print(f"   ✓ Editor can access confidential node: {editor_can_access_confidential}")
    print(f"   ✓ Editor can access private node: {editor_can_access_private}")

    # Create explicit access rule for confidential data
    access_control.create_access_rule(
        "node",
        knowledge_node_2,
        permissions={"knowledge:read", "knowledge:update"},
        user_id=editor_user.user_id,
        created_by=admin_user.user_id,
    )

    # Re-check access
    editor_can_access_confidential_now = access_control.check_access(
        editor_user.user_id, editor_roles, "node", knowledge_node_2, PermissionType.KNOWLEDGE_READ
    )

    print(
        f"   ✓ Editor can access confidential node (after rule): {editor_can_access_confidential_now}"
    )

    # Encryption Demo
    print("\n6. Data Encryption Demo...")

    # Encrypt different types of data
    sensitive_user_data = {
        "user_id": editor_user.user_id,
        "email": editor_user.email,
        "preferences": {"theme": "dark", "notifications": True},
    }

    knowledge_content = "This is sensitive knowledge content about our proprietary algorithms."

    api_key = "sk-1234567890abcdef"

    # Encrypt data
    encrypted_user_data = encryption_manager.encrypt_string(
        str(sensitive_user_data), EncryptionScope.USER_DATA
    )

    encrypted_knowledge = encryption_manager.encrypt_string(
        knowledge_content, EncryptionScope.KNOWLEDGE_CONTENT
    )

    encrypted_api_key = encryption_manager.encrypt_string(api_key, EncryptionScope.API_KEYS)

    print("   ✓ Encrypted user data")
    print("   ✓ Encrypted knowledge content")
    print("   ✓ Encrypted API key")

    # Decrypt data
    decrypted_user_data = encryption_manager.decrypt_string(encrypted_user_data)
    decrypted_knowledge = encryption_manager.decrypt_string(encrypted_knowledge)
    decrypted_api_key = encryption_manager.decrypt_string(encrypted_api_key)

    print("   ✓ Decrypted user data successfully")
    print("   ✓ Decrypted knowledge content successfully")
    print("   ✓ Decrypted API key successfully")

    # Audit and Compliance Demo
    print("\n7. Audit and Compliance Demo...")

    # Log various security events
    audit_logger.log_knowledge_access(
        "Knowledge node accessed",
        user_id=editor_user.user_id,
        resource_id=knowledge_node_1,
        session_id=editor_session.session_id,
        details={"access_method": "direct_query", "query": "search algorithms"},
    )

    audit_logger.log_knowledge_modification(
        "Knowledge node updated",
        user_id=editor_user.user_id,
        resource_id=knowledge_node_1,
        session_id=editor_session.session_id,
        details={"changes": "updated_content", "old_version": "v1.0", "new_version": "v1.1"},
    )

    audit_logger.log_privacy_control(
        "Privacy level changed",
        user_id=admin_user.user_id,
        resource_type="node",
        resource_id=knowledge_node_2,
        details={"old_level": "public", "new_level": "confidential", "reason": "data_sensitivity"},
    )

    print("   ✓ Logged knowledge access event")
    print("   ✓ Logged knowledge modification event")
    print("   ✓ Logged privacy control event")

    # Generate security summary
    security_summary = audit_logger.get_security_summary(days_back=1)
    print(
        f"   ✓ Security summary: {security_summary['total_events']} events, {security_summary['unique_users']} users"
    )

    # Key Management Demo
    print("\n8. Key Management Demo...")

    # Check encryption status
    encryption_status = encryption_manager.get_encryption_status()
    print(f"   ✓ Total encryption keys: {encryption_status['total_keys']}")
    print(f"   ✓ Active keys: {encryption_status['active_keys']}")

    # Rotate a key
    old_key = encryption_manager.get_active_key(EncryptionScope.SESSION_DATA)
    new_key = encryption_manager.rotate_key(EncryptionScope.SESSION_DATA)

    print(f"   ✓ Rotated session data key: {old_key.key_id[:8]}... -> {new_key.key_id[:8]}...")

    # Check for expiring keys
    expiring_keys = encryption_manager.check_key_expiration()
    print(f"   ✓ Keys expiring soon: {len(expiring_keys)}")

    # JWT Token Demo
    print("\n9. JWT Token Demo...")

    # Generate JWT tokens for API access
    admin_token = auth_manager.generate_jwt_token(admin_user, expires_in_hours=1)
    editor_token = auth_manager.generate_jwt_token(editor_user, expires_in_hours=8)

    print("   ✓ Generated admin JWT token")
    print("   ✓ Generated editor JWT token")

    # Verify tokens
    admin_payload = auth_manager.verify_jwt_token(admin_token)
    editor_payload = auth_manager.verify_jwt_token(editor_token)

    print(f"   ✓ Verified admin token: user {admin_payload['username']}")
    print(f"   ✓ Verified editor token: user {editor_payload['username']}")

    # Security Monitoring Demo
    print("\n10. Security Monitoring Demo...")

    # Simulate suspicious activity
    for i in range(6):  # Multiple failed attempts
        auth_manager.authenticate("admin", "wrong_password")

    print("   ✓ Simulated brute force attack (6 failed attempts)")

    # Check recent security events
    from memory_core.security.audit import AuditFilter

    recent_events = audit_logger.query_events(
        AuditFilter(
            start_time=datetime.now(UTC) - timedelta(minutes=5),
            categories=[AuditCategory.SECURITY_INCIDENT],
            limit=5,
        )
    )

    print(f"   ✓ Recent security incidents: {len(recent_events)}")

    # Privacy Statistics
    print("\n11. Privacy and Access Statistics...")

    privacy_stats = access_control.get_privacy_statistics()
    print(f"   ✓ Total resources with privacy controls: {privacy_stats['total_resources']}")
    print(f"   ✓ Active access rules: {privacy_stats['access_rules']}")
    print(f"   ✓ Access decision cache size: {privacy_stats['cached_decisions']}")

    # Role and Permission Statistics
    all_roles = rbac_manager.list_roles()
    all_permissions = rbac_manager.list_permissions()

    print(f"   ✓ Total roles defined: {len(all_roles)}")
    print(f"   ✓ Total permissions defined: {len(all_permissions)}")

    system_roles = [role for role in all_roles if role.is_system_role]
    custom_roles = [role for role in all_roles if not role.is_system_role]

    print(f"   ✓ System roles: {len(system_roles)}")
    print(f"   ✓ Custom roles: {len(custom_roles)}")

    print("\n=== Security Framework Demo Complete ===")
    print("\nKey Security Features Demonstrated:")
    print("✓ Multi-user authentication with secure password hashing")
    print("✓ Role-based access control with hierarchical permissions")
    print("✓ Fine-grained privacy levels for knowledge resources")
    print("✓ Comprehensive audit logging for compliance")
    print("✓ Strong encryption for data at rest and in transit")
    print("✓ JWT tokens for stateless API authentication")
    print("✓ Automatic security threat detection")
    print("✓ Key rotation and encryption key management")
    print("✓ Access control rules and permission management")
    print("✓ Privacy-aware knowledge access controls")

    print("\nSecurity Metrics:")
    print(f"• Users created: {len(auth_manager.list_users())}")
    print(f"• Active sessions: {len(auth_manager.list_sessions())}")
    print(f"• Audit events logged: {len(audit_logger._memory_events)}")
    print(f"• Encryption keys managed: {encryption_status['total_keys']}")
    print(f"• Privacy levels configured: {len(privacy_stats['privacy_levels'])}")


if __name__ == "__main__":
    asyncio.run(main())
