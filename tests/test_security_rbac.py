"""
Tests for the RBAC (Role-Based Access Control) system.
"""

import pytest
from datetime import datetime, timedelta

from memory_core.security.rbac import (
    RBACManager, Role, Permission, PermissionType
)


class TestPermission:
    """Test the Permission model."""
    
    def test_permission_creation(self):
        """Test creating a permission."""
        permission = Permission(
            permission_id="perm-123",
            name="Read Knowledge",
            type=PermissionType.KNOWLEDGE_READ,
            description="Permission to read knowledge nodes"
        )
        
        assert permission.permission_id == "perm-123"
        assert permission.name == "Read Knowledge"
        assert permission.type == PermissionType.KNOWLEDGE_READ
        assert permission.description == "Permission to read knowledge nodes"
        assert permission.resource_pattern is None
    
    def test_permission_resource_matching(self):
        """Test permission resource pattern matching."""
        # Global permission (no pattern)
        global_perm = Permission(
            permission_id="perm-1",
            name="Global Read",
            type=PermissionType.KNOWLEDGE_READ,
            description="Global read permission"
        )
        
        assert global_perm.matches_resource("any_resource")
        assert global_perm.matches_resource("node_123")
        
        # Wildcard permission
        wildcard_perm = Permission(
            permission_id="perm-2",
            name="Wildcard Read",
            type=PermissionType.KNOWLEDGE_READ,
            description="Wildcard read permission",
            resource_pattern="*"
        )
        
        assert wildcard_perm.matches_resource("any_resource")
        assert wildcard_perm.matches_resource("node_123")
        
        # Specific resource permission
        specific_perm = Permission(
            permission_id="perm-3",
            name="Specific Read",
            type=PermissionType.KNOWLEDGE_READ,
            description="Read permission for specific resource",
            resource_pattern="node_123"
        )
        
        assert specific_perm.matches_resource("node_123")
        assert not specific_perm.matches_resource("node_456")
        
        # Prefix matching
        prefix_perm = Permission(
            permission_id="perm-4",
            name="Prefix Read",
            type=PermissionType.KNOWLEDGE_READ,
            description="Read permission for resources with prefix",
            resource_pattern="user_data_"
        )
        
        assert prefix_perm.matches_resource("user_data_123")
        assert prefix_perm.matches_resource("user_data_abc")
        assert not prefix_perm.matches_resource("system_data_123")
    
    def test_permission_to_dict(self):
        """Test permission serialization."""
        permission = Permission(
            permission_id="perm-123",
            name="Read Knowledge",
            type=PermissionType.KNOWLEDGE_READ,
            description="Permission to read knowledge nodes",
            resource_pattern="node_*",
            metadata={"category": "knowledge"}
        )
        
        perm_dict = permission.to_dict()
        
        assert perm_dict["permission_id"] == "perm-123"
        assert perm_dict["name"] == "Read Knowledge"
        assert perm_dict["type"] == PermissionType.KNOWLEDGE_READ.value
        assert perm_dict["resource_pattern"] == "node_*"
        assert perm_dict["metadata"]["category"] == "knowledge"


class TestRole:
    """Test the Role model."""
    
    def test_role_creation(self):
        """Test creating a role."""
        role = Role(
            role_id="role-123",
            name="Knowledge Editor",
            description="Can edit knowledge content",
            permissions={"perm-1", "perm-2"},
            parent_roles={"parent-role"}
        )
        
        assert role.role_id == "role-123"
        assert role.name == "Knowledge Editor"
        assert role.description == "Can edit knowledge content"
        assert role.permissions == {"perm-1", "perm-2"}
        assert role.parent_roles == {"parent-role"}
        assert not role.is_system_role
    
    def test_role_permission_management(self):
        """Test role permission management."""
        role = Role(
            role_id="role-123",
            name="Test Role",
            description="Test role"
        )
        
        # Add permissions
        role.add_permission("perm-1")
        role.add_permission("perm-2")
        
        assert "perm-1" in role.permissions
        assert "perm-2" in role.permissions
        assert len(role.permissions) == 2
        
        # Remove permission
        role.remove_permission("perm-1")
        assert "perm-1" not in role.permissions
        assert len(role.permissions) == 1
    
    def test_role_parent_management(self):
        """Test role parent role management."""
        role = Role(
            role_id="role-123",
            name="Child Role",
            description="Child role"
        )
        
        # Add parent roles
        role.add_parent_role("parent-1")
        role.add_parent_role("parent-2")
        
        assert "parent-1" in role.parent_roles
        assert "parent-2" in role.parent_roles
        assert len(role.parent_roles) == 2
        
        # Remove parent role
        role.remove_parent_role("parent-1")
        assert "parent-1" not in role.parent_roles
        assert len(role.parent_roles) == 1
    
    def test_role_to_dict(self):
        """Test role serialization."""
        role = Role(
            role_id="role-123",
            name="Test Role",
            description="Test role",
            permissions={"perm-1", "perm-2"},
            parent_roles={"parent-role"},
            metadata={"department": "engineering"},
            is_system_role=True
        )
        
        role_dict = role.to_dict()
        
        assert role_dict["role_id"] == "role-123"
        assert role_dict["name"] == "Test Role"
        assert set(role_dict["permissions"]) == {"perm-1", "perm-2"}
        assert set(role_dict["parent_roles"]) == {"parent-role"}
        assert role_dict["metadata"]["department"] == "engineering"
        assert role_dict["is_system_role"] is True


class TestRBACManager:
    """Test the RBACManager class."""
    
    def setup_method(self):
        """Set up for each test method."""
        self.rbac_manager = RBACManager()
    
    def test_system_permissions_initialization(self):
        """Test that system permissions are properly initialized."""
        permissions = self.rbac_manager.list_permissions()
        
        # Should have all system permissions
        permission_types = {perm.type for perm in permissions}
        
        # Check for key permission types
        assert PermissionType.KNOWLEDGE_CREATE in permission_types
        assert PermissionType.KNOWLEDGE_READ in permission_types
        assert PermissionType.USER_CREATE in permission_types
        assert PermissionType.SYSTEM_ADMIN in permission_types
        
        # Should have a reasonable number of permissions
        assert len(permissions) >= 20
    
    def test_system_roles_initialization(self):
        """Test that system roles are properly initialized."""
        roles = self.rbac_manager.list_roles()
        
        # Should have all system roles
        role_names = {role.name for role in roles}
        
        assert "Super Administrator" in role_names
        assert "Knowledge Administrator" in role_names
        assert "Knowledge Editor" in role_names
        assert "Knowledge Reader" in role_names
        assert "User Manager" in role_names
        assert "System Monitor" in role_names
        
        # Check that super admin has all permissions
        super_admin = self.rbac_manager.get_role_by_name("Super Administrator")
        assert super_admin is not None
        assert super_admin.is_system_role
        
        all_permissions = set(perm.permission_id for perm in self.rbac_manager.list_permissions())
        assert super_admin.permissions == all_permissions
    
    def test_create_custom_permission(self):
        """Test creating custom permissions."""
        permission = self.rbac_manager.create_permission(
            name="Custom Read",
            permission_type=PermissionType.KNOWLEDGE_READ,
            description="Custom read permission",
            resource_pattern="custom_*",
            metadata={"custom": True}
        )
        
        assert permission.name == "Custom Read"
        assert permission.type == PermissionType.KNOWLEDGE_READ
        assert permission.resource_pattern == "custom_*"
        assert permission.metadata["custom"] is True
        
        # Verify permission is stored
        retrieved_perm = self.rbac_manager.get_permission(permission.permission_id)
        assert retrieved_perm.name == "Custom Read"
    
    def test_create_custom_role(self):
        """Test creating custom roles."""
        # Get some permissions
        permissions = self.rbac_manager.list_permissions()[:3]
        permission_ids = {perm.permission_id for perm in permissions}
        
        role = self.rbac_manager.create_role(
            name="Custom Role",
            description="Custom test role",
            permissions=permission_ids,
            metadata={"custom": True}
        )
        
        assert role.name == "Custom Role"
        assert role.description == "Custom test role"
        assert role.permissions == permission_ids
        assert role.metadata["custom"] is True
        assert not role.is_system_role
        
        # Verify role is stored
        retrieved_role = self.rbac_manager.get_role(role.role_id)
        assert retrieved_role.name == "Custom Role"
    
    def test_role_permission_management(self):
        """Test adding and removing permissions from roles."""
        # Create a custom role
        role = self.rbac_manager.create_role(
            name="Test Role",
            description="Test role"
        )
        
        # Get a permission
        permissions = self.rbac_manager.list_permissions()
        test_permission = permissions[0]
        
        # Add permission to role
        result = self.rbac_manager.add_permission_to_role(
            role.role_id,
            test_permission.permission_id
        )
        assert result is True
        
        # Verify permission was added
        updated_role = self.rbac_manager.get_role(role.role_id)
        assert test_permission.permission_id in updated_role.permissions
        
        # Remove permission from role
        result = self.rbac_manager.remove_permission_from_role(
            role.role_id,
            test_permission.permission_id
        )
        assert result is True
        
        # Verify permission was removed
        updated_role = self.rbac_manager.get_role(role.role_id)
        assert test_permission.permission_id not in updated_role.permissions
    
    def test_role_hierarchy_permissions(self):
        """Test role hierarchy and permission inheritance."""
        # Create parent role with permissions
        parent_permissions = set(list(perm.permission_id for perm in self.rbac_manager.list_permissions())[:2])
        parent_role = self.rbac_manager.create_role(
            name="Parent Role",
            description="Parent role",
            permissions=parent_permissions
        )
        
        # Create child role with different permissions
        child_permissions = set(list(perm.permission_id for perm in self.rbac_manager.list_permissions())[2:4])
        child_role = self.rbac_manager.create_role(
            name="Child Role",
            description="Child role",
            permissions=child_permissions,
            parent_roles={parent_role.role_id}
        )
        
        # Get effective permissions for child role (should include parent permissions)
        effective_permissions = self.rbac_manager.get_role_permissions(child_role.role_id, include_inherited=True)
        expected_permissions = parent_permissions | child_permissions
        
        assert effective_permissions == expected_permissions
        
        # Get direct permissions only (should not include parent permissions)
        direct_permissions = self.rbac_manager.get_role_permissions(child_role.role_id, include_inherited=False)
        assert direct_permissions == child_permissions
    
    def test_permission_checking(self):
        """Test permission checking for users."""
        # Get a knowledge read permission
        knowledge_read_perms = [
            perm for perm in self.rbac_manager.list_permissions()
            if perm.type == PermissionType.KNOWLEDGE_READ
        ]
        assert len(knowledge_read_perms) > 0
        
        # Create role with knowledge read permission
        role = self.rbac_manager.create_role(
            name="Reader Role",
            description="Can read knowledge",
            permissions={knowledge_read_perms[0].permission_id}
        )
        
        # Test permission checking
        user_roles = {role.role_id}
        
        # Should have knowledge read permission
        has_permission = self.rbac_manager.check_permission(
            user_roles,
            PermissionType.KNOWLEDGE_READ
        )
        assert has_permission is True
        
        # Should not have knowledge create permission
        has_permission = self.rbac_manager.check_permission(
            user_roles,
            PermissionType.KNOWLEDGE_CREATE
        )
        assert has_permission is False
    
    def test_effective_permissions(self):
        """Test getting effective permissions for a user."""
        # Get some permissions
        permissions = self.rbac_manager.list_permissions()[:5]
        permission_ids = {perm.permission_id for perm in permissions}
        
        # Create role with permissions
        role = self.rbac_manager.create_role(
            name="Test Role",
            description="Test role",
            permissions=permission_ids
        )
        
        # Get effective permissions
        user_roles = {role.role_id}
        effective_permissions = self.rbac_manager.get_effective_permissions(user_roles)
        
        # Should have permissions for the role's permission types
        for perm in permissions:
            assert effective_permissions[perm.type] is True
        
        # Should not have permissions for other types
        other_types = set(PermissionType) - {perm.type for perm in permissions}
        for perm_type in other_types:
            assert effective_permissions[perm_type] is False
    
    def test_user_permissions(self):
        """Test getting all permissions for a user."""
        # Get some permissions
        permissions = self.rbac_manager.list_permissions()[:3]
        permission_ids = {perm.permission_id for perm in permissions}
        
        # Create role with permissions
        role = self.rbac_manager.create_role(
            name="Test Role",
            description="Test role",
            permissions=permission_ids
        )
        
        # Get user permissions
        user_roles = {role.role_id}
        user_permissions = self.rbac_manager.get_user_permissions(user_roles)
        
        # Should return the same permissions
        assert len(user_permissions) == len(permissions)
        returned_permission_ids = {perm.permission_id for perm in user_permissions}
        assert returned_permission_ids == permission_ids
    
    def test_role_hierarchy_validation(self):
        """Test role hierarchy validation for circular dependencies."""
        # Create roles
        role_a = self.rbac_manager.create_role(
            name="Role A",
            description="Role A"
        )
        
        role_b = self.rbac_manager.create_role(
            name="Role B",
            description="Role B",
            parent_roles={role_a.role_id}
        )
        
        # Create circular dependency
        role_a.add_parent_role(role_b.role_id)
        
        # Validation should detect circular dependency
        errors = self.rbac_manager.validate_role_hierarchy()
        assert len(errors) > 0
        assert any("circular dependency" in error.lower() for error in errors)
    
    def test_delete_permission(self):
        """Test deleting permissions."""
        # Create custom permission
        permission = self.rbac_manager.create_permission(
            name="Custom Permission",
            permission_type=PermissionType.KNOWLEDGE_READ,
            description="Custom permission"
        )
        
        # Create role with the permission
        role = self.rbac_manager.create_role(
            name="Test Role",
            description="Test role",
            permissions={permission.permission_id}
        )
        
        # Delete permission
        result = self.rbac_manager.delete_permission(permission.permission_id)
        assert result is True
        
        # Permission should be gone
        deleted_perm = self.rbac_manager.get_permission(permission.permission_id)
        assert deleted_perm is None
        
        # Permission should be removed from role
        updated_role = self.rbac_manager.get_role(role.role_id)
        assert permission.permission_id not in updated_role.permissions
    
    def test_delete_role(self):
        """Test deleting roles."""
        # Create custom role
        role = self.rbac_manager.create_role(
            name="Custom Role",
            description="Custom role"
        )
        
        # Delete role
        result = self.rbac_manager.delete_role(role.role_id)
        assert result is True
        
        # Role should be gone
        deleted_role = self.rbac_manager.get_role(role.role_id)
        assert deleted_role is None
    
    def test_delete_system_role_protection(self):
        """Test that system roles cannot be deleted."""
        # Try to delete a system role
        super_admin = self.rbac_manager.get_role_by_name("Super Administrator")
        assert super_admin is not None
        assert super_admin.is_system_role
        
        result = self.rbac_manager.delete_role(super_admin.role_id)
        assert result is False
        
        # Role should still exist
        existing_role = self.rbac_manager.get_role(super_admin.role_id)
        assert existing_role is not None
    
    def test_update_role(self):
        """Test updating role information."""
        # Create custom role
        role = self.rbac_manager.create_role(
            name="Original Name",
            description="Original description"
        )
        
        # Update role
        result = self.rbac_manager.update_role(
            role.role_id,
            name="Updated Name",
            description="Updated description",
            metadata={"updated": True}
        )
        assert result is True
        
        # Verify updates
        updated_role = self.rbac_manager.get_role(role.role_id)
        assert updated_role.name == "Updated Name"
        assert updated_role.description == "Updated description"
        assert updated_role.metadata["updated"] is True
    
    def test_role_by_name_lookup(self):
        """Test looking up roles by name."""
        # Create custom role
        role = self.rbac_manager.create_role(
            name="Unique Role Name",
            description="Unique role"
        )
        
        # Find by name
        found_role = self.rbac_manager.get_role_by_name("Unique Role Name")
        assert found_role is not None
        assert found_role.role_id == role.role_id
        
        # Non-existent role
        not_found = self.rbac_manager.get_role_by_name("Non-existent Role")
        assert not_found is None