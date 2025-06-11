"""
Role-Based Access Control (RBAC) system for the Memory Engine.

Provides comprehensive role and permission management for controlling
access to knowledge operations and system resources.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, UTC
import uuid

from memory_core.config.config_manager import get_config


logger = logging.getLogger(__name__)


class PermissionType(Enum):
    """Types of permissions in the system."""
    
    # Knowledge operations
    KNOWLEDGE_CREATE = "knowledge:create"
    KNOWLEDGE_READ = "knowledge:read"
    KNOWLEDGE_UPDATE = "knowledge:update"
    KNOWLEDGE_DELETE = "knowledge:delete"
    KNOWLEDGE_SEARCH = "knowledge:search"
    
    # Relationship operations
    RELATIONSHIP_CREATE = "relationship:create"
    RELATIONSHIP_READ = "relationship:read"
    RELATIONSHIP_UPDATE = "relationship:update"
    RELATIONSHIP_DELETE = "relationship:delete"
    
    # System operations
    SYSTEM_ADMIN = "system:admin"
    SYSTEM_CONFIG = "system:config"
    SYSTEM_MONITOR = "system:monitor"
    SYSTEM_BACKUP = "system:backup"
    
    # User management
    USER_CREATE = "user:create"
    USER_READ = "user:read"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"
    USER_MANAGE_ROLES = "user:manage_roles"
    
    # Role management
    ROLE_CREATE = "role:create"
    ROLE_READ = "role:read"
    ROLE_UPDATE = "role:update"
    ROLE_DELETE = "role:delete"
    ROLE_ASSIGN = "role:assign"
    
    # Privacy and security
    PRIVACY_MANAGE = "privacy:manage"
    SECURITY_AUDIT = "security:audit"
    ENCRYPTION_MANAGE = "encryption:manage"


@dataclass
class Permission:
    """Individual permission that can be granted to roles."""
    
    permission_id: str
    name: str
    type: PermissionType
    description: str
    resource_pattern: Optional[str] = None  # For resource-specific permissions
    metadata: Dict[str, any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def matches_resource(self, resource: str) -> bool:
        """
        Check if this permission matches a specific resource.
        
        Args:
            resource: Resource identifier
        
        Returns:
            True if permission applies to the resource
        """
        if not self.resource_pattern:
            return True  # Global permission
        
        # Simple pattern matching (extend for more complex patterns)
        if self.resource_pattern == "*":
            return True
        
        return resource == self.resource_pattern or resource.startswith(self.resource_pattern)
    
    def to_dict(self) -> Dict[str, any]:
        """Convert permission to dictionary representation."""
        return {
            'permission_id': self.permission_id,
            'name': self.name,
            'type': self.type.value,
            'description': self.description,
            'resource_pattern': self.resource_pattern,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class Role:
    """Role that groups permissions and can be assigned to users."""
    
    role_id: str
    name: str
    description: str
    permissions: Set[str] = field(default_factory=set)  # Permission IDs
    parent_roles: Set[str] = field(default_factory=set)  # Inherited roles
    metadata: Dict[str, any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    is_system_role: bool = False  # System roles cannot be deleted
    
    def add_permission(self, permission_id: str) -> None:
        """Add a permission to this role."""
        self.permissions.add(permission_id)
    
    def remove_permission(self, permission_id: str) -> None:
        """Remove a permission from this role."""
        self.permissions.discard(permission_id)
    
    def add_parent_role(self, role_id: str) -> None:
        """Add a parent role for inheritance."""
        self.parent_roles.add(role_id)
    
    def remove_parent_role(self, role_id: str) -> None:
        """Remove a parent role."""
        self.parent_roles.discard(role_id)
    
    def to_dict(self) -> Dict[str, any]:
        """Convert role to dictionary representation."""
        return {
            'role_id': self.role_id,
            'name': self.name,
            'description': self.description,
            'permissions': list(self.permissions),
            'parent_roles': list(self.parent_roles),
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'is_system_role': self.is_system_role
        }


class RBACManager:
    """
    Role-Based Access Control manager for handling roles, permissions,
    and access control decisions.
    """
    
    def __init__(self):
        """Initialize the RBAC manager."""
        self.config = get_config()
        
        # In-memory storage (replace with persistent storage in production)
        self._permissions: Dict[str, Permission] = {}
        self._roles: Dict[str, Role] = {}
        self._role_permissions_cache: Dict[str, Set[str]] = {}
        
        # Initialize system permissions and roles
        self._initialize_system_permissions()
        self._initialize_system_roles()
        
        logger.info("RBACManager initialized")
    
    def _initialize_system_permissions(self) -> None:
        """Initialize default system permissions."""
        system_permissions = [
            # Knowledge permissions
            ("knowledge_create", "Create Knowledge", PermissionType.KNOWLEDGE_CREATE, "Create new knowledge nodes"),
            ("knowledge_read", "Read Knowledge", PermissionType.KNOWLEDGE_READ, "Read knowledge nodes"),
            ("knowledge_update", "Update Knowledge", PermissionType.KNOWLEDGE_UPDATE, "Update existing knowledge nodes"),
            ("knowledge_delete", "Delete Knowledge", PermissionType.KNOWLEDGE_DELETE, "Delete knowledge nodes"),
            ("knowledge_search", "Search Knowledge", PermissionType.KNOWLEDGE_SEARCH, "Search and query knowledge"),
            
            # Relationship permissions
            ("relationship_create", "Create Relationships", PermissionType.RELATIONSHIP_CREATE, "Create relationships between nodes"),
            ("relationship_read", "Read Relationships", PermissionType.RELATIONSHIP_READ, "Read relationship information"),
            ("relationship_update", "Update Relationships", PermissionType.RELATIONSHIP_UPDATE, "Update existing relationships"),
            ("relationship_delete", "Delete Relationships", PermissionType.RELATIONSHIP_DELETE, "Delete relationships"),
            
            # System permissions
            ("system_admin", "System Administration", PermissionType.SYSTEM_ADMIN, "Full system administration access"),
            ("system_config", "System Configuration", PermissionType.SYSTEM_CONFIG, "Manage system configuration"),
            ("system_monitor", "System Monitoring", PermissionType.SYSTEM_MONITOR, "Monitor system performance and health"),
            ("system_backup", "System Backup", PermissionType.SYSTEM_BACKUP, "Perform system backups and restoration"),
            
            # User management permissions
            ("user_create", "Create Users", PermissionType.USER_CREATE, "Create new user accounts"),
            ("user_read", "Read Users", PermissionType.USER_READ, "View user information"),
            ("user_update", "Update Users", PermissionType.USER_UPDATE, "Update user information"),
            ("user_delete", "Delete Users", PermissionType.USER_DELETE, "Delete user accounts"),
            ("user_manage_roles", "Manage User Roles", PermissionType.USER_MANAGE_ROLES, "Assign and remove user roles"),
            
            # Role management permissions
            ("role_create", "Create Roles", PermissionType.ROLE_CREATE, "Create new roles"),
            ("role_read", "Read Roles", PermissionType.ROLE_READ, "View role information"),
            ("role_update", "Update Roles", PermissionType.ROLE_UPDATE, "Update role information"),
            ("role_delete", "Delete Roles", PermissionType.ROLE_DELETE, "Delete roles"),
            ("role_assign", "Assign Roles", PermissionType.ROLE_ASSIGN, "Assign roles to users"),
            
            # Security permissions
            ("privacy_manage", "Manage Privacy", PermissionType.PRIVACY_MANAGE, "Manage privacy levels and access control"),
            ("security_audit", "Security Auditing", PermissionType.SECURITY_AUDIT, "Access security audit logs"),
            ("encryption_manage", "Manage Encryption", PermissionType.ENCRYPTION_MANAGE, "Manage encryption settings"),
        ]
        
        for perm_id, name, perm_type, description in system_permissions:
            permission = Permission(
                permission_id=perm_id,
                name=name,
                type=perm_type,
                description=description
            )
            self._permissions[perm_id] = permission
        
        logger.info(f"Initialized {len(system_permissions)} system permissions")
    
    def _initialize_system_roles(self) -> None:
        """Initialize default system roles."""
        # Super Admin role - all permissions
        super_admin = Role(
            role_id="super_admin",
            name="Super Administrator",
            description="Full system access with all permissions",
            permissions=set(self._permissions.keys()),
            is_system_role=True
        )
        self._roles["super_admin"] = super_admin
        
        # Knowledge Admin role - all knowledge operations
        knowledge_admin = Role(
            role_id="knowledge_admin",
            name="Knowledge Administrator",
            description="Full access to knowledge management operations",
            permissions={
                "knowledge_create", "knowledge_read", "knowledge_update", "knowledge_delete", "knowledge_search",
                "relationship_create", "relationship_read", "relationship_update", "relationship_delete",
                "privacy_manage"
            },
            is_system_role=True
        )
        self._roles["knowledge_admin"] = knowledge_admin
        
        # Knowledge Editor role - create, read, update knowledge
        knowledge_editor = Role(
            role_id="knowledge_editor",
            name="Knowledge Editor",
            description="Create, edit, and manage knowledge content",
            permissions={
                "knowledge_create", "knowledge_read", "knowledge_update", "knowledge_search",
                "relationship_create", "relationship_read", "relationship_update"
            },
            is_system_role=True
        )
        self._roles["knowledge_editor"] = knowledge_editor
        
        # Knowledge Reader role - read-only access
        knowledge_reader = Role(
            role_id="knowledge_reader",
            name="Knowledge Reader",
            description="Read-only access to knowledge and relationships",
            permissions={
                "knowledge_read", "knowledge_search",
                "relationship_read"
            },
            is_system_role=True
        )
        self._roles["knowledge_reader"] = knowledge_reader
        
        # User Manager role - user management operations
        user_manager = Role(
            role_id="user_manager",
            name="User Manager",
            description="Manage user accounts and basic role assignments",
            permissions={
                "user_create", "user_read", "user_update", "user_delete",
                "role_read", "role_assign"
            },
            is_system_role=True
        )
        self._roles["user_manager"] = user_manager
        
        # Monitor role - system monitoring
        monitor = Role(
            role_id="monitor",
            name="System Monitor",
            description="Monitor system performance and health",
            permissions={
                "system_monitor", "security_audit",
                "knowledge_read", "relationship_read"
            },
            is_system_role=True
        )
        self._roles["monitor"] = monitor
        
        logger.info(f"Initialized {len(self._roles)} system roles")
    
    def create_permission(
        self,
        name: str,
        permission_type: PermissionType,
        description: str,
        resource_pattern: Optional[str] = None,
        metadata: Optional[Dict[str, any]] = None
    ) -> Permission:
        """
        Create a new permission.
        
        Args:
            name: Permission name
            permission_type: Type of permission
            description: Permission description
            resource_pattern: Optional resource pattern for resource-specific permissions
            metadata: Additional metadata
        
        Returns:
            Created Permission object
        """
        permission_id = str(uuid.uuid4())
        
        permission = Permission(
            permission_id=permission_id,
            name=name,
            type=permission_type,
            description=description,
            resource_pattern=resource_pattern,
            metadata=metadata or {}
        )
        
        self._permissions[permission_id] = permission
        
        logger.info(f"Created permission: {name} ({permission_id})")
        return permission
    
    def get_permission(self, permission_id: str) -> Optional[Permission]:
        """Get a permission by ID."""
        return self._permissions.get(permission_id)
    
    def list_permissions(self) -> List[Permission]:
        """Get a list of all permissions."""
        return list(self._permissions.values())
    
    def delete_permission(self, permission_id: str) -> bool:
        """
        Delete a permission.
        
        Args:
            permission_id: Permission identifier
        
        Returns:
            True if permission was deleted, False if not found
        """
        permission = self._permissions.pop(permission_id, None)
        if permission:
            # Remove from all roles
            for role in self._roles.values():
                role.remove_permission(permission_id)
            
            # Clear cache
            self._role_permissions_cache.clear()
            
            logger.info(f"Deleted permission: {permission.name} ({permission_id})")
            return True
        return False
    
    def create_role(
        self,
        name: str,
        description: str,
        permissions: Optional[Set[str]] = None,
        parent_roles: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, any]] = None
    ) -> Role:
        """
        Create a new role.
        
        Args:
            name: Role name
            description: Role description
            permissions: Set of permission IDs to assign
            parent_roles: Set of parent role IDs for inheritance
            metadata: Additional metadata
        
        Returns:
            Created Role object
        """
        role_id = str(uuid.uuid4())
        
        role = Role(
            role_id=role_id,
            name=name,
            description=description,
            permissions=permissions or set(),
            parent_roles=parent_roles or set(),
            metadata=metadata or {}
        )
        
        self._roles[role_id] = role
        
        # Clear cache since role hierarchy might have changed
        self._role_permissions_cache.clear()
        
        logger.info(f"Created role: {name} ({role_id})")
        return role
    
    def get_role(self, role_id: str) -> Optional[Role]:
        """Get a role by ID."""
        return self._roles.get(role_id)
    
    def get_role_by_name(self, name: str) -> Optional[Role]:
        """Get a role by name."""
        for role in self._roles.values():
            if role.name == name:
                return role
        return None
    
    def list_roles(self) -> List[Role]:
        """Get a list of all roles."""
        return list(self._roles.values())
    
    def update_role(self, role_id: str, **updates) -> bool:
        """
        Update role information.
        
        Args:
            role_id: Role identifier
            **updates: Fields to update
        
        Returns:
            True if role was updated, False if not found
        """
        role = self._roles.get(role_id)
        if not role:
            return False
        
        for field, value in updates.items():
            if hasattr(role, field):
                setattr(role, field, value)
        
        # Clear cache since role might have changed
        self._role_permissions_cache.clear()
        
        logger.info(f"Updated role: {role.name} ({role_id})")
        return True
    
    def delete_role(self, role_id: str) -> bool:
        """
        Delete a role.
        
        Args:
            role_id: Role identifier
        
        Returns:
            True if role was deleted, False if not found or is system role
        """
        role = self._roles.get(role_id)
        if not role:
            return False
        
        if role.is_system_role:
            logger.warning(f"Cannot delete system role: {role.name} ({role_id})")
            return False
        
        # Remove from all parent role references
        for other_role in self._roles.values():
            other_role.remove_parent_role(role_id)
        
        self._roles.pop(role_id, None)
        
        # Clear cache
        self._role_permissions_cache.clear()
        
        logger.info(f"Deleted role: {role.name} ({role_id})")
        return True
    
    def add_permission_to_role(self, role_id: str, permission_id: str) -> bool:
        """
        Add a permission to a role.
        
        Args:
            role_id: Role identifier
            permission_id: Permission identifier
        
        Returns:
            True if permission was added, False if role or permission not found
        """
        role = self._roles.get(role_id)
        permission = self._permissions.get(permission_id)
        
        if not role or not permission:
            return False
        
        role.add_permission(permission_id)
        
        # Clear cache for this role
        self._role_permissions_cache.pop(role_id, None)
        
        logger.info(f"Added permission {permission.name} to role {role.name}")
        return True
    
    def remove_permission_from_role(self, role_id: str, permission_id: str) -> bool:
        """
        Remove a permission from a role.
        
        Args:
            role_id: Role identifier
            permission_id: Permission identifier
        
        Returns:
            True if permission was removed, False if role not found
        """
        role = self._roles.get(role_id)
        if not role:
            return False
        
        role.remove_permission(permission_id)
        
        # Clear cache for this role
        self._role_permissions_cache.pop(role_id, None)
        
        logger.info(f"Removed permission from role {role.name}")
        return True
    
    def get_role_permissions(self, role_id: str, include_inherited: bool = True) -> Set[str]:
        """
        Get all permissions for a role, optionally including inherited permissions.
        
        Args:
            role_id: Role identifier
            include_inherited: Whether to include permissions from parent roles
        
        Returns:
            Set of permission IDs
        """
        if include_inherited and role_id in self._role_permissions_cache:
            return self._role_permissions_cache[role_id].copy()
        
        role = self._roles.get(role_id)
        if not role:
            return set()
        
        permissions = role.permissions.copy()
        
        if include_inherited:
            # Recursively get permissions from parent roles
            visited = set()
            
            def collect_permissions(current_role_id: str) -> None:
                if current_role_id in visited:
                    return  # Avoid circular dependencies
                
                visited.add(current_role_id)
                current_role = self._roles.get(current_role_id)
                
                if current_role:
                    permissions.update(current_role.permissions)
                    
                    for parent_role_id in current_role.parent_roles:
                        collect_permissions(parent_role_id)
            
            collect_permissions(role_id)
            
            # Cache the result
            self._role_permissions_cache[role_id] = permissions.copy()
        
        return permissions
    
    def check_permission(
        self,
        user_roles: Set[str],
        permission_type: PermissionType,
        resource: Optional[str] = None
    ) -> bool:
        """
        Check if a user has a specific permission.
        
        Args:
            user_roles: Set of role IDs assigned to the user
            permission_type: Type of permission to check
            resource: Optional resource identifier for resource-specific permissions
        
        Returns:
            True if user has the permission, False otherwise
        """
        # Get all permissions for all user roles
        user_permissions = set()
        for role_id in user_roles:
            user_permissions.update(self.get_role_permissions(role_id))
        
        # Check if any permission matches the request
        for permission_id in user_permissions:
            permission = self._permissions.get(permission_id)
            if permission and permission.type == permission_type:
                if resource is None or permission.matches_resource(resource):
                    return True
        
        return False
    
    def get_user_permissions(self, user_roles: Set[str]) -> List[Permission]:
        """
        Get all permissions for a user based on their roles.
        
        Args:
            user_roles: Set of role IDs assigned to the user
        
        Returns:
            List of Permission objects
        """
        user_permission_ids = set()
        for role_id in user_roles:
            user_permission_ids.update(self.get_role_permissions(role_id))
        
        return [
            self._permissions[perm_id]
            for perm_id in user_permission_ids
            if perm_id in self._permissions
        ]
    
    def get_effective_permissions(
        self,
        user_roles: Set[str],
        resource: Optional[str] = None
    ) -> Dict[PermissionType, bool]:
        """
        Get effective permissions for a user on a specific resource.
        
        Args:
            user_roles: Set of role IDs assigned to the user
            resource: Optional resource identifier
        
        Returns:
            Dictionary mapping permission types to boolean values
        """
        result = {}
        
        for permission_type in PermissionType:
            result[permission_type] = self.check_permission(user_roles, permission_type, resource)
        
        return result
    
    def validate_role_hierarchy(self) -> List[str]:
        """
        Validate role hierarchy for circular dependencies.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        def has_circular_dependency(role_id: str, visited: Set[str], path: List[str]) -> bool:
            if role_id in visited:
                cycle_start = path.index(role_id)
                cycle = " -> ".join(path[cycle_start:] + [role_id])
                errors.append(f"Circular dependency detected: {cycle}")
                return True
            
            visited.add(role_id)
            path.append(role_id)
            
            role = self._roles.get(role_id)
            if role:
                for parent_role_id in role.parent_roles:
                    if has_circular_dependency(parent_role_id, visited.copy(), path.copy()):
                        return True
            
            return False
        
        for role_id in self._roles:
            has_circular_dependency(role_id, set(), [])
        
        return errors