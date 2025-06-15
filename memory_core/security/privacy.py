"""
Knowledge privacy levels and access control system.

Provides fine-grained privacy control for knowledge nodes and relationships,
enabling secure multi-user knowledge sharing with configurable access levels.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta, UTC
import uuid

from memory_core.model.knowledge_node import KnowledgeNode
from memory_core.model.relationship import Relationship
from memory_core.security.rbac import RBACManager, PermissionType


logger = logging.getLogger(__name__)


class PrivacyLevel(Enum):
    """Privacy levels for knowledge nodes and relationships."""

    PUBLIC = "public"  # Accessible to all authenticated users
    INTERNAL = "internal"  # Accessible to users within the organization
    CONFIDENTIAL = "confidential"  # Accessible to specific roles/users only
    RESTRICTED = "restricted"  # Accessible to explicitly granted users only
    PRIVATE = "private"  # Accessible only to the owner


@dataclass
class AccessRule:
    """Access control rule for knowledge resources."""

    rule_id: str
    resource_type: str  # "node" or "relationship"
    resource_id: str
    user_id: Optional[str] = None  # Specific user access
    role_id: Optional[str] = None  # Role-based access
    permissions: Set[str] = field(default_factory=set)  # Set of permission types
    conditions: Dict[str, any] = field(default_factory=dict)  # Additional conditions
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: Optional[datetime] = None  # Optional expiration
    created_by: str = ""  # User who created the rule

    def is_valid(self) -> bool:
        """Check if the access rule is still valid."""
        if self.expires_at and datetime.now(UTC) > self.expires_at:
            return False
        return True

    def matches_user(self, user_id: str, user_roles: Set[str]) -> bool:
        """Check if the rule applies to a specific user."""
        if not self.is_valid():
            return False

        # User-specific rule
        if self.user_id and self.user_id == user_id:
            return True

        # Role-based rule
        if self.role_id and self.role_id in user_roles:
            return True

        return False

    def to_dict(self) -> Dict[str, any]:
        """Convert access rule to dictionary representation."""
        return {
            "rule_id": self.rule_id,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "user_id": self.user_id,
            "role_id": self.role_id,
            "permissions": list(self.permissions),
            "conditions": self.conditions,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "created_by": self.created_by,
        }


@dataclass
class KnowledgeAccessMetadata:
    """Access control metadata for knowledge resources."""

    privacy_level: PrivacyLevel
    owner_id: str
    organization_id: Optional[str] = None
    access_groups: Set[str] = field(default_factory=set)
    classification_tags: Set[str] = field(default_factory=set)
    sensitivity_score: float = 0.0  # 0.0 = low, 1.0 = high sensitivity
    retention_policy: Optional[str] = None
    last_accessed: Optional[datetime] = None
    access_count: int = 0

    def to_dict(self) -> Dict[str, any]:
        """Convert metadata to dictionary representation."""
        return {
            "privacy_level": self.privacy_level.value,
            "owner_id": self.owner_id,
            "organization_id": self.organization_id,
            "access_groups": list(self.access_groups),
            "classification_tags": list(self.classification_tags),
            "sensitivity_score": self.sensitivity_score,
            "retention_policy": self.retention_policy,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "access_count": self.access_count,
        }


class KnowledgeAccessControl:
    """
    Knowledge access control system for managing privacy levels
    and fine-grained access permissions.
    """

    def __init__(self, rbac_manager: RBACManager):
        """
        Initialize the knowledge access control system.

        Args:
            rbac_manager: RBAC manager for role-based permissions
        """
        self.rbac_manager = rbac_manager

        # In-memory storage (replace with persistent storage in production)
        self._access_rules: Dict[str, AccessRule] = {}
        self._resource_metadata: Dict[str, KnowledgeAccessMetadata] = {}
        self._access_groups: Dict[str, Set[str]] = {}  # group_id -> user_ids

        # Access control cache
        self._access_cache: Dict[str, Dict[str, bool]] = {}

        logger.info("KnowledgeAccessControl initialized")

    def set_privacy_level(
        self,
        resource_type: str,
        resource_id: str,
        privacy_level: PrivacyLevel,
        owner_id: str,
        organization_id: Optional[str] = None,
    ) -> None:
        """
        Set the privacy level for a knowledge resource.

        Args:
            resource_type: Type of resource ("node" or "relationship")
            resource_id: Resource identifier
            privacy_level: Privacy level to set
            owner_id: Owner user ID
            organization_id: Optional organization identifier
        """
        metadata_key = f"{resource_type}:{resource_id}"

        if metadata_key not in self._resource_metadata:
            self._resource_metadata[metadata_key] = KnowledgeAccessMetadata(
                privacy_level=privacy_level, owner_id=owner_id, organization_id=organization_id
            )
        else:
            self._resource_metadata[metadata_key].privacy_level = privacy_level

        # Clear access cache for this resource
        self._clear_resource_cache(resource_type, resource_id)

        logger.info(f"Set privacy level {privacy_level.value} for {resource_type} {resource_id}")

    def get_privacy_level(self, resource_type: str, resource_id: str) -> Optional[PrivacyLevel]:
        """Get the privacy level for a resource."""
        metadata_key = f"{resource_type}:{resource_id}"
        metadata = self._resource_metadata.get(metadata_key)
        return metadata.privacy_level if metadata else None

    def get_resource_metadata(
        self, resource_type: str, resource_id: str
    ) -> Optional[KnowledgeAccessMetadata]:
        """Get access metadata for a resource."""
        metadata_key = f"{resource_type}:{resource_id}"
        return self._resource_metadata.get(metadata_key)

    def update_resource_metadata(self, resource_type: str, resource_id: str, **updates) -> bool:
        """
        Update access metadata for a resource.

        Args:
            resource_type: Type of resource
            resource_id: Resource identifier
            **updates: Fields to update

        Returns:
            True if metadata was updated, False if not found
        """
        metadata_key = f"{resource_type}:{resource_id}"
        metadata = self._resource_metadata.get(metadata_key)

        if not metadata:
            return False

        for field, value in updates.items():
            if hasattr(metadata, field):
                setattr(metadata, field, value)

        # Clear access cache for this resource
        self._clear_resource_cache(resource_type, resource_id)

        return True

    def create_access_rule(
        self,
        resource_type: str,
        resource_id: str,
        permissions: Set[str],
        user_id: Optional[str] = None,
        role_id: Optional[str] = None,
        conditions: Optional[Dict[str, any]] = None,
        expires_at: Optional[datetime] = None,
        created_by: str = "",
    ) -> AccessRule:
        """
        Create an access rule for a specific resource.

        Args:
            resource_type: Type of resource ("node" or "relationship")
            resource_id: Resource identifier
            permissions: Set of permission types to grant
            user_id: Specific user to grant access to
            role_id: Role to grant access to
            conditions: Additional access conditions
            expires_at: Optional expiration time
            created_by: User who created the rule

        Returns:
            Created AccessRule object
        """
        if not user_id and not role_id:
            raise ValueError("Either user_id or role_id must be specified")

        rule_id = str(uuid.uuid4())

        access_rule = AccessRule(
            rule_id=rule_id,
            resource_type=resource_type,
            resource_id=resource_id,
            user_id=user_id,
            role_id=role_id,
            permissions=permissions,
            conditions=conditions or {},
            expires_at=expires_at,
            created_by=created_by,
        )

        self._access_rules[rule_id] = access_rule

        # Clear access cache for this resource
        self._clear_resource_cache(resource_type, resource_id)

        logger.info(f"Created access rule {rule_id} for {resource_type} {resource_id}")
        return access_rule

    def delete_access_rule(self, rule_id: str) -> bool:
        """
        Delete an access rule.

        Args:
            rule_id: Access rule identifier

        Returns:
            True if rule was deleted, False if not found
        """
        rule = self._access_rules.pop(rule_id, None)
        if rule:
            # Clear access cache for the affected resource
            self._clear_resource_cache(rule.resource_type, rule.resource_id)
            logger.info(f"Deleted access rule {rule_id}")
            return True
        return False

    def get_access_rules(
        self,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[AccessRule]:
        """
        Get access rules with optional filtering.

        Args:
            resource_type: Filter by resource type
            resource_id: Filter by resource ID
            user_id: Filter by user ID

        Returns:
            List of matching access rules
        """
        rules = []

        for rule in self._access_rules.values():
            if resource_type and rule.resource_type != resource_type:
                continue
            if resource_id and rule.resource_id != resource_id:
                continue
            if user_id and rule.user_id != user_id:
                continue

            rules.append(rule)

        return rules

    def create_access_group(self, group_name: str, user_ids: Set[str]) -> str:
        """
        Create an access group for simplified user management.

        Args:
            group_name: Name of the access group
            user_ids: Set of user IDs to include in the group

        Returns:
            Group identifier
        """
        group_id = str(uuid.uuid4())
        self._access_groups[group_id] = user_ids.copy()

        logger.info(f"Created access group '{group_name}' with {len(user_ids)} users")
        return group_id

    def add_user_to_group(self, group_id: str, user_id: str) -> bool:
        """Add a user to an access group."""
        if group_id in self._access_groups:
            self._access_groups[group_id].add(user_id)
            return True
        return False

    def remove_user_from_group(self, group_id: str, user_id: str) -> bool:
        """Remove a user from an access group."""
        if group_id in self._access_groups:
            self._access_groups[group_id].discard(user_id)
            return True
        return False

    def check_access(
        self,
        user_id: str,
        user_roles: Set[str],
        resource_type: str,
        resource_id: str,
        permission_type: PermissionType,
        organization_id: Optional[str] = None,
    ) -> bool:
        """
        Check if a user has access to a specific resource.

        Args:
            user_id: User identifier
            user_roles: Set of user roles
            resource_type: Type of resource ("node" or "relationship")
            resource_id: Resource identifier
            permission_type: Type of permission requested
            organization_id: Optional organization context

        Returns:
            True if access is granted, False otherwise
        """
        # Check cache first
        cache_key = f"{user_id}:{resource_type}:{resource_id}:{permission_type.value}"
        if cache_key in self._access_cache.get(user_id, {}):
            return self._access_cache[user_id][cache_key]

        # Get resource metadata
        metadata = self.get_resource_metadata(resource_type, resource_id)
        if not metadata:
            # No metadata means default privacy level (internal)
            metadata = KnowledgeAccessMetadata(
                privacy_level=PrivacyLevel.INTERNAL, owner_id="system"
            )

        # Update access tracking
        metadata.last_accessed = datetime.now(UTC)
        metadata.access_count += 1

        # Check ownership
        if metadata.owner_id == user_id:
            result = True
        else:
            result = self._evaluate_access(
                user_id,
                user_roles,
                metadata,
                resource_type,
                resource_id,
                permission_type,
                organization_id,
            )

        # Cache the result
        if user_id not in self._access_cache:
            self._access_cache[user_id] = {}
        self._access_cache[user_id][cache_key] = result

        return result

    def _evaluate_access(
        self,
        user_id: str,
        user_roles: Set[str],
        metadata: KnowledgeAccessMetadata,
        resource_type: str,
        resource_id: str,
        permission_type: PermissionType,
        organization_id: Optional[str],
    ) -> bool:
        """Evaluate access based on privacy level and access rules."""

        # Check privacy level restrictions
        if metadata.privacy_level == PrivacyLevel.PRIVATE:
            # Private resources are only accessible to owner
            return False

        if metadata.privacy_level == PrivacyLevel.RESTRICTED:
            # Restricted resources require explicit access rules
            return self._check_explicit_access(
                user_id, user_roles, resource_type, resource_id, permission_type
            )

        if metadata.privacy_level == PrivacyLevel.CONFIDENTIAL:
            # Confidential resources require appropriate role or explicit access
            has_role_access = self.rbac_manager.check_permission(
                user_roles, permission_type, resource_id
            )
            has_explicit_access = self._check_explicit_access(
                user_id, user_roles, resource_type, resource_id, permission_type
            )
            return has_role_access or has_explicit_access

        if metadata.privacy_level == PrivacyLevel.INTERNAL:
            # Internal resources require organization membership or role-based access
            if (
                organization_id
                and metadata.organization_id
                and organization_id == metadata.organization_id
            ):
                return True
            return self.rbac_manager.check_permission(user_roles, permission_type, resource_id)

        if metadata.privacy_level == PrivacyLevel.PUBLIC:
            # Public resources are accessible to all authenticated users (with basic permission check)
            return self.rbac_manager.check_permission(user_roles, permission_type, resource_id)

        return False

    def _check_explicit_access(
        self,
        user_id: str,
        user_roles: Set[str],
        resource_type: str,
        resource_id: str,
        permission_type: PermissionType,
    ) -> bool:
        """Check for explicit access rules."""

        # Get rules for this resource
        resource_rules = [
            rule
            for rule in self._access_rules.values()
            if rule.resource_type == resource_type and rule.resource_id == resource_id
        ]

        for rule in resource_rules:
            if not rule.is_valid():
                continue

            if rule.matches_user(user_id, user_roles):
                if permission_type.value in rule.permissions or "*" in rule.permissions:
                    # Check additional conditions if any
                    if self._evaluate_conditions(rule.conditions, user_id, user_roles):
                        return True

        return False

    def _evaluate_conditions(
        self, conditions: Dict[str, any], user_id: str, user_roles: Set[str]
    ) -> bool:
        """Evaluate additional access conditions."""
        if not conditions:
            return True

        # Time-based conditions
        if "time_range" in conditions:
            time_range = conditions["time_range"]
            current_time = datetime.now(UTC).time()
            start_time = datetime.strptime(time_range["start"], "%H:%M").time()
            end_time = datetime.strptime(time_range["end"], "%H:%M").time()

            if not (start_time <= current_time <= end_time):
                return False

        # IP-based conditions (would need request context)
        if "allowed_ips" in conditions:
            # This would require request context to get user's IP
            pass

        # Custom conditions can be added here

        return True

    def _clear_resource_cache(self, resource_type: str, resource_id: str) -> None:
        """Clear access cache for a specific resource."""
        for user_cache in self._access_cache.values():
            keys_to_remove = [
                key
                for key in user_cache.keys()
                if key.startswith(f"{resource_type}:{resource_id}:")
            ]
            for key in keys_to_remove:
                user_cache.pop(key, None)

    def clear_user_cache(self, user_id: str) -> None:
        """Clear access cache for a specific user."""
        self._access_cache.pop(user_id, None)

    def get_accessible_resources(
        self,
        user_id: str,
        user_roles: Set[str],
        resource_type: str,
        permission_type: PermissionType,
        organization_id: Optional[str] = None,
    ) -> List[str]:
        """
        Get a list of resource IDs that the user can access.

        Args:
            user_id: User identifier
            user_roles: Set of user roles
            resource_type: Type of resource
            permission_type: Type of permission
            organization_id: Optional organization context

        Returns:
            List of accessible resource IDs
        """
        accessible = []

        for metadata_key, metadata in self._resource_metadata.items():
            if not metadata_key.startswith(f"{resource_type}:"):
                continue

            resource_id = metadata_key.split(":", 1)[1]

            if self.check_access(
                user_id, user_roles, resource_type, resource_id, permission_type, organization_id
            ):
                accessible.append(resource_id)

        return accessible

    def audit_access_attempts(self, user_id: str, days_back: int = 30) -> List[Dict[str, any]]:
        """
        Get audit trail of access attempts for a user.

        Args:
            user_id: User identifier
            days_back: Number of days to look back

        Returns:
            List of access attempt records
        """
        cutoff_date = datetime.now(UTC) - timedelta(days=days_back)
        audit_records = []

        # This would typically query an audit log
        # For now, we'll return cached access information
        user_cache = self._access_cache.get(user_id, {})

        for cache_key, granted in user_cache.items():
            parts = cache_key.split(":")
            if len(parts) >= 4:
                resource_type, resource_id, permission_type = parts[1], parts[2], parts[3]

                audit_records.append(
                    {
                        "user_id": user_id,
                        "resource_type": resource_type,
                        "resource_id": resource_id,
                        "permission_type": permission_type,
                        "access_granted": granted,
                        "timestamp": datetime.now(UTC).isoformat(),  # Would be actual timestamp
                    }
                )

        return audit_records

    def get_privacy_statistics(self) -> Dict[str, any]:
        """Get privacy and access control statistics."""
        stats = {
            "total_resources": len(self._resource_metadata),
            "privacy_levels": {},
            "access_rules": len(self._access_rules),
            "access_groups": len(self._access_groups),
            "cached_decisions": sum(len(cache) for cache in self._access_cache.values()),
        }

        # Count resources by privacy level
        for metadata in self._resource_metadata.values():
            level = metadata.privacy_level.value
            stats["privacy_levels"][level] = stats["privacy_levels"].get(level, 0) + 1

        return stats
