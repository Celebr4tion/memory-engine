"""
Authentication framework for the Memory Engine.

Provides user management, session handling, and authentication mechanisms
for secure multi-user access.
"""

import hashlib
import secrets
import time
import uuid
from datetime import datetime, timedelta, UTC
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import bcrypt
import jwt

from memory_core.config.config_manager import get_config


logger = logging.getLogger(__name__)


class UserStatus(Enum):
    """User account status."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"


@dataclass
class User:
    """User model for authentication and authorization."""

    user_id: str
    username: str
    email: str
    password_hash: str
    status: UserStatus = UserStatus.ACTIVE
    roles: Set[str] = field(default_factory=set)
    metadata: Dict[str, any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0

    def is_active(self) -> bool:
        """Check if user account is active."""
        return self.status == UserStatus.ACTIVE

    def has_role(self, role: str) -> bool:
        """Check if user has a specific role."""
        return role in self.roles

    def add_role(self, role: str) -> None:
        """Add a role to the user."""
        self.roles.add(role)

    def remove_role(self, role: str) -> None:
        """Remove a role from the user."""
        self.roles.discard(role)

    def to_dict(self) -> Dict[str, any]:
        """Convert user to dictionary representation."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "status": self.status.value,
            "roles": list(self.roles),
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "failed_login_attempts": self.failed_login_attempts,
        }


@dataclass
class UserSession:
    """User session model for managing authenticated sessions."""

    session_id: str
    user_id: str
    username: str
    roles: Set[str]
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime = field(default_factory=lambda: datetime.now(UTC) + timedelta(hours=24))
    last_activity: datetime = field(default_factory=lambda: datetime.now(UTC))
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    def is_valid(self) -> bool:
        """Check if session is still valid."""
        return datetime.now(UTC) < self.expires_at

    def is_expired(self) -> bool:
        """Check if session has expired."""
        return datetime.now(UTC) >= self.expires_at

    def refresh(self, extend_hours: int = 24) -> None:
        """Refresh session expiration and update last activity."""
        self.last_activity = datetime.now(UTC)
        self.expires_at = datetime.now(UTC) + timedelta(hours=extend_hours)

    def to_dict(self) -> Dict[str, any]:
        """Convert session to dictionary representation."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "username": self.username,
            "roles": list(self.roles),
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
        }


class AuthManager:
    """
    Authentication manager for handling user authentication,
    session management, and security policies.
    """

    def __init__(self, secret_key: Optional[str] = None):
        """
        Initialize the authentication manager.

        Args:
            secret_key: Secret key for JWT tokens and session encryption
        """
        self.config = get_config()
        self.secret_key = secret_key or self.config.get(
            "security.secret_key", secrets.token_urlsafe(32)
        )

        # In-memory storage (replace with persistent storage in production)
        self._users: Dict[str, User] = {}
        self._sessions: Dict[str, UserSession] = {}
        self._username_to_id: Dict[str, str] = {}
        self._email_to_id: Dict[str, str] = {}

        # Security policies
        self.max_failed_attempts = 5
        self.session_timeout_hours = 24
        self.password_min_length = 8

        logger.info("AuthManager initialized")

    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against its hash."""
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))

    def validate_password(self, password: str) -> List[str]:
        """
        Validate password strength.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if len(password) < self.password_min_length:
            errors.append(f"Password must be at least {self.password_min_length} characters long")

        if not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")

        if not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")

        if not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one digit")

        if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            errors.append("Password must contain at least one special character")

        return errors

    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        roles: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, any]] = None,
    ) -> User:
        """
        Create a new user account.

        Args:
            username: Unique username
            email: User's email address
            password: Plain text password
            roles: Set of roles to assign to the user
            metadata: Additional user metadata

        Returns:
            Created User object

        Raises:
            ValueError: If username/email already exists or password is invalid
        """
        # Validate inputs
        if username in self._username_to_id:
            raise ValueError(f"Username '{username}' already exists")

        if email in self._email_to_id:
            raise ValueError(f"Email '{email}' already exists")

        password_errors = self.validate_password(password)
        if password_errors:
            raise ValueError(f"Password validation failed: {'; '.join(password_errors)}")

        # Create user
        user_id = str(uuid.uuid4())
        password_hash = self.hash_password(password)

        user = User(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            roles=roles or set(),
            metadata=metadata or {},
        )

        # Store user
        self._users[user_id] = user
        self._username_to_id[username] = user_id
        self._email_to_id[email] = user_id

        logger.info(f"Created user: {username} ({user_id})")
        return user

    def authenticate(self, username: str, password: str) -> Optional[User]:
        """
        Authenticate a user with username and password.

        Args:
            username: Username or email
            password: Plain text password

        Returns:
            User object if authentication successful, None otherwise
        """
        # Find user by username or email
        user_id = self._username_to_id.get(username) or self._email_to_id.get(username)
        if not user_id:
            logger.warning(f"Authentication failed: User '{username}' not found")
            return None

        user = self._users[user_id]

        # Check if account is locked due to failed attempts
        if user.failed_login_attempts >= self.max_failed_attempts:
            logger.warning(f"Authentication failed: Account '{username}' is locked")
            return None

        # Check if account is active
        if not user.is_active():
            logger.warning(f"Authentication failed: Account '{username}' is not active")
            return None

        # Verify password
        if not self.verify_password(password, user.password_hash):
            user.failed_login_attempts += 1
            logger.warning(
                f"Authentication failed: Invalid password for '{username}' (attempt {user.failed_login_attempts})"
            )
            return None

        # Reset failed attempts and update last login
        user.failed_login_attempts = 0
        user.last_login = datetime.now(UTC)

        logger.info(f"User authenticated: {username}")
        return user

    def create_session(
        self, user: User, ip_address: Optional[str] = None, user_agent: Optional[str] = None
    ) -> UserSession:
        """
        Create a new user session.

        Args:
            user: Authenticated user
            ip_address: Client IP address
            user_agent: Client user agent

        Returns:
            Created UserSession object
        """
        session_id = str(uuid.uuid4())

        session = UserSession(
            session_id=session_id,
            user_id=user.user_id,
            username=user.username,
            roles=user.roles.copy(),
            ip_address=ip_address,
            user_agent=user_agent,
        )

        self._sessions[session_id] = session

        logger.info(f"Created session for user: {user.username} (session: {session_id})")
        return session

    def get_session(self, session_id: str) -> Optional[UserSession]:
        """
        Get a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            UserSession object if found and valid, None otherwise
        """
        session = self._sessions.get(session_id)
        if not session:
            return None

        if session.is_expired():
            self.invalidate_session(session_id)
            return None

        return session

    def invalidate_session(self, session_id: str) -> bool:
        """
        Invalidate a session.

        Args:
            session_id: Session identifier

        Returns:
            True if session was invalidated, False if not found
        """
        session = self._sessions.pop(session_id, None)
        if session:
            logger.info(f"Invalidated session: {session_id} for user: {session.username}")
            return True
        return False

    def refresh_session(self, session_id: str) -> Optional[UserSession]:
        """
        Refresh a session's expiration time.

        Args:
            session_id: Session identifier

        Returns:
            Updated UserSession object if successful, None otherwise
        """
        session = self.get_session(session_id)
        if session:
            session.refresh(self.session_timeout_hours)
            logger.debug(f"Refreshed session: {session_id}")
        return session

    def get_user(self, user_id: str) -> Optional[User]:
        """Get a user by ID."""
        return self._users.get(user_id)

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get a user by username."""
        user_id = self._username_to_id.get(username)
        return self._users.get(user_id) if user_id else None

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get a user by email."""
        user_id = self._email_to_id.get(email)
        return self._users.get(user_id) if user_id else None

    def update_user(self, user_id: str, **updates) -> bool:
        """
        Update user information.

        Args:
            user_id: User identifier
            **updates: Fields to update

        Returns:
            True if user was updated, False if not found
        """
        user = self._users.get(user_id)
        if not user:
            return False

        for field, value in updates.items():
            if hasattr(user, field):
                setattr(user, field, value)

        logger.info(f"Updated user: {user.username} ({user_id})")
        return True

    def delete_user(self, user_id: str) -> bool:
        """
        Delete a user account.

        Args:
            user_id: User identifier

        Returns:
            True if user was deleted, False if not found
        """
        user = self._users.get(user_id)
        if not user:
            return False

        # Remove from all mappings
        self._users.pop(user_id, None)
        self._username_to_id.pop(user.username, None)
        self._email_to_id.pop(user.email, None)

        # Invalidate all sessions for this user
        sessions_to_remove = [
            sid for sid, session in self._sessions.items() if session.user_id == user_id
        ]
        for session_id in sessions_to_remove:
            self.invalidate_session(session_id)

        logger.info(f"Deleted user: {user.username} ({user_id})")
        return True

    def list_users(self) -> List[User]:
        """Get a list of all users."""
        return list(self._users.values())

    def list_sessions(self) -> List[UserSession]:
        """Get a list of all active sessions."""
        return [session for session in self._sessions.values() if session.is_valid()]

    def cleanup_expired_sessions(self) -> int:
        """
        Remove expired sessions.

        Returns:
            Number of sessions cleaned up
        """
        expired_sessions = [sid for sid, session in self._sessions.items() if session.is_expired()]

        for session_id in expired_sessions:
            self.invalidate_session(session_id)

        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

        return len(expired_sessions)

    def generate_jwt_token(self, user: User, expires_in_hours: int = 24) -> str:
        """
        Generate a JWT token for a user.

        Args:
            user: User to generate token for
            expires_in_hours: Token expiration time

        Returns:
            JWT token string
        """
        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "roles": list(user.roles),
            "iat": datetime.now(UTC),
            "exp": datetime.now(UTC) + timedelta(hours=expires_in_hours),
        }

        return jwt.encode(payload, self.secret_key, algorithm="HS256")

    def verify_jwt_token(self, token: str) -> Optional[Dict[str, any]]:
        """
        Verify and decode a JWT token.

        Args:
            token: JWT token string

        Returns:
            Decoded payload if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token has expired")
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")

        return None
