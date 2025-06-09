"""
Tests for the authentication system.
"""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import patch

from memory_core.security.auth import (
    AuthManager, User, UserSession, UserStatus
)


class TestUser:
    """Test the User model."""
    
    def test_user_creation(self):
        """Test creating a user."""
        user = User(
            user_id="test-123",
            username="testuser",
            email="test@example.com",
            password_hash="hashed_password"
        )
        
        assert user.user_id == "test-123"
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.is_active()
        assert len(user.roles) == 0
    
    def test_user_roles(self):
        """Test user role management."""
        user = User(
            user_id="test-123",
            username="testuser",
            email="test@example.com",
            password_hash="hashed_password"
        )
        
        # Test adding roles
        user.add_role("admin")
        user.add_role("editor")
        
        assert user.has_role("admin")
        assert user.has_role("editor")
        assert not user.has_role("viewer")
        assert len(user.roles) == 2
        
        # Test removing role
        user.remove_role("editor")
        assert not user.has_role("editor")
        assert len(user.roles) == 1
    
    def test_user_status(self):
        """Test user status management."""
        user = User(
            user_id="test-123",
            username="testuser",
            email="test@example.com",
            password_hash="hashed_password",
            status=UserStatus.INACTIVE
        )
        
        assert not user.is_active()
        
        user.status = UserStatus.ACTIVE
        assert user.is_active()
    
    def test_user_to_dict(self):
        """Test user serialization."""
        user = User(
            user_id="test-123",
            username="testuser",
            email="test@example.com",
            password_hash="hashed_password"
        )
        user.add_role("admin")
        
        user_dict = user.to_dict()
        
        assert user_dict["user_id"] == "test-123"
        assert user_dict["username"] == "testuser"
        assert user_dict["email"] == "test@example.com"
        assert "admin" in user_dict["roles"]
        assert "password_hash" not in user_dict  # Should not include password


class TestUserSession:
    """Test the UserSession model."""
    
    def test_session_creation(self):
        """Test creating a session."""
        session = UserSession(
            session_id="session-123",
            user_id="user-123",
            username="testuser",
            roles={"admin", "editor"}
        )
        
        assert session.session_id == "session-123"
        assert session.user_id == "user-123"
        assert session.username == "testuser"
        assert session.roles == {"admin", "editor"}
        assert session.is_valid()
        assert not session.is_expired()
    
    def test_session_expiration(self):
        """Test session expiration."""
        # Create expired session
        session = UserSession(
            session_id="session-123",
            user_id="user-123",
            username="testuser",
            roles=set(),
            expires_at=datetime.utcnow() - timedelta(hours=1)
        )
        
        assert not session.is_valid()
        assert session.is_expired()
    
    def test_session_refresh(self):
        """Test session refresh."""
        session = UserSession(
            session_id="session-123",
            user_id="user-123",
            username="testuser",
            roles=set()
        )
        
        original_expires = session.expires_at
        original_activity = session.last_activity
        
        # Wait a small amount and refresh
        import time
        time.sleep(0.01)
        session.refresh()
        
        assert session.expires_at > original_expires
        assert session.last_activity > original_activity
    
    def test_session_to_dict(self):
        """Test session serialization."""
        session = UserSession(
            session_id="session-123",
            user_id="user-123",
            username="testuser",
            roles={"admin"},
            ip_address="192.168.1.1",
            user_agent="Test Agent"
        )
        
        session_dict = session.to_dict()
        
        assert session_dict["session_id"] == "session-123"
        assert session_dict["user_id"] == "user-123"
        assert session_dict["username"] == "testuser"
        assert session_dict["roles"] == ["admin"]
        assert session_dict["ip_address"] == "192.168.1.1"
        assert session_dict["user_agent"] == "Test Agent"


class TestAuthManager:
    """Test the AuthManager class."""
    
    def setup_method(self):
        """Set up for each test method."""
        self.auth_manager = AuthManager(secret_key="test-secret-key")
    
    def test_password_hashing(self):
        """Test password hashing and verification."""
        password = "test_password_123!"
        password_hash = self.auth_manager.hash_password(password)
        
        assert password_hash != password
        assert self.auth_manager.verify_password(password, password_hash)
        assert not self.auth_manager.verify_password("wrong_password", password_hash)
    
    def test_password_validation(self):
        """Test password validation."""
        # Valid password
        valid_password = "ValidPass123!"
        errors = self.auth_manager.validate_password(valid_password)
        assert len(errors) == 0
        
        # Invalid passwords
        short_password = "abc"
        errors = self.auth_manager.validate_password(short_password)
        assert len(errors) > 0
        assert any("8 characters" in error for error in errors)
        
        no_uppercase = "validpass123!"
        errors = self.auth_manager.validate_password(no_uppercase)
        assert any("uppercase" in error for error in errors)
        
        no_digit = "ValidPass!"
        errors = self.auth_manager.validate_password(no_digit)
        assert any("digit" in error for error in errors)
        
        no_special = "ValidPass123"
        errors = self.auth_manager.validate_password(no_special)
        assert any("special character" in error for error in errors)
    
    def test_create_user_success(self):
        """Test successful user creation."""
        user = self.auth_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="ValidPass123!",
            roles={"admin"}
        )
        
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.has_role("admin")
        assert user.is_active()
        
        # Verify user is stored
        retrieved_user = self.auth_manager.get_user(user.user_id)
        assert retrieved_user.username == "testuser"
        
        # Verify username and email mappings
        user_by_username = self.auth_manager.get_user_by_username("testuser")
        assert user_by_username.user_id == user.user_id
        
        user_by_email = self.auth_manager.get_user_by_email("test@example.com")
        assert user_by_email.user_id == user.user_id
    
    def test_create_user_duplicate_username(self):
        """Test creating user with duplicate username."""
        self.auth_manager.create_user(
            username="testuser",
            email="test1@example.com",
            password="ValidPass123!"
        )
        
        with pytest.raises(ValueError, match="Username.*already exists"):
            self.auth_manager.create_user(
                username="testuser",
                email="test2@example.com",
                password="ValidPass123!"
            )
    
    def test_create_user_duplicate_email(self):
        """Test creating user with duplicate email."""
        self.auth_manager.create_user(
            username="testuser1",
            email="test@example.com",
            password="ValidPass123!"
        )
        
        with pytest.raises(ValueError, match="Email.*already exists"):
            self.auth_manager.create_user(
                username="testuser2",
                email="test@example.com",
                password="ValidPass123!"
            )
    
    def test_create_user_invalid_password(self):
        """Test creating user with invalid password."""
        with pytest.raises(ValueError, match="Password validation failed"):
            self.auth_manager.create_user(
                username="testuser",
                email="test@example.com",
                password="weak"
            )
    
    def test_authenticate_success(self):
        """Test successful authentication."""
        # Create user
        user = self.auth_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="ValidPass123!"
        )
        
        # Authenticate with username
        auth_user = self.auth_manager.authenticate("testuser", "ValidPass123!")
        assert auth_user.user_id == user.user_id
        assert auth_user.failed_login_attempts == 0
        assert auth_user.last_login is not None
        
        # Authenticate with email
        auth_user = self.auth_manager.authenticate("test@example.com", "ValidPass123!")
        assert auth_user.user_id == user.user_id
    
    def test_authenticate_wrong_password(self):
        """Test authentication with wrong password."""
        user = self.auth_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="ValidPass123!"
        )
        
        # First failed attempt
        auth_user = self.auth_manager.authenticate("testuser", "wrong_password")
        assert auth_user is None
        
        # Check failed attempts counter
        stored_user = self.auth_manager.get_user(user.user_id)
        assert stored_user.failed_login_attempts == 1
    
    def test_authenticate_nonexistent_user(self):
        """Test authentication with nonexistent user."""
        auth_user = self.auth_manager.authenticate("nonexistent", "password")
        assert auth_user is None
    
    def test_authenticate_locked_account(self):
        """Test authentication with locked account."""
        user = self.auth_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="ValidPass123!"
        )
        
        # Simulate multiple failed attempts
        for _ in range(5):
            self.auth_manager.authenticate("testuser", "wrong_password")
        
        # Account should be locked
        auth_user = self.auth_manager.authenticate("testuser", "ValidPass123!")
        assert auth_user is None
    
    def test_authenticate_inactive_user(self):
        """Test authentication with inactive user."""
        user = self.auth_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="ValidPass123!"
        )
        
        # Deactivate user
        user.status = UserStatus.INACTIVE
        
        # Authentication should fail
        auth_user = self.auth_manager.authenticate("testuser", "ValidPass123!")
        assert auth_user is None
    
    def test_session_management(self):
        """Test session creation and management."""
        user = self.auth_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="ValidPass123!",
            roles={"admin"}
        )
        
        # Create session
        session = self.auth_manager.create_session(
            user,
            ip_address="192.168.1.1",
            user_agent="Test Agent"
        )
        
        assert session.user_id == user.user_id
        assert session.username == user.username
        assert session.roles == user.roles
        assert session.ip_address == "192.168.1.1"
        assert session.user_agent == "Test Agent"
        
        # Retrieve session
        retrieved_session = self.auth_manager.get_session(session.session_id)
        assert retrieved_session.session_id == session.session_id
        
        # Refresh session
        import time
        time.sleep(0.01)  # Small delay to ensure timestamp difference
        refreshed_session = self.auth_manager.refresh_session(session.session_id)
        assert refreshed_session.last_activity >= session.last_activity
        
        # Invalidate session
        result = self.auth_manager.invalidate_session(session.session_id)
        assert result is True
        
        # Session should no longer exist
        invalid_session = self.auth_manager.get_session(session.session_id)
        assert invalid_session is None
    
    def test_session_expiration_cleanup(self):
        """Test cleanup of expired sessions."""
        user = self.auth_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="ValidPass123!"
        )
        
        # Create expired session
        session = self.auth_manager.create_session(user)
        session.expires_at = datetime.utcnow() - timedelta(hours=1)
        
        # Run cleanup first to count expired sessions
        cleaned_count = self.auth_manager.cleanup_expired_sessions()
        assert cleaned_count >= 1
        
        # Try to get expired session after cleanup
        retrieved_session = self.auth_manager.get_session(session.session_id)
        assert retrieved_session is None
    
    def test_user_management_operations(self):
        """Test user management operations."""
        user = self.auth_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="ValidPass123!"
        )
        
        # Update user
        result = self.auth_manager.update_user(
            user.user_id,
            status=UserStatus.SUSPENDED
        )
        assert result is True
        
        updated_user = self.auth_manager.get_user(user.user_id)
        assert updated_user.status == UserStatus.SUSPENDED
        
        # List users
        users = self.auth_manager.list_users()
        assert len(users) == 1
        assert users[0].user_id == user.user_id
        
        # Delete user
        result = self.auth_manager.delete_user(user.user_id)
        assert result is True
        
        deleted_user = self.auth_manager.get_user(user.user_id)
        assert deleted_user is None
    
    def test_jwt_token_operations(self):
        """Test JWT token generation and verification."""
        user = self.auth_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="ValidPass123!",
            roles={"admin", "editor"}
        )
        
        # Generate token
        token = self.auth_manager.generate_jwt_token(user, expires_in_hours=1)
        assert token is not None
        assert isinstance(token, str)
        
        # Verify token
        payload = self.auth_manager.verify_jwt_token(token)
        assert payload is not None
        assert payload["user_id"] == user.user_id
        assert payload["username"] == user.username
        assert set(payload["roles"]) == user.roles
        
        # Test invalid token
        invalid_payload = self.auth_manager.verify_jwt_token("invalid_token")
        assert invalid_payload is None
    
    def test_session_list_and_cleanup(self):
        """Test session listing and cleanup operations."""
        user1 = self.auth_manager.create_user(
            username="user1",
            email="user1@example.com",
            password="ValidPass123!"
        )
        
        user2 = self.auth_manager.create_user(
            username="user2",
            email="user2@example.com",
            password="ValidPass123!"
        )
        
        # Create sessions
        session1 = self.auth_manager.create_session(user1)
        session2 = self.auth_manager.create_session(user2)
        
        # Create expired session
        expired_session = self.auth_manager.create_session(user1)
        expired_session.expires_at = datetime.utcnow() - timedelta(hours=1)
        
        # List active sessions
        active_sessions = self.auth_manager.list_sessions()
        assert len(active_sessions) == 2
        
        # Cleanup expired sessions
        cleaned_count = self.auth_manager.cleanup_expired_sessions()
        assert cleaned_count == 1
        
        # Verify only active sessions remain
        active_sessions = self.auth_manager.list_sessions()
        assert len(active_sessions) == 2