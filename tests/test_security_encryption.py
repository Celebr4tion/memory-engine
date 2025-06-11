"""
Tests for the encryption system.
"""

import pytest
import json
from datetime import datetime, timedelta, UTC

from memory_core.security.encryption import (
    EncryptionManager, EncryptionConfig, EncryptionType, EncryptionScope,
    EncryptionKey, EncryptedData
)


class TestEncryptionConfig:
    """Test the EncryptionConfig class."""
    
    def test_default_config(self):
        """Test default encryption configuration."""
        config = EncryptionConfig()
        
        assert config.default_algorithm == EncryptionType.AES_256_GCM
        assert config.key_rotation_days == 90
        assert config.enable_compression is True
        assert config.enable_integrity_check is True
        assert config.backup_keys_count == 3
        assert config.key_derivation_iterations == 100000
        
        # Check scope configurations
        assert EncryptionScope.KNOWLEDGE_CONTENT in config.scope_configs
        assert EncryptionScope.USER_DATA in config.scope_configs
        assert EncryptionScope.API_KEYS in config.scope_configs
        
        # Check specific scope requirements
        user_data_config = config.scope_configs[EncryptionScope.USER_DATA]
        assert user_data_config['required'] is True
        
        knowledge_config = config.scope_configs[EncryptionScope.KNOWLEDGE_CONTENT]
        assert knowledge_config['required'] is False
    
    def test_custom_config(self):
        """Test custom encryption configuration."""
        custom_scope_configs = {
            EncryptionScope.KNOWLEDGE_CONTENT: {
                'algorithm': EncryptionType.FERNET,
                'compress': False,
                'required': True
            }
        }
        
        config = EncryptionConfig(
            default_algorithm=EncryptionType.FERNET,
            key_rotation_days=30,
            enable_compression=False,
            scope_configs=custom_scope_configs
        )
        
        assert config.default_algorithm == EncryptionType.FERNET
        assert config.key_rotation_days == 30
        assert config.enable_compression is False
        
        knowledge_config = config.scope_configs[EncryptionScope.KNOWLEDGE_CONTENT]
        assert knowledge_config['algorithm'] == EncryptionType.FERNET
        assert knowledge_config['required'] is True


class TestEncryptionKey:
    """Test the EncryptionKey class."""
    
    def test_key_creation(self):
        """Test creating an encryption key."""
        key_data = b"test_key_data_32_bytes_long!!"
        expires_at = datetime.now(UTC) + timedelta(days=90)
        
        key = EncryptionKey(
            key_id="key-123",
            algorithm=EncryptionType.AES_256_GCM,
            scope=EncryptionScope.USER_DATA,
            key_data=key_data,
            created_at=datetime.now(UTC),
            expires_at=expires_at,
            metadata={"custom": "value"}
        )
        
        assert key.key_id == "key-123"
        assert key.algorithm == EncryptionType.AES_256_GCM
        assert key.scope == EncryptionScope.USER_DATA
        assert key.key_data == key_data
        assert key.expires_at == expires_at
        assert key.is_active is True
        assert key.version == 1
        assert key.metadata["custom"] == "value"
    
    def test_key_expiration(self):
        """Test key expiration checking."""
        # Non-expired key
        key = EncryptionKey(
            key_id="key-123",
            algorithm=EncryptionType.AES_256_GCM,
            scope=EncryptionScope.USER_DATA,
            key_data=b"test_key",
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(days=1)
        )
        
        assert not key.is_expired()
        
        # Expired key
        expired_key = EncryptionKey(
            key_id="key-456",
            algorithm=EncryptionType.AES_256_GCM,
            scope=EncryptionScope.USER_DATA,
            key_data=b"test_key",
            created_at=datetime.now(UTC) - timedelta(days=2),
            expires_at=datetime.now(UTC) - timedelta(days=1)
        )
        
        assert expired_key.is_expired()
        
        # Key without expiration
        no_expiry_key = EncryptionKey(
            key_id="key-789",
            algorithm=EncryptionType.AES_256_GCM,
            scope=EncryptionScope.USER_DATA,
            key_data=b"test_key",
            created_at=datetime.now(UTC)
        )
        
        assert not no_expiry_key.is_expired()
    
    def test_key_to_dict(self):
        """Test key serialization."""
        key = EncryptionKey(
            key_id="key-123",
            algorithm=EncryptionType.AES_256_GCM,
            scope=EncryptionScope.USER_DATA,
            key_data=b"secret_key_data",
            created_at=datetime.now(UTC),
            metadata={"purpose": "testing"}
        )
        
        key_dict = key.to_dict()
        
        assert key_dict["key_id"] == "key-123"
        assert key_dict["algorithm"] == EncryptionType.AES_256_GCM.value
        assert key_dict["scope"] == EncryptionScope.USER_DATA.value
        assert key_dict["is_active"] is True
        assert key_dict["metadata"]["purpose"] == "testing"
        
        # Should not include sensitive key data
        assert "key_data" not in key_dict


class TestEncryptedData:
    """Test the EncryptedData class."""
    
    def test_encrypted_data_creation(self):
        """Test creating encrypted data."""
        encrypted_data = EncryptedData(
            data=b"encrypted_content",
            algorithm=EncryptionType.AES_256_GCM,
            key_id="key-123",
            iv=b"initialization_v",
            tag=b"auth_tag",
            metadata={"compressed": True}
        )
        
        assert encrypted_data.data == b"encrypted_content"
        assert encrypted_data.algorithm == EncryptionType.AES_256_GCM
        assert encrypted_data.key_id == "key-123"
        assert encrypted_data.iv == b"initialization_v"
        assert encrypted_data.tag == b"auth_tag"
        assert encrypted_data.metadata["compressed"] is True
        assert encrypted_data.encrypted_at is not None
    
    def test_encrypted_data_base64_serialization(self):
        """Test base64 serialization and deserialization."""
        original = EncryptedData(
            data=b"encrypted_content",
            algorithm=EncryptionType.AES_256_GCM,
            key_id="key-123",
            iv=b"initialization_v",
            tag=b"auth_tag",
            metadata={"compressed": True}
        )
        
        # Serialize to base64
        b64_data = original.to_base64()
        assert isinstance(b64_data, str)
        
        # Deserialize from base64
        restored = EncryptedData.from_base64(b64_data)
        
        assert restored.data == original.data
        assert restored.algorithm == original.algorithm
        assert restored.key_id == original.key_id
        assert restored.iv == original.iv
        assert restored.tag == original.tag
        assert restored.metadata == original.metadata


class TestEncryptionManager:
    """Test the EncryptionManager class."""
    
    def setup_method(self):
        """Set up for each test method."""
        self.encryption_manager = EncryptionManager()
    
    def test_manager_initialization(self):
        """Test encryption manager initialization."""
        # Should have default keys for all scopes
        for scope in EncryptionScope:
            active_key = self.encryption_manager.get_active_key(scope)
            assert active_key is not None
            assert active_key.scope == scope
            assert active_key.is_active
    
    def test_key_generation_aes(self):
        """Test AES key generation."""
        key = self.encryption_manager.generate_key(
            EncryptionType.AES_256_GCM,
            EncryptionScope.USER_DATA,
            expires_in_days=30
        )
        
        assert key.algorithm == EncryptionType.AES_256_GCM
        assert key.scope == EncryptionScope.USER_DATA
        assert len(key.key_data) == 32  # 256 bits
        assert key.expires_at is not None
        assert key.is_active
    
    def test_key_generation_fernet(self):
        """Test Fernet key generation."""
        key = self.encryption_manager.generate_key(
            EncryptionType.FERNET,
            EncryptionScope.SESSION_DATA
        )
        
        assert key.algorithm == EncryptionType.FERNET
        assert key.scope == EncryptionScope.SESSION_DATA
        assert len(key.key_data) == 44  # Fernet key length
        assert key.is_active
    
    def test_key_generation_rsa(self):
        """Test RSA key generation."""
        key = self.encryption_manager.generate_key(
            EncryptionType.RSA_2048,
            EncryptionScope.API_KEYS
        )
        
        assert key.algorithm == EncryptionType.RSA_2048
        assert key.scope == EncryptionScope.API_KEYS
        assert key.is_active
        
        # Should have RSA key pair stored
        assert key.key_id in self.encryption_manager._rsa_keys
        private_pem, public_pem = self.encryption_manager._rsa_keys[key.key_id]
        assert b"BEGIN PRIVATE KEY" in private_pem
        assert b"BEGIN PUBLIC KEY" in public_pem
    
    def test_string_encryption_decryption(self):
        """Test string encryption and decryption."""
        test_string = "This is a secret message!"
        
        # Encrypt
        encrypted_b64 = self.encryption_manager.encrypt_string(
            test_string,
            EncryptionScope.USER_DATA
        )
        
        assert isinstance(encrypted_b64, str)
        assert encrypted_b64 != test_string
        
        # Decrypt
        decrypted_string = self.encryption_manager.decrypt_string(encrypted_b64)
        assert decrypted_string == test_string
    
    def test_data_encryption_decryption_aes(self):
        """Test data encryption and decryption with AES."""
        test_data = {"username": "testuser", "email": "test@example.com"}
        
        # Encrypt
        encrypted_data = self.encryption_manager.encrypt(
            test_data,
            EncryptionScope.USER_DATA
        )
        
        assert encrypted_data.algorithm == EncryptionType.AES_256_GCM
        assert encrypted_data.data != json.dumps(test_data).encode('utf-8')
        assert encrypted_data.iv is not None
        assert encrypted_data.tag is not None
        
        # Decrypt
        decrypted_data = self.encryption_manager.decrypt(encrypted_data)
        assert decrypted_data == test_data
    
    def test_data_encryption_decryption_fernet(self):
        """Test data encryption and decryption with Fernet."""
        # Set up Fernet key for a scope
        fernet_key = self.encryption_manager.generate_key(
            EncryptionType.FERNET,
            EncryptionScope.SESSION_DATA
        )
        self.encryption_manager._set_active_key(EncryptionScope.SESSION_DATA, fernet_key.key_id)
        
        test_data = "Session data string"
        
        # Encrypt
        encrypted_data = self.encryption_manager.encrypt(
            test_data,
            EncryptionScope.SESSION_DATA
        )
        
        assert encrypted_data.algorithm == EncryptionType.FERNET
        assert encrypted_data.data != test_data.encode('utf-8')
        
        # Decrypt
        decrypted_data = self.encryption_manager.decrypt(encrypted_data)
        assert decrypted_data == test_data
    
    def test_data_encryption_decryption_rsa_small(self):
        """Test RSA encryption and decryption with small data."""
        # Set up RSA key
        rsa_key = self.encryption_manager.generate_key(
            EncryptionType.RSA_2048,
            EncryptionScope.API_KEYS
        )
        self.encryption_manager._set_active_key(EncryptionScope.API_KEYS, rsa_key.key_id)
        
        test_data = "Small secret data"
        
        # Encrypt
        encrypted_data = self.encryption_manager.encrypt(
            test_data,
            EncryptionScope.API_KEYS
        )
        
        assert encrypted_data.algorithm == EncryptionType.RSA_2048
        assert encrypted_data.data != test_data.encode('utf-8')
        
        # Decrypt
        decrypted_data = self.encryption_manager.decrypt(encrypted_data)
        assert decrypted_data == test_data
    
    def test_data_encryption_decryption_rsa_large(self):
        """Test RSA encryption and decryption with large data (hybrid mode)."""
        # Set up RSA key
        rsa_key = self.encryption_manager.generate_key(
            EncryptionType.RSA_2048,
            EncryptionScope.API_KEYS
        )
        self.encryption_manager._set_active_key(EncryptionScope.API_KEYS, rsa_key.key_id)
        
        # Large data that requires hybrid encryption
        test_data = "A" * 500  # Larger than RSA chunk size
        
        # Encrypt
        encrypted_data = self.encryption_manager.encrypt(
            test_data,
            EncryptionScope.API_KEYS
        )
        
        assert encrypted_data.algorithm == EncryptionType.RSA_2048
        assert encrypted_data.metadata.get('hybrid', False) is True
        assert encrypted_data.data != test_data.encode('utf-8')
        
        # Decrypt
        decrypted_data = self.encryption_manager.decrypt(encrypted_data)
        assert decrypted_data == test_data
    
    def test_compression_option(self):
        """Test data compression during encryption."""
        # Enable compression for knowledge content
        config = EncryptionConfig()
        config.scope_configs[EncryptionScope.KNOWLEDGE_CONTENT]['compress'] = True
        
        manager = EncryptionManager(config)
        
        # Large repetitive data that compresses well
        test_data = "AAAAAAAAAA" * 1000
        
        # Encrypt with compression
        encrypted_data = manager.encrypt(test_data, EncryptionScope.KNOWLEDGE_CONTENT)
        assert encrypted_data.metadata.get('compressed', False) is True
        
        # Decrypt
        decrypted_data = manager.decrypt(encrypted_data)
        assert decrypted_data == test_data
    
    def test_key_rotation(self):
        """Test key rotation."""
        scope = EncryptionScope.USER_DATA
        
        # Get original key
        original_key = self.encryption_manager.get_active_key(scope)
        original_key_id = original_key.key_id
        
        # Rotate key
        new_key = self.encryption_manager.rotate_key(scope)
        
        assert new_key.key_id != original_key_id
        assert new_key.scope == scope
        assert new_key.is_active
        
        # Original key should be inactive
        assert not original_key.is_active
        
        # New key should be active
        current_active = self.encryption_manager.get_active_key(scope)
        assert current_active.key_id == new_key.key_id
    
    def test_key_expiration_checking(self):
        """Test key expiration checking."""
        # Create a key that expires soon
        almost_expired_key = self.encryption_manager.generate_key(
            EncryptionType.AES_256_GCM,
            EncryptionScope.USER_DATA,
            expires_in_days=1  # Expires in 1 day
        )
        
        # Create an expired key
        expired_key = self.encryption_manager.generate_key(
            EncryptionType.AES_256_GCM,
            EncryptionScope.USER_DATA
        )
        expired_key.expires_at = datetime.now(UTC) - timedelta(days=1)
        
        # Check expiring keys
        expiring_keys = self.encryption_manager.check_key_expiration()
        
        # Should include the almost expired key
        expiring_key_ids = [key.key_id for key in expiring_keys]
        assert almost_expired_key.key_id in expiring_key_ids
        assert expired_key.key_id in expiring_key_ids
    
    def test_auto_key_rotation(self):
        """Test automatic key rotation for expired keys."""
        scope = EncryptionScope.USER_DATA
        
        # Get current active key and mark it as expired
        current_key = self.encryption_manager.get_active_key(scope)
        current_key.expires_at = datetime.now(UTC) - timedelta(days=1)
        
        # Run auto rotation
        rotated = self.encryption_manager.auto_rotate_expired_keys()
        
        # Should have rotated the expired key
        assert len(rotated) >= 1
        
        # Find the rotation for our scope
        rotation_found = False
        for rotated_scope, old_key_id, new_key_id in rotated:
            if rotated_scope == scope:
                assert old_key_id == current_key.key_id
                assert new_key_id != old_key_id
                rotation_found = True
                break
        
        assert rotation_found
        
        # New active key should be different
        new_active_key = self.encryption_manager.get_active_key(scope)
        assert new_active_key.key_id != current_key.key_id
    
    def test_data_integrity_verification(self):
        """Test data integrity verification."""
        test_data = "Test data for integrity check"
        
        # Encrypt data
        encrypted_data = self.encryption_manager.encrypt(
            test_data,
            EncryptionScope.USER_DATA
        )
        
        # Verify intact data
        assert self.encryption_manager.verify_data_integrity(encrypted_data) is True
        
        # Corrupt the data
        corrupted_data = EncryptedData(
            data=b"corrupted_data",
            algorithm=encrypted_data.algorithm,
            key_id=encrypted_data.key_id,
            iv=encrypted_data.iv,
            tag=encrypted_data.tag,
            metadata=encrypted_data.metadata
        )
        
        # Verification should fail
        assert self.encryption_manager.verify_data_integrity(corrupted_data) is False
    
    def test_encryption_status(self):
        """Test getting encryption system status."""
        status = self.encryption_manager.get_encryption_status()
        
        assert 'total_keys' in status
        assert 'active_keys' in status
        assert 'keys_by_algorithm' in status
        assert 'keys_by_scope' in status
        assert 'expiring_keys' in status
        assert 'config' in status
        
        # Should have keys for all scopes
        assert status['active_keys'] == len(EncryptionScope)
        assert status['total_keys'] >= len(EncryptionScope)
        
        # Config should be included
        assert status['config']['default_algorithm'] == EncryptionType.AES_256_GCM.value
    
    def test_public_key_export(self):
        """Test exporting public keys for RSA key pairs."""
        # Generate RSA key
        rsa_key = self.encryption_manager.generate_key(
            EncryptionType.RSA_2048,
            EncryptionScope.API_KEYS
        )
        
        # Export public keys
        public_keys = self.encryption_manager.export_public_keys()
        
        assert rsa_key.key_id in public_keys
        assert "BEGIN PUBLIC KEY" in public_keys[rsa_key.key_id]
        assert "END PUBLIC KEY" in public_keys[rsa_key.key_id]
    
    def test_encryption_with_specific_key(self):
        """Test encryption with a specific key ID."""
        # Generate additional key
        additional_key = self.encryption_manager.generate_key(
            EncryptionType.AES_256_GCM,
            EncryptionScope.USER_DATA
        )
        
        test_data = "Data encrypted with specific key"
        
        # Encrypt with specific key
        encrypted_data = self.encryption_manager.encrypt(
            test_data,
            EncryptionScope.USER_DATA,
            key_id=additional_key.key_id
        )
        
        assert encrypted_data.key_id == additional_key.key_id
        
        # Decrypt
        decrypted_data = self.encryption_manager.decrypt(encrypted_data)
        assert decrypted_data == test_data
    
    def test_invalid_key_id_encryption(self):
        """Test encryption with invalid key ID."""
        test_data = "Test data"
        
        with pytest.raises(ValueError, match="Key .* not found"):
            self.encryption_manager.encrypt(
                test_data,
                EncryptionScope.USER_DATA,
                key_id="nonexistent-key"
            )
    
    def test_decryption_with_missing_key(self):
        """Test decryption when key is missing."""
        # Create encrypted data with non-existent key
        encrypted_data = EncryptedData(
            data=b"some_data",
            algorithm=EncryptionType.AES_256_GCM,
            key_id="missing-key"
        )
        
        with pytest.raises(ValueError, match="Key .* not found"):
            self.encryption_manager.decrypt(encrypted_data)
    
    def test_different_data_types(self):
        """Test encryption of different data types."""
        # String
        string_data = "Test string"
        encrypted = self.encryption_manager.encrypt(string_data, EncryptionScope.USER_DATA)
        decrypted = self.encryption_manager.decrypt(encrypted)
        assert decrypted == string_data
        
        # Dictionary
        dict_data = {"key": "value", "number": 42}
        encrypted = self.encryption_manager.encrypt(dict_data, EncryptionScope.USER_DATA)
        decrypted = self.encryption_manager.decrypt(encrypted)
        assert decrypted == dict_data
        
        # List
        list_data = [1, 2, "three", {"four": 4}]
        encrypted = self.encryption_manager.encrypt(list_data, EncryptionScope.USER_DATA)
        decrypted = self.encryption_manager.decrypt(encrypted)
        assert decrypted == list_data
        
        # Bytes (will be returned as bytes since it's not valid UTF-8 JSON)
        bytes_data = b"binary data"
        encrypted = self.encryption_manager.encrypt(bytes_data, EncryptionScope.USER_DATA)
        decrypted = self.encryption_manager.decrypt(encrypted)
        # Bytes data gets decoded to string if it's valid UTF-8
        assert decrypted == bytes_data.decode('utf-8')