"""
Data encryption system for the Memory Engine.

Provides encryption at rest and in transit for sensitive knowledge data,
user information, and system configurations.
"""

import logging
import base64
import os
import secrets
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import json

# Encryption libraries
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

from memory_core.config.config_manager import get_config


logger = logging.getLogger(__name__)


class EncryptionType(Enum):
    """Types of encryption algorithms."""
    
    AES_256_GCM = "aes_256_gcm"
    FERNET = "fernet"
    RSA_2048 = "rsa_2048"
    RSA_4096 = "rsa_4096"


class EncryptionScope(Enum):
    """Scope of data being encrypted."""
    
    KNOWLEDGE_CONTENT = "knowledge_content"
    USER_DATA = "user_data"
    SYSTEM_CONFIG = "system_config"
    AUDIT_LOGS = "audit_logs"
    SESSION_DATA = "session_data"
    API_KEYS = "api_keys"


@dataclass
class EncryptionConfig:
    """Configuration for encryption settings."""
    
    default_algorithm: EncryptionType = EncryptionType.AES_256_GCM
    key_rotation_days: int = 90
    enable_compression: bool = True
    enable_integrity_check: bool = True
    backup_keys_count: int = 3
    key_derivation_iterations: int = 100000
    
    # Scope-specific configurations
    scope_configs: Dict[EncryptionScope, Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.scope_configs is None:
            self.scope_configs = {
                EncryptionScope.KNOWLEDGE_CONTENT: {
                    'algorithm': EncryptionType.AES_256_GCM,
                    'compress': True,
                    'required': False
                },
                EncryptionScope.USER_DATA: {
                    'algorithm': EncryptionType.AES_256_GCM,
                    'compress': False,
                    'required': True
                },
                EncryptionScope.SYSTEM_CONFIG: {
                    'algorithm': EncryptionType.AES_256_GCM,
                    'compress': False,
                    'required': True
                },
                EncryptionScope.AUDIT_LOGS: {
                    'algorithm': EncryptionType.AES_256_GCM,
                    'compress': True,
                    'required': True
                },
                EncryptionScope.SESSION_DATA: {
                    'algorithm': EncryptionType.FERNET,
                    'compress': False,
                    'required': True
                },
                EncryptionScope.API_KEYS: {
                    'algorithm': EncryptionType.AES_256_GCM,
                    'compress': False,
                    'required': True
                }
            }


@dataclass
class EncryptionKey:
    """Encryption key metadata and data."""
    
    key_id: str
    algorithm: EncryptionType
    scope: EncryptionScope
    key_data: bytes
    created_at: datetime
    expires_at: Optional[datetime] = None
    is_active: bool = True
    version: int = 1
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def is_expired(self) -> bool:
        """Check if the key has expired."""
        return self.expires_at and datetime.utcnow() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without sensitive key data)."""
        return {
            'key_id': self.key_id,
            'algorithm': self.algorithm.value,
            'scope': self.scope.value,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'is_active': self.is_active,
            'version': self.version,
            'metadata': self.metadata
        }


@dataclass
class EncryptedData:
    """Encrypted data container with metadata."""
    
    data: bytes
    algorithm: EncryptionType
    key_id: str
    iv: Optional[bytes] = None  # Initialization vector for AES
    tag: Optional[bytes] = None  # Authentication tag for GCM
    metadata: Dict[str, Any] = None
    encrypted_at: datetime = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.encrypted_at is None:
            self.encrypted_at = datetime.utcnow()
    
    def to_base64(self) -> str:
        """Convert encrypted data to base64 string for storage."""
        data_dict = {
            'data': base64.b64encode(self.data).decode('utf-8'),
            'algorithm': self.algorithm.value,
            'key_id': self.key_id,
            'encrypted_at': self.encrypted_at.isoformat(),
            'metadata': self.metadata
        }
        
        if self.iv:
            data_dict['iv'] = base64.b64encode(self.iv).decode('utf-8')
        
        if self.tag:
            data_dict['tag'] = base64.b64encode(self.tag).decode('utf-8')
        
        return base64.b64encode(json.dumps(data_dict).encode('utf-8')).decode('utf-8')
    
    @classmethod
    def from_base64(cls, b64_data: str) -> 'EncryptedData':
        """Create EncryptedData from base64 string."""
        data_dict = json.loads(base64.b64decode(b64_data.encode('utf-8')).decode('utf-8'))
        
        return cls(
            data=base64.b64decode(data_dict['data'].encode('utf-8')),
            algorithm=EncryptionType(data_dict['algorithm']),
            key_id=data_dict['key_id'],
            iv=base64.b64decode(data_dict['iv'].encode('utf-8')) if 'iv' in data_dict else None,
            tag=base64.b64decode(data_dict['tag'].encode('utf-8')) if 'tag' in data_dict else None,
            metadata=data_dict.get('metadata', {}),
            encrypted_at=datetime.fromisoformat(data_dict['encrypted_at'])
        )


class EncryptionManager:
    """
    Comprehensive encryption manager for data at rest and in transit.
    """
    
    def __init__(self, config: Optional[EncryptionConfig] = None):
        """
        Initialize the encryption manager.
        
        Args:
            config: Encryption configuration
        """
        self.config = config or EncryptionConfig()
        self.app_config = get_config()
        
        # Key storage (in production, use secure key management service)
        self._keys: Dict[str, EncryptionKey] = {}
        self._active_keys: Dict[EncryptionScope, str] = {}  # scope -> key_id
        
        # Encryption backends
        self._fernet_keys: Dict[str, Fernet] = {}
        self._aes_keys: Dict[str, bytes] = {}
        self._rsa_keys: Dict[str, Tuple[bytes, bytes]] = {}  # (private, public)
        
        # Initialize default keys
        self._initialize_default_keys()
        
        logger.info("EncryptionManager initialized")
    
    def _initialize_default_keys(self) -> None:
        """Initialize default encryption keys for each scope."""
        for scope in EncryptionScope:
            scope_config = self.config.scope_configs[scope]
            algorithm = scope_config['algorithm']
            
            key = self.generate_key(algorithm, scope)
            self._set_active_key(scope, key.key_id)
            
            logger.info(f"Generated default {algorithm.value} key for {scope.value}")
    
    def generate_key(
        self,
        algorithm: EncryptionType,
        scope: EncryptionScope,
        expires_in_days: Optional[int] = None
    ) -> EncryptionKey:
        """
        Generate a new encryption key.
        
        Args:
            algorithm: Encryption algorithm
            scope: Scope of the key
            expires_in_days: Optional expiration in days
        
        Returns:
            Generated EncryptionKey
        """
        key_id = secrets.token_hex(16)
        
        if algorithm == EncryptionType.FERNET:
            key_data = Fernet.generate_key()
            self._fernet_keys[key_id] = Fernet(key_data)
            
        elif algorithm in [EncryptionType.AES_256_GCM]:
            key_data = secrets.token_bytes(32)  # 256 bits
            self._aes_keys[key_id] = key_data
            
        elif algorithm in [EncryptionType.RSA_2048, EncryptionType.RSA_4096]:
            key_size = 2048 if algorithm == EncryptionType.RSA_2048 else 4096
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size,
                backend=default_backend()
            )
            
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            public_pem = private_key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            self._rsa_keys[key_id] = (private_pem, public_pem)
            key_data = private_pem
            
        else:
            raise ValueError(f"Unsupported encryption algorithm: {algorithm}")
        
        # Set expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        elif self.config.key_rotation_days > 0:
            expires_at = datetime.utcnow() + timedelta(days=self.config.key_rotation_days)
        
        # Create key metadata
        encryption_key = EncryptionKey(
            key_id=key_id,
            algorithm=algorithm,
            scope=scope,
            key_data=key_data,
            created_at=datetime.utcnow(),
            expires_at=expires_at
        )
        
        self._keys[key_id] = encryption_key
        
        logger.info(f"Generated {algorithm.value} key {key_id} for scope {scope.value}")
        return encryption_key
    
    def _set_active_key(self, scope: EncryptionScope, key_id: str) -> None:
        """Set the active key for a scope."""
        if key_id not in self._keys:
            raise ValueError(f"Key {key_id} not found")
        
        self._active_keys[scope] = key_id
        logger.info(f"Set active key for {scope.value}: {key_id}")
    
    def get_active_key(self, scope: EncryptionScope) -> Optional[EncryptionKey]:
        """Get the active key for a scope."""
        key_id = self._active_keys.get(scope)
        return self._keys.get(key_id) if key_id else None
    
    def encrypt(
        self,
        data: Union[str, bytes, Dict, List],
        scope: EncryptionScope,
        key_id: Optional[str] = None
    ) -> EncryptedData:
        """
        Encrypt data using the appropriate algorithm and key.
        
        Args:
            data: Data to encrypt
            scope: Encryption scope
            key_id: Optional specific key ID (uses active key if None)
        
        Returns:
            EncryptedData object
        """
        # Get the key
        if key_id:
            key = self._keys.get(key_id)
            if not key:
                raise ValueError(f"Key {key_id} not found")
        else:
            key = self.get_active_key(scope)
            if not key:
                raise ValueError(f"No active key for scope {scope.value}")
        
        # Convert data to bytes
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif isinstance(data, (dict, list)):
            data_bytes = json.dumps(data, sort_keys=True).encode('utf-8')
        elif isinstance(data, bytes):
            data_bytes = data
        else:
            data_bytes = str(data).encode('utf-8')
        
        # Compress if enabled
        scope_config = self.config.scope_configs[scope]
        if scope_config.get('compress', False) and self.config.enable_compression:
            import gzip
            data_bytes = gzip.compress(data_bytes)
        
        # Encrypt based on algorithm
        if key.algorithm == EncryptionType.FERNET:
            fernet = self._fernet_keys[key.key_id]
            encrypted_data = fernet.encrypt(data_bytes)
            
            return EncryptedData(
                data=encrypted_data,
                algorithm=key.algorithm,
                key_id=key.key_id,
                metadata={'compressed': scope_config.get('compress', False)}
            )
            
        elif key.algorithm == EncryptionType.AES_256_GCM:
            iv = secrets.token_bytes(12)  # 96-bit IV for GCM
            aes_key = self._aes_keys[key.key_id]
            
            cipher = Cipher(
                algorithms.AES(aes_key),
                modes.GCM(iv),
                backend=default_backend()
            )
            
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(data_bytes) + encryptor.finalize()
            
            return EncryptedData(
                data=ciphertext,
                algorithm=key.algorithm,
                key_id=key.key_id,
                iv=iv,
                tag=encryptor.tag,
                metadata={'compressed': scope_config.get('compress', False)}
            )
            
        elif key.algorithm in [EncryptionType.RSA_2048, EncryptionType.RSA_4096]:
            private_pem, public_pem = self._rsa_keys[key.key_id]
            
            public_key = serialization.load_pem_public_key(public_pem, backend=default_backend())
            
            # RSA has size limitations, so we might need to chunk data
            max_chunk_size = (key.algorithm == EncryptionType.RSA_2048 and 190) or 446  # Conservative sizes
            
            if len(data_bytes) <= max_chunk_size:
                encrypted_data = public_key.encrypt(
                    data_bytes,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
            else:
                # For larger data, use hybrid encryption (RSA + AES)
                aes_key = secrets.token_bytes(32)
                iv = secrets.token_bytes(12)
                
                # Encrypt data with AES
                cipher = Cipher(
                    algorithms.AES(aes_key),
                    modes.GCM(iv),
                    backend=default_backend()
                )
                encryptor = cipher.encryptor()
                ciphertext = encryptor.update(data_bytes) + encryptor.finalize()
                
                # Encrypt AES key with RSA
                encrypted_aes_key = public_key.encrypt(
                    aes_key,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                
                # Combine encrypted AES key, IV, tag, and ciphertext
                encrypted_data = encrypted_aes_key + iv + encryptor.tag + ciphertext
            
            return EncryptedData(
                data=encrypted_data,
                algorithm=key.algorithm,
                key_id=key.key_id,
                metadata={
                    'compressed': scope_config.get('compress', False),
                    'hybrid': len(data_bytes) > max_chunk_size
                }
            )
            
        else:
            raise ValueError(f"Unsupported encryption algorithm: {key.algorithm}")
    
    def decrypt(self, encrypted_data: EncryptedData) -> Union[str, bytes, Dict, List]:
        """
        Decrypt data.
        
        Args:
            encrypted_data: EncryptedData object to decrypt
        
        Returns:
            Decrypted data
        """
        key = self._keys.get(encrypted_data.key_id)
        if not key:
            raise ValueError(f"Key {encrypted_data.key_id} not found")
        
        # Decrypt based on algorithm
        if encrypted_data.algorithm == EncryptionType.FERNET:
            fernet = self._fernet_keys[key.key_id]
            decrypted_bytes = fernet.decrypt(encrypted_data.data)
            
        elif encrypted_data.algorithm == EncryptionType.AES_256_GCM:
            if not encrypted_data.iv or not encrypted_data.tag:
                raise ValueError("IV and tag required for AES-GCM decryption")
            
            aes_key = self._aes_keys[key.key_id]
            
            cipher = Cipher(
                algorithms.AES(aes_key),
                modes.GCM(encrypted_data.iv, encrypted_data.tag),
                backend=default_backend()
            )
            
            decryptor = cipher.decryptor()
            decrypted_bytes = decryptor.update(encrypted_data.data) + decryptor.finalize()
            
        elif encrypted_data.algorithm in [EncryptionType.RSA_2048, EncryptionType.RSA_4096]:
            private_pem, _ = self._rsa_keys[key.key_id]
            private_key = serialization.load_pem_private_key(
                private_pem, password=None, backend=default_backend()
            )
            
            if encrypted_data.metadata.get('hybrid', False):
                # Hybrid decryption
                key_size = 256 if encrypted_data.algorithm == EncryptionType.RSA_2048 else 512
                
                encrypted_aes_key = encrypted_data.data[:key_size]
                iv = encrypted_data.data[key_size:key_size + 12]
                tag = encrypted_data.data[key_size + 12:key_size + 12 + 16]
                ciphertext = encrypted_data.data[key_size + 12 + 16:]
                
                # Decrypt AES key
                aes_key = private_key.decrypt(
                    encrypted_aes_key,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                
                # Decrypt data with AES
                cipher = Cipher(
                    algorithms.AES(aes_key),
                    modes.GCM(iv, tag),
                    backend=default_backend()
                )
                decryptor = cipher.decryptor()
                decrypted_bytes = decryptor.update(ciphertext) + decryptor.finalize()
            else:
                # Direct RSA decryption
                decrypted_bytes = private_key.decrypt(
                    encrypted_data.data,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
            
        else:
            raise ValueError(f"Unsupported encryption algorithm: {encrypted_data.algorithm}")
        
        # Decompress if needed
        if encrypted_data.metadata.get('compressed', False):
            import gzip
            decrypted_bytes = gzip.decompress(decrypted_bytes)
        
        # Try to decode as string or JSON
        try:
            text = decrypted_bytes.decode('utf-8')
            
            # Try to parse as JSON
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
                
        except UnicodeDecodeError:
            # Return as bytes if not valid UTF-8
            return decrypted_bytes
    
    def rotate_key(self, scope: EncryptionScope) -> EncryptionKey:
        """
        Rotate the active key for a scope.
        
        Args:
            scope: Scope to rotate key for
        
        Returns:
            New active EncryptionKey
        """
        # Get current key
        current_key = self.get_active_key(scope)
        if current_key:
            # Mark current key as inactive
            current_key.is_active = False
        
        # Generate new key with same algorithm
        scope_config = self.config.scope_configs[scope]
        algorithm = scope_config['algorithm']
        
        new_key = self.generate_key(algorithm, scope)
        self._set_active_key(scope, new_key.key_id)
        
        logger.info(f"Rotated key for scope {scope.value}: {current_key.key_id if current_key else 'none'} -> {new_key.key_id}")
        return new_key
    
    def check_key_expiration(self) -> List[EncryptionKey]:
        """
        Check for expired or soon-to-expire keys.
        
        Returns:
            List of keys that are expired or will expire soon
        """
        expiring_keys = []
        warning_threshold = datetime.utcnow() + timedelta(days=7)  # 7 days warning
        
        for key in self._keys.values():
            if key.expires_at and key.expires_at <= warning_threshold:
                expiring_keys.append(key)
        
        return expiring_keys
    
    def auto_rotate_expired_keys(self) -> List[Tuple[EncryptionScope, str, str]]:
        """
        Automatically rotate expired keys.
        
        Returns:
            List of (scope, old_key_id, new_key_id) tuples for rotated keys
        """
        rotated = []
        
        for scope in EncryptionScope:
            key = self.get_active_key(scope)
            if key and key.is_expired():
                new_key = self.rotate_key(scope)
                rotated.append((scope, key.key_id, new_key.key_id))
        
        return rotated
    
    def encrypt_string(self, text: str, scope: EncryptionScope) -> str:
        """
        Convenience method to encrypt a string and return base64 encoded result.
        
        Args:
            text: String to encrypt
            scope: Encryption scope
        
        Returns:
            Base64 encoded encrypted data
        """
        encrypted = self.encrypt(text, scope)
        return encrypted.to_base64()
    
    def decrypt_string(self, encrypted_b64: str) -> str:
        """
        Convenience method to decrypt base64 encoded string.
        
        Args:
            encrypted_b64: Base64 encoded encrypted data
        
        Returns:
            Decrypted string
        """
        encrypted_data = EncryptedData.from_base64(encrypted_b64)
        result = self.decrypt(encrypted_data)
        
        if isinstance(result, str):
            return result
        elif isinstance(result, bytes):
            return result.decode('utf-8')
        else:
            return json.dumps(result)
    
    def get_encryption_status(self) -> Dict[str, Any]:
        """Get encryption system status and statistics."""
        status = {
            'total_keys': len(self._keys),
            'active_keys': len(self._active_keys),
            'keys_by_algorithm': {},
            'keys_by_scope': {},
            'expiring_keys': len(self.check_key_expiration()),
            'config': {
                'default_algorithm': self.config.default_algorithm.value,
                'key_rotation_days': self.config.key_rotation_days,
                'enable_compression': self.config.enable_compression,
                'enable_integrity_check': self.config.enable_integrity_check
            }
        }
        
        for key in self._keys.values():
            # Count by algorithm
            alg = key.algorithm.value
            status['keys_by_algorithm'][alg] = status['keys_by_algorithm'].get(alg, 0) + 1
            
            # Count by scope
            scope = key.scope.value
            status['keys_by_scope'][scope] = status['keys_by_scope'].get(scope, 0) + 1
        
        return status
    
    def export_public_keys(self) -> Dict[str, str]:
        """
        Export public keys for RSA key pairs.
        
        Returns:
            Dictionary mapping key_id to public key PEM
        """
        public_keys = {}
        
        for key_id, key in self._keys.items():
            if key.algorithm in [EncryptionType.RSA_2048, EncryptionType.RSA_4096]:
                _, public_pem = self._rsa_keys[key_id]
                public_keys[key_id] = public_pem.decode('utf-8')
        
        return public_keys
    
    def verify_data_integrity(self, encrypted_data: EncryptedData) -> bool:
        """
        Verify the integrity of encrypted data.
        
        Args:
            encrypted_data: EncryptedData to verify
        
        Returns:
            True if data integrity is verified, False otherwise
        """
        try:
            # Try to decrypt - if successful, data is intact
            self.decrypt(encrypted_data)
            return True
        except Exception as e:
            logger.warning(f"Data integrity check failed: {str(e)}")
            return False