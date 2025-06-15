"""
Centralized Configuration Management System

This module provides a robust configuration system that:
- Centralizes all configuration settings
- Supports environment-specific overrides
- Validates configuration on startup
- Supports dynamic configuration updates
- Provides type-safe access to configuration values
"""

import os
import json
import yaml
import logging
from typing import Any, Dict, Optional, Union, List, Type
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
from datetime import datetime


class Environment(Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class VectorStoreType(Enum):
    MILVUS = "milvus"
    CHROMA = "chroma"
    FAISS = "faiss"
    NUMPY = "numpy"
    QDRANT = "qdrant"


class EmbeddingProviderType(Enum):
    GEMINI = "gemini"
    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OLLAMA = "ollama"


@dataclass
class DatabaseConfig:
    """Database connection configuration"""

    url: str = "sqlite:///memory_engine.db"
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600


@dataclass
class JanusGraphConfig:
    """JanusGraph database configuration"""

    host: str = "localhost"
    port: int = 8182
    use_ssl: bool = False
    username: Optional[str] = None
    password: Optional[str] = None
    connection_timeout: int = 30
    max_retry_attempts: int = 3
    retry_delay: float = 1.0

    @property
    def connection_url(self) -> str:
        protocol = "wss" if self.use_ssl else "ws"
        return f"{protocol}://{self.host}:{self.port}/gremlin"


@dataclass
class MilvusConfig:
    """Milvus vector database configuration"""

    host: str = "localhost"
    port: int = 19530
    user: Optional[str] = None
    password: Optional[str] = None
    collection_name: str = "knowledge_vectors"
    dimension: int = 768
    index_type: str = "IVF_FLAT"
    metric_type: str = "L2"
    nlist: int = 1024
    nprobe: int = 10


@dataclass
class VectorStoreConfig:
    """Vector store configuration"""

    type: VectorStoreType = VectorStoreType.MILVUS
    dimension: int = 768
    similarity_threshold: float = 0.7
    max_results: int = 100
    milvus: MilvusConfig = field(default_factory=MilvusConfig)


@dataclass
class EmbeddingConfig:
    """Legacy embedding model configuration (for backwards compatibility)"""

    model: str = "gemini-embedding-exp-03-07"
    dimension: int = 768  # Can be 768, 1536, or 3072 for gemini-embedding-exp-03-07
    batch_size: int = 32
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: int = 30


@dataclass
class GeminiEmbeddingConfig:
    """Gemini embedding provider configuration"""

    api_key: Optional[str] = None
    model_name: str = "gemini-embedding-exp-03-07"
    dimension: int = 768
    max_batch_size: int = 32
    timeout: int = 30


@dataclass
class OpenAIEmbeddingConfig:
    """OpenAI embedding provider configuration"""

    api_key: Optional[str] = None
    model_name: str = "text-embedding-3-small"
    dimension: int = 1536
    max_batch_size: int = 100
    timeout: int = 30
    organization: Optional[str] = None
    base_url: Optional[str] = None


@dataclass
class SentenceTransformersEmbeddingConfig:
    """Sentence Transformers embedding provider configuration"""

    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    max_batch_size: int = 64
    trust_remote_code: bool = False
    normalize_embeddings: bool = True
    cache_folder: Optional[str] = None


@dataclass
class OllamaEmbeddingConfig:
    """Ollama embedding provider configuration"""

    model_name: str = "nomic-embed-text"
    base_url: str = "http://localhost:11434"
    max_batch_size: int = 32
    timeout: int = 60
    keep_alive: str = "5m"


@dataclass
class ModularEmbeddingConfig:
    """Modular embedding system configuration"""

    provider: EmbeddingProviderType = EmbeddingProviderType.GEMINI
    gemini: GeminiEmbeddingConfig = field(default_factory=GeminiEmbeddingConfig)
    openai: OpenAIEmbeddingConfig = field(default_factory=OpenAIEmbeddingConfig)
    sentence_transformers: SentenceTransformersEmbeddingConfig = field(
        default_factory=SentenceTransformersEmbeddingConfig
    )
    ollama: OllamaEmbeddingConfig = field(default_factory=OllamaEmbeddingConfig)


@dataclass
class ChromaVectorStoreConfig:
    """ChromaDB vector store configuration"""

    path: Optional[str] = None
    collection_name: str = "knowledge_vectors"
    dimension: int = 768
    metric_type: str = "L2"
    batch_size: int = 1000
    host: Optional[str] = None
    port: Optional[int] = None
    ssl: bool = False
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class NumpyVectorStoreConfig:
    """NumPy vector store configuration"""

    collection_name: str = "knowledge_vectors"
    dimension: int = 768
    metric_type: str = "L2"
    persist_path: str = "./data/numpy_vectors"
    auto_save: bool = True
    max_memory_usage: int = 1000


@dataclass
class QdrantVectorStoreConfig:
    """Qdrant vector store configuration"""

    host: str = "localhost"
    port: int = 6333
    collection_name: str = "knowledge_vectors"
    dimension: int = 768
    metric_type: str = "COSINE"
    api_key: Optional[str] = None
    https: bool = False
    timeout: int = 60


@dataclass
class FAISSVectorStoreConfig:
    """FAISS vector store configuration"""

    collection_name: str = "knowledge_vectors"
    dimension: int = 768
    metric_type: str = "L2"
    index_type: str = "IndexFlatL2"
    persist_path: str = "./data/faiss_vectors"
    nlist: int = 1024
    use_gpu: bool = False


@dataclass
class ModularVectorStoreConfig:
    """Modular vector store configuration"""

    backend: VectorStoreType = VectorStoreType.NUMPY
    milvus: MilvusConfig = field(default_factory=MilvusConfig)
    chroma: ChromaVectorStoreConfig = field(default_factory=ChromaVectorStoreConfig)
    numpy: NumpyVectorStoreConfig = field(default_factory=NumpyVectorStoreConfig)
    qdrant: QdrantVectorStoreConfig = field(default_factory=QdrantVectorStoreConfig)
    faiss: FAISSVectorStoreConfig = field(default_factory=FAISSVectorStoreConfig)


@dataclass
class LLMConfig:
    """Large Language Model configuration"""

    model: str = "gemini-2.0-flash-thinking-exp"
    fallback_model: str = "gemini-2.0-flash-exp"
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class APIConfig:
    """API keys and authentication configuration"""

    google_api_key: Optional[str] = None

    @property
    def gemini_api_key(self) -> Optional[str]:
        """Compatibility property - returns the same as google_api_key"""
        return self.google_api_key


@dataclass
class LoggingConfig:
    """Logging configuration"""

    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    enable_console: bool = True


@dataclass
class VersioningConfig:
    """Versioning and revision management configuration"""

    enable_versioning: bool = True
    enable_snapshots: bool = True
    changes_threshold: int = 100
    snapshot_interval: int = 3600  # seconds
    max_revisions: int = 1000
    compression_enabled: bool = True


@dataclass
class SecurityConfig:
    """Security configuration"""

    encrypt_at_rest: bool = False
    encryption_key: Optional[str] = None
    tls_enabled: bool = False
    max_login_attempts: int = 5
    session_timeout: int = 3600  # seconds


@dataclass
class PerformanceConfig:
    """Performance and resource configuration"""

    workers: int = 4
    worker_timeout: int = 300
    max_memory_usage: int = 1024 * 1024 * 1024  # 1GB
    cache_size: int = 1000
    batch_processing_size: int = 100


@dataclass
class JsonFileStorageConfig:
    """JSON file storage configuration"""

    directory: str = "./data/graph"
    pretty_print: bool = True


@dataclass
class SqliteStorageConfig:
    """SQLite storage configuration"""

    database_path: str = "./data/knowledge.db"


@dataclass
class GraphStorageConfig:
    """Graph storage configuration"""

    backend: str = "janusgraph"  # Options: "janusgraph", "json_file", "sqlite"
    janusgraph: JanusGraphConfig = field(default_factory=JanusGraphConfig)
    json_file: JsonFileStorageConfig = field(default_factory=JsonFileStorageConfig)
    sqlite: SqliteStorageConfig = field(default_factory=SqliteStorageConfig)


@dataclass
class StorageConfig:
    """Storage layer configuration"""

    graph: GraphStorageConfig = field(default_factory=GraphStorageConfig)


@dataclass
class AppConfig:
    """Main application configuration"""

    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    testing: bool = False

    # Sub-configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    janusgraph: JanusGraphConfig = field(
        default_factory=JanusGraphConfig
    )  # For backwards compatibility

    # Legacy configurations (for backwards compatibility)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)

    # New modular configurations
    embeddings: ModularEmbeddingConfig = field(default_factory=ModularEmbeddingConfig)
    vector_stores: ModularVectorStoreConfig = field(default_factory=ModularVectorStoreConfig)

    llm: LLMConfig = field(default_factory=LLMConfig)
    api: APIConfig = field(default_factory=APIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    versioning: VersioningConfig = field(default_factory=VersioningConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)


class ConfigValidationError(Exception):
    """Raised when configuration validation fails"""

    pass


class ConfigManager:
    """
    Centralized configuration manager with support for:
    - Environment-specific configurations
    - Configuration validation
    - Dynamic configuration updates
    - File watching and auto-reload
    """

    _instance: Optional["ConfigManager"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Singleton pattern implementation"""
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        # Avoid re-initialization in singleton
        if hasattr(self, "_initialized"):
            return

        self._initialized = True
        # Default to config directory in project root
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            # Look for config directory relative to project root
            project_root = Path(__file__).parent.parent.parent
            self.config_dir = project_root / "config"

        self.config: AppConfig = AppConfig()
        self._file_watchers: Dict[str, float] = {}
        self._watch_thread: Optional[threading.Thread] = None
        self._watch_active = False
        self.logger = logging.getLogger(__name__)

        # Load configuration
        self._load_configuration()

    def _load_configuration(self):
        """Load configuration from multiple sources in priority order"""
        # 1. Load default configuration
        self.config = AppConfig()

        # 2. Load base configuration file
        self._load_from_file("config.yaml")
        self._load_from_file("config.json")

        # 3. Load environment-specific configuration
        env = os.getenv("ENVIRONMENT", "development").lower()
        self._load_from_file(f"environments/config.{env}.yaml")
        self._load_from_file(f"environments/config.{env}.json")

        # 4. Load from environment variables (highest priority)
        self._load_from_environment()

        # 5. Validate configuration
        self._validate_configuration()

        # 6. Start file watching if enabled
        if self.config.environment != Environment.PRODUCTION:
            self._start_file_watching()

    def _load_from_file(self, filename: str):
        """Load configuration from YAML/JSON file"""
        file_path = self.config_dir / filename
        if not file_path.exists():
            return

        try:
            with open(file_path, "r") as f:
                if filename.endswith(".yaml") or filename.endswith(".yml"):
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)

            if data:
                self._update_config_from_dict(data)
                self._file_watchers[str(file_path)] = file_path.stat().st_mtime
                self.logger.info(f"Loaded configuration from {filename}")

        except Exception as e:
            self.logger.warning(f"Failed to load configuration from {filename}: {e}")

    def _load_from_environment(self):
        """Load configuration from environment variables"""
        env_mappings = {
            # Environment
            "ENVIRONMENT": ("environment", lambda x: Environment(x.lower())),
            "DEBUG": ("debug", lambda x: x.lower() in ("true", "1", "yes")),
            # Database
            "DATABASE_URL": ("database.url", str),
            "DATABASE_POOL_SIZE": ("database.pool_size", int),
            # JanusGraph
            "JANUSGRAPH_HOST": ("janusgraph.host", str),
            "JANUSGRAPH_PORT": ("janusgraph.port", int),
            "JANUSGRAPH_USE_SSL": (
                "janusgraph.use_ssl",
                lambda x: x.lower() in ("true", "1", "yes"),
            ),
            "JANUSGRAPH_USERNAME": ("janusgraph.username", str),
            "JANUSGRAPH_PASSWORD": ("janusgraph.password", str),
            # Milvus
            "MILVUS_HOST": ("vector_store.milvus.host", str),
            "MILVUS_PORT": ("vector_store.milvus.port", int),
            "MILVUS_USER": ("vector_store.milvus.user", str),
            "MILVUS_PASSWORD": ("vector_store.milvus.password", str),
            "COLLECTION_NAME": ("vector_store.milvus.collection_name", str),
            # Vector Store
            "VECTOR_STORE_TYPE": ("vector_store.type", lambda x: VectorStoreType(x.lower())),
            "EMBEDDING_DIMENSION": ("embedding.dimension", int),
            # Embedding
            "EMBEDDING_MODEL": ("embedding.model", str),
            # LLM
            "LLM_MODEL": ("llm.model", str),
            "FALLBACK_MODEL": ("llm.fallback_model", str),
            # API Keys
            "GOOGLE_API_KEY": ("api.google_api_key", str),
            "GEMINI_API_KEY": ("api.google_api_key", str),  # For backward compatibility
            # Logging
            "LOG_LEVEL": ("logging.level", lambda x: LogLevel(x.upper())),
            "LOG_FORMAT": ("logging.format", str),
            # Versioning
            "ENABLE_VERSIONING": (
                "versioning.enable_versioning",
                lambda x: x.lower() in ("true", "1", "yes"),
            ),
            "ENABLE_SNAPSHOTS": (
                "versioning.enable_snapshots",
                lambda x: x.lower() in ("true", "1", "yes"),
            ),
            "CHANGES_THRESHOLD": ("versioning.changes_threshold", int),
            # Security
            "ENCRYPT_AT_REST": (
                "security.encrypt_at_rest",
                lambda x: x.lower() in ("true", "1", "yes"),
            ),
            "ENCRYPTION_KEY": ("security.encryption_key", str),
            "TLS_ENABLED": ("security.tls_enabled", lambda x: x.lower() in ("true", "1", "yes")),
            # Performance
            "WORKERS": ("performance.workers", int),
        }

        for env_var, (config_path, converter) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    converted_value = converter(value)
                    self._set_nested_attr(self.config, config_path, converted_value)

                    # Special handling for EMBEDDING_DIMENSION to keep vector store in sync
                    if env_var == "EMBEDDING_DIMENSION":
                        self._set_nested_attr(
                            self.config, "vector_store.dimension", converted_value
                        )
                        self._set_nested_attr(
                            self.config, "vector_store.milvus.dimension", converted_value
                        )

                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Invalid value for {env_var}: {value}, error: {e}")

    def _update_config_from_dict(self, data: Dict[str, Any], prefix: str = ""):
        """Update configuration from dictionary recursively"""
        for key, value in data.items():
            config_path = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                self._update_config_from_dict(value, config_path)
            else:
                try:
                    # Handle enum conversions for file-based config
                    if config_path == "environment" and isinstance(value, str):
                        value = Environment(value.lower())
                    elif config_path == "logging.level" and isinstance(value, str):
                        value = LogLevel(value.upper())
                    elif config_path == "vector_store.type" and isinstance(value, str):
                        value = VectorStoreType(value.lower())

                    self._set_nested_attr(self.config, config_path, value)

                    # Handle dimension synchronization for file-based config
                    if config_path == "embedding.dimension":
                        self._set_nested_attr(self.config, "vector_store.dimension", value)
                        self._set_nested_attr(self.config, "vector_store.milvus.dimension", value)
                    elif config_path == "vector_store.dimension":
                        self._set_nested_attr(self.config, "embedding.dimension", value)
                        self._set_nested_attr(self.config, "vector_store.milvus.dimension", value)

                except AttributeError:
                    self.logger.warning(f"Unknown configuration key: {config_path}")
                except ValueError as e:
                    self.logger.warning(f"Invalid value for {config_path}: {value}, error: {e}")

    def _set_nested_attr(self, obj: Any, path: str, value: Any):
        """Set nested attribute using dot notation"""
        parts = path.split(".")
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    def _validate_configuration(self):
        """Validate configuration settings"""
        errors = []

        # Validate required API keys
        if not self.config.api.google_api_key:
            errors.append("GOOGLE_API_KEY is required")

        # Validate database configuration
        if not self.config.database.url:
            errors.append("DATABASE_URL is required")

        # Validate vector store dimension consistency
        if self.config.vector_store.dimension != self.config.embedding.dimension:
            errors.append(
                f"Vector store dimension ({self.config.vector_store.dimension}) "
                f"must match embedding dimension ({self.config.embedding.dimension})"
            )

        # Validate JanusGraph configuration
        if self.config.janusgraph.port <= 0 or self.config.janusgraph.port > 65535:
            errors.append("JanusGraph port must be between 1 and 65535")

        # Validate Milvus configuration
        if (
            self.config.vector_store.milvus.port <= 0
            or self.config.vector_store.milvus.port > 65535
        ):
            errors.append("Milvus port must be between 1 and 65535")

        # Validate security configuration
        if self.config.security.encrypt_at_rest and not self.config.security.encryption_key:
            errors.append("Encryption key is required when encryption at rest is enabled")

        if errors:
            raise ConfigValidationError(f"Configuration validation failed: {'; '.join(errors)}")

        self.logger.info("Configuration validation passed")

    def _start_file_watching(self):
        """Start watching configuration files for changes"""
        if self._watch_active:
            return

        self._watch_active = True
        self._watch_thread = threading.Thread(target=self._file_watch_loop, daemon=True)
        self._watch_thread.start()
        self.logger.info("Started configuration file watching")

    def _file_watch_loop(self):
        """File watching loop"""
        while self._watch_active:
            try:
                for file_path, last_mtime in list(self._file_watchers.items()):
                    path = Path(file_path)
                    if path.exists():
                        current_mtime = path.stat().st_mtime
                        if current_mtime > last_mtime:
                            self.logger.info(
                                f"Configuration file {file_path} changed, reloading..."
                            )
                            self.reload_configuration()
                            break

                time.sleep(1.0)  # Check every second
            except Exception as e:
                self.logger.error(f"Error in file watching: {e}")
                time.sleep(5.0)  # Wait longer on error

    def reload_configuration(self):
        """Reload configuration from all sources"""
        try:
            self._load_configuration()
            self.logger.info("Configuration reloaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to reload configuration: {e}")
            raise

    def get(self, path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        try:
            obj = self.config
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            return default

    def set(self, path: str, value: Any):
        """Set configuration value using dot notation"""
        self._set_nested_attr(self.config, path, value)
        self._validate_configuration()

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""

        def _asdict_recursive(obj):
            if hasattr(obj, "__dict__"):
                result = {}
                for key, value in obj.__dict__.items():
                    if isinstance(value, Enum):
                        result[key] = value.value
                    elif hasattr(value, "__dict__"):
                        result[key] = _asdict_recursive(value)
                    else:
                        result[key] = value

                # Add APIConfig compatibility properties
                if hasattr(obj, "gemini_api_key") and "google_api_key" in result:
                    result["gemini_api_key"] = obj.gemini_api_key

                return result
            return obj

        return _asdict_recursive(self.config)

    def save_to_file(self, filename: str, format: str = "yaml"):
        """Save current configuration to file"""
        file_path = self.config_dir / filename
        config_dict = self.to_dict()

        with open(file_path, "w") as f:
            if format.lower() == "yaml":
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            else:
                json.dump(config_dict, f, indent=2)

        self.logger.info(f"Configuration saved to {filename}")

    def stop_file_watching(self):
        """Stop file watching"""
        self._watch_active = False
        if self._watch_thread and self._watch_thread.is_alive():
            self._watch_thread.join(timeout=5.0)
        self.logger.info("Stopped configuration file watching")

    def get_embedding_config(self) -> Dict[str, Any]:
        """
        Get embedding configuration for the modular embedding manager.

        Returns:
            Dictionary with provider and provider_config keys
        """
        embeddings_config = self.config.embeddings
        provider_type = embeddings_config.provider.value

        # Get provider-specific configuration
        provider_config = {}
        if provider_type == "gemini":
            gemini_config = embeddings_config.gemini
            provider_config = {
                "api_key": gemini_config.api_key or self.config.api.google_api_key,
                "model_name": gemini_config.model_name,
                "dimension": gemini_config.dimension,
                "max_batch_size": gemini_config.max_batch_size,
                "timeout": gemini_config.timeout,
            }
        elif provider_type == "openai":
            openai_config = embeddings_config.openai
            provider_config = {
                "api_key": openai_config.api_key,
                "model_name": openai_config.model_name,
                "dimension": openai_config.dimension,
                "max_batch_size": openai_config.max_batch_size,
                "timeout": openai_config.timeout,
                "organization": openai_config.organization,
                "base_url": openai_config.base_url,
            }
        elif provider_type == "sentence_transformers":
            st_config = embeddings_config.sentence_transformers
            provider_config = {
                "model_name": st_config.model_name,
                "device": st_config.device,
                "max_batch_size": st_config.max_batch_size,
                "trust_remote_code": st_config.trust_remote_code,
                "normalize_embeddings": st_config.normalize_embeddings,
                "cache_folder": st_config.cache_folder,
            }
        elif provider_type == "ollama":
            ollama_config = embeddings_config.ollama
            provider_config = {
                "model_name": ollama_config.model_name,
                "base_url": ollama_config.base_url,
                "max_batch_size": ollama_config.max_batch_size,
                "timeout": ollama_config.timeout,
                "keep_alive": ollama_config.keep_alive,
            }

        return {"provider": provider_type, "provider_config": provider_config}

    def get_vector_store_config(self) -> Dict[str, Any]:
        """
        Get vector store configuration for the modular embedding manager.

        Returns:
            Dictionary with backend and backend_config keys
        """
        vector_stores_config = self.config.vector_stores
        backend_type = vector_stores_config.backend.value

        # Get backend-specific configuration
        backend_config = {}
        if backend_type == "milvus":
            milvus_config = vector_stores_config.milvus
            backend_config = {
                "host": milvus_config.host,
                "port": milvus_config.port,
                "collection_name": milvus_config.collection_name,
                "dimension": milvus_config.dimension,
                "metric_type": milvus_config.metric_type,
                "index_type": milvus_config.index_type,
                "nlist": milvus_config.nlist,
                "nprobe": milvus_config.nprobe,
                "user": milvus_config.user,
                "password": milvus_config.password,
            }
        elif backend_type == "chroma":
            chroma_config = vector_stores_config.chroma
            backend_config = {
                "path": chroma_config.path,
                "collection_name": chroma_config.collection_name,
                "dimension": chroma_config.dimension,
                "metric_type": chroma_config.metric_type,
                "batch_size": chroma_config.batch_size,
                "host": chroma_config.host,
                "port": chroma_config.port,
                "ssl": chroma_config.ssl,
                "headers": chroma_config.headers,
            }
        elif backend_type == "numpy":
            numpy_config = vector_stores_config.numpy
            backend_config = {
                "collection_name": numpy_config.collection_name,
                "dimension": numpy_config.dimension,
                "metric_type": numpy_config.metric_type,
                "persist_path": numpy_config.persist_path,
                "auto_save": numpy_config.auto_save,
                "max_memory_usage": numpy_config.max_memory_usage,
            }
        elif backend_type == "qdrant":
            qdrant_config = vector_stores_config.qdrant
            backend_config = {
                "host": qdrant_config.host,
                "port": qdrant_config.port,
                "collection_name": qdrant_config.collection_name,
                "dimension": qdrant_config.dimension,
                "metric_type": qdrant_config.metric_type,
                "api_key": qdrant_config.api_key,
                "https": qdrant_config.https,
                "timeout": qdrant_config.timeout,
            }
        elif backend_type == "faiss":
            faiss_config = vector_stores_config.faiss
            backend_config = {
                "collection_name": faiss_config.collection_name,
                "dimension": faiss_config.dimension,
                "metric_type": faiss_config.metric_type,
                "index_type": faiss_config.index_type,
                "persist_path": faiss_config.persist_path,
                "nlist": faiss_config.nlist,
                "use_gpu": faiss_config.use_gpu,
            }

        return {"backend": backend_type, "backend_config": backend_config}


# Global configuration instance
_config_manager: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get the global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def init_config(config_dir: Optional[Union[str, Path]] = None) -> ConfigManager:
    """Initialize the global configuration manager"""
    global _config_manager
    _config_manager = ConfigManager(config_dir)
    return _config_manager
