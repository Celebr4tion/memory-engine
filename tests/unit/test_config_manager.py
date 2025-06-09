"""
Tests for the configuration management system.
"""
import os
import tempfile
import pytest
import json
import yaml
import logging
from pathlib import Path
from unittest.mock import patch

from memory_core.config.config_manager import (
    ConfigManager, 
    AppConfig, 
    Environment, 
    LogLevel, 
    VectorStoreType,
    ConfigValidationError,
    get_config,
    init_config
)


class TestConfigManager:
    """Test the ConfigManager class."""
    
    def setup_method(self):
        """Set up for each test method."""
        # Reset the singleton
        ConfigManager._instance = None
        global _config_manager
        import memory_core.config.config_manager as config_module
        config_module._config_manager = None
    
    def test_singleton_pattern(self):
        """Test that ConfigManager follows singleton pattern."""
        config1 = ConfigManager()
        config2 = ConfigManager()
        assert config1 is config2
    
    def test_default_configuration(self):
        """Test that default configuration is properly loaded."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ConfigManager(temp_dir)
            
            # Check default values
            assert config.config.environment == Environment.DEVELOPMENT
            assert config.config.database.url == "sqlite:///memory_engine.db"
            assert config.config.janusgraph.host == "localhost"
            assert config.config.janusgraph.port == 8182
            assert config.config.vector_store.type == VectorStoreType.MILVUS
            assert config.config.embedding.model == "gemini-embedding-exp-03-07"
            assert config.config.llm.model == "gemini-2.0-flash-thinking-exp"
    
    def test_yaml_config_loading(self):
        """Test loading configuration from YAML file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            
            # Create a test config file
            test_config = {
                "environment": "testing",
                "debug": True,
                "database": {
                    "url": "postgresql://test:test@localhost:5432/testdb"
                },
                "janusgraph": {
                    "host": "test-host",
                    "port": 9999
                }
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(test_config, f)
            
            # Clear environment variables to test file loading
            env_vars = {'GOOGLE_API_KEY': 'test-key'}
            with patch.dict(os.environ, env_vars, clear=True):
                config = ConfigManager(temp_dir)
                
                # Check that values were loaded
                assert config.config.environment == Environment.TESTING
                assert config.config.debug == True
                assert config.config.database.url == "postgresql://test:test@localhost:5432/testdb"
                assert config.config.janusgraph.host == "test-host"
                assert config.config.janusgraph.port == 9999
    
    def test_json_config_loading(self):
        """Test loading configuration from JSON file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            
            # Create a test config file
            test_config = {
                "environment": "staging",
                "embedding": {
                    "model": "custom-embedding-model",
                    "dimension": 1024
                },
                "llm": {
                    "temperature": 0.8
                }
            }
            
            with open(config_path, 'w') as f:
                json.dump(test_config, f)
            
            # Clear environment variables to test file loading
            env_vars = {'GOOGLE_API_KEY': 'test-key'}
            with patch.dict(os.environ, env_vars, clear=True):
                config = ConfigManager(temp_dir)
                
                # Check that values were loaded
                assert config.config.environment == Environment.STAGING
                assert config.config.embedding.model == "custom-embedding-model"
                assert config.config.embedding.dimension == 1024
                assert config.config.llm.temperature == 0.8
    
    def test_environment_specific_config(self):
        """Test loading environment-specific configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create proper directory structure
            environments_dir = Path(temp_dir) / "environments"
            environments_dir.mkdir()
            
            # Create base config
            base_config = {"database": {"url": "sqlite:///base.db"}}
            with open(Path(temp_dir) / "config.yaml", 'w') as f:
                yaml.dump(base_config, f)
            
            # Create environment-specific config in environments directory
            env_config = {"database": {"url": "sqlite:///dev.db"}}
            with open(environments_dir / "config.development.yaml", 'w') as f:
                yaml.dump(env_config, f)
            
            # Clear environment variables and set only what we need
            env_vars = {'ENVIRONMENT': 'development', 'GOOGLE_API_KEY': 'test-key'}
            with patch.dict(os.environ, env_vars, clear=True):
                config = ConfigManager(temp_dir)
                # Environment-specific config should override base config
                assert config.config.database.url == "sqlite:///dev.db"
    
    def test_environment_variable_override(self):
        """Test that environment variables override file configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create config file
            test_config = {"janusgraph": {"host": "file-host", "port": 8182}}
            with open(Path(temp_dir) / "config.yaml", 'w') as f:
                yaml.dump(test_config, f)
            
            # Set environment variables
            env_vars = {
                'JANUSGRAPH_HOST': 'env-host',
                'JANUSGRAPH_PORT': '9999',
                'DEBUG': 'true',
                'GOOGLE_API_KEY': 'test-api-key'
            }
            
            with patch.dict(os.environ, env_vars):
                config = ConfigManager(temp_dir)
                
                # Environment variables should override file config
                assert config.config.janusgraph.host == "env-host"
                assert config.config.janusgraph.port == 9999
                assert config.config.debug == True
                assert config.config.api.gemini_api_key == "test-api-key"
    
    def test_configuration_validation_success(self):
        """Test successful configuration validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            env_vars = {
                'GOOGLE_API_KEY': 'valid-api-key',
                'DATABASE_URL': 'sqlite:///test.db'
            }
            
            with patch.dict(os.environ, env_vars):
                # Should not raise any exception
                config = ConfigManager(temp_dir)
                assert config.config.api.gemini_api_key == "valid-api-key"
    
    def test_configuration_validation_failure(self):
        """Test configuration validation failures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Missing required API key should cause validation to fail
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(ConfigValidationError) as exc_info:
                    ConfigManager(temp_dir)
                assert "GOOGLE_API_KEY is required" in str(exc_info.value)
    
    def test_dimension_mismatch_validation(self):
        """Test validation of dimension mismatch between vector store and embedding."""
        with tempfile.TemporaryDirectory() as temp_dir:
            env_vars = {'GOOGLE_API_KEY': 'test-key'}
            with patch.dict(os.environ, env_vars, clear=True):
                # Create config and manually set mismatched dimensions
                config = ConfigManager(temp_dir)
                
                # Manually set mismatched dimensions to test validation
                config.config.vector_store.dimension = 1024
                config.config.vector_store.milvus.dimension = 1024
                config.config.embedding.dimension = 768
                
                # Now validation should fail due to dimension mismatch
                with pytest.raises(ConfigValidationError) as exc_info:
                    config._validate_configuration()
                assert "must match embedding dimension" in str(exc_info.value)
    
    def test_get_and_set_methods(self):
        """Test the get and set methods for configuration values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            env_vars = {'GOOGLE_API_KEY': 'test-key'}
            with patch.dict(os.environ, env_vars):
                config = ConfigManager(temp_dir)
                
                # Test get method
                assert config.get('janusgraph.host') == 'localhost'
                assert config.get('nonexistent.key', 'default') == 'default'
                
                # Test set method
                config.set('janusgraph.host', 'new-host')
                assert config.get('janusgraph.host') == 'new-host'
                assert config.config.janusgraph.host == 'new-host'
    
    def test_to_dict_method(self):
        """Test converting configuration to dictionary."""
        with tempfile.TemporaryDirectory() as temp_dir:
            env_vars = {'GOOGLE_API_KEY': 'test-key'}
            with patch.dict(os.environ, env_vars):
                config = ConfigManager(temp_dir)
                config_dict = config.to_dict()
                
                assert isinstance(config_dict, dict)
                assert config_dict['environment'] == 'development'
                assert config_dict['janusgraph']['host'] == 'localhost'
                assert config_dict['api']['gemini_api_key'] == 'test-key'
    
    def test_save_to_file(self):
        """Test saving configuration to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            env_vars = {'GOOGLE_API_KEY': 'test-key'}
            with patch.dict(os.environ, env_vars):
                config = ConfigManager(temp_dir)
                
                # Test YAML save
                yaml_file = "test_config.yaml"
                config.save_to_file(yaml_file, 'yaml')
                
                yaml_path = Path(temp_dir) / yaml_file
                assert yaml_path.exists()
                
                with open(yaml_path, 'r') as f:
                    saved_config = yaml.safe_load(f)
                assert saved_config['environment'] == 'development'
                
                # Test JSON save
                json_file = "test_config.json"
                config.save_to_file(json_file, 'json')
                
                json_path = Path(temp_dir) / json_file
                assert json_path.exists()
                
                with open(json_path, 'r') as f:
                    saved_config = json.load(f)
                assert saved_config['environment'] == 'development'
    
    def test_global_config_functions(self):
        """Test global configuration functions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            env_vars = {'GOOGLE_API_KEY': 'test-key'}
            with patch.dict(os.environ, env_vars):
                # Test init_config
                config1 = init_config(temp_dir)
                assert isinstance(config1, ConfigManager)
                
                # Test get_config
                config2 = get_config()
                assert config1 is config2


class TestConfigDataClasses:
    """Test the configuration data classes."""
    
    def test_janusgraph_config_connection_url(self):
        """Test JanusGraphConfig connection URL property."""
        from memory_core.config.config_manager import JanusGraphConfig
        
        # Test without SSL
        config = JanusGraphConfig(host="test-host", port=8182, use_ssl=False)
        assert config.connection_url == "ws://test-host:8182/gremlin"
        
        # Test with SSL
        config_ssl = JanusGraphConfig(host="secure-host", port=8183, use_ssl=True)
        assert config_ssl.connection_url == "wss://secure-host:8183/gremlin"
    
    def test_api_config_fallback(self):
        """Test APIConfig fallback from google_api_key to gemini_api_key."""
        from memory_core.config.config_manager import APIConfig
        
        # Test fallback when google_api_key is set
        config = APIConfig(google_api_key="google-key")
        assert config.gemini_api_key == "google-key"
        
        # Test when google_api_key is None
        config2 = APIConfig(google_api_key=None)
        assert config2.gemini_api_key is None
    
    def test_enum_values(self):
        """Test enum value conversions."""
        assert Environment.DEVELOPMENT.value == "development"
        assert LogLevel.INFO.value == "INFO"
        assert VectorStoreType.MILVUS.value == "milvus"