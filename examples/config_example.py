#!/usr/bin/env python3
"""
Configuration System Example

This example demonstrates how to use the Memory Engine's configuration system.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from memory_core.config import get_config, init_config, ConfigValidationError, Environment


def basic_usage_example():
    """Example of basic configuration usage."""
    print("=== Basic Configuration Usage ===")
    
    try:
        # Get the global configuration instance
        config = get_config()
        
        # Access configuration values
        print(f"Environment: {config.config.environment.value}")
        print(f"Debug mode: {config.config.debug}")
        print(f"Database URL: {config.config.database.url}")
        print(f"JanusGraph host: {config.config.janusgraph.host}")
        print(f"JanusGraph port: {config.config.janusgraph.port}")
        print(f"Vector store type: {config.config.vector_store.type.value}")
        print(f"Embedding model: {config.config.embedding.model}")
        print(f"LLM model: {config.config.llm.model}")
        
        # Check if API key is configured
        if config.config.api.gemini_api_key:
            print(f"Gemini API key: {'*' * 20}...{config.config.api.gemini_api_key[-4:]}")
        else:
            print("⚠️  Gemini API key not configured")
            
    except ConfigValidationError as e:
        print(f"❌ Configuration validation failed: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")


def get_set_example():
    """Example of getting and setting configuration values."""
    print("\n=== Get/Set Configuration Values ===")
    
    config = get_config()
    
    # Get values using dot notation
    host = config.get('janusgraph.host')
    port = config.get('janusgraph.port', 8182)  # with default
    nonexistent = config.get('nonexistent.key', 'default_value')
    
    print(f"JanusGraph host: {host}")
    print(f"JanusGraph port: {port}")
    print(f"Nonexistent key: {nonexistent}")
    
    # Set configuration values
    print("\nSetting new values...")
    config.set('janusgraph.host', 'custom-host')
    config.set('janusgraph.port', 9999)
    
    print(f"New JanusGraph host: {config.get('janusgraph.host')}")
    print(f"New JanusGraph port: {config.get('janusgraph.port')}")


def environment_specific_example():
    """Example of environment-specific behavior."""
    print("\n=== Environment-Specific Behavior ===")
    
    config = get_config()
    
    # Environment-specific logic
    if config.config.environment == Environment.PRODUCTION:
        print("🏭 Running in PRODUCTION mode")
        print(f"  - Security features enabled: {config.config.security.encrypt_at_rest}")
        print(f"  - TLS enabled: {config.config.security.tls_enabled}")
        print(f"  - Workers: {config.config.performance.workers}")
        
    elif config.config.environment == Environment.DEVELOPMENT:
        print("🛠️  Running in DEVELOPMENT mode")
        print(f"  - Debug mode: {config.config.debug}")
        print(f"  - Log level: {config.config.logging.level.value}")
        print(f"  - File watching enabled for config reload")
        
    elif config.config.environment == Environment.TESTING:
        print("🧪 Running in TESTING mode")
        print(f"  - In-memory database: {config.config.database.url}")
        print(f"  - Versioning disabled: {not config.config.versioning.enable_versioning}")
        
    else:
        print(f"Running in {config.config.environment.value.upper()} mode")


def configuration_to_dict_example():
    """Example of converting configuration to dictionary."""
    print("\n=== Configuration to Dictionary ===")
    
    config = get_config()
    config_dict = config.to_dict()
    
    # Print some interesting sections
    print("Database configuration:")
    for key, value in config_dict['database'].items():
        print(f"  {key}: {value}")
    
    print("\nJanusGraph configuration:")
    for key, value in config_dict['janusgraph'].items():
        print(f"  {key}: {value}")
    
    print("\nVector store configuration:")
    for key, value in config_dict['vector_store'].items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                print(f"    {subkey}: {subvalue}")
        else:
            print(f"  {key}: {value}")


def save_configuration_example():
    """Example of saving configuration to file."""
    print("\n=== Save Configuration Example ===")
    
    config = get_config()
    
    # Save current configuration
    try:
        config.save_to_file('current_config.yaml', 'yaml')
        print("✅ Configuration saved to current_config.yaml")
        
        config.save_to_file('current_config.json', 'json')
        print("✅ Configuration saved to current_config.json")
        
    except Exception as e:
        print(f"❌ Error saving configuration: {e}")


def demonstrate_validation():
    """Demonstrate configuration validation."""
    print("\n=== Configuration Validation Example ===")
    
    # Show what happens with invalid configuration
    print("Current configuration validation status:")
    try:
        config = get_config()
        print("✅ Configuration is valid")
        
        # Try to set invalid values to show validation
        try:
            config.set('janusgraph.port', 70000)  # Invalid port
            print("❌ Validation should have failed for invalid port")
        except Exception as e:
            print(f"✅ Validation caught invalid port: {e}")
            
    except ConfigValidationError as e:
        print(f"❌ Configuration validation failed: {e}")
        print("💡 Make sure GEMINI_API_KEY is set in your environment")


def main():
    """Main function to run all examples."""
    print("Memory Engine Configuration System Examples")
    print("=" * 50)
    
    # Set a dummy API key if not set for demonstration purposes
    if not os.getenv('GEMINI_API_KEY'):
        print("⚠️  GEMINI_API_KEY not found in environment.")
        print("Setting a dummy key for demonstration purposes...")
        os.environ['GEMINI_API_KEY'] = 'dummy-key-for-demo'
    
    # Set encryption key for production environment if needed
    if os.getenv('ENVIRONMENT', '').lower() == 'production' and not os.getenv('ENCRYPTION_KEY'):
        print("⚠️  Production environment requires ENCRYPTION_KEY.")
        print("Setting a dummy encryption key for demonstration purposes...")
        os.environ['ENCRYPTION_KEY'] = 'dummy-encryption-key-for-demo'
    
    # Run examples
    basic_usage_example()
    get_set_example()
    environment_specific_example()
    configuration_to_dict_example()
    save_configuration_example()
    demonstrate_validation()
    
    print("\n" + "=" * 50)
    print("Configuration examples completed!")
    print("\nTo run with different environments, set the ENVIRONMENT variable:")
    print("  ENVIRONMENT=production python examples/config_example.py")
    print("  ENVIRONMENT=testing python examples/config_example.py")


if __name__ == "__main__":
    main()