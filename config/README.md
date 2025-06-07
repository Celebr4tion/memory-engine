# Configuration Directory

This directory contains all configuration files for the Memory Engine project.

## Structure

```
config/
├── README.md                    # This file
├── config.yaml                 # Base configuration (default values)
└── environments/               # Environment-specific configurations
    ├── config.development.yaml # Development environment
    ├── config.testing.yaml     # Testing environment  
    ├── config.staging.yaml     # Staging environment
    └── config.production.yaml  # Production environment
```

## Usage

The configuration system automatically loads:
1. Base configuration from `config.yaml`
2. Environment-specific overrides from `environments/config.{ENVIRONMENT}.yaml`
3. Environment variables (highest priority)

## Environment Variables

Set the `ENVIRONMENT` variable to specify which environment configuration to load:

```bash
export ENVIRONMENT=development  # Uses config.development.yaml
export ENVIRONMENT=testing      # Uses config.testing.yaml
export ENVIRONMENT=staging      # Uses config.staging.yaml
export ENVIRONMENT=production   # Uses config.production.yaml
```

For more details, see the [Configuration System Documentation](../docs/configuration_system.md).