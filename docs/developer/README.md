# Developer Documentation

Documentation for developers working on the Memory Engine project.

## Available Guides

### [Setup Guide](setup_guide.md)
Complete development environment setup including dependencies, services, and tools.

### [Architecture](architecture.md)
System architecture overview, components, and design decisions.

### [Configuration System](configuration_system.md)
Comprehensive guide to the configuration management system including advanced features.

## Development Workflow

### Initial Setup
1. Clone the repository
2. Run setup script: `./scripts/setup.sh`
3. Activate virtual environment: `source .venv/bin/activate`
4. Set up environment variables in `.env`

### Daily Development
1. Run tests: `./scripts/test.sh unit`
2. Run linting: `./scripts/test.sh lint`
3. Format code: `./scripts/test.sh format`
4. Run all checks: `./scripts/test.sh check`

### Testing
- **Unit Tests**: `./scripts/test.sh unit` (fast, no external dependencies)
- **Integration Tests**: `./scripts/test.sh integration` (requires services)
- **Coverage**: `./scripts/test.sh coverage`

### Code Standards
- Follow PEP 8 style guidelines
- Use type hints where possible
- Write comprehensive tests
- Document complex functions and classes
- Keep functions focused and small

### Project Structure
```
memory_core/          # Main package
├── config/          # Configuration management
├── core/            # Core engine components
├── db/              # Database adapters
├── embeddings/      # Embedding and vector operations
├── ingestion/       # Data ingestion and processing
├── model/           # Data models
└── ...
```

For detailed architecture information, see [architecture.md](architecture.md).