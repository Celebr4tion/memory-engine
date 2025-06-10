# Contributing to Memory Engine

Thank you for your interest in contributing to Memory Engine! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Security](#security)
- [Community](#community)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. We are committed to providing a welcoming and inclusive environment for all contributors.

### Our Standards

**Positive behaviors include:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behaviors include:**
- Harassment, discrimination, or exclusionary behavior
- Trolling, insulting/derogatory comments, and personal attacks
- Public or private harassment
- Publishing others' private information without permission
- Other conduct which could reasonably be considered inappropriate

## Getting Started

### Prerequisites

Before contributing, ensure you have:

1. **Python 3.9+** installed
2. **Git** for version control
3. **JanusGraph** (for graph storage testing)
4. **Milvus** (for vector storage testing)
5. Basic familiarity with knowledge graphs and NLP concepts

### First-Time Contributors

1. **Star the repository** to show your support
2. **Fork the repository** to your GitHub account
3. **Read the documentation** in the `docs/` directory
4. **Look for "good first issue" labels** on GitHub issues
5. **Join our community discussions** for questions and support

### Areas for Contribution

We welcome contributions in various areas:

- **Core Features**: Knowledge ingestion, querying, synthesis
- **Security**: Authentication, authorization, encryption
- **Performance**: Optimization, caching, scalability
- **Quality**: Knowledge quality assessment and enhancement
- **Monitoring**: Performance monitoring, alerting, observability
- **Documentation**: User guides, API docs, tutorials
- **Testing**: Unit tests, integration tests, performance tests
- **Examples**: Usage examples, tutorials, demos

## Development Setup

### 1. Clone and Setup

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/memory-engine.git
cd memory-engine

# Add upstream remote
git remote add upstream https://github.com/Celebr4tion/memory-engine.git

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov flake8 black isort mypy
```

### 2. Environment Configuration

```bash
# Copy example configuration
cp config/config.yaml config/config.local.yaml

# Set environment variables
export GOOGLE_API_KEY="your-api-key"
export OPENAI_API_KEY="your-openai-key"  # if using OpenAI

# For testing
export MEMORY_ENGINE_ENV="testing"
```

### 3. Database Setup

#### JanusGraph (Graph Storage)
```bash
# Download and start JanusGraph
# See docs/developer/setup_guide.md for detailed instructions
```

#### Milvus (Vector Storage)
```bash
# Install and start Milvus
# See docs/developer/setup_guide.md for detailed instructions
```

### 4. Verify Setup

```bash
# Run tests to verify setup
python -m pytest tests/integration/ -v

# Run a basic example
python examples/basic_usage.py
```

## Contributing Guidelines

### Branch Strategy

We use GitFlow branching strategy:

- **main**: Production-ready code
- **development**: Integration branch for features
- **feature/**: Feature development branches
- **hotfix/**: Critical fixes for production
- **release/**: Release preparation branches

### Feature Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout development
   git pull upstream development
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Follow coding standards
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**:
   ```bash
   # Run full test suite
   python -m pytest

   # Run specific tests
   python -m pytest tests/test_your_feature.py

   # Check code quality
   flake8 memory_core/
   black --check memory_core/
   isort --check memory_core/
   ```

4. **Commit and push**:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request** to the `development` branch

### Commit Message Convention

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(auth): add JWT token authentication
fix(query): resolve cache invalidation issue
docs(security): add encryption documentation
test(rbac): add role hierarchy tests
```

## Pull Request Process

### Before Submitting

1. **Ensure tests pass**:
   ```bash
   python -m pytest
   ```

2. **Check code quality**:
   ```bash
   flake8 memory_core/
   black memory_core/
   isort memory_core/
   ```

3. **Update documentation** if needed

4. **Add tests** for new functionality

5. **Update CHANGELOG.md** if applicable

### Pull Request Template

When creating a PR, include:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Testing
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No merge conflicts
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs tests and quality checks
2. **Code Review**: Maintainers review code for quality and design
3. **Testing**: Additional testing if needed
4. **Approval**: At least one maintainer approval required
5. **Merge**: Squash and merge to development branch

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://peps.python.org/pep-0008/) with some modifications:

- **Line Length**: 100 characters (instead of 79)
- **Imports**: Use `isort` for import organization
- **Formatting**: Use `black` for code formatting
- **Type Hints**: Use type hints for all public functions

### Code Quality Tools

```bash
# Format code
black memory_core/
isort memory_core/

# Check style
flake8 memory_core/

# Type checking
mypy memory_core/

# Security scanning
bandit -r memory_core/
```

### Documentation Strings

Use Google-style docstrings:

```python
def example_function(param1: str, param2: int) -> bool:
    """Brief description of the function.
    
    Longer description if needed, explaining the purpose,
    behavior, and any important details.
    
    Args:
        param1: Description of param1.
        param2: Description of param2.
        
    Returns:
        Description of return value.
        
    Raises:
        ValueError: Description of when this is raised.
        
    Example:
        >>> result = example_function("test", 42)
        >>> print(result)
        True
    """
    return True
```

### Security Guidelines

1. **Never commit secrets** (API keys, passwords, etc.)
2. **Use environment variables** for configuration
3. **Follow secure coding practices**:
   - Input validation
   - SQL injection prevention
   - XSS protection
   - Authentication and authorization
4. **Security testing** for security-related changes
5. **Follow our security policy** in SECURITY.md

## Testing

### Test Types

1. **Unit Tests**: Test individual components
2. **Integration Tests**: Test component interactions
3. **Performance Tests**: Test performance characteristics
4. **Security Tests**: Test security features

### Testing Guidelines

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=memory_core --cov-report=html

# Run specific test categories
python -m pytest tests/unit/           # Unit tests
python -m pytest tests/integration/   # Integration tests
python -m pytest tests/test_security_*.py  # Security tests

# Run performance tests
python -m pytest tests/test_performance_*.py -v
```

### Writing Tests

1. **Test Structure**: Use AAA pattern (Arrange, Act, Assert)
2. **Test Names**: Descriptive test names
3. **Fixtures**: Use pytest fixtures for setup
4. **Mocking**: Mock external dependencies
5. **Coverage**: Aim for >90% test coverage

Example test:

```python
def test_knowledge_node_creation():
    """Test that knowledge nodes are created correctly."""
    # Arrange
    content = "Test knowledge content"
    metadata = {"source": "test"}
    
    # Act
    node = KnowledgeNode(content=content, metadata=metadata)
    
    # Assert
    assert node.content == content
    assert node.metadata["source"] == "test"
    assert node.node_id is not None
```

### Test Data

- Use **fake data** for testing (never real user data)
- **Clean up** test data after tests
- **Isolate tests** to avoid dependencies between tests

## Documentation

### Documentation Standards

1. **User Documentation**: Clear, example-driven guides
2. **API Documentation**: Complete API reference with examples
3. **Developer Documentation**: Architecture, setup, contributing guides
4. **Code Documentation**: Comprehensive docstrings

### Documentation Structure

```
docs/
├── README.md                 # Overview and navigation
├── user/                     # User-facing documentation
│   ├── configuration.md
│   └── troubleshooting.md
├── developer/                # Developer documentation
│   ├── architecture.md
│   └── setup_guide.md
├── api/                      # API documentation
│   └── api_reference.md
└── security/                 # Security documentation
    └── README.md
```

### Writing Documentation

1. **Clear Structure**: Use headings and table of contents
2. **Examples**: Include practical examples
3. **Cross-References**: Link to related documentation
4. **Up-to-Date**: Keep documentation current with code changes

## Security

### Security Considerations

When contributing security-related code:

1. **Follow security best practices**
2. **Add comprehensive tests**
3. **Update security documentation**
4. **Consider threat models**
5. **Review security implications**

### Reporting Security Issues

**Do not create public issues for security vulnerabilities.**

Report security issues to: @Celebr4tion on GitHub

See [SECURITY.md](SECURITY.md) for detailed security reporting guidelines.

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and community discussions
- **Pull Request Reviews**: Code review discussions

### Getting Help

1. **Documentation**: Check existing documentation first
2. **GitHub Discussions**: Ask questions in discussions
3. **GitHub Issues**: Create issues for bugs or feature requests
4. **Code Review**: Request review feedback during PR process

### Recognition

We recognize contributors through:

- **Contributor List**: Listed in README.md
- **Release Notes**: Mentioned in CHANGELOG.md
- **GitHub Recognition**: GitHub contributor statistics

## Release Process

### Versioning

We use [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: Backward-compatible functionality additions
- **PATCH**: Backward-compatible bug fixes

### Release Workflow

1. **Feature Development**: Features merged to development
2. **Release Branch**: Create release branch from development
3. **Testing**: Comprehensive testing of release candidate
4. **Documentation**: Update documentation and CHANGELOG.md
5. **Release**: Merge to main and tag release
6. **Distribution**: Publish release packages

## Questions?

If you have questions about contributing:

1. Check the [documentation](docs/)
2. Search [existing issues](https://github.com/Celebr4tion/memory-engine/issues)
3. Create a [new discussion](https://github.com/Celebr4tion/memory-engine/discussions)
4. Contact maintainers through GitHub

Thank you for contributing to Memory Engine! 🚀