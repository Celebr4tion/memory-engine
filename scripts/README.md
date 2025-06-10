# Scripts Directory

This directory contains utility scripts for development, testing, and deployment.

## Available Scripts

### setup.sh
Development environment setup script.

```bash
./scripts/setup.sh
```

Features:
- Checks Python version compatibility
- Creates virtual environment
- Installs dependencies
- Sets up pre-commit hooks
- Creates .env template
- Creates necessary directories

### test.sh
Comprehensive test runner with multiple options.

```bash
./scripts/test.sh [command] [options]
```

Commands:
- `all` - Run all tests
- `unit` - Run unit tests only (fast)
- `integration` - Run integration tests (requires services)
- `component` - Run component tests
- `coverage` - Run tests with coverage report
- `lint` - Run linting checks
- `format` - Format code with black
- `check` - Run all checks (lint + tests)

Examples:
```bash
./scripts/test.sh unit -v              # Unit tests with verbose output
./scripts/test.sh integration -x       # Integration tests, stop on failure
./scripts/test.sh --file test_config_manager.py  # Run specific test file
./scripts/test.sh coverage             # Tests with coverage report
```

## Usage

Make scripts executable first:
```bash
chmod +x scripts/*.sh
```

Then run them from the project root:
```bash
./scripts/setup.sh
./scripts/test.sh unit
```

## Requirements

- Bash shell
- Python 3.8+
- Virtual environment recommended

Optional tools for enhanced functionality:
- `black` for code formatting
- `flake8` for linting
- `mypy` for type checking
- `pytest-cov` for coverage reports
- `pre-commit` for git hooks