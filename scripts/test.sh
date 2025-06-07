#!/bin/bash
# Memory Engine Test Runner Script
# Provides convenient commands for running different types of tests

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_color() {
    printf "${1}${2}${NC}\n"
}

print_usage() {
    echo "Memory Engine Test Runner"
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  all         Run all tests"
    echo "  unit        Run unit tests only (fast)"
    echo "  integration Run integration tests only (requires services)"
    echo "  component   Run component tests only"
    echo "  coverage    Run tests with coverage report"
    echo "  lint        Run linting checks"
    echo "  format      Format code with black"
    echo "  check       Run all checks (lint + tests)"
    echo ""
    echo "Options:"
    echo "  -v, --verbose    Verbose output"
    echo "  -x, --stop       Stop on first failure"
    echo "  --file FILE      Run specific test file"
    echo ""
    echo "Examples:"
    echo "  $0 unit -v              # Run unit tests with verbose output"
    echo "  $0 integration -x       # Run integration tests, stop on failure"
    echo "  $0 --file test_config   # Run specific test file"
    echo "  $0 coverage             # Run tests with coverage"
}

# Check if virtual environment is activated
check_venv() {
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        print_color $YELLOW "Warning: Virtual environment not activated"
        print_color $YELLOW "Consider running: source .venv/bin/activate"
        echo ""
    fi
}

# Run unit tests
run_unit_tests() {
    print_color $BLUE "Running unit tests..."
    pytest tests/unit/ "$@"
}

# Run integration tests
run_integration_tests() {
    print_color $BLUE "Running integration tests..."
    print_color $YELLOW "Note: Requires JanusGraph and Milvus services"
    pytest tests/integration/ "$@"
}

# Run component tests
run_component_tests() {
    print_color $BLUE "Running component tests..."
    pytest tests/test_*.py "$@"
}

# Run all tests
run_all_tests() {
    print_color $BLUE "Running all tests..."
    pytest tests/ "$@"
}

# Run tests with coverage
run_coverage() {
    print_color $BLUE "Running tests with coverage..."
    pytest --cov=memory_core --cov-report=html --cov-report=term tests/ "$@"
    print_color $GREEN "Coverage report generated in htmlcov/"
}

# Run linting
run_lint() {
    print_color $BLUE "Running linting checks..."
    
    if command -v flake8 &> /dev/null; then
        print_color $BLUE "Running flake8..."
        flake8 memory_core/ tests/ examples/ --max-line-length=100
    fi
    
    if command -v mypy &> /dev/null; then
        print_color $BLUE "Running mypy..."
        mypy memory_core/ --ignore-missing-imports
    fi
}

# Format code
run_format() {
    print_color $BLUE "Formatting code..."
    
    if command -v black &> /dev/null; then
        black memory_core/ tests/ examples/ scripts/
        print_color $GREEN "Code formatted with black"
    else
        print_color $RED "black not installed. Install with: pip install black"
        exit 1
    fi
}

# Run all checks
run_check() {
    print_color $BLUE "Running all checks..."
    run_lint
    run_unit_tests "$@"
    print_color $GREEN "All checks passed!"
}

# Parse command line arguments
check_venv

case "${1:-all}" in
    all)
        shift
        run_all_tests "$@"
        ;;
    unit)
        shift
        run_unit_tests "$@"
        ;;
    integration)
        shift
        run_integration_tests "$@"
        ;;
    component)
        shift
        run_component_tests "$@"
        ;;
    coverage)
        shift
        run_coverage "$@"
        ;;
    lint)
        run_lint
        ;;
    format)
        run_format
        ;;
    check)
        shift
        run_check "$@"
        ;;
    --file)
        if [ -z "$2" ]; then
            print_color $RED "Error: --file requires a filename"
            exit 1
        fi
        shift 2
        pytest "tests/**/*${1}*" "$@"
        ;;
    -h|--help|help)
        print_usage
        ;;
    *)
        print_color $RED "Unknown command: $1"
        print_usage
        exit 1
        ;;
esac