#!/bin/bash
# Memory Engine Setup Script
# Sets up development environment and dependencies

set -e

echo "Setting up Memory Engine development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.8+ required, found $python_version"
    exit 1
fi

echo "✓ Python $python_version detected"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Set up pre-commit hooks (if available)
if command -v pre-commit &> /dev/null; then
    echo "Setting up pre-commit hooks..."
    pre-commit install
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p logs
mkdir -p data

# Copy environment template if .env doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env template..."
    cat > .env << EOF
# Memory Engine Environment Variables
# Copy this file and set your actual values

# Required: Gemini API Key
GEMINI_API_KEY=your-gemini-api-key-here

# Optional: Override default configuration
# ENVIRONMENT=development
# LOG_LEVEL=INFO

# Database Configuration
# DATABASE_URL=sqlite:///memory_engine.db

# JanusGraph Configuration  
# JANUSGRAPH_HOST=localhost
# JANUSGRAPH_PORT=8182

# Milvus Configuration
# MILVUS_HOST=localhost
# MILVUS_PORT=19530
EOF
    echo "Created .env template - please update with your API keys"
fi

echo ""
echo "✓ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Update .env file with your API keys"
echo "2. Start services: docker-compose up -d (optional, for JanusGraph/Milvus)"
echo "3. Run tests: pytest"
echo "4. Try examples: python examples/config_example.py"
echo ""
echo "For more information, see docs/setup_guide.md"