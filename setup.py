"""
Setup configuration for Memory Engine package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements from requirements.txt
requirements = []
try:
    with open('requirements.txt') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
except FileNotFoundError:
    pass

setup(
    name="memory-engine",
    version="0.1.0",
    author="Memory Engine Team",
    author_email="contact@memory-engine.dev",
    description="A comprehensive knowledge graph and memory management system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/memory-engine/memory-engine",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Database :: Database Engines/Servers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
        "mcp": [
            "mcp>=0.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "memory-engine=memory_core.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "memory_core": [
            "config/*.yaml",
            "config/environments/*.yaml",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/memory-engine/memory-engine/issues",
        "Source": "https://github.com/memory-engine/memory-engine",
        "Documentation": "https://memory-engine.readthedocs.io/",
    },
    keywords="knowledge-graph memory-management ai llm embeddings vector-database graph-database",
)