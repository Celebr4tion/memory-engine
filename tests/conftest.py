"""
Pytest configuration file.
"""
import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "integration: mark a test as an integration test") 