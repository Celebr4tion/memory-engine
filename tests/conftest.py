"""
Pytest configuration file.
"""
import pytest
import sys
import collections.abc

# Python 3.13 removed collections.MutableMapping
# This patch fixes libraries that still use it
if not hasattr(collections, 'MutableMapping'):
    collections.MutableMapping = collections.abc.MutableMapping


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "integration: mark a test as an integration test")