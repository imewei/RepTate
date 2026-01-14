"""Tests for figure cache integration.

Feature: 005-gui-performance-integration (T040)
Tests User Story 4: Efficient Dataset Switching
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


class TestFigureCache:
    """Tests for FigureCache functionality."""

    def test_cache_creation(self):
        """FigureCache should be creatable with max_size."""
        from RepTate.gui.performance import FigureCache

        cache = FigureCache(max_size=10)
        assert cache.max_size == 10
        assert cache.current_size == 0

    def test_cache_get_creates_on_miss(self):
        """Cache get should create figure on miss."""
        from RepTate.gui.performance import FigureCache

        cache = FigureCache(max_size=5)
        figure = MagicMock()
        factory = MagicMock(return_value=figure)

        result = cache.get("test_key", factory)

        factory.assert_called_once()
        assert result is figure
        assert cache.current_size == 1

    def test_cache_get_returns_cached_on_hit(self):
        """Cache get should return cached figure on hit."""
        from RepTate.gui.performance import FigureCache

        cache = FigureCache(max_size=5)
        figure = MagicMock()
        factory = MagicMock(return_value=figure)

        # First call - cache miss
        cache.get("test_key", factory)

        # Second call - cache hit (factory should NOT be called again)
        factory.reset_mock()
        result = cache.get("test_key", factory)

        factory.assert_not_called()
        assert result is figure

    def test_cache_lru_eviction(self):
        """Cache should evict LRU items when full."""
        from RepTate.gui.performance import FigureCache

        cache = FigureCache(max_size=2)

        fig1 = MagicMock()
        fig2 = MagicMock()
        fig3 = MagicMock()

        cache.get("key1", lambda: fig1)
        cache.get("key2", lambda: fig2)

        # Cache is full (size 2)
        assert cache.current_size == 2

        # Adding third should evict first (LRU)
        cache.get("key3", lambda: fig3)

        assert cache.current_size == 2

    def test_cache_invalidate(self):
        """invalidate should remove specific key."""
        from RepTate.gui.performance import FigureCache

        cache = FigureCache(max_size=5)
        figure = MagicMock()

        cache.get("test_key", lambda: figure)
        assert cache.current_size == 1

        cache.invalidate("test_key")
        assert cache.current_size == 0

    def test_cache_invalidate_prefix(self):
        """invalidate_prefix should remove all matching keys."""
        from RepTate.gui.performance import FigureCache

        cache = FigureCache(max_size=10)

        # Add several figures with common prefix
        cache.get("app1:ds1", lambda: MagicMock())
        cache.get("app1:ds2", lambda: MagicMock())
        cache.get("app2:ds1", lambda: MagicMock())

        assert cache.current_size == 3

        # Invalidate all app1 figures
        cache.invalidate_prefix("app1:")

        assert cache.current_size == 1

    def test_cache_clear(self):
        """clear should remove all entries."""
        from RepTate.gui.performance import FigureCache

        cache = FigureCache(max_size=5)

        cache.get("key1", lambda: MagicMock())
        cache.get("key2", lambda: MagicMock())

        cache.clear()

        assert cache.current_size == 0


class TestCacheKeyGeneration:
    """Tests for cache key generation in QApplicationWindow."""

    def test_cache_key_method_exists(self):
        """QApplicationWindow should have _get_cache_key method."""
        from RepTate.gui.QApplicationWindow import QApplicationWindow

        assert hasattr(QApplicationWindow, "_get_cache_key")

    def test_cache_key_format(self):
        """Cache key should follow {app_name}:{dataset_id} format."""
        from RepTate.gui.QApplicationWindow import QApplicationWindow

        # Create a simple mock object with the required attribute
        class MockWindow:
            appname = "TestApp"

        mock = MockWindow()

        # Call the method using the unbound function pattern
        key = QApplicationWindow._get_cache_key(mock, "dataset123")

        assert key == "TestApp:dataset123"
        assert ":" in key
