"""
Tests for FigureCache.

Feature: 004-gui-performance-optimizations
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from RepTate.gui.performance.figure_cache import (
    FigureCache,
    _reset_global_cache,
    get_figure_cache,
)


@pytest.fixture(autouse=True)
def reset_cache():
    """Reset the global cache before and after each test."""
    _reset_global_cache()
    yield
    _reset_global_cache()


@pytest.fixture
def mock_figure():
    """Create a mock matplotlib Figure."""
    figure = MagicMock()
    figure.number = 1
    return figure


@pytest.fixture
def figure_factory(mock_figure):
    """Create a factory that returns mock figures."""
    call_count = [0]

    def factory():
        call_count[0] += 1
        fig = MagicMock()
        fig.number = call_count[0]
        return fig

    return factory


class TestFigureCache:
    """Tests for FigureCache class."""

    def test_init_with_valid_max_size(self):
        """Test initialization with valid max_size."""
        cache = FigureCache(max_size=5)

        assert cache.max_size == 5
        assert cache.current_size == 0

    def test_init_with_invalid_max_size_raises_error(self):
        """Test initialization with invalid max_size raises ValueError."""
        with pytest.raises(ValueError, match="max_size must be at least 1"):
            FigureCache(max_size=0)

        with pytest.raises(ValueError, match="max_size must be at least 1"):
            FigureCache(max_size=-1)

    def test_cache_hit_returns_same_figure(self, figure_factory):
        """Test cache hit returns the same figure instance."""
        cache = FigureCache(max_size=5)

        fig1 = cache.get("key1", figure_factory)
        fig2 = cache.get("key1", figure_factory)

        assert fig1 is fig2
        assert cache.current_size == 1

    def test_cache_hit_moves_entry_to_most_recently_used(self, figure_factory):
        """Test cache hit moves entry to MRU position."""
        cache = FigureCache(max_size=3)

        # Add three figures
        cache.get("key1", figure_factory)
        cache.get("key2", figure_factory)
        cache.get("key3", figure_factory)

        # Access key1 (moves it to MRU)
        cache.get("key1", figure_factory)

        # Check order: key2 should be LRU now
        keys = list(cache._cache.keys())
        assert keys == ["key2", "key3", "key1"]

    def test_cache_miss_invokes_factory(self, figure_factory):
        """Test cache miss invokes the factory function."""
        cache = FigureCache(max_size=5)

        fig1 = cache.get("key1", figure_factory)
        fig2 = cache.get("key2", figure_factory)

        # Each should have different figure number from factory
        assert fig1.number == 1
        assert fig2.number == 2
        assert cache.current_size == 2

    @patch("RepTate.gui.performance.figure_cache.plt.close")
    def test_lru_eviction_when_at_capacity(self, mock_close, figure_factory):
        """Test LRU eviction when cache reaches capacity."""
        cache = FigureCache(max_size=2)

        fig1 = cache.get("key1", figure_factory)
        cache.get("key2", figure_factory)

        # Adding third should evict key1 (LRU)
        cache.get("key3", figure_factory)

        assert cache.current_size == 2
        assert "key1" not in cache._cache
        assert "key2" in cache._cache
        assert "key3" in cache._cache
        mock_close.assert_called_once_with(fig1)

    @patch("RepTate.gui.performance.figure_cache.plt.close")
    def test_evicted_figures_are_closed(self, mock_close, figure_factory):
        """Test evicted figures are properly closed with plt.close()."""
        cache = FigureCache(max_size=1)

        fig1 = cache.get("key1", figure_factory)
        cache.get("key2", figure_factory)

        mock_close.assert_called_once_with(fig1)

    @patch("RepTate.gui.performance.figure_cache.plt.close")
    def test_invalidate_removes_specific_entry(self, mock_close, figure_factory):
        """Test invalidate removes a specific entry."""
        cache = FigureCache(max_size=5)

        fig1 = cache.get("key1", figure_factory)
        cache.get("key2", figure_factory)

        cache.invalidate("key1")

        assert cache.current_size == 1
        assert "key1" not in cache._cache
        assert "key2" in cache._cache
        mock_close.assert_called_once_with(fig1)

    def test_invalidate_nonexistent_key_is_noop(self):
        """Test invalidate with nonexistent key does nothing."""
        cache = FigureCache(max_size=5)

        # Should not raise
        cache.invalidate("nonexistent")

        assert cache.current_size == 0

    @patch("RepTate.gui.performance.figure_cache.plt.close")
    def test_invalidate_prefix_removes_matching_entries(self, mock_close, figure_factory):
        """Test invalidate_prefix removes all entries with matching prefix."""
        cache = FigureCache(max_size=10)

        # Add figures with different prefixes
        cache.get("app1:dataset1", figure_factory)
        cache.get("app1:dataset2", figure_factory)
        cache.get("app2:dataset1", figure_factory)

        cache.invalidate_prefix("app1:")

        assert cache.current_size == 1
        assert "app1:dataset1" not in cache._cache
        assert "app1:dataset2" not in cache._cache
        assert "app2:dataset1" in cache._cache
        assert mock_close.call_count == 2

    def test_invalidate_prefix_no_matches_is_noop(self, figure_factory):
        """Test invalidate_prefix with no matches does nothing."""
        cache = FigureCache(max_size=5)

        cache.get("key1", figure_factory)

        cache.invalidate_prefix("nonexistent:")

        assert cache.current_size == 1

    @patch("RepTate.gui.performance.figure_cache.plt.close")
    def test_clear_removes_all_entries(self, mock_close, figure_factory):
        """Test clear removes all entries."""
        cache = FigureCache(max_size=10)

        cache.get("key1", figure_factory)
        cache.get("key2", figure_factory)
        cache.get("key3", figure_factory)

        cache.clear()

        assert cache.current_size == 0
        assert mock_close.call_count == 3

    def test_current_size_reflects_cache_state(self, figure_factory):
        """Test current_size accurately reflects cache state."""
        cache = FigureCache(max_size=5)

        assert cache.current_size == 0

        cache.get("key1", figure_factory)
        assert cache.current_size == 1

        cache.get("key2", figure_factory)
        assert cache.current_size == 2

        cache.invalidate("key1")
        assert cache.current_size == 1

        cache.clear()
        assert cache.current_size == 0


class TestGetFigureCache:
    """Tests for get_figure_cache singleton factory."""

    def test_singleton_returns_same_instance(self):
        """Test get_figure_cache returns the same instance."""
        cache1 = get_figure_cache()
        cache2 = get_figure_cache()

        assert cache1 is cache2

    def test_singleton_with_max_size(self):
        """Test singleton can be created with max_size."""
        cache = get_figure_cache(max_size=20)

        # Note: max_size only used on first creation
        assert cache.max_size == 20

    def test_singleton_max_size_ignored_after_creation(self):
        """Test max_size is ignored after first creation."""
        cache1 = get_figure_cache(max_size=5)
        cache2 = get_figure_cache(max_size=100)

        # Second call should return same instance with original max_size
        assert cache1 is cache2
        assert cache2.max_size == 5

    def test_default_max_size(self):
        """Test default max_size is 10."""
        cache = get_figure_cache()

        assert cache.max_size == 10
