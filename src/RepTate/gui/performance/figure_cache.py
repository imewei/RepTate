"""
Figure Cache for efficient matplotlib figure memory management.

Provides an LRU (Least Recently Used) cache for matplotlib figures to
reduce memory allocation overhead when switching between datasets.
Ensures proper cleanup by calling plt.close() on evicted figures.

Feature: 004-gui-performance-optimizations
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from RepTate.gui.performance._types import FigureFactory

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# Singleton instance
_global_cache: FigureCache | None = None


class FigureCache:
    """LRU cache for matplotlib figures.

    Caches figures by key to avoid recreating them when switching between
    datasets. Uses OrderedDict for O(1) LRU tracking. When the cache
    reaches capacity, the least recently used figure is evicted and
    properly closed with plt.close().

    Usage:
        cache = get_figure_cache(max_size=10)

        # Get or create a figure
        fig = cache.get("dataset_1", lambda: plt.figure())

        # Invalidate when dataset is deleted
        cache.invalidate("dataset_1")

        # Invalidate all figures for an application
        cache.invalidate_prefix("app_name:")

    Attributes:
        max_size: Maximum number of figures to cache.
        current_size: Current number of cached figures.
    """

    def __init__(self, max_size: int = 10) -> None:
        """Initialize the figure cache.

        Args:
            max_size: Maximum number of figures to cache (default: 10).
        """
        if max_size < 1:
            raise ValueError("max_size must be at least 1")

        self._max_size = max_size
        self._cache: OrderedDict[str, Figure] = OrderedDict()

    @property
    def max_size(self) -> int:
        """Maximum number of figures to cache."""
        return self._max_size

    @property
    def current_size(self) -> int:
        """Current number of cached figures."""
        return len(self._cache)

    def get(self, key: str, factory: FigureFactory) -> Figure:
        """Get a cached figure or create a new one.

        If the key exists in the cache, moves it to the most recently
        used position and returns it. Otherwise, creates a new figure
        using the factory, caches it, and returns it.

        Args:
            key: Unique identifier for the figure.
            factory: Callable that creates a new Figure if needed.

        Returns:
            The cached or newly created Figure.
        """
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            logger.debug(f"Cache hit for figure: {key}")
            return self._cache[key]

        # Cache miss - create new figure
        logger.debug(f"Cache miss for figure: {key}")
        figure = factory()

        # Evict LRU if at capacity
        if len(self._cache) >= self._max_size:
            self._evict_lru()

        self._cache[key] = figure
        return figure

    def invalidate(self, key: str) -> None:
        """Remove a specific figure from the cache.

        If the key exists, the figure is closed and removed from cache.
        No-op if the key doesn't exist.

        Args:
            key: Unique identifier for the figure to remove.
        """
        if key in self._cache:
            figure = self._cache.pop(key)
            self._close_figure(figure)
            logger.debug(f"Invalidated figure: {key}")

    def invalidate_prefix(self, prefix: str) -> None:
        """Remove all figures with keys starting with the given prefix.

        Useful for clearing all figures associated with an application
        or dataset group.

        Args:
            prefix: Key prefix to match.
        """
        keys_to_remove = [k for k in self._cache if k.startswith(prefix)]
        for key in keys_to_remove:
            self.invalidate(key)

        if keys_to_remove:
            logger.debug(f"Invalidated {len(keys_to_remove)} figures with prefix: {prefix}")

    def clear(self) -> None:
        """Remove all figures from the cache.

        All cached figures are properly closed with plt.close().
        """
        for figure in self._cache.values():
            self._close_figure(figure)

        self._cache.clear()
        logger.debug("Cleared all cached figures")

    def _evict_lru(self) -> None:
        """Evict the least recently used figure from the cache."""
        if self._cache:
            key, figure = self._cache.popitem(last=False)
            self._close_figure(figure)
            logger.debug(f"Evicted LRU figure: {key}")

    def _close_figure(self, figure: Figure) -> None:
        """Close a matplotlib figure to free memory.

        Args:
            figure: The figure to close.
        """
        try:
            plt.close(figure)
        except Exception as e:
            logger.warning(f"Error closing figure: {e}")


def get_figure_cache(max_size: int = 10) -> FigureCache:
    """Get or create the global figure cache.

    Args:
        max_size: Maximum cache size. Only used when creating the
            cache for the first time.

    Returns:
        Singleton FigureCache instance.
    """
    global _global_cache

    if _global_cache is None:
        _global_cache = FigureCache(max_size)

    return _global_cache


def _reset_global_cache() -> None:
    """Reset the global cache (for testing only)."""
    global _global_cache
    if _global_cache is not None:
        _global_cache.clear()
    _global_cache = None


__all__ = ["FigureCache", "get_figure_cache"]
