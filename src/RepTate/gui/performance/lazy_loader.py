"""
Lazy Module Loader for deferred imports.

Provides a mechanism to defer import of heavy modules until first access,
reducing application startup time. Supports optional loading indicators
and graceful error handling.

Feature: 004-gui-performance-optimizations
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


class LazyModuleLoader:
    """Lazy loader for Python modules.

    Defers import of a module until first attribute access. This is useful
    for reducing startup time by not loading heavy modules until they're
    actually needed.

    Usage:
        loader = LazyModuleLoader('heavy.module.path', display_name='Heavy Module')

        # Module not loaded yet
        assert not loader.is_loaded

        # First attribute access triggers load
        SomeClass = loader.SomeClass  # Module loads here

        # Now it's loaded
        assert loader.is_loaded

    Attributes:
        module_path: Full import path of the module.
        display_name: Human-readable name for UI feedback.
        is_loaded: Whether the module has been loaded.
    """

    def __init__(
        self,
        module_path: str,
        display_name: str | None = None,
        on_loading: Callable[[], None] | None = None,
    ) -> None:
        """Initialize the lazy loader.

        Args:
            module_path: Full import path (e.g., 'RepTate.applications.ApplicationLVE').
            display_name: Human-readable name for loading indicator.
            on_loading: Callback invoked when loading starts.
        """
        self._module_path = module_path
        self._display_name = display_name or module_path.split(".")[-1]
        self._on_loading = on_loading
        self._module: Any | None = None
        self._load_error: ImportError | None = None

    @property
    def module_path(self) -> str:
        """Full import path of the module."""
        return self._module_path

    @property
    def display_name(self) -> str:
        """Human-readable name for UI feedback."""
        return self._display_name

    @property
    def is_loaded(self) -> bool:
        """Whether the module has been loaded."""
        return self._module is not None

    def load(self) -> Any:
        """Force immediate module load.

        Returns:
            The loaded module.

        Raises:
            ImportError: If the module cannot be imported.
        """
        if self._module is not None:
            return self._module

        if self._load_error is not None:
            raise self._load_error

        # Invoke loading callback if provided
        if self._on_loading is not None:
            try:
                self._on_loading()
            except Exception as e:
                logger.warning(f"Loading callback failed for {self._module_path}: {e}")

        try:
            logger.debug(f"Loading module: {self._module_path}")
            self._module = importlib.import_module(self._module_path)
            logger.debug(f"Successfully loaded: {self._module_path}")
            return self._module
        except ImportError as e:
            self._load_error = e
            logger.error(f"Failed to import {self._module_path}: {e}")
            raise

    def __getattr__(self, name: str) -> Any:
        """Lazy attribute access triggers module load.

        Args:
            name: The attribute name to access.

        Returns:
            The attribute from the loaded module.

        Raises:
            ImportError: If the module cannot be imported.
            AttributeError: If the attribute doesn't exist in the module.
        """
        # Avoid infinite recursion for special attributes
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        module = self.load()
        try:
            return getattr(module, name)
        except AttributeError:
            raise AttributeError(
                f"Module '{self._module_path}' has no attribute '{name}'"
            )

    def __repr__(self) -> str:
        """Return string representation."""
        status = "loaded" if self.is_loaded else "not loaded"
        return f"<LazyModuleLoader({self._module_path!r}, {status})>"


def create_lazy_loader(
    module_path: str,
    display_name: str | None = None,
    on_loading: Callable[[], None] | None = None,
) -> LazyModuleLoader:
    """Factory function to create a LazyModuleLoader.

    Args:
        module_path: Full import path.
        display_name: Human-readable name for UI feedback.
        on_loading: Callback invoked when loading starts.

    Returns:
        A configured LazyModuleLoader instance.
    """
    return LazyModuleLoader(module_path, display_name, on_loading)


__all__ = ["LazyModuleLoader", "create_lazy_loader"]
