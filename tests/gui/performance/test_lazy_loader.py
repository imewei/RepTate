"""
Tests for LazyModuleLoader.

Feature: 004-gui-performance-optimizations
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from RepTate.gui.performance.lazy_loader import LazyModuleLoader, create_lazy_loader


class TestLazyModuleLoader:
    """Tests for LazyModuleLoader class."""

    def test_module_not_loaded_until_first_access(self):
        """Test module is not loaded until first attribute access."""
        loader = LazyModuleLoader("os.path")

        assert loader.is_loaded is False

    def test_is_loaded_returns_correct_state(self):
        """Test is_loaded returns correct state before and after load."""
        loader = LazyModuleLoader("os.path")

        assert loader.is_loaded is False

        # Access an attribute to trigger load
        _ = loader.join

        assert loader.is_loaded is True

    def test_load_forces_immediate_import(self):
        """Test load() forces immediate import."""
        loader = LazyModuleLoader("os.path")

        module = loader.load()

        assert loader.is_loaded is True
        assert module is not None
        assert hasattr(module, "join")

    def test_getattr_returns_correct_module_attribute(self):
        """Test __getattr__ returns correct module attribute."""
        loader = LazyModuleLoader("os.path")

        join_func = loader.join

        import os.path

        assert join_func is os.path.join

    def test_on_loading_callback_invoked_before_import(self):
        """Test on_loading callback is invoked before import."""
        callback = MagicMock()
        loader = LazyModuleLoader("os.path", on_loading=callback)

        loader.load()

        callback.assert_called_once()

    def test_on_loading_callback_only_called_once(self):
        """Test on_loading callback is only called on first load."""
        callback = MagicMock()
        loader = LazyModuleLoader("os.path", on_loading=callback)

        loader.load()
        loader.load()

        callback.assert_called_once()

    def test_import_error_caught_and_reported(self):
        """Test ImportError is caught and reported gracefully."""
        loader = LazyModuleLoader("nonexistent.module.path")

        with pytest.raises(ImportError) as exc_info:
            loader.load()

        assert "nonexistent" in str(exc_info.value)

    def test_import_error_cached_and_reraises(self):
        """Test ImportError is cached and re-raised on subsequent calls."""
        loader = LazyModuleLoader("nonexistent.module.path")

        with pytest.raises(ImportError):
            loader.load()

        # Second call should raise the same error
        with pytest.raises(ImportError):
            loader.load()

    def test_module_cached_after_first_load(self):
        """Test module is cached after first load."""
        loader = LazyModuleLoader("os.path")

        module1 = loader.load()
        module2 = loader.load()

        assert module1 is module2

    def test_attribute_access_caches_module(self):
        """Test attribute access caches the module."""
        loader = LazyModuleLoader("os.path")

        _ = loader.join
        _ = loader.dirname

        # Both should use the same cached module
        assert loader.is_loaded is True

    def test_module_path_property(self):
        """Test module_path property returns correct value."""
        loader = LazyModuleLoader("some.module.path")

        assert loader.module_path == "some.module.path"

    def test_display_name_defaults_to_module_name(self):
        """Test display_name defaults to last part of module path."""
        loader = LazyModuleLoader("some.module.path")

        assert loader.display_name == "path"

    def test_display_name_can_be_overridden(self):
        """Test display_name can be explicitly set."""
        loader = LazyModuleLoader("os.path", display_name="Path Utilities")

        assert loader.display_name == "Path Utilities"

    def test_repr_shows_load_status(self):
        """Test __repr__ shows correct load status."""
        loader = LazyModuleLoader("os.path")

        assert "not loaded" in repr(loader)

        loader.load()

        assert "loaded" in repr(loader)

    def test_private_attributes_raise_attribute_error(self):
        """Test accessing private attributes raises AttributeError."""
        loader = LazyModuleLoader("os.path")

        with pytest.raises(AttributeError):
            _ = loader._private_attr

    def test_nonexistent_attribute_raises_attribute_error(self):
        """Test accessing nonexistent attribute raises AttributeError."""
        loader = LazyModuleLoader("os.path")

        with pytest.raises(AttributeError) as exc_info:
            _ = loader.nonexistent_function

        assert "nonexistent_function" in str(exc_info.value)

    def test_callback_failure_does_not_prevent_load(self):
        """Test callback failure doesn't prevent module loading."""

        def failing_callback():
            raise ValueError("Callback failed")

        loader = LazyModuleLoader("os.path", on_loading=failing_callback)

        # Should still load despite callback failure
        module = loader.load()
        assert module is not None


class TestCreateLazyLoader:
    """Tests for create_lazy_loader factory function."""

    def test_factory_creates_loader(self):
        """Test factory function creates a LazyModuleLoader instance."""
        loader = create_lazy_loader("os.path")

        assert isinstance(loader, LazyModuleLoader)
        assert loader.module_path == "os.path"

    def test_factory_passes_display_name(self):
        """Test factory passes display_name to loader."""
        loader = create_lazy_loader("os.path", display_name="Path Module")

        assert loader.display_name == "Path Module"

    def test_factory_passes_on_loading_callback(self):
        """Test factory passes on_loading callback to loader."""
        callback = MagicMock()
        loader = create_lazy_loader("os.path", on_loading=callback)

        loader.load()

        callback.assert_called_once()
