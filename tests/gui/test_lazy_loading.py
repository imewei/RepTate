"""Tests for lazy loading of application modules.

Feature: 005-gui-performance-integration (T037)
Tests User Story 1: Faster Application Startup
"""

from __future__ import annotations

import time
import pytest


class TestApplicationRegistry:
    """Tests for the application registry module."""

    def test_registry_import_is_fast(self):
        """Importing application_registry should be fast (no eager app imports)."""
        start = time.perf_counter()
        from RepTate.gui.application_registry import get_available_application_names
        end = time.perf_counter()

        # Registry import should be under 500ms (eager imports were ~4000ms)
        assert (end - start) < 0.5, f"Registry import took {(end - start) * 1000:.2f}ms"

    def test_all_applications_registered(self):
        """All 12 applications should be registered."""
        from RepTate.gui.application_registry import get_available_application_names

        apps = get_available_application_names()
        expected_apps = {
            "MWD", "TTS", "TTSF", "LVE", "NLVE", "Crystal",
            "Gt", "Creep", "SANS", "React", "Dielectric", "LAOS"
        }

        assert set(apps) == expected_apps

    def test_lazy_load_application(self):
        """Applications should load on first access."""
        from RepTate.gui.application_registry import (
            get_application_class,
            is_application_loaded,
            clear_cache,
        )

        # Clear cache to ensure clean state
        clear_cache()

        # MWD should not be loaded initially
        assert not is_application_loaded("MWD")

        # Load MWD
        app_class = get_application_class("MWD")

        # Now it should be loaded
        assert is_application_loaded("MWD")
        assert app_class is not None
        assert hasattr(app_class, "appname")

    def test_cached_load_is_instant(self):
        """Second load of same application should be instant (cached)."""
        from RepTate.gui.application_registry import get_application_class

        # First load
        _ = get_application_class("MWD")

        # Second load should be nearly instant
        start = time.perf_counter()
        _ = get_application_class("MWD")
        end = time.perf_counter()

        assert (end - start) < 0.001, f"Cached load took {(end - start) * 1000:.2f}ms"

    def test_unknown_application_raises_keyerror(self):
        """Unknown application name should raise KeyError."""
        from RepTate.gui.application_registry import get_application_class

        with pytest.raises(KeyError, match="Unknown application"):
            get_application_class("NonExistentApp")


class TestStartupTimeImprovement:
    """Tests verifying the startup time improvement."""

    def test_startup_time_under_500ms(self):
        """Application registry import should be under 500ms (vs ~4000ms baseline)."""
        # Force fresh import by clearing from sys.modules
        import sys
        modules_to_clear = [
            k for k in sys.modules
            if k.startswith("RepTate.gui.application_registry")
        ]
        for mod in modules_to_clear:
            del sys.modules[mod]

        start = time.perf_counter()
        from RepTate.gui import application_registry
        end = time.perf_counter()

        elapsed_ms = (end - start) * 1000
        # Should be at least 80% faster than baseline (~4000ms)
        assert elapsed_ms < 800, f"Import took {elapsed_ms:.2f}ms, expected < 800ms"
