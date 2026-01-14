"""Tests for blitting manager integration.

Feature: 005-gui-performance-integration (T039)
Tests User Story 3: Responsive Interactive Plotting
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


class TestBlittingManager:
    """Tests for BlittingManager functionality."""

    def test_blitting_manager_creation(self):
        """BlittingManager should be creatable with a canvas."""
        from RepTate.gui.performance import create_blitting_manager

        canvas = MagicMock()
        canvas.supports_blit = True

        manager = create_blitting_manager(canvas)
        assert manager is not None

    def test_blitting_manager_supports_blit_detection(self):
        """BlittingManager should detect if backend supports blitting."""
        from RepTate.gui.performance.blitting import BlittingManager

        canvas = MagicMock()
        canvas.supports_blit = True

        manager = BlittingManager(canvas)
        assert manager._supports_blit is True

    def test_blitting_manager_fallback_mode(self):
        """BlittingManager should fallback gracefully if blitting not supported."""
        from RepTate.gui.performance.blitting import BlittingManager

        canvas = MagicMock()
        canvas.supports_blit = False

        manager = BlittingManager(canvas)
        assert manager._supports_blit is False

        # Should not fail even when blitting not supported
        manager.start_blit([])
        manager.update()
        manager.end_blit()

    def test_start_blit_captures_background(self):
        """start_blit should capture the canvas background."""
        from RepTate.gui.performance.blitting import BlittingManager

        canvas = MagicMock()
        canvas.supports_blit = True

        manager = BlittingManager(canvas)

        artist = MagicMock()
        manager.start_blit([artist])

        # Background should be captured
        canvas.copy_from_bbox.assert_called()

    def test_end_blit_triggers_draw(self):
        """end_blit should trigger a full canvas draw."""
        from RepTate.gui.performance.blitting import BlittingManager

        canvas = MagicMock()
        canvas.supports_blit = True

        manager = BlittingManager(canvas)
        manager.start_blit([])
        manager.end_blit()

        canvas.draw.assert_called()

    def test_disconnect_removes_handlers(self):
        """disconnect should clean up event handlers."""
        from RepTate.gui.performance.blitting import BlittingManager

        canvas = MagicMock()
        canvas.supports_blit = True

        manager = BlittingManager(canvas)
        manager.disconnect()

        # Manager should be in clean state
        assert manager._background is None
        assert manager._active is False


class TestBlittingIntegration:
    """Integration tests for blitting in QApplicationWindow."""

    def test_animated_artists_helper_exists(self):
        """QApplicationWindow should have _get_animated_artists method."""
        # This is a documentation test - the method exists in QApplicationWindow
        from RepTate.gui.QApplicationWindow import QApplicationWindow

        assert hasattr(QApplicationWindow, "_get_animated_artists")

    def test_blit_mouse_handlers_exist(self):
        """QApplicationWindow should have blitting mouse event handlers."""
        from RepTate.gui.QApplicationWindow import QApplicationWindow

        assert hasattr(QApplicationWindow, "_on_blit_mouse_press")
        assert hasattr(QApplicationWindow, "_on_blit_mouse_release")
