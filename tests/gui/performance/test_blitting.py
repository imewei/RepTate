"""
Tests for BlittingManager.

Feature: 004-gui-performance-optimizations
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from RepTate.gui.performance.blitting import BlittingManager, create_blitting_manager


class TestBlittingManager:
    """Tests for BlittingManager class."""

    def test_initialization_with_mock_canvas(self, mock_canvas):
        """Test BlittingManager initializes correctly with mock canvas."""
        manager = BlittingManager(mock_canvas)

        assert manager.canvas is mock_canvas
        assert manager.is_active is False
        assert manager._supports_blit is True

    def test_initialization_detects_blit_support(self, mock_canvas):
        """Test that initialization detects blitting support from canvas."""
        mock_canvas.supports_blit = False
        manager = BlittingManager(mock_canvas)

        assert manager._supports_blit is False

    def test_start_blit_marks_artists_as_animated(self, mock_canvas, mock_artist):
        """Test start_blit marks provided artists as animated."""
        manager = BlittingManager(mock_canvas)
        artists = [mock_artist]

        manager.start_blit(artists)

        mock_artist.set_animated.assert_called_once_with(True)
        assert manager.is_active is True

    def test_start_blit_captures_background(self, mock_canvas, mock_artist):
        """Test start_blit captures canvas background."""
        manager = BlittingManager(mock_canvas)

        manager.start_blit([mock_artist])

        mock_canvas.draw.assert_called()
        mock_canvas.copy_from_bbox.assert_called()

    def test_start_blit_noop_when_blit_not_supported(self, mock_canvas, mock_artist):
        """Test start_blit does nothing when blitting not supported."""
        mock_canvas.supports_blit = False
        manager = BlittingManager(mock_canvas)

        manager.start_blit([mock_artist])

        mock_artist.set_animated.assert_not_called()
        assert manager.is_active is False

    def test_update_calls_correct_canvas_methods(self, mock_canvas, mock_artist):
        """Test update calls restore_region, draw_artist, and update."""
        manager = BlittingManager(mock_canvas)
        mock_artist.axes = MagicMock()

        manager.start_blit([mock_artist])
        # Reset mocks after start_blit calls
        mock_canvas.restore_region.reset_mock()
        mock_canvas.update.reset_mock()

        manager.update()

        mock_canvas.restore_region.assert_called_once()
        mock_artist.axes.draw_artist.assert_called_once_with(mock_artist)
        mock_canvas.update.assert_called_once()

    def test_update_noop_when_not_active(self, mock_canvas):
        """Test update does nothing when blitting is not active."""
        manager = BlittingManager(mock_canvas)

        manager.update()

        mock_canvas.restore_region.assert_not_called()
        mock_canvas.update.assert_not_called()

    def test_end_blit_restores_non_animated_state(self, mock_canvas, mock_artist):
        """Test end_blit restores artists to non-animated state."""
        manager = BlittingManager(mock_canvas)

        manager.start_blit([mock_artist])
        mock_artist.set_animated.reset_mock()

        manager.end_blit()

        mock_artist.set_animated.assert_called_once_with(False)
        assert manager.is_active is False

    def test_end_blit_performs_full_redraw(self, mock_canvas, mock_artist):
        """Test end_blit triggers a full canvas redraw."""
        manager = BlittingManager(mock_canvas)

        manager.start_blit([mock_artist])
        mock_canvas.draw.reset_mock()

        manager.end_blit()

        mock_canvas.draw.assert_called_once()

    def test_end_blit_noop_when_not_active(self, mock_canvas):
        """Test end_blit does nothing when not active."""
        manager = BlittingManager(mock_canvas)

        manager.end_blit()

        # draw should not be called if we never started
        # (only the initial draw from start_blit would happen)
        mock_canvas.draw.assert_not_called()

    def test_fallback_behavior_when_blit_not_supported(self, mock_canvas, mock_artist):
        """Test graceful fallback when supports_blit is False."""
        mock_canvas.supports_blit = False
        manager = BlittingManager(mock_canvas)

        # These should all be no-ops
        manager.start_blit([mock_artist])
        manager.update()
        manager.end_blit()

        # Artist should never be marked as animated
        mock_artist.set_animated.assert_not_called()

    def test_resize_event_recaptures_background(self, mock_canvas, mock_artist):
        """Test that draw events recapture background when active."""
        manager = BlittingManager(mock_canvas)

        manager.start_blit([mock_artist])
        mock_canvas.copy_from_bbox.reset_mock()

        # Simulate a draw event (e.g., from resize)
        manager._on_draw(MagicMock())

        mock_canvas.copy_from_bbox.assert_called_once()

    def test_resize_event_ignored_when_not_active(self, mock_canvas):
        """Test that draw events are ignored when not active."""
        manager = BlittingManager(mock_canvas)

        manager._on_draw(MagicMock())

        mock_canvas.copy_from_bbox.assert_not_called()

    def test_frame_skip_logic_skips_rapid_updates(self, mock_canvas, mock_artist):
        """Test frame-skip logic prevents rapid successive updates."""
        manager = BlittingManager(mock_canvas)
        mock_artist.axes = MagicMock()

        manager.start_blit([mock_artist])

        # First update should succeed
        manager.update()
        update_count_1 = mock_canvas.update.call_count

        # Immediate second update should be skipped
        manager.update()
        update_count_2 = mock_canvas.update.call_count

        # Only one update should have happened
        # (Note: start_blit doesn't call update, only draw)
        assert update_count_2 == update_count_1

    def test_frame_skip_allows_updates_after_interval(self, mock_canvas, mock_artist):
        """Test frame-skip allows updates after minimum interval."""
        manager = BlittingManager(mock_canvas)
        mock_artist.axes = MagicMock()

        manager.start_blit([mock_artist])

        # First update
        manager.update()

        # Manually set last update time to past
        manager._last_update_time = time.perf_counter() * 1000 - 20  # 20ms ago

        # This update should be allowed
        mock_canvas.update.reset_mock()
        manager.update()

        mock_canvas.update.assert_called_once()

    def test_disconnect_removes_event_handler(self, mock_canvas):
        """Test disconnect removes the draw event handler."""
        manager = BlittingManager(mock_canvas)
        original_cid = manager._draw_cid

        manager.disconnect()

        mock_canvas.mpl_disconnect.assert_called_once_with(original_cid)
        assert manager._draw_cid is None


class TestCreateBlittingManager:
    """Tests for create_blitting_manager factory function."""

    def test_factory_creates_manager(self, mock_canvas):
        """Test factory function creates a BlittingManager instance."""
        manager = create_blitting_manager(mock_canvas)

        assert isinstance(manager, BlittingManager)
        assert manager.canvas is mock_canvas

    def test_factory_returns_new_instance_each_call(self, mock_canvas):
        """Test factory creates new instance on each call."""
        manager1 = create_blitting_manager(mock_canvas)
        manager2 = create_blitting_manager(mock_canvas)

        assert manager1 is not manager2
