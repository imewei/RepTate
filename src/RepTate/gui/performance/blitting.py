"""
Matplotlib Blitting Manager for fast interactive plot updates.

Provides background caching and selective artist redraw for interactive
operations like zoom, pan, and drag. Uses canvas.update() instead of
canvas.blit() to avoid memory leaks in Qt backends.

Feature: 004-gui-performance-optimizations
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.artist import Artist
    from matplotlib.backend_bases import DrawEvent
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg


class BlittingManager:
    """Manager for matplotlib blitting operations.

    Manages background caching and selective artist redraw for interactive
    operations like zoom, pan, and drag. Automatically falls back to full
    redraws on backends that don't support blitting.

    Usage:
        blit_manager = BlittingManager(canvas)

        # On mouse press (start interaction)
        blit_manager.start_blit([zoom_rect, cursor])

        # On mouse drag (during interaction)
        blit_manager.update()

        # On mouse release (end interaction)
        blit_manager.end_blit()

    Attributes:
        canvas: The matplotlib FigureCanvasQTAgg being managed.
        is_active: Whether blitting is currently active.
    """

    # Minimum interval between frame updates (ms) for frame-skip logic
    _MIN_FRAME_INTERVAL_MS: float = 16.0  # ~60 FPS max

    def __init__(self, canvas: FigureCanvasQTAgg) -> None:
        """Initialize the blitting manager.

        Args:
            canvas: The matplotlib canvas to manage.
        """
        self._canvas = canvas
        self._background: Any | None = None
        self._animated_artists: list[Artist] = []
        self._active: bool = False
        self._supports_blit: bool = getattr(canvas, "supports_blit", True)
        self._last_update_time: float = 0.0

        # Connect to draw event for resize handling
        self._draw_cid = canvas.mpl_connect("draw_event", self._on_draw)

    @property
    def canvas(self) -> FigureCanvasQTAgg:
        """The matplotlib canvas being managed."""
        return self._canvas

    @property
    def is_active(self) -> bool:
        """Whether blitting is currently active."""
        return self._active

    def start_blit(self, artists: Sequence[Artist]) -> None:
        """Begin blitting operation.

        Caches the background and marks artists as animated for selective
        redraw during interactive operations.

        Args:
            artists: Artists that will be redrawn during animation.
        """
        if not self._supports_blit:
            return

        self._animated_artists = list(artists)
        self._active = True
        self._last_update_time = 0.0

        # Mark artists as animated
        for artist in self._animated_artists:
            artist.set_animated(True)

        # Draw everything to get a clean background, then cache it
        self._canvas.draw()
        self._capture_background()

    def update(self) -> None:
        """Perform blitting update.

        Restores background, redraws animated artists, and updates canvas.
        Must be called between start_blit() and end_blit().

        If blitting is not supported or not active, this is a no-op.
        Implements frame-skip logic to avoid overwhelming the render pipeline
        during rapid input events.
        """
        if not self._active or not self._supports_blit:
            return

        # Frame-skip logic: skip update if too soon after last one
        if self._should_skip_frame():
            return

        if self._background is None:
            return

        # Restore the background
        self._canvas.restore_region(self._background)

        # Redraw only the animated artists
        for artist in self._animated_artists:
            if artist.axes is not None:
                artist.axes.draw_artist(artist)

        # Use canvas.update() instead of canvas.blit() to avoid memory leak
        # in Qt backends (see matplotlib issue #10949)
        self._canvas.update()

        self._last_update_time = time.perf_counter() * 1000

    def end_blit(self) -> None:
        """End blitting operation.

        Restores artists to non-animated state and performs full redraw.
        """
        if not self._active:
            return

        self._active = False

        # Restore artists to non-animated state
        for artist in self._animated_artists:
            artist.set_animated(False)

        self._animated_artists = []
        self._background = None

        # Perform full redraw to finalize
        self._canvas.draw()

    def _capture_background(self) -> None:
        """Capture the current canvas state as background."""
        if self._canvas.figure is not None:
            self._background = self._canvas.copy_from_bbox(
                self._canvas.figure.bbox
            )

    def _on_draw(self, event: DrawEvent) -> None:
        """Handle draw events to recapture background after resize.

        Args:
            event: The matplotlib draw event.
        """
        if self._active and self._supports_blit:
            # Recapture background after resize
            self._capture_background()

    def _should_skip_frame(self) -> bool:
        """Check if the current frame should be skipped.

        Implements frame-skip logic to handle rapid zoom/pan operations
        that generate events faster than the render pipeline can keep up.

        Returns:
            True if this frame should be skipped, False otherwise.
        """
        if self._last_update_time == 0.0:
            return False

        current_time = time.perf_counter() * 1000
        elapsed = current_time - self._last_update_time
        return elapsed < self._MIN_FRAME_INTERVAL_MS

    def disconnect(self) -> None:
        """Disconnect event handlers.

        Call this when the manager is no longer needed to prevent memory leaks.
        """
        if self._draw_cid is not None:
            self._canvas.mpl_disconnect(self._draw_cid)
            self._draw_cid = None


def create_blitting_manager(canvas: FigureCanvasQTAgg) -> BlittingManager:
    """Factory function to create a BlittingManager.

    Args:
        canvas: The matplotlib canvas to manage.

    Returns:
        A configured BlittingManager instance.
    """
    return BlittingManager(canvas)


__all__ = ["BlittingManager", "create_blitting_manager"]
