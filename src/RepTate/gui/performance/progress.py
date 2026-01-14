"""
Progress Reporter for thread-safe progress updates.

Provides a mechanism for worker threads to safely update UI progress
indicators via Qt signals. Supports both determinate (percentage) and
indeterminate (busy spinner) modes.

Feature: 004-gui-performance-optimizations
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QTimer, Signal, Slot

from RepTate.gui.performance._signals import BaseWorkerSignals

if TYPE_CHECKING:
    from PySide6.QtWidgets import QProgressBar

logger = logging.getLogger(__name__)


class ProgressSignals(BaseWorkerSignals):
    """Signals for progress reporting.

    Extends BaseWorkerSignals with progress-specific signals.

    Signals:
        finished: Emitted when the operation completes (inherited).
        error: Emitted when an error occurs (inherited).
        progress: Emitted with progress value (0-100).
        status: Emitted with status message.
        indeterminate: Emitted to switch between modes.
    """

    progress = Signal(int)
    status = Signal(str)
    indeterminate = Signal(bool)


class ProgressReporter(QObject):
    """Thread-safe progress reporter for long-running operations.

    Allows worker threads to safely update UI progress indicators via
    Qt signals. Supports both determinate (percentage-based) and
    indeterminate (busy spinner) modes.

    The reporter can delay showing the progress indicator until a
    threshold time has passed, preventing flicker for fast operations.

    Usage:
        reporter = ProgressReporter(progress_bar, threshold_ms=500)

        # In worker thread:
        reporter.start(total=100, message="Processing...")
        for i in range(100):
            do_work(i)
            reporter.update(i + 1, f"Item {i+1}/100")
        reporter.finish("Complete")

    Attributes:
        threshold_ms: Delay before showing indicator (ms).
    """

    def __init__(
        self,
        progress_bar: QProgressBar,
        threshold_ms: int = 500,
    ) -> None:
        """Initialize the progress reporter.

        Args:
            progress_bar: The UI progress bar widget.
            threshold_ms: Delay before showing indicator (default: 500).
        """
        super().__init__()

        self._progress_bar = progress_bar
        self._threshold_ms = threshold_ms
        self._signals = ProgressSignals()
        self._started_at: float = 0.0
        self._total: int | None = None
        self._visible: bool = False

        # Timer for delayed visibility
        self._threshold_timer = QTimer(self)
        self._threshold_timer.setSingleShot(True)
        self._threshold_timer.timeout.connect(self._show_progress_bar)

        # Connect signals to UI updates (using queued connection for thread safety)
        self._signals.progress.connect(self._update_progress)
        self._signals.status.connect(self._update_status)
        self._signals.indeterminate.connect(self._set_indeterminate)
        self._signals.finished.connect(self._on_finished)
        self._signals.error.connect(self._on_error)

    @property
    def threshold_ms(self) -> int:
        """Delay before showing indicator (ms)."""
        return self._threshold_ms

    def start(self, total: int | None = None, message: str = "") -> None:
        """Start progress tracking.

        Args:
            total: Total steps (None for indeterminate/busy mode).
            message: Initial status message.
        """
        self._started_at = time.perf_counter() * 1000
        self._total = total
        self._visible = False

        # Set indeterminate mode if no total provided
        if total is None:
            self._signals.indeterminate.emit(True)
        else:
            self._signals.indeterminate.emit(False)
            self._signals.progress.emit(0)

        if message:
            self._signals.status.emit(message)

        # Start threshold timer for delayed visibility
        self._threshold_timer.start(self._threshold_ms)

    def update(self, current: int, message: str = "") -> None:
        """Update progress.

        Thread-safe: Can be called from worker threads.

        Args:
            current: Current step (0 to total).
            message: Optional status message update.
        """
        if self._total is not None and self._total > 0:
            # Calculate percentage, clamped to 0-100
            percentage = int((current / self._total) * 100)
            percentage = max(0, min(100, percentage))
            self._signals.progress.emit(percentage)

        if message:
            self._signals.status.emit(message)

    def set_indeterminate(self, busy: bool) -> None:
        """Switch between determinate and indeterminate modes.

        Thread-safe: Can be called from worker threads.

        Args:
            busy: True for busy spinner, False for percentage bar.
        """
        self._signals.indeterminate.emit(busy)

    def finish(self, message: str = "") -> None:
        """Mark progress as complete.

        Thread-safe: Can be called from worker threads.

        Args:
            message: Final status message.
        """
        if message:
            self._signals.status.emit(message)

        self._signals.finished.emit()

    def error(self, message: str = "") -> None:
        """Report an error state.

        Thread-safe: Can be called from worker threads.

        Args:
            message: Error message to display.
        """
        self._signals.error.emit(message)

    @Slot()
    def _show_progress_bar(self) -> None:
        """Show the progress bar after threshold delay."""
        self._visible = True
        self._progress_bar.setVisible(True)

    @Slot(int)
    def _update_progress(self, value: int) -> None:
        """Update the progress bar value.

        Args:
            value: Progress value (0-100).
        """
        self._progress_bar.setValue(value)

    @Slot(str)
    def _update_status(self, message: str) -> None:
        """Update the progress bar text format.

        Args:
            message: Status message.
        """
        self._progress_bar.setFormat(message)

    @Slot(bool)
    def _set_indeterminate(self, busy: bool) -> None:
        """Switch progress bar mode.

        Args:
            busy: True for indeterminate (busy spinner) mode.
        """
        if busy:
            self._progress_bar.setRange(0, 0)  # Indeterminate mode
        else:
            self._progress_bar.setRange(0, 100)  # Normal mode

    @Slot()
    def _on_finished(self) -> None:
        """Handle operation completion."""
        self._threshold_timer.stop()

        # Check if we should hide (operation completed before threshold)
        elapsed = time.perf_counter() * 1000 - self._started_at
        if elapsed < self._threshold_ms and not self._visible:
            # Don't show if we completed before the threshold
            pass
        else:
            # Reset and hide after completion
            self._progress_bar.setValue(100)

        # Always hide after a short delay to show completion
        QTimer.singleShot(500, self._hide_progress_bar)

    @Slot(str)
    def _on_error(self, message: str) -> None:
        """Handle error state.

        Args:
            message: Error message.
        """
        self._threshold_timer.stop()
        self._progress_bar.setFormat(f"Error: {message}")
        self._progress_bar.setVisible(True)

        # Hide after showing error
        QTimer.singleShot(3000, self._hide_progress_bar)

    def _hide_progress_bar(self) -> None:
        """Hide the progress bar and reset state."""
        self._progress_bar.setVisible(False)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("%p%")
        self._visible = False


def create_progress_reporter(
    progress_bar: QProgressBar,
    threshold_ms: int = 500,
) -> ProgressReporter:
    """Factory function to create a ProgressReporter.

    Args:
        progress_bar: The UI progress bar widget.
        threshold_ms: Delay before showing indicator.

    Returns:
        A configured ProgressReporter instance.
    """
    return ProgressReporter(progress_bar, threshold_ms)


__all__ = ["ProgressSignals", "ProgressReporter", "create_progress_reporter"]
