"""
Batch Update Context Manager for flicker-free widget updates.

Provides a context manager that disables widget repainting during bulk
data operations to prevent flicker and improve performance.

Feature: 004-gui-performance-optimizations
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget


class BatchUpdateContext:
    """Context manager for batch widget updates.

    Disables widget signals and updates during bulk operations to prevent
    flicker. Automatically restores the original state on exit, even if
    an exception occurs.

    Usage:
        with BatchUpdateContext(table_widget):
            for row in large_dataset:
                add_row_to_table(row)
        # Single repaint occurs here

    Attributes:
        widget: The QWidget being batch-updated.
    """

    def __init__(self, widget: QWidget) -> None:
        """Initialize the batch update context.

        Args:
            widget: The widget to batch-update.
        """
        self._widget = widget
        self._signals_blocked: bool = False
        self._updates_enabled: bool = True

    @property
    def widget(self) -> QWidget:
        """The QWidget being batch-updated."""
        return self._widget

    def __enter__(self) -> BatchUpdateContext:
        """Enter batch update mode.

        Stores the current state and disables signals and updates.

        Returns:
            This context manager instance.
        """
        # Store original state
        self._signals_blocked = self._widget.signalsBlocked()
        self._updates_enabled = self._widget.updatesEnabled()

        # Disable signals and updates
        self._widget.blockSignals(True)
        self._widget.setUpdatesEnabled(False)

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit batch update mode and trigger repaint.

        Restores the original state and triggers a single update.
        Always executes, even if an exception occurred.

        Args:
            exc_type: Exception type if an exception was raised.
            exc_val: Exception value if an exception was raised.
            exc_tb: Exception traceback if an exception was raised.
        """
        # Always restore original state
        self._widget.blockSignals(self._signals_blocked)
        self._widget.setUpdatesEnabled(self._updates_enabled)

        # Trigger a single repaint
        self._widget.update()


@contextmanager
def batch_updates(widget: QWidget) -> Iterator[BatchUpdateContext]:
    """Context manager factory for batch widget updates.

    Convenience function that creates a BatchUpdateContext and yields it.

    Args:
        widget: The widget to batch-update.

    Yields:
        The BatchUpdateContext instance.

    Example:
        with batch_updates(table_widget):
            for row in large_dataset:
                add_row_to_table(row)
    """
    context = BatchUpdateContext(widget)
    with context:
        yield context


__all__ = ["BatchUpdateContext", "batch_updates"]
