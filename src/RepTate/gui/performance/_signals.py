"""
Shared signal base classes for the performance module.

Provides common Qt signals used by WorkerSignals (thread_pool.py) and
ProgressSignals (progress.py) to avoid code duplication.

Feature: 004-gui-performance-optimizations
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal


class BaseWorkerSignals(QObject):
    """Base class for worker thread signals.

    Provides common signals for completion and error handling that
    are shared across different worker types.

    Signals:
        finished: Emitted when the operation completes successfully.
        error: Emitted when an error occurs, with the error message.
    """

    finished = Signal()
    error = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        """Initialize the signals.

        Args:
            parent: Optional parent QObject for Qt object hierarchy.
        """
        super().__init__(parent)


__all__ = ["BaseWorkerSignals"]
