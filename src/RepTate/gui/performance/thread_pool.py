"""
Thread Pool for parallel calculations.

Provides a shared thread pool for managing concurrent calculations with
resource limiting and cancellation support. Uses Qt's QThreadPool for
proper integration with the GUI event loop.

Feature: 004-gui-performance-optimizations
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Callable, TypeVar

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from RepTate.gui.performance._signals import BaseWorkerSignals

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Singleton instance
_global_pool: CalculationThreadPool | None = None


class WorkerSignals(BaseWorkerSignals):
    """Signals for calculation workers.

    Extends BaseWorkerSignals with a result signal for returning
    computation results.

    Signals:
        finished: Emitted when the operation completes (inherited).
        error: Emitted when an error occurs (inherited).
        result: Emitted with the computation result.
    """

    result = Signal(object)


class CalculationRunnable(QRunnable):
    """A reusable calculation worker for the thread pool.

    Wraps a callable function and executes it in a worker thread,
    emitting signals for completion, error, and result.

    Attributes:
        signals: WorkerSignals instance for communication.
        is_cancelled: Whether cancellation has been requested.
    """

    def __init__(
        self,
        func: Callable[..., T],
        *args: Any,
        on_complete: Callable[[T], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the runnable.

        Args:
            func: The function to execute.
            *args: Positional arguments for the function.
            on_complete: Callback when function completes successfully.
            on_error: Callback when function raises an exception.
            **kwargs: Keyword arguments for the function.
        """
        super().__init__()
        self.setAutoDelete(True)

        self._func = func
        self._args = args
        self._kwargs = kwargs
        self._on_complete = on_complete
        self._on_error = on_error
        self._cancelled = False

        self.signals = WorkerSignals()

        # Connect callbacks if provided
        if on_complete is not None:
            self.signals.result.connect(on_complete)
        if on_error is not None:
            self.signals.error.connect(
                lambda msg: on_error(Exception(msg)) if on_error else None
            )

    @property
    def is_cancelled(self) -> bool:
        """Whether cancellation has been requested."""
        return self._cancelled

    def run(self) -> None:
        """Execute the calculation.

        Runs the function with the provided arguments and emits
        appropriate signals based on the result.
        """
        if self._cancelled:
            return

        try:
            result = self._func(*self._args, **self._kwargs)

            if not self._cancelled:
                self.signals.result.emit(result)
                self.signals.finished.emit()
        except Exception as e:
            if not self._cancelled:
                logger.error(f"Calculation error: {e}")
                self.signals.error.emit(str(e))

    def cancel(self) -> None:
        """Request cancellation of this calculation.

        Note: This sets a flag that the calculation can check, but
        cannot forcefully stop a running calculation. The calculation
        function should periodically check is_cancelled.
        """
        self._cancelled = True


class CalculationThreadPool:
    """Shared thread pool for managing parallel calculations.

    Provides a centralized pool for running calculations with resource
    limiting and task tracking.

    Attributes:
        max_threads: Maximum number of concurrent worker threads.
        active_count: Number of currently running tasks.
    """

    def __init__(self, max_threads: int | None = None) -> None:
        """Initialize the thread pool.

        Args:
            max_threads: Maximum concurrent workers. Defaults to CPU count.
        """
        self._pool = QThreadPool.globalInstance()

        if max_threads is None:
            max_threads = os.cpu_count() or 4

        # Clamp to reasonable range
        max_threads = max(1, min(max_threads, (os.cpu_count() or 4) * 2))

        self._max_threads = max_threads
        self._pool.setMaxThreadCount(max_threads)

        self._active_tasks: dict[str, CalculationRunnable] = {}

    @property
    def max_threads(self) -> int:
        """Maximum number of concurrent worker threads."""
        return self._max_threads

    @property
    def active_count(self) -> int:
        """Number of currently active threads in the pool."""
        return self._pool.activeThreadCount()

    def submit(
        self,
        func: Callable[..., T],
        *args: Any,
        on_complete: Callable[[T], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
        **kwargs: Any,
    ) -> str:
        """Submit a calculation for execution.

        Args:
            func: The function to execute.
            *args: Positional arguments for the function.
            on_complete: Callback when function completes successfully.
            on_error: Callback when function raises an exception.
            **kwargs: Keyword arguments for the function.

        Returns:
            Task ID for tracking/cancellation.
        """
        task_id = str(uuid.uuid4())

        runnable = CalculationRunnable(
            func,
            *args,
            on_complete=on_complete,
            on_error=on_error,
            **kwargs,
        )

        # Track the task
        self._active_tasks[task_id] = runnable

        # Clean up tracking when finished
        def cleanup():
            self._active_tasks.pop(task_id, None)

        runnable.signals.finished.connect(cleanup)
        runnable.signals.error.connect(lambda _: cleanup())

        # Start the task
        self._pool.start(runnable)

        logger.debug(f"Submitted task {task_id}")
        return task_id

    def cancel(self, task_id: str) -> bool:
        """Request cancellation of a running task.

        Args:
            task_id: ID returned from submit().

        Returns:
            True if cancellation was requested, False if task not found.
        """
        runnable = self._active_tasks.get(task_id)
        if runnable is None:
            return False

        runnable.cancel()
        self._active_tasks.pop(task_id, None)
        logger.debug(f"Cancelled task {task_id}")
        return True

    def wait_all(self, timeout_ms: int | None = None) -> bool:
        """Wait for all tasks to complete.

        Args:
            timeout_ms: Maximum wait time in milliseconds (None = forever).

        Returns:
            True if all tasks completed, False if timeout.
        """
        if timeout_ms is None:
            self._pool.waitForDone(-1)
            return True
        else:
            return self._pool.waitForDone(timeout_ms)

    def clear(self) -> None:
        """Cancel all pending tasks and clear the pool.

        Note: This cancels pending tasks but cannot stop running ones.
        """
        self._pool.clear()
        for task_id in list(self._active_tasks.keys()):
            self.cancel(task_id)


def get_calculation_pool(max_threads: int | None = None) -> CalculationThreadPool:
    """Get or create the global calculation thread pool.

    Args:
        max_threads: Override default thread limit (CPU cores).
            Only used when creating the pool for the first time.

    Returns:
        Singleton thread pool instance.
    """
    global _global_pool

    if _global_pool is None:
        _global_pool = CalculationThreadPool(max_threads)

    return _global_pool


def _reset_global_pool() -> None:
    """Reset the global pool (for testing only)."""
    global _global_pool
    if _global_pool is not None:
        _global_pool.clear()
        _global_pool.wait_all(1000)
    _global_pool = None


__all__ = [
    "WorkerSignals",
    "CalculationRunnable",
    "CalculationThreadPool",
    "get_calculation_pool",
]
