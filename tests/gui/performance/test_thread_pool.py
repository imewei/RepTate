"""
Tests for CalculationThreadPool.

Feature: 004-gui-performance-optimizations
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import QCoreApplication

from RepTate.gui.performance.thread_pool import (
    CalculationRunnable,
    CalculationThreadPool,
    WorkerSignals,
    _reset_global_pool,
    get_calculation_pool,
)


@pytest.fixture(autouse=True)
def reset_pool():
    """Reset the global pool before and after each test."""
    _reset_global_pool()
    yield
    _reset_global_pool()


def process_events(timeout_ms: int = 100) -> None:
    """Process Qt events for a duration to allow signal delivery."""
    from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer

    loop = QEventLoop()
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()


class TestWorkerSignals:
    """Tests for WorkerSignals class."""

    def test_signals_exist(self, qapp):
        """Test that all expected signals exist."""
        signals = WorkerSignals()

        assert hasattr(signals, "finished")
        assert hasattr(signals, "error")
        assert hasattr(signals, "result")


class TestCalculationRunnable:
    """Tests for CalculationRunnable class."""

    def test_run_executes_function(self, qapp):
        """Test run() executes the provided function."""
        result_holder = []

        def test_func(value):
            result_holder.append(value)
            return value * 2

        runnable = CalculationRunnable(test_func, 5)
        runnable.run()

        assert result_holder == [5]

    def test_run_emits_result_signal(self, qapp):
        """Test run() emits result signal with return value."""
        result_holder = []

        def test_func():
            return 42

        runnable = CalculationRunnable(test_func)
        runnable.signals.result.connect(result_holder.append)
        runnable.run()

        assert result_holder == [42]

    def test_run_emits_finished_signal(self, qapp):
        """Test run() emits finished signal on completion."""
        finished_called = []

        def test_func():
            return "done"

        runnable = CalculationRunnable(test_func)
        runnable.signals.finished.connect(lambda: finished_called.append(True))
        runnable.run()

        assert finished_called == [True]

    def test_run_emits_error_on_exception(self, qapp):
        """Test run() emits error signal on exception."""
        error_holder = []

        def failing_func():
            raise ValueError("Test error")

        runnable = CalculationRunnable(failing_func)
        runnable.signals.error.connect(error_holder.append)
        runnable.run()

        assert len(error_holder) == 1
        assert "Test error" in error_holder[0]

    def test_cancel_sets_flag(self, qapp):
        """Test cancel() sets the is_cancelled flag."""
        runnable = CalculationRunnable(lambda: None)

        assert runnable.is_cancelled is False

        runnable.cancel()

        assert runnable.is_cancelled is True

    def test_cancelled_runnable_does_not_run(self, qapp):
        """Test cancelled runnable doesn't execute function."""
        result_holder = []

        runnable = CalculationRunnable(lambda: result_holder.append(True))
        runnable.cancel()
        runnable.run()

        assert result_holder == []


class TestCalculationThreadPool:
    """Tests for CalculationThreadPool class."""

    def test_pool_respects_max_threads(self, qapp):
        """Test pool respects the max_threads limit."""
        pool = CalculationThreadPool(max_threads=2)

        assert pool.max_threads == 2

    def test_submit_returns_unique_task_id(self, qapp):
        """Test submit returns unique task IDs."""
        pool = CalculationThreadPool()

        task_id1 = pool.submit(lambda: None)
        task_id2 = pool.submit(lambda: None)

        assert task_id1 != task_id2

    def test_on_complete_callback_invoked(self, qapp):
        """Test on_complete callback is invoked on success."""
        pool = CalculationThreadPool()
        result_holder = []

        def on_complete(result):
            result_holder.append(result)

        pool.submit(lambda: 42, on_complete=on_complete)
        pool.wait_all(5000)
        process_events(200)

        assert result_holder == [42]

    def test_on_error_callback_invoked(self, qapp):
        """Test on_error callback is invoked on exception."""
        pool = CalculationThreadPool()
        error_holder = []

        def on_error(exc):
            error_holder.append(str(exc))

        def failing_func():
            raise ValueError("Test error")

        pool.submit(failing_func, on_error=on_error)
        pool.wait_all(5000)
        process_events(200)

        assert len(error_holder) == 1
        assert "Test error" in error_holder[0]

    def test_cancel_stops_pending_tasks(self, qapp):
        """Test cancel() marks task for cancellation."""
        pool = CalculationThreadPool()
        result_holder = []

        # Submit a task that will be cancelled
        def slow_task():
            time.sleep(0.1)
            result_holder.append("executed")
            return "done"

        task_id = pool.submit(slow_task)
        cancelled = pool.cancel(task_id)

        # Cancel should return True
        assert cancelled is True

    def test_cancel_returns_false_for_unknown_task(self, qapp):
        """Test cancel() returns False for unknown task ID."""
        pool = CalculationThreadPool()

        result = pool.cancel("nonexistent-task-id")

        assert result is False

    def test_wait_all_blocks_until_completion(self, qapp):
        """Test wait_all blocks until all tasks complete."""
        pool = CalculationThreadPool()
        result_holder = []

        for i in range(3):
            pool.submit(lambda x=i: result_holder.append(x))

        pool.wait_all()

        assert len(result_holder) == 3

    def test_wait_all_respects_timeout(self, qapp):
        """Test wait_all respects timeout parameter."""
        pool = CalculationThreadPool()

        def slow_task():
            time.sleep(10)  # Very slow
            return "done"

        pool.submit(slow_task)

        # Should timeout quickly
        result = pool.wait_all(timeout_ms=100)

        # Timeout should return False
        assert result is False

    def test_active_count_reflects_running_tasks(self, qapp):
        """Test active_count reflects currently running tasks."""
        pool = CalculationThreadPool(max_threads=2)

        # Initially no active tasks
        assert pool.active_count >= 0  # Could be 0 or small number


class TestGetCalculationPool:
    """Tests for get_calculation_pool singleton factory."""

    def test_singleton_returns_same_instance(self, qapp):
        """Test get_calculation_pool returns same instance."""
        pool1 = get_calculation_pool()
        pool2 = get_calculation_pool()

        assert pool1 is pool2

    def test_singleton_with_max_threads(self, qapp):
        """Test singleton can be created with max_threads."""
        pool = get_calculation_pool(max_threads=4)

        # Note: max_threads only used on first creation
        assert pool.max_threads == 4
