"""
Integration tests for GUI performance optimizations.

Tests interaction between different performance components.

Feature: 004-gui-performance-optimizations
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtCore import QEventLoop, QTimer

from RepTate.gui.performance.batch_update import batch_updates
from RepTate.gui.performance.figure_cache import FigureCache, _reset_global_cache
from RepTate.gui.performance.lazy_loader import LazyModuleLoader
from RepTate.gui.performance.progress import ProgressReporter, create_progress_reporter
from RepTate.gui.performance.thread_pool import (
    CalculationThreadPool,
    _reset_global_pool,
)


def process_events(timeout_ms: int = 100) -> None:
    """Process Qt events for a duration to allow signal delivery."""
    loop = QEventLoop()
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset all singleton instances before and after each test."""
    _reset_global_pool()
    _reset_global_cache()
    yield
    _reset_global_pool()
    _reset_global_cache()


class TestBlittingWithFigureCache:
    """Test interaction between blitting and figure cache."""

    def test_cached_figures_can_be_used_with_blitting(self, qapp, mock_canvas):
        """Test that cached figures work correctly with blitting manager."""
        from RepTate.gui.performance.blitting import BlittingManager

        cache = FigureCache(max_size=5)
        figures_created = []

        def create_figure():
            fig = MagicMock()
            fig.canvas = mock_canvas
            figures_created.append(fig)
            return fig

        # Get figure from cache
        fig1 = cache.get("test:1", create_figure)

        # Use with blitting manager
        manager = BlittingManager(mock_canvas)
        mock_artist = MagicMock()
        mock_artist.set_animated = MagicMock()
        mock_artist.axes = MagicMock()
        mock_artist.axes.figure = fig1
        mock_artist.axes.figure.canvas = mock_canvas

        manager.start_blit([mock_artist])
        assert manager.is_active

        manager.end_blit()
        assert not manager.is_active

        # Figure should still be in cache
        fig2 = cache.get("test:1", create_figure)
        assert fig1 is fig2
        assert len(figures_created) == 1

    @patch("RepTate.gui.performance.figure_cache.plt.close")
    def test_invalidated_figures_release_resources(self, mock_close, qapp):
        """Test that invalidating cached figures properly releases resources."""
        cache = FigureCache(max_size=5)

        mock_fig = MagicMock()
        cache.get("test:1", lambda: mock_fig)

        # Invalidate should close the figure
        cache.invalidate("test:1")

        mock_close.assert_called_once_with(mock_fig)
        assert cache.current_size == 0


class TestThreadPoolWithProgress:
    """Test interaction between thread pool and progress indicators."""

    def test_thread_pool_task_with_progress_reporting(self, qapp, qprogress_bar):
        """Test that thread pool tasks can report progress."""
        pool = CalculationThreadPool(max_threads=2)
        reporter = create_progress_reporter(qprogress_bar, threshold_ms=0)

        result_holder = []

        def task_with_progress():
            reporter.start(total=100, message="Processing...")
            for i in range(0, 101, 10):
                reporter.update(i, f"Step {i}")
            reporter.finish("Complete")
            return "done"

        def on_complete(result):
            result_holder.append(result)

        pool.submit(task_with_progress, on_complete=on_complete)
        pool.wait_all(5000)
        process_events(200)

        assert result_holder == ["done"]

    def test_thread_pool_error_reports_via_progress(self, qapp, qprogress_bar):
        """Test that thread pool errors can be reported via progress."""
        pool = CalculationThreadPool(max_threads=2)
        reporter = create_progress_reporter(qprogress_bar, threshold_ms=0)

        error_messages = []

        def failing_task():
            reporter.start(total=100, message="Processing...")
            reporter.update(50, "Halfway...")
            raise ValueError("Test error")

        def on_error(exc):
            reporter.error(str(exc))
            error_messages.append(str(exc))

        pool.submit(failing_task, on_error=on_error)
        pool.wait_all(5000)
        process_events(200)

        assert len(error_messages) == 1
        assert "Test error" in error_messages[0]

    def test_multiple_concurrent_tasks_with_separate_progress(self, qapp):
        """Test multiple concurrent tasks can have separate progress tracking."""
        pool = CalculationThreadPool(max_threads=4)
        results = []

        def task(task_id):
            # Each task could have its own progress reporter in practice
            return f"task_{task_id}_complete"

        def on_complete(result):
            results.append(result)

        for i in range(4):
            pool.submit(task, i, on_complete=on_complete)

        pool.wait_all(5000)
        process_events(300)

        assert len(results) == 4
        assert all("complete" in r for r in results)


class TestBatchUpdatesWithLazyLoading:
    """Test interaction between batch updates and lazy loading."""

    def test_batch_updates_during_lazy_loaded_operation(self, qapp, qtable_widget):
        """Test that batch updates work correctly during lazy-loaded operations."""
        from PySide6.QtWidgets import QTableWidgetItem

        # Simulate a lazy-loaded module
        loader = LazyModuleLoader("collections")

        # Perform batch updates while using lazy-loaded module
        with batch_updates(qtable_widget):
            # Access the lazy-loaded module
            OrderedDict = loader.OrderedDict

            # Add items to table
            qtable_widget.setRowCount(10)
            qtable_widget.setColumnCount(2)

            for i in range(10):
                item = OrderedDict([("key", i)])
                qtable_widget.setItem(i, 0, QTableWidgetItem(str(item["key"])))

        # Verify module was loaded
        assert loader.is_loaded

        # Verify table was updated
        assert qtable_widget.rowCount() == 10

    def test_lazy_loading_with_progress_indicator(self, qapp, qprogress_bar):
        """Test that lazy loading can show progress indicator."""
        reporter = create_progress_reporter(qprogress_bar, threshold_ms=0)

        loading_started = []

        def on_loading():
            loading_started.append(True)
            reporter.start(total=None, message="Loading module...")

        loader = LazyModuleLoader("collections", on_loading=on_loading)

        # Trigger load
        _ = loader.OrderedDict

        # Callback should have been invoked
        assert loading_started == [True]
        assert loader.is_loaded


class TestCacheInvalidationOnDatasetDelete:
    """Test figure cache invalidation patterns."""

    @patch("RepTate.gui.performance.figure_cache.plt.close")
    def test_invalidate_prefix_for_application_close(self, mock_close):
        """Test invalidate_prefix clears all figures for an application."""
        cache = FigureCache(max_size=20)

        # Create figures for multiple applications
        for app in ["LVE", "TTS", "MWD"]:
            for ds in range(3):
                cache.get(f"{app}:dataset_{ds}", MagicMock)

        assert cache.current_size == 9

        # Close one application
        cache.invalidate_prefix("LVE:")

        assert cache.current_size == 6
        assert mock_close.call_count == 3

    @patch("RepTate.gui.performance.figure_cache.plt.close")
    def test_invalidate_specific_dataset(self, mock_close):
        """Test invalidate clears specific dataset figure."""
        cache = FigureCache(max_size=20)

        figs = {}
        for ds in range(5):
            key = f"LVE:dataset_{ds}"
            figs[key] = MagicMock()
            cache.get(key, lambda k=key: figs[k])

        assert cache.current_size == 5

        # Delete one dataset
        cache.invalidate("LVE:dataset_2")

        assert cache.current_size == 4
        mock_close.assert_called_once_with(figs["LVE:dataset_2"])


class TestModuleImports:
    """Test that all performance modules can be imported correctly."""

    def test_import_from_package(self):
        """Test importing from the performance package."""
        from RepTate.gui.performance import (
            BatchUpdateContext,
            BlittingManager,
            CalculationThreadPool,
            FigureCache,
            LazyModuleLoader,
            ProgressReporter,
            batch_updates,
            create_blitting_manager,
            create_lazy_loader,
            create_progress_reporter,
            get_calculation_pool,
            get_figure_cache,
        )

        # All imports should succeed
        assert BlittingManager is not None
        assert BatchUpdateContext is not None
        assert LazyModuleLoader is not None
        assert CalculationThreadPool is not None
        assert ProgressReporter is not None
        assert FigureCache is not None

    def test_type_definitions_available(self):
        """Test that type definitions are accessible."""
        from RepTate.gui.performance._types import (
            ErrorCallback,
            FigureFactory,
            ProgressCallback,
        )

        assert FigureFactory is not None
        assert ProgressCallback is not None
        assert ErrorCallback is not None

    def test_base_signals_available(self):
        """Test that base signals class is accessible."""
        from RepTate.gui.performance._signals import BaseWorkerSignals

        assert BaseWorkerSignals is not None
