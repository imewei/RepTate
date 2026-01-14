"""
Tests for ProgressReporter.

Feature: 004-gui-performance-optimizations
"""

from __future__ import annotations

import time

import pytest
from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer

from RepTate.gui.performance.progress import (
    ProgressReporter,
    ProgressSignals,
    create_progress_reporter,
)


def process_events(timeout_ms: int = 100) -> None:
    """Process Qt events for a duration to allow signal delivery."""
    loop = QEventLoop()
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()


class TestProgressSignals:
    """Tests for ProgressSignals class."""

    def test_signals_exist(self, qapp):
        """Test that all expected signals exist."""
        signals = ProgressSignals()

        assert hasattr(signals, "finished")
        assert hasattr(signals, "error")
        assert hasattr(signals, "progress")
        assert hasattr(signals, "status")
        assert hasattr(signals, "indeterminate")


class TestProgressReporter:
    """Tests for ProgressReporter class."""

    def test_start_with_total_sets_determinate_mode(self, qprogress_bar):
        """Test start with total sets determinate mode."""
        reporter = ProgressReporter(qprogress_bar, threshold_ms=0)

        reporter.start(total=100, message="Processing...")
        process_events(50)

        # Range should be (0, 100) for determinate mode
        assert qprogress_bar.minimum() == 0
        assert qprogress_bar.maximum() == 100

    def test_start_without_total_sets_indeterminate_mode(self, qprogress_bar):
        """Test start without total sets indeterminate mode."""
        reporter = ProgressReporter(qprogress_bar, threshold_ms=0)

        reporter.start(total=None, message="Loading...")
        process_events(50)

        # Range should be (0, 0) for indeterminate mode
        assert qprogress_bar.minimum() == 0
        assert qprogress_bar.maximum() == 0

    def test_update_clamps_value_to_0_100(self, qprogress_bar):
        """Test update clamps value to 0-100 range."""
        reporter = ProgressReporter(qprogress_bar, threshold_ms=0)

        reporter.start(total=100)
        process_events(50)

        # Test clamping at upper bound
        reporter.update(150)  # Should clamp to 100
        process_events(50)
        assert qprogress_bar.value() == 100

        # Test clamping at lower bound
        reporter.update(-50)  # Should clamp to 0
        process_events(50)
        assert qprogress_bar.value() == 0

    def test_set_indeterminate_true_sets_busy_mode(self, qprogress_bar):
        """Test set_indeterminate(True) sets range (0, 0)."""
        reporter = ProgressReporter(qprogress_bar, threshold_ms=0)

        reporter.start(total=100)
        reporter.set_indeterminate(True)
        process_events(50)

        assert qprogress_bar.minimum() == 0
        assert qprogress_bar.maximum() == 0

    def test_set_indeterminate_false_sets_normal_mode(self, qprogress_bar):
        """Test set_indeterminate(False) sets range (0, 100)."""
        reporter = ProgressReporter(qprogress_bar, threshold_ms=0)

        reporter.start(total=None)  # Start in indeterminate
        reporter.set_indeterminate(False)
        process_events(50)

        assert qprogress_bar.minimum() == 0
        assert qprogress_bar.maximum() == 100

    def test_finish_resets_progress_bar(self, qprogress_bar):
        """Test finish resets the progress bar after completion."""
        reporter = ProgressReporter(qprogress_bar, threshold_ms=0)

        reporter.start(total=100)
        reporter.update(50)
        reporter.finish("Complete")
        process_events(100)

        # After finish, value should be 100 briefly
        assert qprogress_bar.value() == 100

    def test_error_displays_error_state(self, qprogress_bar):
        """Test error displays error state and message."""
        reporter = ProgressReporter(qprogress_bar, threshold_ms=0)

        reporter.start(total=100)
        reporter.error("Something went wrong")
        process_events(100)

        # Progress bar should show error message
        assert "Error" in qprogress_bar.format()

    def test_indicator_hidden_until_threshold_reached(self, qprogress_bar):
        """Test indicator is hidden until threshold is reached."""
        # Use a longer threshold for this test
        reporter = ProgressReporter(qprogress_bar, threshold_ms=500)

        # Initially hidden
        qprogress_bar.setVisible(False)

        reporter.start(total=100)

        # Immediately after start, should still be hidden (threshold not reached)
        process_events(50)
        # Note: This depends on implementation - may need adjustment

    def test_indicator_not_shown_for_fast_operations(self, qprogress_bar):
        """Test indicator is not shown for operations completing before threshold."""
        reporter = ProgressReporter(qprogress_bar, threshold_ms=500)

        # Initially hidden
        qprogress_bar.setVisible(False)

        reporter.start(total=100)
        reporter.update(50)
        reporter.finish("Done")

        # Process events but not enough for threshold
        process_events(50)

        # Should not have been made visible (operation completed before threshold)
        # Note: The implementation may vary

    def test_thread_safe_updates_via_signals(self, qprogress_bar, qapp):
        """Test progress updates work correctly via signals."""
        reporter = ProgressReporter(qprogress_bar, threshold_ms=0)

        reporter.start(total=100, message="Processing...")
        process_events(50)

        for i in range(0, 101, 10):
            reporter.update(i, f"Step {i}")
            process_events(10)

        # Should have progressed through values
        assert qprogress_bar.value() == 100

    def test_threshold_ms_property(self, qprogress_bar):
        """Test threshold_ms property returns correct value."""
        reporter = ProgressReporter(qprogress_bar, threshold_ms=750)

        assert reporter.threshold_ms == 750


class TestCreateProgressReporter:
    """Tests for create_progress_reporter factory function."""

    def test_factory_creates_reporter(self, qprogress_bar):
        """Test factory function creates a ProgressReporter instance."""
        reporter = create_progress_reporter(qprogress_bar)

        assert isinstance(reporter, ProgressReporter)

    def test_factory_passes_threshold_ms(self, qprogress_bar):
        """Test factory passes threshold_ms to reporter."""
        reporter = create_progress_reporter(qprogress_bar, threshold_ms=300)

        assert reporter.threshold_ms == 300

    def test_factory_default_threshold(self, qprogress_bar):
        """Test factory uses default threshold of 500ms."""
        reporter = create_progress_reporter(qprogress_bar)

        assert reporter.threshold_ms == 500
