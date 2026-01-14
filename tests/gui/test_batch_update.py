"""Tests for batch update context manager integration.

Feature: 005-gui-performance-integration (T038)
Tests User Story 2: Flicker-Free Project Loading
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


class TestBatchUpdateContext:
    """Tests for BatchUpdateContext functionality."""

    def test_context_manager_protocol(self):
        """BatchUpdateContext should implement context manager protocol."""
        from RepTate.gui.performance import BatchUpdateContext

        widget = MagicMock()
        widget.signalsBlocked.return_value = False
        widget.updatesEnabled.return_value = True

        ctx = BatchUpdateContext(widget)

        # Should have __enter__ and __exit__
        assert hasattr(ctx, "__enter__")
        assert hasattr(ctx, "__exit__")

    def test_signals_blocked_during_batch(self):
        """Signals should be blocked during batch update."""
        from RepTate.gui.performance import BatchUpdateContext

        widget = MagicMock()
        widget.signalsBlocked.return_value = False
        widget.updatesEnabled.return_value = True

        with BatchUpdateContext(widget):
            # Verify blockSignals was called with True
            widget.blockSignals.assert_called_with(True)

    def test_updates_disabled_during_batch(self):
        """Widget updates should be disabled during batch update."""
        from RepTate.gui.performance import BatchUpdateContext

        widget = MagicMock()
        widget.signalsBlocked.return_value = False
        widget.updatesEnabled.return_value = True

        with BatchUpdateContext(widget):
            # Verify setUpdatesEnabled was called with False
            widget.setUpdatesEnabled.assert_called_with(False)

    def test_state_restored_on_exit(self):
        """Original state should be restored on context exit."""
        from RepTate.gui.performance import BatchUpdateContext

        widget = MagicMock()
        widget.signalsBlocked.return_value = False
        widget.updatesEnabled.return_value = True

        with BatchUpdateContext(widget):
            pass

        # Verify state was restored
        calls = widget.blockSignals.call_args_list
        # Last call should restore original state (False)
        assert calls[-1][0][0] is False

        calls = widget.setUpdatesEnabled.call_args_list
        # Last call should restore original state (True)
        assert calls[-1][0][0] is True

    def test_state_restored_on_exception(self):
        """Original state should be restored even if exception occurs."""
        from RepTate.gui.performance import BatchUpdateContext

        widget = MagicMock()
        widget.signalsBlocked.return_value = False
        widget.updatesEnabled.return_value = True

        with pytest.raises(ValueError):
            with BatchUpdateContext(widget):
                raise ValueError("Test exception")

        # Verify state was still restored
        calls = widget.setUpdatesEnabled.call_args_list
        assert calls[-1][0][0] is True

    def test_update_called_on_exit(self):
        """Widget update() should be called on context exit."""
        from RepTate.gui.performance import BatchUpdateContext

        widget = MagicMock()
        widget.signalsBlocked.return_value = False
        widget.updatesEnabled.return_value = True

        with BatchUpdateContext(widget):
            pass

        widget.update.assert_called_once()


class TestBatchUpdatesHelper:
    """Tests for the batch_updates helper function."""

    def test_batch_updates_creates_context(self):
        """batch_updates should create and yield a BatchUpdateContext."""
        from RepTate.gui.performance import batch_updates, BatchUpdateContext

        widget = MagicMock()
        widget.signalsBlocked.return_value = False
        widget.updatesEnabled.return_value = True

        with batch_updates(widget) as ctx:
            assert isinstance(ctx, BatchUpdateContext)
            assert ctx.widget is widget
