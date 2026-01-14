"""Tests for accessibility improvements.

Feature: 005-gui-performance-integration (T041)
Tests User Story 5: Keyboard Navigation Readiness
Tests User Story 6: Clear Visual State Indication
"""

from __future__ import annotations

import pytest
from pathlib import Path


class TestFocusIndicators:
    """Tests for focus ring styles in QSS."""

    @pytest.fixture
    def qss_content(self) -> str:
        """Load the reptate.qss stylesheet content."""
        qss_path = Path(__file__).parent.parent.parent / "src" / "RepTate" / "gui" / "styles" / "reptate.qss"
        return qss_path.read_text()

    def test_base_focus_ring_exists(self, qss_content: str):
        """Base focus ring style should exist in QSS."""
        assert "*:focus" in qss_content or ":focus" in qss_content

    def test_pushbutton_focus_style(self, qss_content: str):
        """QPushButton should have focus style."""
        assert "QPushButton:focus" in qss_content

    def test_toolbutton_focus_style(self, qss_content: str):
        """QToolButton should have focus style."""
        assert "QToolButton:focus" in qss_content

    def test_combobox_focus_style(self, qss_content: str):
        """QComboBox should have focus style."""
        assert "QComboBox:focus" in qss_content

    def test_lineedit_focus_style(self, qss_content: str):
        """QLineEdit should have focus style."""
        assert "QLineEdit:focus" in qss_content

    def test_spinbox_focus_style(self, qss_content: str):
        """QSpinBox should have focus style."""
        assert "QSpinBox:focus" in qss_content

    def test_focus_border_color(self, qss_content: str):
        """Focus border should use blue color (#1976d2)."""
        # Check that the Material Design blue is used for focus
        assert "#1976d2" in qss_content


class TestContrastCompliance:
    """Tests for WCAG AA contrast compliance."""

    @pytest.fixture
    def qss_content(self) -> str:
        """Load the reptate.qss stylesheet content."""
        qss_path = Path(__file__).parent.parent.parent / "src" / "RepTate" / "gui" / "styles" / "reptate.qss"
        return qss_path.read_text()

    def test_disabled_button_style_exists(self, qss_content: str):
        """Disabled button style should exist."""
        assert "QPushButton:disabled" in qss_content

    def test_disabled_background_color(self, qss_content: str):
        """Disabled background should use WCAG AA compliant color."""
        # #e0e0e0 provides better contrast than original #bdbdbd
        assert "#e0e0e0" in qss_content

    def test_disabled_text_color(self, qss_content: str):
        """Disabled text should use WCAG AA compliant color."""
        # #616161 provides 4.9:1 contrast ratio (passes WCAG AA)
        assert "#616161" in qss_content


class TestSplitterHandles:
    """Tests for splitter handle accessibility."""

    @pytest.fixture
    def qss_content(self) -> str:
        """Load the reptate.qss stylesheet content."""
        qss_path = Path(__file__).parent.parent.parent / "src" / "RepTate" / "gui" / "styles" / "reptate.qss"
        return qss_path.read_text()

    def test_horizontal_splitter_handle_size(self, qss_content: str):
        """Horizontal splitter should have 6px width."""
        assert "QSplitter::handle:horizontal" in qss_content
        # Check for 6px width (increased from 4px)
        assert "width: 6px" in qss_content or "width:6px" in qss_content

    def test_vertical_splitter_handle_size(self, qss_content: str):
        """Vertical splitter should have 6px height."""
        assert "QSplitter::handle:vertical" in qss_content
        # Check for 6px height (increased from 4px)
        assert "height: 6px" in qss_content or "height:6px" in qss_content


class TestLoggerWidget:
    """Tests for logger widget styling."""

    @pytest.fixture
    def qss_content(self) -> str:
        """Load the reptate.qss stylesheet content."""
        qss_path = Path(__file__).parent.parent.parent / "src" / "RepTate" / "gui" / "styles" / "reptate.qss"
        return qss_path.read_text()

    def test_logger_widget_selector(self, qss_content: str):
        """Logger widget should have object name selector in QSS."""
        assert "QTextBrowser#loggerWidget" in qss_content

    def test_logger_background_color(self, qss_content: str):
        """Logger should have light yellow background."""
        assert "#ffffde" in qss_content

