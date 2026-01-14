"""
Pytest fixtures for GUI performance tests.

Provides common fixtures for testing Qt widgets and matplotlib integration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

    from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp() -> Generator[QApplication, None, None]:
    """Create a QApplication instance for the test session.

    This fixture ensures a single QApplication exists for all tests,
    as Qt only allows one QApplication per process.
    """
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    yield app

    # Don't delete the app - let it be cleaned up naturally


@pytest.fixture
def mock_canvas() -> MagicMock:
    """Create a mock matplotlib FigureCanvasQTAgg for testing.

    The mock provides common canvas attributes and methods needed
    for blitting operations.
    """
    canvas = MagicMock()
    canvas.supports_blit = True
    canvas.figure = MagicMock()
    canvas.figure.bbox = MagicMock()
    canvas.copy_from_bbox = MagicMock(return_value=MagicMock())
    canvas.restore_region = MagicMock()
    canvas.update = MagicMock()
    canvas.draw = MagicMock()
    canvas.draw_idle = MagicMock()
    return canvas


@pytest.fixture
def mock_artist() -> MagicMock:
    """Create a mock matplotlib Artist for testing."""
    artist = MagicMock()
    artist.set_animated = MagicMock()
    artist.get_animated = MagicMock(return_value=False)
    artist.axes = MagicMock()
    artist.axes.bbox = MagicMock()
    return artist


@pytest.fixture
def mock_axes() -> MagicMock:
    """Create a mock matplotlib Axes for testing."""
    axes = MagicMock()
    axes.bbox = MagicMock()
    axes.draw_artist = MagicMock()
    return axes


@pytest.fixture
def qtable_widget(qapp):
    """Create a QTableWidget for testing batch updates."""
    from PySide6.QtWidgets import QTableWidget

    table = QTableWidget()
    table.setRowCount(10)
    table.setColumnCount(5)
    yield table
    table.deleteLater()


@pytest.fixture
def qtree_widget(qapp):
    """Create a QTreeWidget for testing batch updates."""
    from PySide6.QtWidgets import QTreeWidget

    tree = QTreeWidget()
    tree.setColumnCount(3)
    yield tree
    tree.deleteLater()


@pytest.fixture
def qprogress_bar(qapp):
    """Create a QProgressBar for testing progress indicators."""
    from PySide6.QtWidgets import QProgressBar

    progress = QProgressBar()
    yield progress
    progress.deleteLater()
