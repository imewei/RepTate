"""Pytest fixtures for application integration tests.

Provides fixtures for testing complete RepTate application workflows
including data loading, theory creation, and fitting operations.

These tests require the Qt event loop and test data files.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

    from PySide6.QtWidgets import QApplication


# =============================================================================
# Custom Markers
# =============================================================================

def pytest_configure(config):
    """Register custom markers for application tests."""
    config.addinivalue_line(
        "markers", "gui: marks tests that require Qt GUI (deselect with '-m \"not gui\"')"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "requires_data: marks tests that require test data files"
    )


# =============================================================================
# Qt Application Fixtures
# =============================================================================

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


@pytest.fixture(scope="module")
def application_manager(qapp):
    """Create QApplicationManager instance for testing.

    This fixture provides a RepTate application manager with
    single-thread mode enabled for deterministic testing.

    Yields:
        QApplicationManager instance (not shown)
    """
    import logging

    from RepTate.core.CmdBase import CalcMode, CmdBase
    from RepTate.gui.QApplicationManager import QApplicationManager

    # Set single-thread mode for deterministic tests
    CmdBase.calcmode = CalcMode.singlethread

    manager = QApplicationManager(loglevel=logging.WARNING)
    manager.setStyleSheet("QTabBar::tab { color:black; height: 22px; }")

    yield manager

    # Cleanup
    manager.close()


# =============================================================================
# Data Directory Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent.parent.parent


@pytest.fixture(scope="session")
def data_dir(project_root) -> Path:
    """Return the path to the data directory."""
    return project_root / "data"


@pytest.fixture(scope="session")
def pi_linear_dir(data_dir) -> Path:
    """Return path to PI_LINEAR data directory."""
    return data_dir / "PI_LINEAR"


@pytest.fixture(scope="session")
def pi_star_dir(data_dir) -> Path:
    """Return path to PI_STAR data directory."""
    return data_dir / "PI_STAR"


@pytest.fixture(scope="session")
def dow_dir(data_dir) -> Path:
    """Return path to DOW data directory."""
    return data_dir / "DOW"


@pytest.fixture(scope="session")
def gt_dir(data_dir) -> Path:
    """Return path to Gt data directory."""
    return data_dir / "Gt"


@pytest.fixture(scope="session")
def mwd_dir(data_dir) -> Path:
    """Return path to MWD data directory."""
    return data_dir / "MWD"


@pytest.fixture(scope="session")
def react_dir(data_dir) -> Path:
    """Return path to React data directory."""
    return data_dir / "React"


@pytest.fixture(scope="session")
def nlve_extension_dir(data_dir) -> Path:
    """Return path to NLVE_Extension data directory."""
    return data_dir / "NLVE_Extension"


# =============================================================================
# Helper Functions
# =============================================================================

def skip_if_no_data(data_path: Path, reason: str = "Test data not found"):
    """Skip test if data file/directory doesn't exist."""
    if not data_path.exists():
        pytest.skip(f"{reason}: {data_path}")


# =============================================================================
# Application Creation Fixtures
# =============================================================================

@pytest.fixture
def lve_app(application_manager):
    """Create an LVE application instance.

    Yields:
        Tuple of (application_manager, app_name) for the LVE app.
    """
    application_manager.handle_new_app("LVE")
    app_name = f"LVE{application_manager.application_counter}"
    yield application_manager, app_name


@pytest.fixture
def tts_app(application_manager):
    """Create a TTS application instance.

    Yields:
        Tuple of (application_manager, app_name) for the TTS app.
    """
    application_manager.handle_new_app("TTS")
    app_name = f"TTS{application_manager.application_counter}"
    yield application_manager, app_name


@pytest.fixture
def gt_app(application_manager):
    """Create a Gt application instance.

    Yields:
        Tuple of (application_manager, app_name) for the Gt app.
    """
    application_manager.handle_new_app("Gt")
    app_name = f"Gt{application_manager.application_counter}"
    yield application_manager, app_name


@pytest.fixture
def mwd_app(application_manager):
    """Create an MWD application instance.

    Yields:
        Tuple of (application_manager, app_name) for the MWD app.
    """
    application_manager.handle_new_app("MWD")
    app_name = f"MWD{application_manager.application_counter}"
    yield application_manager, app_name


@pytest.fixture
def nlve_app(application_manager):
    """Create an NLVE application instance.

    Yields:
        Tuple of (application_manager, app_name) for the NLVE app.
    """
    application_manager.handle_new_app("NLVE")
    app_name = f"NLVE{application_manager.application_counter}"
    yield application_manager, app_name


@pytest.fixture
def react_app(application_manager):
    """Create a React application instance.

    Yields:
        Tuple of (application_manager, app_name) for the React app.
    """
    application_manager.handle_new_app("React")
    app_name = f"React{application_manager.application_counter}"
    yield application_manager, app_name
