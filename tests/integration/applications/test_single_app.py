"""Integration tests for single application startup.

Tests validate basic application creation and window initialization.
These are smoke tests to ensure each application type can be created.

Migrated from legacy RepTate_SingleApp.py.
"""

from __future__ import annotations

import pytest


@pytest.mark.gui
class TestApplicationStartup:
    """Test basic application startup for all application types."""

    def test_lve_app_creation(self, application_manager):
        """Test LVE application can be created."""
        manager = application_manager
        manager.handle_new_app("LVE")
        app_name = f"LVE{manager.application_counter}"

        assert app_name in manager.applications
        assert manager.applications[app_name] is not None

    def test_tts_app_creation(self, application_manager):
        """Test TTS application can be created."""
        manager = application_manager
        manager.handle_new_app("TTS")
        app_name = f"TTS{manager.application_counter}"

        assert app_name in manager.applications

    def test_gt_app_creation(self, application_manager):
        """Test Gt application can be created."""
        manager = application_manager
        manager.handle_new_app("Gt")
        app_name = f"Gt{manager.application_counter}"

        assert app_name in manager.applications

    def test_mwd_app_creation(self, application_manager):
        """Test MWD application can be created."""
        manager = application_manager
        manager.handle_new_app("MWD")
        app_name = f"MWD{manager.application_counter}"

        assert app_name in manager.applications

    def test_nlve_app_creation(self, application_manager):
        """Test NLVE application can be created."""
        manager = application_manager
        manager.handle_new_app("NLVE")
        app_name = f"NLVE{manager.application_counter}"

        assert app_name in manager.applications

    def test_react_app_creation(self, application_manager):
        """Test React application can be created."""
        manager = application_manager
        manager.handle_new_app("React")
        app_name = f"React{manager.application_counter}"

        assert app_name in manager.applications

    def test_creep_app_creation(self, application_manager):
        """Test Creep application can be created."""
        manager = application_manager
        manager.handle_new_app("Creep")
        app_name = f"Creep{manager.application_counter}"

        assert app_name in manager.applications

    def test_laos_app_creation(self, application_manager):
        """Test LAOS application can be created."""
        manager = application_manager
        manager.handle_new_app("LAOS")
        app_name = f"LAOS{manager.application_counter}"

        assert app_name in manager.applications


@pytest.mark.gui
class TestApplicationWindowProperties:
    """Test application window properties after creation."""

    def test_lve_app_has_datasets_dict(self, lve_app):
        """Test LVE application has datasets dictionary."""
        manager, app_name = lve_app
        assert hasattr(manager.applications[app_name], "datasets")
        assert isinstance(manager.applications[app_name].datasets, dict)

    def test_lve_app_has_theories_access(self, lve_app):
        """Test LVE application can access theories through datasets."""
        manager, app_name = lve_app
        # Create a dataset first
        manager.applications[app_name].datasets  # Access datasets dict
        # App should have available_theories attribute
        assert hasattr(manager.applications[app_name], "available_theories")
