"""Integration tests for Gt (Relaxation Modulus) application.

Tests validate complete Gt application workflows including:
- Loading G(t) relaxation data files
- Creating and fitting Gt theories (Maxwell Modes, Rouse)
- Time-domain analysis

Migrated from legacy RepTate_Test_Gt.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import skip_if_no_data


@pytest.mark.gui
@pytest.mark.slow
@pytest.mark.requires_data
class TestGtMaxwellModes:
    """Test Gt application with Maxwell Modes theory."""

    def test_load_gt_data(self, gt_app, gt_dir):
        """Test loading G(t) relaxation data file."""
        manager, app_name = gt_app

        data_file = gt_dir / "C0224_NVT_450K_1atm.gt"
        skip_if_no_data(data_file)

        # Load data file
        manager.applications[app_name].new_tables_from_files([str(data_file)])

        # Verify dataset was created
        assert "Set1" in manager.applications[app_name].datasets

    def test_maxwell_modes_time_fit(self, gt_app, gt_dir):
        """Test Maxwell Modes theory fitting for G(t) data."""
        manager, app_name = gt_app

        data_file = gt_dir / "C0224_NVT_450K_1atm.gt"
        skip_if_no_data(data_file)

        # Load data
        manager.applications[app_name].new_tables_from_files([str(data_file)])

        # Create Maxwell Modes theory
        manager.applications[app_name].datasets["Set1"].new_theory("Maxwell Modes")

        # Verify theory was created
        assert len(manager.applications[app_name].datasets["Set1"].theories) > 0

        # Minimize error
        manager.applications[app_name].datasets["Set1"].handle_actionMinimize_Error()


@pytest.mark.gui
@pytest.mark.slow
@pytest.mark.requires_data
class TestGtRouse:
    """Test Gt application with Rouse theory."""

    def test_rouse_fit(self, gt_app, gt_dir):
        """Test Rouse theory fitting for G(t) data."""
        manager, app_name = gt_app

        data_file = gt_dir / "C0224_NVT_450K_1atm.gt"
        skip_if_no_data(data_file)

        # Load data
        manager.applications[app_name].new_tables_from_files([str(data_file)])

        # Create Rouse theory
        manager.applications[app_name].datasets["Set1"].new_theory("Rouse")

        # Minimize error
        manager.applications[app_name].datasets["Set1"].handle_actionMinimize_Error()
