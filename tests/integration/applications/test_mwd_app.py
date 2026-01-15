"""Integration tests for MWD (Molecular Weight Distribution) application.

Tests validate complete MWD application workflows including:
- Loading GPC (gel permeation chromatography) data files
- Creating MWD theories (Discretize MWD, LogNormal, GEX)
- Molecular weight distribution analysis

Migrated from legacy RepTate_Test_MWD.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import skip_if_no_data


@pytest.mark.gui
@pytest.mark.slow
@pytest.mark.requires_data
class TestMWDDiscretize:
    """Test MWD application with Discretize MWD theory."""

    def test_load_gpc_data(self, mwd_app, mwd_dir):
        """Test loading GPC data file."""
        manager, app_name = mwd_app

        data_file = mwd_dir / "Munstedt_PSIII.gpc"
        skip_if_no_data(data_file)

        # Load data file
        manager.applications[app_name].new_tables_from_files([str(data_file)])

        # Verify dataset was created
        assert "Set1" in manager.applications[app_name].datasets

    def test_discretize_mwd_theory(self, mwd_app, mwd_dir):
        """Test Discretize MWD theory creation."""
        manager, app_name = mwd_app

        data_file = mwd_dir / "Munstedt_PSIII.gpc"
        skip_if_no_data(data_file)

        # Load data
        manager.applications[app_name].new_tables_from_files([str(data_file)])

        # Create Discretize MWD theory
        manager.applications[app_name].datasets["Set1"].new_theory("Discretize MWD")

        # Verify theory was created
        assert len(manager.applications[app_name].datasets["Set1"].theories) > 0


@pytest.mark.gui
@pytest.mark.slow
@pytest.mark.requires_data
class TestMWDDistributions:
    """Test MWD application with distribution theories."""

    def test_lognormal_theory(self, mwd_app, mwd_dir):
        """Test LogNormal distribution theory."""
        manager, app_name = mwd_app

        data_file = mwd_dir / "Munstedt_PSIII.gpc"
        skip_if_no_data(data_file)

        # Load data
        manager.applications[app_name].new_tables_from_files([str(data_file)])

        # Create LogNormal theory
        manager.applications[app_name].datasets["Set1"].new_theory("LogNormal")

    def test_gex_theory(self, mwd_app, mwd_dir):
        """Test GEX (Generalized Exponential) distribution theory."""
        manager, app_name = mwd_app

        data_file = mwd_dir / "Munstedt_PSIII.gpc"
        skip_if_no_data(data_file)

        # Load data
        manager.applications[app_name].new_tables_from_files([str(data_file)])

        # Create GEX theory
        manager.applications[app_name].datasets["Set1"].new_theory("GEX")


@pytest.mark.gui
@pytest.mark.slow
@pytest.mark.requires_data
class TestMWDMultipleTheories:
    """Test MWD application with multiple theories simultaneously."""

    def test_multiple_mwd_theories(self, mwd_app, mwd_dir):
        """Test creating multiple MWD theories on same dataset."""
        manager, app_name = mwd_app

        data_file = mwd_dir / "Munstedt_PSIII.gpc"
        skip_if_no_data(data_file)

        # Load data
        manager.applications[app_name].new_tables_from_files([str(data_file)])

        # Create multiple theories
        manager.applications[app_name].datasets["Set1"].new_theory("Discretize MWD")
        manager.applications[app_name].datasets["Set1"].new_theory("LogNormal")
        manager.applications[app_name].datasets["Set1"].new_theory("GEX")

        # Verify all theories were created
        assert len(manager.applications[app_name].datasets["Set1"].theories) >= 3
