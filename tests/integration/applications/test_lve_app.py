"""Integration tests for LVE (Linear Viscoelasticity) application.

Tests validate complete LVE application workflows including:
- Loading frequency sweep data files
- Creating and fitting LVE theories (Likhtman-McLeish, Rouse, Carreau-Yasuda, Maxwell, DTD)
- Minimization and error calculation

Migrated from legacy RepTate_Test_LVE.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import skip_if_no_data


@pytest.mark.gui
@pytest.mark.slow
@pytest.mark.requires_data
class TestLVELikhtmanMcLeish:
    """Test LVE application with Likhtman-McLeish theory."""

    def test_load_pi_linear_data(self, lve_app, pi_linear_dir):
        """Test loading PI linear data files into LVE application."""
        manager, app_name = lve_app

        data_files = [
            pi_linear_dir / "PI_13.5k_T-35.tts",
            pi_linear_dir / "PI_23.4k_T-35.tts",
            pi_linear_dir / "PI_33.6k_T-35.tts",
            pi_linear_dir / "PI_94.9k_T-35.tts",
            pi_linear_dir / "PI_225.9k_T-35.tts",
            pi_linear_dir / "PI_483.1k_T-35.tts",
            pi_linear_dir / "PI_634.5k_T-35.tts",
            pi_linear_dir / "PI_1131k_T-35.tts",
        ]

        # Check if data files exist
        for f in data_files:
            skip_if_no_data(f)

        # Load data files
        file_paths = [str(f) for f in data_files]
        manager.applications[app_name].new_tables_from_files(file_paths)

        # Verify dataset was created
        assert "Set1" in manager.applications[app_name].datasets

    def test_likhtman_mcleish_theory_fit(self, lve_app, pi_linear_dir):
        """Test Likhtman-McLeish theory creation and fitting."""
        manager, app_name = lve_app

        data_file = pi_linear_dir / "PI_94.9k_T-35.tts"
        skip_if_no_data(data_file)

        # Load data
        manager.applications[app_name].new_tables_from_files([str(data_file)])

        # Create Likhtman-McLeish theory
        manager.applications[app_name].datasets["Set1"].new_theory("Likhtman-McLeish")

        # Verify theory was created
        assert len(manager.applications[app_name].datasets["Set1"].theories) > 0

        # Minimize error
        manager.applications[app_name].datasets["Set1"].handle_actionMinimize_Error()

    def test_rouse_theory_fit(self, lve_app, pi_linear_dir):
        """Test Rouse theory creation and fitting."""
        manager, app_name = lve_app

        data_file = pi_linear_dir / "PI_13.5k_T-35.tts"
        skip_if_no_data(data_file)

        # Load data
        manager.applications[app_name].new_tables_from_files([str(data_file)])

        # Create Rouse theory
        manager.applications[app_name].datasets["Set1"].new_theory("Rouse")

        # Verify theory was created
        assert len(manager.applications[app_name].datasets["Set1"].theories) > 0

        # Minimize error
        manager.applications[app_name].datasets["Set1"].handle_actionMinimize_Error()


@pytest.mark.gui
@pytest.mark.slow
@pytest.mark.requires_data
class TestLVECarreauYasuda:
    """Test LVE application with Carreau-Yasuda theory."""

    def test_carreau_yasuda_theory_fit(self, lve_app, pi_linear_dir):
        """Test Carreau-Yasuda theory with eta* view."""
        manager, app_name = lve_app

        data_file = pi_linear_dir / "PI_483.1k_T-35.tts"
        skip_if_no_data(data_file)

        # Load data
        manager.applications[app_name].new_tables_from_files([str(data_file)])

        # Switch view to log(eta*)
        manager.applications[app_name].view_switch("logetastar")

        # Create Carreau-Yasuda theory
        manager.applications[app_name].datasets["Set1"].new_theory("Carreau-Yasuda")

        # Minimize error
        manager.applications[app_name].datasets["Set1"].handle_actionMinimize_Error()


@pytest.mark.gui
@pytest.mark.slow
@pytest.mark.requires_data
class TestLVEMaxwell:
    """Test LVE application with Maxwell Modes theory."""

    def test_maxwell_modes_theory_fit(self, lve_app, pi_linear_dir):
        """Test Maxwell Modes theory fitting."""
        manager, app_name = lve_app

        data_file = pi_linear_dir / "PI_483.1k_T-35.tts"
        skip_if_no_data(data_file)

        # Load data
        manager.applications[app_name].new_tables_from_files([str(data_file)])

        # Create Maxwell Modes theory
        manager.applications[app_name].datasets["Set1"].new_theory("Maxwell Modes")

        # Minimize error
        manager.applications[app_name].datasets["Set1"].handle_actionMinimize_Error()


@pytest.mark.gui
@pytest.mark.slow
@pytest.mark.requires_data
class TestLVEDTDStars:
    """Test LVE application with DTD Stars theory."""

    def test_dtd_stars_theory_fit(self, lve_app, pi_star_dir):
        """Test DTD Stars theory fitting for star polymers."""
        manager, app_name = lve_app

        data_files = [
            pi_star_dir / "S6Z8.1T40.tts",
            pi_star_dir / "S6Z12T40.tts",
            pi_star_dir / "S6Z16T40.tts",
        ]

        # Check if data files exist
        for f in data_files:
            skip_if_no_data(f)

        # Load data
        file_paths = [str(f) for f in data_files]
        manager.applications[app_name].new_tables_from_files(file_paths)

        # Create DTD Stars theory
        manager.applications[app_name].datasets["Set1"].new_theory("DTD Stars")

        # Minimize error
        manager.applications[app_name].datasets["Set1"].handle_actionMinimize_Error()
