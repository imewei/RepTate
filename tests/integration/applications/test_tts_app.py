"""Integration tests for TTS (Time-Temperature Superposition) application.

Tests validate complete TTS application workflows including:
- Loading oscillatory data at multiple temperatures
- Creating and fitting TTS theories (Automatic TTS Shift, WLF Shift)
- Time-temperature superposition operations

Migrated from legacy RepTate_Test_TTS.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import skip_if_no_data


@pytest.mark.gui
@pytest.mark.slow
@pytest.mark.requires_data
class TestTTSShift:
    """Test TTS application with shift theories."""

    def test_load_oscillatory_data(self, tts_app, pi_linear_dir):
        """Test loading oscillatory data files for TTS analysis."""
        manager, app_name = tts_app

        osc_dir = pi_linear_dir / "osc"

        data_files = [
            osc_dir / "PI223k-14c_-45C_FS2_PP10.osc",
            osc_dir / "PI223k-14c_-40C_FS_PP10.osc",
            osc_dir / "PI223k-14c_-30C_FS_PP10.osc",
            osc_dir / "PI223k-14_-10C_FS_PP10.osc",
            osc_dir / "PI223k-14c_-20C_FS_PP10.osc",
            osc_dir / "PI223k-14b_0C_FS4_PP10.osc",
            osc_dir / "PI223k-14_10C_FS_PP10.osc",
            osc_dir / "PI223k-14b_25C_FS3_PP10.osc",
            osc_dir / "PI223k-14_25C_FS3_PP10.osc",
            osc_dir / "PI223k-14c_30C_FS3_PP10.osc",
            osc_dir / "PI223k-14_40C_FS_PP10.osc",
            osc_dir / "PI223k-14_50C_FS_PP10.osc",
        ]

        # Check if data files exist
        for f in data_files:
            skip_if_no_data(f)

        # Load data files
        file_paths = [str(f) for f in data_files]
        manager.applications[app_name].new_tables_from_files(file_paths)

        # Verify dataset was created
        assert "Set1" in manager.applications[app_name].datasets

    def test_automatic_tts_shift(self, tts_app, pi_linear_dir):
        """Test Automatic TTS Shift theory."""
        manager, app_name = tts_app

        osc_dir = pi_linear_dir / "osc"

        # Use subset of files for faster test
        data_files = [
            osc_dir / "PI223k-14c_-30C_FS_PP10.osc",
            osc_dir / "PI223k-14c_-20C_FS_PP10.osc",
            osc_dir / "PI223k-14b_0C_FS4_PP10.osc",
            osc_dir / "PI223k-14_10C_FS_PP10.osc",
        ]

        for f in data_files:
            skip_if_no_data(f)

        # Load data
        file_paths = [str(f) for f in data_files]
        manager.applications[app_name].new_tables_from_files(file_paths)

        # Create Automatic TTS Shift theory
        manager.applications[app_name].datasets["Set1"].new_theory("Automatic TTS Shift")

        # Minimize error
        manager.applications[app_name].datasets["Set1"].handle_actionMinimize_Error()

    def test_wlf_shift(self, tts_app, pi_linear_dir):
        """Test WLF Shift theory."""
        manager, app_name = tts_app

        osc_dir = pi_linear_dir / "osc"

        # Use subset of files for faster test
        data_files = [
            osc_dir / "PI223k-14c_-30C_FS_PP10.osc",
            osc_dir / "PI223k-14c_-20C_FS_PP10.osc",
            osc_dir / "PI223k-14b_0C_FS4_PP10.osc",
            osc_dir / "PI223k-14_10C_FS_PP10.osc",
        ]

        for f in data_files:
            skip_if_no_data(f)

        # Load data
        file_paths = [str(f) for f in data_files]
        manager.applications[app_name].new_tables_from_files(file_paths)

        # Create WLF Shift theory
        manager.applications[app_name].datasets["Set1"].new_theory("WLF Shift")

        # Minimize error
        manager.applications[app_name].datasets["Set1"].handle_actionMinimize_Error()
