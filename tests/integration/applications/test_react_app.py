"""Integration tests for React (Reaction) application.

Tests validate complete React application workflows including:
- Loading reaction data files
- Creating React theories (Tobita CSTR, Tobita Batch, React Mix, Multi-Met CSTR)
- BOB Architecture theory

Migrated from legacy RepTate_Test_React.py and RepTate_Test_BoB_polyconf.py.
"""

from __future__ import annotations

from pathlib import Path
from time import sleep

import pytest

from .conftest import skip_if_no_data


@pytest.mark.gui
@pytest.mark.slow
@pytest.mark.requires_data
class TestReactTobita:
    """Test React application with Tobita theories."""

    def test_load_react_data(self, react_app, react_dir):
        """Test loading reaction data file."""
        manager, app_name = react_app

        data_file = react_dir / "out1.reac"
        skip_if_no_data(data_file)

        # Load data file
        manager.applications[app_name].new_tables_from_files([str(data_file)])

        # Verify dataset was created
        assert "Set1" in manager.applications[app_name].datasets

    def test_tobita_cstr_theory(self, react_app, react_dir):
        """Test Tobita CSTR theory creation."""
        manager, app_name = react_app

        data_file = react_dir / "out1.reac"
        skip_if_no_data(data_file)

        # Load data
        manager.applications[app_name].new_tables_from_files([str(data_file)])

        # Create Tobita CSTR theory
        manager.applications[app_name].datasets["Set1"].new_theory("Tobita CSTR")

        # Verify theory was created
        assert len(manager.applications[app_name].datasets["Set1"].theories) > 0

    def test_tobita_batch_theory(self, react_app, react_dir):
        """Test Tobita Batch theory creation."""
        manager, app_name = react_app

        data_file = react_dir / "out1.reac"
        skip_if_no_data(data_file)

        # Load data
        manager.applications[app_name].new_tables_from_files([str(data_file)])

        # Create Tobita Batch theory
        manager.applications[app_name].datasets["Set1"].new_theory("Tobita Batch")

        # Verify theory was created
        assert len(manager.applications[app_name].datasets["Set1"].theories) > 0


@pytest.mark.gui
@pytest.mark.slow
@pytest.mark.requires_data
class TestReactMix:
    """Test React application with React Mix theory."""

    def test_react_mix_theory(self, react_app, react_dir):
        """Test React Mix theory creation."""
        manager, app_name = react_app

        data_file = react_dir / "out1.reac"
        skip_if_no_data(data_file)

        # Load data
        manager.applications[app_name].new_tables_from_files([str(data_file)])

        # Create React Mix theory
        manager.applications[app_name].datasets["Set1"].new_theory("React Mix")

        # Verify theory was created
        assert len(manager.applications[app_name].datasets["Set1"].theories) > 0


@pytest.mark.gui
@pytest.mark.slow
@pytest.mark.requires_data
class TestReactMultiMet:
    """Test React application with Multi-Met CSTR theory."""

    def test_multi_met_cstr_theory(self, react_app, react_dir):
        """Test Multi-Met CSTR theory creation."""
        manager, app_name = react_app

        data_file = react_dir / "out1.reac"
        skip_if_no_data(data_file)

        # Load data
        manager.applications[app_name].new_tables_from_files([str(data_file)])

        # Create Multi-Met CSTR theory
        manager.applications[app_name].datasets["Set1"].new_theory("Multi-Met CSTR")

        # Verify theory was created
        assert len(manager.applications[app_name].datasets["Set1"].theories) > 0


@pytest.mark.gui
@pytest.mark.slow
@pytest.mark.requires_data
class TestReactBOBArchitecture:
    """Test React application with BOB Architecture theory."""

    def test_bob_architecture_theory(self, react_app, react_dir):
        """Test BOB Architecture theory creation."""
        manager, app_name = react_app

        data_file = react_dir / "out1.reac"
        skip_if_no_data(data_file)

        # Load data
        manager.applications[app_name].new_tables_from_files([str(data_file)])

        # Create BOB Architecture theory
        manager.applications[app_name].datasets["Set1"].new_theory("BOB Architecture")

        # Verify theory was created
        assert len(manager.applications[app_name].datasets["Set1"].theories) > 0


@pytest.mark.gui
@pytest.mark.slow
@pytest.mark.requires_data
class TestReactMultipleTheories:
    """Test React application with multiple theories."""

    def test_create_multiple_react_theories(self, react_app, react_dir):
        """Test creating multiple React theories sequentially."""
        manager, app_name = react_app

        data_file = react_dir / "out1.reac"
        skip_if_no_data(data_file)

        # Load data
        manager.applications[app_name].new_tables_from_files([str(data_file)])

        # Create multiple theories with small delays
        theories_to_create = [
            "Tobita CSTR",
            "Tobita Batch",
            "React Mix",
            "Multi-Met CSTR",
        ]

        for theory_name in theories_to_create:
            manager.applications[app_name].datasets["Set1"].new_theory(theory_name)
            # Small delay between theory creation
            sleep(0.1)

        # Verify all theories were created
        assert len(manager.applications[app_name].datasets["Set1"].theories) >= len(theories_to_create)
