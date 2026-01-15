"""Integration tests for NLVE (Non-Linear Viscoelasticity) application.

Tests validate complete NLVE application workflows including:
- Loading shear and extensional flow data
- Creating NLVE theories (Rolie-Poly, Pom-Pom, Rolie-Double-Poly)
- Mode copying from LVE Maxwell fits
- Start-up shear and uniaxial extension analysis

Migrated from legacy RepTate_Test_NLVE.py and RepTate_Test_NLVE_RDP.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import skip_if_no_data


@pytest.mark.gui
@pytest.mark.slow
@pytest.mark.requires_data
class TestNLVERoliePoly:
    """Test NLVE application with Rolie-Poly theory."""

    def test_startup_shear_roliepoly(self, application_manager, dow_dir):
        """Test Rolie-Poly theory for start-up shear flow."""
        manager = application_manager

        # First create LVE app to get Maxwell modes
        manager.handle_new_app("LVE")
        lve_name = f"LVE{manager.application_counter}"

        lve_data = dow_dir / "Linear_Rheology_TTS" / "DOWLDPEL150R_160C.tts"
        skip_if_no_data(lve_data)

        # Load LVE data and fit Maxwell modes
        manager.applications[lve_name].new_tables_from_files([str(lve_data)])
        manager.applications[lve_name].datasets["Set1"].new_theory("Maxwell Modes")
        manager.applications[lve_name].datasets["Set1"].handle_actionMinimize_Error()

        # Create NLVE app for shear data
        manager.handle_new_app("NLVE")
        nlve_name = f"NLVE{manager.application_counter}"

        shear_dir = dow_dir / "Non-Linear_Rheology" / "Start-up_Shear"
        shear_files = [
            shear_dir / "My_dow150-160-1 shear.shear",
            shear_dir / "My_dow150-160-01 shear.shear",
            shear_dir / "My_dow150-160-001 shear.shear",
            shear_dir / "My_dow150-160-3 shear.shear",
            shear_dir / "My_dow150-160-03 shear.shear",
            shear_dir / "My_dow150-160-003 shear.shear",
            shear_dir / "My_dow150-160-0003 shear.shear",
        ]

        for f in shear_files:
            skip_if_no_data(f)

        # Load shear data
        file_paths = [str(f) for f in shear_files]
        manager.applications[nlve_name].new_tables_from_files(file_paths)

        # Create Rolie-Poly theory
        manager.applications[nlve_name].datasets["Set1"].new_theory("Rolie-Poly")

        # Copy Maxwell modes from LVE fit
        rp_theory = list(manager.applications[nlve_name].datasets["Set1"].theories.values())[0]
        rp_theory.do_copy_modes(f"{lve_name}.Set1.MM1")

        # Minimize error
        manager.applications[nlve_name].datasets["Set1"].handle_actionMinimize_Error()

    def test_uniaxial_extension_roliepoly(self, application_manager, dow_dir):
        """Test Rolie-Poly theory for uniaxial extension flow."""
        manager = application_manager

        # First create LVE app to get Maxwell modes
        manager.handle_new_app("LVE")
        lve_name = f"LVE{manager.application_counter}"

        lve_data = dow_dir / "Linear_Rheology_TTS" / "DOWLDPEL150R_160C.tts"
        skip_if_no_data(lve_data)

        # Load LVE data and fit Maxwell modes
        manager.applications[lve_name].new_tables_from_files([str(lve_data)])
        manager.applications[lve_name].datasets["Set1"].new_theory("Maxwell Modes")
        manager.applications[lve_name].datasets["Set1"].handle_actionMinimize_Error()

        # Create NLVE app for extension data
        manager.handle_new_app("NLVE")
        nlve_name = f"NLVE{manager.application_counter}"

        ext_dir = dow_dir / "Non-Linear_Rheology" / "Start-up_extension"
        ext_files = [
            ext_dir / "My_dow150-160-01.uext",
            ext_dir / "My_dow150-160-001.uext",
            ext_dir / "My_dow150-160-0001.uext",
            ext_dir / "My_dow150-160-03.uext",
            ext_dir / "My_dow150-160-003.uext",
            ext_dir / "My_dow150-160-0003.uext",
        ]

        for f in ext_files:
            skip_if_no_data(f)

        # Load extension data
        file_paths = [str(f) for f in ext_files]
        manager.applications[nlve_name].new_tables_from_files(file_paths)

        # Create Rolie-Poly theory
        manager.applications[nlve_name].datasets["Set1"].new_theory("Rolie-Poly")

        # Select extensional flow
        rp_theory = list(manager.applications[nlve_name].datasets["Set1"].theories.values())[0]
        rp_theory.select_extensional_flow()

        # Copy Maxwell modes from LVE fit
        rp_theory.do_copy_modes(f"{lve_name}.Set1.MM1")

        # Minimize error
        manager.applications[nlve_name].datasets["Set1"].handle_actionMinimize_Error()


@pytest.mark.gui
@pytest.mark.slow
@pytest.mark.requires_data
class TestNLVEPomPom:
    """Test NLVE application with Pom-Pom theory."""

    def test_pompom_theory(self, application_manager, dow_dir):
        """Test Pom-Pom theory creation for extension data."""
        manager = application_manager

        # First create LVE app to get Maxwell modes
        manager.handle_new_app("LVE")
        lve_name = f"LVE{manager.application_counter}"

        lve_data = dow_dir / "Linear_Rheology_TTS" / "DOWLDPEL150R_160C.tts"
        skip_if_no_data(lve_data)

        # Load LVE data and fit Maxwell modes
        manager.applications[lve_name].new_tables_from_files([str(lve_data)])
        manager.applications[lve_name].datasets["Set1"].new_theory("Maxwell Modes")
        manager.applications[lve_name].datasets["Set1"].handle_actionMinimize_Error()

        # Create NLVE app
        manager.handle_new_app("NLVE")
        nlve_name = f"NLVE{manager.application_counter}"

        ext_dir = dow_dir / "Non-Linear_Rheology" / "Start-up_extension"
        ext_file = ext_dir / "My_dow150-160-01.uext"
        skip_if_no_data(ext_file)

        # Load extension data
        manager.applications[nlve_name].new_tables_from_files([str(ext_file)])

        # Create Pom-Pom theory
        manager.applications[nlve_name].datasets["Set1"].new_theory("Pom-Pom")

        # Verify theory was created
        assert len(manager.applications[nlve_name].datasets["Set1"].theories) > 0


@pytest.mark.gui
@pytest.mark.slow
@pytest.mark.requires_data
class TestNLVERolieDoublePoly:
    """Test NLVE application with Rolie-Double-Poly theory."""

    def test_rolie_double_poly_shear(self, nlve_app, dow_dir):
        """Test Rolie-Double-Poly theory for shear data."""
        manager, app_name = nlve_app

        shear_dir = dow_dir / "Non-Linear_Rheology" / "Start-up_Shear"
        shear_files = [
            shear_dir / "My_dow150-160-1 shear.shear",
            shear_dir / "My_dow150-160-01 shear.shear",
            shear_dir / "My_dow150-160-001 shear.shear",
        ]

        for f in shear_files:
            skip_if_no_data(f)

        # Load shear data
        file_paths = [str(f) for f in shear_files]
        manager.applications[app_name].new_tables_from_files(file_paths)

        # Create Rolie-Double-Poly theory
        manager.applications[app_name].datasets["Set1"].new_theory("Rolie-Double-Poly")

        # Verify theory was created
        assert len(manager.applications[app_name].datasets["Set1"].theories) > 0

    def test_rolie_double_poly_extension(self, nlve_app, nlve_extension_dir):
        """Test Rolie-Double-Poly theory for extension data."""
        manager, app_name = nlve_app

        ext_files = [
            nlve_extension_dir / "Minegishi_spiked_PS_0_572.uext",
            nlve_extension_dir / "Minegishi_spiked_PS_0_013.uext",
            nlve_extension_dir / "Minegishi_spiked_PS_0_047.uext",
            nlve_extension_dir / "Minegishi_spiked_PS_0_097.uext",
        ]

        for f in ext_files:
            skip_if_no_data(f)

        # Load extension data
        file_paths = [str(f) for f in ext_files]
        manager.applications[app_name].new_tables_from_files(file_paths)

        # Create Rolie-Double-Poly theory
        manager.applications[app_name].datasets["Set1"].new_theory("Rolie-Double-Poly")

        # Verify theory was created
        assert len(manager.applications[app_name].datasets["Set1"].theories) > 0
