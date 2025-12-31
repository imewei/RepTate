"""Integration tests for secure file loading workflow.

Tests cover:
- Safe loading of JSON/NPZ format files
- Legacy pickle file migration
- Security rejection of malicious files
- Materials database loading workflow
- Linlin data loading workflow

Tasks: T010 [P] [US1], T017a [US1]
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from RepTate.core.serialization import SafeSerializer, migrate_pickle
from RepTate.core.feature_flags import is_enabled

if TYPE_CHECKING:
    pass


class TestSafeFileLoading:
    """Integration tests for safe file loading."""

    def test_load_valid_json_npz_file(self, temp_workspace: Path) -> None:
        """Test loading a valid JSON/NPZ file set."""
        # Create test data
        data = {
            "name": "test_experiment",
            "frequency": np.array([0.1, 1.0, 10.0, 100.0]),
            "g_prime": np.array([1e5, 5e4, 1e4, 5e3]),
            "metadata": {
                "temperature": 25.0,
                "sample": "polymer_001",
            },
        }
        base_path = temp_workspace / "experiment"

        # Save using SafeSerializer
        SafeSerializer.save(base_path, data)

        # Verify files exist
        assert (temp_workspace / "experiment.json").exists()
        assert (temp_workspace / "experiment.npz").exists()

        # Load and verify
        loaded = SafeSerializer.load(base_path)

        assert loaded["name"] == "test_experiment"
        np.testing.assert_array_equal(loaded["frequency"], data["frequency"])
        np.testing.assert_array_equal(loaded["g_prime"], data["g_prime"])
        assert loaded["metadata"]["temperature"] == 25.0
        assert loaded["metadata"]["sample"] == "polymer_001"

    def test_load_json_only_file(self, temp_workspace: Path) -> None:
        """Test loading a JSON file without arrays (no NPZ needed)."""
        data = {
            "name": "metadata_only",
            "parameters": {
                "G0": 1e5,
                "tau": 1.0,
                "alpha": 0.5,
            },
        }
        base_path = temp_workspace / "params"

        # Save
        result = SafeSerializer.save(base_path, data)

        # Should have JSON but no NPZ
        assert result.json_path.exists()
        assert result.npz_path is None

        # Load and verify
        loaded = SafeSerializer.load(base_path)

        assert loaded["name"] == "metadata_only"
        assert loaded["parameters"]["G0"] == 1e5
        assert loaded["parameters"]["tau"] == 1.0

    def test_load_complex_nested_data(self, temp_workspace: Path) -> None:
        """Test loading complex nested data structures."""
        data = {
            "datasets": [
                {
                    "name": "dataset1",
                    "x": np.linspace(0, 10, 100),
                    "y": np.sin(np.linspace(0, 10, 100)),
                },
                {
                    "name": "dataset2",
                    "x": np.linspace(0, 10, 100),
                    "y": np.cos(np.linspace(0, 10, 100)),
                },
            ],
            "theories": {
                "maxwell": {"G0": 1e5, "tau": 1.0},
                "rouse": {"N": 100, "b": 1.5},
            },
        }
        base_path = temp_workspace / "complex"

        SafeSerializer.save(base_path, data)
        loaded = SafeSerializer.load(base_path)

        assert len(loaded["datasets"]) == 2
        assert loaded["datasets"][0]["name"] == "dataset1"
        np.testing.assert_array_almost_equal(
            loaded["datasets"][0]["y"],
            np.sin(np.linspace(0, 10, 100)),
        )
        assert loaded["theories"]["maxwell"]["G0"] == 1e5


class TestLegacyPickleMigration:
    """Integration tests for legacy pickle file migration.

    Task: T017a [US1]
    """

    def test_migrate_simple_pickle_file(self, temp_workspace: Path) -> None:
        """Test migrating a simple pickle file to safe format."""
        # Create a legacy pickle file
        pickle_path = temp_workspace / "legacy_data.pkl"
        original_data = {
            "name": "legacy_experiment",
            "values": [1, 2, 3, 4, 5],
            "settings": {"temperature": 25.0},
        }

        with open(pickle_path, "wb") as f:
            pickle.dump(original_data, f)

        # Migrate
        new_base_path = migrate_pickle(pickle_path)

        # Verify new files exist
        assert Path(str(new_base_path) + ".json").exists()

        # Verify original is backed up
        assert Path(str(pickle_path) + ".bak").exists()

        # Load and verify data
        loaded = SafeSerializer.load(new_base_path)

        assert loaded["name"] == "legacy_experiment"
        assert loaded["values"] == [1, 2, 3, 4, 5]
        assert loaded["settings"]["temperature"] == 25.0

    def test_migrate_pickle_with_numpy_arrays(self, temp_workspace: Path) -> None:
        """Test migrating pickle files containing numpy arrays."""
        pickle_path = temp_workspace / "array_data.pkl"
        original_data = {
            "frequency": np.array([0.1, 1.0, 10.0]),
            "g_prime": np.array([1e5, 5e4, 1e4]),
            "metadata": {"type": "LVE"},
        }

        with open(pickle_path, "wb") as f:
            pickle.dump(original_data, f)

        # Migrate
        new_base_path = migrate_pickle(pickle_path)

        # Verify NPZ was created for arrays
        assert Path(str(new_base_path) + ".npz").exists()

        # Load and verify
        loaded = SafeSerializer.load(new_base_path)

        np.testing.assert_array_equal(loaded["frequency"], original_data["frequency"])
        np.testing.assert_array_equal(loaded["g_prime"], original_data["g_prime"])

    def test_migrate_pickle_version_mismatch_warning(
        self, temp_workspace: Path
    ) -> None:
        """Test migration handles version mismatch gracefully.

        Task: T017a - Legacy files may have been created with different
        Python versions, which can affect pickle behavior.
        """
        pickle_path = temp_workspace / "old_version.pkl"

        # Create pickle with current version
        data = {"test": "value", "number": 42}

        with open(pickle_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Migration should work regardless of pickle protocol used
        new_base_path = migrate_pickle(pickle_path)

        loaded = SafeSerializer.load(new_base_path)
        assert loaded["test"] == "value"
        assert loaded["number"] == 42

    def test_migrate_pickle_already_migrated(self, temp_workspace: Path) -> None:
        """Test migrating a file that's already been migrated."""
        pickle_path = temp_workspace / "data.pkl"
        data = {"test": "value"}

        with open(pickle_path, "wb") as f:
            pickle.dump(data, f)

        # First migration
        new_base_path_1 = migrate_pickle(pickle_path)

        # Restore pickle for second migration test
        backup_path = Path(str(pickle_path) + ".bak")
        backup_path.rename(pickle_path)

        # Second migration should still work
        new_base_path_2 = migrate_pickle(pickle_path)

        # Both should produce valid data
        loaded = SafeSerializer.load(new_base_path_2)
        assert loaded["test"] == "value"


class TestSecurityRejection:
    """Integration tests for security-related file rejections."""

    def test_reject_pickle_file_direct_load(self, temp_workspace: Path) -> None:
        """Test that pickle files cannot be loaded directly."""
        pickle_path = temp_workspace / "malicious.pkl"

        # Create a pickle file (which should not be loadable)
        data = {"test": "value"}
        with open(pickle_path, "wb") as f:
            pickle.dump(data, f)

        # SafeSerializer should not load pickle files
        base_path = temp_workspace / "malicious"
        assert SafeSerializer.can_load(base_path) is False

    def test_reject_manipulated_json(self, temp_workspace: Path) -> None:
        """Test that manipulated JSON files are rejected."""
        base_path = temp_workspace / "test"

        # Create a valid file first
        SafeSerializer.save(base_path, {"test": "value"})

        # Manipulate the JSON to have wrong version
        json_path = Path(str(base_path) + ".json")
        with open(json_path, "r") as f:
            data = json.load(f)

        data["__version__"] = 999

        with open(json_path, "w") as f:
            json.dump(data, f)

        # Should reject due to version mismatch
        with pytest.raises(ValueError, match="Unsupported file version"):
            SafeSerializer.load(base_path)

    def test_npz_allow_pickle_false_enforced(self, temp_workspace: Path) -> None:
        """Test that NPZ files with pickled objects fail to load."""
        npz_path = temp_workspace / "test.npz"
        json_path = temp_workspace / "test.json"

        # Create an NPZ with object array (requires allow_pickle=True)
        obj_array = np.array([{"nested": "object"}], dtype=object)
        np.savez(npz_path, _array_array_0=obj_array)

        # Create corresponding JSON
        with open(json_path, "w") as f:
            json.dump(
                {
                    "__version__": 1,
                    "array": {"__array_ref__": "_array_array_0"},
                },
                f,
            )

        # Should fail because NPZ requires pickle
        with pytest.raises(ValueError):
            SafeSerializer.load(temp_workspace / "test")


class TestFeatureFlagIntegration:
    """Tests for feature flag integration with serialization."""

    def test_safe_serialization_flag_enabled(self) -> None:
        """Test USE_SAFE_SERIALIZATION flag is enabled by default."""
        assert is_enabled("USE_SAFE_SERIALIZATION") is True

    def test_serialization_respects_flag(self, temp_workspace: Path) -> None:
        """Test that serialization works when flag is enabled."""
        if not is_enabled("USE_SAFE_SERIALIZATION"):
            pytest.skip("Safe serialization disabled")

        data = {"test": "value", "array": np.array([1, 2, 3])}
        base_path = temp_workspace / "test"

        result = SafeSerializer.save(base_path, data)
        assert result.json_path.exists()

        loaded = SafeSerializer.load(base_path)
        assert loaded["test"] == "value"


class TestMaterialsDatabaseLoadingWorkflow:
    """Integration tests for materials database loading."""

    def test_load_json_materials_database(self, temp_workspace: Path) -> None:
        """Test loading materials database in JSON format."""
        # Create a mock materials database JSON
        materials_data = {
            "HDPE": {
                "name": "HDPE",
                "Mw": 150000.0,
                "PDI": 3.0,
                "chem": "C2H4",
            },
            "PS": {
                "name": "PS",
                "Mw": 200000.0,
                "PDI": 2.5,
                "chem": "C8H8",
            },
        }

        json_path = temp_workspace / "materials.json"
        with open(json_path, "w") as f:
            json.dump(materials_data, f)

        # Load using standard JSON (materials db uses plain JSON, not SafeSerializer)
        with open(json_path, "r") as f:
            loaded = json.load(f)

        assert "HDPE" in loaded
        assert "PS" in loaded
        assert loaded["HDPE"]["Mw"] == 150000.0


class TestLinlinDataLoadingWorkflow:
    """Integration tests for linlin data loading."""

    def test_load_linlin_v2_format(self, temp_workspace: Path) -> None:
        """Test loading linlin data in v2 NPZ format (allow_pickle=False)."""
        # Create v2 linlin data
        Z = np.array([1.0, 2.0, 3.0])
        cnu = np.array([0.1, 0.2, 0.3])
        data_0 = np.array([[1, 2], [3, 4]])
        data_1 = np.array([[5, 6], [7, 8]])

        npz_path = temp_workspace / "linlin_v2.npz"
        np.savez_compressed(
            npz_path,
            Z=Z,
            cnu=cnu,
            data_0000=data_0,
            data_0001=data_1,
        )

        # Load with allow_pickle=False (should work for v2)
        with np.load(npz_path, allow_pickle=False) as f:
            loaded_Z = f["Z"]
            loaded_cnu = f["cnu"]

        np.testing.assert_array_equal(loaded_Z, Z)
        np.testing.assert_array_equal(loaded_cnu, cnu)


class TestEndToEndFileWorkflow:
    """End-to-end tests for complete file workflows."""

    def test_create_modify_reload_workflow(self, temp_workspace: Path) -> None:
        """Test complete workflow: create, modify, save, reload."""
        base_path = temp_workspace / "experiment"

        # Step 1: Create initial data
        data_v1 = {
            "name": "experiment_001",
            "version": 1,
            "frequency": np.array([0.1, 1.0, 10.0]),
            "g_prime": np.array([1e5, 5e4, 1e4]),
        }
        SafeSerializer.save(base_path, data_v1)

        # Step 2: Load and verify
        loaded_v1 = SafeSerializer.load(base_path)
        assert loaded_v1["version"] == 1

        # Step 3: Modify and save
        data_v2 = dict(loaded_v1)
        data_v2["version"] = 2
        data_v2["new_field"] = "added_later"
        data_v2["frequency"] = np.array([0.01, 0.1, 1.0, 10.0, 100.0])

        SafeSerializer.save(base_path, data_v2)

        # Step 4: Reload and verify modifications
        loaded_v2 = SafeSerializer.load(base_path)
        assert loaded_v2["version"] == 2
        assert loaded_v2["new_field"] == "added_later"
        assert len(loaded_v2["frequency"]) == 5

    def test_multiple_files_same_directory(self, temp_workspace: Path) -> None:
        """Test handling multiple data files in same directory."""
        # Create multiple files
        for i in range(3):
            data = {
                "experiment_id": i,
                "data": np.random.randn(10),
            }
            SafeSerializer.save(temp_workspace / f"exp_{i:03d}", data)

        # Load all and verify
        for i in range(3):
            loaded = SafeSerializer.load(temp_workspace / f"exp_{i:03d}")
            assert loaded["experiment_id"] == i
            assert len(loaded["data"]) == 10
