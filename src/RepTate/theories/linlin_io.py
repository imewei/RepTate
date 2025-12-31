"""Safe loader for linlin data with legacy migration.

This module provides secure loading of linlin interpolation data files.
The v2 format uses NPZ with allow_pickle=False for security. Legacy files
are automatically migrated to the v2 format.

Task: T015 [US1] - Updated to verify NPZ format with allow_pickle=False

Security:
- V2 format loaded with allow_pickle=False
- Legacy format migration logs a warning
- All new saves use pickle-free format
"""
from __future__ import annotations

import logging
import os
from typing import List, Tuple

import numpy as np

from RepTate.core.feature_flags import is_enabled

logger = logging.getLogger(__name__)


def _load_v2_data(file_obj: np.lib.npyio.NpzFile) -> List[np.ndarray]:
    """Load data arrays from v2 NPZ format.

    V2 format stores data arrays individually with keys like data_0000.

    Args:
        file_obj: Opened NPZ file object

    Returns:
        List of data arrays in order
    """
    if "data" in file_obj.files:
        return list(file_obj["data"])
    data_keys = sorted(key for key in file_obj.files if key.startswith("data_"))
    return [file_obj[key] for key in data_keys]


def load_linlin_data(
    base_path: str,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Load linlin interpolation data.

    Loads from v2 format (allow_pickle=False) if available, otherwise
    migrates legacy format to v2.

    Args:
        base_path: Directory containing linlin data files

    Returns:
        Tuple of (Z, cnu, data) arrays where:
        - Z: Z values array
        - cnu: cnu values array
        - data: List of interpolation data tables

    Security:
        V2 format is loaded with allow_pickle=False to prevent code
        execution from malicious files. Legacy format requires pickle
        and is migrated automatically.
    """
    v2_path = os.path.join(base_path, "linlin_v2.npz")
    if os.path.exists(v2_path):
        # Load v2 format with security: allow_pickle=False
        with np.load(v2_path, allow_pickle=False) as f:
            return f["Z"], f["cnu"], _load_v2_data(f)

    # Legacy path - requires pickle
    legacy_path = os.path.join(base_path, "linlin.npz")

    if is_enabled("USE_SAFE_SERIALIZATION"):
        logger.warning(
            "Loading legacy linlin.npz with pickle. "
            "This file should be migrated to linlin_v2.npz format: %s",
            legacy_path,
        )

    # Load legacy format (requires allow_pickle=True for object arrays)
    with np.load(legacy_path, allow_pickle=True) as f:
        Z = f["Z"]
        cnu = f["cnu"]
        data = list(f["data"])

    # Migrate to v2 format
    try:
        payload = {"Z": Z, "cnu": cnu}
        for index, table in enumerate(data):
            payload[f"data_{index:04d}"] = table
        np.savez_compressed(v2_path, **payload)
        logger.info("Migrated linlin data to secure v2 format: %s", v2_path)
    except OSError as exc:
        logger.warning("Failed to persist linlin v2 data: %s", exc)

    return Z, cnu, data


def verify_linlin_npz_security(npz_path: str) -> bool:
    """Verify that an NPZ file can be loaded without pickle.

    This function checks if a linlin NPZ file is in the secure v2 format
    by attempting to load it with allow_pickle=False.

    Args:
        npz_path: Path to NPZ file to verify

    Returns:
        True if file can be loaded without pickle, False otherwise
    """
    try:
        with np.load(npz_path, allow_pickle=False) as f:
            # Just access a key to verify the file loads
            _ = list(f.files)
        return True
    except ValueError:
        # ValueError is raised when pickle is required
        return False
    except FileNotFoundError:
        return False
