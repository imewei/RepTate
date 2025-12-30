"""Safe loader for linlin data with legacy migration."""
from __future__ import annotations

import logging
import os
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _load_v2_data(file_obj: np.lib.npyio.NpzFile) -> List[np.ndarray]:
    if "data" in file_obj.files:
        return list(file_obj["data"])
    data_keys = sorted(key for key in file_obj.files if key.startswith("data_"))
    return [file_obj[key] for key in data_keys]


def load_linlin_data(base_path: str) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    v2_path = os.path.join(base_path, "linlin_v2.npz")
    if os.path.exists(v2_path):
        with np.load(v2_path, allow_pickle=False) as f:
            return f["Z"], f["cnu"], _load_v2_data(f)

    legacy_path = os.path.join(base_path, "linlin.npz")
    with np.load(legacy_path, allow_pickle=True) as f:
        Z = f["Z"]
        cnu = f["cnu"]
        data = list(f["data"])
    try:
        payload = {"Z": Z, "cnu": cnu}
        for index, table in enumerate(data):
            payload[f"data_{index:04d}"] = table
        np.savez_compressed(v2_path, **payload)
        logger.info("Migrated linlin data to %s", v2_path)
    except OSError as exc:
        logger.warning("Failed to persist linlin v2 data: %s", exc)
    return Z, cnu, data
