"""Helpers for loading/saving the materials database without pickle."""
from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

import numpy as np

try:
    from RepTate.tools import polymer_data
except ImportError:  # allow standalone usage from tools directory
    import polymer_data

logger = logging.getLogger(__name__)


@contextmanager
def _legacy_sys_path(path: str) -> None:
    added = False
    if path not in sys.path:
        sys.path.insert(0, path)
        added = True
    try:
        yield
    finally:
        if added:
            sys.path.remove(path)


def _coerce_polymer(data: dict[str, Any]) -> polymer_data.polymer:
    if "data" in data and isinstance(data["data"], dict):
        data = data["data"]
    return polymer_data.polymer(**data)


def _serialize_db(db: Dict[str, polymer_data.polymer]) -> Dict[str, dict[str, Any]]:
    return {key: value.data for key, value in db.items()}


def load_materials_json(path: str) -> Dict[str, polymer_data.polymer]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return {key: _coerce_polymer(data) for key, data in payload.items()}


def load_legacy_npy(path: str) -> Dict[str, polymer_data.polymer]:
    legacy_dir = os.path.dirname(os.path.abspath(path))
    with _legacy_sys_path(legacy_dir):
        legacy = np.load(path, allow_pickle=True).item()
    if not isinstance(legacy, dict):
        raise ValueError(f"Expected legacy dict in {path}")
    result: Dict[str, polymer_data.polymer] = {}
    for key, value in legacy.items():
        payload = value.__dict__ if hasattr(value, "__dict__") else value
        result[key] = _coerce_polymer(payload)
    return result


def write_materials_json(db: Dict[str, polymer_data.polymer], path: str) -> None:
    payload = _serialize_db(db)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
    logger.info("Wrote materials database JSON to %s", path)


def load_default_materials(base_path: str) -> Dict[str, polymer_data.polymer]:
    json_path = os.path.join(base_path, "materials_database.json")
    if os.path.exists(json_path):
        return load_materials_json(json_path)

    legacy_path = os.path.join(base_path, "materials_database.npy")
    if os.path.exists(legacy_path):
        logger.warning("Using legacy materials database at %s", legacy_path)
        db = load_legacy_npy(legacy_path)
        try:
            write_materials_json(db, json_path)
        except OSError as exc:
            logger.warning("Failed to persist materials JSON: %s", exc)
        return db

    logger.warning("No materials database found in %s", base_path)
    return {}


def load_user_materials(appdata_path: str, home_path: str | None = None) -> Dict[str, polymer_data.polymer]:
    appdata_dir = Path(appdata_path)
    appdata_dir.mkdir(parents=True, exist_ok=True)
    json_path = appdata_dir / "user_database.json"
    if json_path.exists():
        return load_materials_json(str(json_path))

    legacy_candidates = [appdata_dir / "user_database.npy"]
    if home_path:
        legacy_candidates.append(Path(home_path) / "user_database.npy")

    for legacy_path in legacy_candidates:
        if legacy_path.exists():
            logger.warning("Migrating legacy user database from %s", legacy_path)
            db = load_legacy_npy(str(legacy_path))
            try:
                write_materials_json(db, str(json_path))
            except OSError as exc:
                logger.warning("Failed to persist user materials JSON: %s", exc)
            return db

    return {}
