"""Persistence for inference resume state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ResumeStore:
    """Store/restore resume checkpoints for inference runs."""

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, result_id: str, payload: dict[str, Any]) -> Path:
        path = self.base_dir / f"{result_id}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        return path

    def load(self, result_id: str) -> dict[str, Any]:
        path = self.base_dir / f"{result_id}.json"
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
