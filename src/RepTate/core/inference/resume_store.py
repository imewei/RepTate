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
        """Save inference resume state to a JSON file.

        Persists the resume checkpoint as a JSON file with sorted keys and indentation
        for readability. The file is named using the result_id.

        Args:
            result_id: Unique identifier for the inference run, used as the filename.
            payload: Dictionary containing resume state data such as warm-start parameters,
                MCMC state, or chain metadata. Must be JSON-serializable.

        Returns:
            Path to the saved JSON file in the format {base_dir}/{result_id}.json.
        """
        path = self.base_dir / f"{result_id}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        return path

    def load(self, result_id: str) -> dict[str, Any]:
        """Load inference resume state from a JSON file.

        Reads the resume checkpoint from a JSON file identified by result_id.

        Args:
            result_id: Unique identifier for the inference run, used to locate the file.

        Returns:
            Dictionary containing the resume state data that was previously saved.

        Raises:
            FileNotFoundError: If no resume file exists for the given result_id.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        path = self.base_dir / f"{result_id}.json"
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
