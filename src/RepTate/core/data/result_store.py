"""Persistence helpers for fit and posterior results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from RepTate.core.models.results import FitResultRecord, PosteriorResultRecord


class ResultStore:
    """Store and load deterministic fit and posterior results to disk."""

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.fit_dir = self.base_dir / "fits"
        self.posterior_dir = self.base_dir / "posteriors"
        self.fit_dir.mkdir(exist_ok=True)
        self.posterior_dir.mkdir(exist_ok=True)

    def save_fit(self, record: FitResultRecord) -> Path:
        """Save a deterministic fit result record to disk as JSON.

        Persists the fit result to the fits subdirectory with a filename based
        on the result_id. The record is serialized to JSON with sorted keys and
        indentation for readability.

        Args:
            record: FitResultRecord containing fit parameters, metrics, and metadata.

        Returns:
            Path to the saved JSON file in the fits directory.

        Raises:
            OSError: If the file cannot be written due to permission or disk errors.
        """
        path = self.fit_dir / f"{record.result_id}.json"
        _write_json(path, record.__dict__)
        return path

    def load_fit(self, result_id: str) -> FitResultRecord:
        """Load a deterministic fit result record from disk.

        Reads a previously saved fit result from the fits subdirectory and
        reconstructs the FitResultRecord object from the JSON payload.

        Args:
            result_id: Unique identifier for the fit result to load.

        Returns:
            FitResultRecord reconstructed from the saved JSON data.

        Raises:
            FileNotFoundError: If no fit result exists with the given result_id.
            json.JSONDecodeError: If the file contains invalid JSON.
            TypeError: If the JSON structure does not match FitResultRecord fields.
        """
        path = self.fit_dir / f"{result_id}.json"
        payload = _read_json(path)
        return FitResultRecord(**payload)

    def save_posterior(self, record: PosteriorResultRecord) -> Path:
        """Save a Bayesian posterior result record to disk as JSON.

        Persists the posterior result to the posteriors subdirectory with a
        filename based on the result_id. The record is serialized to JSON with
        sorted keys and indentation for readability.

        Args:
            record: PosteriorResultRecord containing posterior samples, diagnostics,
                and metadata.

        Returns:
            Path to the saved JSON file in the posteriors directory.

        Raises:
            OSError: If the file cannot be written due to permission or disk errors.
        """
        path = self.posterior_dir / f"{record.result_id}.json"
        _write_json(path, record.__dict__)
        return path

    def load_posterior(self, result_id: str) -> PosteriorResultRecord:
        """Load a Bayesian posterior result record from disk.

        Reads a previously saved posterior result from the posteriors subdirectory
        and reconstructs the PosteriorResultRecord object from the JSON payload.

        Args:
            result_id: Unique identifier for the posterior result to load.

        Returns:
            PosteriorResultRecord reconstructed from the saved JSON data.

        Raises:
            FileNotFoundError: If no posterior result exists with the given result_id.
            json.JSONDecodeError: If the file contains invalid JSON.
            TypeError: If the JSON structure does not match PosteriorResultRecord fields.
        """
        path = self.posterior_dir / f"{result_id}.json"
        payload = _read_json(path)
        return PosteriorResultRecord(**payload)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)
