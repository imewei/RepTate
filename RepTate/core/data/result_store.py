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
        path = self.fit_dir / f"{record.result_id}.json"
        _write_json(path, record.__dict__)
        return path

    def load_fit(self, result_id: str) -> FitResultRecord:
        path = self.fit_dir / f"{result_id}.json"
        payload = _read_json(path)
        return FitResultRecord(**payload)

    def save_posterior(self, record: PosteriorResultRecord) -> Path:
        path = self.posterior_dir / f"{record.result_id}.json"
        _write_json(path, record.__dict__)
        return path

    def load_posterior(self, result_id: str) -> PosteriorResultRecord:
        path = self.posterior_dir / f"{result_id}.json"
        payload = _read_json(path)
        return PosteriorResultRecord(**payload)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)
