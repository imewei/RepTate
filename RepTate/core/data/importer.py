"""Dataset import mapping for RepTate datasets."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from RepTate.core.data.dataset_io import DatasetPayload, load_csv_dataset
from RepTate.core.models.results import DatasetRecord


def import_dataset(path: str | Path, *, dataset_id: str) -> tuple[DatasetRecord, DatasetPayload]:
    payload = load_csv_dataset(path)
    record = DatasetRecord(
        dataset_id=dataset_id,
        name=Path(path).stem,
        source_location=payload.source_location,
        metadata=payload.metadata,
    )
    return record, payload


def dataset_summary(record: DatasetRecord, payload: DatasetPayload) -> dict[str, Any]:
    summary = asdict(record)
    summary["columns"] = payload.columns
    summary["row_count"] = int(payload.data.shape[0])
    return summary
