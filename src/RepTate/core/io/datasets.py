"""Dataset import and export helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import uuid

import numpy as np
import numpy.typing as npt

from RepTate.core.models import DatasetRecord


@dataclass(frozen=True)
class DatasetPayload:
    record: DatasetRecord
    data: npt.NDArray[np.floating]


def load_dataset(path: str | Path) -> DatasetPayload:
    """Load a dataset from delimited text or JSON."""
    source = Path(path)
    suffix = source.suffix.lower()
    dataset_id = str(uuid.uuid4())
    name = source.stem

    if suffix in {".csv", ".tsv"}:
        delimiter = "," if suffix == ".csv" else "\t"
        data = np.loadtxt(source, delimiter=delimiter)
        if data.size == 0 or not np.isfinite(data).all():
            raise ValueError("Dataset contains non-finite or empty values.")
        record = DatasetRecord(dataset_id=dataset_id, name=name, source_location=str(source))
        return DatasetPayload(record=record, data=data)

    if suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        data = np.asarray(payload.get("data", []), dtype=float)
        if data.size == 0 or not np.isfinite(data).all():
            raise ValueError("Dataset contains non-finite or empty values.")
        metadata = payload.get("metadata", {})
        record = DatasetRecord(
            dataset_id=dataset_id,
            name=payload.get("name", name),
            source_location=str(source),
            metadata=metadata,
        )
        return DatasetPayload(record=record, data=data)

    raise ValueError(f"Unsupported dataset format: {suffix}")


def export_dataset(payload: DatasetPayload, path: str | Path, *, fmt: str) -> None:
    """Export a dataset to delimited text or JSON."""
    destination = Path(path)
    fmt_lower = fmt.lower()

    if fmt_lower in {"csv", "tsv"}:
        delimiter = "," if fmt_lower == "csv" else "\t"
        np.savetxt(destination, payload.data, delimiter=delimiter)
        return

    if fmt_lower == "json":
        destination.write_text(
            json.dumps(
                {
                    "name": payload.record.name,
                    "metadata": payload.record.metadata,
                    "data": payload.data.tolist(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return

    raise ValueError(f"Unsupported export format: {fmt}")
