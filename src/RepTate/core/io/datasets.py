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
    """Container for dataset metadata and numerical data.

    This immutable dataclass pairs a dataset record (containing metadata like
    name, source location, and custom attributes) with the corresponding
    numerical data array.

    Attributes:
        record: Dataset metadata including ID, name, source location, and
            custom metadata dictionary.
        data: Numerical array containing the dataset values. Expected to be
            a 2D array where rows represent observations and columns represent
            variables.
    """
    record: DatasetRecord
    data: npt.NDArray[np.floating]


def load_dataset(path: str | Path) -> DatasetPayload:
    """Load a dataset from delimited text or JSON.

    Supported formats include CSV (comma-delimited), TSV (tab-delimited),
    and JSON. For delimited formats, the entire file is parsed as numerical
    data. For JSON format, the file must contain a dict with optional "name",
    "data", and "metadata" keys.

    Args:
        path: Path to the dataset file. Supported extensions are .csv, .tsv,
            and .json. The file extension determines the parsing strategy.

    Returns:
        DatasetPayload containing the parsed dataset record (with auto-generated
        UUID, name derived from file stem, and source location) and the numerical
        data array.

    Raises:
        ValueError: If the dataset contains non-finite values, is empty, or has
            an unsupported file extension.
    """
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
    """Export a dataset to delimited text or JSON.

    Writes the dataset to disk in the specified format. For delimited formats
    (CSV, TSV), only the numerical data is written. For JSON format, the output
    includes the dataset name, metadata dictionary, and data array.

    Args:
        payload: Dataset payload containing the record metadata and numerical
            data to export.
        path: Destination file path where the dataset will be written.
        fmt: Output format specifier. Supported values are "csv", "tsv", and
            "json" (case-insensitive).

    Raises:
        ValueError: If the format specifier is not supported.
    """
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
