"""Dataset import mapping for RepTate datasets."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from RepTate.core.data.dataset_io import DatasetPayload, load_csv_dataset
from RepTate.core.models.results import DatasetRecord


def import_dataset(path: str | Path, *, dataset_id: str) -> tuple[DatasetRecord, DatasetPayload]:
    """Import a dataset file and create both record and payload objects.

    Loads a CSV dataset from disk and creates a DatasetRecord for tracking
    metadata and a DatasetPayload containing the actual numeric data. The
    dataset name is derived from the filename stem.

    Args:
        path: Path to the dataset file to import. Can be a string or Path object.
        dataset_id: Unique identifier to assign to this dataset record.

    Returns:
        A tuple containing:
            - DatasetRecord: Metadata record with dataset_id, name, source
              location, and metadata dictionary.
            - DatasetPayload: JAX-backed payload with columns, data array, and
              metadata.

    Raises:
        FileNotFoundError: If the file does not exist at the specified path.
        ValueError: If the dataset is empty or contains invalid data.
    """
    payload = load_csv_dataset(path)
    record = DatasetRecord(
        dataset_id=dataset_id,
        name=Path(path).stem,
        source_location=payload.source_location,
        metadata=payload.metadata,
    )
    return record, payload


def dataset_summary(record: DatasetRecord, payload: DatasetPayload) -> dict[str, Any]:
    """Generate a summary dictionary combining record metadata and payload info.

    Converts a DatasetRecord to a dictionary and enriches it with column names
    and row count from the corresponding DatasetPayload. Useful for inspection,
    logging, and serialization.

    Args:
        record: DatasetRecord containing metadata (dataset_id, name, source, etc.).
        payload: DatasetPayload containing the actual data array and column names.

    Returns:
        Dictionary with all record fields plus 'columns' (list of column names)
        and 'row_count' (integer number of data rows).
    """
    summary = asdict(record)
    summary["columns"] = payload.columns
    summary["row_count"] = int(payload.data.shape[0])
    return summary
