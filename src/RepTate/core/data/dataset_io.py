"""Dataset input/output helpers for JAX-native workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Iterable

import jax.numpy as jnp


@dataclass(frozen=True)
class DatasetPayload:
    """Container for raw dataset arrays and basic metadata."""

    source_location: str
    columns: list[str]
    data: jnp.ndarray
    metadata: dict[str, str]


def load_csv_dataset(path: str | Path, *, delimiter: str | None = None) -> DatasetPayload:
    """Load a CSV-like dataset and return a JAX-backed payload."""
    source = Path(path)
    rows = _read_csv_rows(source, delimiter=delimiter)
    if not rows:
        raise ValueError(f"Dataset is empty: {source}")

    header, data_rows = _split_header(rows)
    data = jnp.asarray(data_rows, dtype=float)
    _validate_numeric_array(data, source)

    if not header:
        header = [f"col_{idx}" for idx in range(data.shape[1])]

    return DatasetPayload(
        source_location=str(source),
        columns=header,
        data=data,
        metadata={},
    )


def _read_csv_rows(source: Path, *, delimiter: str | None = None) -> list[list[str]]:
    if not source.exists():
        raise FileNotFoundError(f"Dataset not found: {source}")
    with source.open(newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter or ",")
        return [row for row in reader if row]


def _split_header(rows: list[list[str]]) -> tuple[list[str], list[list[float]]]:
    header: list[str] = []
    data_start = 0
    if rows and not _row_is_numeric(rows[0]):
        header = [cell.strip() for cell in rows[0]]
        data_start = 1

    data_rows: list[list[float]] = []
    for row in rows[data_start:]:
        data_rows.append([float(cell) for cell in row])
    return header, data_rows


def _row_is_numeric(row: Iterable[str]) -> bool:
    for cell in row:
        try:
            float(cell)
        except ValueError:
            return False
    return True


def _validate_numeric_array(data: jnp.ndarray, source: Path) -> None:
    if data.size == 0:
        raise ValueError(f"Dataset has no numeric rows: {source}")
    if not bool(jnp.all(jnp.isfinite(data))):
        raise ValueError(f"Dataset contains non-finite values: {source}")
