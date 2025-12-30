"""Result record models for deterministic fits and Bayesian inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class DatasetRecord:
    dataset_id: str
    name: str
    source_location: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelRecord:
    model_id: str
    name: str
    parameters: dict[str, Any]


@dataclass
class FitResultRecord:
    result_id: str
    dataset_id: str
    model_id: str
    parameter_estimates: dict[str, float]
    diagnostics: dict[str, Any]
    residuals: list[float]
    execution_context: dict[str, Any]
    status: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class PosteriorResultRecord:
    result_id: str
    fit_result_id: str
    sample_traces: dict[str, list[float]]
    summary_statistics: dict[str, Any]
    chain_metadata: dict[str, Any]
    resume_state: dict[str, Any]
    status: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
