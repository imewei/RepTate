"""Shared typed data structures for RepTate workflows."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ParameterBounds:
    name: str
    lower: float | None
    upper: float | None


@dataclass(frozen=True)
class FitProblem:
    dataset_id: str
    model_id: str
    parameters: dict[str, float]
    bounds: dict[str, ParameterBounds] = field(default_factory=dict)
    options: dict[str, float | int | str | bool] = field(default_factory=dict)


@dataclass(frozen=True)
class FitDiagnostics:
    status: str
    metrics: dict[str, float]
    messages: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class UncertaintySummary:
    parameter_summaries: dict[str, dict[str, float]]
    diagnostics: dict[str, float]
    messages: list[str] = field(default_factory=list)
