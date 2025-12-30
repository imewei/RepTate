"""Visualization state models for UI rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VisualizationState:
    dataset_id: str
    fit_result_id: str | None
    posterior_result_id: str | None
    view_configuration: dict[str, Any] = field(default_factory=dict)
    export_configuration: dict[str, Any] = field(default_factory=dict)
