"""Visualization state models for UI rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VisualizationState:
    """Mutable state for UI visualization and rendering configuration.

    Tracks which dataset and analysis results are currently being visualized,
    along with the user's view preferences and export settings. This state
    object is used to synchronize the UI display with the underlying data
    and analysis results.

    Attributes:
        dataset_id: Identifier of the dataset being visualized.
        fit_result_id: Optional identifier of the deterministic fit result
            being displayed. None if no fit is active.
        posterior_result_id: Optional identifier of the Bayesian posterior
            result being displayed. None if no posterior is active.
        view_configuration: Dictionary of view settings (e.g., plot type,
            axis scales, zoom level, color scheme, overlays).
        export_configuration: Dictionary of export settings (e.g., file format,
            resolution, DPI, include metadata).
    """
    dataset_id: str
    fit_result_id: str | None
    posterior_result_id: str | None
    view_configuration: dict[str, Any] = field(default_factory=dict)
    export_configuration: dict[str, Any] = field(default_factory=dict)
