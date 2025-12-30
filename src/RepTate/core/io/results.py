"""Fit and uncertainty results export helpers."""

from __future__ import annotations

from pathlib import Path
import json

from RepTate.core.models import FitResultRecord, PosteriorResultRecord


def format_results_summary(
    fit_result: FitResultRecord, uncertainty_result: PosteriorResultRecord | None = None
) -> str:
    """Format a human-readable summary of fit and uncertainty results."""
    lines = [
        f"Fit result: {fit_result.result_id}",
        f"Status: {fit_result.status}",
        f"Parameters: {fit_result.parameter_estimates}",
        f"Diagnostics: {fit_result.diagnostics}",
    ]
    if uncertainty_result is not None:
        lines.extend(
            [
                "",
                f"Uncertainty result: {uncertainty_result.result_id}",
                f"Status: {uncertainty_result.status}",
                f"Summary: {uncertainty_result.summary_statistics}",
            ]
        )
    return "\n".join(lines)


def export_results_json(
    fit_result: FitResultRecord,
    uncertainty_result: PosteriorResultRecord | None,
    destination: str | Path,
) -> None:
    """Export fit and uncertainty results to JSON."""
    payload = {
        "fit_result": fit_result.__dict__,
        "uncertainty_result": uncertainty_result.__dict__ if uncertainty_result else None,
    }
    Path(destination).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_results_summary(
    fit_result: FitResultRecord,
    uncertainty_result: PosteriorResultRecord | None,
    destination: str | Path,
) -> None:
    """Export a human-readable summary of results."""
    Path(destination).write_text(
        format_results_summary(fit_result, uncertainty_result), encoding="utf-8"
    )
