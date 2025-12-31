"""Fit and uncertainty results export helpers."""

from __future__ import annotations

from pathlib import Path
import json

from RepTate.core.models import FitResultRecord, PosteriorResultRecord


def format_results_summary(
    fit_result: FitResultRecord, uncertainty_result: PosteriorResultRecord | None = None
) -> str:
    """Format a human-readable summary of fit and uncertainty results.

    Creates a multi-line text summary containing fit result ID, status,
    parameter estimates, and diagnostics. If uncertainty quantification
    results are provided, appends posterior result ID, status, and summary
    statistics.

    Args:
        fit_result: Record containing deterministic fit results including
            parameter estimates, residuals, and diagnostic information.
        uncertainty_result: Optional record containing Bayesian posterior
            samples, summary statistics, and MCMC chain metadata. If None,
            only the fit result is included in the summary.

    Returns:
        Multi-line string summary formatted for human readability. Lines are
        separated by newline characters.
    """
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
    """Export fit and uncertainty results to JSON.

    Serializes both fit and uncertainty results to a JSON file with 2-space
    indentation. The output contains a "fit_result" key with all fit record
    fields and an "uncertainty_result" key (null if not provided).

    Args:
        fit_result: Record containing deterministic fit results including
            parameter estimates, residuals, and diagnostic information.
        uncertainty_result: Optional record containing Bayesian posterior
            samples, summary statistics, and MCMC chain metadata. If None,
            the "uncertainty_result" key in the output will be null.
        destination: Path where the JSON file will be written. Parent directory
            must exist.
    """
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
    """Export a human-readable summary of results.

    Writes a text file containing a formatted summary of fit and uncertainty
    results. The summary includes result IDs, status, parameter estimates,
    diagnostics, and (if available) posterior summary statistics.

    Args:
        fit_result: Record containing deterministic fit results including
            parameter estimates, residuals, and diagnostic information.
        uncertainty_result: Optional record containing Bayesian posterior
            samples, summary statistics, and MCMC chain metadata. If None,
            only the fit result is included in the summary.
        destination: Path where the text summary file will be written. Parent
            directory must exist.
    """
    Path(destination).write_text(
        format_results_summary(fit_result, uncertainty_result), encoding="utf-8"
    )
