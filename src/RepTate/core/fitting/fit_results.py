"""Assemble deterministic fit results into record structures."""

from __future__ import annotations

from typing import Any

from RepTate.core.fitting.nlsq_fit import FitDiagnostics, FitResult
from RepTate.core.models.results import FitResultRecord


def build_fit_record(
    *,
    fit_result: FitResult,
    diagnostics: FitDiagnostics,
    dataset_id: str,
    model_id: str,
    result_id: str,
    execution_context: dict[str, Any],
) -> FitResultRecord:
    """Assemble a deterministic fit result into a persistent record structure.

    Combines the fit result parameters, diagnostics, and metadata into a
    standardized FitResultRecord that can be stored, serialized, or transmitted.

    Args:
        fit_result: The fit result containing parameter estimates, covariance,
            and residuals from the NLSQ optimization.
        diagnostics: Fit diagnostics including number of function evaluations
            and convergence status.
        dataset_id: Unique identifier for the dataset used in the fit.
        model_id: Unique identifier for the theoretical model being fitted.
        result_id: Unique identifier for this particular fit result.
        execution_context: Dictionary containing execution metadata such as
            timestamps, environment info, or configuration settings.

    Returns:
        FitResultRecord: A frozen record containing all fit information with
            status marked as "completed".
    """
    return FitResultRecord(
        result_id=result_id,
        dataset_id=dataset_id,
        model_id=model_id,
        parameter_estimates=fit_result.parameters,
        diagnostics=diagnostics.as_dict(),
        residuals=fit_result.residuals,
        execution_context=execution_context,
        status="completed",
    )
