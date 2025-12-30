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
