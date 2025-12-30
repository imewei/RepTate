"""Controller for Bayesian inference workflow."""

from __future__ import annotations

from typing import Any, Callable

from RepTate.core.compute.service_api import ComputeService
from RepTate.core.models.results import FitResultRecord


class InferenceController:
    """Bridges UI triggers to the NumPyro NUTS pipeline."""

    def __init__(self, compute_service: ComputeService) -> None:
        self._service = compute_service

    def run_inference(
        self,
        model: Callable[..., Any],
        fit_record: FitResultRecord,
        *,
        result_id: str,
        rng_seed: int = 0,
        num_warmup: int = 1000,
        num_samples: int = 1000,
    ) -> dict[str, Any]:
        record = self._service.run_inference(
            model,
            fit_record=fit_record,
            result_id=result_id,
            rng_seed=rng_seed,
            num_warmup=num_warmup,
            num_samples=num_samples,
        )
        return record.__dict__
