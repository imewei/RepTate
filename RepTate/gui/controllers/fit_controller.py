"""Controller for deterministic fit workflow."""

from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp

from RepTate.core.compute.service_api import ComputeService


class FitController:
    """Thin controller that bridges UI events to computation."""

    def __init__(self, compute_service: ComputeService) -> None:
        self._service = compute_service

    def run_fit(
        self,
        model_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        xdata: jnp.ndarray,
        ydata: jnp.ndarray,
        *,
        p0: jnp.ndarray | None,
        bounds: tuple[float, float],
        dataset_id: str,
        model_id: str,
        result_id: str,
    ) -> dict[str, Any]:
        record = self._service.run_fit(
            model_fn,
            xdata,
            ydata,
            p0=p0,
            bounds=bounds,
            result_id=result_id,
            dataset_id=dataset_id,
            model_id=model_id,
        )
        return record.__dict__
