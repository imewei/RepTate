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
        """Execute deterministic nonlinear least squares fit and return results.

        Bridges UI-triggered fit requests to the compute service, which performs
        JAX-accelerated curve fitting using NLSQ optimization. The method returns
        a dictionary representation of the fit result record for UI consumption.

        Args:
            model_fn: Theory function that takes x data and parameters, returns
                predicted y values. Signature: f(x: ndarray, params: ndarray) -> ndarray.
            xdata: Independent variable data (e.g., frequency, time) as 1D JAX array.
            ydata: Dependent variable data (e.g., modulus, relaxation) as 1D JAX array.
            p0: Initial parameter guess as 1D JAX array. If None, service uses
                default initialization strategy.
            bounds: Tuple of (lower_bound, upper_bound) applied to all parameters.
                Use (-inf, inf) for unconstrained optimization.
            dataset_id: Unique identifier for the source dataset being fit.
            model_id: Unique identifier for the theory/model being applied.
            result_id: Unique identifier for this particular fit result, used
                for exporting and result tracking.

        Returns:
            Dictionary containing fit results with keys:
                - result_id: str, unique identifier for this fit
                - dataset_id: str, source dataset identifier
                - model_id: str, theory/model identifier
                - parameters: optimized parameter values
                - residuals: fit residuals (ydata - model_prediction)
                - success: bool, whether optimization converged
                - Additional optimization metadata from NLSQ
        """
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
