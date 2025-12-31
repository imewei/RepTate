"""Warm-start preparation for Bayesian inference."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from RepTate.core.models.results import FitResultRecord


def prepare_warm_start(fit_record: FitResultRecord) -> dict[str, Any]:
    """Prepare warm-start parameters for Bayesian inference from optimization results.

    Converts parameter estimates from a previous maximum likelihood or least-squares
    fit into JAX arrays suitable for initializing MCMC samplers. This warm-starting
    approach improves sampler convergence by starting from a high-probability region
    of the posterior.

    Args:
        fit_record: Previous optimization result containing parameter estimates and
            metadata from a fitting procedure.

    Returns:
        Dictionary containing:
            - "initial_params": Parameter estimates converted to JAX arrays, mapping
              parameter names to their optimized values
            - "fit_result_id": Identifier of the source fit for provenance tracking
    """
    params = fit_record.parameter_estimates
    return {
        "initial_params": {k: jnp.asarray(v) for k, v in params.items()},
        "fit_result_id": fit_record.result_id,
    }
