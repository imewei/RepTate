"""Warm-start preparation for Bayesian inference."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from RepTate.core.models.results import FitResultRecord


def prepare_warm_start(fit_record: FitResultRecord) -> dict[str, Any]:
    params = fit_record.parameter_estimates
    return {
        "initial_params": {k: jnp.asarray(v) for k, v in params.items()},
        "fit_result_id": fit_record.result_id,
    }
