"""Shared JAX kernels for deterministic model evaluation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from RepTate.core.jax_ops.model_api import ModelKernel


def evaluate_model(kernel: ModelKernel, x: jnp.ndarray, params: dict[str, float]) -> jnp.ndarray:
    """Evaluate a model kernel on input data.

    Args:
        kernel: Callable model kernel function conforming to ModelKernel protocol.
        x: Input data array (independent variable values).
        params: Dictionary of model parameters (e.g., {"slope": 1.0, "intercept": 0.0}).

    Returns:
        jnp.ndarray: Model predictions evaluated at input points x using
            the provided parameters.
    """
    return kernel(x, params)


@jax.jit
def linear_kernel(x: jnp.ndarray, params: dict[str, float]) -> jnp.ndarray:
    """Simple linear kernel for testing and scaffolding.

    JIT-compiled for optimal performance on repeated evaluations (FR-014).

    Args:
        x: Input data array (independent variable values).
        params: Dictionary containing "slope" and "intercept" keys.
            Defaults to slope=1.0 and intercept=0.0 if keys are missing.

    Returns:
        jnp.ndarray: Linear function values computed as slope * x + intercept.
    """
    slope = params.get("slope", 1.0)
    intercept = params.get("intercept", 0.0)
    return slope * x + intercept
