"""Shared JAX kernels for deterministic model evaluation."""

from __future__ import annotations

import jax.numpy as jnp

from RepTate.core.jax_ops.model_api import ModelKernel


def evaluate_model(kernel: ModelKernel, x: jnp.ndarray, params: dict[str, float]) -> jnp.ndarray:
    """Evaluate a model kernel on input data."""
    return kernel(x, params)


def linear_kernel(x: jnp.ndarray, params: dict[str, float]) -> jnp.ndarray:
    """Simple linear kernel for testing and scaffolding."""
    slope = params.get("slope", 1.0)
    intercept = params.get("intercept", 0.0)
    return slope * x + intercept
