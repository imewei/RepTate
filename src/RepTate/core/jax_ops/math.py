"""JAX-only numerical helpers for stable computations."""

from __future__ import annotations

import jax.numpy as jnp


def clamp(value: jnp.ndarray, lower: float, upper: float) -> jnp.ndarray:
    """Clamp values into the inclusive range [lower, upper]."""
    return jnp.clip(value, lower, upper)


def safe_log(value: jnp.ndarray, *, min_value: float = 1e-12) -> jnp.ndarray:
    """Compute log with a floor to avoid invalid values."""
    return jnp.log(jnp.clip(value, min_value, None))


def safe_divide(numerator: jnp.ndarray, denominator: jnp.ndarray) -> jnp.ndarray:
    """Divide with zeros mapped to inf using JAX semantics."""
    return numerator / denominator


def squared_error(observed: jnp.ndarray, predicted: jnp.ndarray) -> jnp.ndarray:
    """Compute squared residuals."""
    residuals = observed - predicted
    return residuals * residuals
