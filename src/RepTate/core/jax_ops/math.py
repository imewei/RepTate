"""JAX-only numerical helpers for stable computations."""

from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.jit
def clamp(value: jnp.ndarray, lower: float, upper: float) -> jnp.ndarray:
    """Clamp values into the inclusive range [lower, upper].

    Args:
        value: Input array to clamp.
        lower: Minimum value (inclusive).
        upper: Maximum value (inclusive).

    Returns:
        jnp.ndarray: Array with all values constrained to [lower, upper].
    """
    return jnp.clip(value, lower, upper)


@jax.jit
def safe_log(value: jnp.ndarray, *, min_value: float = 1e-12) -> jnp.ndarray:
    """Compute log with a floor to avoid invalid values.

    Args:
        value: Input array for logarithm computation.
        min_value: Minimum value floor to prevent log(0) or log(negative).
            Defaults to 1e-12.

    Returns:
        jnp.ndarray: Natural logarithm of the clamped input, preventing
            NaN or -inf results from non-positive values.
    """
    return jnp.log(jnp.clip(value, min_value, None))


@jax.jit
def safe_divide(numerator: jnp.ndarray, denominator: jnp.ndarray) -> jnp.ndarray:
    """Divide with zeros mapped to inf using JAX semantics.

    Args:
        numerator: Dividend array.
        denominator: Divisor array.

    Returns:
        jnp.ndarray: Element-wise division result. Division by zero yields
            inf with the appropriate sign following JAX/IEEE 754 semantics.
    """
    return numerator / denominator


@jax.jit
def squared_error(observed: jnp.ndarray, predicted: jnp.ndarray) -> jnp.ndarray:
    """Compute squared residuals.

    Args:
        observed: Observed or experimental data values.
        predicted: Model-predicted values.

    Returns:
        jnp.ndarray: Element-wise squared differences (observed - predicted)^2,
            commonly used for least-squares fitting and loss computation.
    """
    residuals = observed - predicted
    return residuals * residuals
