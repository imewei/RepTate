"""JAX array helpers for device placement and dtype policy."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

JaxArray = jax.Array


def default_float_dtype() -> jnp.dtype:
    """Return the default floating-point dtype for numerical kernels."""
    return jnp.float64


def as_jax_array(data: Any, *, dtype: jnp.dtype | None = None) -> JaxArray:
    """Convert input data into a JAX array with the configured dtype."""
    target_dtype = dtype or default_float_dtype()
    return jnp.asarray(data, dtype=target_dtype)


def ensure_device(data: JaxArray, *, device: jax.Device | None = None) -> JaxArray:
    """Ensure data is placed on the requested device."""
    if device is None:
        return data
    return jax.device_put(data, device=device)


def zeros_like(data: JaxArray) -> JaxArray:
    """Create a zero-filled array matching the input shape and dtype."""
    return jnp.zeros_like(data)
