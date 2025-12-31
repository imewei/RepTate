"""JAX array helpers for device placement and dtype policy."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

JaxArray = jax.Array


def default_float_dtype() -> jnp.dtype:
    """Return the default floating-point dtype for numerical kernels.

    Returns:
        jnp.dtype: The default floating-point data type (float64) used for
            high-precision numerical computations in RepTate.
    """
    return jnp.float64


def as_jax_array(data: Any, *, dtype: jnp.dtype | None = None) -> JaxArray:
    """Convert input data into a JAX array with the configured dtype.

    Args:
        data: Input data to convert (numpy array, list, scalar, etc.).
        dtype: Target data type. If None, uses default_float_dtype().

    Returns:
        JaxArray: JAX array representation of the input data with the
            specified dtype.
    """
    target_dtype = dtype or default_float_dtype()
    return jnp.asarray(data, dtype=target_dtype)


def ensure_device(data: JaxArray, *, device: jax.Device | None = None) -> JaxArray:
    """Ensure data is placed on the requested device.

    Args:
        data: JAX array to be placed on device.
        device: Target JAX device (CPU, GPU, TPU). If None, returns data
            unchanged on its current device.

    Returns:
        JaxArray: The input array transferred to the specified device, or
            unchanged if device is None.
    """
    if device is None:
        return data
    return jax.device_put(data, device=device)


def zeros_like(data: JaxArray) -> JaxArray:
    """Create a zero-filled array matching the input shape and dtype.

    Args:
        data: Reference JAX array whose shape and dtype will be matched.

    Returns:
        JaxArray: A new array with the same shape and dtype as data,
            filled with zeros.
    """
    return jnp.zeros_like(data)
