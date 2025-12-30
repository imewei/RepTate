"""Typed protocol for JAX model kernels."""

from __future__ import annotations

from typing import Protocol

import jax.numpy as jnp


class ModelKernel(Protocol):
    """Callable interface for JAX model kernels."""

    def __call__(self, x: jnp.ndarray, params: dict[str, float]) -> jnp.ndarray:
        ...
