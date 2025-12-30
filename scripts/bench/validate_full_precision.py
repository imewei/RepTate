"""Validate full-precision numerical routines."""

from __future__ import annotations

import jax.numpy as jnp

from RepTate.core.jax_ops.models import linear_kernel


def main() -> None:
    x = jnp.linspace(0.0, 1.0, 10, dtype=jnp.float64)
    y = linear_kernel(x, {"slope": 1.0, "intercept": 0.0})
    assert y.dtype == jnp.float64, "Expected float64 outputs for full precision."
    print("Full precision check passed.")


if __name__ == "__main__":
    main()
