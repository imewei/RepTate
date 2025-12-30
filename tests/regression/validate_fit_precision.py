"""Regression check for deterministic fit precision."""

from __future__ import annotations

import jax.numpy as jnp

from RepTate.core.fitting.nlsq_fit import run_nlsq_fit


def _linear_model(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    return params[0] * x + params[1]


def main() -> None:
    xdata = jnp.linspace(0.0, 10.0, 50)
    true_params = jnp.array([2.0, -1.0])
    ydata = _linear_model(xdata, true_params)
    result, _ = run_nlsq_fit(
        lambda x, p: _linear_model(x, p),
        xdata,
        ydata,
        p0=jnp.array([1.0, 0.0]),
    )
    assert jnp.allclose(result.parameters_array, true_params, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    main()
