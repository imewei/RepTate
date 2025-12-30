"""Benchmark deterministic fit timing for SC-001/SC-002."""

from __future__ import annotations

import time

import jax.numpy as jnp

from RepTate.core.fitting.nlsq_fit import run_nlsq_fit


def _linear_model(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    return params[0] * x + params[1]


def main() -> None:
    xdata = jnp.linspace(0.0, 10.0, 1000)
    ydata = _linear_model(xdata, jnp.array([1.5, 0.5]))
    start = time.perf_counter()
    run_nlsq_fit(lambda x, p: _linear_model(x, p), xdata, ydata, p0=jnp.array([1.0, 0.0]))
    duration = time.perf_counter() - start
    print(f"Deterministic fit timing: {duration:.3f}s")


if __name__ == "__main__":
    main()
