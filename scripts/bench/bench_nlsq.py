"""Benchmark deterministic fitting performance."""

from __future__ import annotations

import time

import jax.numpy as jnp

from RepTate.core.fitting.nlsq_fit import run_nlsq_fit
from RepTate.core.jax_ops.models import linear_kernel


def main() -> None:
    x = jnp.linspace(0.0, 10.0, 1000)
    y = linear_kernel(x, {"slope": 2.0, "intercept": 1.0})
    start = time.time()
    run_nlsq_fit(lambda xv, params: linear_kernel(xv, {"slope": params[0], "intercept": params[1]}), x, y)
    elapsed = time.time() - start
    print(f"NLSQ fit elapsed: {elapsed:.3f}s")


if __name__ == "__main__":
    main()
