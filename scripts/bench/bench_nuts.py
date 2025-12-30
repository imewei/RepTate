"""Benchmark NUTS sampling performance."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from RepTate.core.bayes import build_likelihood, default_priors, run_nuts
from RepTate.core.jax_ops.models import linear_kernel


def main() -> None:
    x = jnp.linspace(0.0, 1.0, 100)
    y = linear_kernel(x, {"slope": 1.0, "intercept": 0.0})
    priors = default_priors()
    model = build_likelihood(linear_kernel, x, y, priors)
    rng = jax.random.PRNGKey(0)
    mcmc = run_nuts(model, rng, num_warmup=100, num_samples=200, init_params={"sigma": 0.1})
    print(mcmc.get_samples().keys())


if __name__ == "__main__":
    main()
