"""NUTS runner for Bayesian inference."""

from __future__ import annotations

from typing import Callable

import jax
import numpyro
from numpyro.infer import MCMC, NUTS


def run_nuts(
    model: Callable[[], None],
    rng_key: jax.Array,
    *,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    init_params: dict[str, float] | None = None,
) -> MCMC:
    """Run NUTS sampling for the given model."""
    kernel = NUTS(model, init_strategy=numpyro.infer.init_to_value(values=init_params))
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key)
    return mcmc
