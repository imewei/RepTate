"""NumPyro model definitions for RepTate."""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from RepTate.core.jax_ops.models import evaluate_model


def build_likelihood(
    kernel: Callable[[jnp.ndarray, dict[str, float]], jnp.ndarray],
    xdata: jnp.ndarray,
    ydata: jnp.ndarray,
    priors: dict[str, dist.Distribution],
) -> Callable[[], None]:
    """Build a NumPyro model with shared JAX kernels."""

    def model() -> None:
        params = {}
        for name, prior in priors.items():
            params[name] = numpyro.sample(name, prior)
        sigma = params.get("sigma", 1.0)
        mu = evaluate_model(kernel, xdata, params)
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=ydata)

    return model
