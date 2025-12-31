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
    """Run NUTS sampling for the given model.

    Performs Markov Chain Monte Carlo (MCMC) sampling using the No-U-Turn
    Sampler (NUTS) algorithm to generate samples from the posterior distribution
    defined by the NumPyro model. NUTS is an adaptive Hamiltonian Monte Carlo
    method that efficiently explores the parameter space.

    Args:
        model: A NumPyro model function (callable with no arguments) that
            defines the probabilistic model including priors and likelihood.
            Typically created by build_likelihood.
        rng_key: JAX random number generator key for reproducible sampling.
            Create with jax.random.PRNGKey(seed).
        num_warmup: Number of warmup (burn-in) iterations to tune sampler
            parameters. These samples are discarded. Default is 1000.
        num_samples: Number of posterior samples to collect after warmup.
            Default is 1000.
        init_params: Optional dictionary of initial parameter values to start
            the sampling. If None, NumPyro uses default initialization strategy.
            Keys should match parameter names in the model.

    Returns:
        MCMC object containing the sampling results. Use mcmc.get_samples() to
        retrieve posterior samples as a dictionary of arrays, where each key
        corresponds to a parameter and values are samples of shape
        (num_samples, *param_shape).
    """
    kernel = NUTS(model, init_strategy=numpyro.infer.init_to_value(values=init_params))
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key)
    return mcmc
