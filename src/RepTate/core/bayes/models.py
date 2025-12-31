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
    """Build a NumPyro model with shared JAX kernels.

    Constructs a NumPyro probabilistic model by combining prior distributions,
    a JAX kernel function for model evaluation, and likelihood specification.
    The model samples parameters from priors, evaluates predictions using the
    kernel, and defines a Normal likelihood for observed data.

    Args:
        kernel: A JAX-compatible function that evaluates the model predictions.
            Takes input data (jnp.ndarray) and parameter dictionary as arguments,
            returns predicted values as jnp.ndarray.
        xdata: Input data array for model evaluation. Shape depends on the
            specific kernel function requirements.
        ydata: Observed output data array that the model should fit. Must be
            compatible in shape with kernel predictions.
        priors: Dictionary mapping parameter names to NumPyro distribution
            objects. Each distribution defines the prior belief for that parameter.
            Should include 'sigma' for observation noise if not handled elsewhere.

    Returns:
        A NumPyro model function (callable with no arguments) that can be used
        with NUTS or other MCMC samplers. The model samples parameters from
        priors and conditions on observed data through a Normal likelihood.
    """

    def model() -> None:
        """NumPyro model definition for Bayesian inference.

        Samples model parameters from their prior distributions, evaluates
        the kernel function to compute predictions, and defines the likelihood
        of observations assuming Normal-distributed errors with standard
        deviation sigma.

        The model performs the following steps:
        1. Sample each parameter from its prior distribution
        2. Extract sigma (observation noise) from parameters or use default
        3. Evaluate the kernel to get predicted mean values
        4. Define Normal likelihood for observed data conditioned on predictions
        """
        params = {}
        for name, prior in priors.items():
            params[name] = numpyro.sample(name, prior)
        sigma = params.get("sigma", 1.0)
        mu = evaluate_model(kernel, xdata, params)
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=ydata)

    return model
