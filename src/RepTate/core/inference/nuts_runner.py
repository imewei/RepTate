"""NumPyro NUTS runner for Bayesian inference."""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import MCMC, NUTS

from RepTate.core.inference.warm_start import prepare_warm_start
from RepTate.core.models.results import FitResultRecord, PosteriorResultRecord


def run_nuts_inference(
    model: Callable[..., Any],
    *,
    fit_record: FitResultRecord,
    result_id: str,
    rng_seed: int,
    num_warmup: int,
    num_samples: int,
    model_kwargs: dict[str, Any] | None = None,
) -> PosteriorResultRecord:
    """Run NUTS (No-U-Turn Sampler) Bayesian inference on a probabilistic model.

    Performs Markov Chain Monte Carlo sampling using the NUTS algorithm from NumPyro.
    The function initializes the sampler with warm-start parameters from a previous
    optimization fit, runs the MCMC chains, and returns posterior samples with summary
    statistics.

    Args:
        model: A NumPyro probabilistic model function that defines the generative process
            and priors. Should accept model_kwargs as keyword arguments.
        fit_record: Previous optimization result containing parameter estimates used for
            warm-starting the MCMC sampler.
        result_id: Unique identifier for this inference run, used to track and store results.
        rng_seed: Random number generator seed for reproducibility of sampling.
        num_warmup: Number of warmup (adaptation) steps for the sampler to tune step size
            and mass matrix.
        num_samples: Number of posterior samples to draw after warmup.
        model_kwargs: Optional keyword arguments to pass to the model function, such as
            observed data or configuration parameters.

    Returns:
        PosteriorResultRecord containing:
            - sample_traces: Flattened posterior samples for each parameter
            - summary_statistics: Mean and standard deviation for each parameter
            - chain_metadata: Number of warmup steps, samples, and chains
            - resume_state: Warm-start parameters for potential resumption
            - status: Completion status of the inference run
    """
    warm_start = prepare_warm_start(fit_record)
    rng_key = jax.random.PRNGKey(rng_seed)
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key, **(model_kwargs or {}))
    samples = mcmc.get_samples(group_by_chain=True)
    summary = summarize_samples(samples)
    return PosteriorResultRecord(
        result_id=result_id,
        fit_result_id=fit_record.result_id,
        sample_traces=_to_python(samples),
        summary_statistics=summary,
        chain_metadata={
            "num_warmup": num_warmup,
            "num_samples": num_samples,
            "num_chains": samples[next(iter(samples))].shape[0],
        },
        resume_state={"warm_start": warm_start},
        status="completed",
    )


def summarize_samples(samples: dict[str, jnp.ndarray]) -> dict[str, dict[str, float]]:
    """Compute summary statistics for posterior samples.

    Calculates mean and standard deviation for each parameter across all chains
    and samples. All samples are flattened before computing statistics.

    Args:
        samples: Dictionary mapping parameter names to arrays of posterior samples.
            Arrays can have shape (num_chains, num_samples, ...) and will be flattened.

    Returns:
        Dictionary mapping parameter names to their summary statistics. Each parameter
        has a nested dictionary with keys:
            - "mean": Mean value across all samples
            - "std": Standard deviation across all samples
    """
    summaries: dict[str, dict[str, float]] = {}
    for name, values in samples.items():
        flat = jnp.ravel(values)
        summaries[name] = {
            "mean": float(jnp.mean(flat)),
            "std": float(jnp.std(flat)),
        }
    return summaries


def _to_python(samples: dict[str, jnp.ndarray]) -> dict[str, list[float]]:
    return {name: jnp.ravel(values).tolist() for name, values in samples.items()}
