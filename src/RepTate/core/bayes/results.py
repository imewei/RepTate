"""Posterior results and diagnostics helpers."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from numpyro.diagnostics import summary


@dataclass(frozen=True)
class PosteriorSummary:
    """Summary statistics for posterior samples.

    Stores computed statistics for each parameter in the posterior distribution,
    including mean, standard deviation, and quantiles. This provides a compact
    representation of the posterior without storing all samples.

    Attributes:
        statistics: Nested dictionary where outer keys are parameter names and
            inner dictionaries contain statistic names mapped to float values.
            Typical statistics include 'mean', 'std', 'median', and quantiles
            like '5%', '95%' for credible intervals.
    """
    statistics: dict[str, dict[str, float]]


@dataclass(frozen=True)
class PosteriorDiagnostics:
    """Diagnostic metrics for MCMC sampling quality.

    Contains convergence and efficiency diagnostics for assessing the quality
    of posterior samples from MCMC. These metrics help determine whether the
    sampler has converged and whether sufficient independent samples were
    obtained.

    Attributes:
        r_hat: Dictionary mapping parameter names to Gelman-Rubin R-hat
            convergence diagnostic values. Values close to 1.0 (typically
            < 1.01) indicate convergence. Values > 1.1 suggest chains have
            not converged.
        ess: Dictionary mapping parameter names to effective sample size (ESS)
            values. Represents the number of independent samples. Higher is
            better; values > 400 per chain are generally adequate for inference.
    """
    r_hat: dict[str, float]
    ess: dict[str, float]


def summarize_posterior(samples: dict[str, jnp.ndarray]) -> PosteriorSummary:
    """Compute summary statistics for posterior samples.

    Processes raw MCMC samples to calculate summary statistics for each parameter,
    including central tendency, spread, and credible intervals. This provides a
    human-readable summary of the posterior distribution.

    Args:
        samples: Dictionary mapping parameter names to arrays of posterior samples.
            Each array has shape (num_samples, *param_shape) where num_samples
            is the number of MCMC samples collected. Typically obtained from
            mcmc.get_samples().

    Returns:
        PosteriorSummary object containing nested dictionaries of statistics.
        For each parameter, includes mean, standard deviation, quantiles, and
        other descriptive statistics computed by NumPyro's summary function.
    """
    stats = summary(samples)
    statistics = {key: {k: float(v) for k, v in val.items()} for key, val in stats.items()}
    return PosteriorSummary(statistics=statistics)


def diagnostics_from_samples(samples: dict[str, jnp.ndarray]) -> PosteriorDiagnostics:
    """Extract MCMC diagnostic metrics from posterior samples.

    Computes convergence and efficiency diagnostics to assess the quality of
    MCMC sampling. These diagnostics help determine whether the sampler has
    adequately explored the posterior distribution and whether additional
    samples are needed.

    Args:
        samples: Dictionary mapping parameter names to arrays of posterior samples.
            Each array has shape (num_samples, *param_shape). Typically obtained
            from mcmc.get_samples(). For R-hat calculation, multiple chains should
            be used.

    Returns:
        PosteriorDiagnostics object containing R-hat convergence statistics and
        effective sample sizes (ESS) for each parameter. Use these to verify
        sampling quality before interpreting results: R-hat < 1.01 and ESS > 400
        per chain are typical quality thresholds.
    """
    stats = summary(samples)
    r_hat = {key: float(val["r_hat"]) for key, val in stats.items()}
    ess = {key: float(val["n_eff"]) for key, val in stats.items()}
    return PosteriorDiagnostics(r_hat=r_hat, ess=ess)
