"""Orchestration for deterministic fit and Bayesian inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from RepTate.core.bayes import build_likelihood, default_priors, run_nuts
from RepTate.core.bayes.results import PosteriorDiagnostics, PosteriorSummary, diagnostics_from_samples, summarize_posterior
from RepTate.core.fitting.nlsq_fit import FitResult, FitDiagnostics, run_nlsq_fit


@dataclass(frozen=True)
class FitAndSampleResult:
    """Combined result from deterministic fit and Bayesian inference.

    Encapsulates both the point estimates from NLSQ optimization and the
    posterior distribution from MCMC sampling, enabling uncertainty quantification
    while maintaining computational efficiency through warm-starting.

    Attributes:
        fit_result: Deterministic fit result containing parameter point estimates,
            covariance, and residuals from NLSQ optimization.
        fit_diagnostics: Fit diagnostics including convergence status and function
            evaluation count.
        posterior_summary: Summary statistics (mean, std, quantiles) from the
            posterior distribution.
        posterior_diagnostics: MCMC diagnostics including R-hat, effective sample
            size, and divergence counts.
        raw_samples: Dictionary mapping parameter names to their MCMC sample chains.
            Shape (n_samples,) for each parameter.
    """
    fit_result: FitResult
    fit_diagnostics: FitDiagnostics
    posterior_summary: PosteriorSummary
    posterior_diagnostics: PosteriorDiagnostics
    raw_samples: dict[str, jnp.ndarray]


def fit_and_sample(
    kernel: Callable[[jnp.ndarray, dict[str, float]], jnp.ndarray],
    xdata: jnp.ndarray,
    ydata: jnp.ndarray,
    *,
    priors: dict[str, object] | None = None,
    rng_key: jax.Array | None = None,
) -> FitAndSampleResult:
    """Run deterministic fit then warm-started NUTS sampling.

    Two-stage inference pipeline that first performs fast deterministic optimization
    using NLSQ, then uses the fit result to warm-start Bayesian MCMC sampling for
    full uncertainty quantification.

    Args:
        kernel: Model kernel function taking (xdata, parameters_dict) and returning
            model predictions. Signature: kernel(x, params) -> Array where params
            is a dict with string keys (e.g., {"p0": 1.0, "p1": 2.0}).
        xdata: Independent variable data. Shape (n_points,) or (n_points, n_features).
        ydata: Dependent variable data to fit. Shape (n_points,).
        priors: Dictionary specifying prior distributions for parameters. If None,
            uses default priors (typically weakly informative). Keys should match
            parameter names, values are NumPyro distribution objects.
        rng_key: JAX random key for MCMC sampling. If None, uses PRNGKey(0) for
            reproducibility.

    Returns:
        FitAndSampleResult: Combined result containing:
            - Deterministic fit results (point estimates, covariance)
            - Fit diagnostics (convergence status, nfev)
            - Posterior summary statistics (mean, std, quantiles)
            - MCMC diagnostics (R-hat, ESS, divergences)
            - Raw MCMC sample chains for all parameters
    """
    fit_result, fit_diagnostics = run_nlsq_fit(
        lambda x, params: kernel(x, {f"p{i}": float(v) for i, v in enumerate(params)}),
        xdata,
        ydata,
    )
    priors = priors or default_priors()
    model = build_likelihood(kernel, xdata, ydata, priors)
    rng = rng_key or jax.random.PRNGKey(0)
    mcmc = run_nuts(model, rng, init_params=fit_result.warm_start)
    samples = mcmc.get_samples()
    summary = summarize_posterior(samples)
    diagnostics = diagnostics_from_samples(samples)
    return FitAndSampleResult(
        fit_result=fit_result,
        fit_diagnostics=fit_diagnostics,
        posterior_summary=summary,
        posterior_diagnostics=diagnostics,
        raw_samples=samples,
    )
