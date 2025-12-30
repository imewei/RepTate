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
    """Run deterministic fit then warm-started NUTS sampling."""
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
