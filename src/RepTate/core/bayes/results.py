"""Posterior results and diagnostics helpers."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from numpyro.diagnostics import summary


@dataclass(frozen=True)
class PosteriorSummary:
    statistics: dict[str, dict[str, float]]


@dataclass(frozen=True)
class PosteriorDiagnostics:
    r_hat: dict[str, float]
    ess: dict[str, float]


def summarize_posterior(samples: dict[str, jnp.ndarray]) -> PosteriorSummary:
    stats = summary(samples)
    statistics = {key: {k: float(v) for k, v in val.items()} for key, val in stats.items()}
    return PosteriorSummary(statistics=statistics)


def diagnostics_from_samples(samples: dict[str, jnp.ndarray]) -> PosteriorDiagnostics:
    stats = summary(samples)
    r_hat = {key: float(val["r_hat"]) for key, val in stats.items()}
    ess = {key: float(val["n_eff"]) for key, val in stats.items()}
    return PosteriorDiagnostics(r_hat=r_hat, ess=ess)
