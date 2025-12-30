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
