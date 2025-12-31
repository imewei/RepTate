"""Controller for Bayesian inference workflow."""

from __future__ import annotations

from typing import Any, Callable

from RepTate.core.compute.service_api import ComputeService
from RepTate.core.models.results import FitResultRecord


class InferenceController:
    """Bridges UI triggers to the NumPyro NUTS pipeline."""

    def __init__(self, compute_service: ComputeService) -> None:
        self._service = compute_service

    def run_inference(
        self,
        model: Callable[..., Any],
        fit_record: FitResultRecord,
        *,
        result_id: str,
        rng_seed: int = 0,
        num_warmup: int = 1000,
        num_samples: int = 1000,
    ) -> dict[str, Any]:
        """Execute Bayesian inference using NumPyro NUTS sampler and return posterior.

        Bridges UI-triggered inference requests to the compute service, which runs
        MCMC sampling to obtain posterior distributions over model parameters. Uses
        the fit result as initialization and data context for the Bayesian model.

        Args:
            model: NumPyro probabilistic model function with signature
                model(x, y, fit_params, ...). Must define priors using numpyro.sample
                and likelihood using numpyro.sample with obs= parameter.
            fit_record: Previous deterministic fit result containing optimized
                parameters (used as initialization), data references, and model ID.
            result_id: Unique identifier for this inference result, used for
                tracking and exporting posterior samples.
            rng_seed: Random number generator seed for reproducible MCMC sampling.
                Default is 0.
            num_warmup: Number of MCMC warmup/burn-in iterations to discard.
                Default is 1000. Increase for complex posteriors.
            num_samples: Number of posterior samples to draw after warmup.
                Default is 1000. Increase for better posterior approximation.

        Returns:
            Dictionary containing inference results with keys:
                - result_id: str, unique identifier for this inference run
                - samples: posterior parameter samples from MCMC chains
                - diagnostics: MCMC diagnostics (e.g., r_hat, n_eff, divergences)
                - fit_record: reference to the input fit result
                - Additional metadata from NumPyro NUTS sampler
        """
        record = self._service.run_inference(
            model,
            fit_record=fit_record,
            result_id=result_id,
            rng_seed=rng_seed,
            num_warmup=num_warmup,
            num_samples=num_samples,
        )
        return record.__dict__
