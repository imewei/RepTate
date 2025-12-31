"""Computation service boundary between UI and numerical kernels."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import jax.numpy as jnp

from RepTate.core.compute.hardware import warn_if_no_accelerator
from RepTate.core.data.dataset_io import DatasetPayload, load_csv_dataset
from RepTate.core.data.result_store import ResultStore
from RepTate.core.fitting.nlsq_fit import FitDiagnostics, run_nlsq_fit
from RepTate.core.inference.nuts_runner import run_nuts_inference
from RepTate.core.models.model_registry import ModelRegistry
from RepTate.core.models.results import FitResultRecord, PosteriorResultRecord


class ComputeService:
    """Defines the core computation interface used by UI controllers."""

    def __init__(self, result_dir: str | Path) -> None:
        self._registry = ModelRegistry.from_theories()
        self._store = ResultStore(result_dir)

    def list_models(self) -> list[dict[str, Any]]:
        """List all available theoretical models registered in the system.

        Returns:
            list[dict[str, Any]]: List of model specification dictionaries containing
                model metadata (e.g., model_id, name, parameters, description) from the
                internal model registry. Each dictionary represents a serialized
                ModelSpec instance.
        """
        return [asdict(spec) for spec in self._registry.list_models()]

    def import_dataset(self, path: str | Path) -> DatasetPayload:
        """Load experimental data from a CSV file.

        Args:
            path (str | Path): Filesystem path to the CSV file containing experimental
                data. Expected format is determined by the dataset_io module.

        Returns:
            DatasetPayload: Structured dataset containing x-values, y-values, and
                metadata extracted from the CSV file.
        """
        return load_csv_dataset(path)

    def run_fit(
        self,
        model_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        xdata: jnp.ndarray,
        ydata: jnp.ndarray,
        *,
        p0: jnp.ndarray | None = None,
        bounds: tuple[float, float] = (-jnp.inf, jnp.inf),
        result_id: str,
        dataset_id: str,
        model_id: str,
    ) -> FitResultRecord:
        """Execute nonlinear least-squares fitting on experimental data.

        Args:
            model_fn (Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]): Model function
                mapping (parameters, xdata) to predicted ydata.
            xdata (jnp.ndarray): Independent variable values from experimental data.
            ydata (jnp.ndarray): Dependent variable values to fit.
            p0 (jnp.ndarray | None): Initial parameter guess. If None, uses model
                defaults or random initialization.
            bounds (tuple[float, float]): Parameter bounds as (lower, upper) tuple.
                Defaults to unbounded optimization.
            result_id (str): Unique identifier for this fit result.
            dataset_id (str): Reference to the source dataset.
            model_id (str): Reference to the theoretical model being fit.

        Returns:
            FitResultRecord: Complete fit result including optimized parameters,
                residuals, diagnostics (convergence status, covariance, etc.), and
                execution metadata. Result is persisted to disk via ResultStore.
        """
        warn_if_no_accelerator()
        fit_result, diagnostics = run_nlsq_fit(
            model_fn,
            xdata,
            ydata,
            p0=p0,
            bounds=bounds,
        )
        record = FitResultRecord(
            result_id=result_id,
            dataset_id=dataset_id,
            model_id=model_id,
            parameter_estimates=fit_result.parameters,
            diagnostics=diagnostics.as_dict(),
            residuals=fit_result.residuals,
            execution_context={"backend": "jax"},
            status="completed",
        )
        self._store.save_fit(record)
        return record

    def run_inference(
        self,
        model: Callable[..., Any],
        *,
        fit_record: FitResultRecord,
        result_id: str,
        rng_seed: int = 0,
        num_warmup: int = 1000,
        num_samples: int = 1000,
    ) -> PosteriorResultRecord:
        """Execute Bayesian inference using NUTS sampling.

        Args:
            model (Callable[..., Any]): NumPyro probabilistic model defining priors and
                likelihood function.
            fit_record (FitResultRecord): Previous fit result providing initialization
                and data context for inference.
            result_id (str): Unique identifier for this inference result.
            rng_seed (int): Random number generator seed for reproducibility. Defaults
                to 0.
            num_warmup (int): Number of NUTS warmup iterations for sampler adaptation.
                Defaults to 1000.
            num_samples (int): Number of posterior samples to draw after warmup.
                Defaults to 1000.

        Returns:
            PosteriorResultRecord: Posterior samples, diagnostics (ESS, Rhat,
                divergences, tree depth), and metadata. Result is persisted to disk
                via ResultStore.
        """
        warn_if_no_accelerator()
        record = run_nuts_inference(
            model,
            fit_record=fit_record,
            result_id=result_id,
            rng_seed=rng_seed,
            num_warmup=num_warmup,
            num_samples=num_samples,
        )
        self._store.save_posterior(record)
        return record

    def resume_inference(self, result_id: str) -> PosteriorResultRecord:
        """Load a previously computed inference result from disk.

        Args:
            result_id (str): Unique identifier of the posterior result to retrieve.

        Returns:
            PosteriorResultRecord: Previously computed posterior samples and diagnostics
                loaded from persistent storage.
        """
        return self._store.load_posterior(result_id)


__all__ = ["ComputeService", "DatasetPayload", "FitDiagnostics"]
