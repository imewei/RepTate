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
        return [asdict(spec) for spec in self._registry.list_models()]

    def import_dataset(self, path: str | Path) -> DatasetPayload:
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
        return self._store.load_posterior(result_id)


__all__ = ["ComputeService", "DatasetPayload", "FitDiagnostics"]
