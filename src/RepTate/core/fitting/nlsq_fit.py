"""Deterministic fitting utilities powered by NLSQ."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Literal

import jax
import jax.numpy as jnp

from RepTate.core.fitting.nlsq_optimize import fit
from RepTate.core.io.datasets import DatasetPayload, export_dataset

logger = logging.getLogger("RepTate")


@dataclass(frozen=True)
class FitResult:
    """Deterministic nonlinear least-squares fit result.

    Contains parameter estimates, covariance matrix, residuals, and warm-start
    values for subsequent optimization or Bayesian inference.

    Attributes:
        parameters: Dictionary mapping parameter names (p0, p1, ...) to their
            fitted values.
        parameters_array: JAX array of fitted parameter values in sequential order.
        covariance: Parameter covariance matrix estimated from the Jacobian at
            the optimal point. Shape (n_params, n_params).
        residuals: List of residual values (ydata - model predictions) at the
            optimal parameter values.
        warm_start: Dictionary of parameter values suitable for warm-starting
            subsequent fits or MCMC chains. Same as parameters.
    """
    parameters: dict[str, float]
    parameters_array: jnp.ndarray
    covariance: jnp.ndarray
    residuals: list[float]
    warm_start: dict[str, float]


@dataclass(frozen=True)
class FitDiagnostics:
    """Diagnostic information from a deterministic fit.

    Captures convergence status and function evaluation count for monitoring
    fit quality and computational cost.

    Attributes:
        nfev: Number of function evaluations performed during optimization.
            May be None if not tracked by the optimizer.
        status: Convergence status string (e.g., "success", "max_iter_reached").
    """
    nfev: int | None
    status: str

    def as_dict(self) -> dict[str, object]:
        """Convert diagnostics to a dictionary for serialization.

        Returns:
            dict[str, object]: Dictionary with keys "nfev" and "status".
        """
        return {"nfev": self.nfev, "status": self.status}


def run_nlsq_fit(
    model_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    xdata: jnp.ndarray,
    ydata: jnp.ndarray,
    *,
    p0: jnp.ndarray | None = None,
    bounds: tuple[float, float] = (-jnp.inf, jnp.inf),
    workflow: Literal["auto", "auto_global", "hpc"] | None = "auto",
    show_progress: bool = False,
) -> tuple[FitResult, FitDiagnostics]:
    """Execute deterministic nonlinear least-squares fit using NLSQ.

    Fits a nonlinear model to data by minimizing the sum of squared residuals.
    Uses the NLSQ library for automatic differentiation and trust-region optimization
    with workflow-based memory management for automatic handling of large datasets.

    Args:
        model_fn: Callable taking (xdata, parameters) and returning model predictions.
            Both inputs and output should be JAX arrays.
        xdata: Independent variable data. Shape (n_points,) or (n_points, n_features).
        ydata: Dependent variable data to fit. Shape (n_points,).
        p0: Initial parameter guess. If None, NLSQ will use zeros or heuristics.
        bounds: Parameter bounds as (lower, upper). Can be scalars (applied to all
            parameters) or arrays matching parameter shape. Default is unbounded.
        workflow: Memory management strategy. One of:
            - "auto" (default): Memory-aware local optimization with automatic
              chunking/streaming for large datasets.
            - "auto_global": Memory-aware global optimization (requires bounds).
            - "hpc": Global optimization with checkpointing for HPC environments.
        show_progress: Display progress bar for large dataset operations.

    Returns:
        tuple[FitResult, FitDiagnostics]: A tuple containing:
            - FitResult with parameter estimates, covariance, and residuals
            - FitDiagnostics with convergence status
    """

    # JIT-compile the model function for optimal performance (FR-013)
    jit_model_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    try:
        jit_model_fn = jax.jit(model_fn)
    except Exception as e:
        # Fallback to non-JIT if compilation fails (edge case handling)
        logger.warning(f"JIT compilation failed, using non-JIT fallback: {e}")
        jit_model_fn = model_fn

    def curve_model(x: jnp.ndarray, *params: float) -> jnp.ndarray:
        """Adapter function to unpack variadic parameters for fit.

        Args:
            x: Independent variable array.
            *params: Unpacked parameter values.

        Returns:
            jnp.ndarray: Model predictions at x with given parameters.
        """
        return jit_model_fn(x, jnp.asarray(params))

    # Run fit with GPU memory exhaustion fallback to CPU (T029a)
    try:
        popt, pcov = fit(
            curve_model,
            xdata,
            ydata,
            p0=p0,
            bounds=bounds,
            workflow=workflow,
            show_progress=show_progress,
        )
    except (jax.errors.XlaRuntimeError, MemoryError) as e:
        # GPU memory exhaustion - fallback to CPU
        logger.info(f"GPU memory exhaustion detected, falling back to CPU: {e}")
        with jax.default_device(jax.devices("cpu")[0]):
            popt, pcov = fit(
                curve_model,
                xdata,
                ydata,
                p0=p0,
                bounds=bounds,
                workflow=workflow,
                show_progress=show_progress,
            )

    residuals = jnp.asarray(ydata - model_fn(xdata, jnp.asarray(popt)))
    parameters = {f"p{i}": float(val) for i, val in enumerate(popt)}
    result = FitResult(
        parameters=parameters,
        parameters_array=jnp.asarray(popt),
        covariance=jnp.asarray(pcov),
        residuals=[float(val) for val in residuals],
        warm_start=parameters,
    )
    diagnostics = FitDiagnostics(nfev=None, status="success")
    return result, diagnostics


def export_fit_inputs(
    payload: DatasetPayload, destination: str, *, fmt: str = "json"
) -> None:
    """Export dataset inputs used for fitting to disk.

    Saves the dataset payload (data and metadata) to a file for archival,
    reproducibility, or sharing. Delegates to the core export_dataset function.

    Args:
        payload: DatasetPayload containing the dataset record and numerical data
            to export.
        destination: File path where the dataset should be written. The file
            extension is ignored; format is determined by the fmt parameter.
        fmt: Output format. Supported values are "json" (default), "csv", or "tsv".

    Raises:
        ValueError: If fmt is not a supported format.
    """
    export_dataset(payload, destination, fmt=fmt)
