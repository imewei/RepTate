"""Deterministic fitting utilities powered by NLSQ."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp

from RepTate.core.fitting.nlsq_optimize import curve_fit


@dataclass(frozen=True)
class FitResult:
    parameters: dict[str, float]
    parameters_array: jnp.ndarray
    covariance: jnp.ndarray
    residuals: list[float]


@dataclass(frozen=True)
class FitDiagnostics:
    nfev: int | None
    status: str

    def as_dict(self) -> dict[str, object]:
        return {"nfev": self.nfev, "status": self.status}


def run_nlsq_fit(
    model_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    xdata: jnp.ndarray,
    ydata: jnp.ndarray,
    *,
    p0: jnp.ndarray | None = None,
    bounds: tuple[float, float] = (-jnp.inf, jnp.inf),
) -> tuple[FitResult, FitDiagnostics]:
    def curve_model(x, *params):
        return model_fn(x, jnp.asarray(params))

    popt, pcov = curve_fit(curve_model, xdata, ydata, p0=p0, bounds=bounds)
    residuals = jnp.asarray(ydata - model_fn(xdata, jnp.asarray(popt)))
    parameters = {f"p{i}": float(val) for i, val in enumerate(popt)}
    result = FitResult(
        parameters=parameters,
        parameters_array=jnp.asarray(popt),
        covariance=jnp.asarray(pcov),
        residuals=[float(val) for val in residuals],
    )
    diagnostics = FitDiagnostics(nfev=None, status="success")
    return result, diagnostics
