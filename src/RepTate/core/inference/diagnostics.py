"""MCMC convergence diagnostics and reproducibility metadata."""

from __future__ import annotations

import importlib.metadata
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import arviz as az
import jax
import jax.numpy as jnp
import numpyro

if TYPE_CHECKING:
    from numpyro.infer import MCMC

logger = logging.getLogger("RepTate")


@dataclass(frozen=True)
class ConvergenceDiagnostics:
    """MCMC convergence diagnostic metrics.

    Contains R-hat, effective sample size (ESS), divergence counts, and any
    warning messages from the diagnostic checks. All metrics are computed via ArviZ.

    Attributes:
        r_hat: R-hat (potential scale reduction factor) for each parameter.
            Values > 1.01 indicate potential non-convergence.
        ess_bulk: Bulk effective sample size for each parameter.
            Values < 400 indicate insufficient samples for reliable inference.
        ess_tail: Tail effective sample size for each parameter.
            Low values indicate poor tail estimation.
        divergences: Count of divergent transitions during sampling.
            Any divergences indicate potential bias in posterior estimates.
        warnings: List of warning messages generated during diagnostic checks.
    """

    r_hat: dict[str, float]
    ess_bulk: dict[str, float]
    ess_tail: dict[str, float]
    divergences: int
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReproducibilityInfo:
    """Metadata required to reproduce an inference run.

    Captures the random seed, software versions, and model configuration
    needed to recreate identical results on compatible systems.

    Attributes:
        rng_seed: Random seed used for NUTS sampling.
        jax_version: JAX library version string.
        numpyro_version: NumPyro library version string.
        reptate_version: RepTate package version string.
        model_config: Snapshot of model configuration parameters (model_kwargs).
    """

    rng_seed: int
    jax_version: str
    numpyro_version: str
    reptate_version: str
    model_config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DiagnosticsReport:
    """User-facing summary of MCMC diagnostics.

    Combines raw diagnostics with interpretation flags and human-readable
    summary text for display in the UI or logs.

    Attributes:
        converged: True if all R-hat values <= 1.01 and not 100% divergent.
        diagnostics: The underlying ConvergenceDiagnostics data.
        summary_text: Human-readable summary of convergence status.
    """

    converged: bool
    diagnostics: ConvergenceDiagnostics
    summary_text: str


def compute_diagnostics(mcmc: MCMC) -> ConvergenceDiagnostics:
    """Compute MCMC convergence diagnostics from a completed NUTS run.

    Uses ArviZ to compute R-hat and ESS metrics, and extracts divergence
    information from NumPyro's extra fields. Emits warnings for any
    parameters with R-hat > 1.01 or ESS < 400.

    Args:
        mcmc: A NumPyro MCMC object after run() has completed.

    Returns:
        ConvergenceDiagnostics containing R-hat, ESS, divergence count,
        and any warning messages.
    """
    # Convert NumPyro MCMC to ArviZ InferenceData
    idata = az.from_numpyro(mcmc)  # type: ignore[no-untyped-call]

    # Compute R-hat for each parameter
    r_hat_data = az.rhat(idata)  # type: ignore[no-untyped-call]
    r_hat: dict[str, float] = {}
    for var_name in r_hat_data.data_vars:
        val = r_hat_data[var_name].values
        # Handle scalar vs array values
        if hasattr(val, "item"):
            r_hat[str(var_name)] = float(val.item())
        else:
            r_hat[str(var_name)] = float(val)

    # Compute bulk ESS for each parameter
    ess_bulk_data = az.ess(idata, method="bulk")  # type: ignore[no-untyped-call]
    ess_bulk: dict[str, float] = {}
    for var_name in ess_bulk_data.data_vars:
        val = ess_bulk_data[var_name].values
        if hasattr(val, "item"):
            ess_bulk[str(var_name)] = float(val.item())
        else:
            ess_bulk[str(var_name)] = float(val)

    # Compute tail ESS for each parameter
    ess_tail_data = az.ess(idata, method="tail")  # type: ignore[no-untyped-call]
    ess_tail: dict[str, float] = {}
    for var_name in ess_tail_data.data_vars:
        val = ess_tail_data[var_name].values
        if hasattr(val, "item"):
            ess_tail[str(var_name)] = float(val.item())
        else:
            ess_tail[str(var_name)] = float(val)

    # Extract divergence count from NumPyro extra fields
    extra_fields = mcmc.get_extra_fields()
    diverging = extra_fields.get("diverging", jnp.array([]))
    divergences = int(jnp.sum(diverging))

    # Generate warnings for diagnostic issues
    warnings: list[str] = []

    for param, val in r_hat.items():
        if val > 1.01:
            msg = f"R-hat for {param} is {val:.3f} > 1.01"
            warnings.append(msg)
            logger.warning(msg)

    for param, val in ess_bulk.items():
        if val < 400:
            msg = f"ESS for {param} is {val:.0f} < 400"
            warnings.append(msg)
            logger.warning(msg)

    return ConvergenceDiagnostics(
        r_hat=r_hat,
        ess_bulk=ess_bulk,
        ess_tail=ess_tail,
        divergences=divergences,
        warnings=warnings,
    )


def collect_reproducibility_info(
    rng_seed: int,
    model_config: dict[str, Any] | None = None,
) -> ReproducibilityInfo:
    """Collect software versions and configuration for reproducibility.

    Gathers version information from JAX, NumPyro, and RepTate packages
    along with the random seed and model configuration used for inference.

    Args:
        rng_seed: The random seed used for NUTS sampling.
        model_config: Optional model configuration parameters (model_kwargs).

    Returns:
        ReproducibilityInfo containing seed, versions, and configuration.
    """
    # Get RepTate version, falling back to importlib.metadata if __version__ unavailable
    try:
        reptate_version = importlib.metadata.version("RepTate")
    except importlib.metadata.PackageNotFoundError:
        reptate_version = "unknown"

    return ReproducibilityInfo(
        rng_seed=rng_seed,
        jax_version=jax.__version__,
        numpyro_version=numpyro.__version__,
        reptate_version=reptate_version,
        model_config=model_config or {},
    )


def create_diagnostics_report(diagnostics: ConvergenceDiagnostics) -> DiagnosticsReport:
    """Create a user-facing diagnostics report.

    Analyzes the convergence diagnostics and generates a human-readable
    summary with overall convergence status.

    Args:
        diagnostics: ConvergenceDiagnostics from compute_diagnostics().

    Returns:
        DiagnosticsReport with converged flag and summary text.
    """
    # Check convergence criteria
    all_rhat_ok = all(val <= 1.01 for val in diagnostics.r_hat.values())
    converged = all_rhat_ok and len(diagnostics.warnings) == 0

    # Build summary text
    lines = []
    if converged:
        lines.append("Convergence: OK")
        lines.append("R-hat: all parameters <= 1.01")
    else:
        lines.append("Convergence: WARNINGS DETECTED")
        if not all_rhat_ok:
            lines.append("R-hat: some parameters > 1.01")

    if diagnostics.ess_bulk:
        min_ess = min(diagnostics.ess_bulk.values())
        max_ess = max(diagnostics.ess_bulk.values())
        lines.append(f"ESS (bulk): min={min_ess:.0f}, max={max_ess:.0f}")

    if diagnostics.ess_tail:
        min_ess = min(diagnostics.ess_tail.values())
        max_ess = max(diagnostics.ess_tail.values())
        lines.append(f"ESS (tail): min={min_ess:.0f}, max={max_ess:.0f}")

    lines.append(f"Divergences: {diagnostics.divergences}")

    summary_text = "\n".join(lines)

    return DiagnosticsReport(
        converged=converged,
        diagnostics=diagnostics,
        summary_text=summary_text,
    )
