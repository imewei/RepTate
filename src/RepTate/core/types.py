"""Shared typed data structures for RepTate workflows."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ParameterBounds:
    """Parameter bounds specification for optimization.

    Defines lower and upper bounds for a single fitting parameter.

    Attributes:
        name: Parameter identifier (must match parameter key in FitProblem.parameters)
        lower: Minimum allowed value (None for unbounded below)
        upper: Maximum allowed value (None for unbounded above)
    """
    name: str
    lower: float | None
    upper: float | None


@dataclass(frozen=True)
class FitProblem:
    """Complete specification of a fitting problem.

    Encapsulates all information needed to execute a fit: dataset, model,
    initial parameters, bounds, and solver options.

    Attributes:
        dataset_id: Unique identifier for the dataset being fitted
        model_id: Unique identifier for the theoretical model
        parameters: Initial parameter values (parameter_name -> value)
        bounds: Parameter bounds (parameter_name -> ParameterBounds)
        options: Solver-specific options (e.g., max_iterations, tolerance)
    """
    dataset_id: str
    model_id: str
    parameters: dict[str, float]
    bounds: dict[str, ParameterBounds] = field(default_factory=dict)
    options: dict[str, float | int | str | bool] = field(default_factory=dict)


@dataclass(frozen=True)
class FitDiagnostics:
    """Fitting results and diagnostic information.

    Contains the outcome status, quality metrics, and diagnostic messages
    from a fitting operation.

    Attributes:
        status: Fit outcome ('success', 'max_iterations', 'failed', etc.)
        metrics: Quality metrics (chi2, reduced_chi2, aic, bic, etc.)
        messages: Diagnostic messages and warnings from the fitting procedure
    """
    status: str
    metrics: dict[str, float]
    messages: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class UncertaintySummary:
    """Parameter uncertainty estimates from Bayesian inference.

    Summarizes posterior distributions for fitted parameters including
    point estimates, credible intervals, and convergence diagnostics.

    Attributes:
        parameter_summaries: Per-parameter statistics (mean, std, median, q5, q95, etc.)
        diagnostics: Convergence diagnostics (r_hat, ess_bulk, ess_tail, etc.)
        messages: Warnings about convergence issues or sampling problems
    """
    parameter_summaries: dict[str, dict[str, float]]
    diagnostics: dict[str, float]
    messages: list[str] = field(default_factory=list)
