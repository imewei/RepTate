"""Result record models for deterministic fits and Bayesian inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class DatasetRecord:
    """Immutable record of a dataset used in fitting or inference.

    Captures the essential metadata about a dataset, including its unique
    identifier, human-readable name, source file or location, and any
    additional metadata needed for provenance tracking.

    Attributes:
        dataset_id: Unique identifier for the dataset.
        name: Human-readable name for the dataset.
        source_location: File path or URI where the dataset originated.
        metadata: Additional key-value metadata (e.g., units, processing history).
    """
    dataset_id: str
    name: str
    source_location: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelRecord:
    """Immutable record of a model configuration used in fitting or inference.

    Stores the model identifier, name, and parameter specifications. This record
    captures the model definition independently of any particular fit result.

    Attributes:
        model_id: Unique identifier for the model (typically the class name).
        name: Human-readable name for the model.
        parameters: Dictionary of parameter names to their specifications
            (e.g., initial values, bounds, priors).
    """
    model_id: str
    name: str
    parameters: dict[str, Any]


@dataclass
class FitResultRecord:
    """Mutable record of a deterministic model fitting result.

    Stores the complete output of a model fitting operation, including
    parameter estimates, diagnostic metrics, residuals, and execution
    metadata. This record links a specific dataset and model configuration
    to the optimization results.

    Attributes:
        result_id: Unique identifier for this fit result.
        dataset_id: Identifier of the dataset used for fitting.
        model_id: Identifier of the model used for fitting.
        parameter_estimates: Dictionary mapping parameter names to their
            fitted values.
        diagnostics: Dictionary of fit quality metrics (e.g., chi-squared,
            R-squared, convergence status).
        residuals: List of residual values (observed - predicted).
        execution_context: Dictionary capturing execution environment details
            (e.g., optimizer settings, runtime, system info).
        status: String indicating fit status (e.g., 'success', 'failed',
            'converged', 'max_iterations').
        created_at: ISO 8601 timestamp of when the fit was created.
    """
    result_id: str
    dataset_id: str
    model_id: str
    parameter_estimates: dict[str, float]
    diagnostics: dict[str, Any]
    residuals: list[float]
    execution_context: dict[str, Any]
    status: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class PosteriorResultRecord:
    """Mutable record of a Bayesian inference posterior sampling result.

    Stores the complete output of a Bayesian inference operation, including
    MCMC sample traces, summary statistics, chain diagnostics, and state
    information for resuming sampling. This record is linked to a prior
    deterministic fit result that provides initialization.

    Attributes:
        result_id: Unique identifier for this posterior result.
        fit_result_id: Identifier of the deterministic fit result used for
            initialization or comparison.
        sample_traces: Dictionary mapping parameter names to lists of sampled
            values across all chains.
        summary_statistics: Dictionary of posterior summary metrics (e.g., mean,
            std, quantiles, ESS, R-hat).
        chain_metadata: Dictionary of MCMC chain diagnostics and settings
            (e.g., num_chains, num_samples, warmup_steps, sampler).
        resume_state: Dictionary containing serialized sampler state for
            resuming interrupted sampling runs.
        status: String indicating sampling status (e.g., 'completed', 'running',
            'failed', 'diverged').
        created_at: ISO 8601 timestamp of when the sampling was created.
    """
    result_id: str
    fit_result_id: str
    sample_traces: dict[str, list[float]]
    summary_statistics: dict[str, Any]
    chain_metadata: dict[str, Any]
    resume_state: dict[str, Any]
    status: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
