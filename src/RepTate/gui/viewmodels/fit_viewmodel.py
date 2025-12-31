"""Viewmodel for deterministic fit results."""

from __future__ import annotations

from dataclasses import dataclass

from RepTate.core.models import FitResultRecord


@dataclass(frozen=True)
class FitViewModel:
    """Viewmodel for presenting deterministic fit results in the GUI.

    This immutable viewmodel wraps FitResultRecord and exposes properties
    for displaying optimization results including fit status and parameter
    estimates from least-squares or other deterministic fitting methods.

    Attributes:
        result: The underlying FitResultRecord containing fit results.
    """

    result: FitResultRecord

    @property
    def status(self) -> str:
        """Get the fit status message.

        Returns:
            str: Status description (e.g., "converged", "failed", "max iterations").
        """
        return self.result.status

    @property
    def parameters(self) -> dict[str, float]:
        """Get the estimated parameter values.

        Returns:
            dict[str, float]: Mapping of parameter names to their fitted values.
        """
        return self.result.parameter_estimates
