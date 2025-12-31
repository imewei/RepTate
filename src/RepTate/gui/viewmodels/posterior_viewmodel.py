"""Viewmodel for uncertainty summaries and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

from RepTate.core.models import PosteriorResultRecord


@dataclass(frozen=True)
class PosteriorViewModel:
    """Viewmodel for presenting uncertainty summaries and diagnostics.

    This immutable viewmodel wraps PosteriorResultRecord and exposes properties
    for displaying Bayesian inference results including MCMC sampling status
    and posterior summary statistics (means, credible intervals, etc.).

    Attributes:
        result: The underlying PosteriorResultRecord containing MCMC results.
    """

    result: PosteriorResultRecord

    @property
    def status(self) -> str:
        """Get the MCMC sampling status message.

        Returns:
            str: Status description (e.g., "converged", "divergent", "max samples").
        """
        return self.result.status

    @property
    def summaries(self) -> dict[str, object]:
        """Get the posterior summary statistics.

        Returns:
            dict[str, object]: Mapping of parameter names to their summary statistics
                including mean, standard deviation, credible intervals, and diagnostics.
        """
        return self.result.summary_statistics
