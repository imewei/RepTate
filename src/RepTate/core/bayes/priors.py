"""Default prior definitions for Bayesian inference."""

from __future__ import annotations

import numpyro.distributions as dist


def default_priors() -> dict[str, dist.Distribution]:
    """Return default priors for model parameters."""
    return {
        "sigma": dist.HalfNormal(1.0),
    }
