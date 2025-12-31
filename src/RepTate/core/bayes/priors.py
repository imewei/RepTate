"""Default prior definitions for Bayesian inference."""

from __future__ import annotations

import numpyro.distributions as dist


def default_priors() -> dict[str, dist.Distribution]:
    """Return default priors for model parameters.

    Provides a minimal set of default prior distributions for Bayesian inference.
    Currently includes only the observation noise parameter (sigma). Users should
    extend this dictionary with model-specific parameter priors before building
    the likelihood.

    Returns:
        Dictionary mapping parameter names to NumPyro distribution objects.
        Contains:
        - 'sigma': HalfNormal(1.0) distribution for observation noise standard
          deviation. The half-normal prior ensures sigma is positive and
          weakly regularizes toward smaller noise values.
    """
    return {
        "sigma": dist.HalfNormal(1.0),
    }
