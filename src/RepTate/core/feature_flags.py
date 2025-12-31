"""Feature flag system for safe rollout control.

This module provides a centralized feature flag mechanism that supports:
- Default flag values for new features
- Environment variable overrides (REPTATE_<FLAG_NAME>=true|false)
- Runtime flag queries

Environment variables override defaults. Example:
    REPTATE_USE_SAFE_EVAL=false python -m RepTate

Usage:
    from RepTate.core.feature_flags import is_enabled, FEATURES

    if is_enabled('USE_SAFE_EVAL'):
        # Use new safe_eval implementation
        pass
    else:
        # Fall back to legacy implementation
        pass
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class FeatureFlag:
    """Definition of a feature flag.

    Attributes:
        name: Flag identifier (used in FEATURES dict and env vars)
        default: Default value when no override is set
        description: Human-readable description of the feature
    """
    name: str
    default: bool
    description: str


# Feature flag definitions
# Environment variables use the pattern: REPTATE_<FLAG_NAME>=true|false
_FLAG_DEFINITIONS: Final[tuple[FeatureFlag, ...]] = (
    FeatureFlag(
        name='USE_SAFE_EVAL',
        default=True,
        description='Use safe expression evaluator instead of eval() (Sprint 1)'
    ),
    FeatureFlag(
        name='USE_SAFE_SERIALIZATION',
        default=True,
        description='Use JSON/NPZ serialization instead of pickle (Sprint 2)'
    ),
    FeatureFlag(
        name='USE_JAX_OPTIMIZATION',
        default=True,
        description='Use JAX-based optimization instead of scipy.optimize (Sprint 7-8)'
    ),
)


# Default feature values (exported for documentation/inspection)
FEATURES: Final[dict[str, bool]] = {
    flag.name: flag.default for flag in _FLAG_DEFINITIONS
}


def _parse_bool(value: str) -> bool:
    """Parse a string value to boolean.

    Args:
        value: String to parse (case-insensitive)

    Returns:
        True for 'true', '1', 'yes', 'on'
        False for 'false', '0', 'no', 'off'

    Raises:
        ValueError: If value cannot be parsed as boolean
    """
    lower_value = value.lower().strip()
    if lower_value in ('true', '1', 'yes', 'on'):
        return True
    if lower_value in ('false', '0', 'no', 'off'):
        return False
    raise ValueError(f"Cannot parse '{value}' as boolean")


def is_enabled(flag_name: str) -> bool:
    """Check if a feature flag is enabled.

    Checks for environment variable override first (REPTATE_<FLAG_NAME>),
    then falls back to default value.

    Args:
        flag_name: Name of the feature flag (e.g., 'USE_SAFE_EVAL')

    Returns:
        True if the feature is enabled, False otherwise

    Raises:
        KeyError: If flag_name is not a recognized feature flag

    Examples:
        >>> # With default values (no env override)
        >>> is_enabled('USE_SAFE_EVAL')
        True

        >>> # Unknown flag raises KeyError
        >>> is_enabled('UNKNOWN_FLAG')
        Traceback (most recent call last):
            ...
        KeyError: "Unknown feature flag: 'UNKNOWN_FLAG'"
    """
    if flag_name not in FEATURES:
        raise KeyError(f"Unknown feature flag: '{flag_name}'")

    # Check for environment variable override
    env_var_name = f'REPTATE_{flag_name}'
    env_value = os.environ.get(env_var_name)

    if env_value is not None:
        try:
            return _parse_bool(env_value)
        except ValueError:
            # Invalid env value, fall back to default
            import warnings
            warnings.warn(
                f"Invalid value '{env_value}' for {env_var_name}, "
                f"using default: {FEATURES[flag_name]}",
                UserWarning,
                stacklevel=2
            )

    return FEATURES[flag_name]


def get_all_flags() -> dict[str, bool]:
    """Get the current state of all feature flags.

    Returns a dictionary of all flags with their effective values
    (after applying environment variable overrides).

    Returns:
        Dictionary mapping flag names to their effective boolean values
    """
    return {flag_name: is_enabled(flag_name) for flag_name in FEATURES}


def get_flag_info() -> dict[str, dict[str, str | bool]]:
    """Get detailed information about all feature flags.

    Returns:
        Dictionary mapping flag names to their metadata including:
        - default: The default value
        - current: The current effective value
        - description: Human-readable description
        - env_var: The environment variable name for overriding
    """
    result: dict[str, dict[str, str | bool]] = {}
    for flag in _FLAG_DEFINITIONS:
        result[flag.name] = {
            'default': flag.default,
            'current': is_enabled(flag.name),
            'description': flag.description,
            'env_var': f'REPTATE_{flag.name}',
        }
    return result
