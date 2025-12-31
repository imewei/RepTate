"""Enhanced feature flag system with rollout control and YAML configuration.

This module extends the base feature flag system with:
- Percentage-based rollout (0-100%)
- User/session-based flags
- Environment variable overrides (REPTATE_FEATURE_*)
- YAML-based configuration
- Runtime monitoring and metrics

Usage:
    from RepTate.core.feature_flags_enhanced import FeatureFlagManager

    flag_manager = FeatureFlagManager()
    if flag_manager.is_enabled('use_jax_integration', user_id='session_123'):
        # Use new JAX implementation
        pass
    else:
        # Fall back to legacy scipy
        pass
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

logger = logging.getLogger(__name__)


@dataclass
class FeatureFlagConfig:
    """Configuration for a single feature flag.

    Attributes:
        name: Unique identifier for the flag
        enabled: Default enabled state
        rollout_percentage: Percentage of users to enable (0-100)
        description: Human-readable description
        tags: Tags for categorization (e.g., ['migration', 'gui'])
        depends_on: List of flags that must be enabled for this flag
        environment_override: Whether to allow env var override
    """
    name: str
    enabled: bool = False
    rollout_percentage: int = 0
    description: str = ""
    tags: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    environment_override: bool = True


@dataclass
class FlagEvaluation:
    """Result of a feature flag evaluation.

    Attributes:
        flag_name: Name of the evaluated flag
        enabled: Whether the flag is enabled for this evaluation
        reason: Reason for the decision
        context: Additional context (user_id, session, etc.)
    """
    flag_name: str
    enabled: bool
    reason: str
    context: dict[str, Any] = field(default_factory=dict)


class FeatureFlagManager:
    """Manages feature flags with advanced rollout capabilities.

    This manager supports:
    - YAML configuration file loading
    - Environment variable overrides (REPTATE_FEATURE_<FLAG_NAME>=true|false)
    - Percentage-based rollout using consistent hashing
    - Dependency resolution between flags
    - Thread-safe flag evaluation
    - Metrics collection for observability
    """

    def __init__(
        self,
        config_path: Path | str | None = None,
        enable_metrics: bool = True
    ):
        """Initialize the feature flag manager.

        Args:
            config_path: Path to YAML configuration file (optional)
            enable_metrics: Whether to collect metrics for flag usage
        """
        self._flags: dict[str, FeatureFlagConfig] = {}
        self._lock = threading.RLock()
        self._enable_metrics = enable_metrics
        self._evaluations: list[FlagEvaluation] = []
        self._evaluation_counts: dict[str, dict[str, int]] = {}

        # Initialize default flags for RepTate migration
        self._init_default_flags()

        # Load configuration from YAML if provided
        if config_path:
            self.load_config(config_path)

    def _init_default_flags(self) -> None:
        """Initialize default feature flags for RepTate migration."""
        default_flags = [
            FeatureFlagConfig(
                name='use_jax_integration',
                enabled=True,
                rollout_percentage=100,
                description='Use JAX-based optimization instead of scipy.optimize',
                tags=['migration', 'jax', 'optimization'],
                depends_on=[],
            ),
            FeatureFlagConfig(
                name='use_decomposed_gui',
                enabled=False,
                rollout_percentage=0,
                description='Use decomposed GUI controllers instead of god classes',
                tags=['refactoring', 'gui', 'architecture'],
                depends_on=[],
            ),
            FeatureFlagConfig(
                name='use_jax_native',
                enabled=False,
                rollout_percentage=0,
                description='Use JAX implementations instead of native C libraries',
                tags=['migration', 'jax', 'native'],
                depends_on=['use_jax_integration'],
            ),
            FeatureFlagConfig(
                name='use_safe_eval',
                enabled=True,
                rollout_percentage=100,
                description='Use safe expression evaluator instead of eval()',
                tags=['security', 'migration'],
                depends_on=[],
            ),
            FeatureFlagConfig(
                name='use_safe_serialization',
                enabled=True,
                rollout_percentage=100,
                description='Use JSON/NPZ serialization instead of pickle',
                tags=['security', 'migration', 'serialization'],
                depends_on=[],
            ),
        ]

        for flag_config in default_flags:
            self._flags[flag_config.name] = flag_config

    def load_config(self, config_path: Path | str) -> None:
        """Load feature flag configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Raises:
            ImportError: If PyYAML is not installed
            FileNotFoundError: If config file doesn't exist
            ValueError: If config format is invalid
        """
        if not HAS_YAML:
            raise ImportError(
                "PyYAML is required for YAML configuration. "
                "Install with: pip install pyyaml"
            )

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        if not isinstance(config_data, dict) or 'flags' not in config_data:
            raise ValueError(
                "Invalid config format. Expected 'flags' key at root level."
            )

        with self._lock:
            for flag_name, flag_data in config_data['flags'].items():
                self._flags[flag_name] = FeatureFlagConfig(
                    name=flag_name,
                    enabled=flag_data.get('enabled', False),
                    rollout_percentage=flag_data.get('rollout_percentage', 0),
                    description=flag_data.get('description', ''),
                    tags=flag_data.get('tags', []),
                    depends_on=flag_data.get('depends_on', []),
                    environment_override=flag_data.get('environment_override', True),
                )

        logger.info(f"Loaded {len(config_data['flags'])} flags from {config_path}")

    def is_enabled(
        self,
        flag_name: str,
        user_id: str | None = None,
        context: dict[str, Any] | None = None
    ) -> bool:
        """Check if a feature flag is enabled.

        The evaluation follows this precedence:
        1. Environment variable override (if allowed)
        2. Dependency check (all depends_on must be enabled)
        3. Rollout percentage with consistent hashing

        Args:
            flag_name: Name of the feature flag
            user_id: Optional user/session ID for consistent rollout
            context: Additional context for evaluation

        Returns:
            True if the feature is enabled, False otherwise

        Raises:
            KeyError: If flag_name is not recognized
        """
        if flag_name not in self._flags:
            raise KeyError(f"Unknown feature flag: '{flag_name}'")

        flag_config = self._flags[flag_name]
        eval_context = context or {}
        if user_id:
            eval_context['user_id'] = user_id

        # Check environment override
        if flag_config.environment_override:
            env_var_name = f'REPTATE_FEATURE_{flag_name.upper()}'
            env_value = os.environ.get(env_var_name)
            if env_value is not None:
                enabled = self._parse_bool(env_value)
                self._record_evaluation(
                    flag_name, enabled, 'environment_override', eval_context
                )
                return enabled

        # Check dependencies
        for dep_flag in flag_config.depends_on:
            if not self.is_enabled(dep_flag, user_id, context):
                self._record_evaluation(
                    flag_name, False, f'dependency_failed:{dep_flag}', eval_context
                )
                return False

        # Check base enabled state
        if not flag_config.enabled:
            self._record_evaluation(
                flag_name, False, 'disabled', eval_context
            )
            return False

        # Check rollout percentage
        if flag_config.rollout_percentage >= 100:
            self._record_evaluation(
                flag_name, True, 'full_rollout', eval_context
            )
            return True

        if flag_config.rollout_percentage <= 0:
            self._record_evaluation(
                flag_name, False, 'zero_rollout', eval_context
            )
            return False

        # Use consistent hashing for rollout
        enabled = self._check_rollout(flag_name, user_id, flag_config.rollout_percentage)
        reason = 'rollout_enabled' if enabled else 'rollout_disabled'
        self._record_evaluation(flag_name, enabled, reason, eval_context)
        return enabled

    def _check_rollout(
        self,
        flag_name: str,
        user_id: str | None,
        percentage: int
    ) -> bool:
        """Check if user is in rollout percentage using consistent hashing.

        Args:
            flag_name: Name of the flag
            user_id: User/session ID (uses 'default' if None)
            percentage: Rollout percentage (0-100)

        Returns:
            True if user is in the rollout percentage
        """
        user_id = user_id or 'default'
        hash_input = f"{flag_name}:{user_id}"
        hash_value = hashlib.sha256(hash_input.encode()).hexdigest()
        hash_int = int(hash_value[:8], 16)
        bucket = hash_int % 100
        return bucket < percentage

    def _parse_bool(self, value: str) -> bool:
        """Parse string value to boolean.

        Args:
            value: String to parse

        Returns:
            Boolean value

        Raises:
            ValueError: If value cannot be parsed
        """
        lower_value = value.lower().strip()
        if lower_value in ('true', '1', 'yes', 'on'):
            return True
        if lower_value in ('false', '0', 'no', 'off'):
            return False
        raise ValueError(f"Cannot parse '{value}' as boolean")

    def _record_evaluation(
        self,
        flag_name: str,
        enabled: bool,
        reason: str,
        context: dict[str, Any]
    ) -> None:
        """Record a flag evaluation for metrics.

        Args:
            flag_name: Name of the flag
            enabled: Whether it was enabled
            reason: Reason for the decision
            context: Evaluation context
        """
        if not self._enable_metrics:
            return

        with self._lock:
            evaluation = FlagEvaluation(flag_name, enabled, reason, context)
            self._evaluations.append(evaluation)

            # Update counts
            if flag_name not in self._evaluation_counts:
                self._evaluation_counts[flag_name] = {'enabled': 0, 'disabled': 0}

            if enabled:
                self._evaluation_counts[flag_name]['enabled'] += 1
            else:
                self._evaluation_counts[flag_name]['disabled'] += 1

    def get_flag_status(self, flag_name: str) -> dict[str, Any]:
        """Get detailed status of a feature flag.

        Args:
            flag_name: Name of the flag

        Returns:
            Dictionary with flag configuration and metrics

        Raises:
            KeyError: If flag_name is not recognized
        """
        if flag_name not in self._flags:
            raise KeyError(f"Unknown feature flag: '{flag_name}'")

        flag_config = self._flags[flag_name]
        with self._lock:
            counts = self._evaluation_counts.get(
                flag_name, {'enabled': 0, 'disabled': 0}
            )

        return {
            'name': flag_config.name,
            'enabled': flag_config.enabled,
            'rollout_percentage': flag_config.rollout_percentage,
            'description': flag_config.description,
            'tags': flag_config.tags,
            'depends_on': flag_config.depends_on,
            'evaluation_count': counts['enabled'] + counts['disabled'],
            'enabled_count': counts['enabled'],
            'disabled_count': counts['disabled'],
        }

    def get_all_flags(self) -> dict[str, dict[str, Any]]:
        """Get status of all feature flags.

        Returns:
            Dictionary mapping flag names to their status
        """
        return {
            flag_name: self.get_flag_status(flag_name)
            for flag_name in self._flags
        }

    def export_metrics(self) -> dict[str, Any]:
        """Export metrics for observability dashboard.

        Returns:
            Dictionary containing evaluation metrics
        """
        with self._lock:
            return {
                'total_evaluations': len(self._evaluations),
                'flags': self.get_all_flags(),
                'recent_evaluations': [
                    {
                        'flag': e.flag_name,
                        'enabled': e.enabled,
                        'reason': e.reason,
                        'context': e.context,
                    }
                    for e in self._evaluations[-100:]  # Last 100 evaluations
                ],
            }

    def update_flag(
        self,
        flag_name: str,
        enabled: bool | None = None,
        rollout_percentage: int | None = None
    ) -> None:
        """Update a feature flag configuration at runtime.

        Args:
            flag_name: Name of the flag to update
            enabled: New enabled state (optional)
            rollout_percentage: New rollout percentage (optional)

        Raises:
            KeyError: If flag_name is not recognized
            ValueError: If rollout_percentage is not in range [0, 100]
        """
        if flag_name not in self._flags:
            raise KeyError(f"Unknown feature flag: '{flag_name}'")

        if rollout_percentage is not None and not 0 <= rollout_percentage <= 100:
            raise ValueError(
                f"Rollout percentage must be in range [0, 100], got {rollout_percentage}"
            )

        with self._lock:
            if enabled is not None:
                self._flags[flag_name].enabled = enabled
            if rollout_percentage is not None:
                self._flags[flag_name].rollout_percentage = rollout_percentage

        logger.info(f"Updated flag '{flag_name}': enabled={enabled}, rollout={rollout_percentage}")


# Global singleton instance
_global_manager: FeatureFlagManager | None = None
_manager_lock = threading.Lock()


def get_feature_flag_manager() -> FeatureFlagManager:
    """Get the global feature flag manager instance.

    Returns:
        Global FeatureFlagManager instance
    """
    global _global_manager

    if _global_manager is None:
        with _manager_lock:
            if _global_manager is None:
                # Check for config file in standard locations
                config_paths = [
                    Path.cwd() / 'feature_flags.yaml',
                    Path.home() / '.reptate' / 'feature_flags.yaml',
                    Path('/etc/reptate/feature_flags.yaml'),
                ]

                config_path = None
                for path in config_paths:
                    if path.exists():
                        config_path = path
                        break

                _global_manager = FeatureFlagManager(config_path=config_path)

    return _global_manager


def is_enabled(flag_name: str, user_id: str | None = None) -> bool:
    """Convenience function to check if a feature flag is enabled.

    Args:
        flag_name: Name of the feature flag
        user_id: Optional user/session ID

    Returns:
        True if the feature is enabled, False otherwise
    """
    manager = get_feature_flag_manager()
    return manager.is_enabled(flag_name, user_id)
