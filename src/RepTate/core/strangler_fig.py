"""Strangler fig pattern integration utilities.

This module provides high-level utilities that combine feature flags,
circuit breakers, and dual-run patterns for safe migration.

Usage:
    from RepTate.core.strangler_fig import (
        with_migration,
        migrate_with_validation,
        MigrationDecorator
    )

    # Simple migration with automatic fallback
    @with_migration(
        name='jax_optimize',
        flag='use_jax_integration',
        legacy_impl=scipy_optimize
    )
    def jax_optimize(params):
        # New JAX implementation
        return optimized_params

    # Migration with result validation
    result = migrate_with_validation(
        new_impl=jax_function,
        legacy_impl=scipy_function,
        args=(data,),
        name='optimization',
        validate=True
    )
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, TypeVar

from RepTate.core.circuit_breaker import CircuitBreakerConfig, CircuitBreakerRegistry
from RepTate.core.dual_run import ComparisonStrategy, DualRunner, DualRunRegistry
from RepTate.core.feature_flags_enhanced import get_feature_flag_manager
from RepTate.core.migration_observability import get_dashboard

logger = logging.getLogger(__name__)

T = TypeVar('T')


class MigrationDecorator:
    """Decorator for migrating function implementations.

    This combines feature flags, circuit breakers, and dual-run validation
    into a single decorator for easy migration.
    """

    def __init__(
        self,
        name: str,
        flag: str,
        legacy_impl: Callable[..., T] | None = None,
        validate: bool = False,
        comparison_strategy: ComparisonStrategy = ComparisonStrategy.NUMERICAL_CLOSE,
        circuit_breaker_config: CircuitBreakerConfig | None = None,
        **comparison_kwargs: Any
    ):
        """Initialize the migration decorator.

        Args:
            name: Unique name for this migration
            flag: Feature flag name to check
            legacy_impl: Legacy implementation to fall back to
            validate: Whether to run dual validation
            comparison_strategy: Strategy for comparing results
            circuit_breaker_config: Circuit breaker configuration
            **comparison_kwargs: Additional kwargs for comparison (rtol, atol, etc.)
        """
        self.name = name
        self.flag = flag
        self.legacy_impl = legacy_impl
        self.validate = validate
        self.comparison_strategy = comparison_strategy
        self.comparison_kwargs = comparison_kwargs

        # Get or create circuit breaker
        self.breaker = CircuitBreakerRegistry.get_breaker(
            name, circuit_breaker_config
        )

        # Get or create dual runner if validation is enabled
        self.runner: DualRunner | None = None
        if validate and legacy_impl:
            self.runner = DualRunRegistry.get_runner(
                name, comparison_strategy, **comparison_kwargs
            )

        # Get dashboard for metrics
        self.dashboard = get_dashboard()

    def __call__(self, new_impl: Callable[..., T]) -> Callable[..., T]:
        """Decorate the new implementation.

        Args:
            new_impl: New implementation to migrate to

        Returns:
            Decorated function
        """
        @functools.wraps(new_impl)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            """Execute migration with feature flag, circuit breaker, and validation.

            Args:
                *args: Positional arguments for the implementation
                **kwargs: Keyword arguments for the implementation

            Returns:
                Result from new implementation or legacy fallback
            """
            import time

            start_time = time.perf_counter()

            try:
                # Check feature flag
                flag_manager = get_feature_flag_manager()
                flag_enabled = flag_manager.is_enabled(self.flag)

                # If flag is disabled, use legacy implementation
                if not flag_enabled:
                    if self.legacy_impl is None:
                        raise RuntimeError(
                            f"Migration '{self.name}' flag '{self.flag}' is disabled "
                            f"but no legacy implementation provided"
                        )

                    logger.debug(
                        f"Migration '{self.name}' - flag disabled, using legacy"
                    )
                    result = self.legacy_impl(*args, **kwargs)
                    duration = time.perf_counter() - start_time
                    self.dashboard.log_migration_event(
                        self.name, 'flag_disabled', duration
                    )
                    return result

                # Flag is enabled - check if we should validate
                if self.validate and self.runner and self.legacy_impl:
                    # Run dual validation
                    logger.debug(
                        f"Migration '{self.name}' - running dual validation"
                    )
                    result = self.runner.run(
                        new_impl, self.legacy_impl, args, kwargs
                    )
                    duration = time.perf_counter() - start_time
                    self.dashboard.log_migration_event(
                        self.name, 'validated', duration
                    )
                    return result

                # No validation - use circuit breaker for safety
                result = self.breaker.call(
                    new_impl, self.legacy_impl, *args, **kwargs
                )
                duration = time.perf_counter() - start_time

                # Log success or fallback
                if self.breaker.state.value == 'open':
                    self.dashboard.log_migration_event(
                        self.name, 'fallback', duration
                    )
                else:
                    self.dashboard.log_migration_event(
                        self.name, 'success', duration
                    )

                return result

            except Exception as e:
                duration = time.perf_counter() - start_time
                self.dashboard.log_migration_event(
                    self.name, 'failure', duration,
                    metadata={'error': str(e), 'error_type': type(e).__name__}
                )
                raise

        return wrapper


def with_migration(
    name: str,
    flag: str,
    legacy_impl: Callable[..., T] | None = None,
    validate: bool = False,
    **kwargs: Any
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator factory for migrating functions.

    Args:
        name: Unique name for this migration
        flag: Feature flag name to check
        legacy_impl: Legacy implementation to fall back to
        validate: Whether to run dual validation
        **kwargs: Additional arguments for MigrationDecorator

    Returns:
        Decorator function

    Example:
        @with_migration('jax_optimize', 'use_jax_integration', legacy_impl=scipy_optimize)
        def jax_optimize(params):
            return optimized_params
    """
    return MigrationDecorator(name, flag, legacy_impl, validate, **kwargs)


def migrate_with_validation(
    new_impl: Callable[..., T],
    legacy_impl: Callable[..., T],
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    name: str = 'migration',
    validate: bool = True,
    comparison_strategy: ComparisonStrategy = ComparisonStrategy.NUMERICAL_CLOSE,
    **comparison_kwargs: Any
) -> T:
    """Execute a migration with validation.

    This is a functional interface (non-decorator) for one-off migrations.

    Args:
        new_impl: New implementation
        legacy_impl: Legacy implementation
        args: Positional arguments
        kwargs: Keyword arguments
        name: Unique name for this migration
        validate: Whether to run validation
        comparison_strategy: Strategy for comparing results
        **comparison_kwargs: Additional comparison arguments

    Returns:
        Result from new implementation

    Example:
        result = migrate_with_validation(
            new_impl=jax_function,
            legacy_impl=scipy_function,
            args=(data,),
            name='optimization'
        )
    """
    kwargs = kwargs or {}

    if validate:
        runner = DualRunRegistry.get_runner(
            name, comparison_strategy, **comparison_kwargs
        )
        return runner.run(new_impl, legacy_impl, args, kwargs)

    # No validation - use circuit breaker
    breaker = CircuitBreakerRegistry.get_breaker(name)
    return breaker.call(new_impl, legacy_impl, *args, **kwargs)


def create_gradual_rollout(
    name: str,
    flag: str,
    new_impl: Callable[..., T],
    legacy_impl: Callable[..., T],
    initial_percentage: int = 10,
    increment: int = 10,
    validation_runs: int = 100,
    success_threshold: float = 0.99
) -> Callable[..., T]:
    """Create a function with gradual percentage-based rollout.

    This creates a wrapped function that uses percentage-based rollout
    controlled by the feature flag system.

    Args:
        name: Unique name for this migration
        flag: Feature flag name
        new_impl: New implementation
        legacy_impl: Legacy implementation
        initial_percentage: Initial rollout percentage
        increment: Percentage increment for gradual rollout
        validation_runs: Number of runs to validate before increasing percentage
        success_threshold: Success rate threshold (0.0-1.0)

    Returns:
        Wrapped function with gradual rollout

    Example:
        optimized_func = create_gradual_rollout(
            'jax_opt', 'use_jax_integration',
            jax_optimize, scipy_optimize
        )
    """
    flag_manager = get_feature_flag_manager()

    # Set initial rollout percentage
    try:
        flag_manager.update_flag(flag, enabled=True, rollout_percentage=initial_percentage)
    except KeyError:
        logger.warning(f"Flag '{flag}' not found, cannot set rollout percentage")

    @functools.wraps(new_impl)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        """Execute gradual rollout based on feature flag percentage.

        Args:
            *args: Positional arguments for the implementation
            **kwargs: Keyword arguments for the implementation (special: _migration_user_id)

        Returns:
            Result from new implementation or legacy based on rollout percentage
        """
        # Check if user is in rollout
        user_id = kwargs.pop('_migration_user_id', None)

        if flag_manager.is_enabled(flag, user_id=user_id):
            # Use new implementation
            breaker = CircuitBreakerRegistry.get_breaker(name)
            return breaker.call(new_impl, legacy_impl, *args, **kwargs)

        # Use legacy implementation
        return legacy_impl(*args, **kwargs)

    return wrapper


def get_migration_health() -> dict[str, Any]:
    """Get health status of all migrations.

    Returns:
        Dictionary with migration health information including alerts
    """
    dashboard = get_dashboard()
    status = dashboard.get_status()
    alerts = dashboard.get_health_alerts()

    return {
        'overall_health': 'healthy' if len(alerts) == 0 else 'degraded',
        'overall_migration_percentage': status['overall_migration_percentage'],
        'alerts': alerts,
        'components': status['components'],
        'timestamp': status['timestamp'],
    }


def print_migration_summary() -> None:
    """Print a human-readable migration summary to stdout."""
    dashboard = get_dashboard()
    print(dashboard.get_summary())


def export_migration_report(output_file: str) -> None:
    """Export comprehensive migration report to file.

    Args:
        output_file: Path to output JSON file
    """
    dashboard = get_dashboard()
    dashboard.export_metrics(output_file)
    logger.info(f"Migration report exported to {output_file}")
