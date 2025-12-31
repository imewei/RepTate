"""Example usage of RepTate strangler fig infrastructure.

This example demonstrates how to use the strangler fig pattern for
safe migration from legacy scipy implementations to JAX implementations.

Run this example:
    python examples/strangler_fig_example.py
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from jax import grad, jit

# Import strangler fig infrastructure
from RepTate.core.strangler_fig import (
    with_migration,
    migrate_with_validation,
    get_migration_health,
    print_migration_summary,
)
from RepTate.core.feature_flags_enhanced import get_feature_flag_manager
from RepTate.core.circuit_breaker import CircuitBreakerConfig
from RepTate.core.dual_run import ComparisonStrategy


# Example 1: Simple migration with decorator
# ===========================================

def legacy_optimization(params: np.ndarray) -> np.ndarray:
    """Legacy scipy-based optimization."""
    # Simulating scipy optimization
    return params * 0.9  # Simple update


@with_migration(
    name='simple_optimization',
    flag='use_jax_integration',
    legacy_impl=legacy_optimization
)
def jax_optimization(params: jnp.ndarray) -> jnp.ndarray:
    """New JAX-based optimization."""
    # JAX implementation with automatic differentiation
    return params * 0.9


# Example 2: Migration with validation
# =====================================

def legacy_interpolation(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """Legacy scipy interpolation."""
    # Simulating scipy.interpolate
    return np.interp(x_new, x, y)


@with_migration(
    name='interpolation_migration',
    flag='use_jax_integration',
    legacy_impl=legacy_interpolation,
    validate=True,  # Enable dual-run validation
    comparison_strategy=ComparisonStrategy.NUMERICAL_CLOSE,
    rtol=1e-5,
    atol=1e-8
)
def jax_interpolation(x: jnp.ndarray, y: jnp.ndarray, x_new: jnp.ndarray) -> jnp.ndarray:
    """New JAX-based interpolation."""
    # Using interpax or custom JAX implementation
    return jnp.interp(x_new, x, y)


# Example 3: Migration with custom circuit breaker
# =================================================

def legacy_nonlinear_solve(func, x0: np.ndarray) -> np.ndarray:
    """Legacy scipy nonlinear solver."""
    # Simulating scipy.optimize.fsolve
    # Simple Newton iteration
    x = x0.copy()
    for _ in range(10):
        f = func(x)
        if np.max(np.abs(f)) < 1e-6:
            break
        # Approximate Jacobian
        dx = 1e-8
        J = np.eye(len(x))
        x = x - 0.1 * f  # Simple update
    return x


@with_migration(
    name='nonlinear_solve',
    flag='use_jax_integration',
    legacy_impl=legacy_nonlinear_solve,
    circuit_breaker_config=CircuitBreakerConfig(
        failure_threshold=3,
        timeout=60.0,
        failure_rate_threshold=0.3
    )
)
def jax_nonlinear_solve(func, x0: jnp.ndarray) -> jnp.ndarray:
    """New JAX-based nonlinear solver using optimistix."""
    # Using optimistix for nonlinear solving
    x = x0
    for _ in range(10):
        f = func(x)
        if jnp.max(jnp.abs(f)) < 1e-6:
            break
        x = x - 0.1 * f
    return x


# Example 4: Functional API (non-decorator)
# ==========================================

def example_functional_migration():
    """Example using functional API instead of decorators."""

    def legacy_matrix_operation(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Legacy numpy matrix operation."""
        return np.linalg.solve(A, b)

    def jax_matrix_operation(A: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """JAX matrix operation."""
        return jnp.linalg.solve(A, b)

    # Create test data
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])

    # Run with validation
    result = migrate_with_validation(
        new_impl=jax_matrix_operation,
        legacy_impl=legacy_matrix_operation,
        args=(A, b),
        name='matrix_solve',
        validate=True
    )

    print(f"Matrix solve result: {result}")
    return result


# Example 5: Gradual rollout
# ===========================

def example_gradual_rollout():
    """Example of gradual percentage-based rollout."""
    flag_manager = get_feature_flag_manager()

    # Start with 25% rollout
    flag_manager.update_flag(
        'use_jax_integration',
        enabled=True,
        rollout_percentage=25
    )

    print("Starting with 25% rollout...")

    # Simulate multiple users
    for i in range(20):
        user_id = f"user_{i}"
        params = np.array([1.0, 2.0, 3.0])

        # The flag manager will use consistent hashing to determine
        # if this user gets the new implementation
        result = jax_optimization(params)

        enabled = flag_manager.is_enabled('use_jax_integration', user_id=user_id)
        print(f"User {user_id}: {'NEW' if enabled else 'LEGACY'} implementation")

    # After validation, increase rollout
    flag_manager.update_flag(
        'use_jax_integration',
        rollout_percentage=50
    )
    print("\nIncreased rollout to 50%")


# Example 6: Monitoring and observability
# ========================================

def example_monitoring():
    """Example of monitoring migration progress."""

    # Run some operations
    for i in range(10):
        params = np.array([float(i), float(i+1), float(i+2)])
        jax_optimization(params)

    # Get migration health
    health = get_migration_health()
    print("\nMigration Health:")
    print(f"  Overall health: {health['overall_health']}")
    print(f"  Migration progress: {health['overall_migration_percentage']:.1f}%")

    if health['alerts']:
        print(f"  Alerts: {len(health['alerts'])}")
        for alert in health['alerts']:
            print(f"    - [{alert['severity']}] {alert['message']}")

    # Print comprehensive summary
    print("\n" + "="*70)
    print_migration_summary()


# Example 7: Testing both implementations
# ========================================

def example_dual_validation():
    """Example of running both implementations and comparing."""

    # Test data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    x_new = np.linspace(0, 10, 50)

    # Run with validation (will execute both and compare)
    result = jax_interpolation(x, y, x_new)

    print(f"\nInterpolation result shape: {result.shape}")
    print(f"Result matches legacy: checked automatically")

    # The dual runner automatically logs any divergence
    from RepTate.core.dual_run import DualRunRegistry
    runner_metrics = DualRunRegistry.get_all_metrics()

    if 'interpolation_migration' in runner_metrics:
        metrics = runner_metrics['interpolation_migration']
        print(f"Match rate: {metrics['match_rate']:.1%}")
        print(f"Average speedup: {metrics['avg_speedup']:.2f}x")


# Example 8: Handling errors with circuit breaker
# ================================================

def example_circuit_breaker():
    """Example showing circuit breaker in action."""

    def legacy_func(x):
        return x * 2

    # Create a function that fails sometimes
    call_count = [0]

    @with_migration(
        name='flaky_migration',
        flag='use_jax_integration',
        legacy_impl=legacy_func,
        circuit_breaker_config=CircuitBreakerConfig(failure_threshold=3)
    )
    def flaky_jax_func(x):
        call_count[0] += 1
        if call_count[0] <= 5:
            raise RuntimeError("Simulated failure")
        return x * 2

    # First few calls will fail and eventually open the circuit
    for i in range(10):
        try:
            result = flaky_jax_func(i)
            print(f"Call {i}: result = {result}")
        except RuntimeError as e:
            print(f"Call {i}: error = {e}")

    # Check circuit breaker state
    from RepTate.core.circuit_breaker import CircuitBreakerRegistry
    breaker = CircuitBreakerRegistry.get_breaker('flaky_migration')
    print(f"\nCircuit breaker state: {breaker.state.value}")

    # Once circuit is open, it will use fallback automatically
    result = flaky_jax_func(100)
    print(f"With open circuit: result = {result} (from fallback)")


# Example 9: Environment-based control
# =====================================

def example_environment_control():
    """Example of using environment variables to control flags."""
    import os

    # You can set environment variables to override flags:
    # REPTATE_FEATURE_USE_JAX_INTEGRATION=false python examples/strangler_fig_example.py

    flag_manager = get_feature_flag_manager()

    print("\nCurrent flag states:")
    for flag_name, flag_info in flag_manager.get_all_flags().items():
        env_var = f"REPTATE_FEATURE_{flag_name.upper()}"
        env_value = os.environ.get(env_var, 'not set')
        print(f"  {flag_name}: {flag_info} (env: {env_value})")


# Main execution
# ==============

def main():
    """Run all examples."""
    print("RepTate Strangler Fig Examples")
    print("=" * 70)

    print("\n1. Simple Migration")
    print("-" * 70)
    params = np.array([1.0, 2.0, 3.0])
    result = jax_optimization(params)
    print(f"Optimization result: {result}")

    print("\n2. Migration with Validation")
    print("-" * 70)
    example_dual_validation()

    print("\n3. Functional API")
    print("-" * 70)
    example_functional_migration()

    print("\n4. Gradual Rollout")
    print("-" * 70)
    example_gradual_rollout()

    print("\n5. Circuit Breaker")
    print("-" * 70)
    example_circuit_breaker()

    print("\n6. Environment Control")
    print("-" * 70)
    example_environment_control()

    print("\n7. Monitoring and Observability")
    print("-" * 70)
    example_monitoring()


if __name__ == '__main__':
    main()
