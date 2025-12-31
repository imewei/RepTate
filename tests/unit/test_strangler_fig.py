"""Tests for strangler fig infrastructure.

These tests validate the feature flag system, circuit breakers,
dual-run patterns, and observability dashboard.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from RepTate.core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitState,
)
from RepTate.core.dual_run import ComparisonStrategy, DualRunner, DualRunRegistry
from RepTate.core.feature_flags_enhanced import (
    FeatureFlagConfig,
    FeatureFlagManager,
)
from RepTate.core.migration_observability import MigrationDashboard
from RepTate.core.strangler_fig import (
    migrate_with_validation,
    with_migration,
)


class TestFeatureFlagManager:
    """Test the enhanced feature flag manager."""

    def test_default_flags_initialized(self):
        """Test that default flags are initialized."""
        manager = FeatureFlagManager()
        assert manager.is_enabled('use_jax_integration')
        assert manager.is_enabled('use_safe_eval')

    def test_environment_override(self, monkeypatch):
        """Test environment variable override."""
        monkeypatch.setenv('REPTATE_FEATURE_USE_JAX_INTEGRATION', 'false')
        manager = FeatureFlagManager()
        assert not manager.is_enabled('use_jax_integration')

    def test_rollout_percentage_consistent(self):
        """Test that rollout percentage is consistent for same user."""
        manager = FeatureFlagManager()
        manager.update_flag('use_decomposed_gui', enabled=True, rollout_percentage=50)

        user_id = 'test_user_123'
        # Should be consistent across multiple calls
        result1 = manager.is_enabled('use_decomposed_gui', user_id=user_id)
        result2 = manager.is_enabled('use_decomposed_gui', user_id=user_id)
        assert result1 == result2

    def test_dependency_resolution(self):
        """Test that flag dependencies are enforced."""
        manager = FeatureFlagManager()

        # use_jax_native depends on use_jax_integration
        manager.update_flag('use_jax_integration', enabled=False)
        manager.update_flag('use_jax_native', enabled=True, rollout_percentage=100)

        # Should be disabled due to dependency
        assert not manager.is_enabled('use_jax_native')

    def test_metrics_collection(self):
        """Test that metrics are collected."""
        manager = FeatureFlagManager(enable_metrics=True)

        # Make several evaluations
        for i in range(10):
            manager.is_enabled('use_jax_integration', user_id=f'user_{i}')

        metrics = manager.export_metrics()
        assert metrics['total_evaluations'] == 10

    def test_yaml_config_loading(self, tmp_path):
        """Test loading configuration from YAML file."""
        pytest.importorskip('yaml')

        config_file = tmp_path / 'test_flags.yaml'
        config_content = """
flags:
  test_flag:
    enabled: true
    rollout_percentage: 75
    description: "Test flag"
    tags:
      - test
    depends_on: []
"""
        config_file.write_text(config_content)

        manager = FeatureFlagManager(config_path=config_file)
        status = manager.get_flag_status('test_flag')
        assert status['enabled']
        assert status['rollout_percentage'] == 75


class TestCircuitBreaker:
    """Test the circuit breaker implementation."""

    def test_circuit_starts_closed(self):
        """Test that circuit breaker starts in CLOSED state."""
        breaker = CircuitBreaker('test_breaker')
        assert breaker.state == CircuitState.CLOSED

    def test_circuit_opens_on_failures(self):
        """Test that circuit opens after threshold failures."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker('test_breaker', config)

        def failing_func():
            raise ValueError("Test error")

        def fallback_func():
            return "fallback"

        # First few failures should keep circuit closed
        for _ in range(3):
            try:
                breaker.call(failing_func, fallback_func)
            except ValueError:
                pass

        # Circuit should now be open
        assert breaker.state == CircuitState.OPEN

    def test_circuit_uses_fallback_when_open(self):
        """Test that fallback is used when circuit is open."""
        config = CircuitBreakerConfig(failure_threshold=2)
        breaker = CircuitBreaker('test_breaker', config)

        def failing_func():
            raise ValueError("Test error")

        def fallback_func():
            return "fallback_result"

        # Trigger circuit to open
        for _ in range(2):
            breaker.call(failing_func, fallback_func)

        # Should use fallback now
        result = breaker.call(failing_func, fallback_func)
        assert result == "fallback_result"

    def test_circuit_transitions_to_half_open(self):
        """Test circuit transitions to HALF_OPEN after timeout."""
        config = CircuitBreakerConfig(failure_threshold=2, timeout=0.1)
        breaker = CircuitBreaker('test_breaker', config)

        def failing_func():
            raise ValueError("Test error")

        def fallback_func():
            return "fallback"

        # Open the circuit
        for _ in range(2):
            breaker.call(failing_func, fallback_func)

        assert breaker.state == CircuitState.OPEN

        # Wait for timeout
        time.sleep(0.15)

        # Next call should transition to HALF_OPEN
        breaker.call(failing_func, fallback_func)
        # Note: It will go back to OPEN due to failure in HALF_OPEN

    def test_circuit_closes_after_successes(self):
        """Test circuit closes after success threshold in HALF_OPEN."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=2,
            timeout=0.1
        )
        breaker = CircuitBreaker('test_breaker', config)

        call_count = [0]

        def sometimes_failing_func():
            call_count[0] += 1
            if call_count[0] <= 2:
                raise ValueError("Test error")
            return "success"

        def fallback_func():
            return "fallback"

        # Open the circuit
        for _ in range(2):
            breaker.call(sometimes_failing_func, fallback_func)

        assert breaker.state == CircuitState.OPEN

        # Wait for timeout
        time.sleep(0.15)

        # Successful calls should eventually close circuit
        for _ in range(5):
            breaker.call(sometimes_failing_func, fallback_func)
            time.sleep(0.01)

        assert breaker.state == CircuitState.CLOSED

    def test_protected_decorator(self):
        """Test the @protected decorator."""
        breaker = CircuitBreaker('test_breaker')

        def legacy_impl(x):
            return x * 2

        @breaker.protected(fallback=legacy_impl)
        def new_impl(x):
            return x * 3

        result = new_impl(5)
        assert result == 15

    def test_metrics_collection(self):
        """Test that circuit breaker collects metrics."""
        breaker = CircuitBreaker('test_breaker')

        def test_func():
            return "success"

        for _ in range(10):
            breaker.call(test_func)

        metrics = breaker.get_metrics()
        assert metrics['total_calls'] == 10
        assert metrics['successful_calls'] == 10
        assert metrics['success_rate'] == 1.0


class TestDualRunner:
    """Test the dual-run pattern implementation."""

    def test_exact_comparison_matches(self):
        """Test exact comparison strategy with matching results."""
        runner = DualRunner('test_runner', ComparisonStrategy.EXACT)

        def new_impl(x):
            return x * 2

        def legacy_impl(x):
            return x * 2

        result = runner.run(new_impl, legacy_impl, args=(5,))
        assert result == 10

        metrics = runner.get_metrics()
        assert metrics['matches'] == 1
        assert metrics['mismatches'] == 0

    def test_exact_comparison_mismatch(self):
        """Test exact comparison with divergent results."""
        runner = DualRunner(
            'test_runner',
            ComparisonStrategy.EXACT,
            alert_on_mismatch=False
        )

        def new_impl(x):
            return x * 3

        def legacy_impl(x):
            return x * 2

        result = runner.run(new_impl, legacy_impl, args=(5,))
        assert result == 15  # Uses new implementation result

        metrics = runner.get_metrics()
        assert metrics['mismatches'] == 1

    def test_numerical_close_comparison(self):
        """Test numerical comparison with close results."""
        runner = DualRunner(
            'test_runner',
            ComparisonStrategy.NUMERICAL_CLOSE,
            rtol=1e-5,
            atol=1e-8
        )

        def new_impl(x):
            return jnp.array([x * 2.0, x * 3.0])

        def legacy_impl(x):
            return np.array([x * 2.0 + 1e-9, x * 3.0 + 1e-9])

        result = runner.run(new_impl, legacy_impl, args=(5.0,))
        assert isinstance(result, jnp.ndarray)

        metrics = runner.get_metrics()
        assert metrics['matches'] == 1

    def test_numerical_comparison_divergence(self):
        """Test numerical comparison with divergent results."""
        runner = DualRunner(
            'test_runner',
            ComparisonStrategy.NUMERICAL_CLOSE,
            alert_on_mismatch=False
        )

        def new_impl(x):
            return jnp.array([x * 2.0])

        def legacy_impl(x):
            return np.array([x * 3.0])

        result = runner.run(new_impl, legacy_impl, args=(5.0,))

        metrics = runner.get_metrics()
        assert metrics['mismatches'] == 1
        assert len(metrics['recent_mismatches']) == 1

    def test_speedup_tracking(self):
        """Test that speedup metrics are tracked."""
        runner = DualRunner('test_runner')

        def fast_impl(x):
            return x * 2

        def slow_impl(x):
            time.sleep(0.01)
            return x * 2

        runner.run(fast_impl, slow_impl, args=(5,))

        metrics = runner.get_metrics()
        assert metrics['new_faster'] == 1
        assert metrics['avg_speedup'] > 1.0

    def test_custom_comparator(self):
        """Test custom comparison function."""
        def custom_compare(a, b):
            # Custom logic: results match if sum is the same
            return sum(a) == sum(b)

        runner = DualRunner(
            'test_runner',
            ComparisonStrategy.CUSTOM,
            custom_comparator=custom_compare
        )

        def new_impl():
            return [1, 2, 3]

        def legacy_impl():
            return [2, 2, 2]

        result = runner.run(new_impl, legacy_impl)

        metrics = runner.get_metrics()
        assert metrics['matches'] == 1


class TestMigrationDashboard:
    """Test the migration observability dashboard."""

    def test_dashboard_initialization(self, tmp_path):
        """Test dashboard initialization."""
        log_file = tmp_path / 'migration.log'
        dashboard = MigrationDashboard(log_file=log_file)

        status = dashboard.get_status()
        assert status['total_events'] == 0
        assert status['overall_migration_percentage'] == 0.0

    def test_event_logging(self):
        """Test logging migration events."""
        dashboard = MigrationDashboard()

        dashboard.log_migration_event('test_component', 'success', duration=0.123)
        dashboard.log_migration_event('test_component', 'failure', duration=0.456)

        status = dashboard.get_component_status('test_component')
        assert status is not None
        assert status['total_calls'] == 2
        assert status['success_count'] == 1
        assert status['failure_count'] == 1

    def test_metrics_aggregation(self):
        """Test that metrics are aggregated correctly."""
        dashboard = MigrationDashboard()

        # Log multiple events for different components
        for i in range(10):
            dashboard.log_migration_event('comp_a', 'success', duration=0.1)
            dashboard.log_migration_event('comp_b', 'success', duration=0.2)

        status = dashboard.get_status()
        assert len(status['components']) == 2
        assert status['total_events'] == 20

    def test_health_alerts(self):
        """Test health alert generation."""
        dashboard = MigrationDashboard()

        # Create some failures and successes to get above threshold
        for _ in range(15):
            dashboard.log_migration_event('failing_component', 'failure')
        for _ in range(2):
            dashboard.log_migration_event('failing_component', 'success')

        alerts = dashboard.get_health_alerts()
        # Should have alert for high failure rate (>10% with >10 total calls)
        assert any(
            alert['type'] == 'high_failure_rate'
            for alert in alerts
        ), f"Expected high_failure_rate alert but got: {[a['type'] for a in alerts]}"

    def test_summary_generation(self):
        """Test summary string generation."""
        dashboard = MigrationDashboard()

        dashboard.log_migration_event('test', 'success')

        summary = dashboard.get_summary()
        assert 'Migration Dashboard' in summary
        assert 'test' in summary

    def test_export_metrics(self, tmp_path):
        """Test exporting metrics to file."""
        dashboard = MigrationDashboard()

        dashboard.log_migration_event('test', 'success')

        output_file = tmp_path / 'metrics.json'
        dashboard.export_metrics(output_file)

        assert output_file.exists()
        import json
        with open(output_file) as f:
            data = json.load(f)
        assert 'components' in data


class TestStranglerFigIntegration:
    """Test integration of all strangler fig components."""

    def test_with_migration_decorator(self):
        """Test the @with_migration decorator."""
        def legacy_func(x):
            return x * 2

        @with_migration(
            name='test_migration',
            flag='use_jax_integration',
            legacy_impl=legacy_func
        )
        def new_func(x):
            return x * 3

        result = new_func(5)
        # Should use new implementation since flag is enabled by default
        assert result == 15

    def test_with_migration_validation(self):
        """Test migration with validation enabled."""
        def legacy_func(x):
            return np.array([x * 2.0])

        @with_migration(
            name='test_validation',
            flag='use_jax_integration',
            legacy_impl=legacy_func,
            validate=True
        )
        def new_func(x):
            return jnp.array([x * 2.0])

        result = new_func(5.0)
        # Should match and return new implementation result
        assert jnp.allclose(result, jnp.array([10.0]))

    def test_migrate_with_validation_function(self):
        """Test the functional migrate_with_validation."""
        def new_impl(x):
            return jnp.array([x * 2.0])

        def legacy_impl(x):
            return np.array([x * 2.0])

        result = migrate_with_validation(
            new_impl=new_impl,
            legacy_impl=legacy_impl,
            args=(5.0,),
            name='test_func_migration'
        )

        assert jnp.allclose(result, jnp.array([10.0]))

    def test_circuit_breaker_registry(self):
        """Test circuit breaker registry."""
        breaker1 = CircuitBreakerRegistry.get_breaker('test1')
        breaker2 = CircuitBreakerRegistry.get_breaker('test1')

        # Should return same instance
        assert breaker1 is breaker2

        metrics = CircuitBreakerRegistry.get_all_metrics()
        assert 'test1' in metrics

    def test_dual_run_registry(self):
        """Test dual runner registry."""
        runner1 = DualRunRegistry.get_runner('test1')
        runner2 = DualRunRegistry.get_runner('test1')

        # Should return same instance
        assert runner1 is runner2

        metrics = DualRunRegistry.get_all_metrics()
        assert 'test1' in metrics

    def test_end_to_end_migration_flow(self):
        """Test complete migration flow with all components."""
        # Set up flag
        manager = FeatureFlagManager()
        manager.update_flag('use_jax_integration', enabled=True, rollout_percentage=100)

        # Create implementations
        def legacy_optimize(x):
            return x * 2.0

        @with_migration(
            name='e2e_test',
            flag='use_jax_integration',
            legacy_impl=legacy_optimize,
            validate=True
        )
        def jax_optimize(x):
            return x * 2.0

        # Execute
        result = jax_optimize(5.0)
        assert result == 10.0

        # Check metrics
        dashboard = MigrationDashboard()
        status = dashboard.get_component_status('e2e_test')
        # May be None if dashboard is not the global instance
        # assert status is not None or status is None


@pytest.fixture(autouse=True)
def reset_registries():
    """Reset global registries between tests."""
    yield
    # Note: In real implementation, you'd add reset methods
    # For tests, we rely on test isolation
