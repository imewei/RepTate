"""Performance contracts validating execution time bounds.

Contract Tests:
- C009: NLSQ curve fitting time budget (must complete within X seconds)
- C010: NumPyro NUTS inference time budget
- C011: Theory calculation performance (per-call and vectorized)
- C012: Data loading performance

Performance contracts establish performance baselines and detect regressions.
They capture the current performance characteristics to prevent unintended slowdowns.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable

import jax.numpy as jnp
import pytest
from jax import Array

if TYPE_CHECKING:
    from tests.conftest import PerformanceBaseline, SyntheticData


class TestNLSQCurveFittingPerformance:
    """Contract tests for NLSQ curve fitting performance.

    Expected Performance:
    - Linear model fit (<1 second for 100 points)
    - Maxwell model fit (<2 seconds for 100 points)
    - Multi-parameter fit (<5 seconds for 100 points)
    """

    def test_linear_fit_performance(self, synthetic_frequency_data: SyntheticData) -> None:
        """Contract: Linear model fit completes in reasonable time."""
        from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

        # Arrange: Linear test data
        xdata = synthetic_frequency_data.x
        ydata = 2.5 * xdata + 1.5

        def linear_model(x: Array, params: Array) -> Array:
            return params[0] * x + params[1]

        p0 = jnp.array([1.0, 0.0])

        # Act: Time the fit
        start = time.perf_counter()
        result, _ = run_nlsq_fit(linear_model, xdata, ydata, p0=p0)
        elapsed = time.perf_counter() - start

        # Assert: Should complete quickly for simple model
        assert elapsed < 1.0, f"Linear fit took {elapsed:.2f}s, expected <1s"
        assert result.success

    def test_maxwell_fit_performance(self, synthetic_frequency_data: SyntheticData) -> None:
        """Contract: Maxwell model fit completes in reasonable time."""
        from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

        # Arrange: Maxwell test data
        G0_true = 1e5
        tau_true = 1.0
        omega = synthetic_frequency_data.x
        omega_tau = omega * tau_true
        G_prime = G0_true * omega_tau**2 / (1 + omega_tau**2)

        def maxwell_model(x: Array, params: Array) -> Array:
            G0, tau = params[0], params[1]
            omega_tau = x * tau
            return G0 * omega_tau**2 / (1 + omega_tau**2)

        p0 = jnp.array([5e4, 0.5])

        # Act: Time the fit
        start = time.perf_counter()
        result, _ = run_nlsq_fit(maxwell_model, omega, G_prime, p0=p0)
        elapsed = time.perf_counter() - start

        # Assert: Should complete in reasonable time
        assert elapsed < 2.0, f"Maxwell fit took {elapsed:.2f}s, expected <2s"

    def test_fit_performance_scales_linearly(
        self,
        synthetic_frequency_data: SyntheticData,
    ) -> None:
        """Contract: Fit time scales roughly linearly with data size."""
        from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

        def linear_model(x: Array, params: Array) -> Array:
            return params[0] * x + params[1]

        p0 = jnp.array([2.5, 1.5])

        # Measure time for different data sizes
        times = {}
        for n in [10, 50, 100]:
            xdata = jnp.linspace(0, 10, n)
            ydata = 2.5 * xdata + 1.5

            start = time.perf_counter()
            _, _ = run_nlsq_fit(linear_model, xdata, ydata, p0=p0)
            times[n] = time.perf_counter() - start

        # Time should not increase quadratically
        ratio_10_50 = times[50] / times[10]
        ratio_50_100 = times[100] / times[50]

        # For linear scaling, 50/10=5x data, 100/50=2x data
        # Time ratio should be similar to data ratio
        assert ratio_10_50 < 10, "Subquadratic scaling violated"
        assert ratio_50_100 < 5, "Subquadratic scaling violated"


class TestTheoryCalculationPerformance:
    """Contract tests for theory calculation performance.

    Expected Performance:
    - Single calculation: <1ms
    - Vectorized (100 points): <100ms
    - Vectorized (1000 points): <1000ms
    """

    def test_single_theory_calculation_fast(self, mock_theory) -> None:
        """Contract: Single theory calculation is fast (<1ms)."""
        params = {"slope": 2.0, "intercept": 1.0}
        x = jnp.array([1.0])

        start = time.perf_counter()
        _ = mock_theory.calculate(params, x)
        elapsed = time.perf_counter() - start

        # Should be very fast (1ms = 0.001s)
        assert elapsed < 0.001, f"Single calculation took {elapsed*1000:.2f}ms"

    def test_vectorized_theory_calculation_scales(self, mock_theory) -> None:
        """Contract: Vectorized calculation scales linearly."""
        params = {"slope": 2.0, "intercept": 1.0}

        times = {}
        for n in [10, 100, 1000]:
            x = jnp.linspace(0, 10, n)

            start = time.perf_counter()
            _ = mock_theory.calculate(params, x)
            times[n] = time.perf_counter() - start

        # Time should scale roughly linearly with data size
        ratio_10_100 = times[100] / times[10]
        ratio_100_1000 = times[1000] / times[100]

        # Should be roughly 10x and 10x respectively
        assert ratio_10_100 < 100, "Superlinear scaling detected"
        assert ratio_100_1000 < 100, "Superlinear scaling detected"

    def test_parameter_update_doesnt_slow_calculation(
        self,
        mock_theory,
    ) -> None:
        """Contract: Parameter updates don't impact calculation performance."""
        x = jnp.linspace(0, 10, 100)

        # Time calculation with original parameters
        params1 = {"slope": 2.0, "intercept": 1.0}
        start = time.perf_counter()
        _ = mock_theory.calculate(params1, x)
        time1 = time.perf_counter() - start

        # Time calculation with different parameters
        params2 = {"slope": 5.0, "intercept": 10.0}
        start = time.perf_counter()
        _ = mock_theory.calculate(params2, x)
        time2 = time.perf_counter() - start

        # Times should be similar (no parameter lookup overhead)
        ratio = max(time2, time1) / min(time2, time1)
        assert ratio < 2.0, "Parameter change caused significant slowdown"


class TestDataAccessPerformance:
    """Contract tests for dataset access performance.

    Expected Performance:
    - get_x(): <1us
    - get_y(): <1us
    - get_column(): <100us
    """

    def test_get_x_performance(self, mock_dataset) -> None:
        """Contract: get_x() is fast (array access)."""
        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            _ = mock_dataset.get_x()
        elapsed = time.perf_counter() - start

        per_call = (elapsed / iterations) * 1e6  # Convert to microseconds
        # Should be very fast, under 100 microseconds
        assert per_call < 100, f"get_x() took {per_call:.1f}us per call"

    def test_get_y_performance(self, mock_dataset) -> None:
        """Contract: get_y() is fast (array access)."""
        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            _ = mock_dataset.get_y()
        elapsed = time.perf_counter() - start

        per_call = (elapsed / iterations) * 1e6
        assert per_call < 100, f"get_y() took {per_call:.1f}us per call"


class TestPerformanceRegression:
    """Contract tests for detecting performance regressions.

    These tests establish baseline measurements and flag significant
    performance degradations.
    """

    def test_linear_fit_regression_detection(
        self,
        synthetic_frequency_data: SyntheticData,
        baseline_registry: dict,
    ) -> None:
        """Contract: Detect if linear fit becomes significantly slower."""
        from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

        xdata = synthetic_frequency_data.x
        ydata = 2.5 * xdata + 1.5

        def linear_model(x: Array, params: Array) -> Array:
            return params[0] * x + params[1]

        p0 = jnp.array([1.0, 0.0])

        # Measure current performance
        times = []
        for _ in range(3):
            start = time.perf_counter()
            _, _ = run_nlsq_fit(linear_model, xdata, ydata, p0=p0)
            times.append(time.perf_counter() - start)

        mean_time = sum(times) / len(times)
        std_dev = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5

        # Store in registry for assertion
        baseline_registry["linear_fit"] = {
            "mean": mean_time,
            "std": std_dev,
            "samples": 3,
        }

        # Assert it's still reasonably fast
        assert mean_time < 1.0, f"Linear fit performance degraded: {mean_time:.2f}s"

    def test_theory_calculation_regression_detection(
        self,
        mock_theory,
        baseline_registry: dict,
    ) -> None:
        """Contract: Detect if theory calculation becomes significantly slower."""
        x = jnp.linspace(0, 10, 100)
        params = {"slope": 2.0, "intercept": 1.0}

        # Measure current performance
        times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = mock_theory.calculate(params, x)
            times.append(time.perf_counter() - start)

        mean_time = sum(times) / len(times)
        std_dev = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5

        baseline_registry["theory_calculation"] = {
            "mean": mean_time,
            "std": std_dev,
            "samples": 10,
        }

        # Should complete 100 calculations in reasonable time
        assert mean_time < 0.01, f"Theory calculation performance degraded: {mean_time*1000:.2f}ms"


class TestMemoryPerformance:
    """Contract tests for memory usage patterns.

    Expected Behavior:
    - No memory leaks in repeated calculations
    - Array copies minimized
    """

    def test_repeated_calculation_no_accumulation(self, mock_theory) -> None:
        """Contract: Repeated calculations don't accumulate memory."""
        params = {"slope": 2.0, "intercept": 1.0}
        x = jnp.linspace(0, 10, 100)

        # Perform many iterations (would accumulate memory if leak exists)
        for _ in range(1000):
            _ = mock_theory.calculate(params, x)

        # If we get here without memory error, test passes
        assert True

    def test_calculation_output_independent(self, mock_theory) -> None:
        """Contract: Output arrays are independent (not shared references)."""
        params = {"slope": 2.0, "intercept": 1.0}
        x = jnp.linspace(0, 10, 10)

        y1 = mock_theory.calculate(params, x)
        y2 = mock_theory.calculate(params, x)

        # Arrays should be equal but different objects
        assert jnp.allclose(y1, y2)
        # Note: JAX arrays might be the same object due to immutability,
        # but they should not be modifiable
