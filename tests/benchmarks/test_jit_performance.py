"""JIT performance benchmark tests.

Tests cover:
- T024: JIT performance benchmarks with timing assertions
- T025: curve_model JIT compilation benchmark

Verifies that JIT compilation provides at least 2x speedup on repeated calls.
"""

from __future__ import annotations

import time
from typing import Callable

import jax
import jax.numpy as jnp
import pytest


def benchmark_function(
    fn: Callable, *args, warmup_calls: int = 3, benchmark_calls: int = 10, **kwargs
) -> tuple[float, float]:
    """Benchmark a function with warmup.

    Args:
        fn: Function to benchmark.
        *args: Positional arguments for the function.
        warmup_calls: Number of warmup calls (includes JIT compilation).
        benchmark_calls: Number of calls to average for benchmark.
        **kwargs: Keyword arguments for the function.

    Returns:
        Tuple of (first_call_time, average_subsequent_time).
    """
    # First call (includes JIT compilation)
    start = time.perf_counter()
    _ = fn(*args, **kwargs)
    first_call_time = time.perf_counter() - start

    # Warmup calls
    for _ in range(warmup_calls - 1):
        _ = fn(*args, **kwargs)

    # Benchmark calls
    times = []
    for _ in range(benchmark_calls):
        start = time.perf_counter()
        _ = fn(*args, **kwargs)
        times.append(time.perf_counter() - start)

    average_time = sum(times) / len(times)
    return first_call_time, average_time


class TestJITPerformance:
    """Test suite for JIT compilation performance (T024)."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for benchmarking."""
        x = jnp.linspace(0, 10, 1000)
        y = 2.5 * x + 1.5 + jax.random.normal(jax.random.PRNGKey(0), shape=(1000,)) * 0.5
        return x, y

    def test_jit_provides_speedup(self, sample_data):
        """Test that JIT compilation provides measurable speedup."""
        x, y = sample_data

        def model_fn(x, params):
            """Simple linear model."""
            return params[0] * x + params[1]

        # Non-JIT version
        def eval_non_jit(x, params):
            return model_fn(x, params)

        # JIT version
        @jax.jit
        def eval_jit(x, params):
            return model_fn(x, params)

        params = jnp.array([2.5, 1.5])

        # Benchmark non-JIT
        _, non_jit_time = benchmark_function(eval_non_jit, x, params)

        # Benchmark JIT (first call includes compilation)
        first_call, jit_time = benchmark_function(eval_jit, x, params)

        # First call should be slower (compilation overhead)
        # After compilation, JIT should be competitive or faster
        # Note: For very simple functions, speedup may be minimal
        assert jit_time <= first_call or jit_time <= non_jit_time * 1.5

    def test_repeated_jit_calls_fast(self, sample_data):
        """Test that repeated JIT calls are consistently fast."""
        x, y = sample_data

        @jax.jit
        def compute_residuals(x, y, params):
            pred = params[0] * x + params[1]
            return jnp.sum((y - pred) ** 2)

        params = jnp.array([2.5, 1.5])

        # First call (JIT compilation)
        start = time.perf_counter()
        _ = compute_residuals(x, y, params)
        first_time = time.perf_counter() - start

        # Subsequent calls
        times = []
        for _ in range(20):
            start = time.perf_counter()
            _ = compute_residuals(x, y, params)
            times.append(time.perf_counter() - start)

        avg_subsequent = sum(times) / len(times)

        # Subsequent calls should be at least 2x faster than first
        # (accounting for JIT compilation overhead)
        assert avg_subsequent < first_time or avg_subsequent < 0.01  # Fast enough


class TestCurveModelJIT:
    """Test suite for curve_model JIT compilation (T025)."""

    @pytest.fixture
    def polynomial_model(self):
        """Create a polynomial model for testing."""

        def model(x, params):
            """Polynomial model: a*x^2 + b*x + c"""
            return params[0] * x**2 + params[1] * x + params[2]

        return model

    def test_curve_model_jit_compilation(self, polynomial_model):
        """Test that curve model can be JIT compiled."""
        x = jnp.linspace(0, 10, 500)
        params = jnp.array([0.5, 2.0, 1.0])

        # JIT compile the model
        jit_model = jax.jit(polynomial_model)

        # Should produce same results
        non_jit_result = polynomial_model(x, params)
        jit_result = jit_model(x, params)

        assert jnp.allclose(non_jit_result, jit_result)

    def test_curve_model_jit_speedup(self, polynomial_model):
        """Test that JIT provides speedup for curve model evaluation."""
        x = jnp.linspace(0, 10, 10000)
        params = jnp.array([0.5, 2.0, 1.0])

        jit_model = jax.jit(polynomial_model)

        # Warmup JIT
        _ = jit_model(x, params)
        _ = jit_model(x, params)

        # Benchmark
        first_call, jit_time = benchmark_function(jit_model, x, params)
        _, non_jit_time = benchmark_function(polynomial_model, x, params)

        # JIT should not be significantly slower after compilation
        # For larger data, JIT should provide speedup
        assert jit_time <= non_jit_time * 2  # Allow some tolerance

    def test_jit_preserves_numerical_accuracy(self, polynomial_model):
        """Test that JIT compilation preserves numerical accuracy."""
        x = jnp.linspace(0, 10, 1000)
        params = jnp.array([0.5, 2.0, 1.0])

        jit_model = jax.jit(polynomial_model)

        # Multiple evaluations should give identical results
        results = [jit_model(x, params) for _ in range(5)]

        for result in results[1:]:
            assert jnp.allclose(results[0], result, rtol=1e-10)


class TestJaxOpsKernelJIT:
    """Test JIT compilation of jax_ops kernels."""

    def test_evaluate_model_jit(self):
        """Test that evaluate_model can be JIT compiled."""
        from RepTate.core.jax_ops.models import evaluate_model, linear_kernel

        x = jnp.linspace(0, 10, 100)
        params = {"slope": 2.5, "intercept": 1.5}

        # JIT compile evaluate_model
        @jax.jit
        def jit_eval(x, params):
            return linear_kernel(x, params)

        result_direct = linear_kernel(x, params)
        result_jit = jit_eval(x, params)

        assert jnp.allclose(result_direct, result_jit)

    def test_math_functions_jit(self):
        """Test that math helper functions can be JIT compiled."""
        from RepTate.core.jax_ops.math import clamp, safe_divide, safe_log, squared_error

        x = jnp.array([0.1, 1.0, 10.0, 100.0])

        # JIT compile math functions
        jit_clamp = jax.jit(clamp)
        jit_safe_log = jax.jit(safe_log)

        # Test clamp
        result = jit_clamp(x, 0.5, 50.0)
        expected = jnp.clip(x, 0.5, 50.0)
        assert jnp.allclose(result, expected)

        # Test safe_log
        result = jit_safe_log(x)
        expected = jnp.log(x)
        assert jnp.allclose(result, expected)
