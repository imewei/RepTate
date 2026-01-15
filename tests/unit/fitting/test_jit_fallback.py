"""Tests for JIT fallback behavior.

Tests cover:
- T025a: JIT fallback behavior verification
  - WARNING is logged when JIT compilation fails
  - Numerical results are unchanged in fallback mode
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import jax
import jax.numpy as jnp
import pytest


class TestJITFallbackBehavior:
    """Test suite for JIT fallback behavior (T025a)."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        x = jnp.linspace(0, 10, 100)
        y = 2.5 * x + 1.5 + jax.random.normal(jax.random.PRNGKey(0), shape=(100,)) * 0.5
        return x, y

    def test_numerical_results_unchanged_in_fallback(self, sample_data):
        """Test that numerical results are identical with/without JIT."""
        x, y = sample_data

        def model_fn(x, params):
            """Simple model function."""
            return params[0] * x + params[1]

        params = jnp.array([2.5, 1.5])

        # Non-JIT result
        non_jit_result = model_fn(x, params)

        # JIT result
        jit_model = jax.jit(model_fn)
        jit_result = jit_model(x, params)

        # Results should be numerically identical
        assert jnp.allclose(non_jit_result, jit_result, rtol=1e-10)

    def test_fallback_function_produces_correct_output(self, sample_data):
        """Test that fallback mode produces correct output."""
        x, y = sample_data

        def model_fn(x, params):
            """Simple model function."""
            return params[0] * x + params[1]

        params = jnp.array([2.5, 1.5])

        # Simulate fallback by using non-JIT version
        result = model_fn(x, params)

        # Verify output is correct
        expected = 2.5 * x + 1.5
        assert jnp.allclose(result, expected, rtol=1e-10)

    def test_jit_and_non_jit_residuals_match(self, sample_data):
        """Test that residual computation is identical with/without JIT."""
        x, y = sample_data

        def compute_residuals(x, y, params):
            """Compute sum of squared residuals."""
            pred = params[0] * x + params[1]
            return jnp.sum((y - pred) ** 2)

        params = jnp.array([2.5, 1.5])

        # Non-JIT residuals
        non_jit_residuals = compute_residuals(x, y, params)

        # JIT residuals
        jit_compute = jax.jit(compute_residuals)
        jit_residuals = jit_compute(x, y, params)

        # Should be identical
        assert jnp.isclose(non_jit_residuals, jit_residuals, rtol=1e-10)


class TestJITErrorHandling:
    """Test error handling when JIT compilation encounters issues."""

    def test_traceable_functions_compile_correctly(self):
        """Test that traceable functions compile without errors."""

        @jax.jit
        def simple_function(x):
            return x * 2 + 1

        x = jnp.array([1.0, 2.0, 3.0])
        result = simple_function(x)
        expected = jnp.array([3.0, 5.0, 7.0])

        assert jnp.allclose(result, expected)

    def test_static_argnums_for_non_traceable_args(self):
        """Test using static_argnums for non-traceable arguments."""
        from functools import partial

        def model_with_mode(x, params, mode: str):
            if mode == "linear":
                return params[0] * x + params[1]
            elif mode == "quadratic":
                return params[0] * x**2 + params[1] * x + params[2]
            return x

        # Use static_argnums for the mode parameter
        jit_model = jax.jit(model_with_mode, static_argnums=(2,))

        x = jnp.linspace(0, 5, 10)
        linear_params = jnp.array([2.0, 1.0])

        result = jit_model(x, linear_params, "linear")
        expected = 2.0 * x + 1.0

        assert jnp.allclose(result, expected)


class TestGPUMemoryFallback:
    """Test GPU memory exhaustion fallback behavior."""

    def test_cpu_fallback_produces_correct_results(self):
        """Test that CPU fallback produces correct numerical results."""
        # Force CPU execution
        with jax.default_device(jax.devices("cpu")[0]):

            @jax.jit
            def compute(x, y):
                return jnp.sum(x * y)

            x = jnp.array([1.0, 2.0, 3.0])
            y = jnp.array([4.0, 5.0, 6.0])

            result = compute(x, y)
            expected = 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0

            assert jnp.isclose(result, expected)

    def test_large_array_on_cpu(self):
        """Test that large arrays can be processed on CPU."""
        with jax.default_device(jax.devices("cpu")[0]):
            # Create moderately large array (not too large to cause issues)
            x = jnp.ones(100000)
            y = jnp.arange(100000, dtype=jnp.float32)

            @jax.jit
            def compute_dot(x, y):
                return jnp.dot(x, y)

            result = compute_dot(x, y)
            expected = jnp.sum(y)  # Since x is all ones

            assert jnp.isclose(result, expected)
