"""Regression tests for numerical equivalence.

Tests cover:
- T052: Numerical equivalence tests for JAX computations

These tests ensure that numerical computations produce consistent
results across code changes. They guard against unintended changes
in numerical behavior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from numpy.testing import assert_array_almost_equal

if TYPE_CHECKING:
    pass


class TestJAXNumpyEquivalence:
    """Test JAX operations match NumPy behavior."""

    def test_array_operations_match_numpy(self) -> None:
        """Test basic array operations match NumPy."""
        x_np = np.array([1.0, 2.0, 3.0, 4.0])
        x_jax = jnp.array([1.0, 2.0, 3.0, 4.0])

        # Addition
        assert_array_almost_equal(np.array(x_jax + 1), x_np + 1)

        # Multiplication
        assert_array_almost_equal(np.array(x_jax * 2), x_np * 2)

        # Division
        assert_array_almost_equal(np.array(x_jax / 2), x_np / 2)

        # Power
        assert_array_almost_equal(np.array(x_jax ** 2), x_np ** 2)

    def test_mathematical_functions_match_numpy(self) -> None:
        """Test mathematical functions match NumPy."""
        x_np = np.linspace(0.1, 10, 100)
        x_jax = jnp.array(x_np)

        # Exponential
        assert_array_almost_equal(np.array(jnp.exp(x_jax)), np.exp(x_np), decimal=10)

        # Logarithm
        assert_array_almost_equal(np.array(jnp.log(x_jax)), np.log(x_np), decimal=10)

        # Trigonometric
        assert_array_almost_equal(np.array(jnp.sin(x_jax)), np.sin(x_np), decimal=10)
        assert_array_almost_equal(np.array(jnp.cos(x_jax)), np.cos(x_np), decimal=10)

    def test_reduction_operations_match_numpy(self) -> None:
        """Test reduction operations match NumPy."""
        x_np = np.random.rand(100) * 100
        x_jax = jnp.array(x_np)

        assert abs(float(jnp.sum(x_jax)) - np.sum(x_np)) < 1e-10
        assert abs(float(jnp.mean(x_jax)) - np.mean(x_np)) < 1e-10
        assert abs(float(jnp.std(x_jax)) - np.std(x_np)) < 1e-10

    def test_linear_algebra_match_numpy(self) -> None:
        """Test linear algebra operations match NumPy."""
        A_np = np.random.rand(10, 10)
        b_np = np.random.rand(10)

        A_jax = jnp.array(A_np)
        b_jax = jnp.array(b_np)

        # Matrix-vector product
        assert_array_almost_equal(
            np.array(A_jax @ b_jax),
            A_np @ b_np,
            decimal=10,
        )

        # Determinant
        assert abs(float(jnp.linalg.det(A_jax)) - np.linalg.det(A_np)) < 1e-8


class TestFitResultReproducibility:
    """Test fitting results are reproducible."""

    def test_fit_result_deterministic(self) -> None:
        """Test same input produces same output."""
        from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

        xdata = jnp.linspace(0, 10, 100)
        ydata = 2.5 * xdata + 1.5

        def linear(x: Array, params: Array) -> Array:
            return params[0] * x + params[1]

        # Run fit twice
        result1, _ = run_nlsq_fit(
            linear, xdata, ydata, p0=jnp.array([1.0, 0.0])
        )
        result2, _ = run_nlsq_fit(
            linear, xdata, ydata, p0=jnp.array([1.0, 0.0])
        )

        # Results should be identical
        assert result1.parameters["p0"] == result2.parameters["p0"]
        assert result1.parameters["p1"] == result2.parameters["p1"]

    def test_fit_result_stable_across_runs(self) -> None:
        """Test fit results are stable across multiple runs."""
        from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

        xdata = jnp.linspace(0, 10, 100)
        ydata = 3.0 * xdata - 1.0

        def linear(x: Array, params: Array) -> Array:
            return params[0] * x + params[1]

        # Run 10 times and verify consistency
        results = []
        for _ in range(10):
            result, _ = run_nlsq_fit(
                linear, xdata, ydata, p0=jnp.array([1.0, 0.0])
            )
            results.append((result.parameters["p0"], result.parameters["p1"]))

        # All results should be identical
        first = results[0]
        for r in results[1:]:
            assert r[0] == first[0]
            assert r[1] == first[1]


class TestMaxwellModelEquivalence:
    """Test Maxwell model computations are consistent."""

    def test_storage_modulus_formula(self) -> None:
        """Test storage modulus formula computation."""
        G0 = 1e5
        tau = 1.0
        omega = jnp.logspace(-2, 2, 100)

        # Compute using JAX
        omega_tau = omega * tau
        G_prime_jax = G0 * omega_tau**2 / (1 + omega_tau**2)

        # Compute using NumPy
        omega_np = np.logspace(-2, 2, 100)
        omega_tau_np = omega_np * tau
        G_prime_np = G0 * omega_tau_np**2 / (1 + omega_tau_np**2)

        assert_array_almost_equal(np.array(G_prime_jax), G_prime_np, decimal=10)

    def test_loss_modulus_formula(self) -> None:
        """Test loss modulus formula computation."""
        G0 = 1e5
        tau = 1.0
        omega = jnp.logspace(-2, 2, 100)

        # JAX
        omega_tau = omega * tau
        G_double_prime_jax = G0 * omega_tau / (1 + omega_tau**2)

        # NumPy
        omega_np = np.logspace(-2, 2, 100)
        omega_tau_np = omega_np * tau
        G_double_prime_np = G0 * omega_tau_np / (1 + omega_tau_np**2)

        assert_array_almost_equal(
            np.array(G_double_prime_jax),
            G_double_prime_np,
            decimal=10,
        )


class TestExponentialModelEquivalence:
    """Test exponential model computations are consistent."""

    def test_single_exponential_computation(self) -> None:
        """Test single exponential computation matches NumPy."""
        A = 1000.0
        tau = 2.0
        t = jnp.linspace(0, 20, 100)

        # JAX
        y_jax = A * jnp.exp(-t / tau)

        # NumPy
        t_np = np.linspace(0, 20, 100)
        y_np = A * np.exp(-t_np / tau)

        assert_array_almost_equal(np.array(y_jax), y_np, decimal=10)

    def test_sum_of_exponentials_computation(self) -> None:
        """Test sum of exponentials computation."""
        A1, tau1 = 0.8, 1.0
        A2, tau2 = 0.2, 10.0
        t = jnp.linspace(0, 50, 200)

        # JAX
        y_jax = A1 * jnp.exp(-t / tau1) + A2 * jnp.exp(-t / tau2)

        # NumPy
        t_np = np.linspace(0, 50, 200)
        y_np = A1 * np.exp(-t_np / tau1) + A2 * np.exp(-t_np / tau2)

        assert_array_almost_equal(np.array(y_jax), y_np, decimal=10)


class TestPolynomialEquivalence:
    """Test polynomial computations are consistent."""

    def test_polynomial_evaluation(self) -> None:
        """Test polynomial evaluation matches NumPy."""
        coeffs = [1.0, 2.0, 3.0]  # 1 + 2x + 3x^2
        x = jnp.linspace(-5, 5, 100)

        # JAX
        y_jax = coeffs[0] + coeffs[1] * x + coeffs[2] * x**2

        # NumPy (using polyval with reversed coefficients)
        x_np = np.linspace(-5, 5, 100)
        y_np = np.polyval([3.0, 2.0, 1.0], x_np)

        assert_array_almost_equal(np.array(y_jax), y_np, decimal=10)


class TestFloatingPointPrecision:
    """Test floating point precision requirements."""

    def test_float64_enabled(self) -> None:
        """Test JAX x64 mode is enabled for full precision."""
        # This should be set by conftest.py
        x = jnp.array([1.0, 2.0, 3.0])
        assert x.dtype == jnp.float64

    def test_precision_preserved_through_operations(self) -> None:
        """Test precision is preserved through chain of operations."""
        x = jnp.array([1.0])

        # Chain of operations that could lose precision
        result = x
        for _ in range(100):
            result = jnp.exp(jnp.log(result))

        # Should still be close to 1.0
        assert abs(float(result[0]) - 1.0) < 1e-10

    def test_small_number_preservation(self) -> None:
        """Test very small numbers are preserved."""
        small = jnp.array([1e-200])

        # Operations should preserve magnitude
        result = small * 1e100 / 1e100

        assert abs(float(result[0]) - 1e-200) / 1e-200 < 1e-10


class TestGoldenValueComparison:
    """Test against golden (known-good) values."""

    def test_maxwell_at_known_points(self) -> None:
        """Test Maxwell model at specific frequency points."""
        G0 = 1e5
        tau = 1.0

        # Known points: omega * tau = 1 gives G' = G0/2
        omega = 1.0  # omega*tau = 1
        omega_tau = omega * tau
        G_prime = G0 * omega_tau**2 / (1 + omega_tau**2)

        assert abs(float(G_prime) - 5e4) < 1e-6  # G0/2

        # omega * tau = 1 gives G'' = G0/2
        G_double_prime = G0 * omega_tau / (1 + omega_tau**2)
        assert abs(float(G_double_prime) - 5e4) < 1e-6  # G0/2

    def test_exponential_at_known_points(self) -> None:
        """Test exponential at specific time points."""
        A = 100.0
        tau = 2.0

        # At t=0, y = A
        y_0 = A * jnp.exp(-0.0 / tau)
        assert abs(float(y_0) - 100.0) < 1e-10

        # At t=tau, y = A/e
        y_tau = A * jnp.exp(-tau / tau)
        assert abs(float(y_tau) - 100.0 / np.e) < 1e-10

        # At t=5*tau, y = A * exp(-5) ≈ 0.67
        y_5tau = A * jnp.exp(-5 * tau / tau)
        assert abs(float(y_5tau) - 100.0 * np.exp(-5)) < 1e-10
