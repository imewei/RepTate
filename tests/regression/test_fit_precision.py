"""Regression tests for fit precision.

Tests cover:
- T051: Enhanced precision validation tests for NLSQ fitting

These tests ensure that fitting precision remains consistent across
code changes and different platforms. They serve as regression guards
to detect numerical precision degradation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import pytest
from jax import Array

if TYPE_CHECKING:
    pass


class TestLinearFitPrecision:
    """Test linear model fitting precision."""

    def test_linear_fit_exact_recovery(self) -> None:
        """Test exact recovery of linear parameters from clean data."""
        from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

        true_slope = 2.0
        true_intercept = -1.0
        xdata = jnp.linspace(0, 10, 50)
        ydata = true_slope * xdata + true_intercept

        def linear_model(x: Array, params: Array) -> Array:
            return params[0] * x + params[1]

        result, _ = run_nlsq_fit(
            linear_model,
            xdata,
            ydata,
            p0=jnp.array([1.0, 0.0]),
        )

        # Constitution requirement: 1e-10 tolerance for JAX comparisons
        assert jnp.allclose(
            result.parameters_array,
            jnp.array([true_slope, true_intercept]),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_linear_fit_small_slope(self) -> None:
        """Test fitting with very small slope values."""
        from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

        true_slope = 1e-6
        true_intercept = 1.0
        xdata = jnp.linspace(0, 1e6, 100)
        ydata = true_slope * xdata + true_intercept

        def linear_model(x: Array, params: Array) -> Array:
            return params[0] * x + params[1]

        result, _ = run_nlsq_fit(
            linear_model,
            xdata,
            ydata,
            p0=jnp.array([1e-7, 0.5]),
        )

        assert abs(result.parameters["p0"] - true_slope) / abs(true_slope) < 1e-6
        assert abs(result.parameters["p1"] - true_intercept) / abs(true_intercept) < 1e-6

    def test_linear_fit_large_values(self) -> None:
        """Test fitting with large parameter values."""
        from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

        true_slope = 1e6
        true_intercept = 1e8
        xdata = jnp.linspace(0, 100, 50)
        ydata = true_slope * xdata + true_intercept

        def linear_model(x: Array, params: Array) -> Array:
            return params[0] * x + params[1]

        result, _ = run_nlsq_fit(
            linear_model,
            xdata,
            ydata,
            p0=jnp.array([1e5, 1e7]),
        )

        assert abs(result.parameters["p0"] - true_slope) / true_slope < 1e-6
        assert abs(result.parameters["p1"] - true_intercept) / true_intercept < 1e-6


class TestMaxwellFitPrecision:
    """Test Maxwell model fitting precision."""

    def test_maxwell_storage_modulus_precision(self) -> None:
        """Test precision of Maxwell storage modulus fitting."""
        from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

        G0_true = 1e5
        tau_true = 1.0
        omega = jnp.logspace(-2, 2, 100)
        omega_tau = omega * tau_true
        G_prime = G0_true * omega_tau**2 / (1 + omega_tau**2)

        def maxwell_storage(x: Array, params: Array) -> Array:
            G0, tau = params[0], params[1]
            omega_tau_model = x * tau
            return G0 * omega_tau_model**2 / (1 + omega_tau_model**2)

        result, _ = run_nlsq_fit(
            maxwell_storage,
            omega,
            G_prime,
            p0=jnp.array([5e4, 0.5]),
        )

        # Expect very high precision for clean synthetic data
        assert abs(result.parameters["p0"] - G0_true) / G0_true < 1e-8
        assert abs(result.parameters["p1"] - tau_true) / tau_true < 1e-8

    def test_maxwell_loss_modulus_precision(self) -> None:
        """Test precision of Maxwell loss modulus fitting."""
        from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

        G0_true = 2e5
        tau_true = 0.5
        omega = jnp.logspace(-2, 2, 100)
        omega_tau = omega * tau_true
        G_double_prime = G0_true * omega_tau / (1 + omega_tau**2)

        def maxwell_loss(x: Array, params: Array) -> Array:
            G0, tau = params[0], params[1]
            omega_tau_model = x * tau
            return G0 * omega_tau_model / (1 + omega_tau_model**2)

        result, _ = run_nlsq_fit(
            maxwell_loss,
            omega,
            G_double_prime,
            p0=jnp.array([1e5, 0.3]),
        )

        assert abs(result.parameters["p0"] - G0_true) / G0_true < 1e-6
        assert abs(result.parameters["p1"] - tau_true) / tau_true < 1e-6


class TestExponentialFitPrecision:
    """Test exponential decay fitting precision."""

    def test_single_exponential_precision(self) -> None:
        """Test single exponential decay precision."""
        from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

        A_true = 1000.0
        tau_true = 2.0
        t = jnp.linspace(0, 20, 100)
        y = A_true * jnp.exp(-t / tau_true)

        def exp_decay(x: Array, params: Array) -> Array:
            return params[0] * jnp.exp(-x / params[1])

        result, _ = run_nlsq_fit(
            exp_decay,
            t,
            y,
            p0=jnp.array([500.0, 1.0]),
        )

        assert abs(result.parameters["p0"] - A_true) / A_true < 1e-8
        assert abs(result.parameters["p1"] - tau_true) / tau_true < 1e-8

    def test_stretched_exponential_precision(self) -> None:
        """Test stretched exponential (KWW) precision."""
        from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

        A_true = 100.0
        tau_true = 1.0
        beta_true = 0.5  # Stretching exponent
        t = jnp.linspace(0.01, 10, 100)
        y = A_true * jnp.exp(-jnp.power(t / tau_true, beta_true))

        def kww(x: Array, params: Array) -> Array:
            A, tau, beta = params
            return A * jnp.exp(-jnp.power(x / tau, beta))

        result, _ = run_nlsq_fit(
            kww,
            t,
            y,
            p0=jnp.array([50.0, 0.5, 0.3]),
        )

        assert abs(result.parameters["p0"] - A_true) / A_true < 1e-6
        assert abs(result.parameters["p1"] - tau_true) / tau_true < 1e-6
        assert abs(result.parameters["p2"] - beta_true) / beta_true < 1e-6


class TestResidualPrecision:
    """Test residual calculation precision."""

    def test_zero_residuals_for_exact_fit(self) -> None:
        """Test residuals are near-zero for exact data."""
        from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

        xdata = jnp.linspace(0, 10, 50)
        ydata = 2.0 * xdata + 1.0

        def linear(x: Array, params: Array) -> Array:
            return params[0] * x + params[1]

        result, _ = run_nlsq_fit(
            linear,
            xdata,
            ydata,
            p0=jnp.array([1.0, 0.0]),
        )

        # All residuals should be near machine precision
        residuals_array = jnp.array(result.residuals)
        max_residual = jnp.max(jnp.abs(residuals_array))

        assert max_residual < 1e-10

    def test_residual_sum_of_squares(self) -> None:
        """Test residual sum of squares matches expectation."""
        from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

        xdata = jnp.linspace(0, 10, 100)
        ydata = 3.0 * xdata - 2.0

        def linear(x: Array, params: Array) -> Array:
            return params[0] * x + params[1]

        result, _ = run_nlsq_fit(
            linear,
            xdata,
            ydata,
            p0=jnp.array([1.0, 0.0]),
        )

        # Sum of squared residuals should be near zero
        residuals_array = jnp.array(result.residuals)
        ssr = jnp.sum(residuals_array**2)

        assert ssr < 1e-18


class TestCovariancePrecision:
    """Test covariance matrix precision."""

    def test_covariance_symmetry(self) -> None:
        """Test covariance matrix is symmetric."""
        from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

        xdata = jnp.linspace(0, 10, 100)
        ydata = 2.0 * xdata + 1.0

        def linear(x: Array, params: Array) -> Array:
            return params[0] * x + params[1]

        result, _ = run_nlsq_fit(
            linear,
            xdata,
            ydata,
            p0=jnp.array([1.0, 0.0]),
        )

        cov = result.covariance
        assert jnp.allclose(cov, cov.T, rtol=1e-10, atol=1e-10)

    def test_covariance_positive_diagonal(self) -> None:
        """Test covariance diagonal is non-negative."""
        from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

        xdata = jnp.linspace(0, 10, 100)
        ydata = 2.0 * xdata + 1.0

        def linear(x: Array, params: Array) -> Array:
            return params[0] * x + params[1]

        result, _ = run_nlsq_fit(
            linear,
            xdata,
            ydata,
            p0=jnp.array([1.0, 0.0]),
        )

        diagonal = jnp.diag(result.covariance)
        assert jnp.all(diagonal >= 0)


class TestNumericalStabilityPrecision:
    """Test numerical stability edge cases."""

    def test_near_singular_jacobian(self) -> None:
        """Test handling of near-singular Jacobian situations."""
        from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

        # Data with low variability
        xdata = jnp.linspace(0, 0.001, 100)
        ydata = 1.0 + xdata * 1e-10  # Very flat line

        def linear(x: Array, params: Array) -> Array:
            return params[0] + params[1] * x

        # Should still converge reasonably
        result, diagnostics = run_nlsq_fit(
            linear,
            xdata,
            ydata,
            p0=jnp.array([1.0, 0.0]),
        )

        assert diagnostics.status == "success"
        # Intercept should be close to 1.0
        assert abs(result.parameters["p0"] - 1.0) < 0.01

    def test_wide_parameter_range(self) -> None:
        """Test fitting with parameters spanning many orders of magnitude."""
        from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

        # Model: y = A * exp(-x/tau) where A ~ 1e6, tau ~ 1e-3
        A_true = 1e6
        tau_true = 1e-3
        xdata = jnp.linspace(0, 0.01, 100)
        ydata = A_true * jnp.exp(-xdata / tau_true)

        def exp_model(x: Array, params: Array) -> Array:
            return params[0] * jnp.exp(-x / params[1])

        result, _ = run_nlsq_fit(
            exp_model,
            xdata,
            ydata,
            p0=jnp.array([5e5, 5e-4]),
        )

        # Should recover parameters within tolerance
        assert abs(result.parameters["p0"] - A_true) / A_true < 0.01
        assert abs(result.parameters["p1"] - tau_true) / tau_true < 0.01
