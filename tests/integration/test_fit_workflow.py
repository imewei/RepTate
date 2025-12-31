"""Integration tests for end-to-end fit workflow.

Tests cover:
- T049: Complete fitting workflow from data load to result export

The fit workflow integration tests validate:
1. Data preparation and normalization
2. Model definition and parameter setup
3. Fitting execution with NLSQ
4. Result extraction and validation
5. Error/uncertainty propagation
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

if TYPE_CHECKING:
    from pathlib import Path


class TestNLSQFitWorkflow:
    """End-to-end tests for NLSQ fitting workflow."""

    def test_linear_model_fit_workflow(self) -> None:
        """Test complete linear model fitting workflow."""
        from RepTate.core.fitting.nlsq_fit import run_nlsq_fit, FitResult

        # Step 1: Generate synthetic data
        true_slope = 2.5
        true_intercept = 1.0
        xdata = jnp.linspace(0, 10, 100)
        ydata = true_slope * xdata + true_intercept

        # Step 2: Define model function
        def linear_model(x: Array, params: Array) -> Array:
            return params[0] * x + params[1]

        # Step 3: Set initial guess
        p0 = jnp.array([1.0, 0.0])

        # Step 4: Run fit
        result, diagnostics = run_nlsq_fit(
            linear_model,
            xdata,
            ydata,
            p0=p0,
        )

        # Step 5: Validate results
        assert isinstance(result, FitResult)
        assert diagnostics.status == "success"

        # Verify recovered parameters match true values
        assert abs(result.parameters["p0"] - true_slope) < 1e-10
        assert abs(result.parameters["p1"] - true_intercept) < 1e-10

        # Verify residuals are near zero
        assert all(abs(r) < 1e-10 for r in result.residuals)

        # Verify covariance matrix has correct shape
        assert result.covariance.shape == (2, 2)

    def test_maxwell_model_fit_workflow(self) -> None:
        """Test Maxwell model fitting workflow for rheology data."""
        from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

        # Step 1: Generate synthetic Maxwell model data
        # G'(omega) = G0 * (omega*tau)^2 / (1 + (omega*tau)^2)
        G0_true = 1e5
        tau_true = 1.0
        omega = jnp.logspace(-2, 2, 100)
        omega_tau = omega * tau_true
        G_prime = G0_true * omega_tau**2 / (1 + omega_tau**2)

        # Step 2: Define Maxwell storage modulus model
        def maxwell_storage(x: Array, params: Array) -> Array:
            G0, tau = params[0], params[1]
            omega_tau_model = x * tau
            return G0 * omega_tau_model**2 / (1 + omega_tau_model**2)

        # Step 3: Fit with initial guess
        p0 = jnp.array([5e4, 0.5])  # Initial guess away from true values

        result, diagnostics = run_nlsq_fit(
            maxwell_storage,
            omega,
            G_prime,
            p0=p0,
        )

        # Step 4: Validate fit quality
        assert diagnostics.status == "success"
        assert abs(result.parameters["p0"] - G0_true) / G0_true < 0.01  # 1% tolerance
        assert abs(result.parameters["p1"] - tau_true) / tau_true < 0.01

    def test_exponential_decay_fit_workflow(self) -> None:
        """Test exponential decay fitting workflow."""
        from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

        # Step 1: Generate exponential decay data
        A_true = 1000.0
        tau_true = 2.5
        t = jnp.linspace(0, 20, 100)
        y = A_true * jnp.exp(-t / tau_true)

        # Step 2: Define exponential model
        def exp_decay(x: Array, params: Array) -> Array:
            A, tau = params[0], params[1]
            return A * jnp.exp(-x / tau)

        # Step 3: Fit
        p0 = jnp.array([500.0, 1.0])
        result, _ = run_nlsq_fit(exp_decay, t, y, p0=p0)

        # Step 4: Validate
        assert abs(result.parameters["p0"] - A_true) / A_true < 0.01
        assert abs(result.parameters["p1"] - tau_true) / tau_true < 0.01

    def test_fit_with_bounds_workflow(self) -> None:
        """Test fitting workflow with parameter bounds."""
        from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

        # Data that would fit slope = 10
        xdata = jnp.linspace(0, 1, 50)
        ydata = 10.0 * xdata

        def linear(x: Array, params: Array) -> Array:
            return params[0] * x

        # Bound slope to [0, 5] - should hit upper bound
        result, _ = run_nlsq_fit(
            linear,
            xdata,
            ydata,
            p0=jnp.array([1.0]),
            bounds=(0.0, 5.0),
        )

        # Should be at or near upper bound
        assert result.parameters["p0"] <= 5.0


class TestFitWorkflowWithNoise:
    """Test fitting workflow with noisy data."""

    def test_linear_fit_with_noise(self) -> None:
        """Test linear fit recovers approximate parameters from noisy data."""
        from RepTate.core.fitting.nlsq_fit import run_nlsq_fit
        import jax

        # True parameters
        true_slope = 3.0
        true_intercept = 2.0

        # Generate noisy data
        key = jax.random.PRNGKey(42)
        xdata = jnp.linspace(0, 10, 100)
        noise = jax.random.normal(key, shape=xdata.shape) * 0.5
        ydata = true_slope * xdata + true_intercept + noise

        def linear_model(x: Array, params: Array) -> Array:
            return params[0] * x + params[1]

        result, _ = run_nlsq_fit(
            linear_model,
            xdata,
            ydata,
            p0=jnp.array([1.0, 0.0]),
        )

        # With noise, accept larger tolerance
        assert abs(result.parameters["p0"] - true_slope) < 0.5
        assert abs(result.parameters["p1"] - true_intercept) < 1.0

    def test_maxwell_fit_with_noise(self) -> None:
        """Test Maxwell model fit with realistic noise levels."""
        from RepTate.core.fitting.nlsq_fit import run_nlsq_fit
        import jax

        G0_true = 1e5
        tau_true = 1.0

        key = jax.random.PRNGKey(123)
        omega = jnp.logspace(-2, 2, 100)
        omega_tau = omega * tau_true
        G_prime_clean = G0_true * omega_tau**2 / (1 + omega_tau**2)

        # Add 5% relative noise
        noise = jax.random.normal(key, shape=G_prime_clean.shape) * 0.05 * G_prime_clean
        G_prime_noisy = G_prime_clean + noise

        def maxwell_storage(x: Array, params: Array) -> Array:
            G0, tau = params[0], params[1]
            omega_tau_model = x * tau
            return G0 * omega_tau_model**2 / (1 + omega_tau_model**2)

        result, _ = run_nlsq_fit(
            maxwell_storage,
            omega,
            G_prime_noisy,
            p0=jnp.array([5e4, 0.5]),
        )

        # With 5% noise, expect ~5% accuracy
        assert abs(result.parameters["p0"] - G0_true) / G0_true < 0.1
        assert abs(result.parameters["p1"] - tau_true) / tau_true < 0.1


class TestFitResultExport:
    """Test fit result export functionality."""

    def test_result_parameters_dict_format(self) -> None:
        """Test result.parameters is a usable dictionary."""
        from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

        xdata = jnp.linspace(0, 1, 50)
        ydata = 2.0 * xdata + 1.0

        def linear(x: Array, params: Array) -> Array:
            return params[0] * x + params[1]

        result, _ = run_nlsq_fit(linear, xdata, ydata, p0=jnp.array([1.0, 0.0]))

        # Verify dict format
        assert isinstance(result.parameters, dict)
        assert "p0" in result.parameters
        assert "p1" in result.parameters

        # Values should be plain floats
        assert isinstance(result.parameters["p0"], float)
        assert isinstance(result.parameters["p1"], float)

    def test_result_warm_start_for_continuation(self) -> None:
        """Test warm_start field can be used for continuation."""
        from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

        xdata = jnp.linspace(0, 1, 50)
        ydata = 2.0 * xdata + 1.0

        def linear(x: Array, params: Array) -> Array:
            return params[0] * x + params[1]

        result, _ = run_nlsq_fit(linear, xdata, ydata, p0=jnp.array([1.0, 0.0]))

        # warm_start should equal parameters
        assert result.warm_start == result.parameters

        # warm_start should be usable for next fit
        assert isinstance(result.warm_start, dict)
        warm_values = jnp.array(list(result.warm_start.values()))
        assert warm_values.shape == (2,)


class TestMultiParameterFitting:
    """Test fitting with multiple parameters."""

    def test_three_parameter_fit(self) -> None:
        """Test fitting with three parameters."""
        from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

        # Model: y = a * x^2 + b * x + c
        a_true, b_true, c_true = 0.5, -2.0, 3.0
        xdata = jnp.linspace(-5, 5, 100)
        ydata = a_true * xdata**2 + b_true * xdata + c_true

        def quadratic(x: Array, params: Array) -> Array:
            return params[0] * x**2 + params[1] * x + params[2]

        result, _ = run_nlsq_fit(
            quadratic,
            xdata,
            ydata,
            p0=jnp.array([0.1, 0.0, 0.0]),
        )

        assert abs(result.parameters["p0"] - a_true) < 1e-8
        assert abs(result.parameters["p1"] - b_true) < 1e-8
        assert abs(result.parameters["p2"] - c_true) < 1e-8

    def test_four_parameter_fit(self) -> None:
        """Test fitting with four parameters (two-mode Maxwell)."""
        from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

        # Two-mode Maxwell: G' = G1*(w*t1)^2/(1+(w*t1)^2) + G2*(w*t2)^2/(1+(w*t2)^2)
        G1_true, tau1_true = 1e5, 1.0
        G2_true, tau2_true = 5e4, 0.1

        omega = jnp.logspace(-3, 3, 200)

        def two_mode_maxwell(x: Array, params: Array) -> Array:
            G1, tau1, G2, tau2 = params
            wt1 = x * tau1
            wt2 = x * tau2
            return G1 * wt1**2 / (1 + wt1**2) + G2 * wt2**2 / (1 + wt2**2)

        G_prime = two_mode_maxwell(omega, jnp.array([G1_true, tau1_true, G2_true, tau2_true]))

        result, _ = run_nlsq_fit(
            two_mode_maxwell,
            omega,
            G_prime,
            p0=jnp.array([5e4, 0.5, 2e4, 0.05]),
        )

        # Four parameters in result
        assert len(result.parameters) == 4
        assert result.covariance.shape == (4, 4)
