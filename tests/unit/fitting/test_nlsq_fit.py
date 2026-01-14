"""Unit tests for nlsq_fit module.

Tests cover:
- T045: FitResult, FitDiagnostics dataclasses and run_nlsq_fit function

The nlsq_fit module provides deterministic curve fitting using NLSQ.
"""

from __future__ import annotations

import pytest
import jax.numpy as jnp
from jax import Array


class TestFitResultDataclass:
    """Test FitResult dataclass."""

    def test_fit_result_creation(self) -> None:
        """Test FitResult can be created with all fields."""
        from RepTate.core.fitting.nlsq_fit import FitResult

        result = FitResult(
            parameters={"p0": 1.0, "p1": 2.0},
            parameters_array=jnp.array([1.0, 2.0]),
            covariance=jnp.eye(2),
            residuals=[0.1, -0.1, 0.05],
            warm_start={"p0": 1.0, "p1": 2.0},
        )

        assert result.parameters == {"p0": 1.0, "p1": 2.0}
        assert len(result.residuals) == 3

    def test_fit_result_is_frozen(self) -> None:
        """Test FitResult is immutable."""
        from RepTate.core.fitting.nlsq_fit import FitResult

        result = FitResult(
            parameters={"p0": 1.0},
            parameters_array=jnp.array([1.0]),
            covariance=jnp.eye(1),
            residuals=[0.0],
            warm_start={"p0": 1.0},
        )

        with pytest.raises(AttributeError):
            result.parameters = {"p0": 2.0}  # type: ignore[misc]


class TestFitDiagnosticsDataclass:
    """Test FitDiagnostics dataclass."""

    def test_diagnostics_creation(self) -> None:
        """Test FitDiagnostics can be created."""
        from RepTate.core.fitting.nlsq_fit import FitDiagnostics

        diag = FitDiagnostics(nfev=10, status="success")

        assert diag.nfev == 10
        assert diag.status == "success"

    def test_diagnostics_as_dict(self) -> None:
        """Test as_dict method returns dictionary."""
        from RepTate.core.fitting.nlsq_fit import FitDiagnostics

        diag = FitDiagnostics(nfev=42, status="converged")
        result = diag.as_dict()

        assert result == {"nfev": 42, "status": "converged"}

    def test_diagnostics_nfev_none(self) -> None:
        """Test diagnostics with None nfev."""
        from RepTate.core.fitting.nlsq_fit import FitDiagnostics

        diag = FitDiagnostics(nfev=None, status="unknown")
        result = diag.as_dict()

        assert result["nfev"] is None


class TestRunNlsqFit:
    """Test run_nlsq_fit function."""

    def test_linear_fit(self) -> None:
        """Test fitting a linear model y = a*x + b."""
        from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

        # Generate synthetic data: y = 2*x + 1
        xdata = jnp.linspace(0, 10, 50)
        ydata = 2.0 * xdata + 1.0

        def linear_model(x: Array, params: Array) -> Array:
            a, b = params[0], params[1]
            return a * x + b

        result, diagnostics = run_nlsq_fit(
            linear_model,
            xdata,
            ydata,
            p0=jnp.array([1.0, 0.0]),
        )

        assert abs(result.parameters["p0"] - 2.0) < 1e-6
        assert abs(result.parameters["p1"] - 1.0) < 1e-6
        assert diagnostics.status == "success"

    def test_quadratic_fit(self) -> None:
        """Test fitting a quadratic model y = a*x^2 + b*x + c."""
        from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

        # Generate synthetic data: y = 0.5*x^2 - 2*x + 3
        xdata = jnp.linspace(-5, 5, 100)
        ydata = 0.5 * xdata**2 - 2.0 * xdata + 3.0

        def quadratic_model(x: Array, params: Array) -> Array:
            a, b, c = params[0], params[1], params[2]
            return a * x**2 + b * x + c

        result, _ = run_nlsq_fit(
            quadratic_model,
            xdata,
            ydata,
            p0=jnp.array([0.1, -0.1, 0.1]),
        )

        assert abs(result.parameters["p0"] - 0.5) < 1e-5
        assert abs(result.parameters["p1"] - (-2.0)) < 1e-5
        assert abs(result.parameters["p2"] - 3.0) < 1e-5

    def test_exponential_fit(self) -> None:
        """Test fitting an exponential decay model."""
        from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

        # Generate synthetic data: y = A * exp(-t/tau)
        xdata = jnp.linspace(0, 10, 100)
        A_true, tau_true = 1000.0, 2.0
        ydata = A_true * jnp.exp(-xdata / tau_true)

        def exp_model(x: Array, params: Array) -> Array:
            A, tau = params[0], params[1]
            return A * jnp.exp(-x / tau)

        result, _ = run_nlsq_fit(
            exp_model,
            xdata,
            ydata,
            p0=jnp.array([500.0, 1.0]),
        )

        assert abs(result.parameters["p0"] - A_true) / A_true < 0.01
        assert abs(result.parameters["p1"] - tau_true) / tau_true < 0.01

    def test_result_has_covariance(self) -> None:
        """Test result includes covariance matrix."""
        from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

        xdata = jnp.linspace(0, 1, 20)
        ydata = 3.0 * xdata + 2.0

        def linear_model(x: Array, params: Array) -> Array:
            return params[0] * x + params[1]

        result, _ = run_nlsq_fit(
            linear_model,
            xdata,
            ydata,
            p0=jnp.array([1.0, 1.0]),
        )

        assert result.covariance.shape == (2, 2)

    def test_result_has_residuals(self) -> None:
        """Test result includes residuals."""
        from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

        xdata = jnp.linspace(0, 1, 10)
        ydata = 2.0 * xdata

        def linear_model(x: Array, params: Array) -> Array:
            return params[0] * x

        result, _ = run_nlsq_fit(
            linear_model,
            xdata,
            ydata,
            p0=jnp.array([1.0]),
        )

        assert len(result.residuals) == 10
        # Perfect fit should have near-zero residuals
        assert all(abs(r) < 1e-6 for r in result.residuals)

    def test_warm_start_equals_parameters(self) -> None:
        """Test warm_start contains final parameters."""
        from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

        xdata = jnp.linspace(0, 1, 20)
        ydata = xdata + 1.0

        def linear_model(x: Array, params: Array) -> Array:
            return params[0] * x + params[1]

        result, _ = run_nlsq_fit(
            linear_model,
            xdata,
            ydata,
            p0=jnp.array([0.5, 0.5]),
        )

        assert result.warm_start == result.parameters

    def test_workflow_parameter_accepted(self) -> None:
        """Test workflow parameter is accepted for memory-aware optimization."""
        from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

        xdata = jnp.linspace(0, 1, 50)
        ydata = 2.0 * xdata + 1.0

        def linear_model(x: Array, params: Array) -> Array:
            return params[0] * x + params[1]

        # Test with explicit workflow parameter
        result, diagnostics = run_nlsq_fit(
            linear_model,
            xdata,
            ydata,
            p0=jnp.array([1.0, 0.0]),
            workflow="auto",
            show_progress=False,
        )

        assert abs(result.parameters["p0"] - 2.0) < 1e-5
        assert diagnostics.status == "success"

    def test_fit_with_bounds(self) -> None:
        """Test fitting with parameter bounds."""
        from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

        xdata = jnp.linspace(0, 1, 20)
        ydata = 5.0 * xdata  # True slope is 5

        def linear_model(x: Array, params: Array) -> Array:
            return params[0] * x

        # Bound slope between 4 and 6
        result, _ = run_nlsq_fit(
            linear_model,
            xdata,
            ydata,
            p0=jnp.array([1.0]),
            bounds=(4.0, 6.0),
        )

        assert 4.0 <= result.parameters["p0"] <= 6.0


class TestNlsqFitWithNoise:
    """Test fitting with noisy data."""

    def test_linear_fit_with_noise(self) -> None:
        """Test linear fit recovers parameters from noisy data."""
        from RepTate.core.fitting.nlsq_fit import run_nlsq_fit
        import jax

        # Generate noisy data
        key = jax.random.PRNGKey(42)
        xdata = jnp.linspace(0, 10, 100)
        noise = jax.random.normal(key, shape=xdata.shape) * 0.5
        ydata = 2.0 * xdata + 1.0 + noise

        def linear_model(x: Array, params: Array) -> Array:
            return params[0] * x + params[1]

        result, _ = run_nlsq_fit(
            linear_model,
            xdata,
            ydata,
            p0=jnp.array([1.0, 0.0]),
        )

        # With noise, we expect reasonable but not perfect recovery
        assert abs(result.parameters["p0"] - 2.0) < 0.5
        assert abs(result.parameters["p1"] - 1.0) < 1.0
