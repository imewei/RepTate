"""Unit tests for TheoryCompute.

Tests cover:
- T065: Unit tests for TheoryCompute component

These tests validate the TheoryCompute component extracted from QTheory.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

if TYPE_CHECKING:
    pass


class TestTheoryComputeInit:
    """Test TheoryCompute initialization."""

    def test_init_with_logger(self) -> None:
        """Test initialization with custom logger."""
        import logging

        from RepTate.gui.TheoryCompute import TheoryCompute

        logger = logging.getLogger("test")
        compute = TheoryCompute(logger=logger)
        assert compute.logger is logger

    def test_init_default_logger(self) -> None:
        """Test initialization with default logger."""
        from RepTate.gui.TheoryCompute import TheoryCompute

        compute = TheoryCompute()
        assert compute.logger is not None


class TestResidualCalculation:
    """Test residual calculation methods."""

    def test_calculate_residuals(self) -> None:
        """Test basic residual calculation."""
        from RepTate.gui.TheoryCompute import TheoryCompute

        compute = TheoryCompute()
        y_data = jnp.array([1.0, 2.0, 3.0, 4.0])
        y_theory = jnp.array([1.1, 1.9, 3.0, 4.2])

        residuals = compute.calculate_residuals(y_data, y_theory)

        expected = jnp.array([-0.1, 0.1, 0.0, -0.2])
        assert jnp.allclose(residuals, expected, atol=1e-10)

    def test_calculate_residuals_zero(self) -> None:
        """Test residuals are zero for exact match."""
        from RepTate.gui.TheoryCompute import TheoryCompute

        compute = TheoryCompute()
        y = jnp.array([1.0, 2.0, 3.0])

        residuals = compute.calculate_residuals(y, y)

        assert jnp.allclose(residuals, jnp.zeros_like(y), atol=1e-14)


class TestChiSquared:
    """Test chi-squared calculation."""

    def test_chi_squared_unweighted(self) -> None:
        """Test chi-squared without weights."""
        from RepTate.gui.TheoryCompute import TheoryCompute

        compute = TheoryCompute()
        residuals = jnp.array([0.1, -0.1, 0.2])

        chi2 = compute.calculate_chi_squared(residuals)

        expected = 0.01 + 0.01 + 0.04
        assert abs(chi2 - expected) < 1e-10

    def test_chi_squared_weighted(self) -> None:
        """Test chi-squared with weights."""
        from RepTate.gui.TheoryCompute import TheoryCompute

        compute = TheoryCompute()
        residuals = jnp.array([1.0, 1.0])
        weights = jnp.array([1.0, 2.0])

        chi2 = compute.calculate_chi_squared(residuals, weights)

        expected = 1.0 * 1.0 + 2.0 * 1.0
        assert abs(chi2 - expected) < 1e-10


class TestErrorCalculation:
    """Test error calculation methods."""

    def test_error_standard(self) -> None:
        """Test standard relative error calculation."""
        from RepTate.gui.TheoryCompute import TheoryCompute

        compute = TheoryCompute()
        y_data = jnp.array([10.0, 20.0, 30.0])
        y_theory = jnp.array([11.0, 19.0, 30.0])

        error = compute.calculate_error_standard(y_data, y_theory)

        # Relative residuals: -0.1, 0.05, 0
        # Sum of squares: 0.01 + 0.0025 + 0 = 0.0125
        expected = 0.01 + 0.0025 + 0.0
        assert abs(error - expected) < 1e-8

    def test_error_abs(self) -> None:
        """Test absolute error calculation."""
        from RepTate.gui.TheoryCompute import TheoryCompute

        compute = TheoryCompute()
        y_data = jnp.array([1.0, 2.0, 3.0])
        y_theory = jnp.array([1.1, 2.1, 3.1])

        error = compute.calculate_error_abs(y_data, y_theory)

        # Sum of (0.1)^2 = 0.03
        expected = 0.01 + 0.01 + 0.01
        assert abs(error - expected) < 1e-10

    def test_error_log(self) -> None:
        """Test logarithmic error calculation."""
        from RepTate.gui.TheoryCompute import TheoryCompute

        compute = TheoryCompute()
        y_data = jnp.array([10.0, 100.0])
        y_theory = jnp.array([10.0, 100.0])

        error = compute.calculate_error_log(y_data, y_theory)

        # Perfect match should have zero error
        assert error < 1e-20


class TestRSquared:
    """Test R-squared calculation."""

    def test_r_squared_perfect_fit(self) -> None:
        """Test R-squared for perfect fit."""
        from RepTate.gui.TheoryCompute import TheoryCompute

        compute = TheoryCompute()
        y = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        r2 = compute.calculate_r_squared(y, y)

        assert abs(r2 - 1.0) < 1e-10

    def test_r_squared_good_fit(self) -> None:
        """Test R-squared for good fit."""
        from RepTate.gui.TheoryCompute import TheoryCompute

        compute = TheoryCompute()
        y_data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_theory = jnp.array([1.1, 1.9, 3.05, 4.0, 4.95])

        r2 = compute.calculate_r_squared(y_data, y_theory)

        # Should be close to 1 for good fit
        assert r2 > 0.99


class TestXRangeExtension:
    """Test x-range extension functionality."""

    def test_extend_x_range_log(self) -> None:
        """Test log-scale x-range extension."""
        from RepTate.gui.TheoryCompute import TheoryCompute

        compute = TheoryCompute()
        x_data = jnp.array([1.0, 10.0, 100.0])

        x_ext = compute.extend_x_range(
            x_data, xmin_ext=0.1, xmax_ext=1000.0, log_scale=True
        )

        # Check range (with floating-point tolerance)
        assert float(jnp.min(x_ext)) <= 0.1 + 1e-10
        assert float(jnp.max(x_ext)) >= 1000.0 - 1e-6
        # Check log spacing
        assert len(x_ext) > 10

    def test_extend_x_range_linear(self) -> None:
        """Test linear x-range extension."""
        from RepTate.gui.TheoryCompute import TheoryCompute

        compute = TheoryCompute()
        x_data = jnp.array([0.0, 5.0, 10.0])

        x_ext = compute.extend_x_range(
            x_data, xmin_ext=-5.0, xmax_ext=20.0, log_scale=False
        )

        # Check range
        assert float(jnp.min(x_ext)) <= -5.0
        assert float(jnp.max(x_ext)) >= 20.0


class TestInterpolation:
    """Test theory interpolation."""

    def test_interpolate_theory_linear(self) -> None:
        """Test linear interpolation."""
        from RepTate.gui.TheoryCompute import TheoryCompute

        compute = TheoryCompute()
        x_theory = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y_theory = jnp.array([0.0, 2.0, 4.0, 6.0, 8.0])  # y = 2x
        x_data = jnp.array([0.5, 1.5, 2.5])

        y_interp = compute.interpolate_theory(x_theory, y_theory, x_data)

        expected = jnp.array([1.0, 3.0, 5.0])
        assert jnp.allclose(y_interp, expected, atol=0.1)


class TestStatisticalFunctions:
    """Test statistical functions."""

    def test_student_t_cdf_symmetry(self) -> None:
        """Test Student t CDF is symmetric."""
        from RepTate.gui.TheoryCompute import student_t_cdf

        cdf_neg = student_t_cdf(-1.0, 10)
        cdf_pos = student_t_cdf(1.0, 10)

        assert abs(float(cdf_neg) + float(cdf_pos) - 1.0) < 1e-10

    def test_student_t_cdf_at_zero(self) -> None:
        """Test Student t CDF at zero is 0.5."""
        from RepTate.gui.TheoryCompute import student_t_cdf

        cdf_zero = student_t_cdf(0.0, 10)

        assert abs(float(cdf_zero) - 0.5) < 1e-10

    def test_student_t_ppf_inverse(self) -> None:
        """Test PPF is inverse of CDF."""
        from RepTate.gui.TheoryCompute import student_t_cdf, student_t_ppf

        p = 0.95
        dof = 10
        t_val = student_t_ppf(p, dof)
        p_recovered = float(student_t_cdf(t_val, dof))

        assert abs(p - p_recovered) < 1e-6


class TestConfidenceInterval:
    """Test confidence interval calculation."""

    def test_confidence_interval_basic(self) -> None:
        """Test basic confidence interval calculation."""
        from RepTate.gui.TheoryCompute import TheoryCompute

        compute = TheoryCompute()
        n_params = 2
        n_points = 100
        residual_variance = 0.01
        jacobian_gram = jnp.eye(2) * 100.0  # Well-conditioned

        ci = compute.confidence_interval(
            n_params, n_points, residual_variance, jacobian_gram
        )

        assert len(ci) == 2
        assert jnp.all(ci > 0)  # Confidence intervals should be positive
        assert jnp.all(ci < 1.0)  # Should be small for well-conditioned problem

    def test_confidence_interval_insufficient_dof(self) -> None:
        """Test handling of insufficient degrees of freedom."""
        from RepTate.gui.TheoryCompute import TheoryCompute

        compute = TheoryCompute()
        n_params = 5
        n_points = 5  # dof = 0
        residual_variance = 0.01
        jacobian_gram = jnp.eye(5)

        ci = compute.confidence_interval(
            n_params, n_points, residual_variance, jacobian_gram
        )

        # Should return inf for all parameters
        assert jnp.all(jnp.isinf(ci))
