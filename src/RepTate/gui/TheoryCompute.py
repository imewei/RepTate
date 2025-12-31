# RepTate: Rheology of Entangled Polymers: Toolkit for the Analysis of Theory and Experiments
# --------------------------------------------------------------------------------------------------------
#
# Authors:
#     Jorge Ramirez, jorge.ramirez@upm.es
#     Victor Boudara, victor.boudara@gmail.com
#
# Useful links:
#     http://blogs.upm.es/compsoftmatter/software/reptate/
#     https://github.com/jorge-ramirez-upm/RepTate
#     http://reptate.readthedocs.io
#
# --------------------------------------------------------------------------------------------------------
#
# Copyright (2017-2023): Jorge Ramirez, Victor Boudara, Universidad Politécnica de Madrid, University of Leeds
#
# This file is part of RepTate.
#
# RepTate is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RepTate is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RepTate.  If not, see <http://www.gnu.org/licenses/>.
#
# --------------------------------------------------------------------------------------------------------
"""Module TheoryCompute

Extracted from QTheory to handle pure JAX-based computation and error calculation.

This class follows the Single Responsibility Principle by focusing exclusively
on numerical computation without GUI dependencies.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import jax.numpy as jnp
import jax.scipy.special as jsp_special
import numpy as np
from jax import Array

if TYPE_CHECKING:
    from RepTate.core.DataTable import DataTable


def student_t_cdf(x: float, dof: int) -> Array:
    """Compute the CDF of Student's t-distribution using JAX.

    Args:
        x: Value at which to evaluate the CDF.
        dof: Degrees of freedom.

    Returns:
        CDF value at x.
    """
    v = jnp.asarray(dof, dtype=jnp.float64)
    x_arr = jnp.asarray(x, dtype=jnp.float64)
    t2 = x_arr * x_arr
    ibeta = jsp_special.betainc(v / 2.0, 0.5, v / (v + t2))
    return 0.5 * (1.0 + jnp.sign(x_arr) * (1.0 - ibeta))


def student_t_ppf(p: float, dof: int) -> float:
    """Compute the PPF (inverse CDF) of Student's t-distribution.

    Uses bisection method for inverse CDF calculation.

    Args:
        p: Probability value (0 < p < 1).
        dof: Degrees of freedom.

    Returns:
        t-value corresponding to probability p.
    """
    # Bisection method for inverse CDF
    low, high = -100.0, 100.0
    mid = 0.0
    for _ in range(100):
        mid = (low + high) / 2.0
        cdf_val = float(student_t_cdf(mid, dof))
        if abs(cdf_val - p) < 1e-10:
            break
        if cdf_val < p:
            low = mid
        else:
            high = mid
    return mid


class TheoryCompute:
    """Handles pure numerical computation for theory calculations.

    This class extracts computation functionality from QTheory
    to reduce the god class size and improve maintainability.
    All methods use JAX for numerical computation.

    Attributes:
        logger: Logger for this compute module.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initialize TheoryCompute.

        Args:
            logger: Optional logger instance.
        """
        self.logger = logger or logging.getLogger(__name__)

    def calculate_residuals(
        self,
        y_data: Array,
        y_theory: Array,
    ) -> Array:
        """Calculate residuals between data and theory.

        Args:
            y_data: Observed data values.
            y_theory: Theoretical predictions.

        Returns:
            Array of residuals (y_data - y_theory).
        """
        return y_data - y_theory

    def calculate_chi_squared(
        self,
        residuals: Array,
        weights: Array | None = None,
    ) -> float:
        """Calculate chi-squared statistic.

        Args:
            residuals: Array of residuals.
            weights: Optional weights (1/sigma^2).

        Returns:
            Chi-squared value.
        """
        if weights is None:
            weights = jnp.ones_like(residuals)
        chi2 = jnp.sum(weights * residuals**2)
        return float(chi2)

    def calculate_error_standard(
        self,
        y_data: Array,
        y_theory: Array,
    ) -> float:
        """Calculate standard error (sum of squared residuals).

        Args:
            y_data: Observed data values.
            y_theory: Theoretical predictions.

        Returns:
            Sum of squared relative residuals.
        """
        residuals = self.calculate_residuals(y_data, y_theory)
        # Avoid division by zero
        y_safe = jnp.where(jnp.abs(y_data) < 1e-100, 1e-100, y_data)
        relative_residuals = residuals / y_safe
        return float(jnp.sum(relative_residuals**2))

    def calculate_error_abs(
        self,
        y_data: Array,
        y_theory: Array,
    ) -> float:
        """Calculate absolute error (sum of squared residuals).

        Args:
            y_data: Observed data values.
            y_theory: Theoretical predictions.

        Returns:
            Sum of squared absolute residuals.
        """
        residuals = self.calculate_residuals(y_data, y_theory)
        return float(jnp.sum(residuals**2))

    def calculate_error_log(
        self,
        y_data: Array,
        y_theory: Array,
    ) -> float:
        """Calculate logarithmic error.

        Args:
            y_data: Observed data values.
            y_theory: Theoretical predictions.

        Returns:
            Sum of squared log residuals.
        """
        # Ensure positive values for log
        y_data_pos = jnp.maximum(jnp.abs(y_data), 1e-100)
        y_theory_pos = jnp.maximum(jnp.abs(y_theory), 1e-100)
        log_residuals = jnp.log10(y_data_pos) - jnp.log10(y_theory_pos)
        return float(jnp.sum(log_residuals**2))

    def calculate_r_squared(
        self,
        y_data: Array,
        y_theory: Array,
    ) -> float:
        """Calculate R-squared (coefficient of determination).

        Args:
            y_data: Observed data values.
            y_theory: Theoretical predictions.

        Returns:
            R-squared value (0 to 1, higher is better).
        """
        ss_res = jnp.sum((y_data - y_theory) ** 2)
        ss_tot = jnp.sum((y_data - jnp.mean(y_data)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot)
        return float(r2)

    def extend_x_range(
        self,
        x_data: Array,
        xmin_ext: float,
        xmax_ext: float,
        points_per_decade: int = 20,
        log_scale: bool = True,
    ) -> Array:
        """Extend x-range for theory calculation.

        Args:
            x_data: Original x-data array.
            xmin_ext: Minimum extended x value.
            xmax_ext: Maximum extended x value.
            points_per_decade: Points per decade for log scale.
            log_scale: Whether to use log spacing.

        Returns:
            Extended x-array.
        """
        if log_scale:
            if xmin_ext <= 0:
                xmin_ext = float(jnp.min(x_data[x_data > 0]))
            if xmax_ext <= 0:
                xmax_ext = float(jnp.max(x_data[x_data > 0]))
            n_decades = jnp.log10(xmax_ext / xmin_ext)
            n_points = int(n_decades * points_per_decade)
            return jnp.logspace(
                jnp.log10(xmin_ext), jnp.log10(xmax_ext), max(n_points, 10)
            )
        else:
            n_points = max(int((xmax_ext - xmin_ext) * points_per_decade), 10)
            return jnp.linspace(xmin_ext, xmax_ext, n_points)

    def interpolate_theory(
        self,
        x_theory: Array,
        y_theory: Array,
        x_data: Array,
    ) -> Array:
        """Interpolate theory values to data x-points.

        Args:
            x_theory: Theory x-values.
            y_theory: Theory y-values.
            x_data: Data x-values to interpolate to.

        Returns:
            Interpolated y-values at x_data points.
        """
        from interpax import interp1d as jinterp1d

        # Sort theory values by x
        sort_idx = jnp.argsort(x_theory)
        x_sorted = x_theory[sort_idx]
        y_sorted = y_theory[sort_idx]

        # Interpolate
        return jinterp1d(x_data, x_sorted, y_sorted, method="cubic")

    def confidence_interval(
        self,
        n_params: int,
        n_points: int,
        residual_variance: float,
        jacobian_gram: Array,
        confidence_level: float = 0.95,
    ) -> Array:
        """Calculate parameter confidence intervals.

        Args:
            n_params: Number of parameters.
            n_points: Number of data points.
            residual_variance: Estimated residual variance.
            jacobian_gram: J^T J matrix (Gram matrix of Jacobian).
            confidence_level: Confidence level (default 95%).

        Returns:
            Array of confidence interval half-widths for each parameter.
        """
        dof = n_points - n_params
        if dof <= 0:
            self.logger.warning("Insufficient degrees of freedom for confidence interval")
            return jnp.full(n_params, jnp.inf)

        # t-value for confidence level
        alpha = 1.0 - confidence_level
        t_val = student_t_ppf(1.0 - alpha / 2.0, dof)

        # Covariance matrix estimate
        try:
            cov = jnp.linalg.inv(jacobian_gram) * residual_variance
            std_errors = jnp.sqrt(jnp.diag(cov))
            return std_errors * t_val
        except Exception as e:
            self.logger.warning(f"Could not compute confidence interval: {e}")
            return jnp.full(n_params, jnp.inf)
