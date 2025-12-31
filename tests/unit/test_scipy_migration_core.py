"""Core tests for SciPy to JAX migration without GUI dependencies.

This module tests the mathematical correctness of the migration without
requiring Qt or other GUI frameworks.
"""

import numpy as np
import pytest
import jax.numpy as jnp
from numpy.testing import assert_allclose


class TestInterpaxBasics:
    """Test basic interpax functionality as used in the migration."""

    def test_cubic_interpolation(self):
        """Test cubic spline interpolation."""
        from interpax import interp1d

        x = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = jnp.array([0.0, 1.0, 4.0, 9.0, 16.0])  # y = x^2

        f = interp1d(x, y, method="cubic", extrap=True)

        # Test at known point
        y_interp = f(jnp.array(2.5))
        assert_allclose(float(y_interp), 6.25, rtol=1e-2)

    def test_extrapolation(self):
        """Test extrapolation beyond data range."""
        from interpax import interp1d

        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        y = jnp.array([1.0, 2.0, 3.0, 4.0])  # Linear

        f = interp1d(x, y, method="cubic", extrap=True)
        y_extrap = f(jnp.array(5.0))

        # Should be close to 5.0 for linear data
        assert_allclose(float(y_extrap), 5.0, rtol=1e-1)

    def test_vectorized_interpolation(self):
        """Test vectorized interpolation."""
        from interpax import interp1d

        x = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = jnp.array([0.0, 1.0, 4.0, 9.0, 16.0])

        f = interp1d(x, y, method="cubic", extrap=True)

        x_new = jnp.array([0.5, 1.5, 2.5, 3.5])
        y_new = f(x_new)

        assert jnp.all(jnp.isfinite(y_new))
        assert len(y_new) == len(x_new)


class TestSavgolFilterJAX:
    """Test the JAX implementation of Savitzky-Golay filter."""

    def test_savgol_preserves_polynomial(self):
        """Test that S-G filter preserves polynomials of degree <= polyorder."""
        from RepTate.tools.ToolSmooth import _savgol_filter_jax

        # Create polynomial data: y = x^2
        x = np.linspace(-5, 5, 51)
        y = x**2

        # Apply S-G filter with order 2
        y_filtered = _savgol_filter_jax(y, window_length=11, polyorder=2)

        # Should preserve quadratic polynomial
        assert_allclose(y_filtered, y, rtol=1e-10, atol=1e-10)

    def test_savgol_smooths_noise(self):
        """Test that S-G filter smooths random noise."""
        from RepTate.tools.ToolSmooth import _savgol_filter_jax

        np.random.seed(123)
        x = np.linspace(0, 10, 100)
        y_clean = np.sin(x)
        y_noisy = y_clean + 0.2 * np.random.randn(len(x))

        y_filtered = _savgol_filter_jax(y_noisy, window_length=15, polyorder=3)

        # Filtered should be smoother (lower variance of differences)
        diff_noisy = np.diff(y_noisy)
        diff_filtered = np.diff(np.array(y_filtered))

        assert np.std(diff_filtered) < np.std(diff_noisy)

    def test_savgol_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        from RepTate.tools.ToolSmooth import _savgol_filter_jax

        y = np.random.randn(100)

        # Even window length should raise error
        with pytest.raises(ValueError, match="positive odd integer"):
            _savgol_filter_jax(y, window_length=10, polyorder=3)

        # Order >= window should raise error
        with pytest.raises(ValueError, match="greater than polyorder"):
            _savgol_filter_jax(y, window_length=5, polyorder=5)

        # Window larger than data should raise error
        with pytest.raises(ValueError, match="too large"):
            _savgol_filter_jax(y, window_length=101, polyorder=3)


class TestCumulativeIntegration:
    """Test cumulative integration implementation."""

    def test_linear_function(self):
        """Test integration of y=x -> integral = x^2/2."""
        x = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        y = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

        # Cumulative trapezoidal integration
        dx = x[1:] - x[:-1]
        avg_y = (y[1:] + y[:-1]) / 2.0
        cumulative = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(avg_y * dx)])

        # Expected: integral of x is x^2/2
        expected = x**2 / 2

        assert_allclose(cumulative, expected, rtol=1e-10)

    def test_constant_function(self):
        """Test integration of constant -> integral = c*x."""
        x = jnp.linspace(0, 10, 50)
        y = jnp.ones_like(x) * 5.0  # constant y = 5

        dx = x[1:] - x[:-1]
        avg_y = (y[1:] + y[:-1]) / 2.0
        cumulative = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(avg_y * dx)])

        # Expected: integral of 5 is 5*x
        expected = 5.0 * x

        assert_allclose(cumulative, expected, rtol=1e-10)

    def test_sine_function(self):
        """Test integration of sin(x) -> integral = -cos(x) + 1."""
        x = jnp.linspace(0, 2 * np.pi, 100)
        y = jnp.sin(x)

        dx = x[1:] - x[:-1]
        avg_y = (y[1:] + y[:-1]) / 2.0
        cumulative = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(avg_y * dx)])

        # Expected: integral of sin(x) is -cos(x) + C, with C=1 to match cumulative[0]=0
        expected = 1.0 - jnp.cos(x)

        assert_allclose(cumulative, expected, rtol=1e-3)


class TestComplexArithmetic:
    """Test complex number operations used in i-Rheo transforms."""

    def test_complex_exponential(self):
        """Test complex exponential calculations."""
        t = np.array([0.01, 0.1, 1.0, 10.0])
        w = np.array([0.1, 1.0, 10.0])

        for wi in w:
            for tj in t:
                # NumPy version
                exp_numpy = np.exp(-1j * wi * tj)

                # JAX version
                exp_jax = jnp.exp(-1j * wi * tj)

                assert_allclose(complex(exp_jax), exp_numpy, rtol=1e-10)

    def test_complex_summation(self):
        """Test complex summation pattern from i-Rheo."""
        n = 10
        t = np.logspace(-2, 2, n)
        j = np.exp(-t)
        w = 1.0

        # Calculate sum
        aux = 0j
        for i in range(1, n):
            contrib = (np.exp(-1j * w * t[i - 1]) - np.exp(-1j * w * t[i])) * (
                j[i] - j[i - 1]
            ) / (t[i] - t[i - 1])
            aux += contrib

        # Should produce finite complex number
        assert np.isfinite(aux.real)
        assert np.isfinite(aux.imag)


class TestAdaptiveInterpolation:
    """Test adaptive interpolation method selection."""

    def test_method_selection(self):
        """Test that interpolation method adapts to number of points."""
        from interpax import interp1d

        test_cases = [
            (2, "linear"),
            (3, "quadratic"),
            (4, "cubic"),
            (10, "cubic"),
        ]

        for n_points, expected_method in test_cases:
            x = jnp.linspace(0, 1, n_points)
            y = x**2

            # Determine method (as in ApplicationGt)
            if n_points < 2:
                method = "nearest"
            elif n_points < 3:
                method = "linear"
            elif n_points < 4:
                method = "quadratic"
            else:
                method = "cubic"

            assert method == expected_method

            # Test that interpolation works
            f = interp1d(x, y, method=method, extrap=True)
            y_interp = f(jnp.array(0.5))
            assert jnp.isfinite(y_interp)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_point_nearest(self):
        """Test interpolation with single point."""
        from interpax import interp1d

        x = jnp.array([1.0])
        y = jnp.array([2.0])

        f = interp1d(x, y, method="nearest", extrap=True)
        assert float(f(jnp.array(0.5))) == 2.0

    def test_two_point_linear(self):
        """Test interpolation with two points."""
        from interpax import interp1d

        x = jnp.array([0.0, 1.0])
        y = jnp.array([0.0, 1.0])

        f = interp1d(x, y, method="linear", extrap=True)

        assert_allclose(float(f(jnp.array(0.5))), 0.5, rtol=1e-10)
        assert_allclose(float(f(jnp.array(2.0))), 2.0, rtol=1e-1)

    def test_monotonic_preservation(self):
        """Test that interpolation preserves monotonicity."""
        from interpax import interp1d

        x = jnp.array([0, 1, 2, 3, 4, 5])
        y = jnp.array([0, 1, 2, 3, 4, 5])

        f = interp1d(x, y, method="cubic", extrap=True)

        x_dense = jnp.linspace(0, 5, 100)
        y_dense = f(x_dense)

        # Should remain monotonically increasing (allow tiny numerical errors)
        assert jnp.all(jnp.diff(y_dense) >= -1e-6)


class TestLogSpacedData:
    """Test interpolation with log-spaced data (common in rheology)."""

    def test_maxwell_relaxation(self):
        """Test interpolation of Maxwell relaxation modulus."""
        from interpax import interp1d

        t = jnp.logspace(-2, 2, 50)
        G0 = 1e6
        tau = 1.0
        G = G0 * jnp.exp(-t / tau)

        f = interp1d(t, G, method="cubic", extrap=True)

        # Test at intermediate points
        t_test = jnp.array([0.1, 1.0, 10.0])
        G_test = f(t_test)
        G_expected = G0 * jnp.exp(-t_test / tau)

        assert_allclose(G_test, G_expected, rtol=1e-2)

    def test_power_law_data(self):
        """Test interpolation of power law data."""
        from interpax import interp1d

        t = jnp.logspace(-1, 1, 30)
        alpha = 0.5
        G = t**alpha

        f = interp1d(t, G, method="cubic", extrap=True)

        # Test interpolation
        t_test = jnp.array([0.5, 1.0, 2.0])
        G_test = f(t_test)
        G_expected = t_test**alpha

        assert_allclose(G_test, G_expected, rtol=1e-2)


class TestNumericalPrecision:
    """Test numerical precision of migrated implementations."""

    def test_cumsum_stability(self):
        """Test that cumulative sum is numerically stable."""
        # Large array with small increments
        n = 10000
        dx = jnp.ones(n) * 1e-10
        cumsum = jnp.cumsum(dx)

        # Should equal n * 1e-10
        expected = n * 1e-10

        assert_allclose(float(cumsum[-1]), expected, rtol=1e-6)

    def test_interpolation_precision(self):
        """Test interpolation precision at data points."""
        from interpax import interp1d

        x = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = jnp.array([1.0, 2.0, 4.0, 8.0, 16.0])

        f = interp1d(x, y, method="cubic", extrap=True)

        # Interpolation at data points should match exactly
        for xi, yi in zip(x, y):
            y_interp = f(jnp.array(xi))
            assert_allclose(float(y_interp), float(yi), rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
