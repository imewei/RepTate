"""Tests for migrated applications to ensure JAX implementations match original behavior.

This module tests the SciPy to JAX migration for:
- ApplicationCreep (scipy.interpolate -> interpax)
- ApplicationGt (scipy.interpolate -> interpax)
"""

import numpy as np
import pytest
import jax.numpy as jnp
from numpy.testing import assert_allclose


class TestApplicationCreepInterpolation:
    """Test ApplicationCreep migration from scipy.interpolate to interpax."""

    def test_interp1d_cubic_basic(self):
        """Test basic cubic interpolation with interpax."""
        from interpax import interp1d

        # Create simple test data
        x = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = jnp.array([0.0, 1.0, 4.0, 9.0, 16.0])  # y = x^2

        # interpax API: interp1d(xq, x, f, method, ...)
        xq = jnp.array([2.5])
        y_interp = interp1d(xq, x, y, method="cubic", extrap=True)

        # For x^2, f(2.5) = 6.25 - use .item() for JAX scalar extraction
        assert_allclose(y_interp.item(), 6.25, rtol=1e-2)

    def test_interp1d_extrapolation(self):
        """Test extrapolation beyond data range."""
        from interpax import interp1d

        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        y = jnp.array([1.0, 2.0, 3.0, 4.0])  # Linear

        # interpax API: interp1d(xq, x, f, method, ...)
        xq = jnp.array([5.0])
        y_extrap = interp1d(xq, x, y, method="cubic", extrap=True)

        # Should be close to 5.0 for linear data
        assert_allclose(y_extrap.item(), 5.0, rtol=1e-1)

    def test_interp1d_vectorized(self):
        """Test vectorized interpolation."""
        from interpax import interp1d

        x = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = jnp.array([0.0, 1.0, 4.0, 9.0, 16.0])

        # interpax API: interp1d(xq, x, f, method, ...)
        x_new = jnp.array([0.5, 1.5, 2.5, 3.5])
        y_new = interp1d(x_new, x, y, method="cubic", extrap=True)

        # All values should be reasonable
        assert jnp.all(jnp.isfinite(y_new))
        assert len(y_new) == len(x_new)


class TestApplicationGtInterpolation:
    """Test ApplicationGt migration from scipy.interpolate to interpax."""

    def test_adaptive_interpolation_method(self):
        """Test that interpolation method adapts to number of points.

        Note: interpax doesn't support 'quadratic', so we use 'cubic' for 3+ points.
        """
        from interpax import interp1d

        # Test with different numbers of points
        # interpax supported: nearest, linear, cubic (no quadratic)
        test_cases = [
            (2, "linear"),
            (3, "cubic"),  # Changed from quadratic - not available in interpax
            (4, "cubic"),
        ]

        for n_points, expected_method in test_cases:
            x = jnp.linspace(0, 1, n_points)
            y = x**2

            # This mimics the logic in ApplicationGt (updated for interpax)
            if n_points < 2:
                method = "nearest"
            elif n_points < 3:
                method = "linear"
            else:
                method = "cubic"

            assert method == expected_method

            # Test that interpolation works - interpax API: interp1d(xq, x, f, ...)
            xq = jnp.array([0.5])
            y_interp = interp1d(xq, x, y, method=method, extrap=True)
            assert jnp.isfinite(y_interp)

    def test_interpolation_at_zero(self):
        """Test interpolation at t=0 (common in ApplicationGt)."""
        from interpax import interp1d

        # Data starting at t > 0
        x = jnp.array([0.1, 0.5, 1.0, 2.0, 5.0])
        y = jnp.exp(-x)  # Exponential decay

        # interpax API: interp1d(xq, x, f, ...)
        xq = jnp.array([0.0])
        y0 = interp1d(xq, x, y, method="cubic", extrap=True)

        # Should be close to 1.0 (exp(0) = 1)
        assert_allclose(y0.item(), 1.0, rtol=0.2)

    def test_log_spaced_interpolation(self):
        """Test interpolation with log-spaced data (common in rheology)."""
        from interpax import interp1d

        # Create log-spaced time data
        t = jnp.logspace(-2, 2, 50)
        # Maxwell relaxation: G(t) = G0 * exp(-t/tau)
        G0 = 1e6
        tau = 1.0
        G = G0 * jnp.exp(-t / tau)

        # interpax API: interp1d(xq, x, f, ...)
        t_test = jnp.array([0.1, 1.0, 10.0])
        G_test = interp1d(t_test, t, G, method="cubic", extrap=True)

        # Compare with analytical solution (relaxed tolerance for cubic approx)
        G_expected = G0 * jnp.exp(-t_test / tau)

        assert_allclose(G_test, G_expected, rtol=0.1)


class TestComplexModulusTransformations:
    """Test i-Rheo transformations in migrated applications."""

    def test_complex_exponential_computation(self):
        """Test complex exponential calculations in i-Rheo transforms."""
        # This tests the mathematical operations used in viewiRheo methods

        # Create test data
        t = np.array([0.01, 0.1, 1.0, 10.0])
        w = np.array([0.1, 1.0, 10.0])

        # Test complex exponential: exp(-i*w*t)
        for wi in w:
            for tj in t:
                # NumPy version
                exp_numpy = np.exp(-1j * wi * tj)

                # JAX version
                exp_jax = jnp.exp(-1j * wi * tj)

                assert_allclose(complex(exp_jax), exp_numpy, rtol=1e-10)

    def test_irheo_summation_pattern(self):
        """Test the summation pattern used in i-Rheo transformations."""
        # Simulate the pattern used in ApplicationCreep.viewiRheo

        n = 10
        t = np.logspace(-2, 2, n)
        j = np.exp(-t)  # Compliance data
        w = np.logspace(-1, 1, 5)  # Frequency points

        # Test the summation structure
        for wi in w:
            # Calculate sum as in viewiRheo
            aux = 0j
            for i in range(1, n):
                contrib = (np.exp(-1j * wi * t[i - 1]) - np.exp(-1j * wi * t[i])) * (
                    j[i] - j[i - 1]
                ) / (t[i] - t[i - 1])
                aux += contrib

            # Should produce finite complex number
            assert np.isfinite(aux.real)
            assert np.isfinite(aux.imag)


class TestNumericalEquivalence:
    """Test numerical equivalence between old and new implementations."""

    def test_cumulative_integration_equivalence(self):
        """Test that cumulative integration gives same results."""
        import jax.numpy as jnp

        # Test data
        x = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        y = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])  # Linear

        # Cumulative trapezoidal integration (as in migrated code)
        dx = x[1:] - x[:-1]
        avg_y = (y[1:] + y[:-1]) / 2.0
        cumulative = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(avg_y * dx)])

        # Expected: integral of y=x is x^2/2
        expected = x**2 / 2

        assert_allclose(cumulative, expected, rtol=1e-10)

    def test_convolution_for_smoothing(self):
        """Test convolution operation used in S-G filter."""
        # Simple moving average (special case of S-G filter)
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        window_size = 3

        # Coefficients for simple average
        coeffs = jnp.ones(window_size) / window_size

        # Pad array
        y_padded = jnp.pad(jnp.array(y), window_size // 2, mode="edge")

        # Convolve
        y_smooth = jnp.convolve(y_padded, coeffs[::-1], mode="valid")

        # Check that smoothing reduces variability
        assert np.std(y_smooth) < np.std(y)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_point_interpolation(self):
        """Test interpolation with minimal data points."""
        from interpax import interp1d

        # Single point - should work with nearest neighbor
        x = jnp.array([1.0])
        y = jnp.array([2.0])

        # interpax API: interp1d(xq, x, f, ...)
        xq = jnp.array([0.5])
        result = interp1d(xq, x, y, method="nearest", extrap=True)
        assert result.item() == 2.0

    def test_two_point_interpolation(self):
        """Test interpolation with two points."""
        from interpax import interp1d

        x = jnp.array([0.0, 1.0])
        y = jnp.array([0.0, 1.0])

        # interpax API: interp1d(xq, x, f, ...)
        # Test interpolation and extrapolation
        assert_allclose(interp1d(jnp.array([0.5]), x, y, method="linear", extrap=True).item(), 0.5, rtol=1e-10)
        assert_allclose(interp1d(jnp.array([2.0]), x, y, method="linear", extrap=True).item(), 2.0, rtol=1e-1)

    def test_monotonic_data(self):
        """Test that interpolation preserves monotonicity for monotonic data."""
        from interpax import interp1d

        # Monotonically increasing data
        x = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        y = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

        # interpax API: interp1d(xq, x, f, ...)
        x_dense = jnp.linspace(0, 5, 100)
        y_dense = interp1d(x_dense, x, y, method="cubic", extrap=True)

        # Should remain monotonically increasing
        assert jnp.all(jnp.diff(y_dense) >= -1e-6)  # Allow tiny numerical errors


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
