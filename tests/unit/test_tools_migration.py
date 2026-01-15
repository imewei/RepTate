"""Tests for migrated tools to ensure JAX implementations match original behavior.

This module tests the SciPy to JAX migration for:
- ToolIntegral (scipy.integrate.odeint -> JAX cumulative integration)
- ToolInterpolate (scipy.interpolate.interp1d -> interpax)
- ToolSmooth (scipy.signal.savgol_filter -> JAX implementation)

Tool classes (ToolIntegral, ToolSmooth, ToolInterpolate) inherit from QTool
which requires a Qt application. The TestTool* classes are marked as GUI tests.
The TestSavgolFilterJAX tests the pure JAX function without Qt dependencies.
"""

import numpy as np
import pytest
import jax.numpy as jnp
from numpy.testing import assert_allclose


@pytest.mark.gui
class TestToolIntegral:
    """Test ToolIntegral migration from scipy to JAX."""

    def test_simple_integration(self):
        """Test cumulative integration of a simple function."""
        # Create simple test data: y = x
        x = np.linspace(0, 10, 100)
        y = x

        # Expected: integral of x is x^2/2
        expected_integral = x**2 / 2

        # Import and test the tool
        from RepTate.tools.ToolIntegral import ToolIntegral

        # Create minimal mock parent application
        class MockApp:
            def update_all_ds_plots(self):
                pass

        tool = ToolIntegral(parent_app=MockApp())
        x_out, y_out = tool.calculate(x, y)

        # Check that output matches expected integral (within tolerance)
        assert_allclose(y_out, expected_integral, rtol=1e-2)

    def test_integration_with_duplicates(self):
        """Test that duplicate x values are handled correctly."""
        x = np.array([0, 1, 1, 2, 3, 3, 4])
        y = np.array([0, 1, 1, 2, 3, 3, 4])

        from RepTate.tools.ToolIntegral import ToolIntegral

        class MockApp:
            def update_all_ds_plots(self):
                pass

        tool = ToolIntegral(parent_app=MockApp())
        x_out, y_out = tool.calculate(x, y)

        # Should have unique x values
        assert len(np.unique(x_out)) == len(x_out)
        # Integral should be monotonically increasing
        assert np.all(np.diff(y_out) >= 0)


@pytest.mark.gui
class TestToolInterpolate:
    """Test ToolInterpolate migration from scipy to JAX."""

    def test_interpolation_linear(self):
        """Test interpolation at an interior point."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([0.0, 1.0, 4.0, 9.0, 16.0])  # y = x^2

        from RepTate.tools.ToolInterpolate import ToolInterpolateExtrapolate

        class MockApp:
            def update_all_ds_plots(self):
                pass

        tool = ToolInterpolateExtrapolate(parent_app=MockApp())
        tool.parameters["x"].value = 2.5

        # Mock Qprint to capture output
        output = []
        tool.Qprint = lambda x: output.append(x)

        x_out, y_out = tool.calculate(x, y)

        # Check that output was generated
        assert len(output) > 0
        # Check that x and y are unchanged
        assert np.array_equal(x_out, x)
        assert np.array_equal(y_out, y)

    def test_extrapolation(self):
        """Test extrapolation beyond data range."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([1.0, 4.0, 9.0, 16.0])  # y = x^2

        from RepTate.tools.ToolInterpolate import ToolInterpolateExtrapolate

        class MockApp:
            def update_all_ds_plots(self):
                pass

        tool = ToolInterpolateExtrapolate(parent_app=MockApp())
        tool.parameters["x"].value = 5.0

        output = []
        tool.Qprint = lambda x: output.append(x)

        x_out, y_out = tool.calculate(x, y)

        # Should successfully extrapolate (no error)
        assert len(output) > 0


@pytest.mark.gui
class TestToolSmooth:
    """Test ToolSmooth migration from scipy.signal to JAX."""

    def test_smoothing_reduces_noise(self):
        """Test that smoothing reduces high-frequency noise."""
        # Create noisy signal
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y_clean = np.sin(x)
        y_noisy = y_clean + 0.1 * np.random.randn(len(x))

        from RepTate.tools.ToolSmooth import ToolSmooth

        class MockApp:
            def update_all_ds_plots(self):
                pass

        tool = ToolSmooth(parent_app=MockApp())
        tool.parameters["window"].value = 11
        tool.parameters["order"].value = 3

        x_out, y_smooth = tool.calculate(x, y_noisy)

        # Smoothed signal should be closer to clean signal than noisy signal
        error_noisy = np.mean((y_noisy - y_clean) ** 2)
        error_smooth = np.mean((y_smooth - y_clean) ** 2)

        assert error_smooth < error_noisy

    def test_invalid_window_parameter(self):
        """Test that invalid window parameter is rejected."""
        x = np.linspace(0, 10, 100)
        y = np.sin(x)

        from RepTate.tools.ToolSmooth import ToolSmooth

        class MockApp:
            def update_all_ds_plots(self):
                pass

        tool = ToolSmooth(parent_app=MockApp())
        tool.parameters["window"].value = 10  # Even number (invalid)
        tool.parameters["order"].value = 3

        output = []
        tool.Qprint = lambda x: output.append(x)

        x_out, y_out = tool.calculate(x, y)

        # Should return original data and print error
        assert np.array_equal(y_out, y)
        assert any("Invalid window" in str(msg) for msg in output)

    def test_window_larger_than_data(self):
        """Test that window larger than data is rejected."""
        x = np.array([0, 1, 2, 3, 4])
        y = np.array([0, 1, 4, 9, 16])

        from RepTate.tools.ToolSmooth import ToolSmooth

        class MockApp:
            def update_all_ds_plots(self):
                pass

        tool = ToolSmooth(parent_app=MockApp())
        tool.parameters["window"].value = 11
        tool.parameters["order"].value = 3

        output = []
        tool.Qprint = lambda x: output.append(x)

        x_out, y_out = tool.calculate(x, y)

        # Should return original data and print error
        assert np.array_equal(y_out, y)
        assert any("Invalid window" in str(msg) for msg in output)


class TestSavgolFilterJAX:
    """Test the JAX implementation of Savitzky-Golay filter."""

    def test_savgol_preserves_polynomial(self):
        """Test that S-G filter preserves polynomials of degree <= polyorder.

        Note: S-G filters have boundary effects, so we only compare interior
        points where the full window can be applied.
        """
        from RepTate.tools.ToolSmooth import _savgol_filter_jax

        # Create polynomial data: y = x^2
        x = np.linspace(-5, 5, 51)
        y = x**2

        # Apply S-G filter with order 2
        window_length = 11
        y_filtered = _savgol_filter_jax(y, window_length=window_length, polyorder=2)

        # Should preserve quadratic polynomial in interior (exclude boundary effects)
        # The boundary region is half the window size on each end
        half_window = window_length // 2
        interior = slice(half_window, -half_window)
        assert_allclose(y_filtered[interior], y[interior], rtol=1e-10, atol=1e-10)

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
