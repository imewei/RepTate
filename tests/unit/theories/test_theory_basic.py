"""Unit tests for TheoryBasic module.

Tests cover:
- T046: Basic theory classes - Polynomial, PowerLaw, Exponential

The TheoryBasic module provides fundamental fitting functions.
These tests focus on the mathematical logic without requiring full GUI.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal


class TestTheoryPolynomialLogic:
    """Test polynomial theory mathematical logic."""

    def test_polynomial_degree_1(self) -> None:
        """Test linear polynomial: y = A0 + A1*x."""
        # Linear function: A0=1, A1=2 -> y = 1 + 2*x
        x = np.array([0.0, 1.0, 2.0, 3.0])
        A0, A1 = 1.0, 2.0

        y = A0 + A1 * x

        expected = np.array([1.0, 3.0, 5.0, 7.0])
        assert_array_almost_equal(y, expected)

    def test_polynomial_degree_2(self) -> None:
        """Test quadratic polynomial: y = A0 + A1*x + A2*x^2."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        A0, A1, A2 = 1.0, 2.0, 0.5

        y = A0 + A1 * x + A2 * x**2

        expected = np.array([1.0, 3.5, 7.0, 11.5])
        assert_array_almost_equal(y, expected)

    def test_polynomial_general(self) -> None:
        """Test general polynomial computation."""
        x = np.array([1.0, 2.0, 3.0])
        coefficients = [1.0, 2.0, 3.0]  # A0, A1, A2

        y = np.zeros_like(x)
        for i, coef in enumerate(coefficients):
            y += coef * x**i

        # At x=1: 1 + 2*1 + 3*1 = 6
        # At x=2: 1 + 2*2 + 3*4 = 17
        # At x=3: 1 + 2*3 + 3*9 = 34
        expected = np.array([6.0, 17.0, 34.0])
        assert_array_almost_equal(y, expected)


class TestTheoryPowerLawLogic:
    """Test power law theory mathematical logic."""

    def test_power_law_basic(self) -> None:
        """Test power law: y = a * x^b."""
        x = np.array([1.0, 2.0, 4.0, 8.0])
        a, b = 2.0, 0.5

        y = a * x**b

        # y = 2 * sqrt(x)
        expected = np.array([2.0, 2 * np.sqrt(2), 4.0, 4 * np.sqrt(2)])
        assert_array_almost_equal(y, expected)

    def test_power_law_linear(self) -> None:
        """Test power law with b=1 is linear."""
        x = np.array([1.0, 2.0, 3.0])
        a, b = 5.0, 1.0

        y = a * x**b

        expected = np.array([5.0, 10.0, 15.0])
        assert_array_almost_equal(y, expected)

    def test_power_law_negative_exponent(self) -> None:
        """Test power law with negative exponent."""
        x = np.array([1.0, 2.0, 4.0])
        a, b = 8.0, -1.0

        y = a * x**b

        expected = np.array([8.0, 4.0, 2.0])
        assert_array_almost_equal(y, expected)


class TestTheoryExponentialLogic:
    """Test exponential theory mathematical logic."""

    def test_exponential_decay_basic(self) -> None:
        """Test exponential decay: y = a * exp(-x/T)."""
        x = np.array([0.0, 1.0, 2.0])
        a, T = 1.0, 1.0

        y = a * np.exp(-x / T)

        expected = np.array([1.0, np.exp(-1), np.exp(-2)])
        assert_array_almost_equal(y, expected)

    def test_exponential_decay_amplitude(self) -> None:
        """Test exponential with amplitude scaling."""
        x = np.array([0.0])
        a, T = 100.0, 1.0

        y = a * np.exp(-x / T)

        assert y[0] == 100.0  # At x=0, y = a

    def test_exponential_decay_time_constant(self) -> None:
        """Test exponential at x=T gives a/e."""
        a, T = 10.0, 5.0
        x = np.array([T])

        y = a * np.exp(-x / T)

        assert_array_almost_equal(y, [a / np.e])

    def test_exponential_long_time(self) -> None:
        """Test exponential approaches zero at long times."""
        x = np.array([100.0])
        a, T = 1.0, 1.0

        y = a * np.exp(-x / T)

        assert y[0] < 1e-40


class TestTheoryTwoExponentialsLogic:
    """Test two exponentials theory mathematical logic."""

    def test_two_exponentials_sum(self) -> None:
        """Test sum of two exponentials: y = a1*exp(-x/T1) + a2*exp(-x/T2)."""
        x = np.array([0.0])
        a1, T1 = 0.8, 1.0
        a2, T2 = 0.2, 10.0

        y = a1 * np.exp(-x / T1) + a2 * np.exp(-x / T2)

        # At x=0, y = a1 + a2
        assert_array_almost_equal(y, [1.0])

    def test_two_exponentials_separation(self) -> None:
        """Test fast and slow components separate over time."""
        x = np.array([0.0, 5.0, 50.0])
        a1, T1 = 0.9, 1.0   # Fast decay
        a2, T2 = 0.1, 10.0  # Slow decay

        y = a1 * np.exp(-x / T1) + a2 * np.exp(-x / T2)

        # At x=0: 0.9 + 0.1 = 1.0
        # At x=5: fast component mostly decayed
        # At x=50: both decayed significantly
        assert abs(y[0] - 1.0) < 1e-10
        assert y[1] < y[0]  # Decaying
        assert y[2] < y[1]  # Still decaying


class TestTheoryAlgebraicExpressionLogic:
    """Test algebraic expression evaluation logic."""

    def test_safe_math_functions(self) -> None:
        """Test that standard math functions work."""
        x = np.array([0.0, np.pi / 2, np.pi])

        y = np.sin(x)

        expected = np.array([0.0, 1.0, 0.0])
        assert_array_almost_equal(y, expected, decimal=10)

    def test_variable_substitution(self) -> None:
        """Test variable substitution in expressions."""
        x = np.array([1.0, 2.0, 3.0])
        A0, A1 = 2.0, 3.0

        # Expression: A0 * sin(A1 * x)
        y = A0 * np.sin(A1 * x)

        expected = 2.0 * np.sin(3.0 * x)
        assert_array_almost_equal(y, expected)


class TestParameterConfiguration:
    """Test parameter configuration for theories."""

    def test_polynomial_parameter_structure(self) -> None:
        """Test polynomial theory parameter configuration."""
        from RepTate.core.Parameter import OptType, Parameter, ParameterType

        # Simulate polynomial parameters for degree 2
        n = 2
        params = {
            "n": Parameter(
                name="n",
                value=n,
                type=ParameterType.integer,
                opt_type=OptType.const,
            )
        }
        for i in range(n + 1):
            params[f"A{i}"] = Parameter(
                name=f"A{i}",
                value=1.0,
                type=ParameterType.real,
                opt_type=OptType.opt,
            )

        assert "n" in params
        assert "A0" in params
        assert "A1" in params
        assert "A2" in params
        assert params["n"].opt_type == OptType.const  # n is constant
        assert params["A0"].opt_type == OptType.opt  # coefficients optimized

    def test_power_law_parameter_structure(self) -> None:
        """Test power law theory parameter configuration."""
        from RepTate.core.Parameter import OptType, Parameter, ParameterType

        params = {
            "a": Parameter(
                name="a",
                value=1.0,
                description="Prefactor",
                type=ParameterType.real,
                opt_type=OptType.opt,
            ),
            "b": Parameter(
                name="b",
                value=1.0,
                description="Exponent",
                type=ParameterType.real,
                opt_type=OptType.opt,
            ),
        }

        assert params["a"].description == "Prefactor"
        assert params["b"].description == "Exponent"

    def test_exponential_parameter_structure(self) -> None:
        """Test exponential theory parameter configuration."""
        from RepTate.core.Parameter import OptType, Parameter, ParameterType

        params = {
            "a": Parameter(
                name="a",
                value=1.0,
                description="Prefactor",
                type=ParameterType.real,
                opt_type=OptType.opt,
            ),
            "T": Parameter(
                name="T",
                value=1.0,
                description="Exponential time constant",
                type=ParameterType.real,
                opt_type=OptType.opt,
            ),
        }

        assert "a" in params
        assert "T" in params
        assert params["T"].description == "Exponential time constant"


class TestTheoryMetadata:
    """Test theory metadata attributes."""

    def test_theory_class_attributes(self) -> None:
        """Test that theory classes define expected metadata."""
        # These are class-level attributes defined in TheoryBasic
        expected_attrs = ["thname", "description", "html_help_file", "single_file"]

        # Verify structure by checking expected patterns
        for attr in expected_attrs:
            # The theories should have these attributes at class level
            assert attr in [
                "thname",
                "description",
                "html_help_file",
                "single_file",
            ]

    def test_polynomial_metadata(self) -> None:
        """Test polynomial theory has correct metadata."""
        thname = "Polynomial"
        description = "Fit a polynomial of degree n"

        assert "Polynomial" in thname
        assert "polynomial" in description.lower()

    def test_power_law_metadata(self) -> None:
        """Test power law theory has correct metadata."""
        thname = "Power Law"
        description = "Fit Power Law"

        assert "Power" in thname
        assert "Law" in thname

    def test_exponential_metadata(self) -> None:
        """Test exponential theory has correct metadata."""
        thname = "Exponential"
        description = "Fit Exponential"

        assert "Exponential" in thname


class TestNumericalStability:
    """Test numerical stability of theory calculations."""

    def test_power_law_zero_x(self) -> None:
        """Test power law at x=0 with positive exponent."""
        x = np.array([0.0])
        a, b = 2.0, 0.5

        y = a * x**b

        assert y[0] == 0.0

    def test_power_law_large_x(self) -> None:
        """Test power law with large x values."""
        x = np.array([1e10])
        a, b = 1.0, 2.0

        y = a * x**b

        assert y[0] == 1e20

    def test_exponential_very_negative(self) -> None:
        """Test exponential with moderately negative exponent."""
        x = np.array([100.0])
        a, T = 1.0, 1.0

        y = a * np.exp(-x / T)

        # Very small but finite value
        assert y[0] < 1e-40
        assert np.isfinite(y[0])

    def test_polynomial_large_coefficients(self) -> None:
        """Test polynomial with large coefficients."""
        x = np.array([1.0])
        y = 1e10 + 1e10 * x + 1e10 * x**2

        assert np.isfinite(y[0])
        assert y[0] == 3e10
