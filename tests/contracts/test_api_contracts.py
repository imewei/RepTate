"""API contracts validating component integration agreements.

Contract Tests:
- C001: Theory.calculate() input/output contract
- C002: Theory parameter access contract
- C003: Dataset data access contract
- C004: Application theory loading contract

These tests ensure that APIs conform to their documented contracts,
enabling reliable component composition without knowing implementation details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

if TYPE_CHECKING:
    from tests.conftest import MockTheory, SyntheticData


class TestTheoryCalculateContract:
    """Contract tests for Theory.calculate() API.

    The calculate() method must accept:
    - params: dict[str, float] with theory-specific parameter names
    - x: Array with shape (n,)

    And return:
    - Array with same shape as x
    - No NaN unless theoretically required
    """

    def test_calculate_accepts_dict_params(
        self,
        mock_theory: MockTheory,
        synthetic_frequency_data: SyntheticData,
    ) -> None:
        """Contract: calculate() accepts dict[str, float] params."""
        # Arrange
        params = {"slope": 2.0, "intercept": 1.0}
        x = synthetic_frequency_data.x

        # Act
        result = mock_theory.calculate(params, x)

        # Assert: Result is array
        assert isinstance(result, Array)
        assert result.shape == x.shape

    def test_calculate_param_types_enforced(
        self,
        mock_theory: MockTheory,
        synthetic_frequency_data: SyntheticData,
    ) -> None:
        """Contract: calculate() requires float parameter values."""
        # Arrange
        x = synthetic_frequency_data.x

        # Act & Assert: Invalid parameter types should raise
        with pytest.raises((TypeError, ValueError)):
            mock_theory.calculate({"slope": "invalid", "intercept": 1.0}, x)

    def test_calculate_output_shape_matches_input(
        self,
        mock_theory: MockTheory,
    ) -> None:
        """Contract: calculate() output shape matches input shape."""
        test_cases = [
            jnp.array([1.0, 2.0, 3.0]),  # 3 elements
            jnp.logspace(-2, 2, 50),  # 50 elements
            jnp.array([1.0]),  # Single element
        ]

        params = {"slope": 2.0, "intercept": 1.0}

        for x in test_cases:
            result = mock_theory.calculate(params, x)
            assert result.shape == x.shape, (
                f"Output shape {result.shape} != input shape {x.shape}"
            )

    def test_calculate_output_dtype_preserved(
        self,
        mock_theory: MockTheory,
        synthetic_frequency_data: SyntheticData,
    ) -> None:
        """Contract: calculate() preserves dtype (float64)."""
        params = {"slope": 2.0, "intercept": 1.0}
        x = synthetic_frequency_data.x

        result = mock_theory.calculate(params, x)

        # Output should be float64
        assert "float" in str(result.dtype).lower()

    def test_calculate_handles_empty_array(
        self,
        mock_theory: MockTheory,
    ) -> None:
        """Contract: calculate() handles empty arrays gracefully."""
        params = {"slope": 2.0, "intercept": 1.0}
        x = jnp.array([])

        result = mock_theory.calculate(params, x)

        assert result.shape == (0,)

    def test_calculate_deterministic(
        self,
        mock_theory: MockTheory,
        synthetic_frequency_data: SyntheticData,
    ) -> None:
        """Contract: calculate() is deterministic (same input = same output)."""
        params = {"slope": 2.0, "intercept": 1.0}
        x = synthetic_frequency_data.x

        result1 = mock_theory.calculate(params, x)
        result2 = mock_theory.calculate(params, x)

        np.testing.assert_array_equal(result1, result2)

    def test_calculate_boundary_parameter_values(
        self,
        mock_theory: MockTheory,
        synthetic_frequency_data: SyntheticData,
    ) -> None:
        """Contract: calculate() works with boundary parameter values."""
        x = synthetic_frequency_data.x

        boundary_cases = [
            {"slope": 0.0, "intercept": 0.0},
            {"slope": 1e-10, "intercept": 1e10},
            {"slope": -1e5, "intercept": 1e5},
        ]

        for params in boundary_cases:
            result = mock_theory.calculate(params, x)
            assert result.shape == x.shape
            assert jnp.all(jnp.isfinite(result))


class TestTheoryParameterContract:
    """Contract tests for Theory parameter interface.

    Theories must provide access to parameters with:
    - name: str
    - value: float
    - min/max bounds
    - type information
    """

    def test_get_parameters_returns_dict(self, mock_theory: MockTheory) -> None:
        """Contract: get_parameters() returns dict[str, Parameter]."""
        params = mock_theory.get_parameters()
        assert isinstance(params, dict)

    def test_parameter_has_name(self, mock_theory: MockTheory) -> None:
        """Contract: Each parameter has a name attribute."""
        params = mock_theory.get_parameters()
        for param in params.values():
            assert hasattr(param, "name")
            assert isinstance(param.name, str)

    def test_parameter_has_value(self, mock_theory: MockTheory) -> None:
        """Contract: Each parameter has a value attribute."""
        params = mock_theory.get_parameters()
        for param in params.values():
            assert hasattr(param, "value")
            assert isinstance(param.value, (int, float))

    def test_parameter_has_bounds(self, mock_theory: MockTheory) -> None:
        """Contract: Each parameter has min/max bounds."""
        params = mock_theory.get_parameters()
        for param in params.values():
            assert hasattr(param, "min_value")
            assert hasattr(param, "max_value")

    def test_parameter_bounds_consistency(
        self,
        mock_theory: MockTheory,
    ) -> None:
        """Contract: min_value <= value <= max_value."""
        params = mock_theory.get_parameters()
        for param in params.values():
            assert param.min_value <= param.value <= param.max_value


class TestDatasetDataAccessContract:
    """Contract tests for Dataset data access API.

    Datasets must provide:
    - get_x(): Array
    - get_y(): Array
    - get_column(name): Array
    - get_error(): Array | None
    """

    def test_get_x_returns_array(self, mock_dataset) -> None:
        """Contract: get_x() returns Array."""
        x = mock_dataset.get_x()
        assert isinstance(x, Array)

    def test_get_y_returns_array(self, mock_dataset) -> None:
        """Contract: get_y() returns Array."""
        y = mock_dataset.get_y()
        assert isinstance(y, Array)

    def test_x_y_same_length(self, mock_dataset) -> None:
        """Contract: x and y have matching lengths."""
        x = mock_dataset.get_x()
        y = mock_dataset.get_y()
        assert len(x) == len(y)

    def test_get_column_returns_array(self, mock_dataset) -> None:
        """Contract: get_column() returns Array for existing columns."""
        x = mock_dataset.get_column("x")
        assert isinstance(x, Array)

    def test_get_column_raises_on_missing(self, mock_dataset) -> None:
        """Contract: get_column() raises for non-existent columns."""
        with pytest.raises((KeyError, ValueError)):
            mock_dataset.get_column("nonexistent")

    def test_get_error_returns_optional(self, mock_dataset) -> None:
        """Contract: get_error() returns Array | None."""
        error = mock_dataset.get_error()
        assert error is None or isinstance(error, Array)


class TestApplicationTheoryContract:
    """Contract tests for Application theory loading.

    Applications must:
    - get_theories(): list[type] (theory classes, not instances)
    - Each theory class is instantiable
    - Theories implement ITheory protocol
    """

    def test_get_theories_returns_list(self, mock_application) -> None:
        """Contract: get_theories() returns list of theory types."""
        theories = mock_application.get_theories()
        assert isinstance(theories, list)

    def test_theories_are_classes(self, mock_application) -> None:
        """Contract: get_theories() returns classes (not instances)."""
        theories = mock_application.get_theories()
        for theory_class in theories:
            assert isinstance(theory_class, type)

    def test_theories_are_instantiable(self, mock_application) -> None:
        """Contract: Theory classes can be instantiated."""
        theories = mock_application.get_theories()
        for theory_class in theories:
            instance = theory_class()
            assert instance is not None

    def test_theory_instances_implement_contract(
        self,
        mock_application,
    ) -> None:
        """Contract: Theory instances implement ITheory."""
        from RepTate.core.interfaces import ITheory

        theories = mock_application.get_theories()
        for theory_class in theories:
            instance = theory_class()
            # Check that instance has required methods
            assert hasattr(instance, "calculate")
            assert hasattr(instance, "get_parameters")
            assert callable(instance.calculate)
            assert callable(instance.get_parameters)


class TestFitResultContract:
    """Contract tests for fit result data structure.

    Fit results must provide:
    - parameters: dict[str, float]
    - covariance: Array | None
    - residuals: Array
    - success: bool
    - get_uncertainty(param_name): float
    """

    def test_fit_result_has_parameters(self, mock_fit_result) -> None:
        """Contract: FitResult has parameters dict."""
        assert hasattr(mock_fit_result, "parameters")
        assert isinstance(mock_fit_result.parameters, dict)

    def test_fit_result_parameters_are_floats(self, mock_fit_result) -> None:
        """Contract: All parameter values are floats."""
        for name, value in mock_fit_result.parameters.items():
            assert isinstance(name, str)
            assert isinstance(value, (int, float))

    def test_fit_result_has_covariance(self, mock_fit_result) -> None:
        """Contract: FitResult has optional covariance matrix."""
        assert hasattr(mock_fit_result, "covariance")
        cov = mock_fit_result.covariance
        assert cov is None or isinstance(cov, Array)

    def test_fit_result_has_residuals(self, mock_fit_result) -> None:
        """Contract: FitResult has residuals array."""
        assert hasattr(mock_fit_result, "residuals")
        assert isinstance(mock_fit_result.residuals, Array)

    def test_fit_result_has_success_flag(self, mock_fit_result) -> None:
        """Contract: FitResult has success boolean."""
        assert hasattr(mock_fit_result, "success")
        assert isinstance(mock_fit_result.success, (bool, np.bool_))

    def test_fit_result_covariance_shape_matches_params(
        self,
        mock_fit_result,
    ) -> None:
        """Contract: Covariance matrix shape matches parameter count."""
        if mock_fit_result.covariance is not None:
            n_params = len(mock_fit_result.parameters)
            assert mock_fit_result.covariance.shape == (n_params, n_params)

    def test_fit_result_get_uncertainty(self, mock_fit_result) -> None:
        """Contract: get_uncertainty() returns float for valid parameters."""
        param_names = list(mock_fit_result.parameters.keys())
        if param_names:
            uncertainty = mock_fit_result.get_uncertainty(param_names[0])
            assert isinstance(uncertainty, (int, float))
            assert uncertainty >= 0.0

    def test_fit_result_get_uncertainty_raises_on_invalid(
        self,
        mock_fit_result,
    ) -> None:
        """Contract: get_uncertainty() raises for invalid parameter names."""
        with pytest.raises((KeyError, ValueError)):
            mock_fit_result.get_uncertainty("nonexistent_param")
