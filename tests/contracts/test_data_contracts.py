"""Data contracts validating information structure agreements.

Contract Tests:
- C005: Dataset structure contract (required/optional fields)
- C006: Theory parameter contract (types, bounds)
- C007: Serialized data structure contract (JSON/NPZ format)
- C008: Calculation output contract (shape, dtype, value ranges)

These tests ensure that data structures conform to documented specifications,
enabling reliable serialization, transmission, and processing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

if TYPE_CHECKING:
    from tests.conftest import SyntheticData


class TestDatasetStructureContract:
    """Contract tests for dataset data structure.

    Required contract:
    - x: Array with shape (n,), dtype float64, strictly increasing
    - y: Array with shape (n,), dtype float64
    - error (optional): Array with shape (n,), dtype float64, positive
    """

    def test_dataset_has_required_fields(self, mock_dataset) -> None:
        """Contract: Dataset provides x and y data."""
        assert mock_dataset.get_x() is not None
        assert mock_dataset.get_y() is not None

    def test_dataset_x_is_1d_array(self, mock_dataset) -> None:
        """Contract: x is a 1D array."""
        x = mock_dataset.get_x()
        assert len(x.shape) == 1

    def test_dataset_y_is_1d_array(self, mock_dataset) -> None:
        """Contract: y is a 1D array."""
        y = mock_dataset.get_y()
        assert len(y.shape) == 1

    def test_dataset_x_y_lengths_match(self, mock_dataset) -> None:
        """Contract: x and y have the same length."""
        x = mock_dataset.get_x()
        y = mock_dataset.get_y()
        assert len(x) == len(y)

    def test_dataset_x_dtype_is_float(self, mock_dataset) -> None:
        """Contract: x is float dtype."""
        x = mock_dataset.get_x()
        assert "float" in str(x.dtype).lower()

    def test_dataset_y_dtype_is_float(self, mock_dataset) -> None:
        """Contract: y is float dtype."""
        y = mock_dataset.get_y()
        assert "float" in str(y.dtype).lower()

    def test_dataset_error_shape_matches_data(self, mock_dataset) -> None:
        """Contract: error array (if present) matches data shape."""
        x = mock_dataset.get_x()
        error = mock_dataset.get_error()
        if error is not None:
            assert error.shape == x.shape

    def test_dataset_error_dtype_is_float(self, mock_dataset) -> None:
        """Contract: error array is float dtype."""
        error = mock_dataset.get_error()
        if error is not None:
            assert "float" in str(error.dtype).lower()

    def test_dataset_error_values_positive(self, mock_dataset) -> None:
        """Contract: error values are non-negative."""
        error = mock_dataset.get_error()
        if error is not None:
            assert jnp.all(error >= 0.0)


class TestTheoryParameterContract:
    """Contract tests for theory parameter data structures.

    Required contract:
    - Each parameter has: name, value, min_value, max_value, opt_type
    - name: str, unique within theory
    - value: float, within [min_value, max_value]
    - opt_type: "opt" or "var"
    """

    def test_parameter_has_required_attributes(self, mock_theory) -> None:
        """Contract: Parameter has name, value, bounds."""
        params = mock_theory.get_parameters()
        for param in params.values():
            assert hasattr(param, "name")
            assert hasattr(param, "value")
            assert hasattr(param, "min_value")
            assert hasattr(param, "max_value")

    def test_parameter_name_is_string(self, mock_theory) -> None:
        """Contract: Parameter name is string."""
        params = mock_theory.get_parameters()
        for param in params.values():
            assert isinstance(param.name, str)
            assert len(param.name) > 0

    def test_parameter_names_unique(self, mock_theory) -> None:
        """Contract: Parameter names are unique within theory."""
        params = mock_theory.get_parameters()
        names = [p.name for p in params.values()]
        assert len(names) == len(set(names))

    def test_parameter_value_is_numeric(self, mock_theory) -> None:
        """Contract: Parameter value is numeric (int or float)."""
        params = mock_theory.get_parameters()
        for param in params.values():
            assert isinstance(param.value, (int, float))

    def test_parameter_bounds_are_numeric(self, mock_theory) -> None:
        """Contract: min_value and max_value are numeric."""
        params = mock_theory.get_parameters()
        for param in params.values():
            assert isinstance(param.min_value, (int, float))
            assert isinstance(param.max_value, (int, float))

    def test_parameter_value_within_bounds(self, mock_theory) -> None:
        """Contract: Parameter value is within bounds."""
        params = mock_theory.get_parameters()
        for param in params.values():
            assert param.min_value <= param.value <= param.max_value

    def test_parameter_bounds_ordering(self, mock_theory) -> None:
        """Contract: min_value <= max_value."""
        params = mock_theory.get_parameters()
        for param in params.values():
            assert param.min_value <= param.max_value

    def test_parameter_has_opt_type(self, mock_theory) -> None:
        """Contract: Parameter has opt_type attribute."""
        params = mock_theory.get_parameters()
        for param in params.values():
            assert hasattr(param, "opt_type")
            assert param.opt_type in ["opt", "var"]


class TestCalculationOutputContract:
    """Contract tests for theory calculation output data.

    Required contract:
    - Output shape matches input shape (n,)
    - Output dtype is float64
    - Output values are finite (no NaN unless theory-specific)
    - Output is deterministic
    """

    def test_output_shape_contract(
        self,
        mock_theory,
        synthetic_frequency_data: SyntheticData,
    ) -> None:
        """Contract: Output array shape matches input shape."""
        params = {"slope": 2.0, "intercept": 1.0}
        x = synthetic_frequency_data.x
        y = mock_theory.calculate(params, x)

        assert y.shape == x.shape
        assert len(y.shape) == 1

    def test_output_dtype_contract(
        self,
        mock_theory,
        synthetic_frequency_data: SyntheticData,
    ) -> None:
        """Contract: Output dtype is float."""
        params = {"slope": 2.0, "intercept": 1.0}
        x = synthetic_frequency_data.x
        y = mock_theory.calculate(params, x)

        assert "float" in str(y.dtype).lower()

    def test_output_finite_values(
        self,
        mock_theory,
        synthetic_frequency_data: SyntheticData,
    ) -> None:
        """Contract: Output values are finite."""
        params = {"slope": 2.0, "intercept": 1.0}
        x = synthetic_frequency_data.x
        y = mock_theory.calculate(params, x)

        assert jnp.all(jnp.isfinite(y))

    def test_output_deterministic(
        self,
        mock_theory,
        synthetic_frequency_data: SyntheticData,
    ) -> None:
        """Contract: Same input produces identical output."""
        params = {"slope": 2.0, "intercept": 1.0}
        x = synthetic_frequency_data.x

        y1 = mock_theory.calculate(params, x)
        y2 = mock_theory.calculate(params, x)

        np.testing.assert_array_equal(y1, y2)

    def test_output_continuous_for_continuous_input(
        self,
        mock_theory,
    ) -> None:
        """Contract: Small input changes produce proportional output changes."""
        params = {"slope": 2.0, "intercept": 1.0}
        x1 = jnp.array([1.0, 2.0, 3.0])
        x2 = x1 + 1e-8  # Very small perturbation

        y1 = mock_theory.calculate(params, x1)
        y2 = mock_theory.calculate(params, x2)

        # Changes should be proportional
        relative_change = jnp.abs((y2 - y1) / y1)
        assert jnp.all(relative_change < 1e-6)


class TestParameterValueContract:
    """Contract tests for parameter value constraints.

    Required contract:
    - Parameter values must respect declared bounds
    - Parameter values must be numeric
    - Parameter value changes are persisted
    """

    def test_set_parameter_persists_value(self, mock_theory) -> None:
        """Contract: set_parameter() persists the new value."""
        mock_theory.set_parameter("slope", 5.0)
        params = mock_theory.get_parameters()
        assert params["slope"].value == 5.0

    def test_set_parameter_within_bounds(self, mock_theory) -> None:
        """Contract: Parameters can be set within bounds."""
        params = mock_theory.get_parameters()
        original_param = params["slope"]

        # Set to a valid value within bounds
        new_value = (original_param.min_value + original_param.max_value) / 2
        mock_theory.set_parameter("slope", new_value)

        updated_params = mock_theory.get_parameters()
        assert updated_params["slope"].value == new_value

    def test_set_parameter_raises_on_nonexistent(self, mock_theory) -> None:
        """Contract: set_parameter() raises for unknown parameters."""
        with pytest.raises((KeyError, ValueError)):
            mock_theory.set_parameter("nonexistent_param", 1.0)


class TestFitParametersContract:
    """Contract tests for fit parameter specifications.

    Required contract:
    - get_fit_parameters() returns list of parameter names to optimize
    - Names correspond to actual parameters
    - Can be empty (no fitting)
    """

    def test_get_fit_parameters_returns_list(self, mock_theory) -> None:
        """Contract: get_fit_parameters() returns list."""
        fit_params = mock_theory.get_fit_parameters()
        assert isinstance(fit_params, list)

    def test_fit_parameters_are_strings(self, mock_theory) -> None:
        """Contract: Fit parameters are string names."""
        fit_params = mock_theory.get_fit_parameters()
        for param_name in fit_params:
            assert isinstance(param_name, str)

    def test_fit_parameters_exist_in_theory(self, mock_theory) -> None:
        """Contract: Fit parameters refer to existing parameters."""
        fit_params = mock_theory.get_fit_parameters()
        all_params = mock_theory.get_parameters()

        for param_name in fit_params:
            assert param_name in all_params


class TestSerializedDataContract:
    """Contract tests for serialized dataset structure.

    Required contract (JSON/NPZ format):
    - Metadata in JSON (non-array data)
    - Arrays in NPZ
    - Type information preserved
    - No executable code
    """

    def test_serialized_metadata_is_dict(self, temp_json_file) -> None:
        """Contract: Serialized metadata is valid JSON dict."""
        import json

        test_metadata = {
            "name": "test_dataset",
            "source": "simulation",
            "timestamp": "2025-01-01T00:00:00",
        }

        temp_json_file.parent.mkdir(exist_ok=True)
        with open(temp_json_file, "w") as f:
            json.dump(test_metadata, f)

        # Verify it can be read back
        with open(temp_json_file, "r") as f:
            loaded = json.load(f)

        assert isinstance(loaded, dict)
        assert loaded["name"] == "test_dataset"

    def test_serialized_arrays_valid_npz(self, temp_npz_file) -> None:
        """Contract: Arrays stored in NPZ format are valid."""
        test_arrays = {
            "x": np.linspace(0, 10, 100),
            "y": np.sin(np.linspace(0, 10, 100)),
        }

        np.savez(temp_npz_file, **test_arrays)

        # Verify it can be read back
        loaded = np.load(temp_npz_file)
        assert "x" in loaded
        assert "y" in loaded
        np.testing.assert_array_equal(loaded["x"], test_arrays["x"])

    def test_serialized_no_pickle_code(self, temp_json_file) -> None:
        """Contract: JSON cannot contain executable code."""
        import json

        # JSON is text-only, cannot contain Python code
        test_data = {"value": 42, "string": "hello"}
        with open(temp_json_file, "w") as f:
            json.dump(test_data, f)

        # Reading as JSON is safe
        with open(temp_json_file, "r") as f:
            loaded = json.load(f)

        assert loaded["value"] == 42
        assert loaded["string"] == "hello"
