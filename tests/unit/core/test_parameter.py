"""Unit tests for Parameter class.

Tests cover:
- T044: Parameter initialization, types, validation, and optimization flags

The Parameter class defines theory parameters with metadata for fitting,
bounds, and display properties.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestParameterTypes:
    """Test ParameterType enum and type handling."""

    def test_parameter_type_enum_values(self) -> None:
        """Test ParameterType enum has expected values."""
        from RepTate.core.Parameter import ParameterType

        assert ParameterType.real.value == 0
        assert ParameterType.integer.value == 1
        assert ParameterType.discrete_real.value == 2
        assert ParameterType.discrete_integer.value == 3
        assert ParameterType.boolean.value == 4
        assert ParameterType.string.value == 5

    def test_opt_type_enum_values(self) -> None:
        """Test OptType enum has expected values."""
        from RepTate.core.Parameter import OptType

        assert OptType.opt.value == 1
        assert OptType.nopt.value == 2
        assert OptType.const.value == 3


class TestParameterInitialization:
    """Test Parameter construction and defaults."""

    def test_default_initialization(self) -> None:
        """Test Parameter initializes with defaults."""
        from RepTate.core.Parameter import OptType, Parameter, ParameterType

        p = Parameter()

        assert p.name == ""
        assert p.value == 0.0
        assert p.description == ""
        assert p.type == ParameterType.real
        assert p.opt_type == OptType.opt
        assert p.min_value == -np.inf
        assert p.max_value == np.inf
        assert p.display_flag is True
        assert p.discrete_values == []

    def test_real_parameter(self) -> None:
        """Test real parameter initialization."""
        from RepTate.core.Parameter import Parameter, ParameterType

        p = Parameter(
            name="modulus",
            value=1e5,
            description="Storage modulus",
            type=ParameterType.real,
        )

        assert p.name == "modulus"
        assert p.value == 1e5
        assert isinstance(p.value, float)
        assert p.description == "Storage modulus"

    def test_integer_parameter(self) -> None:
        """Test integer parameter type conversion."""
        from RepTate.core.Parameter import Parameter, ParameterType

        p = Parameter(
            name="nmodes",
            value=5.7,  # Will be converted to int
            type=ParameterType.integer,
        )

        assert p.value == 5
        assert isinstance(p.value, int)

    def test_discrete_real_parameter(self) -> None:
        """Test discrete real parameter with allowed values."""
        from RepTate.core.Parameter import Parameter, ParameterType

        p = Parameter(
            name="temperature",
            value=25.0,
            type=ParameterType.discrete_real,
            discrete_values=[20.0, 25.0, 30.0],
        )

        assert p.value == 25.0
        assert isinstance(p.value, float)
        assert p.discrete_values == [20.0, 25.0, 30.0]

    def test_discrete_integer_parameter(self) -> None:
        """Test discrete integer parameter."""
        from RepTate.core.Parameter import Parameter, ParameterType

        p = Parameter(
            name="chain_type",
            value=2,
            type=ParameterType.discrete_integer,
            discrete_values=[1, 2, 3],
        )

        assert p.value == 2
        assert isinstance(p.value, int)

    def test_boolean_parameter_true_values(self) -> None:
        """Test boolean parameter recognizes truthy values."""
        from RepTate.core.Parameter import Parameter, ParameterType

        true_inputs = [True, "true", "True", "1", "t", "T", "y", "yes"]

        for val in true_inputs:
            p = Parameter(name="flag", value=val, type=ParameterType.boolean)
            assert p.value is True, f"Failed for input: {val}"

    def test_boolean_parameter_false_values(self) -> None:
        """Test boolean parameter recognizes falsy values."""
        from RepTate.core.Parameter import Parameter, ParameterType

        false_inputs = [False, "false", "False", "0", "f", "F", "n", "no", ""]

        for val in false_inputs:
            p = Parameter(name="flag", value=val, type=ParameterType.boolean)
            assert p.value is False, f"Failed for input: {val}"

    def test_string_parameter(self) -> None:
        """Test string parameter type."""
        from RepTate.core.Parameter import Parameter, ParameterType

        p = Parameter(
            name="model_name",
            value="Maxwell",
            type=ParameterType.string,
        )

        assert p.value == "Maxwell"
        assert isinstance(p.value, str)


class TestParameterBounds:
    """Test parameter bounds and constraints."""

    def test_bounds_specification(self) -> None:
        """Test setting min and max bounds."""
        from RepTate.core.Parameter import Parameter

        p = Parameter(
            name="tau",
            value=1.0,
            min_value=0.0,
            max_value=100.0,
        )

        assert p.min_value == 0.0
        assert p.max_value == 100.0

    def test_unbounded_parameter(self) -> None:
        """Test parameter with infinite bounds."""
        from RepTate.core.Parameter import Parameter

        p = Parameter(name="unbounded")

        assert p.min_value == -np.inf
        assert p.max_value == np.inf


class TestParameterOptimization:
    """Test optimization type settings."""

    def test_opt_type_opt(self) -> None:
        """Test parameter set to be optimized."""
        from RepTate.core.Parameter import OptType, Parameter

        p = Parameter(name="G", opt_type=OptType.opt)

        assert p.opt_type == OptType.opt

    def test_opt_type_nopt(self) -> None:
        """Test parameter set to not be optimized."""
        from RepTate.core.Parameter import OptType, Parameter

        p = Parameter(name="fixed_G", opt_type=OptType.nopt)

        assert p.opt_type == OptType.nopt

    def test_opt_type_const(self) -> None:
        """Test constant parameter."""
        from RepTate.core.Parameter import OptType, Parameter

        p = Parameter(name="constant", opt_type=OptType.const)

        assert p.opt_type == OptType.const


class TestParameterError:
    """Test parameter error (uncertainty) tracking."""

    def test_initial_error_is_inf(self) -> None:
        """Test error initializes to infinity."""
        from RepTate.core.Parameter import Parameter

        p = Parameter(name="test")

        assert p.error == np.inf

    def test_error_can_be_set(self) -> None:
        """Test error value can be updated."""
        from RepTate.core.Parameter import Parameter

        p = Parameter(name="test")
        p.error = 0.05

        assert p.error == 0.05


class TestParameterCopy:
    """Test parameter copy functionality."""

    def test_copy_method(self) -> None:
        """Test copying parameter values."""
        from RepTate.core.Parameter import OptType, Parameter, ParameterType

        source = Parameter(
            name="source_param",
            value=42.0,
            description="Original",
            type=ParameterType.real,
            opt_type=OptType.nopt,
            min_value=0.0,
            max_value=100.0,
        )

        target = Parameter()
        target.copy(source)

        assert target.name == "source_param"
        assert target.value == 42.0
        assert target.description == "Original"
        assert target.type == ParameterType.real
        assert target.opt_type == OptType.nopt
        assert target.min_value == 0.0
        assert target.max_value == 100.0


class TestParameterStringRepresentation:
    """Test string representation methods."""

    def test_str_representation(self) -> None:
        """Test __str__ returns name=value format."""
        from RepTate.core.Parameter import Parameter

        p = Parameter(name="tau", value=1.5)

        result = str(p)
        assert "tau" in result
        assert "1.5" in result

    def test_repr_representation(self) -> None:
        """Test __repr__ provides full parameter info."""
        from RepTate.core.Parameter import Parameter

        p = Parameter(
            name="G0",
            value=1e5,
            description="Modulus",
        )

        result = repr(p)
        assert "Parameter" in result
        assert "G0" in result
        assert "Modulus" in result


class TestParameterDisplayFlag:
    """Test display_flag property."""

    def test_display_flag_default_true(self) -> None:
        """Test display_flag defaults to True."""
        from RepTate.core.Parameter import Parameter

        p = Parameter(name="visible")

        assert p.display_flag is True

    def test_display_flag_can_be_false(self) -> None:
        """Test display_flag can be set to False."""
        from RepTate.core.Parameter import Parameter

        p = Parameter(name="hidden", display_flag=False)

        assert p.display_flag is False


class TestParameterEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_discrete_values(self) -> None:
        """Test discrete_values defaults to empty list."""
        from RepTate.core.Parameter import Parameter

        p = Parameter(name="test")

        assert p.discrete_values == []
        assert isinstance(p.discrete_values, list)

    def test_very_small_value(self) -> None:
        """Test handling of very small values."""
        from RepTate.core.Parameter import Parameter

        p = Parameter(name="small", value=1e-300)

        assert p.value == 1e-300

    def test_very_large_value(self) -> None:
        """Test handling of very large values."""
        from RepTate.core.Parameter import Parameter

        p = Parameter(name="large", value=1e300)

        assert p.value == 1e300

    def test_negative_value(self) -> None:
        """Test handling of negative values."""
        from RepTate.core.Parameter import Parameter

        p = Parameter(name="negative", value=-42.5)

        assert p.value == -42.5

    def test_zero_value(self) -> None:
        """Test handling of zero value."""
        from RepTate.core.Parameter import Parameter

        p = Parameter(name="zero", value=0.0)

        assert p.value == 0.0
