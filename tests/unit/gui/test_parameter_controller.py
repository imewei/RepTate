"""Unit tests for ParameterController.

Tests cover:
- T066: Unit tests for ParameterController component

These tests validate the ParameterController component extracted from QTheory.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import jax.numpy as jnp
import pytest

if TYPE_CHECKING:
    pass


def create_mock_parameter(
    value: float = 1.0,
    min_value: float | None = None,
    max_value: float | None = None,
    opt_type: str = "nopt",
) -> MagicMock:
    """Create a mock Parameter object."""
    mock = MagicMock()
    mock.value = value
    mock.min_value = min_value
    mock.max_value = max_value
    mock.description = "Test parameter"

    # Set opt_type enum
    from RepTate.core.Parameter import OptType

    if opt_type == "opt":
        mock.opt_type = OptType.opt
    else:
        mock.opt_type = OptType.nopt

    return mock


class TestParameterControllerInit:
    """Test ParameterController initialization."""

    def test_init_with_parameters(self) -> None:
        """Test initialization with parameters."""
        from RepTate.gui.ParameterController import ParameterController

        params = {"a": create_mock_parameter(), "b": create_mock_parameter()}
        controller = ParameterController(parameters=params)

        assert len(controller.parameters) == 2
        assert "a" in controller.parameters
        assert "b" in controller.parameters

    def test_init_empty(self) -> None:
        """Test initialization without parameters."""
        from RepTate.gui.ParameterController import ParameterController

        controller = ParameterController()

        assert len(controller.parameters) == 0
        assert controller.logger is not None


class TestGetSetParameters:
    """Test getting and setting parameters."""

    def test_get_parameter_value(self) -> None:
        """Test getting a parameter value."""
        from RepTate.gui.ParameterController import ParameterController

        params = {"a": create_mock_parameter(value=3.14)}
        controller = ParameterController(parameters=params)

        value = controller.get_parameter_value("a")

        assert value == 3.14

    def test_get_parameter_value_missing(self) -> None:
        """Test getting a missing parameter returns None."""
        from RepTate.gui.ParameterController import ParameterController

        controller = ParameterController(parameters={})

        value = controller.get_parameter_value("nonexistent")

        assert value is None

    def test_set_parameter_value(self) -> None:
        """Test setting a parameter value."""
        from RepTate.gui.ParameterController import ParameterController

        mock_param = create_mock_parameter(value=1.0)
        params = {"a": mock_param}
        controller = ParameterController(parameters=params)

        result = controller.set_parameter_value("a", 2.5)

        assert result is True
        assert mock_param.value == 2.5

    def test_set_parameter_value_missing(self) -> None:
        """Test setting a missing parameter returns False."""
        from RepTate.gui.ParameterController import ParameterController

        controller = ParameterController(parameters={})

        result = controller.set_parameter_value("nonexistent", 1.0)

        assert result is False


class TestFitParameters:
    """Test fit parameter operations."""

    def test_get_fit_parameters(self) -> None:
        """Test getting parameters marked for optimization."""
        from RepTate.gui.ParameterController import ParameterController

        params = {
            "a": create_mock_parameter(value=1.0, opt_type="opt"),
            "b": create_mock_parameter(value=2.0, opt_type="nopt"),
            "c": create_mock_parameter(value=3.0, opt_type="opt"),
        }
        controller = ParameterController(parameters=params)

        fit_params = controller.get_fit_parameters()

        assert "a" in fit_params
        assert "c" in fit_params
        assert "b" not in fit_params
        assert fit_params["a"] == 1.0
        assert fit_params["c"] == 3.0

    def test_get_fit_parameter_array(self) -> None:
        """Test getting fit parameters as array."""
        from RepTate.gui.ParameterController import ParameterController

        params = {
            "a": create_mock_parameter(value=1.0, opt_type="opt"),
            "b": create_mock_parameter(value=2.0, opt_type="opt"),
        }
        controller = ParameterController(parameters=params)

        arr = controller.get_fit_parameter_array()

        assert len(arr) == 2
        assert jnp.allclose(arr, jnp.array([1.0, 2.0]))

    def test_get_fit_parameter_names(self) -> None:
        """Test getting names of fit parameters."""
        from RepTate.gui.ParameterController import ParameterController

        params = {
            "a": create_mock_parameter(value=1.0, opt_type="opt"),
            "b": create_mock_parameter(value=2.0, opt_type="nopt"),
            "c": create_mock_parameter(value=3.0, opt_type="opt"),
        }
        controller = ParameterController(parameters=params)

        names = controller.get_fit_parameter_names()

        assert "a" in names
        assert "c" in names
        assert "b" not in names

    def test_set_fit_parameters_dict(self) -> None:
        """Test setting fit parameters from dict."""
        from RepTate.gui.ParameterController import ParameterController

        mock_a = create_mock_parameter(value=1.0, opt_type="opt")
        mock_b = create_mock_parameter(value=2.0, opt_type="opt")
        params = {"a": mock_a, "b": mock_b}
        controller = ParameterController(parameters=params)

        controller.set_fit_parameters({"a": 10.0, "b": 20.0})

        assert mock_a.value == 10.0
        assert mock_b.value == 20.0

    def test_set_fit_parameters_array(self) -> None:
        """Test setting fit parameters from array."""
        from RepTate.gui.ParameterController import ParameterController

        mock_a = create_mock_parameter(value=1.0, opt_type="opt")
        mock_b = create_mock_parameter(value=2.0, opt_type="opt")
        params = {"a": mock_a, "b": mock_b}
        controller = ParameterController(parameters=params)

        controller.set_fit_parameters(jnp.array([10.0, 20.0]))

        # Values should be updated (though order may vary)
        assert mock_a.value == 10.0 or mock_a.value == 20.0


class TestParameterBounds:
    """Test parameter bounds operations."""

    def test_get_parameter_bounds(self) -> None:
        """Test getting parameter bounds."""
        from RepTate.gui.ParameterController import ParameterController

        params = {
            "a": create_mock_parameter(value=1.0, min_value=0.0, max_value=10.0, opt_type="opt"),
            "b": create_mock_parameter(value=2.0, min_value=-5.0, max_value=5.0, opt_type="opt"),
        }
        controller = ParameterController(parameters=params)

        lower, upper = controller.get_parameter_bounds()

        assert len(lower) == 2
        assert len(upper) == 2
        assert 0.0 in lower
        assert 10.0 in upper

    def test_get_parameter_bounds_unbounded(self) -> None:
        """Test bounds for unbounded parameters."""
        import numpy as np

        from RepTate.gui.ParameterController import ParameterController

        params = {
            "a": create_mock_parameter(value=1.0, min_value=None, max_value=None, opt_type="opt"),
        }
        controller = ParameterController(parameters=params)

        lower, upper = controller.get_parameter_bounds()

        assert lower[0] == -np.inf
        assert upper[0] == np.inf


class TestCopyPasteParameters:
    """Test parameter copy/paste functionality."""

    def test_copy_parameters(self) -> None:
        """Test copying parameters to dict."""
        from RepTate.gui.ParameterController import ParameterController

        params = {
            "a": create_mock_parameter(value=1.0, min_value=0.0, max_value=10.0),
        }
        controller = ParameterController(parameters=params)

        result = controller.copy_parameters()

        assert "a" in result
        assert result["a"]["value"] == 1.0
        assert result["a"]["min"] == 0.0
        assert result["a"]["max"] == 10.0

    def test_paste_parameters(self) -> None:
        """Test pasting parameters from dict."""
        from RepTate.gui.ParameterController import ParameterController

        mock_a = create_mock_parameter(value=1.0)
        mock_b = create_mock_parameter(value=2.0)
        params = {"a": mock_a, "b": mock_b}
        controller = ParameterController(parameters=params)

        updated = controller.paste_parameters({
            "a": {"value": 5.0},
            "b": {"value": 6.0},
            "c": {"value": 7.0},  # Non-existent, should be ignored
        })

        assert updated == 2
        assert mock_a.value == 5.0
        assert mock_b.value == 6.0


class TestModes:
    """Test relaxation mode operations."""

    def test_get_modes_none(self) -> None:
        """Test get_modes returns None when no modes."""
        from RepTate.gui.ParameterController import ParameterController

        params = {"a": create_mock_parameter()}
        controller = ParameterController(parameters=params)

        result = controller.get_modes()

        assert result is None

    def test_get_modes_with_modes(self) -> None:
        """Test get_modes with mode parameters."""
        from RepTate.gui.ParameterController import ParameterController

        params = {
            "tau1": create_mock_parameter(value=1.0),
            "G1": create_mock_parameter(value=1e5),
            "tau2": create_mock_parameter(value=10.0),
            "G2": create_mock_parameter(value=1e4),
        }
        controller = ParameterController(parameters=params)

        result = controller.get_modes()

        assert result is not None
        taus, gs = result
        assert len(taus) == 2
        assert len(gs) == 2
        assert jnp.allclose(taus, jnp.array([1.0, 10.0]))
        assert jnp.allclose(gs, jnp.array([1e5, 1e4]))

    def test_set_modes(self) -> None:
        """Test setting modes."""
        from RepTate.gui.ParameterController import ParameterController

        mock_tau1 = create_mock_parameter()
        mock_g1 = create_mock_parameter()
        params = {"tau1": mock_tau1, "G1": mock_g1}
        controller = ParameterController(parameters=params)

        n_set = controller.set_modes(jnp.array([2.0]), jnp.array([2e5]))

        assert n_set == 1
        assert mock_tau1.value == 2.0
        assert mock_g1.value == 2e5


class TestValidation:
    """Test parameter validation."""

    def test_validate_all_valid(self) -> None:
        """Test validation with all valid parameters."""
        from RepTate.gui.ParameterController import ParameterController

        params = {
            "a": create_mock_parameter(value=5.0, min_value=0.0, max_value=10.0),
        }
        controller = ParameterController(parameters=params)

        errors = controller.validate_all_parameters()

        assert len(errors) == 0

    def test_validate_below_minimum(self) -> None:
        """Test validation catches below minimum."""
        from RepTate.gui.ParameterController import ParameterController

        params = {
            "a": create_mock_parameter(value=-1.0, min_value=0.0, max_value=10.0),
        }
        controller = ParameterController(parameters=params)

        errors = controller.validate_all_parameters()

        assert len(errors) == 1
        assert "below minimum" in errors[0]

    def test_validate_above_maximum(self) -> None:
        """Test validation catches above maximum."""
        from RepTate.gui.ParameterController import ParameterController

        params = {
            "a": create_mock_parameter(value=20.0, min_value=0.0, max_value=10.0),
        }
        controller = ParameterController(parameters=params)

        errors = controller.validate_all_parameters()

        assert len(errors) == 1
        assert "above maximum" in errors[0]
