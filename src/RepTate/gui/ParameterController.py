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
"""Module ParameterController

Extracted from QTheory to handle parameter management, validation,
and serialization.

This class follows the Single Responsibility Principle by focusing exclusively
on parameter operations.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
from jax import Array

if TYPE_CHECKING:
    from RepTate.core.Parameter import Parameter


class ParameterController:
    """Controls parameter management for theories.

    This class extracts parameter functionality from QTheory
    to reduce the god class size and improve maintainability.

    Attributes:
        parameters: Dictionary of Parameter objects.
        logger: Logger for this controller.
    """

    def __init__(
        self,
        parameters: dict[str, "Parameter"] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize ParameterController.

        Args:
            parameters: Dictionary of Parameter objects.
            logger: Optional logger instance.
        """
        self.parameters = parameters or {}
        self.logger = logger or logging.getLogger(__name__)

    def get_parameter_value(self, name: str) -> Any:
        """Get a parameter value by name.

        Args:
            name: Parameter name.

        Returns:
            Parameter value, or None if not found.
        """
        if name in self.parameters:
            return self.parameters[name].value
        self.logger.warning(f"Parameter '{name}' not found")
        return None

    def set_parameter_value(self, name: str, value: Any) -> bool:
        """Set a parameter value by name.

        Args:
            name: Parameter name.
            value: New value.

        Returns:
            True if set successfully, False otherwise.
        """
        if name not in self.parameters:
            self.logger.warning(f"Parameter '{name}' not found")
            return False

        param = self.parameters[name]
        try:
            # Validate and set value
            param.value = value
            return True
        except (ValueError, TypeError) as e:
            self.logger.error(f"Failed to set parameter '{name}': {e}")
            return False

    def get_fit_parameters(self) -> dict[str, float]:
        """Get all parameters that are marked for optimization.

        Returns:
            Dictionary of parameter names to values for fitting.
        """
        from RepTate.core.Parameter import OptType

        fit_params = {}
        for name, param in self.parameters.items():
            if param.opt_type == OptType.opt:
                fit_params[name] = float(param.value)
        return fit_params

    def get_fit_parameter_array(self) -> Array:
        """Get fit parameters as a JAX array.

        Returns:
            Array of parameter values in order.
        """
        fit_params = self.get_fit_parameters()
        return jnp.array(list(fit_params.values()))

    def get_fit_parameter_names(self) -> list[str]:
        """Get names of parameters marked for optimization.

        Returns:
            List of parameter names.
        """
        from RepTate.core.Parameter import OptType

        return [
            name
            for name, param in self.parameters.items()
            if param.opt_type == OptType.opt
        ]

    def set_fit_parameters(self, values: dict[str, float] | Array) -> None:
        """Set values for fit parameters.

        Args:
            values: Either a dict of name->value or array matching
                    get_fit_parameter_names() order.
        """
        if isinstance(values, dict):
            for name, value in values.items():
                self.set_parameter_value(name, value)
        else:
            # Array case - match with names
            names = self.get_fit_parameter_names()
            for name, value in zip(names, values, strict=False):
                self.set_parameter_value(name, float(value))

    def get_parameter_bounds(
        self, param_names: list[str] | None = None
    ) -> tuple[list[float], list[float]]:
        """Get bounds for parameters.

        Args:
            param_names: Optional list of parameter names. If None,
                        uses fit parameters.

        Returns:
            Tuple of (lower_bounds, upper_bounds) lists.
        """
        if param_names is None:
            param_names = self.get_fit_parameter_names()

        lower = []
        upper = []
        for name in param_names:
            param = self.parameters.get(name)
            if param is None:
                lower.append(-np.inf)
                upper.append(np.inf)
            else:
                lower.append(param.min_value if param.min_value is not None else -np.inf)
                upper.append(param.max_value if param.max_value is not None else np.inf)
        return lower, upper

    def copy_parameters(self) -> dict[str, dict[str, Any]]:
        """Copy all parameters to a dictionary format.

        Returns:
            Dictionary with parameter data for serialization.
        """
        result = {}
        for name, param in self.parameters.items():
            result[name] = {
                "value": param.value,
                "min": param.min_value,
                "max": param.max_value,
                "opt_type": str(param.opt_type),
                "description": param.description,
            }
        return result

    def paste_parameters(self, param_dict: dict[str, dict[str, Any]]) -> int:
        """Paste parameters from a dictionary.

        Args:
            param_dict: Dictionary with parameter data.

        Returns:
            Number of parameters successfully updated.
        """
        updated = 0
        for name, data in param_dict.items():
            if name in self.parameters:
                if "value" in data:
                    if self.set_parameter_value(name, data["value"]):
                        updated += 1
        return updated

    def get_modes(self) -> tuple[Array, Array] | None:
        """Get relaxation modes (tau, G) from mode parameters.

        Returns:
            Tuple of (tau_array, G_array) or None if no modes.
        """
        taus = []
        gs = []

        # Look for mode parameters (tau1, G1, tau2, G2, etc.)
        mode_idx = 1
        while True:
            tau_name = f"tau{mode_idx}"
            g_name = f"G{mode_idx}"
            if tau_name in self.parameters and g_name in self.parameters:
                taus.append(float(self.parameters[tau_name].value))
                gs.append(float(self.parameters[g_name].value))
                mode_idx += 1
            else:
                break

        if not taus:
            return None
        return jnp.array(taus), jnp.array(gs)

    def set_modes(self, tau: Array, G: Array) -> int:
        """Set relaxation modes from arrays.

        Args:
            tau: Array of relaxation times.
            G: Array of moduli.

        Returns:
            Number of modes set.
        """
        n_modes = min(len(tau), len(G))
        for i in range(n_modes):
            tau_name = f"tau{i + 1}"
            g_name = f"G{i + 1}"
            if tau_name in self.parameters:
                self.set_parameter_value(tau_name, float(tau[i]))
            if g_name in self.parameters:
                self.set_parameter_value(g_name, float(G[i]))
        return n_modes

    def validate_all_parameters(self) -> list[str]:
        """Validate all parameters are within bounds.

        Returns:
            List of error messages for invalid parameters.
        """
        errors = []
        for name, param in self.parameters.items():
            value = param.value
            if param.min_value is not None and value < param.min_value:
                errors.append(
                    f"Parameter '{name}' = {value} is below minimum {param.min_value}"
                )
            if param.max_value is not None and value > param.max_value:
                errors.append(
                    f"Parameter '{name}' = {value} is above maximum {param.max_value}"
                )
        return errors

    def reset_to_defaults(self) -> None:
        """Reset all parameters to their default values."""
        for param in self.parameters.values():
            if hasattr(param, "default_value") and param.default_value is not None:
                param.value = param.default_value
