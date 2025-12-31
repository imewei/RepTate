# RepTate: Rheology of Entangled Polymers: Toolkit for the Analysis of Tool and Experiments
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
# Copyright (2018-2023): Jorge Ramirez, Victor Boudara, Universidad Politécnica de Madrid, University of Leeds
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
"""Module ToolSmooth

Smooth data by applying a Savitzky-Golay filter
"""
import traceback
import numpy as np
import jax.numpy as jnp
from RepTate.core.Parameter import Parameter, ParameterType
from RepTate.gui.QTool import QTool


def _savgol_filter_jax(y, window_length, polyorder):
    """JAX implementation of Savitzky-Golay filter.

    This function smooths data using a Savitzky-Golay filter, which fits
    successive sub-sets of adjacent data points with a low-degree polynomial
    by the method of linear least squares.

    Args:
        y: Array of values to be filtered
        window_length: The length of the filter window (i.e., number of coefficients).
                      Must be a positive odd integer.
        polyorder: The order of the polynomial used to fit the samples.
                  Must be less than window_length.

    Returns:
        Filtered array of same shape as y
    """
    # Convert to JAX array
    y = jnp.asarray(y)

    # Validate inputs
    if window_length % 2 != 1 or window_length < 1:
        raise ValueError("window_length must be a positive odd integer")
    if window_length < polyorder + 1:
        raise ValueError("window_length must be greater than polyorder")
    if window_length > len(y):
        raise ValueError("window_length is too large for the data")

    # Half window size
    half_window = (window_length - 1) // 2

    # Create the Vandermonde matrix for polynomial fitting
    # x values centered around 0
    x = jnp.arange(-half_window, half_window + 1)

    # Build Vandermonde matrix (each column is x^i for i from 0 to polyorder)
    A = jnp.vander(x, polyorder + 1, increasing=True)

    # Solve for filter coefficients using least squares
    # The middle row of (A^T A)^-1 A^T gives us the filter coefficients
    # for computing the smoothed value at the center point
    coeffs = jnp.linalg.pinv(A)[0]  # Get coefficients for 0-th derivative (smoothing)

    # Apply the filter using convolution
    # Pad the array to handle edges
    y_padded = jnp.pad(y, half_window, mode='edge')

    # Convolve with filter coefficients (reversed for correlation)
    filtered = jnp.convolve(y_padded, coeffs[::-1], mode='valid')

    return filtered


class ToolSmooth(QTool):
    """Smooths the current view data by applying a Savitzky-Golay filter. The smoothing procedure is controlled by means of two parameters: the **window** length (a positive, odd integer), which represents the number of convolution coefficients of the filter, and the **order** of the polynomial used to fit the samples (must be smaller than the window length).
    """

    toolname = "Smooth"
    description = "Smooth Tool"
    citations = []
    # html_help_file = 'http://reptate.readthedocs.io/manual/Tools/template.html'

    def __init__(self, name="", parent_app=None):
        """**Constructor**"""
        super().__init__(name, parent_app)
        self.parameters["window"] = Parameter(
            name="window",
            value=11,
            description="Length of filter window. Positive odd integer, smaller than the size of y and larger than order",
            type=ParameterType.integer,
        )
        self.parameters["order"] = Parameter(
            name="order",
            value=3,
            description="Order of smoothing polynomial (must be smaller than window)",
            type=ParameterType.integer,
        )

        self.update_parameter_table()
        self.parent_application.update_all_ds_plots()

    # add widgets specific to the Tool here:

    def set_param_value(self, name, value):
        """Set parameter value with validation of smoothing constraints.

        Validates that window is a positive odd integer larger than order, and that
        order is non-negative and smaller than window. Prevents invalid parameter
        combinations for Savitzky-Golay filtering.

        Args:
            name (str): Name of the parameter to set ('window' or 'order').
            value: New value for the parameter (will be converted to int).

        Returns:
            tuple[str, bool]: A tuple containing:
                - message (str): Status message or error description.
                - success (bool): True if parameter was set successfully, False otherwise.
        """
        p = self.parameters[name]
        old_value = p.value
        try:
            new_value = int(value)
        except ValueError:
            return "Value must be a integer", False
        message, success = super().set_param_value(name, value)
        if success:
            if name == "window":
                order = self.parameters["order"].value
                if new_value <= order or new_value < 0 or new_value % 2 == 0:
                    p.value = old_value
                    message = (
                        "window must be a positive, odd integer, larger than order"
                    )
                    success = False
            elif name == "order":
                window = self.parameters["window"].value
                if new_value >= window or new_value < 0:
                    p.value = old_value
                    message = "order must be >=0 and smaller than window"
                    success = False

        return message, success


    def calculate(self, x, y, ax=None, color=None, file_parameters=[]):
        """Smooth data using Savitzky-Golay filter.

        Applies a Savitzky-Golay filter that fits successive sub-sets of adjacent
        data points with a polynomial. Uses JAX implementation for computation.
        Validates parameters before filtering.

        Args:
            x (numpy.ndarray): Array of x-coordinates (unchanged).
            y (numpy.ndarray): Array of y-coordinates (data to be smoothed).
            ax (matplotlib.axes.Axes, optional): Matplotlib axes for plotting. Defaults to None.
            color: Color specification for plotting. Defaults to None.
            file_parameters (list): List of file-specific parameters. Defaults to [].

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: Tuple (x, smoothed_y) where smoothed_y
                contains the filtered data. Returns original (x, y) if filtering fails
                or parameters are invalid.
        """
        window = self.parameters["window"].value
        order = self.parameters["order"].value
        if window % 2 == 0:
            self.Qprint("Invalid window (must be an odd number)")
            return x, y
        if window >= len(y):
            self.Qprint("Invalid window (must be smaller than the length of the data)")
            return x, y
        if window <= order:
            self.Qprint("Invalid order (must be smaller than the window)")
            return x, y

        try:
            # Use JAX implementation of Savitzky-Golay filter
            y2_jax = _savgol_filter_jax(y, window, order)
            # Convert back to numpy for consistency
            y2 = np.array(y2_jax)
            return x, y2
        except Exception as e:
            self.Qprint("in ToolSmooth.calculate(): %s" % traceback.format_exc())
            return x, y
