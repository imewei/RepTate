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
"""Module InterpolateExtrapolate

Interpolate/Extrapolate data
"""
import traceback
import numpy as np
import jax.numpy as jnp
from interpax import interp1d
from RepTate.core.Parameter import Parameter, ParameterType, OptType
from RepTate.gui.QTool import QTool


class ToolInterpolateExtrapolate(QTool):
    """Interpolates data"""

    toolname = "Interpolate/Extrapolate"
    description = "Interpolate/Extrapolate from view"
    citations = []
    # html_help_file = 'http://reptate.readthedocs.io/manual/Tools/template.html'

    def __init__(self, name="", parent_app=None):
        """**Constructor**"""
        super().__init__(name, parent_app)
        self.parameters["x"] = Parameter(
            name="x",
            value=1,
            description="x",
            type=ParameterType.real,
            opt_type=OptType.const,
        )

        self.update_parameter_table()
        self.parent_application.update_all_ds_plots()

        # add widgets specific to the Tool here:


    def calculate(self, x, y, ax=None, color=None, file_parameters=[]):
        """Interpolate or extrapolate y value at a specified x coordinate.

        Uses cubic spline interpolation (via interpax) to estimate y at the user-specified
        x value. Handles extrapolation beyond the data range. Results are displayed in a
        formatted table in the tool output area.

        Args:
            x (numpy.ndarray): Array of x-coordinates (data points).
            y (numpy.ndarray): Array of y-coordinates (data points).
            ax (matplotlib.axes.Axes, optional): Matplotlib axes for plotting. Defaults to None.
            color: Color specification for plotting. Defaults to None.
            file_parameters (list): List of file-specific parameters. Defaults to [].

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: Original input arrays (x, y) unchanged.
                The interpolated/extrapolated value is displayed in the tool output.
        """
        xval = self.parameters["x"].value
        xunique, indunique = np.unique(x, return_index=True)
        yunique = y[indunique]
        try:
            # Convert to JAX arrays
            xunique_jax = jnp.array(xunique)
            yunique_jax = jnp.array(yunique)
            xval_jax = jnp.array([xval])  # interpax expects array for query points

            # Interpolate using interpax
            yval_array = interp1d(xval_jax, xunique_jax, yunique_jax, method="cubic", extrap=True)
            yval = float(yval_array[0])

            # Format output table
            table = [
                ["%-10s" % "x", "%-10s" % "y"],
                ["%-10.4e" % xval, "%-10.4e" % yval],
            ]
            self.Qprint(table)
        except Exception as e:
            self.Qprint(
                "in ToolInterpolateExtrapolate.calculate(): %s" % traceback.format_exc()
            )
        return x, y
