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
"""Module ToolIntegral

Integral file for creating a new Tool
"""
import traceback
import numpy as np
import jax.numpy as jnp
from interpax import interp1d
from RepTate.gui.QTool import QTool


class ToolIntegral(QTool):
    """Calculate the integral of y with respect to x, where y is the ordinate and x is the abcissa in the current view. Repeated points in the data are removed before the integral is performed. The data between the point is interpolated with a cubic spline. The total value of the definite integral is shown in the Tool output region.
    If a different integration interval is needed, the Bounds tool can be used before the Integral tool.
    """

    toolname = "Integral"
    description = "Integral of current data/view"
    citations = []
    # html_help_file = 'http://reptate.readthedocs.io/manual/Tools/Integral.html'

    def __init__(self, name="", parent_app=None):
        """**Constructor**"""
        super().__init__(name, parent_app)

        # self.function = self.integral  # main Tool function
        # self.parameters['param1'] = Parameter(
        # name='param1',
        # value=1,
        # description='parameter 1',
        # type=ParameterType.real,
        # opt_type=OptType.const)
        self.update_parameter_table()
        self.parent_application.update_all_ds_plots()


    def calculate(self, x, y, ax=None, color=None, file_parameters=[]):
        """Calculate the cumulative integral of y with respect to x.

        Removes duplicate x values, interpolates data using cubic splines (via interpax),
        and performs cumulative trapezoidal integration. The total integral value is
        displayed in the tool output area.

        Args:
            x (numpy.ndarray): Array of x-coordinates (integration variable).
            y (numpy.ndarray): Array of y-coordinates (integrand).
            ax (matplotlib.axes.Axes, optional): Matplotlib axes for plotting. Defaults to None.
            color: Color specification for plotting. Defaults to None.
            file_parameters (list): List of file-specific parameters. Defaults to [].

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: Tuple (xunique, cumulative_integral) where
                xunique contains unique x values and cumulative_integral contains the
                running integral from x[0] to each x value. Returns original (x, y) if
                calculation fails.
        """
        xunique, indunique = np.unique(x, return_index=True)
        num_rows = len(xunique)
        yunique = y[indunique]
        try:
            # Convert to JAX arrays for interpolation
            xunique_jax = jnp.array(xunique)
            yunique_jax = jnp.array(yunique)

            # Interpolate using interpax (note: interpax is a function, not a factory)
            # We interpolate at the original points to smooth the data
            y_interp = interp1d(xunique_jax, xunique_jax, yunique_jax, method="cubic", extrap=True)

            # Cumulative trapezoidal integration
            dx = xunique_jax[1:] - xunique_jax[:-1]
            avg_y = (y_interp[1:] + y_interp[:-1]) / 2.0
            cumulative = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(avg_y * dx)])

            # Convert back to numpy for plotting
            y2 = np.array(cumulative)

            self.Qprint("<b>I</b> = %g" % y2[-1])
            return xunique, y2
        except Exception as e:
            self.Qprint("in ToolIntegral.calculate(): %s" % traceback.format_exc())
            return x, y

