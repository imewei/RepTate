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
"""Module ToolBounds

Remove data ouside Bounds
"""
import numpy as np
from RepTate.core.Parameter import Parameter, ParameterType
from RepTate.gui.QTool import QTool


class ToolBounds(QTool):
    """Remove points in the current view ïf :math:`x \\notin [x_{min}, x_{max}]` or :math:`y \\notin [y_{min}, y_{max}]`
    """

    toolname = "Bounds"
    description = "Bounds Tool"
    citations = []
    # html_help_file = 'http://reptate.readthedocs.io/manual/Tools/template.html'

    def __init__(self, name="", parent_app=None):
        """**Constructor**"""
        super().__init__(name, parent_app)
        self.parameters["xmin"] = Parameter(
            name="xmin",
            value=-np.Infinity,
            description="Minimum x",
            type=ParameterType.real,
        )
        self.parameters["xmax"] = Parameter(
            name="xmax",
            value=np.Infinity,
            description="Maximum x",
            type=ParameterType.real,
        )
        self.parameters["ymin"] = Parameter(
            name="ymin",
            value=-np.Infinity,
            description="Minimum y",
            type=ParameterType.real,
        )
        self.parameters["ymax"] = Parameter(
            name="ymax",
            value=np.Infinity,
            description="Maximum y",
            type=ParameterType.real,
        )

        self.update_parameter_table()
        self.parent_application.update_all_ds_plots()

    # add widgets specific to the Tool here:

    def set_param_value(self, name, value):
        """Set parameter value with validation of boundary constraints.

        Validates that minimum bounds are less than maximum bounds for both x and y axes.
        Prevents setting invalid boundary values that would create empty or inverted ranges.

        Args:
            name (str): Name of the parameter to set ('xmin', 'xmax', 'ymin', or 'ymax').
            value: New value for the parameter (will be converted to float).

        Returns:
            tuple[str, bool]: A tuple containing:
                - message (str): Status message or error description.
                - success (bool): True if parameter was set successfully, False otherwise.
        """
        p = self.parameters[name]
        old_value = p.value
        try:
            new_value = float(value)
        except ValueError:
            return "Value must be a float", False
        message, success = super().set_param_value(name, value)
        if success:
            if name == "xmax":
                xmin = self.parameters["xmin"].value
                if new_value <= xmin:
                    p.value = old_value
                    message = "xmax must be > xmin"
                    success = False
            elif name == "xmin":
                xmax = self.parameters["xmax"].value
                if new_value >= xmax:
                    p.value = old_value
                    message = "xmin must be < xmax"
                    success = False
            elif name == "ymax":
                ymin = self.parameters["ymin"].value
                if new_value <= ymin:
                    p.value = old_value
                    message = "ymax must be > ymin"
                    success = False
            elif name == "ymin":
                ymax = self.parameters["ymax"].value
                if new_value >= ymax:
                    p.value = old_value
                    message = "ymin must be < ymax"
                    success = False

        return message, success


    def calculate(self, x, y, ax=None, color=None, file_parameters=[]):
        """Filter data points to include only those within specified bounds.

        Removes data points where x is outside [xmin, xmax] or y is outside [ymin, ymax].
        Both conditions must be satisfied for a point to be retained.

        Args:
            x (numpy.ndarray): Array of x-coordinates (abscissa values).
            y (numpy.ndarray): Array of y-coordinates (ordinate values).
            ax (matplotlib.axes.Axes, optional): Matplotlib axes for plotting. Defaults to None.
            color: Color specification for plotting. Defaults to None.
            file_parameters (list): List of file-specific parameters. Defaults to [].

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: Filtered arrays (x2, y2) containing only data
                points that satisfy both x and y boundary conditions.
        """
        xmin = self.parameters["xmin"].value
        xmax = self.parameters["xmax"].value
        ymin = self.parameters["ymin"].value
        ymax = self.parameters["ymax"].value
        conditionx = (x > xmin) * (x < xmax)
        conditiony = (y > ymin) * (y < ymax)
        x2 = np.extract(conditionx * conditiony, x)
        y2 = np.extract(conditionx * conditiony, y)
        return x2, y2
