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
"""Module ToolPowerLaw

Tool to check the power law of some data
"""
from RepTate.core.Parameter import Parameter, ParameterType, OptType
from RepTate.gui.QTool import QTool


class ToolPowerLaw(QTool):
    """Check the power law of the data (or some part of it) by dividing the y coordinate by the x coordinate 
raised to n.
    """

    toolname = "PowerLaw"
    description = "Check the power law of the data"
    citations = []
    # html_help_file = 'http://reptate.readthedocs.io/manual/Tools/template.html'

    def __init__(self, name="", parent_app=None):
        """**Constructor**"""
        super().__init__(name, parent_app)
        self.parameters["n"] = Parameter(
            name="n",
            value=1,
            description="Power law exponent",
            type=ParameterType.real,
            opt_type=OptType.const,
        )

        self.update_parameter_table()
        self.parent_application.update_all_ds_plots()

        # add widgets specific to the Tool here:


    def destructor(self):
        """If the tool needs to clear up memory in a very special way, fill up the contents of this function.
If not, you can safely delete it."""
        pass

    def calculate(self, x, y, ax=None, color=None, file_parameters=[]):
        """Check power law behavior by normalizing y by x raised to power n.

        Divides y-coordinates by x^n to verify if the data follows a power law
        relationship. If the data follows y = A*x^n, the result will be a horizontal
        line at y = A.

        Args:
            x (numpy.ndarray): Array of x-coordinates (independent variable).
            y (numpy.ndarray): Array of y-coordinates (dependent variable).
            ax (matplotlib.axes.Axes, optional): Matplotlib axes for plotting. Defaults to None.
            color: Color specification for plotting. Defaults to None.
            file_parameters (list): List of file-specific parameters. Defaults to [].

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: Tuple (x, y/x^n) where the y values are
                normalized by the power law scaling factor.
        """
        n = self.parameters["n"].value
        return x, y / x ** n

