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
"""Module ToolFindPeaks

FindPeaks file for creating a new Tool
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import numpy.typing as npt
from PySide6.QtGui import QIcon

from RepTate.core.fitting.nlsq_optimize import curve_fit
from RepTate.core.Parameter import Parameter, ParameterType
from RepTate.gui.QTool import QTool


class ToolFindPeaks(QTool):
    """Find peaks (maxima or minima) in the data, as represented by the current view. The option to find the maxima or the minima is specified by the min/max check button (the minpeaks parameter in the command line version). The **threshold** controls the relative height that a peak must have (with respect to the data span) in order to be detected. The **minimum_distance** parameter controls how far from each other the peaks must be in order to be distinguished. The returned peaks correspond to the maximum/minimum data point in the current view. Alternatively, the user can select to fit a parabola to the peaks and find the analytical maximum or minimum of the parabola. The parameter **minimum_distance** also controls the number of points around the maximum data point used to fit the parabola. The peaks are returned in the Tool information area and shown as symbols in the chart.

    The algorithm used to find the peaks can be very inaccurate and slow if the data is noisy and has many local peaks. It is recommended to smooth the data first before finding the peaks.
    """

    toolname = "Find Peaks"
    description = "Find Peaks in current data/view"
    citations = []
    # html_help_file = 'http://reptate.readthedocs.io/manual/Tools/FindPeaks.html'

    def __init__(self, name: str = "", parent_app: object | None = None) -> None:
        """**Constructor**"""
        super().__init__(name, parent_app)
        # self.function = self.findpeaks  # main Tool function
        self.parameters["threshold"] = Parameter(
            name="threshold",
            value=0.3,
            description="threshold for peak detection",
            type=ParameterType.real,
        )
        self.parameters["minimum_distance"] = Parameter(
            name="minimum_distance",
            value=5,
            description="minimum distance (in datapoints) between peaks",
            type=ParameterType.integer,
        )
        self.parameters["minpeaks"] = Parameter(
            name="minpeaks",
            value=False,
            description="Find minimum peaks",
            type=ParameterType.boolean,
            display_flag=False,
        )
        self.parameters["parabola"] = Parameter(
            name="parabola",
            value=False,
            description="Fit Parabola to peak",
            type=ParameterType.boolean,
            display_flag=False,
        )
        self.seriesarray: list[object] = []
        self.axarray: list[object] = []

        self.update_parameter_table()
        self.tb.addSeparator()
        self.minpeaks = self.tb.addAction("Minimum peaks")
        self.minpeaks.setCheckable(True)
        self.handle_minpeaks_button(checked=False)
        connection_id = self.minpeaks.triggered.connect(self.handle_minpeaks_button)
        self.parabola = self.tb.addAction("Fit Parabola")
        self.parabola.setIcon(QIcon(":/Icon8/Images/new_icons/icons8-bell-curve.png"))
        self.parabola.setCheckable(True)
        self.parabola.setChecked(False)
        connection_id = self.parabola.triggered.connect(self.handle_parabola_button)
        self.parent_application.update_all_ds_plots()

        # add widgets specific to the Tool here:

    def handle_minpeaks_button(self, checked: bool) -> None:
        """Handle the minimum/maximum peaks toggle button.

        Updates the button icon and internal parameter based on whether the tool should
        search for minimum peaks (valleys) or maximum peaks. Triggers a plot update.

        Args:
            checked (bool): If True, search for minimum peaks; if False, search for maximum peaks.

        Returns:
            None
        """
        if checked:
            self.minpeaks.setIcon(
                QIcon(":/Icon8/Images/new_icons/icons8-peak-minimum.png")
            )
        else:
            self.minpeaks.setIcon(
                QIcon(":/Icon8/Images/new_icons/icons8-peak-maximum.png")
            )
        self.minpeaks.setChecked(checked)
        self.set_param_value("minpeaks", checked)
        self.parent_application.update_all_ds_plots()

    def handle_parabola_button(self, checked: bool) -> None:
        """Handle the parabola fitting toggle button.

        Controls whether peaks are refined by fitting a parabola to nearby points
        and finding the analytical extremum, or simply using the maximum data point.

        Args:
            checked (bool): If True, fit parabolas to peaks; if False, use raw data maxima.

        Returns:
            None
        """
        self.parabola.setChecked(checked)
        self.set_param_value("parabola", checked)
        self.parent_application.update_all_ds_plots()

    def clean_graphic_stuff(self) -> None:
        """Clean up graphical elements (peak markers) from all plots.

        Removes all peak marker symbols that were previously added to plots and clears
        the internal tracking arrays. Called when the tool is destroyed or reset.

        Returns:
            None
        """
        for s, a in zip(self.seriesarray, self.axarray):
            # a.lines.remove(s)
            s.remove()
        self.seriesarray.clear()
        self.axarray.clear()

    def destructor(self) -> None:
        """This is called when the Tool tab is closed"""
        self.clean_graphic_stuff()

    def calculate(
        self,
        x: npt.NDArray[np.floating],
        y: npt.NDArray[np.floating],
        ax: object | None = None,
        color: object | None = None,
        file_parameters: Sequence[Parameter] | None = None,
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Find and mark peaks (maxima or minima) in the data.

        Detects peaks using threshold-based filtering and minimum distance separation.
        Optionally refines peak positions by fitting parabolas. Displays peak locations
        in the tool output and plots them as diamond markers on the graph.

        Args:
            x (numpy.ndarray): Array of x-coordinates (abscissa values).
            y (numpy.ndarray): Array of y-coordinates (ordinate values).
            ax (matplotlib.axes.Axes, optional): Matplotlib axes for plotting peak markers.
                Defaults to None.
            color: Color specification for peak markers. Defaults to None.
            file_parameters (Sequence[Parameter], optional): List of file-specific parameters.
                Defaults to None.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: Original input arrays (x, y) unchanged.
                Peak information is displayed in the tool output area and plotted as markers.
        """
        if file_parameters is None:
            file_parameters = []
        threshold = self.parameters["threshold"].value
        minimum_distance = self.parameters["minimum_distance"].value
        minpeaks = self.parameters["minpeaks"].value
        parabola = self.parameters["parabola"].value
        if minpeaks:
            y = -y
        thresholdnow = threshold * (np.max(y) - np.min(y)) + np.min(y)
        dy = np.diff(y)
        (zeros,) = np.where(dy == 0)
        if len(zeros) == len(y) - 1:
            print("", end="")
            if minpeaks:
                y = -y
            return x, y
        while len(zeros):
            zerosr = np.hstack([dy[1:], 0.0])
            zerosl = np.hstack([0.0, dy[:-1]])
            dy[zeros] = zerosr[zeros]
            (zeros,) = np.where(dy == 0)
            dy[zeros] = zerosl[zeros]
            (zeros,) = np.where(dy == 0)
        peaks = np.where(
            (np.hstack([dy, 0.0]) < 0.0)
            & (np.hstack([0.0, dy]) > 0.0)
            & (y > thresholdnow)
        )[0]
        if peaks.size > 1 and minimum_distance > 1:
            highest = peaks[np.argsort(y[peaks])][::-1]
            rem = np.ones(y.size, dtype=bool)
            rem[peaks] = False

            for peak in highest:
                if not rem[peak]:
                    sl = slice(
                        max(0, peak - minimum_distance), peak + minimum_distance + 1
                    )
                    rem[sl] = True
                    rem[peak] = False
            peaks = np.arange(y.size)[~rem]

        xp = np.zeros(len(peaks))
        yp = np.zeros(len(peaks))
        if minpeaks:
            self.Qprint("<b>%d</b> Minimum(s) found" % len(peaks))
        else:
            self.Qprint("<b>%d</b> Maximum(s) found" % len(peaks))
        if minpeaks:
            y = -y
        # table='''<table border="1" width="100%">'''
        # table+='''<tr><th>x</th><th>y</th></tr>'''
        table = [
            ["%-10s" % "x", "%-10s" % "y"],
        ]
        if parabola:
            ################################################################
            # Fit parabola to each peak and find analytical position of peak
            func = lambda xx, a, tau, c: a * ((xx - tau) ** 2) + c
            for i, d in enumerate(peaks):
                x_data = x[d - minimum_distance // 2 : d + minimum_distance // 2 + 1]
                y_data = y[d - minimum_distance // 2 : d + minimum_distance // 2 + 1]
                tau = x[d]  # approximation of tau (peak position in x)
                c = y[d]  # approximation of peak amplitude
                a = (
                    np.sign(c)
                    * (-1)
                    * (np.sqrt(abs(c)) / (x_data[-1] - x_data[0])) ** 2
                )
                p0 = (a, tau, c)
                popt, pcov = curve_fit(func, x_data, y_data, p0, maxfev=5000)
                xp[i], yp[i] = popt[1:3]
                # table+='''<tr><td>%.4e</td><td>%.4e</td></tr>'''%(xp[i],yp[i])
                table.append(["%-10.4e" % xp[i], "%-10.4e" % yp[i]])
            #################################################################
        else:
            for i, d in enumerate(peaks):
                xp[i] = x[d]
                yp[i] = y[d]
                # table+='''<tr><td>%.4e</td><td>%.4e</td></tr>'''%(xp[i],yp[i])
                table.append(["%-10.4e" % xp[i], "%-10.4e" % yp[i]])
        # table+='''</table><br>'''
        if len(peaks) > 0:
            self.Qprint(table)
        s = ax.plot(xp, yp)[0]
        s.set_marker("D")
        s.set_linestyle("")
        s.set_markerfacecolor(color)
        s.set_markeredgecolor("black")
        s.set_markeredgewidth(3)
        s.set_markersize(12)
        s.set_alpha(0.5)
        self.seriesarray.append(s)
        self.axarray.append(ax)
        return x, y
