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
"""Module ViewCoordinator

Extracted from QApplicationWindow to manage view switching, multiplot
configuration, and plot coordinate systems.

This class follows the Single Responsibility Principle by focusing exclusively
on view and plot management.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from RepTate.gui.QApplicationWindow import QApplicationWindow


class ViewCoordinator:
    """Coordinates view switching and multiplot management.

    This class extracts view-related functionality from QApplicationWindow
    to reduce the god class size and improve maintainability.

    Attributes:
        parent: The parent QApplicationWindow instance.
        logger: Logger for this coordinator.
    """

    def __init__(self, parent: "QApplicationWindow") -> None:
        """Initialize ViewCoordinator.

        Args:
            parent: The parent QApplicationWindow instance.
        """
        self.parent = parent
        self.logger = logging.getLogger(
            parent.logger.name + ".ViewCoordinator"
        )

    def setup_multiplot(self, nplots: int, ncols: int) -> None:
        """Setup the multiplot configuration.

        Args:
            nplots: Number of plots to display.
            ncols: Number of columns in the multiplot grid.
        """
        from RepTate.core.MultiView import MultiView, PlotOrganizationType

        parent = self.parent
        parent.multiplots = MultiView(
            PlotOrganizationType.OptimalRow, nplots, ncols, parent
        )
        parent.multiplots.plotselecttabWidget.setCurrentIndex(parent.current_viewtab)
        parent.figure = parent.multiplots.figure
        parent.axarr = parent.multiplots.axarr
        parent.canvas = parent.multiplots.canvas

    def delete_multiplot(self) -> None:
        """Delete the multiplot object."""
        del self.parent.multiplots

    def set_views(self) -> None:
        """Set up the available views for the application.

        Called during initialization to configure the default view.
        """
        parent = self.parent
        for view_name in parent.views:
            # Index 0 is the default view
            parent.current_view = parent.views[view_name]
            break

    def switch_view(self, name: str) -> bool:
        """Switch to a named view.

        Args:
            name: The name of the view to switch to.

        Returns:
            True if view was switched successfully, False otherwise.
        """
        parent = self.parent
        if name in parent.views:
            parent.current_view = parent.views[name]
            if parent.current_viewtab == 0:
                pass  # Main view
            else:
                parent.multiviews[parent.current_viewtab - 1] = parent.views[name]

            # Update the plots
            self.update_all_plots()
            return True
        else:
            self.logger.warning(f'View "{name}" not found')
            return False

    def update_all_plots(self) -> None:
        """Update all plots from all datasets."""
        for ds in self.parent.datasets.values():
            ds.do_plot()

    def refresh_plot(self) -> None:
        """Refresh the current plot."""
        self.switch_view(self.parent.current_view.name)

    def change_nplots(self, new_nplots: int) -> None:
        """Change the number of plots displayed.

        Args:
            new_nplots: The new number of plots to display.
        """
        parent = self.parent
        parent.nplots = new_nplots
        parent.multiplots.reorg_fig(new_nplots)

    def get_current_axes(self) -> Any:
        """Get the current matplotlib axes.

        Returns:
            The current axes object.
        """
        parent = self.parent
        if parent.current_viewtab == 0:
            return parent.axarr
        else:
            return parent.axarr[parent.current_viewtab - 1]

    def change_view(self, x_vis: bool = False, y_vis: bool = False) -> None:
        """Change plot view with visibility settings.

        Args:
            x_vis: Whether X axis is visible.
            y_vis: Whether Y axis is visible.
        """
        parent = self.parent
        parent.update_Qplot()

    def handle_view_all_sets(self, checked: bool) -> None:
        """Handle toggling view all datasets.

        Args:
            checked: Whether to show all datasets.
        """
        parent = self.parent
        parent.update_Qplot()

        if parent.actionView_All_Sets.isChecked():
            # All datasets are now visible
            pass
        else:
            # Only current dataset visible
            ds = parent.DataSettabWidget.currentWidget()
            if ds is not None:
                parent.dataset_actions_disabled(
                    len(ds.files) == 0
                )
            else:
                parent.dataset_actions_disabled(True)

    def handle_view_all_theories(self, checked: bool) -> None:
        """Handle toggling view all theories.

        Args:
            checked: Whether to show all theories.
        """
        self.parent.update_Qplot()

    def setup_matplotlib_connections(self) -> None:
        """Setup matplotlib canvas event connections."""
        parent = self.parent
        parent.figure.canvas.mpl_connect("resize_event", parent.resizeplot)
        parent.figure.canvas.mpl_connect("scroll_event", parent.zoom_wheel)
        parent.figure.canvas.mpl_connect("button_press_event", parent.on_press)
        parent.figure.canvas.mpl_connect("motion_notify_event", parent.on_motion)
        parent.figure.canvas.mpl_connect("button_release_event", parent.onrelease)

    def get_view_by_name(self, name: str) -> Any | None:
        """Get a view by its name.

        Args:
            name: The view name.

        Returns:
            The view object or None if not found.
        """
        return self.parent.views.get(name)

    def get_available_views(self) -> list[str]:
        """Get list of available view names.

        Returns:
            List of view names.
        """
        return list(self.parent.views.keys())

    def set_view_tools(self, view_name: str) -> None:
        """Set tools for a specific view.

        This is a placeholder that can be overridden in child applications.
        Called when the view is changed.

        Args:
            view_name: The name of the view.
        """
        pass

    def change_ax_view(self, n_ax: int, view_name: str) -> None:
        """Change the view for a specific axis.

        Args:
            n_ax: The axis index.
            view_name: The view name to switch to.
        """
        parent = self.parent
        tab_ind = parent.multiplots.plotselecttabWidget.currentIndex()
        if tab_ind == 0:
            parent.multiviews[n_ax] = parent.views[view_name]
        else:
            parent.multiviews[tab_ind - 1] = parent.views[view_name]
        self.refresh_plot()
