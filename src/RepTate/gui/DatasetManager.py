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
"""Module DatasetManager

Extracted from QApplicationWindow to manage dataset creation, manipulation,
and lifecycle operations.

This class follows the Single Responsibility Principle by focusing exclusively
on dataset management.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox, QTreeWidgetItem

from RepTate.gui.DataSetWidgetItem import DataSetWidgetItem

if TYPE_CHECKING:
    from RepTate.core.DataTable import DataTable
    from RepTate.gui.QApplicationWindow import QApplicationWindow
    from RepTate.gui.QDataSet import QDataSet


class DatasetManager:
    """Manages dataset creation, manipulation, and lifecycle.

    This class extracts dataset-related functionality from QApplicationWindow
    to reduce the god class size and improve maintainability.

    Attributes:
        parent: The parent QApplicationWindow instance.
        logger: Logger for this manager.
    """

    def __init__(self, parent: "QApplicationWindow") -> None:
        """Initialize DatasetManager.

        Args:
            parent: The parent QApplicationWindow instance.
        """
        self.parent = parent
        self.logger = logging.getLogger(
            parent.logger.name + ".DatasetManager"
        )

    def create_empty_dataset(self, tabname: str = "") -> "QDataSet":
        """Create a new empty dataset.

        Args:
            tabname: Optional name for the dataset tab.

        Returns:
            The newly created QDataSet.
        """
        from RepTate.gui.QDataSet import QDataSet

        parent = self.parent
        parent.num_datasets += 1
        num = parent.num_datasets
        ds_name = f"Set{num}"
        ds = QDataSet(name=ds_name, parent=parent)
        parent.datasets[ds_name] = ds

        if tabname == "":
            tabname = ds_name
        ind = parent.DataSettabWidget.addTab(ds, tabname)

        # Set the new tab as active
        parent.DataSettabWidget.setCurrentIndex(ind)

        # Define the tab column names (header)
        dfile = list(parent.filetypes.values())[0]
        dataset_header = dfile.basic_file_parameters[:]
        dataset_header.insert(0, "File")
        ds.DataSettreeWidget.setHeaderItem(QTreeWidgetItem(dataset_header))
        ds.DataSettreeWidget.setSortingEnabled(True)

        hd = ds.DataSettreeWidget.header()
        hd.setSectionsClickable(True)
        w = ds.DataSettreeWidget.width()
        w /= hd.count()
        for i in range(hd.count()):
            hd.resizeSection(i, int(w))

        # Define the inspector column names (header)
        if num == 1:
            inspect_header = [
                f"{a} [{b}]" for a, b in zip(dfile.col_names, dfile.col_units)
            ]
            parent.inspector_table.setHorizontalHeaderLabels(inspect_header)

        return parent.DataSettabWidget.widget(ind)

    def add_table_to_current_dataset(
        self, dt: "DataTable", ext: str
    ) -> None:
        """Add a DataTable to the current dataset.

        Args:
            dt: The DataTable to add.
            ext: The file extension.
        """
        parent = self.parent
        ds = parent.DataSettabWidget.currentWidget()
        if ds is None:
            self.logger.warning("No current dataset to add table to")
            return

        ds.files.append(dt)
        ds.current_file = dt
        ds.num_files += 1

        # Prepare tree widget item data
        lnew = []
        for param in parent.filetypes[ext].basic_file_parameters:
            s_param = dt.file_parameters.get(param, None)
            if s_param is None:
                dt.file_parameters[param] = "0"
                s_param = "0"
                message = f"Warning: File parameter '{param}' not found in {dt.file_name_short}"
                self.logger.warning(message)
            else:
                try:
                    s_param = f"{float(s_param):.3g}"
                except ValueError:
                    s_param = str(dt.file_parameters[param])
            lnew.append(s_param)

        file_name_short = dt.file_name_short
        lnew.insert(0, file_name_short)
        newitem = DataSetWidgetItem(ds.DataSettreeWidget, lnew)
        newitem.setCheckState(0, Qt.CheckState.Checked)
        parent.dataset_actions_disabled(False)

    def close_dataset(self, index: int) -> bool:
        """Close a dataset at the specified index.

        Args:
            index: The tab index of the dataset to close.

        Returns:
            True if the dataset was closed, False if cancelled.
        """
        parent = self.parent
        ds = parent.DataSettabWidget.widget(index)
        if ds is None:
            return False

        # Remove dataset from registry
        if ds.name in parent.datasets:
            del parent.datasets[ds.name]

        # Remove tab
        parent.DataSettabWidget.removeTab(index)

        # Clean up dataset
        ds.deleteLater()

        # Check if any datasets remain
        if parent.DataSettabWidget.count() == 0:
            parent.dataset_actions_disabled(True)

        return True

    def get_current_dataset(self) -> "QDataSet | None":
        """Get the currently active dataset.

        Returns:
            The current QDataSet or None if no dataset is active.
        """
        return self.parent.DataSettabWidget.currentWidget()

    def ensure_dataset_exists(self) -> "QDataSet":
        """Ensure at least one dataset exists, creating one if necessary.

        Returns:
            The current or newly created QDataSet.
        """
        if self.parent.DataSettabWidget.count() == 0:
            return self.create_empty_dataset()
        return self.get_current_dataset()

    def check_no_param_missing(
        self, newtables: list["DataTable"], ext: str
    ) -> None:
        """Check for missing parameters in new tables.

        Args:
            newtables: List of DataTable objects to check.
            ext: The file extension for looking up expected parameters.
        """
        parent = self.parent
        for dt in newtables:
            e_list = []
            for param in parent.filetypes[ext].basic_file_parameters[:]:
                try:
                    temp = dt.file_parameters[param]
                    if temp == "" or temp == "\n":
                        e_list.append(param)
                except KeyError:
                    e_list.append(param)
            if len(e_list) > 0:
                message = (
                    f"Parameter(s) {{{', '.join(e_list)}}} not found in file "
                    f"'{dt.file_name_short}'\n Value(s) set to 0"
                )
                self.logger.warning(message)
                for e_param in e_list:
                    dt.file_parameters[e_param] = "0"

    def new_tables_from_files(self, paths_to_open: list[str]) -> None:
        """Create new DataTables from a list of file paths.

        Args:
            paths_to_open: List of file paths to open.
        """
        ds = self.ensure_dataset_exists()
        ds.DataSettreeWidget.blockSignals(True)

        success, newtables, ext = ds.do_open(paths_to_open)
        if success is True:
            self.check_no_param_missing(newtables, ext)
            for dt in newtables:
                self.add_table_to_current_dataset(dt, ext)
            ds.do_plot()
            self.parent.update_Qplot()
            ds.set_table_icons(ds.table_icon_list)
        else:
            QMessageBox.about(self.parent, "Open", str(success))

        ds.DataSettreeWidget.blockSignals(False)

    def reload_current_dataset(self) -> None:
        """Reload data for the current dataset."""
        ds = self.get_current_dataset()
        if ds is not None:
            ds.reload_data()

    def update_all_datasets_plots(self) -> None:
        """Update plots for all datasets."""
        for ds in self.parent.datasets.values():
            ds.do_plot()
