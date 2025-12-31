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
"""Module FileIOController

Extracted from QApplicationWindow to manage file I/O operations including
opening, saving, importing, and exporting data.

This class follows the Single Responsibility Principle by focusing exclusively
on file input/output operations.
"""
from __future__ import annotations

import logging
from os.path import dirname, isdir, join
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtWidgets import QFileDialog, QMessageBox

if TYPE_CHECKING:
    from pathlib import Path

    from RepTate.gui.QApplicationWindow import QApplicationWindow


class FileIOController:
    """Controls file input/output operations.

    This class extracts file I/O functionality from QApplicationWindow
    to reduce the god class size and improve maintainability.

    Attributes:
        parent: The parent QApplicationWindow instance.
        logger: Logger for this controller.
        dir_start: Starting directory for file dialogs.
    """

    def __init__(
        self, parent: "QApplicationWindow", dir_start: str | Path = ""
    ) -> None:
        """Initialize FileIOController.

        Args:
            parent: The parent QApplicationWindow instance.
            dir_start: Initial directory for file dialogs.
        """
        self.parent = parent
        self.logger = logging.getLogger(
            parent.logger.name + ".FileIOController"
        )
        self.dir_start = str(dir_start) if dir_start else parent.dir_start

    def open_file_names_dialog(
        self, ext_filter: str = "All Files (*)"
    ) -> list[str]:
        """Open a file dialog to select files.

        Args:
            ext_filter: File extension filter string.

        Returns:
            List of selected file paths.
        """
        qfdlg = QFileDialog(self.parent)
        options = qfdlg.options()
        dialogue_name = "Open"
        selected_files, _ = qfdlg.getOpenFileNames(
            self.parent, dialogue_name, self.dir_start, ext_filter, options=options
        )
        if selected_files:
            self.dir_start = dirname(selected_files[0])
            self.parent.dir_start = self.dir_start
        return selected_files

    def open_dataset(self) -> None:
        """Open files into a new dataset.

        Opens a file dialog and loads selected files into the current
        or a new dataset.
        """
        parent = self.parent
        # Build allowed extensions filter
        allowed_ext = ""
        for ftype in parent.filetypes.values():
            allowed_ext += f"{ftype.name} (*{ftype.extension});;"
        allowed_ext = allowed_ext.rstrip(";")

        paths_to_open = self.open_file_names_dialog(allowed_ext)
        if not paths_to_open:
            return

        parent.new_tables_from_files(paths_to_open)

    def save_view(self) -> None:
        """Save the current view data to text files.

        Opens a folder dialog and saves each active file's view data
        as a text file.
        """
        parent = self.parent
        dialogue_name = "Select Folder for Saving data in Current View as txt"
        folder = QFileDialog.getExistingDirectory(
            parent, dialogue_name, self.dir_start
        )
        if not isdir(folder):
            return

        ds = parent.DataSettabWidget.currentWidget()
        if ds is None:
            return

        nfout = 0
        for f in ds.files:
            if not f.active:
                continue

            series = f.data_table.series
            all_views_data = {}
            max_row_len = 0
            max_view_name_len = 0

            for nx, view in enumerate(parent.multiviews):
                max_view_name_len = max(max_view_name_len, len(view.name) + 3)
                data_view = []
                for i in range(view.n):
                    x, y = series[nx][i].get_data()
                    if i > 0 and np.array_equal(x, data_view[0][0]):
                        # If x1=x2, only use y
                        data_view.append([y])
                    else:
                        data_view.append([x, y])
                    max_row_len = max(
                        max_row_len, len(series[nx][i].get_data()[0])
                    )
                all_views_data[view.name] = data_view

            nviews = len(parent.multiviews)
            output_path = join(folder, f.file_name_short) + "_VIEW.txt"

            with open(output_path, "w") as fout:
                # Header with file parameters
                fout.write(
                    "#view(s)=[%s];"
                    % (", ".join([v.name for v in parent.multiviews]))
                )
                for pname in f.file_parameters:
                    fout.write(f"{pname}={f.file_parameters[pname]};")
                fout.write("\n")

                # Column titles
                field_width = []
                for view_name in all_views_data:
                    view = parent.views[view_name]
                    snames_index = 0
                    for xy in all_views_data[view_name]:
                        if len(xy) > 1:
                            # Case where there is x and y series
                            fw1 = max(len(view.x_label), 15)
                            fw2 = max(len(view.snames[snames_index]), 15)
                            fout.write(f"{view.x_label:<{fw1}s}\t")
                            fout.write(f"{view.snames[snames_index]:<{fw2}s}\t")
                            field_width.extend([fw1, fw2])
                        else:
                            # Case where there is y series only
                            fw = max(len(view.snames[snames_index]), 15)
                            fout.write(f"{view.snames[snames_index]:{fw}s}\t")
                            field_width.append(fw)
                        snames_index += 1
                fout.write("\n")

                # Data lines
                for i in range(max_row_len):
                    fw_index = 0
                    for view_name in all_views_data:
                        data_view = all_views_data[view_name]
                        for xy in data_view:
                            if len(xy) > 1:
                                fw1 = field_width[fw_index]
                                fw2 = field_width[fw_index + 1]
                                fw_index += 2
                                if i < len(xy[0]):
                                    fout.write(f"{xy[0][i]:{fw1}.6g}\t")
                                    fout.write(f"{xy[1][i]:{fw2}.6g}\t")
                                else:
                                    fout.write(f"{'':{fw1}s}\t")
                                    fout.write(f"{'':{fw2}s}\t")
                            else:
                                fw = field_width[fw_index]
                                fw_index += 1
                                if i < len(xy[0]):
                                    fout.write(f"{xy[0][i]:{fw}.6g}\t")
                                else:
                                    fout.write(f"{'':{fw}s}\t")
                    fout.write("\n")
            nfout += 1

        if nfout > 0:
            self.logger.info(f"Saved {nfout} view file(s) to {folder}")

    def save_current_dataset(self) -> None:
        """Save the current dataset to files.

        Opens a save dialog and exports the current dataset.
        """
        parent = self.parent
        ds = parent.DataSettabWidget.currentWidget()
        if ds is None:
            QMessageBox.warning(
                parent, "Save Dataset", "No dataset to save"
            )
            return

        # Delegate to dataset's save method
        ds.save_dataset()

    def save_shift_factors(self) -> None:
        """Save shift factors to a file.

        Exports the current shift factors for all files in the active dataset.
        """
        parent = self.parent
        ds = parent.DataSettabWidget.currentWidget()
        if ds is None:
            return

        fnames = [f.file_name_short for f in ds.files]
        factorsx = [f.data_table.shifts_x for f in ds.files]
        factorsy = [f.data_table.shifts_y for f in ds.files]

        dialogue_name = "Save Shift Factors"
        file_path, _ = QFileDialog.getSaveFileName(
            parent, dialogue_name, self.dir_start, "Text Files (*.txt)"
        )
        if not file_path:
            return

        with open(file_path, "w") as fout:
            fout.write("# Shift factors\n")
            fout.write("# filename\tshift_x\tshift_y\n")
            for fname, sx, sy in zip(fnames, factorsx, factorsy):
                for i in range(len(sx)):
                    fout.write(f"{fname}\t{sx[i]}\t{sy[i]}\n")

        self.logger.info(f"Saved shift factors to {file_path}")

    def import_from_excel(self) -> dict | None:
        """Import data from an Excel file.

        Returns:
            Dictionary with imported data or None if cancelled/failed.
        """
        from RepTate.gui.ImportExcelWindow import ImportExcelWindow

        parent = self.parent
        for ftype in parent.filetypes.values():
            break  # Get first filetype

        if parent.excel_import_gui is None:
            parent.excel_import_gui = ImportExcelWindow(parent=parent, ftype=ftype)

        if parent.excel_import_gui.exec_():
            return parent.excel_import_gui.get_data()
        return None

    def import_from_pasted(self) -> dict | None:
        """Import data from pasted text.

        Returns:
            Dictionary with imported data or None if cancelled/failed.
        """
        from RepTate.gui.ImportFromPastedWindow import ImportFromPastedWindow

        parent = self.parent
        for ftype in parent.filetypes.values():
            break  # Get first filetype

        if parent.pasted_import_gui is None:
            parent.pasted_import_gui = ImportFromPastedWindow(parent=parent, ftype=ftype)
            parent.count_pasted_data = 1

        fname = f"pasted_data_{parent.count_pasted_data}"
        parent.pasted_import_gui.set_fname_dialog(fname)

        if parent.pasted_import_gui.exec_():
            return parent.pasted_import_gui.get_data()
        return None

    def print_plot(self) -> None:
        """Print or save the current plot to a file."""
        parent = self.parent
        file_path, _ = QFileDialog.getSaveFileName(
            parent,
            "Export plot",
            self.dir_start,
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)"
        )
        if file_path:
            parent.figure.savefig(file_path)
            self.logger.info(f"Saved plot to {file_path}")

    def copy_chart_to_clipboard(self) -> None:
        """Copy the current chart to the clipboard."""
        import io

        from PySide6.QtGui import QImage
        from PySide6.QtWidgets import QApplication

        parent = self.parent
        buf = io.BytesIO()
        parent.figure.savefig(buf, format='png', dpi=150)
        buf.seek(0)

        image = QImage()
        image.loadFromData(buf.getvalue())

        clipboard = QApplication.clipboard()
        clipboard.setImage(image)

        self.logger.info("Chart copied to clipboard")
