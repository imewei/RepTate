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
"""Module MenuManager

Extracted from QApplicationWindow to manage menu and toolbar setup, action
connections, and menu state management.

This class follows the Single Responsibility Principle by focusing exclusively
on menu and action management.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from PySide6 import QtCore
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QMenu, QToolBar, QToolButton

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget


class MenuManager:
    """Manages menu creation, action connections, and toolbar setup.

    This class extracts menu-related functionality from QApplicationWindow
    to reduce the god class size and improve maintainability.

    Attributes:
        parent: The parent QApplicationWindow instance.
        action_handlers: Dictionary mapping action names to handler functions.
    """

    def __init__(self, parent: "QWidget") -> None:
        """Initialize MenuManager.

        Args:
            parent: The parent QApplicationWindow instance.
        """
        self.parent = parent
        self.action_handlers: dict[str, Callable] = {}

    def setup_data_inspector_toolbar(self, toolbar: QToolBar) -> None:
        """Setup the data inspector toolbar with actions.

        Args:
            toolbar: The QToolBar to configure.
        """
        toolbar.setIconSize(QtCore.QSize(24, 24))
        toolbar.addAction(self.parent.actionCopy)
        toolbar.addAction(self.parent.actionPaste)
        toolbar.addSeparator()
        toolbar.addAction(self.parent.actionShiftVertically)
        toolbar.addAction(self.parent.actionShiftHorizontally)
        toolbar.addAction(self.parent.actionViewShiftFactors)
        toolbar.addAction(self.parent.actionSaveShiftFactors)
        toolbar.addAction(self.parent.actionResetShiftFactors)

    def setup_tool_toolbar(self, toolbar: QToolBar) -> None:
        """Setup the tool toolbar with actions.

        Args:
            toolbar: The QToolBar to configure.
        """
        toolbar.setIconSize(QtCore.QSize(24, 24))
        toolbar.addAction(self.parent.actionNew_Tool)

    def setup_dataset_toolbar(self, toolbar: QToolBar) -> None:
        """Setup the dataset toolbar with actions and menus.

        Args:
            toolbar: The QToolBar to configure.
        """
        toolbar.setIconSize(QtCore.QSize(24, 24))
        toolbar.addAction(self.parent.actionNew_Empty_Dataset)

        # File operations menu button
        tbut = QToolButton()
        tbut.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        tbut.setDefaultAction(self.parent.actionNew_Dataset_From_File)
        menu = QMenu(self.parent)
        menu.addAction(self.parent.actionAddDummyFiles)
        menu.addAction(self.parent.actionAdd_File_With_Function)
        menu.addAction(self.parent.action_import_from_excel)
        menu.addAction(self.parent.action_import_from_pasted)
        menu.addAction(self.parent.actionSaveDataSet)
        tbut.setMenu(menu)
        toolbar.addWidget(tbut)

        # View all sets menu button
        tbut2 = QToolButton()
        tbut2.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        tbut2.setDefaultAction(self.parent.actionView_All_Sets)
        menu2 = QMenu(self.parent)
        menu2.addAction(self.parent.actionView_All_SetTheories)
        tbut2.setMenu(menu2)
        toolbar.addWidget(tbut2)

        toolbar.addAction(self.parent.actionMarkerSettings)
        toolbar.addAction(self.parent.actionReload_Data)
        toolbar.addAction(self.parent.actionInspect_Data)
        toolbar.addAction(self.parent.actionShowFigureTools)

    def setup_autoscale_action(self, toolbar: QToolBar) -> None:
        """Add autoscale action to toolbar.

        Args:
            toolbar: The QToolBar to add the action to.

        Returns:
            The created autoscale action.
        """
        self.parent.actionAutoscale = toolbar.addAction(
            QIcon(":/Images/Images/new_icons/icons8-padlock-96.png"), "Lock XY axes"
        )
        self.parent.actionAutoscale.setCheckable(True)

    def connect_actions(self) -> None:
        """Connect all action signals to their handlers.

        This method sets up all the signal-slot connections for menu actions.
        """
        parent = self.parent

        # Figure tools and data inspector
        parent.actionShowFigureTools.triggered.connect(parent.viewMPLToolbar)
        parent.actionInspect_Data.triggered.connect(parent.showDataInspector)

        # Dataset actions
        parent.actionNew_Empty_Dataset.triggered.connect(
            parent.handle_createNew_Empty_Dataset
        )
        parent.actionNew_Dataset_From_File.triggered.connect(parent.openDataset)
        parent.actionAddDummyFiles.triggered.connect(parent.addDummyFiles)
        parent.actionAdd_File_With_Function.triggered.connect(parent.addFileFunction)
        parent.action_import_from_excel.triggered.connect(
            parent.handle_action_import_from_excel
        )
        parent.action_import_from_pasted.triggered.connect(
            parent.handle_action_import_from_pasted
        )
        parent.actionSaveDataSet.triggered.connect(
            parent.handle_action_save_current_dataset
        )
        parent.actionReload_Data.triggered.connect(parent.handle_actionReload_Data)
        parent.actionAutoscale.triggered.connect(parent.handle_actionAutoscale)

        # Tool actions
        parent.actionNew_Tool.triggered.connect(parent.handle_actionNewTool)
        parent.TooltabWidget.tabCloseRequested.connect(
            parent.handle_toolTabCloseRequested
        )
        parent.qtabbar.tabMoved.connect(parent.handle_toolTabMoved)

        # View actions
        parent.viewComboBox.currentIndexChanged.connect(parent.handle_change_view)
        parent.actionSave_View.triggered.connect(parent.save_view)
        parent.sp_nviews.valueChanged.connect(parent.sp_nviews_valueChanged)

        # Dataset tab actions
        parent.DataSettabWidget.tabCloseRequested.connect(
            parent.close_data_tab_handler
        )
        parent.DataSettabWidget.tabBarDoubleClicked.connect(
            parent.handle_doubleClickTab
        )
        parent.DataSettabWidget.currentChanged.connect(parent.handle_currentChanged)
        parent.actionView_All_Sets.toggled.connect(parent.handle_actionView_All_Sets)
        parent.actionView_All_SetTheories.triggered.connect(
            parent.handle_actionView_All_SetTheories
        )

        # Shift actions
        parent.actionShiftVertically.triggered.connect(
            parent.handle_actionShiftTriggered
        )
        parent.actionShiftHorizontally.triggered.connect(
            parent.handle_actionShiftTriggered
        )
        parent.actionViewShiftFactors.triggered.connect(
            parent.handle_actionViewShiftTriggered
        )
        parent.actionSaveShiftFactors.triggered.connect(
            parent.handle_actionSaveShiftTriggered
        )
        parent.actionResetShiftFactors.triggered.connect(
            parent.handle_actionResetShiftTriggered
        )

        # Data inspector
        parent.DataInspectordockWidget.visibilityChanged.connect(
            parent.handle_inspectorVisibilityChanged
        )

        # Marker settings
        parent.actionMarkerSettings.triggered.connect(
            parent.handle_actionMarkerSettings
        )

        # Copy/Paste
        parent.actionCopy.triggered.connect(parent.inspector_table.copy)
        parent.actionPaste.triggered.connect(parent.inspector_table.paste)

    def set_dataset_actions_disabled(self, state: bool) -> None:
        """Enable or disable dataset-related actions.

        Args:
            state: True to disable actions, False to enable.
        """
        parent = self.parent
        parent.actionMarkerSettings.setDisabled(state)
        parent.actionSave_View.setDisabled(state)
        parent.actionReload_Data.setDisabled(state)
        parent.actionInspect_Data.setDisabled(state)
        parent.actionShowFigureTools.setDisabled(state)
        parent.actionView_All_Sets.setDisabled(state)
        parent.actionAutoscale.setDisabled(state)

        # Also disable theory actions on datasets
        for ds in parent.datasets.values():
            ds.actionNew_Theory.setDisabled(state)

    def update_autoscale_icon(self, checked: bool) -> None:
        """Update autoscale action icon based on state.

        Args:
            checked: Whether autoscale is enabled.
        """
        if checked:
            self.parent.actionAutoscale.setIcon(
                QIcon(":/Images/Images/new_icons/icons8-unlock-96.png")
            )
        else:
            self.parent.actionAutoscale.setIcon(
                QIcon(":/Images/Images/new_icons/icons8-padlock-96.png")
            )
