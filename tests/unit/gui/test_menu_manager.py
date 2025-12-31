"""Unit tests for MenuManager.

Tests cover:
- T059: Unit tests for MenuManager component

These tests validate the MenuManager component extracted from QApplicationWindow.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    pass


class TestMenuManagerInit:
    """Test MenuManager initialization."""

    def test_init_with_parent(self) -> None:
        """Test MenuManager initializes with parent."""
        from RepTate.gui.MenuManager import MenuManager

        mock_parent = MagicMock()
        manager = MenuManager(mock_parent)

        assert manager.parent is mock_parent
        assert isinstance(manager.action_handlers, dict)

    def test_init_empty_handlers(self) -> None:
        """Test action_handlers starts empty."""
        from RepTate.gui.MenuManager import MenuManager

        mock_parent = MagicMock()
        manager = MenuManager(mock_parent)

        assert len(manager.action_handlers) == 0


class TestDataInspectorToolbar:
    """Test data inspector toolbar setup."""

    def test_setup_data_inspector_toolbar(self) -> None:
        """Test toolbar is configured correctly."""
        from RepTate.gui.MenuManager import MenuManager

        mock_parent = MagicMock()
        manager = MenuManager(mock_parent)
        mock_toolbar = MagicMock()

        manager.setup_data_inspector_toolbar(mock_toolbar)

        # Verify toolbar actions were added
        assert mock_toolbar.addAction.called
        assert mock_toolbar.addSeparator.called
        mock_toolbar.setIconSize.assert_called_once()


class TestToolToolbar:
    """Test tool toolbar setup."""

    def test_setup_tool_toolbar(self) -> None:
        """Test tool toolbar is configured."""
        from RepTate.gui.MenuManager import MenuManager

        mock_parent = MagicMock()
        manager = MenuManager(mock_parent)
        mock_toolbar = MagicMock()

        manager.setup_tool_toolbar(mock_toolbar)

        mock_toolbar.setIconSize.assert_called_once()
        mock_toolbar.addAction.assert_called()


class TestDatasetActionsDisabled:
    """Test dataset action enable/disable."""

    def test_disable_dataset_actions(self) -> None:
        """Test disabling dataset actions."""
        from RepTate.gui.MenuManager import MenuManager

        mock_parent = MagicMock()
        mock_parent.datasets = {}
        manager = MenuManager(mock_parent)

        manager.set_dataset_actions_disabled(True)

        mock_parent.actionMarkerSettings.setDisabled.assert_called_with(True)
        mock_parent.actionSave_View.setDisabled.assert_called_with(True)
        mock_parent.actionReload_Data.setDisabled.assert_called_with(True)

    def test_enable_dataset_actions(self) -> None:
        """Test enabling dataset actions."""
        from RepTate.gui.MenuManager import MenuManager

        mock_parent = MagicMock()
        mock_parent.datasets = {}
        manager = MenuManager(mock_parent)

        manager.set_dataset_actions_disabled(False)

        mock_parent.actionMarkerSettings.setDisabled.assert_called_with(False)
        mock_parent.actionSave_View.setDisabled.assert_called_with(False)
        mock_parent.actionReload_Data.setDisabled.assert_called_with(False)

    def test_disable_includes_theory_actions(self) -> None:
        """Test disabling affects theory actions on datasets."""
        from RepTate.gui.MenuManager import MenuManager

        mock_parent = MagicMock()
        mock_ds = MagicMock()
        mock_parent.datasets = {"Set1": mock_ds}
        manager = MenuManager(mock_parent)

        manager.set_dataset_actions_disabled(True)

        mock_ds.actionNew_Theory.setDisabled.assert_called_with(True)


class TestAutoscaleIcon:
    """Test autoscale icon updates."""

    def test_update_autoscale_icon_checked(self) -> None:
        """Test autoscale icon when checked."""
        from RepTate.gui.MenuManager import MenuManager

        mock_parent = MagicMock()
        manager = MenuManager(mock_parent)

        with patch("RepTate.gui.MenuManager.QIcon"):
            manager.update_autoscale_icon(True)

        mock_parent.actionAutoscale.setIcon.assert_called_once()

    def test_update_autoscale_icon_unchecked(self) -> None:
        """Test autoscale icon when unchecked."""
        from RepTate.gui.MenuManager import MenuManager

        mock_parent = MagicMock()
        manager = MenuManager(mock_parent)

        with patch("RepTate.gui.MenuManager.QIcon"):
            manager.update_autoscale_icon(False)

        mock_parent.actionAutoscale.setIcon.assert_called_once()

    def test_autoscale_icons_are_different(self) -> None:
        """Test checked and unchecked use different icons."""
        from RepTate.gui.MenuManager import MenuManager

        mock_parent = MagicMock()
        manager = MenuManager(mock_parent)

        with patch("RepTate.gui.MenuManager.QIcon") as mock_icon:
            manager.update_autoscale_icon(True)
            call_args_checked = mock_icon.call_args

            mock_icon.reset_mock()
            manager.update_autoscale_icon(False)
            call_args_unchecked = mock_icon.call_args

        # Icons should be created with different paths
        assert call_args_checked is not None
        assert call_args_unchecked is not None
        # The paths should be different (unlock vs padlock)
        assert call_args_checked != call_args_unchecked


class TestConnectActions:
    """Test action connection setup."""

    def test_connect_actions_sets_up_signals(self) -> None:
        """Test connect_actions connects signals to slots."""
        from RepTate.gui.MenuManager import MenuManager

        mock_parent = MagicMock()
        manager = MenuManager(mock_parent)

        manager.connect_actions()

        # Verify key connections were made
        mock_parent.actionShowFigureTools.triggered.connect.assert_called()
        mock_parent.actionInspect_Data.triggered.connect.assert_called()
        mock_parent.actionNew_Empty_Dataset.triggered.connect.assert_called()
        mock_parent.actionNew_Dataset_From_File.triggered.connect.assert_called()

    def test_connect_actions_links_to_handlers(self) -> None:
        """Test connections link to correct handlers."""
        from RepTate.gui.MenuManager import MenuManager

        mock_parent = MagicMock()
        manager = MenuManager(mock_parent)

        manager.connect_actions()

        # Verify specific handler connections
        mock_parent.actionCopy.triggered.connect.assert_called_with(
            mock_parent.inspector_table.copy
        )
        mock_parent.actionPaste.triggered.connect.assert_called_with(
            mock_parent.inspector_table.paste
        )


class TestSetupDatasetToolbar:
    """Test dataset toolbar setup."""

    def test_setup_dataset_toolbar_creates_menus(self) -> None:
        """Test dataset toolbar creates popup menus."""
        from RepTate.gui.MenuManager import MenuManager

        mock_parent = MagicMock()
        manager = MenuManager(mock_parent)
        mock_toolbar = MagicMock()

        with patch("RepTate.gui.MenuManager.QToolButton") as mock_button_cls:
            with patch("RepTate.gui.MenuManager.QMenu") as mock_menu_cls:
                manager.setup_dataset_toolbar(mock_toolbar)

                # Verify menus were created
                assert mock_menu_cls.call_count >= 2
                # Verify tool buttons were created
                assert mock_button_cls.call_count >= 2

    def test_setup_dataset_toolbar_adds_actions(self) -> None:
        """Test dataset toolbar adds required actions."""
        from RepTate.gui.MenuManager import MenuManager

        mock_parent = MagicMock()
        manager = MenuManager(mock_parent)
        mock_toolbar = MagicMock()

        with patch("RepTate.gui.MenuManager.QToolButton"):
            with patch("RepTate.gui.MenuManager.QMenu"):
                manager.setup_dataset_toolbar(mock_toolbar)

                # Verify actions were added
                mock_toolbar.addAction.assert_called()
