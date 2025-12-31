"""Unit tests for DatasetManager.

Tests cover:
- T060: Unit tests for DatasetManager component

These tests validate the DatasetManager component extracted from QApplicationWindow.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    pass


class TestDatasetManagerInit:
    """Test DatasetManager initialization."""

    def test_init_with_parent(self) -> None:
        """Test DatasetManager initializes with parent."""
        from RepTate.gui.DatasetManager import DatasetManager

        mock_parent = MagicMock()
        mock_parent.logger.name = "TestApp"
        manager = DatasetManager(mock_parent)

        assert manager.parent is mock_parent
        assert manager.logger is not None

    def test_logger_is_child_of_parent_logger(self) -> None:
        """Test logger is created as child of parent logger."""
        from RepTate.gui.DatasetManager import DatasetManager

        mock_parent = MagicMock()
        mock_parent.logger.name = "TestApp"
        manager = DatasetManager(mock_parent)

        assert "DatasetManager" in manager.logger.name


class TestGetCurrentDataset:
    """Test getting current dataset."""

    def test_get_current_dataset_returns_widget(self) -> None:
        """Test returns current tab widget."""
        from RepTate.gui.DatasetManager import DatasetManager

        mock_parent = MagicMock()
        mock_parent.logger.name = "TestApp"
        mock_ds = MagicMock()
        mock_parent.DataSettabWidget.currentWidget.return_value = mock_ds
        manager = DatasetManager(mock_parent)

        result = manager.get_current_dataset()

        assert result is mock_ds

    def test_get_current_dataset_returns_none_when_empty(self) -> None:
        """Test returns None when no datasets."""
        from RepTate.gui.DatasetManager import DatasetManager

        mock_parent = MagicMock()
        mock_parent.logger.name = "TestApp"
        mock_parent.DataSettabWidget.currentWidget.return_value = None
        manager = DatasetManager(mock_parent)

        result = manager.get_current_dataset()

        assert result is None


class TestEnsureDatasetExists:
    """Test ensuring dataset exists."""

    def test_ensure_dataset_exists_returns_existing(self) -> None:
        """Test returns existing dataset when present."""
        from RepTate.gui.DatasetManager import DatasetManager

        mock_parent = MagicMock()
        mock_parent.logger.name = "TestApp"
        mock_parent.DataSettabWidget.count.return_value = 1
        mock_ds = MagicMock()
        mock_parent.DataSettabWidget.currentWidget.return_value = mock_ds
        manager = DatasetManager(mock_parent)

        result = manager.ensure_dataset_exists()

        assert result is mock_ds

    def test_ensure_dataset_exists_creates_new_when_empty(self) -> None:
        """Test creates new dataset when none exist."""
        from RepTate.gui.DatasetManager import DatasetManager

        mock_parent = MagicMock()
        mock_parent.logger.name = "TestApp"
        mock_parent.DataSettabWidget.count.return_value = 0
        mock_parent.num_datasets = 0
        mock_parent.datasets = {}
        mock_parent.filetypes = {"ext": MagicMock()}
        mock_parent.filetypes["ext"].basic_file_parameters = ["param1"]
        mock_parent.filetypes["ext"].col_names = ["x"]
        mock_parent.filetypes["ext"].col_units = ["unit"]
        manager = DatasetManager(mock_parent)

        with patch("RepTate.gui.QDataSet.QDataSet") as mock_qds_cls:
            mock_ds = MagicMock()
            mock_qds_cls.return_value = mock_ds
            mock_parent.DataSettabWidget.widget.return_value = mock_ds
            mock_parent.DataSettabWidget.addTab.return_value = 0

            result = manager.create_empty_dataset()

            assert result is mock_ds


class TestCloseDataset:
    """Test closing datasets."""

    def test_close_dataset_removes_from_registry(self) -> None:
        """Test closing removes dataset from parent.datasets."""
        from RepTate.gui.DatasetManager import DatasetManager

        mock_parent = MagicMock()
        mock_parent.logger.name = "TestApp"
        mock_ds = MagicMock()
        mock_ds.name = "Set1"
        mock_parent.DataSettabWidget.widget.return_value = mock_ds
        mock_parent.datasets = {"Set1": mock_ds}
        mock_parent.DataSettabWidget.count.return_value = 0
        manager = DatasetManager(mock_parent)

        result = manager.close_dataset(0)

        assert result is True
        assert "Set1" not in mock_parent.datasets

    def test_close_dataset_returns_false_when_invalid_index(self) -> None:
        """Test returns False for invalid index."""
        from RepTate.gui.DatasetManager import DatasetManager

        mock_parent = MagicMock()
        mock_parent.logger.name = "TestApp"
        mock_parent.DataSettabWidget.widget.return_value = None
        manager = DatasetManager(mock_parent)

        result = manager.close_dataset(999)

        assert result is False

    def test_close_dataset_disables_actions_when_last(self) -> None:
        """Test disables actions when last dataset closed."""
        from RepTate.gui.DatasetManager import DatasetManager

        mock_parent = MagicMock()
        mock_parent.logger.name = "TestApp"
        mock_ds = MagicMock()
        mock_ds.name = "Set1"
        mock_parent.DataSettabWidget.widget.return_value = mock_ds
        mock_parent.datasets = {"Set1": mock_ds}
        mock_parent.DataSettabWidget.count.return_value = 0
        manager = DatasetManager(mock_parent)

        manager.close_dataset(0)

        mock_parent.dataset_actions_disabled.assert_called_with(True)


class TestAddTableToCurrentDataset:
    """Test adding tables to datasets."""

    def test_add_table_increments_file_count(self) -> None:
        """Test adding table increments num_files."""
        from RepTate.gui.DatasetManager import DatasetManager

        mock_parent = MagicMock()
        mock_parent.logger.name = "TestApp"
        mock_ds = MagicMock()
        mock_ds.num_files = 0
        mock_ds.files = []
        mock_parent.DataSettabWidget.currentWidget.return_value = mock_ds
        mock_parent.filetypes = {"ext": MagicMock()}
        mock_parent.filetypes["ext"].basic_file_parameters = []
        manager = DatasetManager(mock_parent)

        mock_dt = MagicMock()
        mock_dt.file_name_short = "test.dat"
        mock_dt.file_parameters = {}

        with patch("RepTate.gui.DatasetManager.DataSetWidgetItem"):
            manager.add_table_to_current_dataset(mock_dt, "ext")

        assert mock_ds.num_files == 1
        assert mock_dt in mock_ds.files

    def test_add_table_handles_missing_parameters(self) -> None:
        """Test handles missing file parameters gracefully."""
        from RepTate.gui.DatasetManager import DatasetManager

        mock_parent = MagicMock()
        mock_parent.logger.name = "TestApp"
        mock_ds = MagicMock()
        mock_ds.num_files = 0
        mock_ds.files = []
        mock_parent.DataSettabWidget.currentWidget.return_value = mock_ds
        mock_parent.filetypes = {"ext": MagicMock()}
        mock_parent.filetypes["ext"].basic_file_parameters = ["missing_param"]
        manager = DatasetManager(mock_parent)

        mock_dt = MagicMock()
        mock_dt.file_name_short = "test.dat"
        # Use a real dict to track the parameter being set
        mock_dt.file_parameters = {}

        with patch("RepTate.gui.DatasetManager.DataSetWidgetItem"):
            manager.add_table_to_current_dataset(mock_dt, "ext")

        # Parameter should be set to "0" when missing
        assert mock_dt.file_parameters["missing_param"] == "0"

    def test_add_table_enables_actions(self) -> None:
        """Test adding table enables dataset actions."""
        from RepTate.gui.DatasetManager import DatasetManager

        mock_parent = MagicMock()
        mock_parent.logger.name = "TestApp"
        mock_ds = MagicMock()
        mock_ds.num_files = 0
        mock_ds.files = []
        mock_parent.DataSettabWidget.currentWidget.return_value = mock_ds
        mock_parent.filetypes = {"ext": MagicMock()}
        mock_parent.filetypes["ext"].basic_file_parameters = []
        manager = DatasetManager(mock_parent)

        mock_dt = MagicMock()
        mock_dt.file_name_short = "test.dat"
        mock_dt.file_parameters = {}

        with patch("RepTate.gui.DatasetManager.DataSetWidgetItem"):
            manager.add_table_to_current_dataset(mock_dt, "ext")

        mock_parent.dataset_actions_disabled.assert_called_with(False)


class TestCheckNoParamMissing:
    """Test parameter checking."""

    def test_check_logs_warning_for_missing_params(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test logs warning when parameters missing."""
        import logging

        from RepTate.gui.DatasetManager import DatasetManager

        mock_parent = MagicMock()
        mock_parent.logger.name = "TestApp"
        mock_parent.filetypes = {"ext": MagicMock()}
        mock_parent.filetypes["ext"].basic_file_parameters = ["param1", "param2"]
        manager = DatasetManager(mock_parent)

        mock_dt = MagicMock()
        mock_dt.file_name_short = "test.dat"
        mock_dt.file_parameters = {}  # All params missing

        with caplog.at_level(logging.WARNING):
            manager.check_no_param_missing([mock_dt], "ext")

        # Check warning was logged
        assert "param1" in caplog.text or "param2" in caplog.text

    def test_check_sets_missing_to_zero(self) -> None:
        """Test missing parameters set to '0'."""
        from RepTate.gui.DatasetManager import DatasetManager

        mock_parent = MagicMock()
        mock_parent.logger.name = "TestApp"
        mock_parent.filetypes = {"ext": MagicMock()}
        mock_parent.filetypes["ext"].basic_file_parameters = ["param1"]
        manager = DatasetManager(mock_parent)

        mock_dt = MagicMock()
        mock_dt.file_name_short = "test.dat"
        mock_dt.file_parameters = {}

        manager.check_no_param_missing([mock_dt], "ext")

        assert mock_dt.file_parameters["param1"] == "0"


class TestUpdateAllDatasetsPlots:
    """Test updating all dataset plots."""

    def test_update_calls_do_plot_on_all_datasets(self) -> None:
        """Test calls do_plot on each dataset."""
        from RepTate.gui.DatasetManager import DatasetManager

        mock_parent = MagicMock()
        mock_parent.logger.name = "TestApp"
        mock_ds1 = MagicMock()
        mock_ds2 = MagicMock()
        mock_parent.datasets = {"Set1": mock_ds1, "Set2": mock_ds2}
        manager = DatasetManager(mock_parent)

        manager.update_all_datasets_plots()

        mock_ds1.do_plot.assert_called_once()
        mock_ds2.do_plot.assert_called_once()


class TestReloadCurrentDataset:
    """Test reloading current dataset."""

    def test_reload_calls_dataset_reload(self) -> None:
        """Test calls reload_data on current dataset."""
        from RepTate.gui.DatasetManager import DatasetManager

        mock_parent = MagicMock()
        mock_parent.logger.name = "TestApp"
        mock_ds = MagicMock()
        mock_parent.DataSettabWidget.currentWidget.return_value = mock_ds
        manager = DatasetManager(mock_parent)

        manager.reload_current_dataset()

        mock_ds.reload_data.assert_called_once()

    def test_reload_handles_no_current_dataset(self) -> None:
        """Test handles case with no current dataset."""
        from RepTate.gui.DatasetManager import DatasetManager

        mock_parent = MagicMock()
        mock_parent.logger.name = "TestApp"
        mock_parent.DataSettabWidget.currentWidget.return_value = None
        manager = DatasetManager(mock_parent)

        # Should not raise
        manager.reload_current_dataset()
