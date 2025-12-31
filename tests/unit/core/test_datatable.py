"""Unit tests for DataTable class.

Tests cover:
- T043: DataTable initialization, data storage, and column operations

The DataTable class stores experimental and theory data as numpy arrays
with associated column metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

if TYPE_CHECKING:
    pass


class TestDataTableInitialization:
    """Test DataTable construction and default values."""

    def test_default_initialization(self) -> None:
        """Test DataTable initializes with empty defaults."""
        from RepTate.core.DataTable import DataTable

        dt = DataTable()

        assert dt.num_columns == 0
        assert dt.num_rows == 0
        assert dt.column_names == []
        assert dt.column_units == []
        assert dt.data.shape == (0, 0)
        assert dt.series == []
        assert dt.extra_tables == {}

    def test_initialization_with_name(self) -> None:
        """Test DataTable accepts name parameter without axes."""
        from RepTate.core.DataTable import DataTable

        dt = DataTable(_name="test_data")

        # Without axes, name is stored but no series created
        assert dt.series == []

    def test_class_constants(self) -> None:
        """Test class constants are defined correctly."""
        from RepTate.core.DataTable import DataTable

        assert DataTable.MAX_NUM_SERIES == 3
        assert DataTable.PICKRADIUS == 10


class TestDataTableDataStorage:
    """Test data storage and manipulation."""

    def test_set_data_array(self) -> None:
        """Test setting data array directly."""
        from RepTate.core.DataTable import DataTable

        dt = DataTable()
        test_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        dt.data = test_data
        dt.num_rows, dt.num_columns = test_data.shape

        assert dt.num_rows == 3
        assert dt.num_columns == 2
        assert_array_equal(dt.data, test_data)

    def test_set_column_metadata(self) -> None:
        """Test setting column names and units."""
        from RepTate.core.DataTable import DataTable

        dt = DataTable()
        dt.data = np.array([[1.0, 2.0], [3.0, 4.0]])
        dt.num_rows, dt.num_columns = dt.data.shape
        dt.column_names = ["frequency", "modulus"]
        dt.column_units = ["rad/s", "Pa"]

        assert dt.column_names == ["frequency", "modulus"]
        assert dt.column_units == ["rad/s", "Pa"]

    def test_extra_tables_storage(self) -> None:
        """Test extra_tables dictionary for additional data."""
        from RepTate.core.DataTable import DataTable

        dt = DataTable()
        extra_data = np.array([1.0, 2.0, 3.0])

        dt.extra_tables["aux_data"] = extra_data

        assert "aux_data" in dt.extra_tables
        assert_array_equal(dt.extra_tables["aux_data"], extra_data)


class TestDataTableColumnOperations:
    """Test column min/max operations."""

    @pytest.fixture
    def populated_datatable(self) -> "DataTable":
        """Create a DataTable with sample data."""
        from RepTate.core.DataTable import DataTable

        dt = DataTable()
        # 3 rows, 2 columns
        dt.data = np.array([
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ])
        dt.num_rows, dt.num_columns = dt.data.shape
        dt.column_names = ["x", "y"]
        return dt

    def test_mincol(self, populated_datatable: "DataTable") -> None:
        """Test minimum value in column."""
        dt = populated_datatable

        assert dt.mincol(0) == 1.0
        assert dt.mincol(1) == 10.0

    def test_maxcol(self, populated_datatable: "DataTable") -> None:
        """Test maximum value in column."""
        dt = populated_datatable

        assert dt.maxcol(0) == 3.0
        assert dt.maxcol(1) == 30.0

    def test_minpositivecol(self) -> None:
        """Test minimum positive value in column."""
        from RepTate.core.DataTable import DataTable

        dt = DataTable()
        # Data with negative and positive values
        dt.data = np.array([
            [-5.0, 10.0],
            [2.0, -20.0],
            [0.0, 5.0],
            [7.0, 15.0],
        ])
        dt.num_rows, dt.num_columns = dt.data.shape

        assert dt.minpositivecol(0) == 2.0
        assert dt.minpositivecol(1) == 5.0

    def test_minpositivecol_no_positives_raises(self) -> None:
        """Test minpositivecol raises when no positive values exist."""
        from RepTate.core.DataTable import DataTable

        dt = DataTable()
        dt.data = np.array([[-1.0], [-2.0], [-3.0]])
        dt.num_rows, dt.num_columns = dt.data.shape

        with pytest.raises(ValueError):
            dt.minpositivecol(0)


class TestDataTableStringRepresentation:
    """Test string representation methods."""

    def test_str_representation(self) -> None:
        """Test __str__ returns data string."""
        from RepTate.core.DataTable import DataTable

        dt = DataTable()
        dt.data = np.array([[1.0, 2.0]])

        result = str(dt)
        assert "1." in result
        assert "2." in result


class TestDataTableEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_data_operations(self) -> None:
        """Test operations on empty data."""
        from RepTate.core.DataTable import DataTable

        dt = DataTable()
        # Empty array operations should work without error
        assert dt.data.size == 0

    def test_single_value_column(self) -> None:
        """Test operations with single-value columns."""
        from RepTate.core.DataTable import DataTable

        dt = DataTable()
        dt.data = np.array([[42.0]])
        dt.num_rows, dt.num_columns = dt.data.shape

        assert dt.mincol(0) == 42.0
        assert dt.maxcol(0) == 42.0

    def test_large_data_array(self) -> None:
        """Test DataTable handles large arrays."""
        from RepTate.core.DataTable import DataTable

        dt = DataTable()
        large_data = np.random.rand(10000, 5)
        dt.data = large_data
        dt.num_rows, dt.num_columns = dt.data.shape

        assert dt.num_rows == 10000
        assert dt.num_columns == 5
        assert dt.mincol(0) >= 0.0
        assert dt.maxcol(0) <= 1.0

    def test_data_type_float(self) -> None:
        """Test data is stored as floating point."""
        from RepTate.core.DataTable import DataTable

        dt = DataTable()
        int_data = np.array([[1, 2], [3, 4]])

        dt.data = int_data.astype(np.float64)

        assert np.issubdtype(dt.data.dtype, np.floating)
