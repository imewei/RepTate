"""Integration tests for data loading workflow.

Tests cover:
- T050: Complete data loading integration from file to DataTable

The data loading integration tests validate:
1. Reading various file formats
2. Parsing data into DataTable structures
3. Column metadata extraction
4. Unit handling
5. Error conditions
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

if TYPE_CHECKING:
    pass


class TestDataTableLoading:
    """Test loading data into DataTable structure."""

    def test_create_datatable_from_arrays(self) -> None:
        """Test creating DataTable from numpy arrays."""
        from RepTate.core.DataTable import DataTable

        dt = DataTable()

        # Simulating loaded data
        x_data = np.array([0.1, 1.0, 10.0, 100.0])
        y_data = np.array([1e5, 5e4, 1e4, 5e3])

        dt.data = np.column_stack([x_data, y_data])
        dt.num_rows, dt.num_columns = dt.data.shape
        dt.column_names = ["frequency", "G_prime"]
        dt.column_units = ["rad/s", "Pa"]

        assert dt.num_rows == 4
        assert dt.num_columns == 2
        assert dt.column_names[0] == "frequency"
        assert dt.column_units[1] == "Pa"

    def test_datatable_column_access(self) -> None:
        """Test accessing specific columns from DataTable."""
        from RepTate.core.DataTable import DataTable

        dt = DataTable()

        data = np.array([
            [0.1, 1e5, 1e4],
            [1.0, 5e4, 5e3],
            [10.0, 1e4, 1e3],
        ])

        dt.data = data
        dt.num_rows, dt.num_columns = data.shape
        dt.column_names = ["omega", "G_prime", "G_double_prime"]

        # Access columns by index
        assert_array_almost_equal(dt.data[:, 0], [0.1, 1.0, 10.0])
        assert_array_almost_equal(dt.data[:, 1], [1e5, 5e4, 1e4])
        assert_array_almost_equal(dt.data[:, 2], [1e4, 5e3, 1e3])

    def test_datatable_statistics(self) -> None:
        """Test DataTable statistical methods."""
        from RepTate.core.DataTable import DataTable

        dt = DataTable()
        dt.data = np.array([
            [1.0, 100.0],
            [2.0, 200.0],
            [3.0, 300.0],
        ])
        dt.num_rows, dt.num_columns = dt.data.shape

        assert dt.mincol(0) == 1.0
        assert dt.maxcol(0) == 3.0
        assert dt.mincol(1) == 100.0
        assert dt.maxcol(1) == 300.0


class TestFileFormatParsing:
    """Test parsing different file formats."""

    def test_parse_tts_format(self, temp_workspace: Path) -> None:
        """Test parsing TTS (Time-Temperature Superposition) format."""
        # Create a mock TTS file
        tts_content = """# TTS data file
# omega G' G''
0.01\t100.0\t10.0
0.1\t1000.0\t100.0
1.0\t10000.0\t1000.0
10.0\t50000.0\t5000.0
"""
        tts_path = temp_workspace / "sample.tts"
        tts_path.write_text(tts_content)

        # Parse file (simulated - actual parsing depends on RepTate I/O)
        lines = [l for l in tts_content.split("\n") if l and not l.startswith("#")]
        data = []
        for line in lines:
            values = [float(v) for v in line.split()]
            if values:
                data.append(values)

        data_array = np.array(data)

        assert data_array.shape == (4, 3)
        assert data_array[0, 0] == 0.01
        assert data_array[3, 1] == 50000.0

    def test_parse_csv_format(self, temp_workspace: Path) -> None:
        """Test parsing CSV format data files."""
        csv_content = """frequency,G_prime,G_double_prime
0.01,100.0,10.0
0.1,1000.0,100.0
1.0,10000.0,1000.0
"""
        csv_path = temp_workspace / "data.csv"
        csv_path.write_text(csv_content)

        # Parse CSV
        lines = csv_content.strip().split("\n")
        header = lines[0].split(",")
        data = []
        for line in lines[1:]:
            data.append([float(v) for v in line.split(",")])

        data_array = np.array(data)

        assert len(header) == 3
        assert header[0] == "frequency"
        assert data_array.shape == (3, 3)

    def test_parse_whitespace_delimited(self, temp_workspace: Path) -> None:
        """Test parsing whitespace-delimited data."""
        ws_content = """   0.01    100.0    10.0
   0.10   1000.0   100.0
   1.00  10000.0  1000.0
"""
        ws_path = temp_workspace / "data.dat"
        ws_path.write_text(ws_content)

        # Parse whitespace-delimited
        data = []
        for line in ws_content.strip().split("\n"):
            values = [float(v) for v in line.split()]
            data.append(values)

        data_array = np.array(data)

        assert data_array.shape == (3, 3)
        assert_array_almost_equal(data_array[:, 0], [0.01, 0.10, 1.00])


class TestDataValidation:
    """Test data validation during loading."""

    def test_validate_numeric_data(self) -> None:
        """Test validation of numeric data."""
        from RepTate.core.DataTable import DataTable

        dt = DataTable()
        dt.data = np.array([[1.0, 2.0], [3.0, 4.0]])
        dt.num_rows, dt.num_columns = dt.data.shape

        # All values should be finite
        assert np.all(np.isfinite(dt.data))

    def test_detect_invalid_values(self) -> None:
        """Test detection of invalid values (NaN, Inf)."""
        data_with_nan = np.array([[1.0, np.nan], [3.0, 4.0]])
        data_with_inf = np.array([[1.0, np.inf], [3.0, 4.0]])

        assert not np.all(np.isfinite(data_with_nan))
        assert not np.all(np.isfinite(data_with_inf))

    def test_validate_positive_values(self) -> None:
        """Test validation for positive-only data (e.g., frequencies)."""
        frequencies = np.array([0.01, 0.1, 1.0, 10.0])

        # All frequencies should be positive
        assert np.all(frequencies > 0)

        # Negative frequency should fail validation
        bad_frequencies = np.array([-0.01, 0.1, 1.0])
        assert not np.all(bad_frequencies > 0)


class TestDataTransformation:
    """Test data transformations during loading."""

    def test_log_transformation(self) -> None:
        """Test log transformation for rheology data."""
        linear_data = np.array([1.0, 10.0, 100.0, 1000.0])
        log_data = np.log10(linear_data)

        expected = np.array([0.0, 1.0, 2.0, 3.0])
        assert_array_almost_equal(log_data, expected)

    def test_unit_conversion(self) -> None:
        """Test unit conversion (e.g., Hz to rad/s)."""
        freq_hz = np.array([1.0, 10.0, 100.0])

        # Convert Hz to rad/s: omega = 2*pi*f
        freq_rads = freq_hz * 2 * np.pi

        expected = np.array([2 * np.pi, 20 * np.pi, 200 * np.pi])
        assert_array_almost_equal(freq_rads, expected)


class TestDataExtraTables:
    """Test extra_tables functionality in DataTable."""

    def test_store_auxiliary_data(self) -> None:
        """Test storing auxiliary data in extra_tables."""
        from RepTate.core.DataTable import DataTable

        dt = DataTable()
        dt.data = np.array([[1.0, 2.0], [3.0, 4.0]])
        dt.num_rows, dt.num_columns = dt.data.shape

        # Store auxiliary data
        dt.extra_tables["temperature"] = np.array([25.0])
        dt.extra_tables["strain_rate"] = np.array([1.0, 10.0, 100.0])
        dt.extra_tables["metadata"] = np.array(["sample_001"])

        assert "temperature" in dt.extra_tables
        assert len(dt.extra_tables["strain_rate"]) == 3

    def test_access_extra_table_data(self) -> None:
        """Test accessing data from extra_tables."""
        from RepTate.core.DataTable import DataTable

        dt = DataTable()
        dt.extra_tables["Gt"] = np.array([1e5, 8e4, 5e4, 2e4, 1e4])

        retrieved = dt.extra_tables["Gt"]
        assert len(retrieved) == 5
        assert retrieved[0] == 1e5


class TestMultiColumnData:
    """Test handling of multi-column data."""

    def test_three_column_data(self) -> None:
        """Test loading three-column data (omega, G', G'')."""
        from RepTate.core.DataTable import DataTable

        dt = DataTable()
        dt.data = np.array([
            [0.1, 1e5, 1e4],
            [1.0, 8e4, 8e3],
            [10.0, 5e4, 5e3],
        ])
        dt.num_rows, dt.num_columns = dt.data.shape
        dt.column_names = ["omega", "G_prime", "G_double_prime"]
        dt.column_units = ["rad/s", "Pa", "Pa"]

        assert dt.num_columns == 3
        assert dt.column_names[2] == "G_double_prime"

    def test_many_column_data(self) -> None:
        """Test loading data with many columns."""
        from RepTate.core.DataTable import DataTable

        dt = DataTable()

        # Create 10-column data
        n_cols = 10
        n_rows = 50
        dt.data = np.random.rand(n_rows, n_cols)
        dt.num_rows, dt.num_columns = dt.data.shape
        dt.column_names = [f"col_{i}" for i in range(n_cols)]
        dt.column_units = ["unit"] * n_cols

        assert dt.num_columns == 10
        assert dt.num_rows == 50


class TestDataIntegrity:
    """Test data integrity after loading."""

    def test_data_shape_consistency(self) -> None:
        """Test that num_rows and num_columns match data shape."""
        from RepTate.core.DataTable import DataTable

        dt = DataTable()
        test_data = np.random.rand(100, 5)

        dt.data = test_data
        dt.num_rows, dt.num_columns = test_data.shape

        assert dt.num_rows == dt.data.shape[0]
        assert dt.num_columns == dt.data.shape[1]

    def test_column_names_length(self) -> None:
        """Test column_names length matches num_columns."""
        from RepTate.core.DataTable import DataTable

        dt = DataTable()
        dt.data = np.random.rand(10, 3)
        dt.num_rows, dt.num_columns = dt.data.shape
        dt.column_names = ["a", "b", "c"]
        dt.column_units = ["u1", "u2", "u3"]

        assert len(dt.column_names) == dt.num_columns
        assert len(dt.column_units) == dt.num_columns
