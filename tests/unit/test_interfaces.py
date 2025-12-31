"""Unit tests for protocol interfaces.

Verifies that mock implementations conform to protocol interfaces
and that runtime_checkable works correctly.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from jax import Array

from src.RepTate.core.interfaces import (
    IApplication,
    IDataset,
    IFitResult,
    ITheory,
)
from tests.conftest import (
    MockApplication,
    MockDataset,
    MockFitResult,
    MockTheory,
)


class TestITheoryProtocol:
    """Tests for ITheory protocol compliance."""

    def test_mock_theory_is_itheory(self, mock_theory: MockTheory) -> None:
        """Verify MockTheory implements ITheory protocol."""
        assert isinstance(mock_theory, ITheory)

    def test_theory_has_name(self, mock_theory: MockTheory) -> None:
        """Verify theory has name property."""
        assert mock_theory.name == "MockTheory"

    def test_theory_has_description(self, mock_theory: MockTheory) -> None:
        """Verify theory has description property."""
        assert mock_theory.description == "Mock theory for testing"

    def test_theory_calculate(self, mock_theory: MockTheory) -> None:
        """Verify theory calculate method works."""
        x = jnp.array([1.0, 2.0, 3.0])
        params = {"slope": 2.0, "intercept": 1.0}
        result = mock_theory.calculate(params, x)
        expected = jnp.array([3.0, 5.0, 7.0])
        assert jnp.allclose(result, expected)

    def test_theory_get_parameters(self, mock_theory: MockTheory) -> None:
        """Verify theory get_parameters returns parameter dict."""
        params = mock_theory.get_parameters()
        assert "slope" in params
        assert "intercept" in params

    def test_theory_set_parameter(self, mock_theory: MockTheory) -> None:
        """Verify theory set_parameter modifies value."""
        mock_theory.set_parameter("slope", 5.0)
        assert mock_theory.get_parameters()["slope"].value == 5.0

    def test_theory_set_parameter_invalid_raises(
        self, mock_theory: MockTheory
    ) -> None:
        """Verify set_parameter raises KeyError for unknown parameter."""
        with pytest.raises(KeyError):
            mock_theory.set_parameter("unknown_param", 1.0)

    def test_theory_get_fit_parameters(self, mock_theory: MockTheory) -> None:
        """Verify get_fit_parameters returns list of names."""
        fit_params = mock_theory.get_fit_parameters()
        assert isinstance(fit_params, list)
        assert "slope" in fit_params
        assert "intercept" in fit_params


class TestIDatasetProtocol:
    """Tests for IDataset protocol compliance."""

    def test_mock_dataset_is_idataset(self, mock_dataset: MockDataset) -> None:
        """Verify MockDataset implements IDataset protocol."""
        assert isinstance(mock_dataset, IDataset)

    def test_dataset_has_name(self, mock_dataset: MockDataset) -> None:
        """Verify dataset has name property."""
        assert mock_dataset.name == "MockDataset"

    def test_dataset_has_columns(self, mock_dataset: MockDataset) -> None:
        """Verify dataset has columns property."""
        assert mock_dataset.columns == ["x", "y"]

    def test_dataset_get_x(self, mock_dataset: MockDataset) -> None:
        """Verify get_x returns JAX array."""
        x = mock_dataset.get_x()
        assert isinstance(x, Array)
        assert x.shape == (100,)

    def test_dataset_get_y(self, mock_dataset: MockDataset) -> None:
        """Verify get_y returns JAX array."""
        y = mock_dataset.get_y()
        assert isinstance(y, Array)
        assert y.shape == (100,)

    def test_dataset_get_column(self, mock_dataset: MockDataset) -> None:
        """Verify get_column retrieves named columns."""
        x = mock_dataset.get_column("x")
        y = mock_dataset.get_column("y")
        assert jnp.allclose(x, mock_dataset.get_x())
        assert jnp.allclose(y, mock_dataset.get_y())

    def test_dataset_get_column_invalid_raises(
        self, mock_dataset: MockDataset
    ) -> None:
        """Verify get_column raises KeyError for unknown column."""
        with pytest.raises(KeyError):
            mock_dataset.get_column("unknown_column")

    def test_dataset_get_error_returns_none(
        self, mock_dataset: MockDataset
    ) -> None:
        """Verify get_error returns None when no error data."""
        assert mock_dataset.get_error() is None


class TestIApplicationProtocol:
    """Tests for IApplication protocol compliance."""

    def test_mock_application_is_iapplication(
        self, mock_application: MockApplication
    ) -> None:
        """Verify MockApplication implements IApplication protocol."""
        assert isinstance(mock_application, IApplication)

    def test_application_has_name(
        self, mock_application: MockApplication
    ) -> None:
        """Verify application has name property."""
        assert mock_application.name == "MockApp"

    def test_application_has_extension(
        self, mock_application: MockApplication
    ) -> None:
        """Verify application has extension property."""
        assert mock_application.extension == "mock"

    def test_application_get_theories(
        self, mock_application: MockApplication
    ) -> None:
        """Verify get_theories returns theory classes."""
        theories = mock_application.get_theories()
        assert MockTheory in theories

    def test_application_get_datasets_initially_empty(
        self, mock_application: MockApplication
    ) -> None:
        """Verify get_datasets returns empty list initially."""
        assert mock_application.get_datasets() == []

    def test_application_add_dataset(
        self, mock_application: MockApplication
    ) -> None:
        """Verify add_dataset adds dataset to application."""
        ds = MockDataset(name="test_ds")
        mock_application.add_dataset(ds)
        assert len(mock_application.get_datasets()) == 1
        assert mock_application.get_datasets()[0].name == "test_ds"

    def test_application_remove_dataset(
        self, mock_application: MockApplication
    ) -> None:
        """Verify remove_dataset removes dataset by name."""
        ds = MockDataset(name="to_remove")
        mock_application.add_dataset(ds)
        mock_application.remove_dataset("to_remove")
        assert len(mock_application.get_datasets()) == 0

    def test_application_remove_dataset_invalid_raises(
        self, mock_application: MockApplication
    ) -> None:
        """Verify remove_dataset raises KeyError for unknown dataset."""
        with pytest.raises(KeyError):
            mock_application.remove_dataset("nonexistent")


class TestIFitResultProtocol:
    """Tests for IFitResult protocol compliance."""

    def test_mock_fit_result_is_ifitresult(
        self, mock_fit_result: MockFitResult
    ) -> None:
        """Verify MockFitResult implements IFitResult protocol."""
        assert isinstance(mock_fit_result, IFitResult)

    def test_fit_result_has_parameters(
        self, mock_fit_result: MockFitResult
    ) -> None:
        """Verify fit result has parameters property."""
        params = mock_fit_result.parameters
        assert "slope" in params
        assert "intercept" in params

    def test_fit_result_has_covariance(
        self, mock_fit_result: MockFitResult
    ) -> None:
        """Verify fit result has covariance matrix."""
        cov = mock_fit_result.covariance
        assert cov is not None
        assert cov.shape == (2, 2)

    def test_fit_result_has_residuals(
        self, mock_fit_result: MockFitResult
    ) -> None:
        """Verify fit result has residuals array."""
        residuals = mock_fit_result.residuals
        assert isinstance(residuals, Array)
        assert residuals.shape == (10,)

    def test_fit_result_has_success(
        self, mock_fit_result: MockFitResult
    ) -> None:
        """Verify fit result has success flag."""
        assert mock_fit_result.success is True

    def test_fit_result_get_uncertainty(
        self, mock_fit_result: MockFitResult
    ) -> None:
        """Verify get_uncertainty returns standard deviation."""
        unc = mock_fit_result.get_uncertainty("slope")
        assert unc > 0

    def test_fit_result_get_uncertainty_invalid_raises(
        self, mock_fit_result: MockFitResult
    ) -> None:
        """Verify get_uncertainty raises KeyError for unknown parameter."""
        with pytest.raises(KeyError):
            mock_fit_result.get_uncertainty("unknown")


class TestSyntheticDataFixtures:
    """Tests for synthetic data fixtures."""

    def test_synthetic_frequency_data(
        self, synthetic_frequency_data
    ) -> None:
        """Verify synthetic frequency data fixture provides valid data."""
        assert synthetic_frequency_data.x.shape == (50,)
        assert synthetic_frequency_data.y.shape == (50,)
        assert synthetic_frequency_data.noise_level == 0.0
        assert "G0" in synthetic_frequency_data.params
        assert "tau" in synthetic_frequency_data.params

    def test_synthetic_frequency_data_noisy(
        self, synthetic_frequency_data_noisy
    ) -> None:
        """Verify noisy synthetic data has expected noise level."""
        assert synthetic_frequency_data_noisy.noise_level == 0.05

    def test_synthetic_time_data(self, synthetic_time_data) -> None:
        """Verify synthetic time data fixture provides valid data."""
        assert synthetic_time_data.x.shape == (50,)
        assert synthetic_time_data.y.shape == (50,)
        assert "G0" in synthetic_time_data.params
        assert "tau" in synthetic_time_data.params

    def test_synthetic_multimode_data(self, synthetic_multimode_data) -> None:
        """Verify multi-mode data has more points and parameters."""
        assert synthetic_multimode_data.x.shape == (100,)
        assert synthetic_multimode_data.y.shape == (100,)
        assert len(synthetic_multimode_data.params) == 6  # 3 modes * 2 params


class TestTempWorkspaceFixtures:
    """Tests for temporary workspace fixtures."""

    def test_temp_workspace_exists(self, temp_workspace) -> None:
        """Verify temp_workspace creates a directory."""
        assert temp_workspace.exists()
        assert temp_workspace.is_dir()

    def test_temp_data_file_exists(self, temp_data_file) -> None:
        """Verify temp_data_file creates a file with content."""
        assert temp_data_file.exists()
        content = temp_data_file.read_text()
        assert "Sample TTS data file" in content


class TestNumericalTolerances:
    """Tests for numerical tolerance fixtures."""

    def test_numerical_tolerance_value(self, numerical_tolerance) -> None:
        """Verify numerical tolerance is 1e-10."""
        assert numerical_tolerance == 1e-10

    def test_relaxed_tolerance_value(self, relaxed_tolerance) -> None:
        """Verify relaxed tolerance is 1e-6."""
        assert relaxed_tolerance == 1e-6
