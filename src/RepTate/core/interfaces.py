"""Protocol interfaces for breaking circular dependencies between modules.

This module defines structural typing protocols that enable:
1. Applications to use theories without importing concrete classes
2. GUI to interact with applications/theories via abstractions
3. Testing with mock implementations
4. Clear module boundaries

All protocols are @runtime_checkable for debugging support.

Version: 1.0.0
Implements: FR-016, FR-017

Usage:
    from RepTate.core.interfaces import ITheory, IApplication, IDataset, IFitResult

    def fit_theory(theory: ITheory, x: np.ndarray, y: np.ndarray) -> dict:
        '''Fit any theory implementation.'''
        params = {p.name: p.value for p in theory.get_parameters().values()}
        prediction = theory.calculate(params, x)
        # ... fitting logic
        return optimized_params
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import jax.numpy as jnp
from jax import Array

if TYPE_CHECKING:
    from RepTate.core.Parameter import Parameter


@runtime_checkable
class ITheory(Protocol):
    """Interface for rheological theory implementations.

    Implemented by: TheoryBasic and all subclasses
    Used by: Applications, FitController, GUI

    This protocol enables theories to be used polymorphically without
    concrete class imports, breaking circular dependencies between
    applications and theory modules.

    Examples:
        >>> def process_theory(theory: ITheory) -> None:
        ...     if not isinstance(theory, ITheory):
        ...         raise TypeError(f"Expected ITheory, got {type(theory)}")
        ...     result = theory.calculate(params, x_data)
    """

    @property
    def name(self) -> str:
        """Theory display name.

        Returns:
            Human-readable name for UI display
        """
        ...

    @property
    def description(self) -> str:
        """Theory description for UI.

        Returns:
            Detailed description of the theory's purpose and usage
        """
        ...

    def calculate(
        self,
        params: dict[str, float],
        x: Array,
    ) -> Array:
        """Calculate theory predictions.

        This method performs the core computation of the theory,
        transforming input data according to the theory's model.

        Args:
            params: Parameter name to value mapping
            x: Input data array (e.g., frequency, time, strain rate)

        Returns:
            Predicted values array matching x shape

        Raises:
            ValueError: If parameters are invalid or out of bounds
        """
        ...

    def get_parameters(self) -> dict[str, Parameter]:
        """Get all parameter definitions.

        Returns:
            Dictionary mapping parameter names to Parameter objects
        """
        ...

    def set_parameter(self, name: str, value: float) -> None:
        """Set a parameter value.

        Args:
            name: Parameter name to set
            value: New value for the parameter

        Raises:
            KeyError: If parameter doesn't exist
            ValueError: If value is out of bounds
        """
        ...

    def get_fit_parameters(self) -> list[str]:
        """Get names of parameters marked for fitting.

        Returns:
            List of parameter names where opt_type is OptType.opt
        """
        ...


@runtime_checkable
class IApplication(Protocol):
    """Interface for RepTate application containers.

    Implemented by: All Application subclasses (ApplicationLVE, ApplicationLAOS, etc.)
    Used by: GUI layer, application manager

    This protocol enables the GUI to work with applications polymorphically
    without importing concrete application classes.
    """

    @property
    def name(self) -> str:
        """Application display name.

        Returns:
            Human-readable name for UI display (e.g., "LVE", "LAOS")
        """
        ...

    @property
    def extension(self) -> str:
        """Default file extension for this app type.

        Returns:
            File extension without leading dot (e.g., "tts", "osc")
        """
        ...

    def get_theories(self) -> list[type[ITheory]]:
        """Get available theory classes for this application.

        Returns:
            List of theory classes that can be used with this application
        """
        ...

    def load_data(self, filepath: str) -> IDataset:
        """Load data from file.

        Args:
            filepath: Path to data file

        Returns:
            Loaded dataset implementing IDataset

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        ...

    def get_datasets(self) -> list[IDataset]:
        """Get all loaded datasets.

        Returns:
            List of datasets currently loaded in this application
        """
        ...

    def add_dataset(self, dataset: IDataset) -> None:
        """Add a dataset to the application.

        Args:
            dataset: Dataset to add
        """
        ...

    def remove_dataset(self, name: str) -> None:
        """Remove a dataset by name.

        Args:
            name: Name of dataset to remove

        Raises:
            KeyError: If dataset with given name doesn't exist
        """
        ...


@runtime_checkable
class IDataset(Protocol):
    """Interface for experimental datasets.

    Implemented by: DataTable
    Used by: Theories for data access, GUI for display

    This protocol provides a consistent interface for accessing
    experimental data regardless of the underlying storage format.
    """

    @property
    def name(self) -> str:
        """Dataset identifier.

        Returns:
            Unique name for this dataset within an application
        """
        ...

    @property
    def columns(self) -> list[str]:
        """Available column names.

        Returns:
            List of column names in this dataset
        """
        ...

    def get_x(self) -> Array:
        """Get primary x-axis data.

        Returns:
            JAX array of x-axis values (typically first column)
        """
        ...

    def get_y(self) -> Array:
        """Get primary y-axis data.

        Returns:
            JAX array of y-axis values (typically second column)
        """
        ...

    def get_column(self, name: str) -> Array:
        """Get data column by name.

        Args:
            name: Column name to retrieve

        Returns:
            JAX array of column values

        Raises:
            KeyError: If column doesn't exist
        """
        ...

    def get_error(self) -> Array | None:
        """Get y-axis error bars if available.

        Returns:
            JAX array of error values, or None if not available
        """
        ...


@runtime_checkable
class IFitResult(Protocol):
    """Interface for curve fitting results.

    Implemented by: FitResult from NLSQ
    Used by: GUI, export functions

    This protocol provides a consistent interface for accessing
    fitting results regardless of the underlying optimizer.
    """

    @property
    def parameters(self) -> dict[str, float]:
        """Optimized parameter values.

        Returns:
            Dictionary mapping parameter names to their optimized values
        """
        ...

    @property
    def covariance(self) -> Array | None:
        """Parameter covariance matrix.

        Returns:
            Covariance matrix as JAX array, or None if not computed
        """
        ...

    @property
    def residuals(self) -> Array:
        """Fit residuals.

        Returns:
            JAX array of residual values (y_observed - y_predicted)
        """
        ...

    @property
    def success(self) -> bool:
        """Whether fit converged.

        Returns:
            True if the optimizer converged successfully
        """
        ...

    def get_uncertainty(self, name: str) -> float:
        """Get parameter uncertainty (standard deviation).

        Computes the standard deviation from the diagonal of
        the covariance matrix.

        Args:
            name: Parameter name

        Returns:
            Standard deviation of the parameter estimate

        Raises:
            KeyError: If parameter name not in results
            ValueError: If covariance matrix is not available
        """
        ...


# Type aliases for common patterns
TheoryType = type[ITheory]
DatasetList = list[IDataset]
ParameterDict = dict[str, float]
