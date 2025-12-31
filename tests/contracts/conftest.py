"""Contract testing fixtures for RepTate.

Contract tests validate integration agreements between components:
- API contracts: Input/output shapes, types, ranges
- Data contracts: Field names, types, null handling
- Performance contracts: Baseline measurements, regression thresholds
- Compatibility contracts: Migration equivalence guarantees

These fixtures provide shared infrastructure for all contract tests.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

if TYPE_CHECKING:
    from collections.abc import Generator


# =============================================================================
# Performance Baseline Structures
# =============================================================================


@dataclass(frozen=True)
class PerformanceBaseline:
    """Performance baseline for contract validation.

    Attributes:
        operation: Name of the operation measured
        mean_time: Mean execution time in seconds
        std_dev: Standard deviation of execution time
        samples: Number of samples taken
        threshold_percent: Acceptable regression percentage
    """

    operation: str
    mean_time: float
    std_dev: float
    samples: int
    threshold_percent: float = 10.0  # Default 10% regression tolerance

    def is_within_threshold(self, measured_time: float) -> bool:
        """Check if measured time is within acceptable threshold.

        Args:
            measured_time: Measured execution time in seconds

        Returns:
            True if within threshold, False otherwise
        """
        threshold = self.mean_time * (1 + self.threshold_percent / 100)
        return measured_time <= threshold

    def get_regression_percent(self, measured_time: float) -> float:
        """Calculate regression percentage.

        Args:
            measured_time: Measured execution time in seconds

        Returns:
            Percentage regression (negative if faster)
        """
        return ((measured_time - self.mean_time) / self.mean_time) * 100


@dataclass(frozen=True)
class ContractViolation:
    """Represents a contract violation.

    Attributes:
        component: Name of the component
        contract_type: Type of contract (API, Data, Performance, Compatibility)
        violation_desc: Description of the violation
        expected: Expected value or specification
        actual: Actual value or behavior
    """

    component: str
    contract_type: str
    violation_desc: str
    expected: Any
    actual: Any

    def __str__(self) -> str:
        return (
            f"[{self.contract_type}] {self.component}: {self.violation_desc}\n"
            f"  Expected: {self.expected}\n"
            f"  Actual: {self.actual}"
        )


# =============================================================================
# Baseline Storage
# =============================================================================


@pytest.fixture
def baseline_storage() -> Generator[Path, None, None]:
    """Provide temporary baseline storage directory.

    Yields:
        Path to a temporary directory for storing baselines.
    """
    with tempfile.TemporaryDirectory(prefix="reptate_baselines_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def baseline_registry() -> dict[str, PerformanceBaseline]:
    """Provide in-memory baseline registry.

    Returns:
        Dictionary mapping operation names to PerformanceBaseline objects.
    """
    return {}


# =============================================================================
# Data Contract Fixtures
# =============================================================================


@dataclass(frozen=True)
class TheoryParameterContract:
    """Contract for theory parameter structure.

    Attributes:
        name: Parameter name
        expected_type: Expected type (float, int, bool)
        has_bounds: Whether parameter has min/max bounds
        description: Parameter description
    """

    name: str
    expected_type: str
    has_bounds: bool
    description: str


@dataclass(frozen=True)
class TheoryCalculationContract:
    """Contract for theory calculation output.

    Attributes:
        input_shape_requirement: Expected input shape pattern
        output_shape_requirement: Expected output shape pattern
        dtype_requirement: Expected output dtype
        nan_allowed: Whether NaN values are allowed
        inf_allowed: Whether Inf values are allowed
    """

    input_shape_requirement: str  # e.g., "(n,)" or "(n, m)"
    output_shape_requirement: str  # Must match input
    dtype_requirement: str  # "float64" or "complex128"
    nan_allowed: bool = False
    inf_allowed: bool = False


@pytest.fixture
def theory_parameter_contracts() -> dict[str, list[TheoryParameterContract]]:
    """Provide theory parameter contracts.

    Returns:
        Dictionary mapping theory names to parameter contracts.
    """
    return {
        "Maxwell": [
            TheoryParameterContract(
                name="G0",
                expected_type="float",
                has_bounds=True,
                description="Plateau modulus",
            ),
            TheoryParameterContract(
                name="tau",
                expected_type="float",
                has_bounds=True,
                description="Relaxation time",
            ),
        ],
        "RoliePoly": [
            TheoryParameterContract(
                name="G0",
                expected_type="float",
                has_bounds=True,
                description="Plateau modulus",
            ),
            TheoryParameterContract(
                name="tau",
                expected_type="float",
                has_bounds=True,
                description="Relaxation time",
            ),
            TheoryParameterContract(
                name="nu",
                expected_type="float",
                has_bounds=True,
                description="Stretch parameter",
            ),
        ],
    }


@pytest.fixture
def dataset_data_contract() -> dict[str, Any]:
    """Provide dataset data structure contract.

    Returns:
        Dictionary defining expected dataset structure.
    """
    return {
        "required_fields": ["x", "y"],
        "optional_fields": ["error", "metadata"],
        "x_requirements": {
            "type": "array",
            "dtype": "float64",
            "shape": "(n,)",
            "strictly_increasing": True,
        },
        "y_requirements": {
            "type": "array",
            "dtype": "float64",
            "shape": "(n,)",
        },
        "error_requirements": {
            "type": "array",
            "dtype": "float64",
            "shape": "(n,)",
            "positive": True,
        },
    }


# =============================================================================
# API Contract Fixtures
# =============================================================================


@dataclass(frozen=True)
class APIEndpointContract:
    """Contract for an API endpoint.

    Attributes:
        name: Endpoint name
        required_params: Required parameter names
        optional_params: Optional parameter names
        param_types: Type specification for each parameter
        return_type: Expected return type
        raises_on: Conditions that should raise exceptions
    """

    name: str
    required_params: list[str]
    optional_params: list[str]
    param_types: dict[str, str]
    return_type: str
    raises_on: list[str]


@pytest.fixture
def theory_api_contract() -> APIEndpointContract:
    """Provide API contract for theory.calculate()."""
    return APIEndpointContract(
        name="theory.calculate",
        required_params=["params", "x"],
        optional_params=[],
        param_types={
            "params": "dict[str, float]",
            "x": "Array",
        },
        return_type="Array",
        raises_on=[
            "invalid_params_type",
            "invalid_x_type",
            "parameter_out_of_bounds",
            "shape_mismatch",
        ],
    )


# =============================================================================
# Serialization Contract Fixtures
# =============================================================================


@dataclass(frozen=True)
class SerializationContract:
    """Contract for serialization round-trip.

    Attributes:
        data_type: Type of data to serialize
        format: Serialization format (json, npz, pickle)
        preserves_precision: Whether precision is fully preserved
        preserves_types: Whether types are fully preserved
        supports_nan: Whether NaN is supported
        supports_complex: Whether complex numbers are supported
    """

    data_type: str
    format: str
    preserves_precision: bool
    preserves_types: bool
    supports_nan: bool
    supports_complex: bool


@pytest.fixture
def serialization_contracts() -> dict[str, SerializationContract]:
    """Provide serialization contracts."""
    return {
        "json_simple": SerializationContract(
            data_type="dict_with_primitives",
            format="json",
            preserves_precision=True,
            preserves_types=False,
            supports_nan=False,
            supports_complex=False,
        ),
        "npz_arrays": SerializationContract(
            data_type="numpy_arrays",
            format="npz",
            preserves_precision=True,
            preserves_types=True,
            supports_nan=True,
            supports_complex=True,
        ),
        "safe_mixed": SerializationContract(
            data_type="mixed_json_npz",
            format="json+npz",
            preserves_precision=True,
            preserves_types=True,
            supports_nan=True,
            supports_complex=True,
        ),
    }


# =============================================================================
# Synthetic Data for Contracts
# =============================================================================


@pytest.fixture
def synthetic_theory_input_data() -> dict[str, Array]:
    """Generate synthetic data for theory input contract validation.

    Returns:
        Dictionary with various input configurations.
    """
    return {
        "frequency_sweep": jnp.logspace(-2, 2, 50),
        "time_domain": jnp.logspace(-2, 2, 50),
        "stress_sweep": jnp.logspace(-3, 2, 100),
        "single_point": jnp.array([1.0]),
        "empty": jnp.array([]),
    }


@pytest.fixture
def synthetic_parameter_sets() -> dict[str, dict[str, float]]:
    """Generate synthetic parameter sets for contract validation.

    Returns:
        Dictionary mapping parameter set names to parameter dictionaries.
    """
    return {
        "maxwell_valid": {"G0": 1e5, "tau": 1.0},
        "maxwell_large_modulus": {"G0": 1e6, "tau": 0.1},
        "maxwell_small_modulus": {"G0": 1e3, "tau": 10.0},
        "rolie_poly_valid": {"G0": 1e5, "tau": 1.0, "nu": 2.0},
    }


# =============================================================================
# Validation Helpers
# =============================================================================


def validate_array_contract(
    array: Array,
    expected_shape_pattern: str,
    expected_dtype: str,
    nan_allowed: bool = False,
    inf_allowed: bool = False,
) -> list[ContractViolation]:
    """Validate array against contract specifications.

    Args:
        array: Array to validate
        expected_shape_pattern: Expected shape pattern like "(n,)" or "(n, m)"
        expected_dtype: Expected dtype like "float64"
        nan_allowed: Whether NaN values are allowed
        inf_allowed: Whether Inf values are allowed

    Returns:
        List of ContractViolation objects (empty if valid)
    """
    violations: list[ContractViolation] = []

    # Check dtype
    actual_dtype = str(array.dtype)
    if expected_dtype not in actual_dtype:
        violations.append(
            ContractViolation(
                component="array",
                contract_type="Data",
                violation_desc="dtype mismatch",
                expected=expected_dtype,
                actual=actual_dtype,
            )
        )

    # Check for NaN/Inf
    has_nan = bool(jnp.any(jnp.isnan(array)))
    if has_nan and not nan_allowed:
        violations.append(
            ContractViolation(
                component="array",
                contract_type="Data",
                violation_desc="Contains NaN",
                expected="no NaN",
                actual="NaN present",
            )
        )

    has_inf = bool(jnp.any(jnp.isinf(array)))
    if has_inf and not inf_allowed:
        violations.append(
            ContractViolation(
                component="array",
                contract_type="Data",
                violation_desc="Contains Inf",
                expected="no Inf",
                actual="Inf present",
            )
        )

    return violations


@pytest.fixture
def contract_validator() -> dict[str, Any]:
    """Provide contract validation functions.

    Returns:
        Dictionary of validation functions.
    """
    return {
        "validate_array": validate_array_contract,
    }
