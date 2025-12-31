"""Shared pytest fixtures for RepTate test suite.

This module provides common fixtures used across unit, integration,
characterization, and regression tests.

Fixtures:
    - Synthetic data fixtures for theory testing
    - Temporary workspace fixtures for file operations
    - JAX CPU enforcement for reproducible tests
    - Mock implementations of protocol interfaces

Usage:
    def test_theory_calculation(synthetic_frequency_data: SyntheticData) -> None:
        x, y = synthetic_frequency_data.x, synthetic_frequency_data.y
        # ... test code
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import pytest
from jax import Array

if TYPE_CHECKING:
    from collections.abc import Generator


# =============================================================================
# JAX CPU Enforcement
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
def enforce_jax_cpu() -> None:
    """Enforce JAX to use CPU for reproducible tests.

    This fixture runs automatically at the start of the test session
    to ensure all JAX computations run on CPU, providing deterministic
    results across different hardware configurations.

    Note:
        This must run before any JAX operations to be effective.
    """
    jax.config.update("jax_platform_name", "cpu")
    # Also set x64 mode for full precision (matching constitution requirement)
    jax.config.update("jax_enable_x64", True)


# =============================================================================
# Synthetic Data Fixtures
# =============================================================================

@dataclass(frozen=True)
class SyntheticData:
    """Container for synthetic test data.

    Attributes:
        x: Independent variable array
        y: Dependent variable array
        params: Parameters used to generate the data
        noise_level: Standard deviation of added noise (0 for clean data)
    """
    x: Array
    y: Array
    params: dict[str, float] = field(default_factory=dict)
    noise_level: float = 0.0


@pytest.fixture
def synthetic_frequency_data() -> SyntheticData:
    """Generate synthetic frequency sweep data for LVE testing.

    Returns:
        SyntheticData with frequency range 0.01 to 100 rad/s,
        simulating a simple Maxwell model response.

    The data follows: G'(omega) = G0 * (omega*tau)^2 / (1 + (omega*tau)^2)
    """
    omega = jnp.logspace(-2, 2, 50)  # 0.01 to 100 rad/s
    G0 = 1e5  # Modulus
    tau = 1.0  # Relaxation time

    # Maxwell model storage modulus
    omega_tau = omega * tau
    G_prime = G0 * omega_tau**2 / (1 + omega_tau**2)

    return SyntheticData(
        x=omega,
        y=G_prime,
        params={"G0": G0, "tau": tau},
        noise_level=0.0,
    )


@pytest.fixture
def synthetic_frequency_data_noisy() -> SyntheticData:
    """Generate synthetic frequency data with noise.

    Returns:
        SyntheticData with 5% Gaussian noise added.
    """
    omega = jnp.logspace(-2, 2, 50)
    G0 = 1e5
    tau = 1.0

    omega_tau = omega * tau
    G_prime = G0 * omega_tau**2 / (1 + omega_tau**2)

    # Add 5% Gaussian noise (reproducible with fixed key)
    key = jax.random.PRNGKey(42)
    noise = jax.random.normal(key, shape=G_prime.shape) * 0.05 * G_prime
    G_prime_noisy = G_prime + noise

    return SyntheticData(
        x=omega,
        y=G_prime_noisy,
        params={"G0": G0, "tau": tau},
        noise_level=0.05,
    )


@pytest.fixture
def synthetic_time_data() -> SyntheticData:
    """Generate synthetic time-domain data for relaxation testing.

    Returns:
        SyntheticData with time range 0.01 to 100 s,
        simulating exponential relaxation.

    The data follows: G(t) = G0 * exp(-t/tau)
    """
    t = jnp.logspace(-2, 2, 50)  # 0.01 to 100 s
    G0 = 1e5  # Initial modulus
    tau = 1.0  # Relaxation time

    # Exponential relaxation
    G_t = G0 * jnp.exp(-t / tau)

    return SyntheticData(
        x=t,
        y=G_t,
        params={"G0": G0, "tau": tau},
        noise_level=0.0,
    )


@pytest.fixture
def synthetic_multimode_data() -> SyntheticData:
    """Generate synthetic multi-mode Maxwell data.

    Returns:
        SyntheticData with 3 Maxwell modes for more complex fitting tests.
    """
    omega = jnp.logspace(-3, 3, 100)

    # Three Maxwell modes
    G_values = jnp.array([1e5, 5e4, 2e4])
    tau_values = jnp.array([10.0, 1.0, 0.1])

    # Sum of Maxwell modes for G'
    G_prime = jnp.zeros_like(omega)
    for G0, tau in zip(G_values, tau_values):
        omega_tau = omega * tau
        G_prime = G_prime + G0 * omega_tau**2 / (1 + omega_tau**2)

    return SyntheticData(
        x=omega,
        y=G_prime,
        params={
            "G1": float(G_values[0]), "tau1": float(tau_values[0]),
            "G2": float(G_values[1]), "tau2": float(tau_values[1]),
            "G3": float(G_values[2]), "tau3": float(tau_values[2]),
        },
        noise_level=0.0,
    )


# =============================================================================
# Temporary Workspace Fixtures
# =============================================================================

@pytest.fixture
def temp_workspace() -> Generator[Path, None, None]:
    """Create a temporary workspace directory for file operations.

    Yields:
        Path to a temporary directory that is automatically cleaned up
        after the test.

    Usage:
        def test_file_save(temp_workspace: Path) -> None:
            filepath = temp_workspace / "test_data.json"
            # ... test code
    """
    with tempfile.TemporaryDirectory(prefix="reptate_test_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_data_file(temp_workspace: Path) -> Path:
    """Create a temporary data file with sample content.

    Args:
        temp_workspace: Parent temporary directory

    Returns:
        Path to a temporary file with sample frequency data.
    """
    filepath = temp_workspace / "sample_data.tts"
    content = """# Sample TTS data file
# omega G' G''
0.01 100.0 10.0
0.1 1000.0 100.0
1.0 10000.0 1000.0
10.0 50000.0 5000.0
100.0 90000.0 9000.0
"""
    filepath.write_text(content)
    return filepath


@pytest.fixture
def temp_json_file(temp_workspace: Path) -> Path:
    """Create a temporary JSON file for serialization tests.

    Args:
        temp_workspace: Parent temporary directory

    Returns:
        Path to a JSON file (not yet created).
    """
    return temp_workspace / "test_data.json"


@pytest.fixture
def temp_npz_file(temp_workspace: Path) -> Path:
    """Create a temporary NPZ file for array serialization tests.

    Args:
        temp_workspace: Parent temporary directory

    Returns:
        Path to an NPZ file (not yet created).
    """
    return temp_workspace / "test_arrays.npz"


# =============================================================================
# Mock Protocol Implementations
# =============================================================================

class MockParameter:
    """Mock Parameter class for testing ITheory interface.

    Implements the minimal interface needed to test theory protocols.
    """

    def __init__(
        self,
        name: str,
        value: float = 0.0,
        min_value: float = -1e10,
        max_value: float = 1e10,
    ) -> None:
        self.name = name
        self.value = value
        self.min_value = min_value
        self.max_value = max_value
        self.opt_type = "opt"  # Simplified for testing


class MockTheory:
    """Mock implementation of ITheory for testing.

    Implements a simple linear model: y = slope * x + intercept
    """

    def __init__(self) -> None:
        self._name = "MockTheory"
        self._description = "Mock theory for testing"
        self._parameters: dict[str, MockParameter] = {
            "slope": MockParameter("slope", value=1.0),
            "intercept": MockParameter("intercept", value=0.0),
        }

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def calculate(
        self,
        params: dict[str, float],
        x: Array,
    ) -> Array:
        slope = params.get("slope", 1.0)
        intercept = params.get("intercept", 0.0)
        return slope * x + intercept

    def get_parameters(self) -> dict[str, MockParameter]:
        return self._parameters

    def set_parameter(self, name: str, value: float) -> None:
        if name not in self._parameters:
            raise KeyError(f"Parameter '{name}' not found")
        self._parameters[name].value = value

    def get_fit_parameters(self) -> list[str]:
        return [name for name in self._parameters]


class MockDataset:
    """Mock implementation of IDataset for testing.

    Provides synthetic data conforming to the IDataset protocol.
    """

    def __init__(
        self,
        name: str = "MockDataset",
        x: Array | None = None,
        y: Array | None = None,
    ) -> None:
        self._name = name
        self._x = x if x is not None else jnp.linspace(0, 10, 100)
        self._y = y if y is not None else jnp.sin(self._x)
        self._columns = ["x", "y"]
        self._error: Array | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def columns(self) -> list[str]:
        return self._columns

    def get_x(self) -> Array:
        return self._x

    def get_y(self) -> Array:
        return self._y

    def get_column(self, name: str) -> Array:
        if name == "x":
            return self._x
        elif name == "y":
            return self._y
        else:
            raise KeyError(f"Column '{name}' not found")

    def get_error(self) -> Array | None:
        return self._error


class MockApplication:
    """Mock implementation of IApplication for testing.

    Provides a minimal application conforming to the IApplication protocol.
    """

    def __init__(self) -> None:
        self._name = "MockApp"
        self._extension = "mock"
        self._datasets: list[MockDataset] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def extension(self) -> str:
        return self._extension

    def get_theories(self) -> list[type[MockTheory]]:
        return [MockTheory]

    def load_data(self, filepath: str) -> MockDataset:
        # Simple mock implementation
        if not Path(filepath).exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        dataset = MockDataset(name=Path(filepath).stem)
        self._datasets.append(dataset)
        return dataset

    def get_datasets(self) -> list[MockDataset]:
        return self._datasets

    def add_dataset(self, dataset: MockDataset) -> None:
        self._datasets.append(dataset)

    def remove_dataset(self, name: str) -> None:
        for i, ds in enumerate(self._datasets):
            if ds.name == name:
                del self._datasets[i]
                return
        raise KeyError(f"Dataset '{name}' not found")


class MockFitResult:
    """Mock implementation of IFitResult for testing.

    Provides synthetic fit results conforming to the IFitResult protocol.
    """

    def __init__(
        self,
        parameters: dict[str, float] | None = None,
        success: bool = True,
    ) -> None:
        self._parameters = parameters or {"slope": 1.0, "intercept": 0.0}
        self._success = success
        self._residuals = jnp.zeros(10)
        self._covariance = jnp.eye(len(self._parameters)) * 0.01

    @property
    def parameters(self) -> dict[str, float]:
        return self._parameters

    @property
    def covariance(self) -> Array | None:
        return self._covariance

    @property
    def residuals(self) -> Array:
        return self._residuals

    @property
    def success(self) -> bool:
        return self._success

    def get_uncertainty(self, name: str) -> float:
        if name not in self._parameters:
            raise KeyError(f"Parameter '{name}' not found")
        if self._covariance is None:
            raise ValueError("Covariance matrix not available")
        # Get index and return sqrt of diagonal element
        param_names = list(self._parameters.keys())
        idx = param_names.index(name)
        return float(jnp.sqrt(self._covariance[idx, idx]))


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_theory() -> MockTheory:
    """Provide a mock theory instance for testing.

    Returns:
        MockTheory instance implementing ITheory protocol.
    """
    return MockTheory()


@pytest.fixture
def mock_dataset() -> MockDataset:
    """Provide a mock dataset instance for testing.

    Returns:
        MockDataset instance implementing IDataset protocol.
    """
    return MockDataset()


@pytest.fixture
def mock_application() -> MockApplication:
    """Provide a mock application instance for testing.

    Returns:
        MockApplication instance implementing IApplication protocol.
    """
    return MockApplication()


@pytest.fixture
def mock_fit_result() -> MockFitResult:
    """Provide a mock fit result instance for testing.

    Returns:
        MockFitResult instance implementing IFitResult protocol.
    """
    return MockFitResult()


# =============================================================================
# Test Data Paths
# =============================================================================

@pytest.fixture
def test_data_dir() -> Path:
    """Return the path to the test data directory.

    Returns:
        Path to the tests directory containing test data files.
    """
    return Path(__file__).parent


@pytest.fixture
def sample_lve_file(test_data_dir: Path) -> Path | None:
    """Return path to a sample LVE data file if it exists.

    Returns:
        Path to test_LVE_LML.txt if it exists, None otherwise.
    """
    filepath = test_data_dir / "test_LVE_LML.txt"
    return filepath if filepath.exists() else None


# =============================================================================
# Numerical Tolerance Fixtures
# =============================================================================

@pytest.fixture
def numerical_tolerance() -> float:
    """Return the numerical tolerance for JAX comparisons.

    Returns:
        1e-10 tolerance as specified in constitution requirements.
    """
    return 1e-10


@pytest.fixture
def relaxed_tolerance() -> float:
    """Return a relaxed tolerance for noisy data comparisons.

    Returns:
        1e-6 tolerance for tests involving noise or approximations.
    """
    return 1e-6
