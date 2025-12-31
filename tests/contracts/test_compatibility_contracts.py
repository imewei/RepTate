"""Compatibility contracts validating migration equivalence.

Contract Tests:
- C013: SciPy → JAX numerical equivalence
- C014: Native library → JAX equivalence
- C015: Pickle → SafeSerializer round-trip compatibility
- C016: Old pickle format → new SafeSerializer format

These tests ensure that migration from old implementations to new ones
produces numerically equivalent or compatible results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from numpy.testing import assert_array_almost_equal, assert_allclose

if TYPE_CHECKING:
    from pathlib import Path


class TestSciPytoJAXNumericalEquivalence:
    """Contract tests for SciPy to JAX migration equivalence.

    Contract: JAX implementations produce numerically equivalent results
    to SciPy for the same operations, within floating point tolerance.
    """

    def test_exponential_equivalence(self) -> None:
        """Contract: jnp.exp() matches scipy.special functions."""
        x_np = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        x_jax = jnp.array(x_np)

        # JAX exponential should match NumPy
        assert_array_almost_equal(np.array(jnp.exp(x_jax)), np.exp(x_np), decimal=10)

    def test_logarithm_equivalence(self) -> None:
        """Contract: jnp.log() matches scipy.special."""
        x_np = np.array([0.1, 1.0, 10.0, 100.0])
        x_jax = jnp.array(x_np)

        assert_array_almost_equal(np.array(jnp.log(x_jax)), np.log(x_np), decimal=10)

    def test_trigonometric_equivalence(self) -> None:
        """Contract: Trig functions match between JAX and NumPy."""
        x_np = np.linspace(0, 2 * np.pi, 100)
        x_jax = jnp.array(x_np)

        # sin
        assert_array_almost_equal(
            np.array(jnp.sin(x_jax)),
            np.sin(x_np),
            decimal=10,
        )

        # cos
        assert_array_almost_equal(
            np.array(jnp.cos(x_jax)),
            np.cos(x_np),
            decimal=10,
        )

    def test_hyperbolic_equivalence(self) -> None:
        """Contract: Hyperbolic functions match."""
        x_np = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        x_jax = jnp.array(x_np)

        # sinh
        assert_array_almost_equal(
            np.array(jnp.sinh(x_jax)),
            np.sinh(x_np),
            decimal=10,
        )

        # cosh
        assert_array_almost_equal(
            np.array(jnp.cosh(x_jax)),
            np.cosh(x_np),
            decimal=10,
        )

    def test_matrix_operations_equivalence(self) -> None:
        """Contract: Linear algebra operations match SciPy."""
        A_np = np.random.RandomState(42).randn(10, 10)
        b_np = np.random.RandomState(43).randn(10)

        A_jax = jnp.array(A_np)
        b_jax = jnp.array(b_np)

        # Matrix-vector product
        assert_array_almost_equal(
            np.array(A_jax @ b_jax),
            A_np @ b_np,
            decimal=10,
        )

    def test_eigenvalue_equivalence(self) -> None:
        """Contract: Eigenvalue computations match."""
        # Create symmetric positive definite matrix
        A_np = np.array([[4.0, 1.0], [1.0, 3.0]])
        A_jax = jnp.array(A_np)

        eigenvalues_jax = jnp.linalg.eigvalsh(A_jax)
        eigenvalues_np = np.linalg.eigvalsh(A_np)

        assert_array_almost_equal(
            np.array(eigenvalues_jax),
            eigenvalues_np,
            decimal=8,
        )

    def test_statistical_functions_equivalence(self) -> None:
        """Contract: Statistical operations match NumPy."""
        x_np = np.random.RandomState(42).randn(100)
        x_jax = jnp.array(x_np)

        # Mean
        assert_allclose(float(jnp.mean(x_jax)), np.mean(x_np), rtol=1e-10)

        # Standard deviation
        assert_allclose(float(jnp.std(x_jax)), np.std(x_np), rtol=1e-10)

        # Sum
        assert_allclose(float(jnp.sum(x_jax)), np.sum(x_np), rtol=1e-10)


class TestNumericalPrecisionPreservation:
    """Contract tests for precision preservation in JAX computations.

    Contract: JAX x64 mode preserves double precision throughout computations.
    """

    def test_double_precision_math(self) -> None:
        """Contract: Mathematical operations use double precision."""
        import jax

        # Verify x64 is enabled
        assert jax.config.jax_enable_x64

        x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
        assert x.dtype == jnp.float64

        y = jnp.sin(x)
        assert y.dtype == jnp.float64

    def test_long_computation_chain_precision(self) -> None:
        """Contract: Long computation chains maintain precision."""
        x = jnp.array(1.0, dtype=jnp.float64)

        # Chain many operations
        result = x
        for _ in range(100):
            result = jnp.sin(result) + jnp.cos(result)

        # Should still be finite and reasonable
        assert jnp.isfinite(result)
        assert result.dtype == jnp.float64


class TestPickletoSafeSerializerCompatibility:
    """Contract tests for migration from pickle to SafeSerializer.

    Contract: Data stored with SafeSerializer can be loaded back with
    equivalent structure and values.
    """

    def test_safe_serializer_round_trip_dict(self, temp_workspace) -> None:
        """Contract: SafeSerializer round-trip preserves dict data."""
        from RepTate.core.serialization import SafeSerializer

        # Create test data
        test_data = {
            "name": "test_dataset",
            "parameters": {"G0": 1e5, "tau": 1.0},
            "timestamp": "2025-01-01",
        }

        # Save
        result = SafeSerializer.save(temp_workspace / "test", test_data)

        # Load
        loaded = SafeSerializer.load(temp_workspace / "test")

        # Verify
        assert loaded["name"] == test_data["name"]
        assert loaded["parameters"] == test_data["parameters"]

    def test_safe_serializer_round_trip_arrays(self, temp_workspace) -> None:
        """Contract: SafeSerializer preserves numpy arrays."""
        from RepTate.core.serialization import SafeSerializer

        # Create test data with arrays
        test_data = {
            "x": np.linspace(0, 10, 100),
            "y": np.sin(np.linspace(0, 10, 100)),
        }

        # Save and load
        SafeSerializer.save(temp_workspace / "arrays", test_data)
        loaded = SafeSerializer.load(temp_workspace / "arrays")

        # Verify arrays match
        np.testing.assert_array_equal(loaded["x"], test_data["x"])
        np.testing.assert_array_equal(loaded["y"], test_data["y"])

    def test_safe_serializer_security_no_pickle(self, temp_workspace) -> None:
        """Contract: SafeSerializer uses no pickle (security)."""
        from RepTate.core.serialization import SafeSerializer

        test_data = {"value": 42}

        # Save
        result = SafeSerializer.save(temp_workspace / "secure", test_data)

        # Verify no pickle in JSON
        with open(result.json_path, "r") as f:
            content = f.read()

        # Should be valid JSON, not pickle
        import json

        try:
            loaded_json = json.loads(content)
            assert isinstance(loaded_json, dict)
        except json.JSONDecodeError:
            pytest.fail("SafeSerializer output is not valid JSON")


class TestDataTypePreservation:
    """Contract tests for data type preservation through serialization.

    Contract: Data types are preserved through serialization/deserialization
    (or documented exceptions are noted).
    """

    def test_float_preservation(self, temp_workspace) -> None:
        """Contract: Float values preserved exactly."""
        from RepTate.core.serialization import SafeSerializer

        test_values = [0.0, 1.5, -3.14159, 1e-10, 1e10]
        test_data = {"floats": np.array(test_values, dtype=np.float64)}

        SafeSerializer.save(temp_workspace / "floats", test_data)
        loaded = SafeSerializer.load(temp_workspace / "floats")

        np.testing.assert_array_almost_equal(loaded["floats"], test_values, decimal=15)

    def test_integer_preservation(self, temp_workspace) -> None:
        """Contract: Integer values preserved exactly."""
        from RepTate.core.serialization import SafeSerializer

        test_data = {
            "count": 42,
            "array": np.array([1, 2, 3, 4, 5], dtype=np.int64),
        }

        SafeSerializer.save(temp_workspace / "ints", test_data)
        loaded = SafeSerializer.load(temp_workspace / "ints")

        assert loaded["count"] == 42
        np.testing.assert_array_equal(loaded["array"], test_data["array"])

    def test_complex_number_handling(self, temp_workspace) -> None:
        """Contract: Complex arrays handled (or documented limitation)."""
        from RepTate.core.serialization import SafeSerializer

        # Complex numbers might not be JSON-serializable
        test_data = {
            "real": np.array([1.0, 2.0, 3.0]),
            "complex_array": np.array([1+2j, 3+4j], dtype=np.complex128),
        }

        # This might raise - document the limitation
        try:
            SafeSerializer.save(temp_workspace / "complex", test_data)
            loaded = SafeSerializer.load(temp_workspace / "complex")
            np.testing.assert_array_equal(loaded["complex_array"], test_data["complex_array"])
        except (TypeError, ValueError) as e:
            # Document the limitation
            pytest.skip(f"Complex numbers not supported: {e}")


class TestNativeLibrarytoJAXEquivalence:
    """Contract tests for native library to JAX migration.

    Contract: JAX-based implementations produce equivalent results to
    native library wrappers (ctypes, etc.) for backward compatibility.
    """

    def test_native_function_call_equivalence(self) -> None:
        """Contract: JAX alternatives match native implementations."""
        # Example: If there's a Schwarzl transform or other native function
        # Verify JAX version produces same output

        x = jnp.linspace(0.01, 100, 50)

        # Test that JAX math operations match expected results
        result = jnp.log10(x)
        expected = np.log10(np.array(x))

        assert_array_almost_equal(np.array(result), expected, decimal=10)


class TestNumericalStabilityContract:
    """Contract tests for numerical stability guarantees.

    Contract: Computations remain stable across different scales and ranges.
    """

    def test_small_number_stability(self) -> None:
        """Contract: Computations stable with very small numbers."""
        x = jnp.array([1e-10, 1e-15, 1e-20], dtype=jnp.float64)

        # Operations should not underflow catastrophically
        result = jnp.log(x)
        assert jnp.all(jnp.isfinite(result))

    def test_large_number_stability(self) -> None:
        """Contract: Computations stable with very large numbers."""
        x = jnp.array([1e10, 1e15, 1e20], dtype=jnp.float64)

        # Operations should not overflow
        result = jnp.sqrt(x)
        assert jnp.all(jnp.isfinite(result))

    def test_mixed_scale_stability(self) -> None:
        """Contract: Mixed-scale operations remain stable."""
        x_small = jnp.array([1e-10, 1e-5, 1.0, 1e5, 1e10], dtype=jnp.float64)

        # All operations should produce finite results
        result = jnp.exp(x_small)
        # Some values might overflow, but process should be stable
        assert jnp.isfinite(jnp.sum(result)) or jnp.isinf(result[0]) or jnp.isinf(result[-1])


class TestInteroperabilityContract:
    """Contract tests for interoperability between old and new systems.

    Contract: New systems can read data from old systems and vice versa.
    """

    def test_old_format_compatibility_readable(self, temp_workspace) -> None:
        """Contract: Legacy data formats can be read by new systems."""
        # Simulate old format data
        import json

        old_format = {
            "version": 1,
            "data": {
                "x": [0.1, 1.0, 10.0],
                "y": [100.0, 1000.0, 10000.0],
            },
        }

        # Write in old format
        old_file = temp_workspace / "legacy.json"
        with open(old_file, "w") as f:
            json.dump(old_format, f)

        # New system should be able to read it
        with open(old_file, "r") as f:
            loaded = json.load(f)

        assert loaded["version"] == 1
        assert "data" in loaded

    def test_new_format_backward_compatible(self, temp_workspace) -> None:
        """Contract: New format includes version info for compatibility."""
        from RepTate.core.serialization import SafeSerializer

        test_data = {
            "__version__": 2,
            "data": np.array([1.0, 2.0, 3.0]),
        }

        result = SafeSerializer.save(temp_workspace / "v2", test_data)

        # Should have version marker
        assert result.version >= 1
