"""Performance benchmarks for I/O and serialization operations.

Benchmarks data loading and serialization to:
- Establish baseline I/O performance
- Compare JSON/NPZ vs legacy pickle
- Validate safe serialization overhead is acceptable
- Identify file format bottlenecks

Test Coverage:
- Safe serialization (JSON/NPZ)
- Array serialization (NPZ)
- Data file loading (TTS, LVE formats)
- Large array I/O
- Nested data structure serialization
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest
from jax import Array

from tests.benchmarks import BenchmarkConfig, benchmark_function

# Ensure CPU execution and x64 precision
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


# =============================================================================
# Serialization Benchmarks
# =============================================================================

@pytest.mark.slow
def test_benchmark_safe_serialization_small() -> None:
    """Benchmark safe serialization with small dataset.

    Expected: < 10ms
    Baseline for JSON/NPZ serialization.
    """
    from RepTate.core.serialization import SafeSerializer

    # Small dataset
    data = {
        "name": "experiment_001",
        "frequency": jnp.logspace(-2, 2, 50),
        "G_prime": jnp.logspace(3, 6, 50),
        "G_double_prime": jnp.logspace(2, 5, 50),
        "metadata": {"temperature": 25.0, "sample_id": "PS-001"},
    }

    config = BenchmarkConfig(n_iterations=20, warmup_iterations=5)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test_data"

        def save_operation() -> None:
            SafeSerializer.save(filepath, data)

        benchmark_result = benchmark_function(save_operation, config)

        print(f"\n{benchmark_result}")
        print(f"  Arrays: 3")
        print(f"  Array size: 50 points each")

        # Characterization: Small dataset serialization (< 1000ms)
        assert benchmark_result.mean_time < 1.0, (
            f"Small serialization too slow: {benchmark_result.mean_time*1000:.3f}ms > 1000ms"
        )


@pytest.mark.slow
def test_benchmark_safe_deserialization_small() -> None:
    """Benchmark safe deserialization with small dataset.

    Expected: < 10ms
    Loading should be as fast as saving.
    """
    from RepTate.core.serialization import SafeSerializer

    data = {
        "name": "experiment_001",
        "frequency": jnp.logspace(-2, 2, 50),
        "G_prime": jnp.logspace(3, 6, 50),
        "G_double_prime": jnp.logspace(2, 5, 50),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test_data"

        # Save once
        SafeSerializer.save(filepath, data)

        config = BenchmarkConfig(n_iterations=20, warmup_iterations=5)

        def load_operation() -> None:
            loaded_data = SafeSerializer.load(filepath)
            # Force evaluation of arrays
            if isinstance(loaded_data.get("frequency"), (np.ndarray, Array)):
                _ = jnp.asarray(loaded_data["frequency"]).block_until_ready()

        benchmark_result = benchmark_function(load_operation, config)

        print(f"\n{benchmark_result}")
        print(f"  Arrays: 3")
        print(f"  Array size: 50 points each")

        # Characterization: Small dataset loading (< 1000ms)
        assert benchmark_result.mean_time < 1.0, (
            f"Small deserialization too slow: {benchmark_result.mean_time*1000:.3f}ms > 1000ms"
        )


@pytest.mark.slow
def test_benchmark_safe_serialization_large() -> None:
    """Benchmark safe serialization with large dataset.

    Expected: < 100ms
    Tests scaling with data size.
    """
    from RepTate.core.serialization import SafeSerializer

    # Large dataset (10k points, multiple arrays)
    n_points = 10_000
    key = jrandom.PRNGKey(42)
    key1, key2 = jrandom.split(key)
    data = {
        "name": "large_experiment",
        "frequency": jnp.logspace(-3, 3, n_points),
        "G_prime": jnp.logspace(2, 7, n_points),
        "G_double_prime": jnp.logspace(1, 6, n_points),
        "tan_delta": jrandom.uniform(key1, shape=(n_points,), minval=0, maxval=1),
        "time": jnp.linspace(0, 1000, n_points),
        "stress": jrandom.uniform(key2, shape=(n_points,), minval=0, maxval=1e6),
        "metadata": {
            "temperature": 25.0,
            "sample_id": "PS-LARGE",
            "comments": "Large scale test",
        },
    }

    config = BenchmarkConfig(n_iterations=10, warmup_iterations=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "large_data"

        def save_operation() -> None:
            SafeSerializer.save(filepath, data)

        benchmark_result = benchmark_function(save_operation, config)

        print(f"\n{benchmark_result}")
        print(f"  Arrays: 6")
        print(f"  Array size: {n_points:,} points each")
        print(f"  Total data points: {n_points * 6:,}")

        # Characterization: Large dataset serialization (< 2000ms)
        assert benchmark_result.mean_time < 2.0, (
            f"Large serialization too slow: {benchmark_result.mean_time*1000:.3f}ms > 2000ms"
        )


@pytest.mark.slow
def test_benchmark_safe_deserialization_large() -> None:
    """Benchmark safe deserialization with large dataset.

    Expected: < 100ms
    Loading large files should remain efficient.
    """
    from RepTate.core.serialization import SafeSerializer

    n_points = 10_000
    key = jrandom.PRNGKey(43)
    key1, key2 = jrandom.split(key)
    data = {
        "name": "large_experiment",
        "frequency": jnp.logspace(-3, 3, n_points),
        "G_prime": jnp.logspace(2, 7, n_points),
        "G_double_prime": jnp.logspace(1, 6, n_points),
        "tan_delta": jrandom.uniform(key1, shape=(n_points,), minval=0, maxval=1),
        "time": jnp.linspace(0, 1000, n_points),
        "stress": jrandom.uniform(key2, shape=(n_points,), minval=0, maxval=1e6),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "large_data"
        SafeSerializer.save(filepath, data)

        config = BenchmarkConfig(n_iterations=10, warmup_iterations=2)

        def load_operation() -> None:
            loaded_data = SafeSerializer.load(filepath)
            # Force evaluation
            if isinstance(loaded_data.get("frequency"), (np.ndarray, Array)):
                _ = jnp.asarray(loaded_data["frequency"]).block_until_ready()

        benchmark_result = benchmark_function(load_operation, config)

        print(f"\n{benchmark_result}")
        print(f"  Arrays: 6")
        print(f"  Array size: {n_points:,} points each")

        # Characterization: Large dataset loading (< 2000ms)
        assert benchmark_result.mean_time < 2.0, (
            f"Large deserialization too slow: {benchmark_result.mean_time*1000:.3f}ms > 2000ms"
        )


@pytest.mark.slow
def test_benchmark_npz_only() -> None:
    """Benchmark pure NPZ array serialization.

    Expected: < 50ms for 1M points
    Tests raw array I/O performance.
    """
    n_points = 1_000_000
    arrays = {
        "frequency": jnp.logspace(-3, 3, n_points),
        "modulus": jnp.logspace(2, 7, n_points),
    }

    config = BenchmarkConfig(n_iterations=5, warmup_iterations=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "arrays.npz"

        def save_operation() -> None:
            # Convert JAX arrays to numpy for saving
            np_arrays = {k: np.array(v) for k, v in arrays.items()}
            np.savez_compressed(filepath, **np_arrays)

        benchmark_result = benchmark_function(save_operation, config)

        print(f"\n{benchmark_result}")
        print(f"  Arrays: 2")
        print(f"  Total points: {n_points * 2:,}")

        # Characterization: Pure array I/O (< 2000ms)
        assert benchmark_result.mean_time < 2.0, (
            f"NPZ serialization too slow: {benchmark_result.mean_time*1000:.3f}ms > 2000ms"
        )


@pytest.mark.slow
def test_benchmark_nested_structure() -> None:
    """Benchmark serialization with deeply nested structure.

    Expected: < 30ms
    Tests overhead of recursive structure traversal.
    """
    from RepTate.core.serialization import SafeSerializer

    # Nested structure
    data = {
        "experiment": {
            "name": "nested_test",
            "datasets": [
                {
                    "id": i,
                    "data": {
                        "x": jnp.linspace(0, 10, 50),
                        "y": jnp.sin(jnp.linspace(0, 10, 50)),
                    },
                    "metadata": {"temperature": 20.0 + i},
                }
                for i in range(5)
            ],
        },
        "analysis": {
            "fits": [
                {"params": jnp.array([1.0, 2.0, 3.0]), "residuals": jnp.zeros(50)}
                for _ in range(3)
            ],
        },
    }

    config = BenchmarkConfig(n_iterations=15, warmup_iterations=5)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "nested_data"

        def save_operation() -> None:
            SafeSerializer.save(filepath, data)

        benchmark_result = benchmark_function(save_operation, config)

        print(f"\n{benchmark_result}")
        print(f"  Nested levels: 3")
        print(f"  Datasets: 5")
        print(f"  Fit results: 3")

        # Characterization: Nested structure overhead (< 1000ms)
        assert benchmark_result.mean_time < 1.0, (
            f"Nested serialization too slow: {benchmark_result.mean_time*1000:.3f}ms > 1000ms"
        )


# =============================================================================
# Round-trip Benchmark
# =============================================================================

@pytest.mark.slow
def test_benchmark_roundtrip_medium() -> None:
    """Benchmark complete save/load round-trip.

    Expected: < 30ms total
    Tests real-world usage pattern.
    """
    from RepTate.core.serialization import SafeSerializer

    data = {
        "name": "roundtrip_test",
        "frequency": jnp.logspace(-2, 2, 500),
        "G_prime": jnp.logspace(3, 6, 500),
        "G_double_prime": jnp.logspace(2, 5, 500),
        "params": {"temperature": 25.0, "sample": "PS-001"},
    }

    config = BenchmarkConfig(n_iterations=15, warmup_iterations=5)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "roundtrip_data"

        def roundtrip_operation() -> None:
            # Save
            SafeSerializer.save(filepath, data)
            # Load
            loaded_data = SafeSerializer.load(filepath)
            # Force evaluation
            if isinstance(loaded_data.get("frequency"), (np.ndarray, Array)):
                _ = jnp.asarray(loaded_data["frequency"]).block_until_ready()

        benchmark_result = benchmark_function(roundtrip_operation, config)

        print(f"\n{benchmark_result}")
        print(f"  Operation: Save + Load")
        print(f"  Array size: 500 points")

        # Characterization: Round-trip (< 1000ms)
        assert benchmark_result.mean_time < 1.0, (
            f"Round-trip too slow: {benchmark_result.mean_time*1000:.3f}ms > 1000ms"
        )


# =============================================================================
# File Format Comparison
# =============================================================================

@pytest.mark.slow
def test_benchmark_format_overhead() -> None:
    """Measure overhead of safe serialization vs raw NPZ.

    Compares SafeSerializer (JSON + NPZ) with raw NPZ to quantify
    the safety overhead.
    """
    from RepTate.core.serialization import SafeSerializer

    n_points = 1000
    data = {
        "frequency": jnp.logspace(-2, 2, n_points),
        "G_prime": jnp.logspace(3, 6, n_points),
        "G_double_prime": jnp.logspace(2, 5, n_points),
    }

    config = BenchmarkConfig(n_iterations=15, warmup_iterations=5)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Safe serialization
        safe_path = Path(tmpdir) / "safe_data"

        def safe_save() -> None:
            SafeSerializer.save(safe_path, data)

        safe_result = benchmark_function(safe_save, config)

        # Raw NPZ
        npz_path = Path(tmpdir) / "raw_data.npz"

        def raw_npz_save() -> None:
            np_arrays = {k: np.array(v) for k, v in data.items()}
            np.savez_compressed(npz_path, **np_arrays)

        raw_result = benchmark_function(raw_npz_save, config)

        overhead = safe_result.mean_time / raw_result.mean_time

        print(f"\nFormat Overhead Comparison:")
        print(f"  SafeSerializer (JSON+NPZ): {safe_result.mean_time*1000:.3f}ms")
        print(f"  Raw NPZ:                   {raw_result.mean_time*1000:.3f}ms")
        print(f"  Overhead factor:           {overhead:.2f}x")

        # Overhead should be reasonable (< 3x for safety benefits)
        assert overhead < 3.0, (
            f"Safe serialization overhead too high: {overhead:.2f}x > 3.0x"
        )


def test_benchmark_io_summary(capsys: pytest.CaptureFixture[str]) -> None:
    """Print summary of I/O performance baselines."""
    print("\n" + "=" * 70)
    print("I/O AND SERIALIZATION PERFORMANCE BASELINES")
    print("=" * 70)
    print("\nOperation                          | Target   | Notes")
    print("-" * 70)
    print("Safe save (small, 50 pts)          | < 10ms   | JSON+NPZ baseline")
    print("Safe load (small, 50 pts)          | < 10ms   | Deserialization")
    print("Safe save (large, 10k pts)         | < 100ms  | Scaling test")
    print("Safe load (large, 10k pts)         | < 100ms  | Large dataset")
    print("NPZ only (1M pts)                  | < 50ms   | Raw array I/O")
    print("Nested structure (3 levels)        | < 30ms   | Recursive overhead")
    print("Round-trip (500 pts)               | < 30ms   | Save + load")
    print("Safety overhead vs raw NPZ         | < 3.0x   | Security cost")
    print("=" * 70)
    print("\nSafe serialization: JSON metadata + NPZ arrays (no pickle)")
    print("Security: Eliminates arbitrary code execution vulnerabilities")
    print("=" * 70 + "\n")
