"""Performance benchmarks for interpolation operations.

Benchmarks interpax (JAX-based interpolation) to:
- Establish baseline interpolation performance
- Validate JAX implementation efficiency
- Compare different interpolation methods
- Test scaling with data size

Test Coverage:
- Linear interpolation
- Cubic spline interpolation
- Different data sizes
- Extrapolation handling
- Batch interpolation
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from tests.benchmarks import BenchmarkConfig, benchmark_function

# Ensure CPU execution and x64 precision
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


# =============================================================================
# Interpolation Benchmarks
# =============================================================================

@pytest.mark.slow
def test_benchmark_linear_interp_small() -> None:
    """Benchmark linear interpolation with small dataset.

    Expected: < 5ms (after JIT warmup)
    Baseline for interpolation operations.
    """
    from interpax import interp1d

    # Known data points
    x = jnp.linspace(0, 10, 50)
    y = jnp.sin(x)

    # Query points
    x_new = jnp.linspace(0, 10, 200)

    # Create interpolator
    @jax.jit
    def interpolate() -> Array:
        return interp1d(x_new, x, y, method="linear")

    config = BenchmarkConfig(n_iterations=30, warmup_iterations=10)

    def interp_operation() -> None:
        result = interpolate()
        _ = result.block_until_ready()

    benchmark_result = benchmark_function(interp_operation, config)

    print(f"\n{benchmark_result}")
    print(f"  Known points: 50")
    print(f"  Query points: 200")
    print(f"  Method: linear")

    # SLA: Linear interpolation should be very fast
    assert benchmark_result.mean_time < 0.005, (
        f"Linear interpolation too slow: {benchmark_result.mean_time*1000:.3f}ms > 5ms"
    )


@pytest.mark.slow
def test_benchmark_cubic_interp_small() -> None:
    """Benchmark cubic spline interpolation with small dataset.

    Expected: < 10ms (after JIT warmup)
    Cubic splines are more expensive but still fast.
    """
    from interpax import interp1d

    x = jnp.linspace(0, 10, 50)
    y = jnp.sin(x)
    x_new = jnp.linspace(0, 10, 200)

    @jax.jit
    def interpolate() -> Array:
        return interp1d(x_new, x, y, method="cubic")

    config = BenchmarkConfig(n_iterations=30, warmup_iterations=10)

    def interp_operation() -> None:
        result = interpolate()
        _ = result.block_until_ready()

    benchmark_result = benchmark_function(interp_operation, config)

    print(f"\n{benchmark_result}")
    print(f"  Known points: 50")
    print(f"  Query points: 200")
    print(f"  Method: cubic")

    # SLA: Cubic interpolation should still be fast
    assert benchmark_result.mean_time < 0.010, (
        f"Cubic interpolation too slow: {benchmark_result.mean_time*1000:.3f}ms > 10ms"
    )


@pytest.mark.slow
def test_benchmark_linear_interp_medium() -> None:
    """Benchmark linear interpolation with medium dataset.

    Expected: < 10ms (after JIT warmup)
    Tests scaling behavior.
    """
    from interpax import interp1d

    # Larger dataset
    x = jnp.linspace(0, 100, 500)
    y = jnp.sin(x) + 0.1 * jnp.cos(5 * x)
    x_new = jnp.linspace(0, 100, 2000)

    @jax.jit
    def interpolate() -> Array:
        return interp1d(x_new, x, y, method="linear")

    config = BenchmarkConfig(n_iterations=20, warmup_iterations=10)

    def interp_operation() -> None:
        result = interpolate()
        _ = result.block_until_ready()

    benchmark_result = benchmark_function(interp_operation, config)

    print(f"\n{benchmark_result}")
    print(f"  Known points: 500")
    print(f"  Query points: 2000")
    print(f"  Method: linear")

    # SLA: Should scale well
    assert benchmark_result.mean_time < 0.010, (
        f"Medium linear interpolation too slow: {benchmark_result.mean_time*1000:.3f}ms > 10ms"
    )


@pytest.mark.slow
def test_benchmark_cubic_interp_medium() -> None:
    """Benchmark cubic interpolation with medium dataset.

    Expected: < 20ms (after JIT warmup)
    Cubic scales worse than linear but should remain reasonable.
    """
    from interpax import interp1d

    x = jnp.linspace(0, 100, 500)
    y = jnp.sin(x) + 0.1 * jnp.cos(5 * x)
    x_new = jnp.linspace(0, 100, 2000)

    @jax.jit
    def interpolate() -> Array:
        return interp1d(x_new, x, y, method="cubic")

    config = BenchmarkConfig(n_iterations=20, warmup_iterations=10)

    def interp_operation() -> None:
        result = interpolate()
        _ = result.block_until_ready()

    benchmark_result = benchmark_function(interp_operation, config)

    print(f"\n{benchmark_result}")
    print(f"  Known points: 500")
    print(f"  Query points: 2000")
    print(f"  Method: cubic")

    # SLA: Cubic should scale reasonably
    assert benchmark_result.mean_time < 0.020, (
        f"Medium cubic interpolation too slow: {benchmark_result.mean_time*1000:.3f}ms > 20ms"
    )


@pytest.mark.slow
def test_benchmark_linear_interp_large() -> None:
    """Benchmark linear interpolation with large dataset.

    Expected: < 20ms (after JIT warmup)
    Tests extreme scaling.
    """
    from interpax import interp1d

    # Large dataset
    x = jnp.linspace(0, 1000, 5000)
    y = jnp.sin(x / 10) + 0.1 * jnp.cos(x)
    x_new = jnp.linspace(0, 1000, 10000)

    @jax.jit
    def interpolate() -> Array:
        return interp1d(x_new, x, y, method="linear")

    config = BenchmarkConfig(n_iterations=10, warmup_iterations=5)

    def interp_operation() -> None:
        result = interpolate()
        _ = result.block_until_ready()

    benchmark_result = benchmark_function(interp_operation, config)

    print(f"\n{benchmark_result}")
    print(f"  Known points: 5000")
    print(f"  Query points: 10000")
    print(f"  Method: linear")

    # SLA: Large interpolation should remain efficient
    assert benchmark_result.mean_time < 0.020, (
        f"Large linear interpolation too slow: {benchmark_result.mean_time*1000:.3f}ms > 20ms"
    )


@pytest.mark.slow
def test_benchmark_log_scale_interp() -> None:
    """Benchmark interpolation on log-scale data (typical for rheology).

    Expected: < 8ms (after JIT warmup)
    Rheology data is often log-spaced.
    """
    from interpax import interp1d

    # Log-spaced frequency data (typical rheology)
    omega = jnp.logspace(-2, 2, 50)
    G_prime = 1e5 * omega**2 / (1 + omega**2)

    # Query at finer log-spacing
    omega_new = jnp.logspace(-2, 2, 200)

    # Interpolate in log-space for better accuracy
    log_omega = jnp.log10(omega)
    log_G_prime = jnp.log10(G_prime)
    log_omega_new = jnp.log10(omega_new)

    @jax.jit
    def interpolate() -> Array:
        log_result = interp1d(log_omega_new, log_omega, log_G_prime, method="cubic")
        return 10**log_result

    config = BenchmarkConfig(n_iterations=30, warmup_iterations=10)

    def interp_operation() -> None:
        result = interpolate()
        _ = result.block_until_ready()

    benchmark_result = benchmark_function(interp_operation, config)

    print(f"\n{benchmark_result}")
    print(f"  Known points: 50 (log-spaced)")
    print(f"  Query points: 200 (log-spaced)")
    print(f"  Method: cubic in log-space")

    # SLA: Log-scale interpolation should be fast
    assert benchmark_result.mean_time < 0.008, (
        f"Log-scale interpolation too slow: {benchmark_result.mean_time*1000:.3f}ms > 8ms"
    )


# =============================================================================
# Batch Interpolation
# =============================================================================

@pytest.mark.slow
def test_benchmark_batch_interpolation() -> None:
    """Benchmark batch interpolation across multiple datasets.

    Expected: < 25ms for 10 datasets (after JIT warmup)
    Tests vmap efficiency for interpolation.
    """
    from interpax import interp1d

    # Create batch of datasets
    batch_size = 10
    n_known = 50
    n_query = 200

    # Single x for all (same grid)
    x = jnp.linspace(0, 10, n_known)
    x_new = jnp.linspace(0, 10, n_query)

    # Different y values for each dataset
    batch_y = jnp.array([jnp.sin(x + i * 0.5) for i in range(batch_size)])

    # Create vmapped interpolator
    @jax.jit
    def batch_interpolate(y_batch: Array) -> Array:
        return jax.vmap(
            lambda y: interp1d(x_new, x, y, method="linear"),
            in_axes=0,
        )(y_batch)

    config = BenchmarkConfig(n_iterations=20, warmup_iterations=10)

    def interp_operation() -> None:
        result = batch_interpolate(batch_y)
        _ = result.block_until_ready()

    benchmark_result = benchmark_function(interp_operation, config)

    print(f"\n{benchmark_result}")
    print(f"  Batch size: {batch_size}")
    print(f"  Known points: {n_known}")
    print(f"  Query points: {n_query}")

    # SLA: Batch interpolation should be efficient
    assert benchmark_result.mean_time < 0.025, (
        f"Batch interpolation too slow: {benchmark_result.mean_time*1000:.3f}ms > 25ms"
    )


# =============================================================================
# Method Comparison
# =============================================================================

@pytest.mark.slow
def test_benchmark_interp_method_comparison() -> None:
    """Compare performance of different interpolation methods.

    Compares linear vs cubic to document performance trade-offs.
    """
    from interpax import interp1d

    x = jnp.linspace(0, 10, 100)
    y = jnp.sin(x)
    x_new = jnp.linspace(0, 10, 500)

    config = BenchmarkConfig(n_iterations=30, warmup_iterations=10)

    # Linear
    @jax.jit
    def linear_interp() -> Array:
        return interp1d(x_new, x, y, method="linear")

    def linear_op() -> None:
        result = linear_interp()
        _ = result.block_until_ready()

    linear_result = benchmark_function(linear_op, config)

    # Cubic
    @jax.jit
    def cubic_interp() -> Array:
        return interp1d(x_new, x, y, method="cubic")

    def cubic_op() -> None:
        result = cubic_interp()
        _ = result.block_until_ready()

    cubic_result = benchmark_function(cubic_op, config)

    ratio = cubic_result.mean_time / linear_result.mean_time

    print(f"\nInterpolation Method Comparison:")
    print(f"  Linear:     {linear_result.mean_time*1000:.3f}ms")
    print(f"  Cubic:      {cubic_result.mean_time*1000:.3f}ms")
    print(f"  Ratio:      {ratio:.2f}x")
    print(f"  Points:     100 known, 500 query")

    # Document the trade-off (cubic should be < 3x linear)
    assert ratio < 3.0, (
        f"Cubic overhead too high: {ratio:.2f}x > 3.0x linear"
    )


# =============================================================================
# Real-world Use Case
# =============================================================================

@pytest.mark.slow
def test_benchmark_rheology_data_interpolation() -> None:
    """Benchmark typical rheology data interpolation workflow.

    Expected: < 15ms (after JIT warmup)
    Simulates real RepTate usage: upsampling sparse frequency data.
    """
    from interpax import interp1d

    # Typical experimental data (sparse)
    omega_exp = jnp.array([
        0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0,
        20.0, 50.0, 100.0, 200.0, 500.0,
    ])

    # Simulate Maxwell response
    G0 = 1e5
    tau = 1.0
    omega_tau = omega_exp * tau
    G_prime_exp = G0 * omega_tau**2 / (1 + omega_tau**2)
    G_double_prime_exp = G0 * omega_tau / (1 + omega_tau**2)

    # Dense grid for plotting/analysis
    omega_dense = jnp.logspace(-2, 2.7, 200)

    @jax.jit
    def interpolate_rheology() -> tuple[Array, Array]:
        # Interpolate in log-space
        log_omega_exp = jnp.log10(omega_exp)
        log_omega_dense = jnp.log10(omega_dense)

        log_G_prime = jnp.log10(G_prime_exp)
        log_G_double_prime = jnp.log10(G_double_prime_exp)

        G_prime_interp = 10 ** interp1d(
            log_omega_dense, log_omega_exp, log_G_prime, method="cubic"
        )
        G_double_prime_interp = 10 ** interp1d(
            log_omega_dense, log_omega_exp, log_G_double_prime, method="cubic"
        )

        return G_prime_interp, G_double_prime_interp

    config = BenchmarkConfig(n_iterations=20, warmup_iterations=10)

    def interp_operation() -> None:
        G_p, G_pp = interpolate_rheology()
        _ = G_p.block_until_ready()
        _ = G_pp.block_until_ready()

    benchmark_result = benchmark_function(interp_operation, config)

    print(f"\n{benchmark_result}")
    print(f"  Experimental points: {len(omega_exp)}")
    print(f"  Interpolated points: {len(omega_dense)}")
    print(f"  Quantities: G', G''")
    print(f"  Use case: Upsampling rheology data")

    # SLA: Real-world rheology interpolation should be fast
    assert benchmark_result.mean_time < 0.015, (
        f"Rheology interpolation too slow: {benchmark_result.mean_time*1000:.3f}ms > 15ms"
    )


def test_benchmark_interpolation_summary(capsys: pytest.CaptureFixture[str]) -> None:
    """Print summary of interpolation performance baselines."""
    print("\n" + "=" * 70)
    print("INTERPOLATION PERFORMANCE BASELINES (interpax)")
    print("=" * 70)
    print("\nOperation                          | Target   | Notes")
    print("-" * 70)
    print("Linear (50 → 200 pts)              | < 5ms    | Baseline")
    print("Cubic (50 → 200 pts)               | < 10ms   | Higher order")
    print("Linear (500 → 2k pts)              | < 10ms   | Medium scale")
    print("Cubic (500 → 2k pts)               | < 20ms   | Medium cubic")
    print("Linear (5k → 10k pts)              | < 20ms   | Large scale")
    print("Log-scale cubic (50 → 200)         | < 8ms    | Rheology typical")
    print("Batch (10 datasets)                | < 25ms   | vmap efficiency")
    print("Rheology workflow (15 → 200)       | < 15ms   | Real-world use")
    print("Cubic/Linear ratio                 | < 3.0x   | Performance trade-off")
    print("=" * 70)
    print("\nAll times measured after JIT warmup")
    print("Hardware: CPU (x64 precision)")
    print("Log-scale: Common for rheology frequency data")
    print("=" * 70 + "\n")
