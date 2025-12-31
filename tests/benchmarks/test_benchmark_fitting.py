"""Performance benchmarks for curve fitting operations.

Benchmarks the NLSQ-based fitting against theoretical baselines to:
- Establish performance expectations for various data sizes
- Detect performance regressions
- Guide optimization of fitting algorithms
- Validate JAX JIT compilation benefits

Test Coverage:
- Linear regression (simple case)
- Nonlinear Maxwell model fitting
- Multi-mode Maxwell fitting
- Large dataset fitting
- Parameter bounds handling
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
# Test Models
# =============================================================================

def linear_model(x: Array, params: Array) -> Array:
    """Linear model: y = a*x + b."""
    return params[0] * x + params[1]


def maxwell_single_mode(omega: Array, params: Array) -> Array:
    """Single Maxwell mode storage modulus.

    params[0] = G0, params[1] = tau
    """
    G0, tau = params[0], params[1]
    omega_tau = omega * tau
    return G0 * omega_tau**2 / (1 + omega_tau**2)


def maxwell_multi_mode(omega: Array, params: Array) -> Array:
    """Multi-mode Maxwell model (3 modes).

    params = [G1, tau1, G2, tau2, G3, tau3]
    """
    G_prime = jnp.zeros_like(omega)
    for i in range(0, len(params), 2):
        G0 = params[i]
        tau = params[i + 1]
        omega_tau = omega * tau
        G_prime = G_prime + G0 * omega_tau**2 / (1 + omega_tau**2)
    return G_prime


# =============================================================================
# Benchmark Tests
# =============================================================================

@pytest.mark.slow
def test_benchmark_linear_fit_small() -> None:
    """Benchmark linear fitting with small dataset (100 points).

    Expected: < 10ms (after JIT warmup)
    This is the baseline for simple operations.
    """
    from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

    # Generate clean linear data
    x = jnp.linspace(0, 10, 100)
    y_true = linear_model(x, jnp.array([2.5, 1.0]))
    p0 = jnp.array([1.0, 0.0])

    config = BenchmarkConfig(n_iterations=20, warmup_iterations=5)

    def fit_operation() -> None:
        result, _ = run_nlsq_fit(linear_model, x, y_true, p0=p0)
        # Force evaluation
        _ = result.parameters_array.block_until_ready()

    benchmark_result = benchmark_function(fit_operation, config)

    print(f"\n{benchmark_result}")

    # SLA: Linear fit should complete in < 10ms
    assert benchmark_result.mean_time < 0.010, (
        f"Linear fit too slow: {benchmark_result.mean_time*1000:.3f}ms > 10ms"
    )


@pytest.mark.slow
def test_benchmark_linear_fit_medium() -> None:
    """Benchmark linear fitting with medium dataset (1000 points).

    Expected: < 20ms (after JIT warmup)
    Tests scaling behavior with data size.
    """
    from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

    x = jnp.linspace(0, 10, 1000)
    y_true = linear_model(x, jnp.array([2.5, 1.0]))
    p0 = jnp.array([1.0, 0.0])

    config = BenchmarkConfig(n_iterations=15, warmup_iterations=5)

    def fit_operation() -> None:
        result, _ = run_nlsq_fit(linear_model, x, y_true, p0=p0)
        _ = result.parameters_array.block_until_ready()

    benchmark_result = benchmark_function(fit_operation, config)

    print(f"\n{benchmark_result}")

    # SLA: Should scale sub-linearly
    assert benchmark_result.mean_time < 0.020, (
        f"Medium linear fit too slow: {benchmark_result.mean_time*1000:.3f}ms > 20ms"
    )


@pytest.mark.slow
def test_benchmark_maxwell_single_mode() -> None:
    """Benchmark single-mode Maxwell model fitting.

    Expected: < 50ms (after JIT warmup)
    This represents typical rheology theory fitting.
    """
    from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

    # Generate synthetic Maxwell data
    omega = jnp.logspace(-2, 2, 50)
    params_true = jnp.array([1e5, 1.0])
    y_true = maxwell_single_mode(omega, params_true)

    # Add small noise for realism
    key = jax.random.PRNGKey(42)
    noise = jax.random.normal(key, shape=y_true.shape) * 0.01 * y_true
    y_noisy = y_true + noise

    p0 = jnp.array([5e4, 0.5])  # Initial guess

    config = BenchmarkConfig(n_iterations=10, warmup_iterations=5)

    def fit_operation() -> None:
        result, _ = run_nlsq_fit(
            maxwell_single_mode,
            omega,
            y_noisy,
            p0=p0,
            bounds=(jnp.array([1e3, 0.01]), jnp.array([1e7, 100.0])),
        )
        _ = result.parameters_array.block_until_ready()

    benchmark_result = benchmark_function(fit_operation, config)

    print(f"\n{benchmark_result}")

    # SLA: Maxwell single mode should complete in < 50ms
    assert benchmark_result.mean_time < 0.050, (
        f"Maxwell single mode fit too slow: {benchmark_result.mean_time*1000:.3f}ms > 50ms"
    )


@pytest.mark.slow
def test_benchmark_maxwell_multi_mode() -> None:
    """Benchmark multi-mode (3 modes) Maxwell model fitting.

    Expected: < 200ms (after JIT warmup)
    This represents complex multi-parameter fitting.
    """
    from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

    # Generate synthetic multi-mode Maxwell data
    omega = jnp.logspace(-3, 3, 100)
    params_true = jnp.array([1e5, 10.0, 5e4, 1.0, 2e4, 0.1])
    y_true = maxwell_multi_mode(omega, params_true)

    # Add noise
    key = jax.random.PRNGKey(42)
    noise = jax.random.normal(key, shape=y_true.shape) * 0.02 * y_true
    y_noisy = y_true + noise

    # Initial guess (perturbed from true values)
    p0 = jnp.array([8e4, 8.0, 6e4, 1.5, 3e4, 0.15])

    config = BenchmarkConfig(n_iterations=8, warmup_iterations=3)

    def fit_operation() -> None:
        result, _ = run_nlsq_fit(
            maxwell_multi_mode,
            omega,
            y_noisy,
            p0=p0,
        )
        _ = result.parameters_array.block_until_ready()

    benchmark_result = benchmark_function(fit_operation, config)

    print(f"\n{benchmark_result}")
    print(f"  Parameters: 6")
    print(f"  Data points: 100")

    # SLA: Multi-mode fit should complete in < 200ms
    assert benchmark_result.mean_time < 0.200, (
        f"Maxwell multi-mode fit too slow: {benchmark_result.mean_time*1000:.3f}ms > 200ms"
    )


@pytest.mark.slow
def test_benchmark_large_dataset() -> None:
    """Benchmark fitting with large dataset (10k points).

    Expected: < 100ms (after JIT warmup)
    Tests performance at scale.
    """
    from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

    # Large linear dataset
    x = jnp.linspace(0, 100, 10_000)
    y_true = linear_model(x, jnp.array([2.5, 1.0]))

    # Add noise
    key = jax.random.PRNGKey(42)
    noise = jax.random.normal(key, shape=y_true.shape) * 0.1
    y_noisy = y_true + noise

    p0 = jnp.array([1.0, 0.0])

    config = BenchmarkConfig(n_iterations=5, warmup_iterations=2)

    def fit_operation() -> None:
        result, _ = run_nlsq_fit(linear_model, x, y_noisy, p0=p0)
        _ = result.parameters_array.block_until_ready()

    benchmark_result = benchmark_function(fit_operation, config)

    print(f"\n{benchmark_result}")
    print(f"  Data points: 10,000")

    # SLA: Large dataset should still be fast (JAX optimization)
    assert benchmark_result.mean_time < 0.100, (
        f"Large dataset fit too slow: {benchmark_result.mean_time*1000:.3f}ms > 100ms"
    )


@pytest.mark.slow
def test_benchmark_fit_with_bounds() -> None:
    """Benchmark fitting with parameter bounds.

    Expected: < 60ms (after JIT warmup)
    Bounds add overhead but should remain efficient.
    """
    from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

    omega = jnp.logspace(-2, 2, 50)
    params_true = jnp.array([1e5, 1.0])
    y_true = maxwell_single_mode(omega, params_true)
    p0 = jnp.array([5e4, 0.5])

    # Define bounds
    lower_bounds = jnp.array([1e3, 0.01])
    upper_bounds = jnp.array([1e7, 100.0])

    config = BenchmarkConfig(n_iterations=10, warmup_iterations=5)

    def fit_operation() -> None:
        result, _ = run_nlsq_fit(
            maxwell_single_mode,
            omega,
            y_true,
            p0=p0,
            bounds=(lower_bounds, upper_bounds),
        )
        _ = result.parameters_array.block_until_ready()

    benchmark_result = benchmark_function(fit_operation, config)

    print(f"\n{benchmark_result}")

    # SLA: Bounded fit overhead should be minimal
    assert benchmark_result.mean_time < 0.060, (
        f"Bounded fit too slow: {benchmark_result.mean_time*1000:.3f}ms > 60ms"
    )


# =============================================================================
# JIT Compilation Benefit Test
# =============================================================================

@pytest.mark.slow
def test_benchmark_jit_speedup() -> None:
    """Measure JIT compilation speedup for fitting.

    This test demonstrates the performance benefit of JAX's JIT compilation
    by comparing first call (with compilation) vs subsequent calls.
    """
    from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

    omega = jnp.logspace(-2, 2, 50)
    params_true = jnp.array([1e5, 1.0])
    y_true = maxwell_single_mode(omega, params_true)
    p0 = jnp.array([5e4, 0.5])

    # Measure first call (includes JIT compilation)
    import time
    start_first = time.perf_counter()
    result1, _ = run_nlsq_fit(maxwell_single_mode, omega, y_true, p0=p0)
    _ = result1.parameters_array.block_until_ready()
    first_call_time = time.perf_counter() - start_first

    # Measure subsequent calls (JIT cached)
    times_subsequent = []
    for _ in range(10):
        start = time.perf_counter()
        result, _ = run_nlsq_fit(maxwell_single_mode, omega, y_true, p0=p0)
        _ = result.parameters_array.block_until_ready()
        times_subsequent.append(time.perf_counter() - start)

    mean_subsequent = jnp.mean(jnp.array(times_subsequent))
    speedup = first_call_time / float(mean_subsequent)

    print(f"\nJIT Compilation Benefit:")
    print(f"  First call (with compilation): {first_call_time*1000:.3f}ms")
    print(f"  Subsequent calls (cached):      {float(mean_subsequent)*1000:.3f}ms")
    print(f"  Speedup factor:                 {speedup:.2f}x")

    # JIT should provide at least 2x speedup
    assert speedup > 2.0, (
        f"JIT speedup insufficient: {speedup:.2f}x < 2.0x expected"
    )


# =============================================================================
# Comparative Baseline (for documentation)
# =============================================================================

def test_benchmark_baseline_summary(capsys: pytest.CaptureFixture[str]) -> None:
    """Print summary of expected performance baselines.

    This test documents the performance SLAs for curve fitting operations.
    """
    print("\n" + "=" * 70)
    print("CURVE FITTING PERFORMANCE BASELINES (JAX + NLSQ)")
    print("=" * 70)
    print("\nOperation                     | Target   | Notes")
    print("-" * 70)
    print("Linear fit (100 pts)          | < 10ms   | Simple baseline")
    print("Linear fit (1000 pts)         | < 20ms   | Scaling test")
    print("Linear fit (10k pts)          | < 100ms  | Large dataset")
    print("Maxwell single mode (50 pts)  | < 50ms   | Typical rheology fit")
    print("Maxwell multi-mode (100 pts)  | < 200ms  | Complex 6-parameter fit")
    print("Bounded fit                   | < 60ms   | Parameter constraints")
    print("JIT speedup factor            | > 2.0x   | Compilation benefit")
    print("=" * 70)
    print("\nAll times measured after JIT warmup (excludes compilation)")
    print("Hardware: CPU (x64 precision)")
    print("=" * 70 + "\n")
