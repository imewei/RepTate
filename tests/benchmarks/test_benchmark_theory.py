"""Performance benchmarks for theory calculations.

Benchmarks rheology theory calculations to:
- Establish performance baselines for theory evaluation
- Identify hot paths for JIT optimization
- Validate vectorization and batching strategies
- Detect performance regressions

Test Coverage:
- Maxwell modes calculation (frequency domain)
- Maxwell modes calculation (time domain)
- Giesekus model
- Complex theory with many parameters
- Batch theory evaluation
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
# Theory Models
# =============================================================================

@jax.jit
def maxwell_modes_frequency(
    omega: Array,
    G_values: Array,
    tau_values: Array,
) -> tuple[Array, Array]:
    """Calculate Maxwell modes in frequency domain.

    Returns:
        (G_prime, G_double_prime) - Storage and loss moduli
    """
    # Vectorized calculation over all modes
    omega_tau = jnp.outer(omega, tau_values)  # Shape: (n_omega, n_modes)
    omega_tau_sq = omega_tau**2

    # Broadcasting for all modes
    G_prime = jnp.sum(
        G_values * omega_tau_sq / (1 + omega_tau_sq),
        axis=1,
    )
    G_double_prime = jnp.sum(
        G_values * omega_tau / (1 + omega_tau_sq),
        axis=1,
    )

    return G_prime, G_double_prime


@jax.jit
def maxwell_modes_time(
    t: Array,
    G_values: Array,
    tau_values: Array,
) -> Array:
    """Calculate Maxwell relaxation modulus in time domain.

    Returns:
        G(t) - Relaxation modulus
    """
    # Vectorized exponential relaxation
    exp_terms = jnp.exp(-jnp.outer(t, 1.0 / tau_values))  # Shape: (n_t, n_modes)
    G_t = jnp.sum(G_values * exp_terms, axis=1)
    return G_t


@jax.jit
def giesekus_model(
    gamma_dot: Array,
    G: float,
    tau: float,
    alpha: float,
) -> Array:
    """Giesekus constitutive model for shear stress.

    Non-linear viscoelastic model commonly used in polymer rheology.

    Args:
        gamma_dot: Shear rate array
        G: Modulus
        tau: Relaxation time
        alpha: Mobility parameter (0 < alpha < 1)

    Returns:
        Shear stress array
    """
    # Simplified steady-state solution
    # Full model would solve nonlinear system
    Wi = tau * gamma_dot  # Weissenberg number

    # Approximate solution for demonstration
    eta0 = G * tau
    numerator = eta0 * Wi
    denominator = 1 + alpha * Wi**2

    return numerator / denominator


@jax.jit
def multimode_giesekus(
    gamma_dot: Array,
    G_values: Array,
    tau_values: Array,
    alpha_values: Array,
) -> Array:
    """Multi-mode Giesekus model.

    Args:
        gamma_dot: Shear rate array
        G_values: Moduli for each mode
        tau_values: Relaxation times for each mode
        alpha_values: Mobility parameters for each mode

    Returns:
        Total shear stress
    """
    stress = jnp.zeros_like(gamma_dot)

    for G, tau, alpha in zip(G_values, tau_values, alpha_values):
        stress = stress + giesekus_model(gamma_dot, G, tau, alpha)

    return stress


# =============================================================================
# Benchmark Tests
# =============================================================================

@pytest.mark.slow
def test_benchmark_maxwell_frequency_single_mode() -> None:
    """Benchmark single Maxwell mode frequency calculation.

    Expected: < 5ms (after JIT warmup)
    This is the baseline for theory calculations.
    """
    omega = jnp.logspace(-2, 2, 100)
    G_values = jnp.array([1e5])
    tau_values = jnp.array([1.0])

    config = BenchmarkConfig(n_iterations=50, warmup_iterations=10)

    def theory_calculation() -> None:
        G_prime, G_double_prime = maxwell_modes_frequency(omega, G_values, tau_values)
        # Force evaluation
        _ = G_prime.block_until_ready()
        _ = G_double_prime.block_until_ready()

    benchmark_result = benchmark_function(theory_calculation, config)

    print(f"\n{benchmark_result}")
    print(f"  Data points: 100")
    print(f"  Modes: 1")

    # SLA: Single mode calculation should be very fast
    assert benchmark_result.mean_time < 0.005, (
        f"Single mode calculation too slow: {benchmark_result.mean_time*1000:.3f}ms > 5ms"
    )


@pytest.mark.slow
def test_benchmark_maxwell_frequency_multimode() -> None:
    """Benchmark multi-mode Maxwell frequency calculation.

    Expected: < 10ms for 20 modes (after JIT warmup)
    Tests vectorization efficiency.
    """
    omega = jnp.logspace(-3, 3, 100)

    # 20 Maxwell modes (typical for polymer relaxation spectrum)
    n_modes = 20
    G_values = jnp.logspace(3, 6, n_modes)
    tau_values = jnp.logspace(-2, 2, n_modes)

    config = BenchmarkConfig(n_iterations=30, warmup_iterations=10)

    def theory_calculation() -> None:
        G_prime, G_double_prime = maxwell_modes_frequency(omega, G_values, tau_values)
        _ = G_prime.block_until_ready()
        _ = G_double_prime.block_until_ready()

    benchmark_result = benchmark_function(theory_calculation, config)

    print(f"\n{benchmark_result}")
    print(f"  Data points: 100")
    print(f"  Modes: {n_modes}")

    # SLA: Multi-mode should scale well with vectorization
    assert benchmark_result.mean_time < 0.010, (
        f"Multi-mode calculation too slow: {benchmark_result.mean_time*1000:.3f}ms > 10ms"
    )


@pytest.mark.slow
def test_benchmark_maxwell_time_domain() -> None:
    """Benchmark Maxwell time domain calculation.

    Expected: < 8ms for 10 modes (after JIT warmup)
    Time domain requires exponential calculations.
    """
    t = jnp.logspace(-3, 2, 100)

    n_modes = 10
    G_values = jnp.logspace(3, 5, n_modes)
    tau_values = jnp.logspace(-2, 1, n_modes)

    config = BenchmarkConfig(n_iterations=30, warmup_iterations=10)

    def theory_calculation() -> None:
        G_t = maxwell_modes_time(t, G_values, tau_values)
        _ = G_t.block_until_ready()

    benchmark_result = benchmark_function(theory_calculation, config)

    print(f"\n{benchmark_result}")
    print(f"  Time points: 100")
    print(f"  Modes: {n_modes}")

    # SLA: Time domain exponentials should be efficient
    assert benchmark_result.mean_time < 0.008, (
        f"Time domain calculation too slow: {benchmark_result.mean_time*1000:.3f}ms > 8ms"
    )


@pytest.mark.slow
def test_benchmark_giesekus_single_mode() -> None:
    """Benchmark single-mode Giesekus model.

    Expected: < 5ms (after JIT warmup)
    Non-linear model baseline.
    """
    gamma_dot = jnp.logspace(-2, 2, 50)
    G = 1e5
    tau = 1.0
    alpha = 0.3

    config = BenchmarkConfig(n_iterations=30, warmup_iterations=10)

    def theory_calculation() -> None:
        stress = giesekus_model(gamma_dot, G, tau, alpha)
        _ = stress.block_until_ready()

    benchmark_result = benchmark_function(theory_calculation, config)

    print(f"\n{benchmark_result}")
    print(f"  Shear rate points: 50")
    print(f"  Modes: 1")

    # SLA: Giesekus single mode should be fast
    assert benchmark_result.mean_time < 0.005, (
        f"Giesekus single mode too slow: {benchmark_result.mean_time*1000:.3f}ms > 5ms"
    )


@pytest.mark.slow
def test_benchmark_giesekus_multimode() -> None:
    """Benchmark multi-mode Giesekus model.

    Expected: < 15ms for 5 modes (after JIT warmup)
    Complex non-linear model.
    """
    gamma_dot = jnp.logspace(-2, 2, 50)

    n_modes = 5
    G_values = jnp.array([1e5, 5e4, 2e4, 1e4, 5e3])
    tau_values = jnp.array([10.0, 1.0, 0.1, 0.01, 0.001])
    alpha_values = jnp.array([0.3, 0.3, 0.3, 0.3, 0.3])

    config = BenchmarkConfig(n_iterations=20, warmup_iterations=10)

    def theory_calculation() -> None:
        stress = multimode_giesekus(gamma_dot, G_values, tau_values, alpha_values)
        _ = stress.block_until_ready()

    benchmark_result = benchmark_function(theory_calculation, config)

    print(f"\n{benchmark_result}")
    print(f"  Shear rate points: 50")
    print(f"  Modes: {n_modes}")

    # SLA: Multi-mode Giesekus should remain efficient
    assert benchmark_result.mean_time < 0.015, (
        f"Giesekus multi-mode too slow: {benchmark_result.mean_time*1000:.3f}ms > 15ms"
    )


@pytest.mark.slow
def test_benchmark_large_spectrum() -> None:
    """Benchmark Maxwell calculation with very large relaxation spectrum.

    Expected: < 20ms for 100 modes (after JIT warmup)
    Tests extreme case for full relaxation spectrum.
    """
    omega = jnp.logspace(-3, 3, 200)

    # Very large spectrum (100 modes)
    n_modes = 100
    G_values = jnp.logspace(2, 6, n_modes)
    tau_values = jnp.logspace(-3, 3, n_modes)

    config = BenchmarkConfig(n_iterations=10, warmup_iterations=5)

    def theory_calculation() -> None:
        G_prime, G_double_prime = maxwell_modes_frequency(omega, G_values, tau_values)
        _ = G_prime.block_until_ready()
        _ = G_double_prime.block_until_ready()

    benchmark_result = benchmark_function(theory_calculation, config)

    print(f"\n{benchmark_result}")
    print(f"  Frequency points: 200")
    print(f"  Modes: {n_modes}")

    # SLA: Even large spectra should be fast with vectorization
    assert benchmark_result.mean_time < 0.020, (
        f"Large spectrum calculation too slow: {benchmark_result.mean_time*1000:.3f}ms > 20ms"
    )


# =============================================================================
# Vectorization Benefit Test
# =============================================================================

@pytest.mark.slow
def test_benchmark_vectorization_benefit() -> None:
    """Demonstrate vectorization speedup vs naive loop.

    Shows the benefit of JAX vectorized operations over Python loops.
    """
    omega = jnp.logspace(-2, 2, 100)
    n_modes = 20
    G_values = jnp.logspace(3, 6, n_modes)
    tau_values = jnp.logspace(-2, 2, n_modes)

    # Vectorized version (JIT compiled)
    config = BenchmarkConfig(n_iterations=20, warmup_iterations=10)

    def vectorized_calc() -> None:
        G_prime, _ = maxwell_modes_frequency(omega, G_values, tau_values)
        _ = G_prime.block_until_ready()

    vectorized_result = benchmark_function(vectorized_calc, config)

    # Naive loop version (for comparison, not JIT compiled)
    def naive_calc() -> None:
        G_prime = jnp.zeros_like(omega)
        for i in range(len(G_values)):
            omega_tau = omega * tau_values[i]
            G_prime = G_prime + G_values[i] * omega_tau**2 / (1 + omega_tau**2)
        _ = G_prime.block_until_ready()

    naive_result = benchmark_function(naive_calc, config)

    speedup = naive_result.mean_time / vectorized_result.mean_time

    print(f"\nVectorization Benefit:")
    print(f"  Naive loop:       {naive_result.mean_time*1000:.3f}ms")
    print(f"  Vectorized (JIT): {vectorized_result.mean_time*1000:.3f}ms")
    print(f"  Speedup factor:   {speedup:.2f}x")

    # Vectorization should provide significant speedup
    assert speedup > 1.5, (
        f"Vectorization benefit too small: {speedup:.2f}x < 1.5x expected"
    )


# =============================================================================
# Batch Processing Test
# =============================================================================

@pytest.mark.slow
def test_benchmark_batch_theory_evaluation() -> None:
    """Benchmark batch theory evaluation across multiple datasets.

    Expected: < 30ms for 10 datasets (after JIT warmup)
    Tests vmap efficiency for batch processing.
    """
    # Create batch of frequency arrays (10 datasets)
    batch_size = 10
    n_points = 50

    # Use vmap for batch processing
    batch_omega = jnp.array([jnp.logspace(-2, 2, n_points) for _ in range(batch_size)])

    G_values = jnp.array([1e5])
    tau_values = jnp.array([1.0])

    # Create vmapped version
    batch_maxwell = jax.vmap(
        lambda omega: maxwell_modes_frequency(omega, G_values, tau_values),
        in_axes=0,
    )

    config = BenchmarkConfig(n_iterations=20, warmup_iterations=10)

    def batch_calculation() -> None:
        results = batch_maxwell(batch_omega)
        # Force evaluation
        _ = results[0].block_until_ready()
        _ = results[1].block_until_ready()

    benchmark_result = benchmark_function(batch_calculation, config)

    print(f"\n{benchmark_result}")
    print(f"  Batch size: {batch_size}")
    print(f"  Points per dataset: {n_points}")

    # SLA: Batch processing should be efficient
    assert benchmark_result.mean_time < 0.030, (
        f"Batch calculation too slow: {benchmark_result.mean_time*1000:.3f}ms > 30ms"
    )


def test_benchmark_theory_summary(capsys: pytest.CaptureFixture[str]) -> None:
    """Print summary of expected theory calculation baselines."""
    print("\n" + "=" * 70)
    print("THEORY CALCULATION PERFORMANCE BASELINES (JAX)")
    print("=" * 70)
    print("\nOperation                          | Target   | Notes")
    print("-" * 70)
    print("Maxwell single mode (100 pts)      | < 5ms    | Baseline")
    print("Maxwell multi-mode (100 pts, 20)   | < 10ms   | Vectorized")
    print("Maxwell time domain (100 pts, 10)  | < 8ms    | Exponentials")
    print("Maxwell large spectrum (200, 100)  | < 20ms   | Extreme case")
    print("Giesekus single mode (50 pts)      | < 5ms    | Non-linear")
    print("Giesekus multi-mode (50 pts, 5)    | < 15ms   | Complex model")
    print("Batch evaluation (10 datasets)     | < 30ms   | vmap efficiency")
    print("Vectorization speedup              | > 1.5x   | vs naive loop")
    print("=" * 70)
    print("\nAll times measured after JIT warmup")
    print("Hardware: CPU (x64 precision)")
    print("=" * 70 + "\n")
