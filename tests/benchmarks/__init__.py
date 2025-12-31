"""Performance benchmarking suite for RepTate.

This module provides comprehensive performance benchmarks to:
- Establish baseline metrics for critical operations
- Detect performance regressions
- Guide JAX optimization strategies
- Validate improvements from SciPy to NLSQ/JAX migration

Benchmark Categories:
- Curve fitting (NLSQ vs theoretical SciPy baseline)
- Theory calculations (Maxwell modes, Giesekus, etc.)
- Data I/O and serialization
- Interpolation (interpax vs scipy.interpolate)
- Array operations and transformations

Usage:
    pytest tests/benchmarks/ --benchmark-only
    pytest tests/benchmarks/test_benchmark_fitting.py -v
"""

from __future__ import annotations

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "timer",
]

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for benchmark execution.

    Attributes:
        n_iterations: Number of iterations to run
        warmup_iterations: Number of warmup iterations (for JIT compilation)
        timeout_seconds: Maximum allowed execution time
        relative_tolerance: Acceptable performance degradation (e.g., 1.1 = 10% slower)
    """
    n_iterations: int = 10
    warmup_iterations: int = 3
    timeout_seconds: float = 60.0
    relative_tolerance: float = 1.2  # Allow 20% degradation


@dataclass
class BenchmarkResult:
    """Results from a benchmark run.

    Attributes:
        name: Benchmark name
        mean_time: Mean execution time in seconds
        std_time: Standard deviation of execution time
        min_time: Minimum execution time
        max_time: Maximum execution time
        iterations: Number of iterations performed
        metadata: Additional benchmark-specific metadata
    """
    name: str
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    iterations: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Format benchmark result for display."""
        return (
            f"{self.name}:\n"
            f"  Mean: {self.mean_time*1000:.3f} ms\n"
            f"  Std:  {self.std_time*1000:.3f} ms\n"
            f"  Min:  {self.min_time*1000:.3f} ms\n"
            f"  Max:  {self.max_time*1000:.3f} ms\n"
            f"  Iterations: {self.iterations}"
        )


class timer:
    """Context manager for timing code blocks.

    Example:
        with timer() as t:
            expensive_operation()
        print(f"Elapsed: {t.elapsed:.3f}s")
    """

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed = time.perf_counter() - self._start


def benchmark_function(
    func: Callable[[], Any],
    config: BenchmarkConfig | None = None,
) -> BenchmarkResult:
    """Benchmark a function with warmup and multiple iterations.

    Args:
        func: Function to benchmark (no arguments)
        config: Benchmark configuration

    Returns:
        BenchmarkResult with timing statistics
    """
    if config is None:
        config = BenchmarkConfig()

    # Warmup iterations (important for JIT compilation)
    for _ in range(config.warmup_iterations):
        func()

    # Actual benchmark iterations
    times: list[float] = []
    for _ in range(config.n_iterations):
        with timer() as t:
            func()
        times.append(t.elapsed)

        if sum(times) > config.timeout_seconds:
            break

    times_array = jnp.array(times)

    return BenchmarkResult(
        name=func.__name__,
        mean_time=float(jnp.mean(times_array)),
        std_time=float(jnp.std(times_array)),
        min_time=float(jnp.min(times_array)),
        max_time=float(jnp.max(times_array)),
        iterations=len(times),
    )
