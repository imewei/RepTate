"""Benchmark tests for performance validation.

Provides utilities for benchmarking functions with warmup and timing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs.

    Attributes:
        n_iterations: Number of timed iterations to run.
        warmup_iterations: Number of warmup iterations before timing.
    """

    n_iterations: int = 10
    warmup_iterations: int = 3


@dataclass
class BenchmarkResult:
    """Result from a benchmark run.

    Attributes:
        mean_time: Mean execution time in seconds.
        min_time: Minimum execution time in seconds.
        max_time: Maximum execution time in seconds.
        std_time: Standard deviation of execution times.
        n_iterations: Number of iterations run.
    """

    mean_time: float
    min_time: float
    max_time: float
    std_time: float
    n_iterations: int

    def __str__(self) -> str:
        """Format benchmark result for display."""
        return (
            f"Benchmark Result:\n"
            f"  Mean: {self.mean_time * 1000:.3f}ms\n"
            f"  Min:  {self.min_time * 1000:.3f}ms\n"
            f"  Max:  {self.max_time * 1000:.3f}ms\n"
            f"  Std:  {self.std_time * 1000:.3f}ms\n"
            f"  Iterations: {self.n_iterations}"
        )


def benchmark_function(
    fn: Callable[[], None],
    config: BenchmarkConfig | None = None,
) -> BenchmarkResult:
    """Benchmark a function with warmup iterations.

    Args:
        fn: Function to benchmark (no arguments, returns None).
        config: Benchmark configuration. Uses defaults if not provided.

    Returns:
        BenchmarkResult with timing statistics.
    """
    if config is None:
        config = BenchmarkConfig()

    # Warmup iterations
    for _ in range(config.warmup_iterations):
        fn()

    # Timed iterations
    times: list[float] = []
    for _ in range(config.n_iterations):
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    # Compute statistics
    mean_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    # Standard deviation
    variance = sum((t - mean_time) ** 2 for t in times) / len(times)
    std_time = variance**0.5

    return BenchmarkResult(
        mean_time=mean_time,
        min_time=min_time,
        max_time=max_time,
        std_time=std_time,
        n_iterations=config.n_iterations,
    )


__all__ = ["BenchmarkConfig", "BenchmarkResult", "benchmark_function"]
