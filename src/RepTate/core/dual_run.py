"""Dual-run pattern for comparing old and new implementations.

This module provides infrastructure for running both legacy and new implementations
in parallel, comparing results, and alerting on divergence. This is essential for
validating that new implementations produce equivalent results.

Usage:
    from RepTate.core.dual_run import DualRunner, ComparisonStrategy

    runner = DualRunner(
        name='jax_optimization',
        comparison_strategy=ComparisonStrategy.NUMERICAL_CLOSE
    )

    result = runner.run(
        new_impl=jax_optimize,
        legacy_impl=scipy_optimize,
        args=(initial_params,),
        kwargs={'tolerance': 1e-6}
    )
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, TypeVar

import numpy as np

logger = logging.getLogger(__name__)


class ComparisonStrategy(Enum):
    """Strategies for comparing results from old and new implementations.

    EXACT: Results must be exactly equal (using ==)
    NUMERICAL_CLOSE: Numerical values must be close (using allclose)
    STRUCTURAL: Results must have same structure (shape, type)
    CUSTOM: Use custom comparison function
    """
    EXACT = "exact"
    NUMERICAL_CLOSE = "numerical_close"
    STRUCTURAL = "structural"
    CUSTOM = "custom"


@dataclass
class ComparisonResult:
    """Result of comparing two implementations.

    Attributes:
        match: Whether the results match according to comparison strategy
        new_result: Result from new implementation
        legacy_result: Result from legacy implementation
        execution_time_new: Time taken by new implementation (seconds)
        execution_time_legacy: Time taken by legacy implementation (seconds)
        difference: Computed difference between results (if applicable)
        error_message: Error message if comparison failed
        timestamp: When the comparison was performed
    """
    match: bool
    new_result: Any
    legacy_result: Any
    execution_time_new: float
    execution_time_legacy: float
    difference: Any = None
    error_message: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DualRunMetrics:
    """Metrics for dual-run executions.

    Attributes:
        total_runs: Total number of dual runs
        matches: Number of times results matched
        mismatches: Number of times results diverged
        new_faster: Number of times new implementation was faster
        legacy_faster: Number of times legacy was faster
        new_errors: Number of errors in new implementation
        legacy_errors: Number of errors in legacy implementation
        avg_speedup: Average speedup factor (new vs legacy)
        mismatch_details: Recent mismatch details
    """
    total_runs: int = 0
    matches: int = 0
    mismatches: int = 0
    new_faster: int = 0
    legacy_faster: int = 0
    new_errors: int = 0
    legacy_errors: int = 0
    avg_speedup: float = 1.0
    mismatch_details: list[dict[str, Any]] = field(default_factory=list)


T = TypeVar('T')


class DualRunner:
    """Runs both old and new implementations in parallel for validation.

    This class executes both implementations, compares their results, and
    collects metrics about performance and correctness.
    """

    def __init__(
        self,
        name: str,
        comparison_strategy: ComparisonStrategy = ComparisonStrategy.NUMERICAL_CLOSE,
        custom_comparator: Callable[[Any, Any], bool] | None = None,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        alert_on_mismatch: bool = True,
        max_mismatch_history: int = 100,
    ):
        """Initialize the dual runner.

        Args:
            name: Unique identifier for this dual runner
            comparison_strategy: Strategy for comparing results
            custom_comparator: Custom comparison function (for CUSTOM strategy)
            rtol: Relative tolerance for numerical comparison
            atol: Absolute tolerance for numerical comparison
            alert_on_mismatch: Whether to log warnings on mismatch
            max_mismatch_history: Maximum number of mismatches to store
        """
        self.name = name
        self.comparison_strategy = comparison_strategy
        self.custom_comparator = custom_comparator
        self.rtol = rtol
        self.atol = atol
        self.alert_on_mismatch = alert_on_mismatch
        self.max_mismatch_history = max_mismatch_history

        self._lock = threading.RLock()
        self._metrics = DualRunMetrics()
        self._speedup_samples: list[float] = []

    def run(
        self,
        new_impl: Callable[..., T],
        legacy_impl: Callable[..., T],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> T:
        """Run both implementations and compare results.

        Args:
            new_impl: New implementation to test
            legacy_impl: Legacy implementation (ground truth)
            args: Positional arguments to pass to both implementations
            kwargs: Keyword arguments to pass to both implementations

        Returns:
            Result from new implementation (if it matches legacy)

        Raises:
            RuntimeError: If results don't match and alert_on_mismatch is True
        """
        kwargs = kwargs or {}

        # Run new implementation
        new_result = None
        new_error = None
        start_time = time.perf_counter()
        try:
            new_result = new_impl(*args, **kwargs)
        except Exception as e:
            new_error = e
            logger.error(
                f"DualRunner '{self.name}' - new implementation failed: "
                f"{type(e).__name__}: {e}"
            )
        new_time = time.perf_counter() - start_time

        # Run legacy implementation
        legacy_result = None
        legacy_error = None
        start_time = time.perf_counter()
        try:
            legacy_result = legacy_impl(*args, **kwargs)
        except Exception as e:
            legacy_error = e
            logger.error(
                f"DualRunner '{self.name}' - legacy implementation failed: "
                f"{type(e).__name__}: {e}"
            )
        legacy_time = time.perf_counter() - start_time

        # Update metrics
        with self._lock:
            self._metrics.total_runs += 1
            if new_error:
                self._metrics.new_errors += 1
            if legacy_error:
                self._metrics.legacy_errors += 1

        # Handle errors
        if new_error and legacy_error:
            raise RuntimeError(
                f"DualRunner '{self.name}' - both implementations failed"
            )
        if legacy_error:
            logger.warning(
                f"DualRunner '{self.name}' - legacy failed, using new result"
            )
            return new_result
        if new_error:
            logger.warning(
                f"DualRunner '{self.name}' - new failed, using legacy result"
            )
            return legacy_result

        # Compare results
        comparison = self._compare_results(
            new_result, legacy_result, new_time, legacy_time
        )

        # Update speedup metrics
        if legacy_time > 0:
            speedup = legacy_time / new_time
            with self._lock:
                self._speedup_samples.append(speedup)
                if len(self._speedup_samples) > 1000:
                    self._speedup_samples.pop(0)
                self._metrics.avg_speedup = sum(self._speedup_samples) / len(
                    self._speedup_samples
                )

                if new_time < legacy_time:
                    self._metrics.new_faster += 1
                else:
                    self._metrics.legacy_faster += 1

        # Handle comparison result
        if comparison.match:
            with self._lock:
                self._metrics.matches += 1
            logger.debug(
                f"DualRunner '{self.name}' - results match "
                f"(new: {new_time:.4f}s, legacy: {legacy_time:.4f}s, "
                f"speedup: {speedup:.2f}x)"
            )
            return new_result

        # Results don't match
        with self._lock:
            self._metrics.mismatches += 1
            self._metrics.mismatch_details.append({
                'timestamp': comparison.timestamp.isoformat(),
                'difference': str(comparison.difference),
                'error_message': comparison.error_message,
                'new_time': new_time,
                'legacy_time': legacy_time,
            })
            # Keep only recent mismatches
            if len(self._metrics.mismatch_details) > self.max_mismatch_history:
                self._metrics.mismatch_details.pop(0)

        error_msg = (
            f"DualRunner '{self.name}' - results diverged: "
            f"{comparison.error_message or 'Unknown difference'}"
        )

        if self.alert_on_mismatch:
            logger.error(error_msg)
            logger.error(f"New result: {new_result}")
            logger.error(f"Legacy result: {legacy_result}")
            logger.error(f"Difference: {comparison.difference}")

        # Return new result but warn about divergence
        return new_result

    def _compare_results(
        self,
        new_result: Any,
        legacy_result: Any,
        new_time: float,
        legacy_time: float,
    ) -> ComparisonResult:
        """Compare results from both implementations.

        Args:
            new_result: Result from new implementation
            legacy_result: Result from legacy implementation
            new_time: Execution time for new implementation
            legacy_time: Execution time for legacy implementation

        Returns:
            ComparisonResult with match status and details
        """
        try:
            if self.comparison_strategy == ComparisonStrategy.EXACT:
                match = self._exact_match(new_result, legacy_result)
                diff = None if match else (new_result, legacy_result)

            elif self.comparison_strategy == ComparisonStrategy.NUMERICAL_CLOSE:
                match, diff = self._numerical_close(new_result, legacy_result)

            elif self.comparison_strategy == ComparisonStrategy.STRUCTURAL:
                match, diff = self._structural_match(new_result, legacy_result)

            elif self.comparison_strategy == ComparisonStrategy.CUSTOM:
                if self.custom_comparator is None:
                    raise ValueError(
                        "CUSTOM strategy requires custom_comparator function"
                    )
                match = self.custom_comparator(new_result, legacy_result)
                diff = None if match else (new_result, legacy_result)

            else:
                raise ValueError(f"Unknown comparison strategy: {self.comparison_strategy}")

            return ComparisonResult(
                match=match,
                new_result=new_result,
                legacy_result=legacy_result,
                execution_time_new=new_time,
                execution_time_legacy=legacy_time,
                difference=diff,
            )

        except Exception as e:
            logger.error(
                f"DualRunner '{self.name}' - comparison failed: "
                f"{type(e).__name__}: {e}"
            )
            return ComparisonResult(
                match=False,
                new_result=new_result,
                legacy_result=legacy_result,
                execution_time_new=new_time,
                execution_time_legacy=legacy_time,
                error_message=str(e),
            )

    def _exact_match(self, new_result: Any, legacy_result: Any) -> bool:
        """Check if results are exactly equal.

        Args:
            new_result: Result from new implementation
            legacy_result: Result from legacy implementation

        Returns:
            True if results are exactly equal
        """
        return new_result == legacy_result

    def _numerical_close(
        self,
        new_result: Any,
        legacy_result: Any
    ) -> tuple[bool, Any]:
        """Check if numerical results are close within tolerance.

        Args:
            new_result: Result from new implementation
            legacy_result: Result from legacy implementation

        Returns:
            Tuple of (match, difference)
        """
        # Convert to numpy arrays for comparison
        try:
            # Handle JAX arrays
            if hasattr(new_result, '__array__'):
                new_array = np.asarray(new_result)
            else:
                new_array = np.atleast_1d(new_result)

            if hasattr(legacy_result, '__array__'):
                legacy_array = np.asarray(legacy_result)
            else:
                legacy_array = np.atleast_1d(legacy_result)

            # Check shapes match
            if new_array.shape != legacy_array.shape:
                return False, f"Shape mismatch: {new_array.shape} vs {legacy_array.shape}"

            # Use allclose for comparison
            match = np.allclose(
                new_array, legacy_array, rtol=self.rtol, atol=self.atol
            )

            if not match:
                diff = np.abs(new_array - legacy_array)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                return False, {
                    'max_absolute_diff': float(max_diff),
                    'mean_absolute_diff': float(mean_diff),
                    'max_relative_diff': float(max_diff / (np.abs(legacy_array).max() + 1e-10)),
                }

            return True, None

        except Exception as e:
            return False, f"Comparison error: {type(e).__name__}: {e}"

    def _structural_match(
        self,
        new_result: Any,
        legacy_result: Any
    ) -> tuple[bool, Any]:
        """Check if results have the same structure.

        Args:
            new_result: Result from new implementation
            legacy_result: Result from legacy implementation

        Returns:
            Tuple of (match, difference description)
        """
        # Check types
        if type(new_result) is not type(legacy_result):
            return False, f"Type mismatch: {type(new_result)} vs {type(legacy_result)}"

        # Check shapes for array-like objects
        if hasattr(new_result, 'shape') and hasattr(legacy_result, 'shape'):
            if new_result.shape != legacy_result.shape:
                return False, f"Shape mismatch: {new_result.shape} vs {legacy_result.shape}"

        # Check dict structure
        if isinstance(new_result, dict):
            if set(new_result.keys()) != set(legacy_result.keys()):
                return False, f"Dict keys mismatch: {new_result.keys()} vs {legacy_result.keys()}"

        # Check list/tuple length
        if isinstance(new_result, (list, tuple)):
            if len(new_result) != len(legacy_result):
                return False, f"Length mismatch: {len(new_result)} vs {len(legacy_result)}"

        return True, None

    def get_metrics(self) -> dict[str, Any]:
        """Get metrics for this dual runner.

        Returns:
            Dictionary containing current metrics
        """
        with self._lock:
            total = self._metrics.total_runs
            if total == 0:
                match_rate = 0.0
            else:
                match_rate = self._metrics.matches / total

            return {
                'name': self.name,
                'total_runs': self._metrics.total_runs,
                'matches': self._metrics.matches,
                'mismatches': self._metrics.mismatches,
                'match_rate': match_rate,
                'new_faster': self._metrics.new_faster,
                'legacy_faster': self._metrics.legacy_faster,
                'new_errors': self._metrics.new_errors,
                'legacy_errors': self._metrics.legacy_errors,
                'avg_speedup': self._metrics.avg_speedup,
                'recent_mismatches': self._metrics.mismatch_details[-10:],
            }


class DualRunRegistry:
    """Global registry for dual runners."""

    _runners: dict[str, DualRunner] = {}
    _lock = threading.Lock()

    @classmethod
    def get_runner(
        cls,
        name: str,
        comparison_strategy: ComparisonStrategy = ComparisonStrategy.NUMERICAL_CLOSE,
        **kwargs: Any
    ) -> DualRunner:
        """Get or create a dual runner.

        Args:
            name: Unique name for the dual runner
            comparison_strategy: Strategy for comparing results
            **kwargs: Additional arguments for DualRunner

        Returns:
            DualRunner instance
        """
        with cls._lock:
            if name not in cls._runners:
                cls._runners[name] = DualRunner(
                    name, comparison_strategy, **kwargs
                )
            return cls._runners[name]

    @classmethod
    def get_all_metrics(cls) -> dict[str, Any]:
        """Get metrics for all registered dual runners.

        Returns:
            Dictionary mapping runner names to their metrics
        """
        with cls._lock:
            return {
                name: runner.get_metrics()
                for name, runner in cls._runners.items()
            }
