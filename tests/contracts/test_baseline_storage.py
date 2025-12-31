"""Baseline storage and management for performance contracts.

This module provides utilities for:
- Recording performance baselines
- Comparing against baselines
- Detecting regressions
- Storing baselines persistently

Baselines are stored in JSON format for version control and review.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator


@dataclass
class BaseliningResult:
    """Result of a baselining operation.

    Attributes:
        operation: Name of the operation
        mean_time: Mean execution time
        std_dev: Standard deviation
        samples: Number of samples
        passed: Whether result meets acceptance criteria
        regression_percent: Percent regression if applicable
    """

    operation: str
    mean_time: float
    std_dev: float
    samples: int
    passed: bool
    regression_percent: float = 0.0


class PerformanceBaseline:
    """Manager for performance baselines."""

    def __init__(self, baseline_file: Path, threshold_percent: float = 10.0) -> None:
        """Initialize baseline manager.

        Args:
            baseline_file: Path to JSON file storing baselines
            threshold_percent: Acceptable regression threshold (%)
        """
        self.baseline_file = baseline_file
        self.threshold_percent = threshold_percent
        self._baselines: dict[str, dict[str, Any]] = {}
        self._load_baselines()

    def _load_baselines(self) -> None:
        """Load baselines from file if it exists."""
        if self.baseline_file.exists():
            with open(self.baseline_file, "r") as f:
                self._baselines = json.load(f)

    def _save_baselines(self) -> None:
        """Save baselines to file."""
        self.baseline_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.baseline_file, "w") as f:
            json.dump(self._baselines, f, indent=2)

    def record_baseline(
        self,
        operation: str,
        mean_time: float,
        std_dev: float,
        samples: int,
    ) -> None:
        """Record a performance baseline.

        Args:
            operation: Name of the operation
            mean_time: Mean execution time
            std_dev: Standard deviation
            samples: Number of samples
        """
        self._baselines[operation] = {
            "mean_time": mean_time,
            "std_dev": std_dev,
            "samples": samples,
            "threshold_percent": self.threshold_percent,
        }
        self._save_baselines()

    def check_regression(
        self,
        operation: str,
        measured_time: float,
    ) -> BaseliningResult:
        """Check if measured time violates baseline.

        Args:
            operation: Name of the operation
            measured_time: Measured execution time

        Returns:
            BaseliningResult with regression information
        """
        if operation not in self._baselines:
            # No baseline yet, this is first measurement
            return BaseliningResult(
                operation=operation,
                mean_time=measured_time,
                std_dev=0.0,
                samples=1,
                passed=True,
                regression_percent=0.0,
            )

        baseline = self._baselines[operation]
        baseline_mean = baseline["mean_time"]
        threshold = baseline_mean * (1 + self.threshold_percent / 100)

        regression_percent = ((measured_time - baseline_mean) / baseline_mean) * 100

        return BaseliningResult(
            operation=operation,
            mean_time=measured_time,
            std_dev=baseline["std_dev"],
            samples=baseline["samples"],
            passed=measured_time <= threshold,
            regression_percent=regression_percent,
        )

    def get_baseline(self, operation: str) -> dict[str, Any] | None:
        """Get baseline for an operation.

        Args:
            operation: Name of the operation

        Returns:
            Baseline dict or None if not found
        """
        return self._baselines.get(operation)

    def clear_baselines(self) -> None:
        """Clear all baselines (for testing)."""
        self._baselines = {}
        if self.baseline_file.exists():
            self.baseline_file.unlink()


class BaselineManager:
    """Manages multiple baselines for different test suites."""

    def __init__(self, baseline_dir: Path) -> None:
        """Initialize baseline manager.

        Args:
            baseline_dir: Directory for storing baseline files
        """
        self.baseline_dir = baseline_dir
        self._baselines: dict[str, PerformanceBaseline] = {}

    def get_baseline(
        self,
        suite: str,
        threshold_percent: float = 10.0,
    ) -> PerformanceBaseline:
        """Get or create a baseline for a test suite.

        Args:
            suite: Name of the test suite
            threshold_percent: Acceptable regression threshold

        Returns:
            PerformanceBaseline manager for the suite
        """
        if suite not in self._baselines:
            baseline_file = self.baseline_dir / f"{suite}_baselines.json"
            self._baselines[suite] = PerformanceBaseline(baseline_file, threshold_percent)

        return self._baselines[suite]


def measure_operation_time(
    func: Callable,
    *args: Any,
    **kwargs: Any,
) -> tuple[float, float, int]:
    """Measure operation execution time with statistics.

    Args:
        func: Function to measure
        *args: Positional arguments to func
        **kwargs: Keyword arguments to func

    Returns:
        Tuple of (mean_time, std_dev, samples)
    """
    # Quick measurement: 3 samples
    times = []
    for _ in range(3):
        start = time.perf_counter()
        func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    mean_time = sum(times) / len(times)
    variance = sum((t - mean_time) ** 2 for t in times) / len(times)
    std_dev = variance ** 0.5

    return mean_time, std_dev, len(times)


# =============================================================================
# Pytest Integration
# =============================================================================


@pytest.fixture
def baseline_manager(tmp_path) -> BaselineManager:
    """Provide a baseline manager for tests.

    Args:
        tmp_path: Temporary directory from pytest

    Yields:
        BaselineManager instance
    """
    return BaselineManager(tmp_path / "baselines")


@pytest.fixture
def nlsq_baseline(baseline_manager: BaselineManager) -> PerformanceBaseline:
    """Provide NLSQ baseline manager."""
    return baseline_manager.get_baseline("nlsq_fitting", threshold_percent=10.0)


@pytest.fixture
def theory_calculation_baseline(
    baseline_manager: BaselineManager,
) -> PerformanceBaseline:
    """Provide theory calculation baseline manager."""
    return baseline_manager.get_baseline("theory_calculation", threshold_percent=5.0)


@pytest.fixture
def data_loading_baseline(baseline_manager: BaselineManager) -> PerformanceBaseline:
    """Provide data loading baseline manager."""
    return baseline_manager.get_baseline("data_loading", threshold_percent=15.0)


# =============================================================================
# Baseline Documentation
# =============================================================================


def generate_baseline_report(baseline_manager: BaselineManager) -> str:
    """Generate a report of all recorded baselines.

    Args:
        baseline_manager: BaselineManager instance

    Returns:
        Formatted report string
    """
    report_lines = [
        "=== RepTate Performance Baselines ===\n",
        "Threshold: 10% for NLSQ, 5% for calculations, 15% for I/O\n",
    ]

    for suite_name, baseline in baseline_manager._baselines.items():
        report_lines.append(f"\n{suite_name}:")

        for operation, data in baseline._baselines.items():
            mean = data["mean_time"]
            std = data["std_dev"]
            samples = data["samples"]
            report_lines.append(
                f"  {operation}:"
                f" {mean*1000:.2f}ms ± {std*1000:.2f}ms ({samples} samples)"
            )

    return "".join(report_lines)


# =============================================================================
# Baseline Validation Helpers
# =============================================================================


class RegressionDetector:
    """Detects performance regressions against baselines."""

    def __init__(self, baseline_manager: BaselineManager) -> None:
        """Initialize detector.

        Args:
            baseline_manager: BaselineManager instance
        """
        self.baseline_manager = baseline_manager
        self.regressions: list[str] = []

    def check_and_record(
        self,
        suite: str,
        operation: str,
        func: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        """Check and record baseline for an operation.

        Args:
            suite: Test suite name
            operation: Operation name
            func: Function to measure
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            True if within threshold, False if regression detected
        """
        baseline = self.baseline_manager.get_baseline(suite)

        mean_time, std_dev, samples = measure_operation_time(func, *args, **kwargs)

        result = baseline.check_regression(operation, mean_time)

        if not result.passed:
            msg = (
                f"Regression detected in {operation}: "
                f"{result.regression_percent:.1f}% slower "
                f"({mean_time*1000:.2f}ms vs baseline {result.mean_time*1000:.2f}ms)"
            )
            self.regressions.append(msg)

        # Always record the new measurement
        baseline.record_baseline(operation, mean_time, std_dev, samples)

        return result.passed

    def get_regression_report(self) -> str:
        """Get report of detected regressions.

        Returns:
            Formatted regression report
        """
        if not self.regressions:
            return "No regressions detected."

        return "Regressions:\n" + "\n".join(f"  - {r}" for r in self.regressions)
