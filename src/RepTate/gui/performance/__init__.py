"""
Performance Optimization Utilities for RepTate GUI.

This module provides utilities for improving GUI responsiveness and performance:

- BlittingManager: Matplotlib blitting for fast interactive plot updates
- BatchUpdateContext: Flicker-free bulk widget updates
- LazyModuleLoader: Deferred module imports for faster startup
- CalculationThreadPool: Shared thread pool for parallel calculations
- ProgressReporter: Thread-safe progress reporting
- FigureCache: LRU cache for matplotlib figures

Feature: 004-gui-performance-optimizations
"""

from __future__ import annotations

# Lazy imports to avoid circular dependencies during module initialization.
# The actual imports are deferred until the classes/functions are accessed.

__all__ = [
    # Blitting
    "BlittingManager",
    "create_blitting_manager",
    # Batch Updates
    "BatchUpdateContext",
    "batch_updates",
    # Lazy Loading
    "LazyModuleLoader",
    "create_lazy_loader",
    # Thread Pool
    "CalculationThreadPool",
    "get_calculation_pool",
    # Progress
    "ProgressReporter",
    "create_progress_reporter",
    # Figure Cache
    "FigureCache",
    "get_figure_cache",
]


def __getattr__(name: str):
    """Lazy import of submodule classes and functions."""
    if name in ("BlittingManager", "create_blitting_manager"):
        from RepTate.gui.performance.blitting import (
            BlittingManager,
            create_blitting_manager,
        )

        return BlittingManager if name == "BlittingManager" else create_blitting_manager

    if name in ("BatchUpdateContext", "batch_updates"):
        from RepTate.gui.performance.batch_update import (
            BatchUpdateContext,
            batch_updates,
        )

        return BatchUpdateContext if name == "BatchUpdateContext" else batch_updates

    if name in ("LazyModuleLoader", "create_lazy_loader"):
        from RepTate.gui.performance.lazy_loader import (
            LazyModuleLoader,
            create_lazy_loader,
        )

        return LazyModuleLoader if name == "LazyModuleLoader" else create_lazy_loader

    if name in ("CalculationThreadPool", "get_calculation_pool"):
        from RepTate.gui.performance.thread_pool import (
            CalculationThreadPool,
            get_calculation_pool,
        )

        return (
            CalculationThreadPool
            if name == "CalculationThreadPool"
            else get_calculation_pool
        )

    if name in ("ProgressReporter", "create_progress_reporter"):
        from RepTate.gui.performance.progress import (
            ProgressReporter,
            create_progress_reporter,
        )

        return (
            ProgressReporter if name == "ProgressReporter" else create_progress_reporter
        )

    if name in ("FigureCache", "get_figure_cache"):
        from RepTate.gui.performance.figure_cache import FigureCache, get_figure_cache

        return FigureCache if name == "FigureCache" else get_figure_cache

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
