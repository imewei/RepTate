"""
Shared type aliases for the performance module.

Feature: 004-gui-performance-optimizations
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# Type alias for a callable that creates a matplotlib Figure
FigureFactory = Callable[[], "Figure"]

# Type alias for progress update callbacks
# Args: (current_value: int, message: str)
ProgressCallback = Callable[[int, str], None]

# Type alias for error callbacks
# Args: (exception: Exception)
ErrorCallback = Callable[[Exception], None]

__all__ = [
    "FigureFactory",
    "ProgressCallback",
    "ErrorCallback",
]
