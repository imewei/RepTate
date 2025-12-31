"""Centralized ctypes library loader with logging."""
from __future__ import annotations

import ctypes
import logging

logger = logging.getLogger(__name__)


def load_ctypes_library(lib_path: str, description: str) -> ctypes.CDLL:
    """Load a C shared library via ctypes with error handling.

    Wraps ctypes.CDLL loading with logging and conversion of OSError to
    ImportError for consistent error handling across RepTate.

    Args:
        lib_path: Absolute or relative path to the shared library file
        description: Human-readable description of the library (for error messages)

    Returns:
        Loaded ctypes.CDLL instance

    Raises:
        ImportError: If the library cannot be loaded (wraps underlying OSError)
    """
    try:
        return ctypes.CDLL(lib_path)
    except OSError as exc:
        logger.error("Failed to load %s from %s: %s", description, lib_path, exc)
        raise ImportError(f"Failed to load {description} from {lib_path}") from exc
