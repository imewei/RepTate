"""Centralized ctypes library loader with logging."""
from __future__ import annotations

import ctypes
import logging

logger = logging.getLogger(__name__)


def load_ctypes_library(lib_path: str, description: str) -> ctypes.CDLL:
    try:
        return ctypes.CDLL(lib_path)
    except OSError as exc:
        logger.error("Failed to load %s from %s: %s", description, lib_path, exc)
        raise ImportError(f"Failed to load {description} from {lib_path}") from exc
