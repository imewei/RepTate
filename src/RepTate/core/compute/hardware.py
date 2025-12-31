"""Hardware detection helpers for CPU/GPU/TPU execution."""

from __future__ import annotations

import logging

import jax

LOG = logging.getLogger(__name__)


def get_available_devices() -> list[str]:
    """Return JAX device platforms present on this system.

    Returns:
        list[str]: Sorted list of platform names (e.g., ['cpu'], ['cpu', 'gpu'],
            ['cpu', 'tpu']) available in the current JAX installation and hardware
            environment.
    """
    platforms = {device.platform for device in jax.devices()}
    return sorted(platforms)


def get_compute_backend() -> str:
    """Return the preferred compute backend (gpu, tpu, or cpu).

    Returns:
        str: The preferred backend in priority order: 'gpu' if available, else 'tpu'
            if available, else 'cpu'. This determines the execution device for
            JAX-based numerical computations.
    """
    platforms = get_available_devices()
    if "gpu" in platforms:
        return "gpu"
    if "tpu" in platforms:
        return "tpu"
    return "cpu"


def warn_if_no_accelerator() -> None:
    """Warn when only CPU execution is available."""
    if get_compute_backend() == "cpu":
        LOG.warning(
            "Accelerator not detected. Running in CPU mode with reduced performance."
        )
