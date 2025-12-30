"""Hardware detection helpers for CPU/GPU/TPU execution."""

from __future__ import annotations

import logging

import jax

LOG = logging.getLogger(__name__)


def get_available_devices() -> list[str]:
    """Return JAX device platforms present on this system."""
    platforms = {device.platform for device in jax.devices()}
    return sorted(platforms)


def get_compute_backend() -> str:
    """Return the preferred compute backend (gpu, tpu, or cpu)."""
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
