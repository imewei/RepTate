"""Generate golden files for numerical regression tests.

Run this ONCE before any major numerical migration to establish a baseline.
These files should only be regenerated after careful review of numerical
changes.

Usage:
    python -m tests.regression.generate_golden
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

GOLDEN_DIR = Path(__file__).parent / "golden"


def generate_maxwell_golden() -> None:
    """Generate golden file for Maxwell model."""
    G0 = 1e5
    tau = 1.0
    omega = np.logspace(-2, 2, 100)

    omega_tau = omega * tau
    G_prime = G0 * omega_tau**2 / (1 + omega_tau**2)
    G_double_prime = G0 * omega_tau / (1 + omega_tau**2)

    np.savez(
        GOLDEN_DIR / "maxwell_model.npz",
        omega=omega,
        G_prime=G_prime,
        G_double_prime=G_double_prime,
        G0=G0,
        tau=tau,
    )
    print("Generated: maxwell_model.npz")


def generate_exponential_golden() -> None:
    """Generate golden file for exponential relaxation."""
    A = 1000.0
    tau = 2.0
    t = np.linspace(0, 20, 100)
    y = A * np.exp(-t / tau)

    np.savez(
        GOLDEN_DIR / "exponential_model.npz",
        t=t,
        y=y,
        A=A,
        tau=tau,
    )
    print("Generated: exponential_model.npz")


def generate_fit_golden() -> None:
    """Generate golden file for fitting results."""
    # Linear fit
    x_linear = np.linspace(0, 10, 100)
    y_linear = 2.5 * x_linear + 1.5

    # Exponential fit
    x_exp = np.linspace(0, 5, 50)
    y_exp = 100.0 * np.exp(-x_exp / 1.5)

    np.savez(
        GOLDEN_DIR / "fit_results.npz",
        x_linear=x_linear,
        y_linear=y_linear,
        expected_slope=2.5,
        expected_intercept=1.5,
        x_exp=x_exp,
        y_exp=y_exp,
        expected_amplitude=100.0,
        expected_tau=1.5,
    )
    print("Generated: fit_results.npz")


def generate_all() -> None:
    """Generate all golden files."""
    GOLDEN_DIR.mkdir(exist_ok=True)
    generate_maxwell_golden()
    generate_exponential_golden()
    generate_fit_golden()
    print(f"\nAll golden files generated in: {GOLDEN_DIR}")


if __name__ == "__main__":
    generate_all()
