"""Lightweight ODE helpers for deterministic integration."""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np


def rk4_integrate(
    deriv: Callable[[np.ndarray, float, Optional[Sequence[float]]], Sequence[float]],
    y0: Sequence[float] | float,
    t: Sequence[float],
    args: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Deterministic RK4 integrator for fixed time grids.

    Implements the fourth-order Runge-Kutta method for solving ordinary
    differential equations dy/dt = deriv(y, t, args) on a fixed time grid.

    Args:
        deriv: Derivative function with signature f(y, t) or f(y, t, args)
            that returns dy/dt. Must return a sequence of floats representing
            the time derivatives of the state variables.
        y0: Initial conditions. Can be a scalar for 1D systems or a sequence
            for multi-dimensional systems.
        t: Time points for integration. Must be a monotonically increasing
            sequence defining the output time grid.
        args: Optional constant parameters passed to the derivative function.

    Returns:
        np.ndarray: Solution array with shape (len(t), len(y0)), where each
            row contains the state vector at the corresponding time point.
    """
    y = np.atleast_1d(np.asarray(y0, dtype=float))
    t_arr = np.asarray(t, dtype=float)
    ys = np.zeros((len(t_arr), y.size), dtype=float)
    ys[0] = y
    for i in range(1, len(t_arr)):
        dt = t_arr[i] - t_arr[i - 1]
        ti = t_arr[i - 1]
        if args is None:
            k1 = np.asarray(deriv(y, ti), dtype=float)
            k2 = np.asarray(deriv(y + 0.5 * dt * k1, ti + 0.5 * dt), dtype=float)
            k3 = np.asarray(deriv(y + 0.5 * dt * k2, ti + 0.5 * dt), dtype=float)
            k4 = np.asarray(deriv(y + dt * k3, ti + dt), dtype=float)
        else:
            k1 = np.asarray(deriv(y, ti, args), dtype=float)
            k2 = np.asarray(
                deriv(y + 0.5 * dt * k1, ti + 0.5 * dt, args), dtype=float
            )
            k3 = np.asarray(
                deriv(y + 0.5 * dt * k2, ti + 0.5 * dt, args), dtype=float
            )
            k4 = np.asarray(deriv(y + dt * k3, ti + dt, args), dtype=float)
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        ys[i] = y
    return ys
