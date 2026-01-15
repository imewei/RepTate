"""Lightweight ODE helpers for deterministic integration."""
from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def rk4_step_jax(
    y: jnp.ndarray, t: float, dt: float, deriv: Callable[[jnp.ndarray, float], jnp.ndarray]
) -> jnp.ndarray:
    """Single RK4 step using JAX arrays.

    JIT-compiled for optimal performance during integration (FR-014).

    Args:
        y: Current state vector.
        t: Current time.
        dt: Time step size.
        deriv: Derivative function f(y, t) returning dy/dt.

    Returns:
        jnp.ndarray: State vector at t + dt.
    """
    k1 = deriv(y, t)
    k2 = deriv(y + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = deriv(y + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = deriv(y + dt * k3, t + dt)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_integrate_jax(
    deriv: Callable[[jnp.ndarray, float], jnp.ndarray],
    y0: jnp.ndarray,
    t: jnp.ndarray,
) -> jnp.ndarray:
    """JAX-based RK4 integrator using scan for efficient compilation.

    JIT-compilable version of RK4 integration using JAX primitives.

    Args:
        deriv: Derivative function f(y, t) returning dy/dt. Must be JAX-traceable.
        y0: Initial state vector.
        t: Time points for integration (1D array).

    Returns:
        jnp.ndarray: Solution array with shape (len(t), len(y0)).
    """

    def step(
        carry: tuple[jnp.ndarray, jnp.ndarray], x: jnp.ndarray
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        y, t_prev = carry
        t_curr = x
        dt = t_curr - t_prev
        y_new = rk4_step_jax(y, t_prev, dt, deriv)
        return (y_new, t_curr), y_new

    y0_arr = jnp.atleast_1d(y0)
    _, ys = jax.lax.scan(step, (y0_arr, t[0]), t[1:])
    return jnp.vstack([y0_arr[None, :], ys])


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
