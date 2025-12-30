"""NLSQ-backed optimization adapters replacing scipy.optimize usage."""

from __future__ import annotations

from typing import Any, Callable, Iterable

import jax.numpy as jnp
import nlsq


class OptimizeResult(dict):
    """Dict-like result with attribute access for compatibility."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def curve_fit(*args: Any, **kwargs: Any):
    """Adapter to nlsq.curve_fit with SciPy-compatible return."""
    result = nlsq.curve_fit(*args, **kwargs)
    if isinstance(result, tuple):
        return result
    return result.popt, result.pcov


def least_squares(fun: Callable[..., Any], x0: Any, **kwargs: Any) -> OptimizeResult:
    solver = nlsq.LeastSquares()
    result = solver.least_squares(fun, x0, **kwargs)
    return OptimizeResult(result)


def minimize(
    fun: Callable[..., Any],
    x0: Any,
    args: Iterable[Any] = (),
    **kwargs: Any,
) -> OptimizeResult:
    """Minimize a scalar objective via NLSQ least-squares."""

    kwargs.pop("method", None)

    def residuals(x):
        value = fun(x, *args)
        return jnp.atleast_1d(value)

    result = least_squares(residuals, x0, **kwargs)
    result["fun"] = fun(result["x"], *args)
    return result


def nnls(a: Any, b: Any):
    """Non-negative least squares via bounded NLSQ."""
    a_arr = jnp.asarray(a)
    b_arr = jnp.asarray(b)

    def residuals(x):
        return a_arr @ x - b_arr

    x0 = jnp.zeros(a_arr.shape[1])
    result = least_squares(residuals, x0, bounds=(0.0, jnp.inf))
    rnorm = jnp.linalg.norm(result["fun"])
    return result["x"], rnorm


def root(
    fun: Callable[..., Any],
    x0: Any,
    args: Iterable[Any] = (),
    **kwargs: Any,
) -> OptimizeResult:
    """Root finding via NLSQ least-squares."""

    method = kwargs.pop("method", None)
    jac = kwargs.pop("jac", None)
    if method not in (None, "trf", "dogbox", "lm"):
        method = "trf"

    def residuals(x):
        return fun(x, *args)

    if callable(jac):
        kwargs["jac"] = jac
    result = least_squares(residuals, x0, method=method or "trf", **kwargs)
    return result


def _unsupported_global(name: str) -> None:
    raise NotImplementedError(
        f"{name} is not available in the NLSQ modernization path. "
        "Use least-squares fitting instead."
    )


def basinhopping(*args: Any, **kwargs: Any):
    _unsupported_global("basinhopping")


def dual_annealing(*args: Any, **kwargs: Any):
    _unsupported_global("dual_annealing")


def differential_evolution(*args: Any, **kwargs: Any):
    _unsupported_global("differential_evolution")


def shgo(*args: Any, **kwargs: Any):
    _unsupported_global("shgo")


def brute(*args: Any, **kwargs: Any):
    _unsupported_global("brute")
