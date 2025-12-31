"""NLSQ-backed optimization adapters replacing scipy.optimize usage."""

from __future__ import annotations

from typing import Any, Callable, Iterable, NoReturn

import jax.numpy as jnp
import nlsq


Array = jnp.ndarray


class OptimizeResult(dict):
    """Dict-like result with attribute access for compatibility."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def curve_fit(*args: Any, **kwargs: Any) -> tuple[Array, Array]:
    """Adapter to nlsq.curve_fit with SciPy-compatible return.

    Wraps NLSQ's curve_fit to ensure consistent tuple output matching SciPy's
    scipy.optimize.curve_fit interface.

    Args:
        *args: Positional arguments passed to nlsq.curve_fit (typically f, xdata,
            ydata, p0).
        **kwargs: Keyword arguments passed to nlsq.curve_fit (e.g., bounds, method,
            jac, ftol).

    Returns:
        tuple[Array, Array]: A tuple containing:
            - popt: Optimal parameter values as a JAX array
            - pcov: Estimated covariance matrix of parameters as a JAX array
    """
    result = nlsq.curve_fit(*args, **kwargs)
    if isinstance(result, tuple):
        return result
    return result.popt, result.pcov


def least_squares(
    fun: Callable[..., Any], x0: Any, **kwargs: Any
) -> OptimizeResult:
    """Solve a nonlinear least-squares problem using NLSQ.

    Minimizes the sum of squares of a vector-valued residual function using
    trust-region optimization with automatic differentiation.

    Args:
        fun: Residual function taking parameters x and returning a vector of
            residuals. Signature: fun(x) -> Array.
        x0: Initial parameter guess. Can be a list, array, or JAX array.
        **kwargs: Additional keyword arguments passed to NLSQ's least_squares
            (e.g., bounds, method, ftol, xtol, max_nfev).

    Returns:
        OptimizeResult: Dictionary-like object with attributes including:
            - x: Optimal parameters
            - fun: Residuals at the optimal parameters
            - cost: Sum of squared residuals at the optimal point
            - jac: Jacobian matrix at the optimal point
            - nfev: Number of function evaluations
            - status: Convergence status code
            - message: Convergence message
    """
    solver = nlsq.LeastSquares()
    result = solver.least_squares(fun, x0, **kwargs)
    return OptimizeResult(result)


def minimize(
    fun: Callable[..., Any],
    x0: Any,
    args: Iterable[Any] = (),
    **kwargs: Any,
) -> OptimizeResult:
    """Minimize a scalar objective via NLSQ least-squares.

    Converts a scalar minimization problem into a least-squares problem by
    wrapping the objective in a residual function.

    Args:
        fun: Scalar objective function to minimize. Signature: fun(x, *args) -> float.
        x0: Initial parameter guess.
        args: Extra arguments passed to the objective function.
        **kwargs: Additional keyword arguments passed to least_squares. The "method"
            argument is ignored (NLSQ selects the method automatically).

    Returns:
        OptimizeResult: Dictionary-like object with attributes including:
            - x: Optimal parameters
            - fun: Scalar objective value at the optimal point
            - nfev: Number of function evaluations
            - status: Convergence status code
            - message: Convergence message
    """

    kwargs.pop("method", None)

    def residuals(x):
        """Convert scalar objective to residual form.

        Args:
            x: Parameter vector.

        Returns:
            jnp.ndarray: 1D array containing the objective value.
        """
        value = fun(x, *args)
        return jnp.atleast_1d(value)

    result = least_squares(residuals, x0, **kwargs)
    result["fun"] = fun(result["x"], *args)
    return result


def nnls(a: Any, b: Any) -> tuple[Array, float]:
    """Non-negative least squares via bounded NLSQ.

    Solves the constrained optimization problem: minimize ||Ax - b||^2
    subject to x >= 0 using NLSQ's bounded least-squares solver.

    Args:
        a: Coefficient matrix. Shape (m, n).
        b: Right-hand side vector. Shape (m,).

    Returns:
        tuple[Array, float]: A tuple containing:
            - x: Non-negative solution vector. Shape (n,).
            - rnorm: Residual norm ||Ax - b|| at the solution.
    """
    a_arr = jnp.asarray(a)
    b_arr = jnp.asarray(b)

    def residuals(x):
        """Linear residual function for NNLS.

        Args:
            x: Parameter vector.

        Returns:
            jnp.ndarray: Residuals Ax - b.
        """
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
    """Root finding via NLSQ least-squares.

    Finds roots of a vector-valued function (i.e., solves fun(x) = 0) by
    minimizing the sum of squared residuals using NLSQ's trust-region methods.

    Args:
        fun: Vector-valued function for which to find roots. Signature:
            fun(x, *args) -> Array.
        x0: Initial parameter guess.
        args: Extra arguments passed to the function.
        **kwargs: Additional keyword arguments passed to least_squares. Supported
            methods are "trf", "dogbox", "lm". The "jac" argument can provide an
            analytical Jacobian.

    Returns:
        OptimizeResult: Dictionary-like object with attributes including:
            - x: Solution vector where fun(x) is approximately zero
            - fun: Residuals at the solution
            - nfev: Number of function evaluations
            - status: Convergence status code
            - message: Convergence message
    """

    method = kwargs.pop("method", None)
    jac = kwargs.pop("jac", None)
    if method not in (None, "trf", "dogbox", "lm"):
        method = "trf"

    def residuals(x):
        """Residual function for root finding.

        Args:
            x: Parameter vector.

        Returns:
            jnp.ndarray: Function values to drive to zero.
        """
        return fun(x, *args)

    if callable(jac):
        kwargs["jac"] = jac
    result = least_squares(residuals, x0, method=method or "trf", **kwargs)
    return result


def _unsupported_global(name: str) -> NoReturn:
    raise NotImplementedError(
        f"{name} is not available in the NLSQ modernization path. "
        "Use least-squares fitting instead."
    )


def basinhopping(*args: Any, **kwargs: Any):
    """Unsupported global optimizer - not available in NLSQ modernization path.

    Raises:
        NotImplementedError: Always raised. Use least-squares fitting instead.
    """
    _unsupported_global("basinhopping")


def dual_annealing(*args: Any, **kwargs: Any):
    """Unsupported global optimizer - not available in NLSQ modernization path.

    Raises:
        NotImplementedError: Always raised. Use least-squares fitting instead.
    """
    _unsupported_global("dual_annealing")


def differential_evolution(*args: Any, **kwargs: Any):
    """Unsupported global optimizer - not available in NLSQ modernization path.

    Raises:
        NotImplementedError: Always raised. Use least-squares fitting instead.
    """
    _unsupported_global("differential_evolution")


def shgo(*args: Any, **kwargs: Any):
    """Unsupported global optimizer - not available in NLSQ modernization path.

    Raises:
        NotImplementedError: Always raised. Use least-squares fitting instead.
    """
    _unsupported_global("shgo")


def brute(*args: Any, **kwargs: Any):
    """Unsupported global optimizer - not available in NLSQ modernization path.

    Raises:
        NotImplementedError: Always raised. Use least-squares fitting instead.
    """
    _unsupported_global("brute")
