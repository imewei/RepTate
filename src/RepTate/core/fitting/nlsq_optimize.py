"""NLSQ-backed optimization adapters replacing scipy.optimize usage."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Literal, NoReturn

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


def fit(
    f: Callable[..., Any],
    xdata: Any,
    ydata: Any,
    p0: Any | None = None,
    sigma: Any | None = None,
    absolute_sigma: bool = False,
    bounds: tuple[Any, Any] = (-jnp.inf, jnp.inf),
    *,
    workflow: Literal["auto", "auto_global", "hpc"] | None = "auto",
    method: str | None = None,
    memory_limit_gb: float | None = None,
    size_threshold: int = 1_000_000,
    show_progress: bool = False,
    chunk_size: int | None = None,
    multistart: bool | None = None,
    n_starts: int | None = None,
    sampler: Literal["lhs", "sobol", "halton"] = "lhs",
    center_on_p0: bool = True,
    scale_factor: float = 1.0,
    **kwargs: Any,
) -> tuple[Array, Array]:
    """Unified curve fitting with workflow-based memory management.

    Wrapper around nlsq.fit() providing automatic strategy selection based on
    dataset size and available memory. This is the recommended entry point for
    curve fitting with large datasets.

    Workflows:
        - "auto" (default): Memory-aware local optimization. Automatically selects
          between standard, chunked, or streaming processing based on dataset size
          and available memory. Bounds are optional.
        - "auto_global": Memory-aware global optimization using multi-start or
          CMA-ES. Requires bounds. Auto-selects CMA-ES when parameter scale ratio
          exceeds 1000.
        - "hpc": Global optimization with checkpointing for HPC environments.
          Requires bounds. Use with `checkpoint_dir` kwarg for fault tolerance.

    Args:
        f: Model function f(x, *params) -> y. Must use jax.numpy operations for
            GPU acceleration and automatic differentiation.
        xdata: Independent variable data.
        ydata: Dependent variable data to fit.
        p0: Initial parameter guess. If None, uses heuristics.
        sigma: Uncertainties in ydata for weighted fitting.
        absolute_sigma: If True, sigma represents absolute uncertainties.
        bounds: Parameter bounds as (lower, upper). Required for "auto_global"
            and "hpc" workflows.
        workflow: Memory management strategy. One of "auto", "auto_global", "hpc".
        method: Optimization algorithm ("trf", "lm", or None for auto).
        memory_limit_gb: Maximum memory usage in GB. If None, auto-detects.
        size_threshold: Dataset size threshold for large dataset processing.
            Default 1,000,000 points.
        show_progress: Display progress bar for long operations.
        chunk_size: Override automatic chunk size calculation.
        multistart: Enable multi-start optimization. Default None (auto).
        n_starts: Number of starting points for multi-start. Default None (10).
        sampler: Sampling strategy for multi-start ("lhs", "sobol", "halton").
        center_on_p0: Center multi-start samples around p0.
        scale_factor: Scale factor for exploration region.
        **kwargs: Additional optimization parameters (ftol, xtol, gtol, max_nfev).

    Returns:
        tuple[Array, Array]: A tuple containing:
            - popt: Optimal parameter values as a JAX array
            - pcov: Estimated covariance matrix of parameters as a JAX array

    Examples:
        Basic usage with automatic memory management:

        >>> from RepTate.core.fitting.nlsq_optimize import fit
        >>> import jax.numpy as jnp
        >>> def model(x, a, b): return a * jnp.exp(-b * x)
        >>> popt, pcov = fit(model, xdata, ydata, p0=[1.0, 0.5])

        Large dataset with progress bar:

        >>> popt, pcov = fit(model, big_xdata, big_ydata, show_progress=True)

        Global optimization for multi-modal problems:

        >>> popt, pcov = fit(
        ...     model, xdata, ydata,
        ...     workflow="auto_global",
        ...     bounds=([0, 0], [10, 5]),
        ...     n_starts=20,
        ... )
    """
    result = nlsq.fit(
        f,
        xdata,
        ydata,
        p0=p0,
        sigma=sigma,
        absolute_sigma=absolute_sigma,
        bounds=bounds,
        workflow=workflow,
        method=method,
        memory_limit_gb=memory_limit_gb,
        size_threshold=size_threshold,
        show_progress=show_progress,
        chunk_size=chunk_size,
        multistart=multistart,
        n_starts=n_starts,
        sampler=sampler,
        center_on_p0=center_on_p0,
        scale_factor=scale_factor,
        **kwargs,
    )
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
