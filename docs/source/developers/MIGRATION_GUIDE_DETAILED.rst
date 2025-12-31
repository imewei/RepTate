=============================================
RepTate Modernization: Developer Migration Guide
=============================================

This guide helps developers understand and use the modern

ized RepTate infrastructure, migrate legacy code, and follow best practices.

.. contents:: Table of Contents
   :local:
   :depth: 3

Introduction
============

RepTate has been modernized to eliminate security vulnerabilities, improve performance, and adopt modern scientific computing practices. This guide covers:

- How to use the new JAX-based fitting infrastructure
- How to use SafeSerializer instead of pickle
- How to use safe_eval instead of eval/exec
- How to work with the Strangler Fig infrastructure
- How to add new theories using modern patterns

Target Audience
---------------

- Developers extending RepTate with new theories/tools
- Maintainers updating legacy code
- Contributors fixing bugs or adding features

Prerequisites
-------------

**Required Knowledge:**

- Python 3.12+ (type hints, dataclasses, pathlib)
- Basic JAX concepts (arrays, JIT compilation)
- Qt/PySide6 basics (for GUI work)

**Recommended Reading:**

- :doc:`MODERNIZATION_ARCHITECTURE` - Architecture overview
- :doc:`testing` - Testing practices
- ``MODERNIZATION_SUMMARY.md`` - Quick reference

Part 1: Using JAX-Based Fitting
================================

Overview
--------

RepTate now uses NLSQ (JAX-based non-linear least squares) instead of scipy.optimize.curve_fit. This provides:

- 8-10x faster curve fitting on CPU
- 50-100x faster on GPU (optional)
- More accurate gradients via auto-differentiation
- Better numerical stability

Migration Pattern
-----------------

**Before (Legacy SciPy):**

.. code-block:: python

    from scipy.optimize import curve_fit
    import numpy as np

    def theory_func(x, param1, param2):
        """Maxwell model."""
        return param1 * np.exp(-x / param2)

    # Fit data
    xdata = np.array([0.1, 1.0, 10.0])
    ydata = np.array([100, 50, 10])
    p0 = [100.0, 1.0]  # Initial guess

    popt, pcov = curve_fit(theory_func, xdata, ydata, p0=p0)

**After (Modern JAX/NLSQ):**

.. code-block:: python

    from RepTate.core.fitting.nlsq_fit import fit_data
    import jax.numpy as jnp
    from jax import jit

    @jit  # JIT compilation for performance
    def theory_func(params, x):
        """Maxwell model (JAX-compatible)."""
        param1, param2 = params
        return param1 * jnp.exp(-x / param2)

    # Fit data
    xdata = jnp.array([0.1, 1.0, 10.0])
    ydata = jnp.array([100, 50, 10])
    p0 = jnp.array([100.0, 1.0])  # Initial guess

    result = fit_data(theory_func, xdata, ydata, p0, bounds=None)
    popt = result.params
    fit_error = result.error

**Key Differences:**

1. Function signature: ``theory_func(params, x)`` instead of ``theory_func(x, *params)``
2. Use ``jax.numpy`` (jnp) instead of ``numpy`` (np)
3. Use ``jnp.array`` instead of ``np.array``
4. Decorate with ``@jit`` for performance
5. Result is ``OptimizationResult`` object (not tuple)

Complete Example: Fitting a Maxwell Model
------------------------------------------

.. code-block:: python

    from RepTate.core.fitting.nlsq_fit import fit_data, OptimizationResult
    import jax.numpy as jnp
    from jax import jit
    from typing import Tuple

    @jit
    def maxwell_model(params: jnp.ndarray, omega: jnp.ndarray) -> jnp.ndarray:
        """Compute G' and G'' for Maxwell model.

        Args:
            params: [G, tau] - modulus and relaxation time
            omega: Angular frequencies

        Returns:
            Concatenated [G', G''] array
        """
        G, tau = params
        omega_tau_sq = (omega * tau) ** 2

        G_prime = G * omega_tau_sq / (1 + omega_tau_sq)
        G_double_prime = G * omega * tau / (1 + omega_tau_sq)

        return jnp.concatenate([G_prime, G_double_prime])

    def fit_maxwell_to_data(
        omega: jnp.ndarray,
        G_prime_data: jnp.ndarray,
        G_double_prime_data: jnp.ndarray,
        initial_guess: Tuple[float, float] = (1e5, 1.0)
    ) -> OptimizationResult:
        """Fit Maxwell model to experimental data.

        Args:
            omega: Angular frequencies
            G_prime_data: Storage modulus data
            G_double_prime_data: Loss modulus data
            initial_guess: (G, tau) initial values

        Returns:
            OptimizationResult with fitted parameters
        """
        # Concatenate data
        ydata = jnp.concatenate([G_prime_data, G_double_prime_data])

        # Parameter bounds
        bounds = (
            jnp.array([1e3, 1e-6]),  # Lower bounds [G_min, tau_min]
            jnp.array([1e8, 1e3])     # Upper bounds [G_max, tau_max]
        )

        # Fit
        result = fit_data(
            maxwell_model,
            omega,
            ydata,
            p0=jnp.array(initial_guess),
            bounds=bounds
        )

        return result

    # Usage
    omega = jnp.logspace(-2, 2, 50)
    G_prime = jnp.array([...])      # Your experimental data
    G_double_prime = jnp.array([...])  # Your experimental data

    result = fit_maxwell_to_data(omega, G_prime, G_double_prime)

    if result.success:
        G_fitted, tau_fitted = result.params
        print(f"Fitted G = {G_fitted:.2e}, tau = {tau_fitted:.4f}")
        print(f"Fit error = {result.error:.4e}")
    else:
        print(f"Fit failed: {result.message}")

Common JAX Patterns
-------------------

**1. Array Operations:**

.. code-block:: python

    # NumPy (old)
    import numpy as np
    result = np.exp(x) + np.sin(y)

    # JAX (new)
    import jax.numpy as jnp
    result = jnp.exp(x) + jnp.sin(y)

**2. Conditional Logic:**

.. code-block:: python

    # NumPy (old) - doesn't work with JIT
    if condition:
        result = compute_a()
    else:
        result = compute_b()

    # JAX (new) - JIT-compatible
    from jax import lax
    result = lax.cond(condition, compute_a, compute_b)

    # Or use jnp.where for element-wise conditions
    result = jnp.where(condition, value_if_true, value_if_false)

**3. Loops:**

.. code-block:: python

    # NumPy (old) - slow with JIT
    result = 0
    for i in range(n):
        result += compute(i)

    # JAX (new) - JIT-optimized
    from jax import lax

    def body_fun(i, carry):
        return carry + compute(i)

    result = lax.fori_loop(0, n, body_fun, 0)

**4. Random Numbers:**

.. code-block:: python

    # NumPy (old) - non-deterministic
    import numpy as np
    random_values = np.random.randn(100)

    # JAX (new) - deterministic with explicit key
    import jax
    key = jax.random.PRNGKey(42)
    random_values = jax.random.normal(key, shape=(100,))

Troubleshooting JAX
-------------------

**Issue 1: "TracerArrayConversionError"**

.. code-block:: python

    # BAD: Using Python control flow on traced arrays
    @jit
    def bad_function(x):
        if x > 0:  # ERROR: Can't use Python if with traced array
            return x ** 2
        else:
            return -x

    # GOOD: Use jnp.where or lax.cond
    @jit
    def good_function(x):
        return jnp.where(x > 0, x ** 2, -x)

**Issue 2: "Side effects not allowed in JIT"**

.. code-block:: python

    # BAD: Modifying external state
    results = []

    @jit
    def bad_function(x):
        results.append(x)  # ERROR: Side effect
        return x ** 2

    # GOOD: Return values instead
    @jit
    def good_function(x):
        return x ** 2

    results = [good_function(x) for x in data]

**Issue 3: "Array shape must be static"**

.. code-block:: python

    # BAD: Dynamic array shapes
    @jit
    def bad_function(x, n):
        return jnp.zeros(n)  # ERROR: n is traced, not static

    # GOOD: Use static_argnames
    from functools import partial

    @partial(jit, static_argnames=['n'])
    def good_function(x, n):
        return jnp.zeros(n)

Part 2: Using SafeSerializer Instead of Pickle
===============================================

Overview
--------

RepTate now uses SafeSerializer (JSON/NPZ format) instead of pickle to eliminate arbitrary code execution vulnerabilities.

Migration Pattern
-----------------

**Before (Legacy Pickle):**

.. code-block:: python

    import pickle
    from pathlib import Path

    # Save
    data = {"param1": 1.0, "param2": 2.0, "results": np.array([1, 2, 3])}
    with open("data.pkl", "wb") as f:
        pickle.dump(data, f)

    # Load
    with open("data.pkl", "rb") as f:
        loaded = pickle.load(f)

**After (Modern SafeSerializer):**

.. code-block:: python

    from RepTate.core.serialization import SafeSerializer
    from pathlib import Path
    import numpy as np

    # Save
    data = {"param1": 1.0, "param2": 2.0, "results": np.array([1, 2, 3])}
    result = SafeSerializer.save(Path("data"), data)
    # Creates: data.json, data.npz

    # Load
    loaded = SafeSerializer.load(Path("data"))

**Key Differences:**

1. Pass ``Path`` object, not string (no file extension)
2. Creates two files: ``.json`` (metadata) and ``.npz`` (arrays)
3. Only whitelisted types allowed (int, float, str, bool, list, dict, ndarray)
4. Functions, lambdas, classes cannot be serialized

Complete Example: Saving Theory State
--------------------------------------

.. code-block:: python

    from RepTate.core.serialization import SafeSerializer
    from pathlib import Path
    import numpy as np

    class TheoryState:
        """State for a theory (modern pattern)."""

        def __init__(self):
            self.parameters = {"G": 1e5, "tau": 1.0}
            self.fitted_params = np.array([1e5, 1.0])
            self.predictions = np.array([100, 50, 10])
            self.fit_error = 0.05

        def save(self, filepath: Path) -> None:
            """Save theory state to file."""
            data = {
                "parameters": self.parameters,
                "fitted_params": self.fitted_params,
                "predictions": self.predictions,
                "fit_error": self.fit_error,
            }
            SafeSerializer.save(filepath, data)

        @classmethod
        def load(cls, filepath: Path) -> "TheoryState":
            """Load theory state from file."""
            data = SafeSerializer.load(filepath)

            state = cls()
            state.parameters = data["parameters"]
            state.fitted_params = data["fitted_params"]
            state.predictions = data["predictions"]
            state.fit_error = data["fit_error"]

            return state

    # Usage
    state = TheoryState()
    state.save(Path("theory_state"))
    # Creates: theory_state.json, theory_state.npz

    loaded_state = TheoryState.load(Path("theory_state"))

Migrating Existing Pickle Files
--------------------------------

Use the provided migration script to convert legacy .pkl files:

.. code-block:: bash

    # Migrate a single file
    python scripts/migrate_pickle_files.py data.pkl

    # Migrate all .pkl files in a directory
    python scripts/migrate_pickle_files.py data_directory/

**What the script does:**

1. Loads .pkl file using pickle (last time!)
2. Converts to SafeSerializer format
3. Saves as .json + .npz
4. Renames original to .pkl.bak (for safety)

**Manual migration:**

.. code-block:: python

    from RepTate.core.serialization import migrate_pickle
    from pathlib import Path

    # Convert legacy pickle file
    new_path = migrate_pickle(Path("legacy_data.pkl"))
    # Creates: legacy_data.json, legacy_data.npz
    # Renames: legacy_data.pkl -> legacy_data.pkl.bak

Handling Unsupported Types
---------------------------

If you get ``TypeError: Cannot serialize type``, convert to supported types:

.. code-block:: python

    # BAD: Trying to serialize a function
    data = {
        "func": lambda x: x ** 2,  # ERROR: Cannot serialize function
    }

    # GOOD: Store function name or parameters instead
    data = {
        "func_name": "power_law",
        "func_params": {"exponent": 2.0},
    }

    # BAD: Trying to serialize a class instance
    data = {
        "model": SomeModel(),  # ERROR: Cannot serialize class
    }

    # GOOD: Extract data from instance
    model = SomeModel()
    data = {
        "model_type": "SomeModel",
        "model_params": model.get_params(),  # Extract dict/array
    }

Part 3: Using safe_eval Instead of eval()
==========================================

Overview
--------

RepTate now uses ``safe_eval`` (AST-based) instead of ``eval()`` to eliminate code injection vulnerabilities.

Migration Pattern
-----------------

**Before (Legacy eval):**

.. code-block:: python

    # UNSAFE: Direct eval() usage
    expr = "A * exp(-t / tau)"
    variables = {"A": 1000.0, "t": 0.5, "tau": 0.1}

    result = eval(expr, {}, variables)  # DANGEROUS: Code injection risk

**After (Modern safe_eval):**

.. code-block:: python

    from RepTate.core.safe_eval import safe_eval

    # SAFE: Whitelist-validated evaluation
    expr = "A * exp(-t / tau)"
    variables = {"A": 1000.0, "t": 0.5, "tau": 0.1}

    result = safe_eval(expr, variables)  # Safe: No code execution

**Key Differences:**

1. Only whitelisted operations allowed
2. No import, attribute access, or subscript
3. Clear error messages for disallowed operations
4. Deterministic behavior (no hidden side effects)

Complete Example: Custom Transformation
----------------------------------------

.. code-block:: python

    from RepTate.core.safe_eval import SafeExpression, safe_eval_array
    import numpy as np

    def apply_custom_transformation(
        data: np.ndarray,
        expression: str,
        variables: dict
    ) -> np.ndarray:
        """Apply user-defined transformation to data.

        Args:
            data: Input data array
            expression: Mathematical expression (e.g., "x ** 2 + sin(x)")
            variables: Additional variables (x is automatically included)

        Returns:
            Transformed data array

        Raises:
            ValueError: If expression contains disallowed operations
        """
        # Parse and validate expression once
        try:
            expr = SafeExpression.parse(expression)
        except ValueError as e:
            raise ValueError(f"Invalid expression: {e}")

        # Apply to each data point
        result = np.zeros_like(data)
        for i, x_value in enumerate(data):
            bindings = {**variables, "x": float(x_value)}
            result[i] = expr.evaluate(bindings)

        return result

    # Usage
    data = np.linspace(0, 10, 100)
    transformed = apply_custom_transformation(
        data,
        expression="x ** 2 + A * sin(omega * x)",
        variables={"A": 1.0, "omega": 2 * np.pi}
    )

Array Expressions (Vectorized)
-------------------------------

For array operations, use ``safe_eval_array``:

.. code-block:: python

    from RepTate.core.safe_eval import safe_eval_array
    import numpy as np

    # Vectorized evaluation (more efficient)
    x = np.linspace(0, 10, 100)
    result = safe_eval_array(
        "A0 + A1 * x + A2 * x ** 2",
        variables={"A0": 1.0, "A1": 2.0, "A2": 0.5, "x": x}
    )
    # result is a 100-element array

Allowed vs Disallowed Operations
---------------------------------

**Allowed (Whitelist):**

- Arithmetic: ``+``, ``-``, ``*``, ``/``, ``**``
- Unary: ``+x``, ``-x``
- Math functions: sin, cos, tan, exp, log, log10, sqrt, abs
- Array functions: arcsin, arctan, sinh, cosh, floor, ceil, mod, etc.
- Constants (array mode): ``e``, ``pi``

**Disallowed (Security):**

- Import: ``import math`` → ERROR
- Attribute access: ``x.method()`` → ERROR
- Subscript: ``arr[0]`` → ERROR
- Lambda: ``lambda x: x**2`` → ERROR
- Comprehension: ``[x**2 for x in range(10)]`` → ERROR
- Comparison: ``x > 0`` → ERROR
- Boolean: ``x and y`` → ERROR
- Builtins: ``eval``, ``exec``, ``compile``, ``open``, etc. → ERROR

Error Handling
--------------

.. code-block:: python

    from RepTate.core.safe_eval import safe_eval, SafeExpression

    # Handle invalid expressions
    try:
        expr = SafeExpression.parse("import os; os.system('ls')")
    except ValueError as e:
        print(f"Disallowed operation: {e}")
        # Output: Disallowed operation: Invalid expression syntax

    # Handle missing variables
    try:
        result = safe_eval("x + y", variables={"x": 1.0})
    except ValueError as e:
        print(f"Missing variable: {e}")
        # Output: Missing variable: Missing variable(s): y

Part 4: Working with Strangler Fig Infrastructure
==================================================

Overview
--------

RepTate uses the Strangler Fig pattern to incrementally decompose god classes. New controllers/managers are extracted while keeping legacy classes operational.

Understanding Strangler Fig
----------------------------

**Concept:**

Like a strangler fig tree that grows around a host tree, new components wrap around legacy code, gradually taking over responsibilities.

**Benefits:**

- No big-bang rewrite (incremental migration)
- Legacy code continues working
- Easy rollback via feature flags
- Can be completed over months (low risk)

Using Extracted Controllers
----------------------------

**Legacy Pattern (Don't Use):**

.. code-block:: python

    # Direct method calls on god class
    class SomeLegacyCode:
        def do_something(self):
            app_window = get_application_window()
            app_window.load_file("data.txt")  # 3000+ LOC class
            app_window.save_results("output.txt")

**Modern Pattern (Use This):**

.. code-block:: python

    # Use extracted controllers
    from RepTate.gui.FileIOController import FileIOController

    class ModernCode:
        def __init__(self, application_window):
            self.file_io = FileIOController(application_window)

        def do_something(self):
            self.file_io.load_files([Path("data.txt")])  # 349 LOC controller
            self.file_io.save_data(Path("output.txt"))

Example: Using Controllers
---------------------------

.. code-block:: python

    from RepTate.gui.FileIOController import FileIOController
    from RepTate.gui.ViewCoordinator import ViewCoordinator
    from RepTate.gui.DatasetManager import DatasetManager
    from pathlib import Path

    class NewFeature:
        """Example of using extracted controllers."""

        def __init__(self, application_window):
            # Initialize controllers
            self.file_io = FileIOController(application_window)
            self.view = ViewCoordinator(application_window)
            self.datasets = DatasetManager(application_window)

        def load_and_plot(self, filepath: Path):
            """Load data and update plot."""
            # Load files
            self.file_io.load_files([filepath])

            # Get active dataset
            dataset = self.datasets.get_active_dataset()

            # Update plot
            self.view.update_plot()

        def save_current_view(self, output_path: Path):
            """Save current view state."""
            self.view.save_view_state(output_path)

Part 5: Adding New Theories (Modern Pattern)
=============================================

Overview
--------

When adding a new theory to RepTate, follow the modern pattern using JAX, safe_eval, and proper architecture.

Theory Template
---------------

.. code-block:: python

    """Theory: [Theory Name]

    Description: [Brief description of the theory]

    References:
        [1] Author et al., Journal, Year
    """

    from __future__ import annotations

    import jax.numpy as jnp
    from jax import jit
    from typing import TYPE_CHECKING

    from RepTate.core.CmdBase import CmdBase, CmdMode
    from RepTate.core.Parameter import Parameter, ParameterType
    from RepTate.core.Theory import Theory
    from RepTate.core.safe_eval import safe_eval

    if TYPE_CHECKING:
        from RepTate.core.DataTable import DataTable


    class TheoryNewModel(Theory):
        """[Theory Name] theory implementation.

        [Detailed description]

        Parameters:
            param1 (float): [Description] (default: [value])
            param2 (float): [Description] (default: [value])
        """

        thname = "NewModel"
        description = "[Brief description]"
        citations = "[Author et al., Journal, Year]"

        def __init__(self, name: str = "", parent_dataset=None, axarr=None):
            """Initialize theory."""
            super().__init__(name, parent_dataset, axarr)

            # Define parameters
            self.parameters["param1"] = Parameter(
                name="param1",
                value=1.0,
                description="[Parameter 1 description]",
                type=ParameterType.real,
                opt_type=OptType.opt,
                min_value=0.0,
                max_value=100.0,
            )
            self.parameters["param2"] = Parameter(
                name="param2",
                value=2.0,
                description="[Parameter 2 description]",
                type=ParameterType.real,
                opt_type=OptType.opt,
                min_value=0.0,
                max_value=10.0,
            )

            # Initialize theory
            self.update_parameter_table()

        @jit
        @staticmethod
        def theory_function(params: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
            """Compute theory predictions (JAX-compatible).

            Args:
                params: [param1, param2, ...]
                x: Independent variable (e.g., frequency, time)

            Returns:
                Predicted values
            """
            param1, param2 = params
            return param1 * jnp.exp(-x / param2)

        def calculate(self, dt: "DataTable") -> bool:
            """Calculate theory for given dataset.

            Args:
                dt: Dataset to calculate for

            Returns:
                True if successful
            """
            # Extract parameters
            param1 = self.parameters["param1"].value
            param2 = self.parameters["param2"].value
            params = jnp.array([param1, param2])

            # Get data
            x = dt.data[:, 0]
            xdata = jnp.array(x)

            # Compute predictions
            ydata = self.theory_function(params, xdata)

            # Store results
            dt.theory[:, 0] = xdata
            dt.theory[:, 1] = ydata

            return True

Complete Example: Maxwell Model Theory
---------------------------------------

.. code-block:: python

    """Theory: Maxwell Model

    Single Maxwell element for linear viscoelasticity.

    References:
        [1] Maxwell, J.C., Phil. Trans. R. Soc. Lond., 1867
    """

    from __future__ import annotations

    import jax.numpy as jnp
    from jax import jit
    from typing import TYPE_CHECKING

    from RepTate.core.Parameter import Parameter, ParameterType, OptType
    from RepTate.core.Theory import Theory

    if TYPE_CHECKING:
        from RepTate.core.DataTable import DataTable


    class TheoryMaxwell(Theory):
        """Maxwell model for linear viscoelasticity.

        Predicts G' and G'' for a single Maxwell element:
            G' = G * (ω*τ)^2 / (1 + (ω*τ)^2)
            G'' = G * (ω*τ) / (1 + (ω*τ)^2)

        Parameters:
            G (float): Modulus (Pa) (default: 1e5)
            tau (float): Relaxation time (s) (default: 1.0)
        """

        thname = "Maxwell"
        description = "Single Maxwell element"
        citations = "Maxwell, J.C., Phil. Trans. R. Soc. Lond., 1867"

        def __init__(self, name: str = "", parent_dataset=None, axarr=None):
            """Initialize Maxwell theory."""
            super().__init__(name, parent_dataset, axarr)

            # Define parameters
            self.parameters["G"] = Parameter(
                name="G",
                value=1e5,
                description="Modulus (Pa)",
                type=ParameterType.real,
                opt_type=OptType.opt,
                min_value=1e3,
                max_value=1e8,
            )
            self.parameters["tau"] = Parameter(
                name="tau",
                value=1.0,
                description="Relaxation time (s)",
                type=ParameterType.real,
                opt_type=OptType.opt,
                min_value=1e-6,
                max_value=1e3,
            )

            self.update_parameter_table()

        @jit
        @staticmethod
        def maxwell_function(params: jnp.ndarray, omega: jnp.ndarray) -> jnp.ndarray:
            """Compute G' and G'' for Maxwell model.

            Args:
                params: [G, tau]
                omega: Angular frequencies (rad/s)

            Returns:
                Concatenated [G', G''] array
            """
            G, tau = params
            omega_tau_sq = (omega * tau) ** 2

            G_prime = G * omega_tau_sq / (1 + omega_tau_sq)
            G_double_prime = G * omega * tau / (1 + omega_tau_sq)

            return jnp.concatenate([G_prime, G_double_prime])

        def calculate(self, dt: "DataTable") -> bool:
            """Calculate Maxwell model for dataset.

            Args:
                dt: Dataset with frequency data

            Returns:
                True if successful
            """
            # Extract parameters
            G = self.parameters["G"].value
            tau = self.parameters["tau"].value
            params = jnp.array([G, tau])

            # Get frequency data
            omega = dt.data[:, 0]
            omega_jax = jnp.array(omega)

            # Compute predictions
            predictions = self.maxwell_function(params, omega_jax)

            # Split into G' and G''
            n = len(omega)
            G_prime = predictions[:n]
            G_double_prime = predictions[n:]

            # Store results
            dt.theory[:, 0] = omega
            dt.theory[:, 1] = G_prime
            dt.theory[:, 2] = G_double_prime

            return True

Best Practices for New Theories
--------------------------------

**1. Use Type Hints:**

.. code-block:: python

    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from RepTate.core.DataTable import DataTable

    def calculate(self, dt: "DataTable") -> bool:
        ...

**2. Use JAX Arrays:**

.. code-block:: python

    import jax.numpy as jnp

    # Convert to JAX array
    xdata = jnp.array(dt.data[:, 0])

    # Compute (JAX-compatible)
    result = jnp.exp(-xdata / tau)

**3. Use JIT Compilation:**

.. code-block:: python

    from jax import jit

    @jit
    @staticmethod
    def theory_function(params, x):
        """JIT-compiled for performance."""
        return ...

**4. Document Parameters:**

.. code-block:: python

    self.parameters["G"] = Parameter(
        name="G",
        value=1e5,
        description="Modulus (Pa)",  # Clear description
        type=ParameterType.real,
        opt_type=OptType.opt,
        min_value=1e3,                # Sensible bounds
        max_value=1e8,
    )

**5. Add Citations:**

.. code-block:: python

    class TheoryNewModel(Theory):
        """..."""
        citations = "Author et al., Journal Name, Year"

**6. Write Tests:**

.. code-block:: python

    # tests/unit/theories/test_theory_new_model.py

    def test_new_model_basic():
        """Test basic functionality."""
        theory = TheoryNewModel()
        params = jnp.array([1.0, 2.0])
        x = jnp.linspace(0, 10, 100)
        result = theory.theory_function(params, x)
        assert result.shape == x.shape

    def test_new_model_bounds():
        """Test parameter bounds."""
        theory = TheoryNewModel()
        assert theory.parameters["param1"].min_value == 0.0
        assert theory.parameters["param1"].max_value == 100.0

Part 6: Testing Your Changes
=============================

Unit Tests
----------

Write unit tests for new components:

.. code-block:: python

    # tests/unit/theories/test_theory_maxwell.py

    import pytest
    import jax.numpy as jnp
    from RepTate.theories.TheoryMaxwell import TheoryMaxwell

    def test_maxwell_function_basic():
        """Test Maxwell function with known values."""
        G = 1e5
        tau = 1.0
        omega = jnp.array([0.1, 1.0, 10.0])

        params = jnp.array([G, tau])
        result = TheoryMaxwell.maxwell_function(params, omega)

        # Check shape
        assert result.shape == (2 * len(omega),)

        # Check values at ω*τ = 1 (45-degree point)
        idx = 1  # omega = 1.0, tau = 1.0
        G_prime = result[idx]
        G_double_prime = result[idx + len(omega)]

        assert jnp.isclose(G_prime, G / 2, rtol=1e-5)
        assert jnp.isclose(G_double_prime, G / 2, rtol=1e-5)

    def test_maxwell_function_bounds():
        """Test parameter bounds."""
        theory = TheoryMaxwell()

        assert theory.parameters["G"].min_value == 1e3
        assert theory.parameters["G"].max_value == 1e8
        assert theory.parameters["tau"].min_value == 1e-6
        assert theory.parameters["tau"].max_value == 1e3

Regression Tests
----------------

Add regression tests for numerical equivalence:

.. code-block:: python

    # tests/regression/test_maxwell_equivalence.py

    import pytest
    import jax.numpy as jnp
    from RepTate.theories.TheoryMaxwell import TheoryMaxwell

    def test_maxwell_golden_master():
        """Test Maxwell model against golden master values."""
        # Golden master values (pre-computed)
        omega = jnp.array([0.1, 1.0, 10.0])
        params = jnp.array([1e5, 1.0])

        expected_G_prime = jnp.array([990.099, 50000.0, 99009.9])
        expected_G_double_prime = jnp.array([9900.99, 50000.0, 9900.99])

        # Compute
        result = TheoryMaxwell.maxwell_function(params, omega)
        n = len(omega)
        G_prime = result[:n]
        G_double_prime = result[n:]

        # Validate (high precision)
        assert jnp.allclose(G_prime, expected_G_prime, rtol=1e-10)
        assert jnp.allclose(G_double_prime, expected_G_double_prime, rtol=1e-10)

Running Tests
-------------

.. code-block:: bash

    # Run all tests
    pytest tests/

    # Run specific test file
    pytest tests/unit/theories/test_theory_maxwell.py

    # Run with coverage
    pytest --cov=RepTate tests/

    # Run specific marker
    pytest -m "not slow" tests/

Conclusion
==========

This guide covered:

1. JAX-based curve fitting (8-10x faster)
2. SafeSerializer (secure JSON/NPZ format)
3. safe_eval (AST-based expression evaluation)
4. Strangler Fig pattern (incremental refactoring)
5. Adding new theories (modern template)
6. Testing practices

**Next Steps:**

- Read :doc:`MODERNIZATION_ARCHITECTURE` for architectural details
- Read :doc:`testing` for testing best practices
- See ``MODERNIZATION_SUMMARY.md`` for migration status
- Check ``TECHNICAL_DEBT_INVENTORY.md`` for remaining work

**Getting Help:**

- Documentation: ``docs/source/developers/``
- Examples: ``tests/`` directory
- Architecture docs: :doc:`../architecture/overview`

**Common Issues:**

- JAX TracerArrayConversionError → Use jnp.where or lax.cond
- SafeSerializer TypeError → Check supported types
- safe_eval ValueError → Check whitelist
