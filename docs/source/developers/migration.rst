Migration Guide
===============

This guide documents breaking changes from the modernization effort and
how to migrate existing code.

Version: 003-reptate-modernization

Breaking Changes
----------------

1. Pickle Files No Longer Supported
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Before:**

.. code-block:: python

   import pickle
   data = pickle.load(open("mydata.rpt", "rb"))

**After:**

Use the migration script to convert pickle files:

.. code-block:: bash

   python scripts/migrate_pickle_files.py mydata.rpt

Or use the new serialization module:

.. code-block:: python

   from RepTate.core.serialization import safe_load
   data = safe_load("mydata.json")

2. eval() Replaced with safe_eval
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Before:**

.. code-block:: python

   result = eval(user_expression)

**After:**

.. code-block:: python

   from RepTate.core.safe_eval import safe_eval
   result = safe_eval(user_expression, {"x": x_value})

safe_eval supports:

- Basic arithmetic: +, -, *, /, **
- Math functions: sin, cos, exp, log, sqrt
- Comparison operators
- Variable substitution

3. scipy.optimize -> NLSQ
^^^^^^^^^^^^^^^^^^^^^^^^^

**Before:**

.. code-block:: python

   from scipy.optimize import curve_fit
   popt, pcov = curve_fit(model, xdata, ydata)

**After:**

.. code-block:: python

   from RepTate.core.fitting.nlsq_optimize import curve_fit
   popt, pcov = curve_fit(model, xdata, ydata)

Or use the higher-level interface:

.. code-block:: python

   from RepTate.core.fitting.nlsq_fit import run_nlsq_fit
   result, diagnostics = run_nlsq_fit(model, xdata, ydata, p0=p0)

4. scipy.interpolate -> interpax
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Before:**

.. code-block:: python

   from scipy.interpolate import interp1d
   f = interp1d(x, y, kind="cubic")

**After:**

.. code-block:: python

   from interpax import interp1d as jinterp1d
   y_interp = jinterp1d(x_new, x, y, method="cubic")

5. numpy -> jax.numpy
^^^^^^^^^^^^^^^^^^^^^

For numerical code in theories:

**Before:**

.. code-block:: python

   import numpy as np
   result = np.exp(-x / tau)

**After:**

.. code-block:: python

   import jax.numpy as jnp
   result = jnp.exp(-x / tau)

6. QApplicationWindow Components Extracted
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Direct access to internal methods may need updating.

**Before:**

.. code-block:: python

   app.setup_data_inspector_toolbar(toolbar)

**After:**

.. code-block:: python

   app.menu_manager.setup_data_inspector_toolbar(toolbar)

Components:

- ``menu_manager``: Menu/toolbar management
- ``dataset_manager``: Dataset lifecycle
- ``view_coordinator``: View switching
- ``file_io_controller``: File operations

7. QTheory Components Extracted
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Before:**

.. code-block:: python

   theory.calculate_error_standard(y_data, y_theory)

**After:**

.. code-block:: python

   theory.compute.calculate_error_standard(y_data, y_theory)

Components:

- ``compute``: Numerical computation (TheoryCompute)
- ``parameter_controller``: Parameter management
- ``fit_controller``: Fitting orchestration

Feature Flags
-------------

Temporarily disable new features if needed:

.. code-block:: bash

   # Disable safe_eval (use legacy eval)
   export REPTATE_USE_SAFE_EVAL=false

   # Disable safe serialization (use pickle)
   export REPTATE_USE_SAFE_SERIALIZATION=false

   # Disable JAX optimization
   export REPTATE_USE_JAX_OPTIMIZATION=false

These should only be used for debugging migration issues.

Numerical Tolerance
-------------------

JAX operations may produce slightly different results due to FP
operation reordering. Expected tolerance is 1e-10.

If you encounter numerical differences:

1. Verify both implementations are correct
2. Check that float64 precision is enabled
3. Compare against golden file values
4. Use appropriate tolerance in assertions

Getting Help
------------

If you encounter migration issues:

1. Check the GitHub issues
2. Run the test suite to verify behavior
3. Review the architecture documentation

For API changes not documented here, check the source code
docstrings or the test files for usage examples.
