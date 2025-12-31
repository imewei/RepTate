Architecture Overview
=====================

This document provides a high-level overview of RepTate's architecture after
the modernization effort (003-reptate-modernization).

Layer Structure
---------------

RepTate follows a layered architecture:

1. **GUI Layer** (``src/RepTate/gui/``)

   - PySide6/Qt-based user interface
   - Decomposed into focused components (<500 LOC each)
   - Components: MenuManager, DatasetManager, ViewCoordinator, FileIOController

2. **Application Layer** (``src/RepTate/applications/``)

   - Domain-specific containers (LVE, LAOS, NLVE, etc.)
   - Manages datasets, views, and theories for each rheology type

3. **Theory Layer** (``src/RepTate/theories/``)

   - Rheological theory implementations
   - Pure computational code using JAX
   - Each theory implements ``calculate()`` method

4. **Core Layer** (``src/RepTate/core/``)

   - Shared infrastructure
   - DataTable, Parameter, View definitions
   - Fitting infrastructure (NLSQ-based)
   - Protocol interfaces for type safety

Key Components
--------------

Data Flow
^^^^^^^^^

.. code-block:: text

   User Input → GUI Layer → Application → Theory → Computation
                    ↑                              ↓
                    └──────── Result Display ←─────┘

The data flows from user input through the GUI to applications, which
coordinate theories for computation. Results flow back for visualization.

Protocol Interfaces
^^^^^^^^^^^^^^^^^^^

Located in ``src/RepTate/core/interfaces.py``:

- ``ITheory``: Standard interface for all theory implementations
- ``IApplication``: Interface for application containers
- ``IDataset``: Interface for experimental data access
- ``IFitResult``: Interface for fitting results

These protocols enable:

- Type-safe interactions between layers
- Easy testing with mock implementations
- Clear module boundaries

Numerical Computation
^^^^^^^^^^^^^^^^^^^^^

All numerical computation uses JAX:

- ``jax.numpy`` replaces NumPy for array operations
- ``nlsq`` provides curve fitting (replaces scipy.optimize)
- ``interpax`` provides interpolation (replaces scipy.interpolate)
- NumPyro enables Bayesian inference with MCMC

Feature Flags
^^^^^^^^^^^^^

Located in ``src/RepTate/core/feature_flags.py``:

- ``USE_SAFE_EVAL``: Use secure expression evaluator
- ``USE_SAFE_SERIALIZATION``: Use JSON/NPZ instead of pickle
- ``USE_JAX_OPTIMIZATION``: Use JAX-based optimization

Environment variables can override defaults (e.g., ``REPTATE_USE_SAFE_EVAL=false``).

Extracted Components
--------------------

QApplicationWindow Decomposition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The original ~3000 LOC ``QApplicationWindow`` was decomposed into:

- ``MenuManager`` (266 LOC): Menu and toolbar setup
- ``DatasetManager`` (275 LOC): Dataset lifecycle management
- ``ViewCoordinator`` (252 LOC): View switching and multiplot
- ``FileIOController`` (349 LOC): File I/O operations

QTheory Decomposition
^^^^^^^^^^^^^^^^^^^^^

The original ~2300 LOC ``QTheory`` was decomposed into:

- ``TheoryCompute`` (324 LOC): Pure numerical computation
- ``ParameterController`` (298 LOC): Parameter management
- ``FitController`` (40 LOC): Fitting orchestration

Testing Strategy
----------------

Test Categories
^^^^^^^^^^^^^^^

1. **Unit Tests** (``tests/unit/``): Test individual components in isolation
2. **Integration Tests** (``tests/integration/``): Test component interactions
3. **Regression Tests** (``tests/regression/``): Guard against numerical changes
4. **Characterization Tests** (``tests/characterization/``): Document existing behavior

Coverage
^^^^^^^^

- Core modules: 31%+ coverage
- All extracted components have dedicated unit tests
- 464+ tests total

See Also
--------

- :doc:`dependencies` - Module dependency diagram
- :doc:`data_flow` - Detailed data flow documentation
- ``tests/`` - Test implementations
