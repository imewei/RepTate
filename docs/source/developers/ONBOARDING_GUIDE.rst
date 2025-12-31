=============================================
RepTate Developer Onboarding Guide
=============================================

Welcome to the RepTate project! This guide will help you get up to speed quickly with the codebase, development workflow, and best practices.

.. contents:: Table of Contents
   :local:
   :depth: 3

Introduction
============

RepTate (Rheology of Entangled Polymers: Toolkit for Analysis of Theory and Experiment) is a scientific application for analyzing polymer rheology data. It provides tools for curve fitting, Bayesian inference, and theoretical model comparison.

**Key Technologies:**

- Python 3.12+
- JAX (numerical computing with JIT compilation)
- NumPyro (Bayesian inference)
- PySide6 (Qt6 GUI framework)
- NLSQ (JAX-based curve fitting)

**Project Status:**

- **Modernization:** 65% complete (on track)
- **Security:** Zero vulnerabilities (pickle/eval eliminated)
- **Performance:** 8-15x faster than legacy (JAX adoption)

Quick Start (30 Minutes)
=========================

Prerequisites
-------------

**Required:**

- Python 3.12 or 3.13
- Git
- 4GB+ RAM
- Linux, macOS, or Windows

**Optional:**

- CUDA toolkit (for GPU acceleration)
- Visual Studio Code (recommended IDE)

Installation
------------

1. **Clone repository:**

   .. code-block:: bash

       git clone https://github.com/jorge-ramirez-upm/RepTate.git
       cd RepTate

2. **Create virtual environment:**

   .. code-block:: bash

       # Using uv (recommended)
       uv venv
       source .venv/bin/activate  # Linux/macOS
       # or
       .venv\Scripts\activate     # Windows

       # Or using standard venv
       python3.12 -m venv .venv
       source .venv/bin/activate

3. **Install dependencies:**

   .. code-block:: bash

       # Using uv (recommended)
       uv pip install -e ".[dev]"

       # Or using pip
       pip install -e ".[dev]"

4. **Verify installation:**

   .. code-block:: bash

       # Run tests
       pytest tests/ -v

       # Start RepTate
       python -m RepTate

First Day: Understanding the Codebase
======================================

Project Structure (15 minutes)
-------------------------------

.. code-block:: text

    RepTate/
    │
    ├── src/RepTate/               # Source code
    │   ├── gui/                   # GUI layer (PySide6)
    │   │   ├── controllers/       # Business logic (MVVM)
    │   │   ├── viewmodels/        # State management
    │   │   ├── widgets/           # Reusable UI components
    │   │   └── QApplication*.py   # Legacy god classes (being decomposed)
    │   │
    │   ├── core/                  # Domain logic
    │   │   ├── fitting/           # JAX-based optimization
    │   │   ├── inference/         # Bayesian inference (NumPyro)
    │   │   ├── bayes/             # Prior definitions
    │   │   ├── serialization.py   # Safe JSON/NPZ format
    │   │   ├── safe_eval.py       # AST-based expression evaluation
    │   │   └── feature_flags.py   # Gradual rollout control
    │   │
    │   ├── applications/          # Domain-specific apps
    │   │   └── Application*.py    # LVE, NLVE, TTS, MWD, etc.
    │   │
    │   ├── theories/              # Mathematical models
    │   │   └── Theory*.py         # Maxwell, RoliePoly, BoB, etc.
    │   │
    │   └── tools/                 # Data processing utilities
    │       └── Tool*.py           # Integral, Smooth, Bounds, etc.
    │
    ├── tests/                     # Test suite
    │   ├── unit/                  # Unit tests
    │   ├── integration/           # Integration tests
    │   ├── regression/            # Golden master tests
    │   └── characterization/      # Legacy behavior capture
    │
    ├── docs/                      # Documentation
    │   └── source/
    │       ├── architecture/      # Architecture docs
    │       └── developers/        # Developer guides
    │
    ├── scripts/                   # Utility scripts
    │   ├── migrate_pickle_files.py
    │   └── verify_scipy_removal.py
    │
    └── pyproject.toml             # Project configuration

Key Files to Read (15 minutes)
-------------------------------

**Start Here:**

1. ``README.rst`` - Project overview
2. ``CLAUDE.md`` - Development guidelines
3. ``MODERNIZATION_SUMMARY.md`` - Migration status (quick reference)

**Core Infrastructure:**

4. ``src/RepTate/core/serialization.py`` - Safe JSON/NPZ serialization
5. ``src/RepTate/core/safe_eval.py`` - Secure expression evaluation
6. ``src/RepTate/core/feature_flags.py`` - Feature flag system

**Architecture:**

7. ``docs/source/developers/MODERNIZATION_ARCHITECTURE.rst`` - Comprehensive architecture
8. ``docs/source/architecture/overview.rst`` - High-level overview

Running Your First Test (10 minutes)
-------------------------------------

.. code-block:: bash

    # Run all tests
    pytest tests/ -v

    # Run specific test file
    pytest tests/unit/core/test_serialization.py -v

    # Run specific test
    pytest tests/unit/core/test_serialization.py::test_save_and_load_basic -v

    # Run with coverage
    pytest --cov=RepTate tests/

**Expected Output:**

.. code-block:: text

    ======================= test session starts =======================
    collected 164 items

    tests/unit/core/test_serialization.py::test_save_and_load_basic PASSED
    tests/unit/core/test_serialization.py::test_save_with_arrays PASSED
    ...

    ===================== 164 passed in 5.23s ======================

First Week: Making Your First Contribution
===========================================

Development Workflow
--------------------

**1. Create a Branch:**

.. code-block:: bash

    # For features
    git checkout -b feature/your-feature-name

    # For bug fixes
    git checkout -b fix/bug-description

**2. Make Changes:**

Follow coding standards:

.. code-block:: python

    # Use type hints
    def process_data(data: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Process data with threshold.

        Args:
            data: Input data array
            threshold: Processing threshold (default: 0.5)

        Returns:
            Processed data array
        """
        return data[data > threshold]

    # Use JAX for numerical operations
    import jax.numpy as jnp
    from jax import jit

    @jit
    def compute_modulus(params: jnp.ndarray, omega: jnp.ndarray) -> jnp.ndarray:
        """Compute modulus (JAX-compiled)."""
        G, tau = params
        return G * (omega * tau) ** 2 / (1 + (omega * tau) ** 2)

**3. Write Tests:**

.. code-block:: python

    # tests/unit/test_your_feature.py

    import pytest
    import jax.numpy as jnp

    def test_compute_modulus_basic():
        """Test modulus computation with known values."""
        params = jnp.array([1e5, 1.0])
        omega = jnp.array([0.1, 1.0, 10.0])

        result = compute_modulus(params, omega)

        # Verify shape
        assert result.shape == omega.shape

        # Verify value at ω*τ = 1
        assert jnp.isclose(result[1], 5e4, rtol=1e-5)

**4. Run Tests:**

.. code-block:: bash

    # Run your tests
    pytest tests/unit/test_your_feature.py -v

    # Run all tests
    pytest tests/ -v

    # Run linting
    ruff check .

**5. Commit Changes:**

.. code-block:: bash

    git add .
    git commit -m "Add feature: description

    - Detail 1
    - Detail 2

    Fixes #123"

**6. Push and Create PR:**

.. code-block:: bash

    git push origin feature/your-feature-name

    # Create PR on GitHub
    # Add description, link to issue

Common Development Tasks
------------------------

**Task 1: Add a New Theory**

See :doc:`MIGRATION_GUIDE_DETAILED` Part 5 for complete template.

Quick example:

.. code-block:: python

    from RepTate.core.Theory import Theory
    from RepTate.core.Parameter import Parameter, ParameterType, OptType
    import jax.numpy as jnp
    from jax import jit

    class TheoryNewModel(Theory):
        """New theory implementation."""

        thname = "NewModel"
        description = "Brief description"
        citations = "Author et al., Year"

        def __init__(self, name="", parent_dataset=None, axarr=None):
            super().__init__(name, parent_dataset, axarr)

            self.parameters["param1"] = Parameter(
                name="param1",
                value=1.0,
                description="Parameter description",
                type=ParameterType.real,
                opt_type=OptType.opt,
                min_value=0.0,
                max_value=100.0,
            )

        @jit
        @staticmethod
        def theory_function(params, x):
            """Compute predictions (JAX-compatible)."""
            param1 = params[0]
            return param1 * jnp.exp(-x)

**Task 2: Fix a Bug**

1. Reproduce the bug (write a failing test)
2. Fix the bug
3. Verify the test passes
4. Run full test suite
5. Commit with "Fix: description"

**Task 3: Improve Performance**

1. Profile the code (identify bottleneck)
2. Optimize (use JAX JIT compilation)
3. Benchmark (measure improvement)
4. Add regression test
5. Document performance gain in commit message

Code Style and Best Practices
------------------------------

**Python Style:**

- Follow PEP 8
- Use type hints (PEP 484)
- Use descriptive variable names
- Keep functions small (<50 LOC)
- Use docstrings (Google style)

**JAX Style:**

- Use ``@jit`` for performance-critical functions
- Use pure functions (no side effects)
- Use ``jnp.where`` instead of ``if/else``
- Use ``lax.fori_loop`` instead of Python loops

**Git Commit Messages:**

.. code-block:: text

    <type>: <subject> (max 50 chars)

    <body> (optional, wrap at 72 chars)

    - Detail 1
    - Detail 2

    Fixes #<issue-number>

**Types:**

- ``feat``: New feature
- ``fix``: Bug fix
- ``docs``: Documentation
- ``style``: Formatting
- ``refactor``: Code restructuring
- ``test``: Adding tests
- ``chore``: Maintenance

First Month: Advanced Topics
=============================

Understanding Modernization Strategy
-------------------------------------

RepTate is in the middle of a modernization effort. Read these to understand the approach:

1. **Strangler Fig Pattern:**

   - Extract new components alongside legacy code
   - Gradually delegate responsibilities
   - Maintain both paths during migration

2. **Feature Flags:**

   - Control which implementation is active
   - Instant rollback via environment variables
   - Gradual rollout (developers → beta → all users)

3. **Testing Strategy:**

   - Unit tests (fast feedback)
   - Integration tests (component interactions)
   - Regression tests (golden masters, numerical equivalence)
   - Characterization tests (capture legacy behavior)

**Reading:**

- :doc:`MODERNIZATION_ARCHITECTURE` - Complete architecture documentation
- :doc:`LESSONS_LEARNED` - What worked, challenges, recommendations
- :doc:`RUNBOOKS_DUAL_SYSTEM` - Operational procedures

Contributing to Modernization
------------------------------

**Opportunities:**

1. **Complete SciPy Removal (2-3 days):**

   - Migrate 6 remaining files to JAX
   - See ``scripts/verify_scipy_removal.py``

2. **God Class Decomposition (ongoing):**

   - Extract responsibilities from QApplicationWindow, QTheory, QDataSet
   - Follow Strangler Fig pattern
   - See extracted controllers in ``gui/`` for examples

3. **Increase Test Coverage (ongoing):**

   - Current: 65-70%, Target: 80%+
   - Focus on GUI integration tests
   - Add characterization tests for legacy behavior

4. **Documentation (ongoing):**

   - Update architecture docs
   - Add code examples
   - Improve API documentation

**Getting Started:**

1. Pick a task from GitHub Issues
2. Read relevant documentation
3. Ask questions (GitHub Discussions or Issues)
4. Submit PR with tests and documentation

Working with JAX
----------------

**Common Patterns:**

**1. JIT Compilation:**

.. code-block:: python

    from jax import jit

    @jit  # Compile for performance
    def fast_function(x):
        return jnp.exp(-x)

    # First call: slow (compilation)
    result1 = fast_function(jnp.array([1.0]))  # ~1-2s

    # Subsequent calls: fast (compiled)
    result2 = fast_function(jnp.array([2.0]))  # ~0.001s

**2. Conditional Logic:**

.. code-block:: python

    # BAD: Python if/else
    @jit
    def bad_function(x):
        if x > 0:  # ERROR: TracerArrayConversionError
            return x ** 2
        else:
            return -x

    # GOOD: jnp.where
    @jit
    def good_function(x):
        return jnp.where(x > 0, x ** 2, -x)

**3. Loops:**

.. code-block:: python

    # BAD: Python for loop
    @jit
    def bad_loop(n):
        result = 0
        for i in range(n):  # Slow with JIT
            result += i ** 2
        return result

    # GOOD: lax.fori_loop
    from jax import lax

    @jit
    def good_loop(n):
        def body_fun(i, carry):
            return carry + i ** 2
        return lax.fori_loop(0, n, body_fun, 0)

**4. Random Numbers:**

.. code-block:: python

    # BAD: numpy random
    import numpy as np
    random_values = np.random.randn(100)  # Non-deterministic

    # GOOD: JAX random with explicit key
    import jax
    key = jax.random.PRNGKey(42)  # Deterministic
    random_values = jax.random.normal(key, shape=(100,))

Working with Feature Flags
---------------------------

**Check if feature is enabled:**

.. code-block:: python

    from RepTate.core.feature_flags import is_enabled

    if is_enabled('USE_SAFE_EVAL'):
        from RepTate.core.safe_eval import safe_eval
        result = safe_eval(expr, variables)
    else:
        # Legacy fallback
        result = eval(expr, {}, variables)

**Override for testing:**

.. code-block:: bash

    # Disable a feature
    REPTATE_USE_SAFE_EVAL=false python -m RepTate

    # Enable a feature
    REPTATE_USE_SAFE_EVAL=true python -m RepTate

**Available flags:**

- ``USE_SAFE_EVAL`` (default: True) - Safe expression evaluator
- ``USE_SAFE_SERIALIZATION`` (default: True) - JSON/NPZ serialization
- ``USE_JAX_OPTIMIZATION`` (default: True) - JAX-based optimization

Testing Best Practices
-----------------------

**Unit Tests:**

.. code-block:: python

    # tests/unit/test_module.py

    import pytest
    import jax.numpy as jnp

    def test_function_basic():
        """Test basic functionality."""
        result = function_under_test(input_data)
        assert result.shape == expected_shape

    def test_function_edge_cases():
        """Test edge cases."""
        # Zero input
        result = function_under_test(jnp.zeros(10))
        assert jnp.all(result == 0)

        # Negative input
        with pytest.raises(ValueError):
            function_under_test(jnp.array([-1.0]))

**Regression Tests:**

.. code-block:: python

    # tests/regression/test_numerical_equivalence.py

    def test_legacy_vs_modern_equivalence():
        """Validate modern implementation matches legacy."""
        # Run legacy
        legacy_result = legacy_function(test_input)

        # Run modern
        modern_result = modern_function(test_input)

        # Validate equivalence (high precision)
        assert jnp.allclose(modern_result, legacy_result, rtol=1e-10)

**Characterization Tests:**

.. code-block:: python

    # tests/characterization/test_legacy_behavior.py

    def test_legacy_behavior_golden_master():
        """Capture existing behavior before refactoring."""
        result = legacy_function(test_input)

        # First run: approve the output (pytest-snapshot)
        # Subsequent runs: fail if behavior changes
        assert result == snapshot("legacy_behavior")

Resources and Support
=====================

Documentation
-------------

**Architecture:**

- :doc:`../architecture/overview` - High-level overview
- :doc:`MODERNIZATION_ARCHITECTURE` - Comprehensive architecture

**Developer Guides:**

- :doc:`MIGRATION_GUIDE_DETAILED` - Using modern infrastructure
- :doc:`RUNBOOKS_DUAL_SYSTEM` - Operational procedures
- :doc:`LESSONS_LEARNED` - Best practices and recommendations

**Quick References:**

- ``MODERNIZATION_SUMMARY.md`` - Migration status
- ``TECHNICAL_DEBT_INVENTORY.md`` - Detailed technical debt analysis
- ``COMPONENT_READINESS_MATRIX.md`` - Component readiness scores

Getting Help
------------

**Questions:**

- GitHub Discussions: https://github.com/jorge-ramirez-upm/RepTate/discussions
- GitHub Issues: https://github.com/jorge-ramirez-upm/RepTate/issues

**Code Review:**

- Submit PR for review
- Tag relevant reviewers
- Be patient (may take 1-3 days)

**Pair Programming:**

- Ask in GitHub Discussions
- Schedule pairing session with experienced developer

Useful Commands
---------------

.. code-block:: bash

    # Run tests
    pytest tests/ -v                          # All tests
    pytest tests/unit/ -v                     # Unit tests only
    pytest tests/regression/ -v               # Regression tests only
    pytest -m "not slow" tests/               # Skip slow tests

    # Run with coverage
    pytest --cov=RepTate --cov-report=html tests/

    # Linting
    ruff check .                              # Check code style
    ruff format .                             # Format code
    mypy src/RepTate/                         # Type checking

    # Start RepTate
    python -m RepTate                         # Normal mode
    REPTATE_USE_SAFE_EVAL=false python -m RepTate  # Legacy eval

    # Migration
    python scripts/migrate_pickle_files.py data.pkl    # Migrate pickle
    python scripts/verify_scipy_removal.py             # Verify SciPy removal

Next Steps
==========

**Week 1:**

1. Read architecture documentation
2. Run tests locally
3. Pick a "good first issue" from GitHub

**Week 2-4:**

1. Make first contribution (bug fix or small feature)
2. Review others' PRs
3. Ask questions (GitHub Discussions)

**Month 2+:**

1. Take on larger tasks (new theory, god class decomposition)
2. Participate in design discussions
3. Help onboard new developers

Welcome to the RepTate team! We're excited to have you contribute to this modernization effort.
