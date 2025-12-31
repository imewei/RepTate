Contributing Guidelines
=======================

This guide explains how to contribute to RepTate.

Development Setup
-----------------

Prerequisites:

- Python 3.12+
- uv (recommended) or pip
- Git

Clone and install:

.. code-block:: bash

   git clone https://github.com/jorge-ramirez-upm/RepTate.git
   cd RepTate
   uv sync  # or: pip install -e ".[dev]"

Verify installation:

.. code-block:: bash

   python -m pytest tests/ -v

Code Style
----------

RepTate follows these conventions:

1. **Formatting**: Use ruff for linting and formatting

   .. code-block:: bash

      ruff check src/
      ruff format src/

2. **Type Hints**: Use explicit type annotations

   .. code-block:: python

      def calculate(self, params: dict[str, float], x: Array) -> Array:
          ...

3. **Imports**: Use explicit imports, avoid star imports

   .. code-block:: python

      # Good
      from jax import numpy as jnp

      # Avoid
      from jax.numpy import *

4. **Docstrings**: Use Google style

   .. code-block:: python

      def calculate_residuals(self, y_data: Array, y_theory: Array) -> Array:
          """Calculate residuals between data and theory.

          Args:
              y_data: Observed data values.
              y_theory: Theoretical predictions.

          Returns:
              Array of residuals (y_data - y_theory).
          """

Testing Requirements
--------------------

All contributions must pass the test suite:

.. code-block:: bash

   # Run all tests
   python -m pytest tests/ -v

   # Run specific test categories
   python -m pytest tests/unit/ -v
   python -m pytest tests/integration/ -v
   python -m pytest tests/regression/ -v

New features must include tests.

Pull Request Process
--------------------

1. Fork the repository
2. Create a feature branch: ``git checkout -b feature/my-feature``
3. Make changes and add tests
4. Run ``ruff check`` and fix any issues
5. Run the full test suite
6. Push and create a pull request

PR checklist:

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] New features have documentation
- [ ] CHANGELOG.md updated if applicable

Feature Flags
-------------

Use feature flags for gradual rollout of new features:

.. code-block:: python

   from RepTate.core.feature_flags import is_enabled

   if is_enabled('MY_NEW_FEATURE'):
       # New implementation
       pass
   else:
       # Legacy implementation
       pass

See ``src/RepTate/core/feature_flags.py`` for examples.

Security Considerations
-----------------------

Never commit:

- API keys or credentials
- Pickle files (use JSON/NPZ instead)
- eval() on user input (use safe_eval)

See :doc:`testing` for security testing guidelines.
