Testing Guide
=============

This guide explains RepTate's testing strategy and how to write tests.

Test Categories
---------------

Unit Tests
^^^^^^^^^^

Location: ``tests/unit/``

Test individual components in isolation.

.. code-block:: python

   class TestParameterController:
       def test_get_parameter_value(self):
           controller = ParameterController(parameters={"a": mock_param})
           value = controller.get_parameter_value("a")
           assert value == expected

Run:

.. code-block:: bash

   python -m pytest tests/unit/ -v

Integration Tests
^^^^^^^^^^^^^^^^^

Location: ``tests/integration/``

Test component interactions and workflows.

.. code-block:: python

   class TestFitWorkflow:
       def test_end_to_end_fit(self):
           # Setup data
           # Run fit
           # Verify results

Run:

.. code-block:: bash

   python -m pytest tests/integration/ -v

Regression Tests
^^^^^^^^^^^^^^^^

Location: ``tests/regression/``

Guard against unintended numerical changes.

.. code-block:: python

   class TestNumericalEquivalence:
       def test_jax_numpy_match(self):
           x_np = np.array([1.0, 2.0, 3.0])
           x_jax = jnp.array([1.0, 2.0, 3.0])
           assert_array_almost_equal(np.array(x_jax), x_np)

Run:

.. code-block:: bash

   python -m pytest tests/regression/ -v

Characterization Tests
^^^^^^^^^^^^^^^^^^^^^^

Location: ``tests/characterization/``

Document existing behavior of complex classes.

Run:

.. code-block:: bash

   python -m pytest tests/characterization/ -v

Running Tests
-------------

All Tests
^^^^^^^^^

.. code-block:: bash

   python -m pytest tests/ -v

With Coverage
^^^^^^^^^^^^^

.. code-block:: bash

   python -m pytest tests/ --cov=src/RepTate --cov-report=html

Parallel Execution
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python -m pytest tests/ -n auto

JAX Configuration
-----------------

Tests use float64 precision for numerical accuracy:

.. code-block:: python

   # tests/conftest.py
   @pytest.fixture(scope="session", autouse=True)
   def enable_jax_x64():
       jax.config.update("jax_enable_x64", True)
       yield

Golden File Testing
-------------------

Location: ``tests/regression/golden/``

Compare against known-good values:

.. code-block:: python

   def test_maxwell_against_golden():
       golden = np.load("tests/regression/golden/maxwell_model.npz")
       result = compute_maxwell(golden["omega"])
       assert_array_almost_equal(result, golden["G_prime"], decimal=10)

Writing New Tests
-----------------

Template
^^^^^^^^

.. code-block:: python

   """Unit tests for MyComponent.

   Tests cover:
   - T0XX: Test task reference from tasks.md
   """
   from __future__ import annotations

   import pytest
   import jax.numpy as jnp

   class TestMyComponent:
       """Test MyComponent functionality."""

       def test_basic_operation(self):
           """Test basic operation description."""
           # Arrange
           component = MyComponent()

           # Act
           result = component.do_something()

           # Assert
           assert result == expected

Best Practices
^^^^^^^^^^^^^^

1. One assertion per test (when possible)
2. Use descriptive test names
3. Use fixtures for common setup
4. Mock external dependencies
5. Test edge cases and error conditions

Mocking Qt
^^^^^^^^^^

For GUI tests, mock PySide6 components:

.. code-block:: python

   from unittest.mock import MagicMock, patch

   def test_menu_setup():
       with patch("RepTate.gui.MenuManager.QAction"):
           manager = MenuManager(parent=MagicMock())
           # Test without actual Qt widgets

Performance Testing
-------------------

Ensure no significant regressions:

.. code-block:: python

   import time

   def test_fit_performance():
       start = time.perf_counter()
       run_nlsq_fit(model, x, y, p0=p0)
       elapsed = time.perf_counter() - start

       assert elapsed < 1.0  # Must complete in under 1 second
