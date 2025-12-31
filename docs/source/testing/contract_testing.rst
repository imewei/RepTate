Contract Testing Framework
===========================

Overview
--------

Contract testing validates **integration agreements between components**. Unlike unit tests that test components in isolation, contract tests ensure that:

- APIs conform to documented specifications
- Data structures maintain required fields and types
- Performance stays within acceptable bounds
- Migrations preserve numerical equivalence
- Systems remain interoperable

RepTate implements four categories of contracts:

1. **API Contracts**: Function signatures, parameter types, return types
2. **Data Contracts**: Field names, data types, null handling, constraints
3. **Performance Contracts**: Execution time budgets, baseline measurements
4. **Compatibility Contracts**: Migration equivalence guarantees


Quick Start
-----------

Run all contract tests:

.. code-block:: bash

    pytest tests/contracts/ -v

Run specific contract category:

.. code-block:: bash

    pytest tests/contracts/test_api_contracts.py -v
    pytest tests/contracts/test_data_contracts.py -v
    pytest tests/contracts/test_performance_contracts.py -v
    pytest tests/contracts/test_compatibility_contracts.py -v


API Contracts
-------------

API contracts validate that functions accept the correct types and return the expected results.

Example: ``Theory.calculate()`` Contract
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The contract specifies:

.. code-block:: python

    # Contract: Theory.calculate(params, x)
    # - params: dict[str, float] with theory-specific parameter names
    # - x: Array with shape (n,)
    # Returns: Array with same shape as x

    def test_calculate_accepts_dict_params(mock_theory, synthetic_frequency_data):
        params = {"slope": 2.0, "intercept": 1.0}
        x = synthetic_frequency_data.x
        result = mock_theory.calculate(params, x)

        assert isinstance(result, Array)
        assert result.shape == x.shape

Consumers depend on this contract. If you modify ``calculate()``, update tests first.

Theory Parameter Contract
^^^^^^^^^^^^^^^^^^^^^^^^^

Theories must provide parameters with:

- ``name``: str (unique within theory)
- ``value``: float (within bounds)
- ``min_value``, ``max_value``: bounds
- ``opt_type``: "opt" or "var"

.. code-block:: python

    theory_params = theory.get_parameters()
    for param in theory_params.values():
        assert param.min_value <= param.value <= param.max_value


Dataset Data Access Contract
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Datasets must provide:

.. code-block:: python

    # Required methods
    x = dataset.get_x()  # Array, shape (n,), float64
    y = dataset.get_y()  # Array, shape (n,), float64
    col = dataset.get_column("name")  # Array for any column
    error = dataset.get_error()  # Array or None


Data Contracts
--------------

Data contracts define the structure and constraints on data flowing between components.

Dataset Structure Contract
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Contract: Dataset has x and y data with matching shapes
    contract = {
        "required_fields": ["x", "y"],
        "optional_fields": ["error", "metadata"],
        "x_requirements": {
            "type": "array",
            "dtype": "float64",
            "shape": "(n,)",
            "strictly_increasing": True,
        },
        "y_requirements": {
            "type": "array",
            "dtype": "float64",
            "shape": "(n,)",
        },
    }

Test this contract:

.. code-block:: python

    def test_dataset_x_y_match(dataset):
        x = dataset.get_x()
        y = dataset.get_y()
        assert len(x) == len(y), "x and y must have same length"
        assert x.dtype == y.dtype, "x and y must have same dtype"


Calculation Output Contract
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Theory calculations must produce:

- Same shape as input
- float64 dtype
- Finite values (no NaN/Inf unless documented)
- Deterministic results

.. code-block:: python

    def test_calculation_output(theory, x):
        params = {"G0": 1e5, "tau": 1.0}
        y1 = theory.calculate(params, x)
        y2 = theory.calculate(params, x)

        assert y1.shape == x.shape
        assert y1.dtype == y2.dtype
        assert jnp.allclose(y1, y2)  # Deterministic


Serialization Contract
^^^^^^^^^^^^^^^^^^^^^^

Data serialized with JSON/NPZ format:

- Metadata in JSON (text, human-readable)
- Arrays in NPZ (binary, efficient)
- No pickle (security)
- Round-trip preserves values

.. code-block:: python

    from RepTate.core.serialization import SafeSerializer

    # Save: dict with metadata + arrays
    test_data = {
        "name": "dataset_001",
        "x": np.array([...]),
        "y": np.array([...]),
    }
    result = SafeSerializer.save(Path("output"), test_data)

    # Load: gets same structure back
    loaded = SafeSerializer.load(Path("output"))
    assert loaded["name"] == test_data["name"]
    np.testing.assert_array_equal(loaded["x"], test_data["x"])


Performance Contracts
---------------------

Performance contracts establish time budgets for operations and detect regressions.

NLSQ Curve Fitting Baselines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Expected performance:

- Linear model (100 points): < 1 second
- Maxwell model (100 points): < 2 seconds
- Multi-parameter fit (100 points): < 5 seconds

Test:

.. code-block:: python

    def test_linear_fit_performance(synthetic_frequency_data):
        from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

        xdata = synthetic_frequency_data.x
        ydata = 2.5 * xdata + 1.5
        p0 = jnp.array([1.0, 0.0])

        start = time.perf_counter()
        result, _ = run_nlsq_fit(linear_model, xdata, ydata, p0=p0)
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, f"Fit took {elapsed:.2f}s"


Theory Calculation Baselines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Expected performance:

- Single calculation: < 1ms
- Vectorized (100 points): < 100ms
- Vectorized (1000 points): < 1000ms

The performance must scale subquadratically with data size.


Regression Detection
^^^^^^^^^^^^^^^^^^^^

Use ``RegressionDetector`` to check baselines:

.. code-block:: python

    from tests.contracts.test_baseline_storage import RegressionDetector

    detector = RegressionDetector(baseline_manager)
    passed = detector.check_and_record(
        suite="nlsq_fitting",
        operation="linear_fit",
        func=run_nlsq_fit,
        model, xdata, ydata, p0=p0,
    )

    if not passed:
        print(detector.get_regression_report())


Compatibility Contracts
-----------------------

Compatibility contracts ensure that migrations preserve behavior.

SciPy → JAX Numerical Equivalence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

JAX implementations must produce numerically equivalent results to SciPy:

.. code-block:: python

    def test_exponential_equivalence():
        x_np = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        x_jax = jnp.array(x_np)

        # Same results within floating point tolerance
        assert_array_almost_equal(
            np.array(jnp.exp(x_jax)),
            np.exp(x_np),
            decimal=10,
        )

    def test_trigonometric_equivalence():
        x_np = np.linspace(0, 2*np.pi, 100)
        x_jax = jnp.array(x_np)

        assert_array_almost_equal(
            np.array(jnp.sin(x_jax)),
            np.sin(x_np),
            decimal=10,
        )


Pickle → SafeSerializer Compatibility
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Old pickle-based data must be readable/writable in new format:

.. code-block:: python

    def test_safe_serializer_round_trip(temp_workspace):
        from RepTate.core.serialization import SafeSerializer

        test_data = {
            "x": np.array([0.1, 1.0, 10.0]),
            "y": np.array([100, 1000, 10000]),
        }

        # Save in new format
        SafeSerializer.save(temp_workspace / "data", test_data)

        # Load back
        loaded = SafeSerializer.load(temp_workspace / "data")

        # Same data
        np.testing.assert_array_equal(loaded["x"], test_data["x"])


Numerical Precision Preservation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

JAX x64 mode must preserve double precision:

.. code-block:: python

    def test_double_precision_math():
        import jax
        assert jax.config.jax_enable_x64

        x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
        y = jnp.sin(x)

        assert y.dtype == jnp.float64


CI/CD Integration
-----------------

Contract tests run automatically in GitHub Actions:

.. code-block:: bash

    .github/workflows/contract-tests.yml

The workflow:

1. Runs on every push and pull request
2. Enforces API contracts (blocks merge if violated)
3. Checks compatibility (warnings if violated)
4. Measures performance (detects regressions)
5. Uploads coverage report

Quality Gates
^^^^^^^^^^^^^

Merge blocked if:

- **API contracts fail**: Component interface violated
- **Data contracts fail**: Data structure requirements violated
- **Fit precision contracts fail**: Accuracy degraded

Warnings (non-blocking):

- **Performance regression**: Significant slowdown detected
- **Compatibility failures**: Migration issues detected


Best Practices
--------------

1. **Specify contracts before implementation**: Write contract tests first
2. **Keep contracts explicit**: Document assumptions clearly
3. **Use fixtures for common data**: Reuse synthetic data across tests
4. **Record baselines regularly**: Capture current performance
5. **Review regression reports**: Understand performance changes
6. **Update contracts carefully**: Changing contracts requires careful review

Example Workflow
^^^^^^^^^^^^^^^^

To add a new contract:

1. Create a new test file in ``tests/contracts/``
2. Import fixtures from ``conftest.py``
3. Write tests that validate the contract
4. Document the contract in docstrings
5. Add to ``pytest.ini`` if needed
6. Run: ``pytest tests/contracts/new_test.py -v``
7. Commit test file with implementation


Troubleshooting
---------------

Tests fail with "contract violation"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This means a component changed its interface or behavior in a breaking way.

Steps:

1. Review the test message - what contract was violated?
2. Check if change is intentional
3. If intentional, update the contract test
4. If unintentional, fix the component
5. Document why contract changed (in commit message)

Performance regression detected
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Check if performance degradation is expected (new feature, refactoring):

.. code-block:: bash

    # View current baselines
    cat .baselines/*/

    # If expected, update baseline
    pytest tests/contracts/test_performance_contracts.py \
        --update-baseline

    # Commit new baseline
    git add .baselines/
    git commit -m "Update performance baselines for <reason>"


References
----------

- :doc:`/testing/overview` - Testing strategy overview
- :doc:`/testing/unit_tests` - Unit testing guide
- :doc:`/testing/integration_tests` - Integration testing guide
- :doc:`/architecture/interfaces` - Component interface definitions

See Also
--------

- `Consumer-Driven Contract Testing <https://martinfowler.com/articles/consumerDrivenContracts.html>`_
- `Pact Framework <https://pact.foundation/>`_ - Contract testing for microservices
- `Contract Testing Guide <https://contracttestingguide.com/>`_
