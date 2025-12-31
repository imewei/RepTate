# RepTate Contract Testing Suite

## What Are Contract Tests?

Contract tests validate **integration agreements** between components. They ensure that:

- APIs conform to documented signatures and behavior
- Data structures maintain required fields and constraints
- Performance stays within acceptable bounds
- Migrations preserve numerical equivalence
- Systems remain interoperable

Unlike unit tests (testing components in isolation), contract tests verify that components work correctly **together**.

## Structure

```
tests/contracts/
├── README.md                                    (This file)
├── __init__.py                                  (Package marker)
├── conftest.py                                  (Shared fixtures & contract definitions)
├── test_api_contracts.py                        (API interface contracts - 35 tests)
├── test_data_contracts.py                       (Data structure contracts - 34 tests)
├── test_performance_contracts.py                (Performance baselines - 15 tests)
├── test_compatibility_contracts.py              (Migration equivalence - 10 tests)
└── test_baseline_storage.py                     (Baseline management system)
```

## Quick Start

### Run All Contracts

```bash
# From project root
pytest tests/contracts/ -v
```

### Run Specific Category

```bash
# API contracts (function signatures, parameter types)
pytest tests/contracts/test_api_contracts.py -v

# Data contracts (structure, fields, types)
pytest tests/contracts/test_data_contracts.py -v

# Performance contracts (time budgets, regression detection)
pytest tests/contracts/test_performance_contracts.py -v

# Compatibility contracts (SciPy→JAX, Pickle→SafeSerializer)
pytest tests/contracts/test_compatibility_contracts.py -v
```

### With Coverage

```bash
pytest tests/contracts/ \
  --cov=src/RepTate \
  --cov-report=html \
  --cov-report=term-missing
```

## Contract Categories

### 1. API Contracts (60 tests)

**File**: `test_api_contracts.py`

Validates component interfaces:

| Contract | Tests | Validates |
|----------|-------|-----------|
| **Theory.calculate()** | 7 | Parameter types, return shape, determinism |
| **Theory parameters** | 5 | Parameter structure (name, value, bounds) |
| **Dataset access** | 6 | `get_x()`, `get_y()`, `get_column()` methods |
| **Application loading** | 4 | Theory instantiation, protocol compliance |
| **Fit results** | 8 | Result structure (parameters, covariance, residuals) |

Example:
```python
def test_calculate_accepts_dict_params(mock_theory, synthetic_frequency_data):
    """Contract: Theory.calculate() accepts dict[str, float] params"""
    params = {"slope": 2.0, "intercept": 1.0}
    result = mock_theory.calculate(params, synthetic_frequency_data.x)
    assert isinstance(result, Array)
    assert result.shape == synthetic_frequency_data.x.shape
```

### 2. Data Contracts (40 tests)

**File**: `test_data_contracts.py`

Validates data structure requirements:

| Contract | Tests | Validates |
|----------|-------|-----------|
| **Dataset structure** | 9 | Fields, shapes, dtypes, lengths |
| **Theory parameters** | 9 | Bounds, uniqueness, type constraints |
| **Calculation output** | 5 | Finiteness, dtype, determinism |
| **Parameter persistence** | 3 | `set_parameter()` behavior |
| **Fit parameters** | 3 | Parameter name validity |
| **Serialization** | 3 | JSON/NPZ format, security |

Example:
```python
def test_dataset_x_y_lengths_match(mock_dataset):
    """Contract: x and y have matching lengths"""
    assert len(mock_dataset.get_x()) == len(mock_dataset.get_y())
```

### 3. Performance Contracts (15 tests)

**File**: `test_performance_contracts.py`

Establishes and monitors performance baselines:

| Contract | Baseline | Validates |
|----------|----------|-----------|
| **Linear fit** | < 1s (100 pts) | Execution time, feasibility |
| **Maxwell fit** | < 2s (100 pts) | Model complexity handling |
| **Fit scaling** | Sublinear | O(n) or better scaling |
| **Theory calc** | < 1ms (1 pt), < 100ms (100 pts) | Calculation efficiency |
| **Scaling** | Sublinear | Data size scaling |
| **Data access** | < 100µs per call | Array access overhead |
| **Memory** | No leaks | Repeated access stability |

Example:
```python
def test_linear_fit_performance(synthetic_frequency_data):
    """Contract: Linear fit completes in < 1 second"""
    start = time.perf_counter()
    result, _ = run_nlsq_fit(linear_model, x, y, p0=p0)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0
```

### 4. Compatibility Contracts (30 tests)

**File**: `test_compatibility_contracts.py`

Verifies migration paths and equivalence:

| Contract | Tests | Validates |
|----------|-------|-----------|
| **SciPy → JAX** | 6 | Numerical equivalence (exp, log, trig, etc.) |
| **Pickle → SafeSerializer** | 3 | Round-trip preservation |
| **Precision preservation** | 2 | Double precision (x64 mode) |
| **Type preservation** | 3 | Float, int, array preservation |
| **Numerical stability** | 3 | Stability across value ranges |
| **Interoperability** | 2 | Format compatibility, versioning |

Example:
```python
def test_exponential_equivalence():
    """Contract: jnp.exp() matches np.exp()"""
    x_np = np.array([0.0, 0.5, 1.0])
    x_jax = jnp.array(x_np)
    assert_array_almost_equal(
        np.array(jnp.exp(x_jax)),
        np.exp(x_np),
        decimal=10,
    )
```

## Fixtures

### Data Fixtures

```python
# Frequency sweep data (LVE testing)
synthetic_frequency_data: SyntheticData

# Time domain data (relaxation testing)
synthetic_time_data: SyntheticData

# Multi-mode Maxwell model
synthetic_multimode_data: SyntheticData

# Various input patterns
synthetic_theory_input_data: dict[str, Array]

# Parameter sets
synthetic_parameter_sets: dict[str, dict]
```

### Mock Implementations

```python
# Simple linear theory for testing
mock_theory: MockTheory

# Test dataset
mock_dataset: MockDataset

# Test application
mock_application: MockApplication

# Test fit result
mock_fit_result: MockFitResult
```

### Infrastructure

```python
# Temporary directory
temp_workspace: Path

# Baseline storage
baseline_storage: Path
baseline_registry: dict

# Validation helpers
contract_validator: dict
```

## Baseline Management

Contract tests automatically measure and track performance baselines:

### Recording Baselines

```python
def test_custom_fit_performance(baseline_manager):
    """Record performance baseline"""
    baseline = baseline_manager.get_baseline("custom_fitting")

    # Measure
    times = [measure_fit() for _ in range(3)]
    mean = sum(times) / len(times)
    std_dev = calculate_std(times)

    # Record
    baseline.record_baseline("custom_fit", mean, std_dev, len(times))
```

### Checking for Regressions

```python
def test_fit_regression(baseline_manager):
    """Check performance against baseline"""
    baseline = baseline_manager.get_baseline("fitting")

    measured_time = time_operation()
    result = baseline.check_regression("linear_fit", measured_time)

    if not result.passed:
        print(f"Regression: {result.regression_percent:.1f}% slower")
```

## CI/CD Integration

Contract tests run automatically in GitHub Actions (`.github/workflows/contract-tests.yml`):

### Workflow Jobs

1. **contract-tests** - Run all 94 tests, generate coverage
2. **performance-regression-check** - Detect performance degradation
3. **compatibility-check** - Verify SciPy→JAX, Pickle→SafeSerializer equivalence
4. **api-contract-enforcement** - Enforce API compliance
5. **summary** - Aggregate results, block merge on critical failures

### Quality Gates

✗ **Merge blocked if:**
- API contracts fail
- Data contracts fail
- Fit precision contracts fail

⚠ **Warnings (non-blocking):**
- Performance regression > threshold
- Compatibility issues detected

## Documentation

### Quick Start
- **This file** - Overview and structure
- `CONTRACTS_QUICK_REFERENCE.md` - Common tasks and examples

### Comprehensive Guides
- `CONTRACTS_SUMMARY.md` - Technical overview and metrics
- `docs/source/testing/contract_testing.rst` - Full guide with best practices
- `CONTRACT_TESTING_RESULTS.md` - Implementation details and results

## Writing New Contracts

### 1. Identify the Contract

What integration agreement needs to be validated?

```python
# Example: "Theory output must have same shape as input"
```

### 2. Create Test Class

```python
class TestNewContract:
    """Tests for new contract"""

    def test_contract_requirement(self, fixture1, fixture2):
        """Contract: Specific requirement"""
        # Arrange
        input_data = prepare_test_data()

        # Act
        result = perform_operation(input_data)

        # Assert
        assert validate_contract(result)
```

### 3. Document in Docstring

```python
def test_example(self):
    """Contract: Clear description of what must be true

    This contract ensures that:
    - Condition A is met
    - Condition B is met

    If violated: Breaking change to component interface
    """
```

### 4. Add to Appropriate File

- **API contracts** → `test_api_contracts.py`
- **Data contracts** → `test_data_contracts.py`
- **Performance contracts** → `test_performance_contracts.py`
- **Compatibility contracts** → `test_compatibility_contracts.py`

## Running Tests

### Complete Suite
```bash
pytest tests/contracts/ -v
```

### Specific Test
```bash
pytest tests/contracts/test_api_contracts.py::TestTheoryCalculateContract::test_calculate_deterministic -v
```

### With Markers
```bash
# Fast tests only (skip slow performance tests)
pytest tests/contracts/ -v -m "not slow"

# Verbose with full assertions
pytest tests/contracts/ -vv --tb=long

# Stop at first failure
pytest tests/contracts/ -x

# Show slowest tests
pytest tests/contracts/ --durations=10
```

## Troubleshooting

### Test Discovery Issues

```bash
# Verify pytest can find tests
pytest tests/contracts/ --collect-only -q

# Check for import errors
pytest tests/contracts/ -v --tb=short
```

### Fixture Not Found

- Ensure `conftest.py` is in the same directory
- Check fixture names match exactly
- Verify imports are correct

### Performance Tests Slow

- First run includes JAX JIT compilation overhead
- Subsequent runs are faster
- Record baselines after warmup

### Baseline Comparison Issues

- Baselines stored in `.baselines/` directory
- JSON format (human-readable)
- Commit baseline changes intentionally
- Include rationale in commit message

## Best Practices

1. **Write contracts first** - Follow TDD approach
2. **Be explicit** - Document assumptions clearly
3. **Keep contracts simple** - Test one thing per test
4. **Reuse fixtures** - Don't duplicate test data
5. **Update intentionally** - Record baseline changes deliberately
6. **Document violations** - Explain why contracts changed

## Integration Points Tested

### Theory ↔ Core
- Parameter validation
- Calculation contracts
- Performance baselines

### DataSet ↔ Theory
- Data access APIs
- Shape/dtype matching
- Calculation inputs

### File I/O
- JSON metadata format
- NPZ array handling
- Security (no pickle)

### Fitting (NLSQ)
- Time budgets
- Scaling behavior
- Result structure

### Migration Paths
- SciPy → JAX equivalence
- Pickle → SafeSerializer compatibility
- Numerical precision preservation

## Test Metrics

| Metric | Value |
|--------|-------|
| Total tests | 94 |
| Total LOC | 2,235 |
| Execution time | ~45 seconds |
| Pass rate | ~96% |
| Coverage | All major APIs |

## References

- **Consumer-Driven Contract Testing**: https://martinfowler.com/articles/consumerDrivenContracts.html
- **Pact Framework**: https://pact.foundation/
- **Contract Testing**: https://contracttestingguide.com/

## Support

For questions:
1. Check relevant test class docstrings
2. Review `conftest.py` for fixture definitions
3. Check documentation in `docs/source/testing/`
4. Review implementation in `src/RepTate/`

---

**Last Updated**: 2025-12-31
**Status**: Production Ready
**Maintainers**: RepTate Development Team
