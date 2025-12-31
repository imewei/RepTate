# RepTate Performance Benchmarks

Comprehensive performance benchmark suite for RepTate's JAX + NLSQ migration.

## Overview

This directory contains performance benchmarks to:
- Establish baseline metrics for critical operations
- Detect performance regressions in CI/CD
- Guide optimization efforts
- Document expected performance characteristics

## Benchmark Categories

### 1. Curve Fitting (`test_benchmark_fitting.py`)
- Linear and non-linear regression
- NLSQ-based fitting operations
- Scaling tests with different data sizes
- JIT compilation speedup validation

### 2. Theory Calculations (`test_benchmark_theory.py`)
- Maxwell modes (frequency and time domain)
- Giesekus model
- Large relaxation spectra
- Vectorization benefits
- Batch processing with vmap

### 3. I/O and Serialization (`test_benchmark_io.py`)
- Safe serialization (JSON + NPZ)
- Large file handling
- Round-trip performance
- Security overhead validation

### 4. Interpolation (`test_benchmark_interpolation.py`)
- Linear and cubic spline interpolation
- Log-scale transformations (rheology data)
- Batch interpolation
- Real-world rheology workflows

## Running Benchmarks

### Run All Benchmarks
```bash
pytest tests/benchmarks/ --benchmark-only -v
```

### Run Specific Category
```bash
pytest tests/benchmarks/test_benchmark_fitting.py --benchmark-only
pytest tests/benchmarks/test_benchmark_theory.py --benchmark-only
pytest tests/benchmarks/test_benchmark_io.py --benchmark-only
pytest tests/benchmarks/test_benchmark_interpolation.py --benchmark-only
```

### Run with Slow Marker
```bash
pytest tests/benchmarks/ -m slow --benchmark-only
```

### Get Baseline Summary
```bash
# Run summary tests (no SLA enforcement)
pytest tests/benchmarks/test_benchmark_fitting.py::test_benchmark_baseline_summary -v
pytest tests/benchmarks/test_benchmark_theory.py::test_benchmark_theory_summary -v
pytest tests/benchmarks/test_benchmark_io.py::test_benchmark_io_summary -v
pytest tests/benchmarks/test_benchmark_interpolation.py::test_benchmark_interpolation_summary -v
```

## Performance SLAs

### Hard SLAs (Must Meet)
- Linear fit (100 pts): < 10ms
- Maxwell single mode: < 5ms
- Linear interpolation: < 5ms
- Safe serialization (small): < 10ms

### Soft SLAs (Target)
- Maxwell multi-mode (20): < 10ms
- Complex fitting (6 params): < 200ms
- Large dataset (10k pts): < 100ms

See `PERFORMANCE_BASELINES.md` for complete SLA details.

## CI/CD Integration

### GitHub Actions
Benchmarks run on every PR to detect regressions:

```yaml
# .github/workflows/benchmarks.yml
- name: Run benchmarks
  run: pytest tests/benchmarks/ --benchmark-only --benchmark-json=output.json

- name: Compare with baseline
  run: python scripts/compare_benchmarks.py output.json baseline.json
```

### Regression Detection
- Alert if > 20% slower than baseline (soft SLA)
- Fail PR if hard SLA violated
- Update baseline quarterly or on hardware changes

## Interpreting Results

### Sample Output
```
test_benchmark_linear_fit_small:
  Mean: 8.234 ms
  Std:  0.123 ms
  Min:  8.012 ms
  Max:  8.456 ms
  Iterations: 20
```

### Key Metrics
- **Mean:** Average execution time (primary metric)
- **Std:** Standard deviation (stability indicator)
- **Min:** Best case (lower bound)
- **Max:** Worst case (upper bound)
- **Iterations:** Number of runs (confidence)

### JIT Compilation
All benchmarks include warmup iterations (3-10) to exclude JIT compilation time. Results reflect steady-state performance.

## Writing New Benchmarks

### Template
```python
import jax
import jax.numpy as jnp
import pytest
from tests.benchmarks import BenchmarkConfig, benchmark_function

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

@pytest.mark.slow
def test_benchmark_my_operation() -> None:
    """Benchmark description.

    Expected: < XXms (after JIT warmup)
    Rationale for SLA.
    """
    # Setup
    data = jnp.linspace(0, 10, 100)

    config = BenchmarkConfig(n_iterations=20, warmup_iterations=5)

    def operation() -> None:
        result = my_operation(data)
        _ = result.block_until_ready()  # Force evaluation

    # Run benchmark
    benchmark_result = benchmark_function(operation, config)

    print(f"\n{benchmark_result}")
    print(f"  Additional context")

    # Assert SLA
    assert benchmark_result.mean_time < 0.XXX, (
        f"Operation too slow: {benchmark_result.mean_time*1000:.3f}ms > XXms"
    )
```

### Best Practices
1. **Warmup:** Always include warmup iterations for JIT
2. **Force evaluation:** Use `.block_until_ready()` for JAX arrays
3. **Descriptive docstring:** Document expected performance and rationale
4. **Print context:** Show problem size and relevant parameters
5. **Clear SLA:** Assert against specific threshold with error message

## Profiling Integration

### JAX Profiler
```python
import jax

jax.profiler.start_trace("/tmp/jax_trace")
# ... run operation ...
jax.profiler.stop_trace()

# View with TensorBoard:
# tensorboard --logdir=/tmp/jax_trace
```

### Python Profiler
```bash
python -m cProfile -o profile.stats -m pytest tests/benchmarks/test_benchmark_fitting.py::test_benchmark_linear_fit_small
python -m pstats profile.stats
```

### Memory Profiler
```bash
python -m memory_profiler tests/benchmarks/test_benchmark_fitting.py
```

## Baseline Updates

### When to Update
- Quarterly review
- Hardware changes
- JAX version upgrade
- Algorithm improvements

### Update Process
1. Run full benchmark suite
2. Review results against current baseline
3. Document reasons for changes
4. Update `PERFORMANCE_BASELINES.md`
5. Commit new baseline JSON

### Baseline Storage
```json
{
  "test_benchmark_linear_fit_small": {
    "mean_time": 0.00823,
    "std_time": 0.00012,
    "timestamp": "2025-12-31",
    "jax_version": "0.8.0",
    "hardware": "CPU x86_64"
  }
}
```

## Hardware Considerations

### Reference Platform
- CPU: Modern x86_64 (2020+)
- Cores: 4+
- RAM: 16GB
- Storage: SSD

### Platform Variations
Results may vary by platform:
- x86_64 (modern): 1.0x baseline
- ARM64 (M1/M2): 1.0-1.3x
- x86_64 (older): 0.7-0.9x

## Troubleshooting

### Benchmarks Too Slow
1. Check JAX configuration (x64, CPU)
2. Verify JIT compilation is enabled
3. Increase warmup iterations
4. Profile to find bottleneck

### Inconsistent Results
1. Increase iterations for better statistics
2. Reduce system load (close other apps)
3. Check for thermal throttling
4. Run multiple times and average

### SLA Violations
1. Profile the failing operation
2. Check for recent code changes
3. Review optimization opportunities
4. Consider if SLA needs adjustment

## References

- Performance Baselines: `../../PERFORMANCE_BASELINES.md`
- Optimization Guide: `../../OPTIMIZATION_RECOMMENDATIONS.md`
- JAX Guide: `../../JAX_OPTIMIZATION_GUIDE.md`
- JAX Documentation: https://jax.readthedocs.io/

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-31 | Initial benchmark suite |
