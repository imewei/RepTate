# SciPy to JAX Migration Guide

This guide helps developers understand the SciPy to JAX migration and how to work with the new JAX-based implementations.

---

## Quick Reference

### Interpolation

**Old (SciPy):**
```python
from scipy.interpolate import interp1d

x = np.array([0, 1, 2, 3, 4])
y = np.array([0, 1, 4, 9, 16])

# Create interpolator
f = interp1d(x, y, kind='cubic', fill_value='extrapolate')

# Use it
y_new = f(np.array([0.5, 1.5, 2.5]))
```

**New (interpax):**
```python
from interpax import interp1d
import jax.numpy as jnp

x = jnp.array([0, 1, 2, 3, 4])
y = jnp.array([0, 1, 4, 9, 16])

# Interpolate (functional API - no factory pattern)
x_query = jnp.array([0.5, 1.5, 2.5])
y_new = interp1d(x_query, x, y, method='cubic', extrap=True)
```

**Key Differences:**
- Functional API: `interp1d(xq, x, f, ...)` not `interp1d(x, f, ...)(xq)`
- Parameter names: `method` not `kind`, `extrap` not `fill_value`
- Returns values directly, not a callable

---

### Integration

**Old (SciPy ODE Solver):**
```python
from scipy.integrate import odeint
from scipy.interpolate import interp1d

# Create interpolator
f = interp1d(x, y, kind='cubic', fill_value='extrapolate')

# Integrate using ODE solver
func = lambda y0, t: f(t)
y_integral = odeint(func, [0], x)
```

**New (Trapezoidal Rule):**
```python
import jax.numpy as jnp
from interpax import interp1d

# Interpolate at points
x_jax = jnp.array(x)
y_jax = jnp.array(y)
y_interp = interp1d(x_jax, x_jax, y_jax, method='cubic', extrap=True)

# Cumulative trapezoidal integration
dx = x_jax[1:] - x_jax[:-1]
avg_y = (y_interp[1:] + y_interp[:-1]) / 2.0
y_integral = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(avg_y * dx)])
```

**Why Changed:**
- Trapezoidal rule is more direct for cumulative integration
- No need for ODE solver overhead
- More numerically stable
- Better matches the mathematical operation

---

### Smoothing (Savitzky-Golay Filter)

**Old (SciPy):**
```python
from scipy.signal import savgol_filter

y_smooth = savgol_filter(y, window_length=11, polyorder=3)
```

**New (Custom JAX Implementation):**
```python
from RepTate.tools.ToolSmooth import _savgol_filter_jax

y_smooth = _savgol_filter_jax(y, window_length=11, polyorder=3)
```

**Implementation:**
- Pure JAX implementation using Vandermonde matrices
- Preserves polynomials up to specified order
- Edge handling via padding
- Numerically equivalent to SciPy version

---

## Common Patterns

### Pattern 1: Interpolation at Single Point

```python
from interpax import interp1d
import jax.numpy as jnp

# Data
x = jnp.array([0, 1, 2, 3, 4])
y = jnp.array([0, 1, 4, 9, 16])

# Query single point (needs to be array)
x_query = jnp.array([2.5])
y_value = interp1d(x_query, x, y, method='cubic', extrap=True)[0]
```

### Pattern 2: Interpolation at Multiple Points

```python
# Query multiple points
x_query = jnp.array([0.5, 1.5, 2.5, 3.5])
y_values = interp1d(x_query, x, y, method='cubic', extrap=True)
```

### Pattern 3: Adaptive Interpolation Method

```python
# Choose method based on number of data points
n_points = len(x)
if n_points < 2:
    method = 'nearest'
elif n_points < 3:
    method = 'linear'
elif n_points < 4:
    method = 'quadratic'
else:
    method = 'cubic'

y_values = interp1d(x_query, x, y, method=method, extrap=True)
```

---

## interpax Methods

| Method | Description | Min Points | Continuity |
|--------|-------------|------------|------------|
| `'nearest'` | Nearest neighbor | 1 | C⁻¹ |
| `'linear'` | Linear interpolation | 2 | C⁰ |
| `'cubic'` | Local cubic splines | 4 | C¹ |
| `'cubic2'` | Natural cubic splines | 4 | C² |
| `'quadratic'` | Quadratic interpolation | 3 | C¹ |
| `'monotonic'` | Monotonicity-preserving | 4 | C¹ |
| `'akima'` | Akima splines | 5 | C¹ |

---

## JAX Array Conversion

### When to Convert

Convert to JAX arrays when:
- Passing data to interpax functions
- Using JAX operations (jnp.*)
- Need differentiability

Convert back to NumPy when:
- Returning from tool/application methods
- Plotting or displaying results
- Interfacing with non-JAX code

### Example

```python
import numpy as np
import jax.numpy as jnp

# Input (often NumPy from data files)
x_np = np.array([0, 1, 2, 3, 4])
y_np = np.array([0, 1, 4, 9, 16])

# Convert for JAX operations
x_jax = jnp.array(x_np)
y_jax = jnp.array(y_np)

# Perform JAX operations
result_jax = interp1d(x_jax, x_jax, y_jax, method='cubic', extrap=True)

# Convert back for output
result_np = np.array(result_jax)
```

---

## Testing Migration Code

### Unit Test Template

```python
import numpy as np
import pytest
from numpy.testing import assert_allclose


class TestMyMigration:
    def test_numerical_equivalence(self):
        """Test that new implementation matches old behavior."""
        # Test data
        x = np.array([0, 1, 2, 3, 4])
        y = np.array([0, 1, 4, 9, 16])

        # New implementation
        from my_module import my_new_function
        result_new = my_new_function(x, y)

        # Expected result (from analytical solution or old implementation)
        result_expected = x**2

        # Assert equivalence
        assert_allclose(result_new, result_expected, rtol=1e-10)
```

### Run Tests

```bash
# Run specific test file
pytest tests/unit/test_scipy_migration_core.py -v

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/RepTate --cov-report=html
```

---

## Performance Considerations

### JIT Compilation

JAX can JIT-compile functions for better performance:

```python
from jax import jit

@jit
def my_computation(x, y):
    return interp1d(x, x, y, method='cubic', extrap=True)

# First call compiles
result1 = my_computation(x1, y1)  # Slow (compilation)

# Subsequent calls are fast
result2 = my_computation(x2, y2)  # Fast (cached)
```

### Vectorization

Use `jax.vmap` for vectorizing operations:

```python
from jax import vmap

# Vectorize over first axis
vectorized_interp = vmap(
    lambda xi: interp1d(xi, x, y, method='cubic', extrap=True)
)

# Apply to multiple query points efficiently
results = vectorized_interp(x_queries)
```

---

## Troubleshooting

### Issue: "TypeError: interp1d() missing 1 required positional argument: 'f'"

**Cause:** Using SciPy-style API with interpax

**Solution:** Switch to functional API:
```python
# Wrong (SciPy style)
f = interp1d(x, y, method='cubic')
result = f(x_query)

# Correct (interpax style)
result = interp1d(x_query, x, y, method='cubic')
```

### Issue: "Shape mismatch in interpolation"

**Cause:** Query points not an array

**Solution:** Wrap scalar in array:
```python
# Wrong
y_val = interp1d(2.5, x, y, method='cubic')

# Correct
y_val = interp1d(jnp.array([2.5]), x, y, method='cubic')[0]
```

### Issue: "Invalid method for interpolation"

**Cause:** Not enough data points for chosen method

**Solution:** Use adaptive method selection:
```python
n = len(x)
if n < 4:
    method = 'linear' if n >= 2 else 'nearest'
else:
    method = 'cubic'

result = interp1d(x_query, x, y, method=method)
```

---

## Migration Checklist

When migrating SciPy code to JAX:

- [ ] Replace `scipy.interpolate.interp1d` with `interpax.interp1d`
- [ ] Update to functional API (no factory pattern)
- [ ] Convert arrays to JAX arrays where needed
- [ ] Replace `scipy.integrate.odeint` with appropriate JAX alternative
- [ ] Replace `scipy.signal` functions with JAX implementations
- [ ] Update parameter names (`kind` → `method`, etc.)
- [ ] Add tests for numerical equivalence
- [ ] Verify no scipy imports remain (except `jax.scipy`)
- [ ] Update documentation

---

## Best Practices

1. **Type Annotations:** Use JAX array type hints
   ```python
   from jaxtyping import Array, Float

   def my_function(x: Float[Array, "n"]) -> Float[Array, "n"]:
       ...
   ```

2. **Functional Style:** Avoid side effects
   ```python
   # Good
   def process(x):
       return x * 2

   # Avoid
   result = []
   def process(x):
       result.append(x * 2)
   ```

3. **Test Coverage:** Maintain >90% coverage for migrated code

4. **Documentation:** Document migration decisions in comments

5. **Performance:** Profile before optimizing with JIT

---

## Resources

- [JAX Documentation](https://jax.readthedocs.io/)
- [interpax GitHub](https://github.com/f0uriest/interpax)
- [JAX NumPy API](https://jax.readthedocs.io/en/latest/jax.numpy.html)
- [Migration Summary](../SCIPY_TO_JAX_MIGRATION_COMPLETE.md)

---

## Getting Help

If you encounter issues:

1. Check this guide first
2. Review existing migrated files for patterns
3. Run the verification script: `python scripts/verify_scipy_removal.py`
4. Check test suite for examples: `tests/unit/test_scipy_migration_core.py`
5. Consult the interpax documentation

---

**Last Updated:** 2025-12-31
