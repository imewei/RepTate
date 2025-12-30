# Validation Checklist

## Numerical Routines

- Confirm deterministic fitting uses full precision (float64) for core kernels.
- Confirm no approximation shortcuts (subsampling, randomized SVD) are used.
- Run `scripts/bench/validate_full_precision.py` after changes to numerical code.
