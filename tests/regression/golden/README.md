# Golden Files for Numerical Regression Tests

This directory contains golden (reference) output files for verifying numerical
equivalence of JAX-based implementations against known-good values.

## Purpose

Golden files provide a stable baseline for detecting:
1. Unintended changes in numerical output
2. Precision regressions during refactoring
3. Platform-specific floating-point variations

## File Format

Files are stored as NumPy `.npz` format for exact binary reproducibility:
- `maxwell_model.npz`: Reference values for Maxwell model calculations
- `exponential_model.npz`: Reference values for exponential relaxation
- `fit_results.npz`: Reference fitting results for known test cases

## Generation

Golden files were generated on:
- Platform: Linux 6.8.0 (x86_64)
- Python: 3.12+
- JAX: 0.8.0+
- Float precision: float64 (x64 mode enabled)

## Usage

The `test_numerical_equivalence.py` tests compare current outputs against these
golden files with tolerance of 1e-10 to accommodate JAX FP operation reordering.

## Regeneration

To regenerate golden files (requires careful review):
```bash
python -c "from tests.regression.generate_golden import generate_all; generate_all()"
```
