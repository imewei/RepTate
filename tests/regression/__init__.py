"""Regression tests for RepTate numerical consistency.

Regression tests ensure that numerical results remain consistent across
code changes, particularly during the SciPy to JAX migration.

Test files:
    - validate_fit_precision.py: Validate fitting precision
    - validate_inference_precision.py: Validate Bayesian inference precision
    - benchmark_fit_timing.py: Benchmark fitting performance
    - benchmark_export_timing.py: Benchmark export performance

Tolerance: 1e-10 for JAX equivalence tests (per constitution)
"""
