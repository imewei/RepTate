# Quickstart: JAX-First Modernization

## Prerequisites
- Python 3.12+
- A working C/C++ compiler toolchain suitable for scientific Python packages
- Optional accelerator drivers for GPU/TPU usage (CPU-only is supported)

## Setup
1. Create and activate a virtual environment.
2. Install project dependencies from the repository configuration.
3. Verify the application starts in CPU mode.

## Run
- Launch the desktop application from the repository root.
- Load a sample dataset and run a deterministic fit.
- Start Bayesian inference to produce posterior summaries.

## Test
- Run the test suite for unit, integration, and regression coverage.

## Notes
- CPU execution is the default; acceleration is optional and should warn when unavailable.
- Exports include plots, numeric results, and raw posterior traces.

Validated: 2025-12-29
