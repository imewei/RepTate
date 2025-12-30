# RepTate Architecture

## Overview

RepTate is organized as a single Python package under `src/RepTate/` with clear
separation between computation, fitting, Bayesian inference, and UI layers.

## Module Boundaries

- `core/`: Domain models, parameters, data types, and shared interfaces.
- `core/jax_ops/`: JAX-only numerical kernels and model interfaces.
- `core/fitting/`: NLSQ fitting orchestration and workflow pipelines.
- `core/bayes/`: Bayesian inference models and diagnostics.
- `core/io/`: Dataset and results import/export helpers.
- `gui/`: PyQt UI, viewmodels, and plotting widgets.
- `theories/`: Physics model implementations that call JAX kernels.
- `tools/`: Auxiliary tools and utilities.

## Extension Points

- Add new models by implementing kernels in `core/jax_ops/models.py` and
  exposing them through theory modules in `theories/`.
- Add new fit workflows in `core/fitting/` that reuse existing kernels.
- Add new UI features by creating viewmodels in `gui/viewmodels/` and widgets in
  `gui/widgets/`.
