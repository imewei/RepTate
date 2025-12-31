Data Flow Documentation
=======================

This document describes how data flows through RepTate during typical operations.

Loading Data
------------

.. graphviz::

   digraph load_data {
       rankdir=LR;
       node [shape=box];

       FileSystem [shape=folder];
       FileIOController;
       DataTable;
       QDataSet;
       View;
       Plot;

       FileSystem -> FileIOController [label="read file"];
       FileIOController -> DataTable [label="parse columns"];
       DataTable -> QDataSet [label="add to dataset"];
       QDataSet -> View [label="transform"];
       View -> Plot [label="render"];
   }

Steps:

1. User selects file(s) via ``FileIOController.open_file_names_dialog()``
2. ``DataTable`` parses the file into columns (x, y, optional error)
3. ``QDataSet`` receives the DataTable and manages the collection
4. ``ViewCoordinator`` applies the current ``View`` transform
5. Data is rendered to the matplotlib plot

Theory Calculation
------------------

.. graphviz::

   digraph calculate {
       rankdir=TB;
       node [shape=box];

       DataTable [label="Data\n(x, y)"];
       ParameterController;
       TheoryCompute;
       TheoryTable [label="Theory\n(x_th, y_th)"];
       View;
       Plot;

       DataTable -> ParameterController [label="get x range"];
       ParameterController -> TheoryCompute [label="params dict"];
       DataTable -> TheoryCompute [label="x data"];
       TheoryCompute -> TheoryTable [label="calculate()"];
       TheoryTable -> View [label="transform"];
       DataTable -> View [label="transform"];
       View -> Plot [label="overlay"];
   }

Steps:

1. ``ParameterController`` provides current parameter values
2. ``TheoryCompute`` receives x-data and parameters
3. Theory's ``calculate()`` method produces predictions
4. Results stored in theory's DataTable
5. Both data and theory rendered through same View

Curve Fitting
-------------

.. graphviz::

   digraph fit {
       rankdir=TB;
       node [shape=box];

       Data [label="Experimental Data\n(x, y)"];
       ParameterController;
       FitController;
       nlsq_fit;
       FitResult;
       UpdateParams [label="Update Parameters"];

       Data -> FitController [label="y data"];
       ParameterController -> FitController [label="initial params"];
       FitController -> nlsq_fit [label="model, data, p0"];
       nlsq_fit -> FitResult [label="optimized params"];
       FitResult -> UpdateParams;
       UpdateParams -> ParameterController;
   }

Steps:

1. ``FitController`` collects data and initial parameters
2. ``run_nlsq_fit()`` performs deterministic curve fitting using NLSQ
3. Optimized parameters returned in ``FitResult``
4. ``ParameterController`` updates theory parameters
5. Theory recalculates with new parameters

Bayesian Inference (Optional)
-----------------------------

For uncertainty quantification:

.. graphviz::

   digraph bayes {
       rankdir=TB;
       node [shape=box];

       FitResult [label="NLSQ Fit Result\n(warm start)"];
       NumPyroModel;
       NUTS;
       Samples;
       Posterior;

       FitResult -> NumPyroModel [label="prior means"];
       NumPyroModel -> NUTS [label="model"];
       NUTS -> Samples [label="MCMC"];
       Samples -> Posterior [label="analyze"];
   }

Steps:

1. NLSQ fit provides warm-start values for MCMC
2. NumPyro model defined with priors centered on NLSQ results
3. NUTS sampler explores posterior distribution
4. Samples analyzed for uncertainty estimates

Error Calculation Modes
-----------------------

RepTate supports multiple error calculation modes:

1. **Standard** (relative): ``sum((y_data - y_theory)^2 / y_data^2)``
2. **Absolute**: ``sum((y_data - y_theory)^2)``
3. **Logarithmic**: ``sum((log(y_data) - log(y_theory))^2)``

Implemented in ``TheoryCompute``:

.. code-block:: python

   def calculate_error_standard(self, y_data, y_theory):
       residuals = self.calculate_residuals(y_data, y_theory)
       y_safe = jnp.where(jnp.abs(y_data) < 1e-100, 1e-100, y_data)
       relative_residuals = residuals / y_safe
       return float(jnp.sum(relative_residuals**2))

All computations use JAX with float64 precision for numerical accuracy
within 1e-10 tolerance.
