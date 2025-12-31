============
Strangler Fig
============

High-level migration infrastructure for gradual legacy code replacement.

.. module:: RepTate.core.strangler_fig
   :synopsis: Strangler fig pattern implementation

Overview
--------

The ``strangler_fig`` module provides the main API for migrating from legacy
code to new implementations. It combines feature flags, circuit breakers,
and dual-run validation into a simple decorator-based interface.

Quick Start
-----------

.. code-block:: python

   from RepTate.core.strangler_fig import with_migration

   def legacy_optimize(params):
       return scipy.optimize.curve_fit(...)

   @with_migration(
       name='jax_optimize',
       flag='use_jax_integration',
       legacy_impl=legacy_optimize,
       validate=True,
       rtol=1e-5
   )
   def jax_optimize(params):
       return nlsq.curve_fit(...)

   # Will use JAX if flag enabled, validate against legacy, fallback on error

Decorator Usage
---------------

.. code-block:: python

   @with_migration(
       name='feature_name',           # Unique identifier
       flag='feature_flag_name',      # Feature flag to check
       legacy_impl=legacy_function,   # Fallback function
       validate=True,                 # Compare results
       rtol=1e-5,                     # Relative tolerance
       atol=1e-8                      # Absolute tolerance
   )
   def new_implementation(params):
       return modern_result

Functional API
--------------

.. code-block:: python

   from RepTate.core.strangler_fig import migrate_with_validation

   result = migrate_with_validation(
       new_impl=jax_optimize,
       legacy_impl=scipy_optimize,
       args=(params,),
       flag='use_jax_integration'
   )

Classes
-------

.. autoclass:: MigrationConfig
   :members:

.. autoclass:: MigrationResult
   :members:

Decorators
----------

.. autodecorator:: with_migration

Functions
---------

.. autofunction:: migrate_with_validation
.. autofunction:: get_migration_health
.. autofunction:: get_migration_summary

Integration
-----------

The strangler fig integrates with:

- **Feature Flags**: Control which implementation runs
- **Circuit Breaker**: Auto-fallback on repeated failures
- **Dual-Run**: Validate new vs legacy results
- **Observability**: Track migration progress

See Also
--------

- :doc:`feature_flags` - Feature flag configuration
- :doc:`circuit_breaker` - Automatic fallback mechanism
- :doc:`../../../developers/MIGRATION_GUIDE_DETAILED` - Full migration guide
