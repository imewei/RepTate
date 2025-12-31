=============
Feature Flags
=============

Feature flag system for gradual rollout of new implementations.

.. module:: RepTate.core.feature_flags
   :synopsis: Feature flags for gradual migration

Overview
--------

The ``feature_flags`` module enables gradual rollout of new code paths using
the strangler fig pattern. Features can be toggled via configuration files
or environment variables.

Quick Start
-----------

.. code-block:: python

   from RepTate.core.feature_flags import is_enabled, FeatureFlagManager

   # Check if a feature is enabled
   if is_enabled("use_jax_integration"):
       result = jax_optimize(params)
   else:
       result = legacy_optimize(params)

   # Using the manager directly
   manager = FeatureFlagManager.get_instance()
   manager.enable("use_decomposed_gui")

Configuration
-------------

Feature flags can be configured via YAML:

.. code-block:: yaml

   # feature_flags.yaml
   flags:
     use_jax_integration:
       enabled: true
       rollout_percentage: 100
       description: "Use JAX-based optimization"

     use_decomposed_gui:
       enabled: false
       rollout_percentage: 0
       description: "Use extracted GUI controllers"

Environment overrides:

.. code-block:: bash

   export REPTATE_FEATURE_USE_JAX_INTEGRATION=true

Classes
-------

.. autoclass:: FeatureFlagManager
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: FeatureFlag
   :members:

Functions
---------

.. autofunction:: is_enabled
.. autofunction:: get_flag
.. autofunction:: set_flag

Available Flags
---------------

==================== ========= ==================================
Flag                 Default   Description
==================== ========= ==================================
use_jax_integration  true      JAX-based optimization
use_decomposed_gui   false     Extracted GUI controllers
use_jax_native       false     JAX instead of C libraries
use_safe_eval        true      Safe expression evaluation
use_safe_serialization true    JSON/NPZ serialization
==================== ========= ==================================
