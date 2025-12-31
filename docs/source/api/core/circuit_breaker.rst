===============
Circuit Breaker
===============

Automatic fallback mechanism for resilient code execution.

.. module:: RepTate.core.circuit_breaker
   :synopsis: Circuit breaker pattern for fallback

Overview
--------

The ``circuit_breaker`` module implements the circuit breaker pattern,
providing automatic fallback to legacy code when new implementations fail.

States
------

The circuit breaker has three states:

**CLOSED** (normal operation)
   - Requests go to the new implementation
   - Failures are counted
   - Transitions to OPEN when failure threshold exceeded

**OPEN** (fallback mode)
   - Requests go directly to legacy implementation
   - New implementation is bypassed
   - After timeout, transitions to HALF_OPEN

**HALF_OPEN** (testing recovery)
   - One test request goes to new implementation
   - Success → CLOSED, Failure → OPEN

Quick Start
-----------

.. code-block:: python

   from RepTate.core.circuit_breaker import CircuitBreaker

   cb = CircuitBreaker(
       name='jax_optimizer',
       failure_threshold=5,
       recovery_timeout=60.0,
       fallback=legacy_optimizer
   )

   @cb.protect
   def jax_optimizer(params):
       return jax_result

   # Will fallback to legacy_optimizer after 5 failures

Configuration
-------------

.. code-block:: python

   CircuitBreaker(
       name='unique_name',           # Identifier
       failure_threshold=5,          # Failures before opening
       recovery_timeout=60.0,        # Seconds before half-open
       success_threshold=3,          # Successes to close
       fallback=fallback_fn          # Fallback function
   )

Classes
-------

.. autoclass:: CircuitBreaker
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: CircuitBreakerState
   :members:

.. autoclass:: CircuitBreakerRegistry
   :members:

Functions
---------

.. autofunction:: get_circuit_breaker
.. autofunction:: reset_all_circuit_breakers
