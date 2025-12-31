=========
Safe Eval
=========

Secure expression evaluation replacing Python's unsafe ``eval()``.

.. module:: RepTate.core.safe_eval
   :synopsis: AST-based safe expression evaluation

Overview
--------

The ``safe_eval`` module provides a secure alternative to Python's built-in
``eval()`` function, which can execute arbitrary code. This module uses
AST (Abstract Syntax Tree) parsing to validate expressions before evaluation.

Quick Start
-----------

.. code-block:: python

   from RepTate.core.safe_eval import safe_eval, SafeExpression

   # Evaluate simple expressions
   result = safe_eval("x + 2 * y", {"x": 1, "y": 3})  # Returns 7

   # Using SafeExpression for repeated evaluation
   expr = SafeExpression("sin(x) + cos(y)")
   result1 = expr.evaluate({"x": 0, "y": 0})  # Returns 1.0
   result2 = expr.evaluate({"x": 3.14, "y": 0})  # Returns ~1.0

Allowed Operations
------------------

**Arithmetic:** ``+``, ``-``, ``*``, ``/``, ``//``, ``%``, ``**``

**Comparison:** ``<``, ``<=``, ``>``, ``>=``, ``==``, ``!=``

**Math Functions:** ``sin``, ``cos``, ``tan``, ``exp``, ``log``, ``log10``,
``sqrt``, ``abs``, ``min``, ``max``, ``sum``

**Constants:** ``pi``, ``e``

Blocked Operations
------------------

For security, the following are NOT allowed:

- Import statements
- Function definitions
- Class definitions
- Attribute access (e.g., ``obj.__class__``)
- Subscript assignment
- exec/eval calls

Classes
-------

.. autoclass:: SafeExpression
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: SafeExpressionValidator
   :members:

.. autoclass:: SafeExpressionEvaluator
   :members:

Functions
---------

.. autofunction:: safe_eval
.. autofunction:: validate_expression

Exceptions
----------

.. autoexception:: UnsafeExpressionError
   :members:

Security Notes
--------------

- All expressions are parsed into an AST before evaluation
- Dangerous node types (Import, Exec, Eval) are rejected
- Attribute access is blocked to prevent ``__class__`` exploits
- Resource limits can be configured to prevent DoS attacks
