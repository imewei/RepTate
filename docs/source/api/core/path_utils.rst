==========
Path Utils
==========

Safe path handling utilities preventing path traversal attacks.

.. module:: RepTate.core.path_utils
   :synopsis: Secure path handling

Overview
--------

The ``path_utils`` module provides utilities for safe file path handling,
preventing path traversal attacks (e.g., ``../../../etc/passwd``).

Quick Start
-----------

.. code-block:: python

   from RepTate.core.path_utils import SafePath, validate_path

   # Validate a path is within allowed directory
   safe = SafePath('/data/reptate')
   resolved = safe.resolve('user_file.json')  # OK
   resolved = safe.resolve('../../../etc/passwd')  # Raises SecurityError

   # Quick validation
   if validate_path(user_input, base_dir='/data'):
       process_file(user_input)

Classes
-------

.. autoclass:: SafePath
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autofunction:: validate_path
.. autofunction:: is_safe_path
.. autofunction:: normalize_path

Exceptions
----------

.. autoexception:: PathTraversalError
   :members:

Security Notes
--------------

- Always use SafePath for user-provided file paths
- Never construct paths with string concatenation
- Validate paths before any file operations
