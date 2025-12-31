=============
Serialization
=============

Safe serialization utilities replacing pickle with JSON/NPZ formats.

.. module:: RepTate.core.serialization
   :synopsis: Safe serialization without pickle

Overview
--------

The ``serialization`` module provides secure alternatives to Python's pickle,
which is vulnerable to arbitrary code execution attacks. All data is serialized
using JSON for metadata and NPZ for numerical arrays.

Quick Start
-----------

.. code-block:: python

   from RepTate.core.serialization import SafeSerializer

   # Save data safely
   data = {
       'parameters': {'tau': 1.5, 'G': 1e6},
       'arrays': np.array([1, 2, 3])
   }
   SafeSerializer.save('output.json', data)

   # Load data safely
   loaded = SafeSerializer.load('output.json')

Classes
-------

.. autoclass:: SafeSerializer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: NumpyEncoder
   :members:

Functions
---------

.. autofunction:: save_safe
.. autofunction:: load_safe

Security Notes
--------------

- Never use ``pickle.load()`` on untrusted data
- This module validates all inputs before processing
- Large arrays are stored in separate NPZ files for efficiency
