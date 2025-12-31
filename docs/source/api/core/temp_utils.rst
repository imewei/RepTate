==========
Temp Utils
==========

Temporary file management utilities with automatic cleanup.

.. module:: RepTate.core.temp_utils
   :synopsis: Temporary file management

Overview
--------

The ``temp_utils`` module provides utilities for safe temporary file
handling with automatic cleanup, preventing resource leaks.

Quick Start
-----------

.. code-block:: python

   from RepTate.core.temp_utils import TempFileManager, temp_directory

   # Using context manager for automatic cleanup
   with temp_directory() as tmpdir:
       filepath = tmpdir / 'output.json'
       save_data(filepath)
       process(filepath)
   # Directory and contents automatically deleted

   # Using the manager for more control
   manager = TempFileManager()
   tmpfile = manager.create_temp_file(suffix='.json')
   # ... use file ...
   manager.cleanup()  # Or use atexit registration

Classes
-------

.. autoclass:: TempFileManager
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autofunction:: temp_directory
.. autofunction:: temp_file
.. autofunction:: cleanup_temp_files

Context Managers
----------------

.. code-block:: python

   # Temporary directory
   with temp_directory(prefix='reptate_') as tmpdir:
       # Use tmpdir as Path object
       pass

   # Temporary file
   with temp_file(suffix='.json') as tmpfile:
       # Use tmpfile as Path object
       pass
