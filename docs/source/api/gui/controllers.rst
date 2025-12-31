===============
GUI Controllers
===============

Extracted business logic controllers following MVVM pattern.

.. module:: RepTate.gui
   :synopsis: GUI controller modules

Overview
--------

These controllers were extracted from the legacy god classes
(QApplicationWindow, QTheory, QDataSet) to improve maintainability
and testability.

DatasetManager
==============

.. module:: RepTate.gui.DatasetManager
   :synopsis: Dataset lifecycle management

Manages dataset creation, deletion, and state.

.. autoclass:: DatasetManager
   :members:
   :undoc-members:
   :show-inheritance:

FileIOController
================

.. module:: RepTate.gui.FileIOController
   :synopsis: File operations

Handles file loading, saving, and format conversion.

.. autoclass:: FileIOController
   :members:
   :undoc-members:
   :show-inheritance:

MenuManager
===========

.. module:: RepTate.gui.MenuManager
   :synopsis: Menu management

Creates and manages application menus and actions.

.. autoclass:: MenuManager
   :members:
   :undoc-members:
   :show-inheritance:

ParameterController
===================

.. module:: RepTate.gui.ParameterController
   :synopsis: Parameter management

Handles parameter validation, updates, and synchronization.

.. autoclass:: ParameterController
   :members:
   :undoc-members:
   :show-inheritance:

TheoryCompute
=============

.. module:: RepTate.gui.TheoryCompute
   :synopsis: Theory calculations

Orchestrates theory calculations and fitting operations.

.. autoclass:: TheoryCompute
   :members:
   :undoc-members:
   :show-inheritance:

ViewCoordinator
===============

.. module:: RepTate.gui.ViewCoordinator
   :synopsis: View synchronization

Coordinates view updates and plot synchronization.

.. autoclass:: ViewCoordinator
   :members:
   :undoc-members:
   :show-inheritance:

Usage Pattern
-------------

Controllers are typically used through dependency injection:

.. code-block:: python

   class QApplicationWindow:
       def __init__(self):
           self.file_io = FileIOController(self)
           self.dataset_manager = DatasetManager(self)
           self.menu_manager = MenuManager(self)
           self.view_coordinator = ViewCoordinator(self)

       def load_file(self, path):
           return self.file_io.load(path)
