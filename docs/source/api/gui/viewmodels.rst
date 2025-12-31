==========
ViewModels
==========

MVVM ViewModels for GUI state management.

.. module:: RepTate.gui.viewmodels
   :synopsis: MVVM ViewModel implementations

Overview
--------

ViewModels provide a clean separation between GUI logic and presentation,
making the code more testable and maintainable.

FitViewModel
============

.. autoclass:: RepTate.gui.viewmodels.fit_viewmodel.FitViewModel
   :members:
   :undoc-members:
   :show-inheritance:

PosteriorViewModel
==================

.. autoclass:: RepTate.gui.viewmodels.posterior_viewmodel.PosteriorViewModel
   :members:
   :undoc-members:
   :show-inheritance:

Usage Pattern
-------------

.. code-block:: python

   from RepTate.gui.viewmodels import FitViewModel

   # Create ViewModel with model reference
   fit_vm = FitViewModel(theory=my_theory)

   # Bind to view updates
   fit_vm.on_result_changed.connect(view.update_display)

   # Execute fitting
   fit_vm.run_fit(parameters)

   # Access results
   print(fit_vm.best_parameters)
   print(fit_vm.residuals)
