Module Dependencies
===================

This document describes the dependency structure of RepTate modules after
the modernization effort.

Dependency Graph
----------------

.. graphviz::

   digraph dependencies {
       rankdir=TB;
       node [shape=box, style=filled, fillcolor=lightblue];

       // Layers
       subgraph cluster_gui {
           label="GUI Layer";
           style=filled;
           fillcolor=lightcyan;

           QApplicationWindow;
           QTheory;
           QDataSet;
           MenuManager;
           DatasetManager;
           ViewCoordinator;
           FileIOController;
           TheoryCompute;
           ParameterController;
           FitController;
       }

       subgraph cluster_app {
           label="Application Layer";
           style=filled;
           fillcolor=lightyellow;

           ApplicationLVE;
           ApplicationLAOS;
           ApplicationNLVE;
           OtherApps [label="..."];
       }

       subgraph cluster_theory {
           label="Theory Layer";
           style=filled;
           fillcolor=lightgreen;

           TheoryBasic;
           TheoryMaxwell;
           TheoryRoliePoly;
           OtherTheories [label="..."];
       }

       subgraph cluster_core {
           label="Core Layer";
           style=filled;
           fillcolor=lightpink;

           DataTable;
           Parameter;
           View;
           interfaces;
           nlsq_fit;
           feature_flags;
           safe_eval;
           serialization;
       }

       // Dependencies
       QApplicationWindow -> {MenuManager DatasetManager ViewCoordinator FileIOController};
       QTheory -> {TheoryCompute ParameterController FitController};

       ApplicationLVE -> QApplicationWindow;
       ApplicationLAOS -> QApplicationWindow;
       ApplicationNLVE -> QApplicationWindow;

       TheoryBasic -> QTheory;
       TheoryMaxwell -> TheoryBasic;
       TheoryRoliePoly -> TheoryBasic;

       TheoryCompute -> nlsq_fit;
       ParameterController -> Parameter;
       DatasetManager -> DataTable;

       nlsq_fit -> interfaces;
       TheoryBasic -> interfaces;
   }

Import Rules
------------

To prevent circular imports:

1. **Core modules import nothing from GUI/Application layers**

2. **GUI components use Protocol interfaces for typing**

3. **Applications import theories inside ``__init__``**

Example (ApplicationLAOS):

.. code-block:: python

   class ApplicationLAOS(QApplicationWindow):
       def __init__(self, name="LAOS", parent=None, **kwargs):
           # Import theories INSIDE constructor to avoid circular import
           from RepTate.theories.TheoryRoliePoly import TheoryRoliePoly
           from RepTate.theories.TheoryUCM import TheoryUCM

           super().__init__(name, parent)
           self.theories[TheoryRoliePoly.thname] = TheoryRoliePoly

Protocol Interfaces
-------------------

Located in ``src/RepTate/core/interfaces.py``:

.. code-block:: python

   @runtime_checkable
   class ITheory(Protocol):
       @property
       def name(self) -> str: ...
       def calculate(self, params: dict, x: Array) -> Array: ...
       def get_parameters(self) -> dict[str, Parameter]: ...

   @runtime_checkable
   class IApplication(Protocol):
       @property
       def name(self) -> str: ...
       def get_theories(self) -> list[type[ITheory]]: ...
       def load_data(self, filepath: str) -> IDataset: ...

External Dependencies
---------------------

Core numerical stack:

- **JAX** (>=0.8.0): Array computation, autodiff
- **NLSQ** (>=0.4.1): Curve fitting
- **interpax**: Interpolation
- **NumPyro** (>=0.14.0): Bayesian inference

GUI:

- **PySide6** (>=6.6.0): Qt bindings

See ``pyproject.toml`` for complete dependency list.
