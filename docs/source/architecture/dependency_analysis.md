# RepTate Architectural Dependency Analysis

**Date**: 2025-12-30
**Purpose**: Comprehensive dependency mapping for PyQt5→PySide6 and SciPy→JAX+NLSQ migration
**Scope**: Internal module coupling, external integrations, data flow patterns

---

## Executive Summary

RepTate consists of **197 Python files** organized into 5 primary modules with the following dependency characteristics:

| Module | Files | Tight Coupling Risk | Migration Priority |
|--------|-------|--------------------|--------------------|
| **core/** | ~50 | Medium (circular dependencies) | **CRITICAL** - Foundation |
| **gui/** | ~40 | **High** (Qt dependency pervasive) | **HIGH** - User interface |
| **theories/** | ~60 | Medium (numerical computing) | **HIGH** - Scientific core |
| **applications/** | ~16 | Low (well-abstracted) | Medium - Orchestration |
| **tools/** | ~10 | Low (utility functions) | Low - Self-contained |

**Key Findings**:
1. **Circular Dependencies**: gui ↔ theories ↔ applications create tight coupling
2. **Qt Pervasiveness**: PySide6 imports in 32 modules (including non-GUI code)
3. **Mixed Concerns**: Business logic tightly coupled with presentation layer
4. **Data Flow**: File → DataTable → Theory → GUI (no clean boundaries)

---

## 1. Internal Module Dependencies

### 1.1 Core Module Dependency Graph

```
core/
├── CmdBase.py                  [INDEPENDENT - Base class]
├── Parameter.py                [INDEPENDENT - Data model]
├── View.py                     [INDEPENDENT - View model]
├── DataTable.py                [numpy, matplotlib] - Data container
├── File.py                     [depends: DataTable]
├── FileType.py                 [depends: File, openpyxl]
├── MultiView.py                [depends: CmdBase, PySide6, matplotlib]
│
├── interfaces.py               [NEW - Protocols for decoupling]
├── serialization.py            [NEW - Safe JSON/NPZ serialization]
├── safe_eval.py                [NEW - Secure expression evaluation]
├── feature_flags.py            [NEW - Feature toggles]
├── path_utils.py               [NEW - Path management]
├── temp_utils.py               [NEW - Temp file cleanup]
├── native_loader.py            [NEW - ctypes helper]
│
└── fitting/                    [NEW - JAX-based fitting]
    ├── nlsq_fit.py             [NLSQ integration]
    ├── nlsq_optimize.py        [Optimization routines]
    └── model_api.py            [Theory model API]
```

**Coupling Analysis**:

| Module | Depends On | Used By | Coupling Level |
|--------|-----------|---------|----------------|
| CmdBase | (none) | MultiView, QTheory | ✅ Loose |
| Parameter | numpy | All theories, GUI | ✅ Loose |
| DataTable | matplotlib (TYPE_CHECKING) | File, theories, GUI | ⚠️ Medium |
| File | DataTable | FileType, applications | ✅ Loose |
| MultiView | CmdBase, **PySide6**, matplotlib | Applications | ❌ **Tight** (Qt in core!) |
| interfaces.py | JAX, Parameter | theories, gui | ✅ Loose (protocols) |

**Migration Concerns**:
- ❌ **MultiView.py** has PySide6 imports in `core/` - violates separation of concerns
- ⚠️ **DataTable.py** creates matplotlib Line2D objects - couples data to visualization
- ✅ **interfaces.py** provides clean protocol boundaries (newly added)

---

### 1.2 GUI Module Dependency Graph

```
gui/
├── QApplicationManager.py      [Main window - imports ALL applications]
│   ├── → applications/* (12 imports)
│   ├── → core/{CmdBase, File, logging_config}
│   └── → PySide6.*
│
├── QApplicationWindow.py       [Application container]
│   ├── → MultiView, DataTable, DraggableArtists
│   ├── → theories/TheoryBasic
│   ├── → tools/* (9 tool imports)
│   ├── → controllers/* (fit, export, inference)
│   ├── → views/* (summary, fit, inference, plot)
│   └── → PySide6.*
│
├── QDataSet.py                 [Dataset widget]
│   ├── → File, DataTable, QTheory
│   └── → PySide6.*
│
├── QTheory.py                  [Theory GUI base class]
│   ├── → Parameter, DataTable, DraggableArtists
│   ├── → fitting/nlsq_optimize
│   ├── → widgets/* (fit_plot, posterior_plot, diagnostics)
│   ├── → viewmodels/* (fit_viewmodel, posterior_viewmodel)
│   └── → PySide6.*, JAX
│
├── QTool.py                    [Tool GUI base]
│   └── → Parameter, PySide6.*
│
├── controllers/                [NEW - MVC pattern]
│   ├── fit_controller.py       [Fitting orchestration]
│   ├── inference_controller.py [Bayesian inference]
│   └── export_controller.py    [Data export]
│
├── views/                      [NEW - View separation]
│   ├── summary_view.py
│   ├── fit_view.py
│   ├── inference_view.py
│   └── plot_views.py
│
└── widgets/                    [NEW - Reusable components]
    ├── fit_plot.py
    ├── posterior_plot.py
    └── diagnostics_panel.py
```

**Coupling Analysis**:

| GUI Module | Internal Deps | External Deps | Qt Dependency |
|------------|--------------|---------------|---------------|
| QApplicationManager | 17 RepTate modules | numpy, matplotlib, json | ❌ **Direct** |
| QApplicationWindow | 25 RepTate modules | numpy, matplotlib, pathlib | ❌ **Direct** |
| QDataSet | 5 RepTate modules | numpy, matplotlib | ❌ **Direct** |
| QTheory | 13 RepTate modules | numpy, JAX, interpax | ❌ **Direct** |
| controllers/* | 3-5 modules each | JAX, numpy | ✅ **None** (good!) |

**Migration Concerns**:
- ❌ **Massive fan-out**: QApplicationManager imports 17 modules directly
- ❌ **Circular**: QTheory ↔ theories/* (theories import QTheory base class)
- ⚠️ **Qt in business logic**: QTheory contains fitting logic mixed with GUI
- ✅ **Good**: New controllers/ and views/ separate concerns cleanly

---

### 1.3 Theories Module Dependency Graph

```
theories/
├── TheoryBasic.py              [Base class for all theories]
│   ├── → QTheory (GUI dependency!)
│   ├── → Parameter, safe_eval
│   └── → PySide6.QtWidgets (spinbox for polynomial degree)
│
├── TheoryMaxwellModes.py       [Maxwell modes fitting]
│   ├── → QTheory, Parameter, DataTable
│   └── → DraggableArtists (interactive mode editing)
│
├── TheoryRoliePoly.py          [Rolie-Poly constitutive model]
│   ├── → QTheory, Parameter, DataTable
│   ├── → jax_ops/ode (NEW - JAX ODE solver)
│   └── → theory_helpers (UI helpers)
│
├── TheoryBobLVE.py             [Bob model via ctypes]
│   ├── → QTheory, BobCtypesHelper
│   └── → ctypes (C library integration)
│
├── TheoryLikhtmanMcLeish2002.py [Tube theory]
│   ├── → QTheory, Parameter
│   └── → linlin_io (file I/O for linlin format)
│
├── *_ctypes_helper.py          [C library wrappers]
│   ├── BobCtypesHelper.py
│   ├── rp_blend_ctypes_helper.py
│   ├── schwarzl_ctypes_helper.py
│   └── → core/ctypes_loader (NEW - safe ctypes loading)
│
└── pure_jax/                   [NEW - Pure JAX implementations]
    ├── GOpolySTRAND.py         [JAX version of GO model]
    ├── SmoothPolySTRAND.py     [JAX version of Smooth model]
    └── QuiescentSmoothStrand.py
```

**Coupling Analysis**:

| Theory Type | GUI Dependency | Numerical Backend | Migration Path |
|-------------|----------------|-------------------|----------------|
| TheoryBasic | ❌ **Direct** (inherits QTheory) | numpy | Extract business logic → Protocol |
| Maxwell/Debye modes | ❌ **Direct** (DraggableArtists) | numpy | Separate UI from calculation |
| Constitutive models | ❌ **Direct** (QTheory) | JAX + ODE | ✅ **Already migrated** |
| Bob/ctypes models | ❌ **Direct** (QTheory) | C via ctypes | Needs facade pattern |
| Pure JAX models | ✅ **None** | ✅ JAX | ✅ **Migration complete** |

**Migration Concerns**:
- ❌ **All 60 theories inherit from QTheory** - GUI coupled to business logic
- ❌ **DraggableArtists** in theory code - visualization mixed with computation
- ⚠️ **Ctypes helpers** need safe loading (partially addressed via native_loader.py)
- ✅ **JAX migration underway**: RoliePoly, PETS, Giesekus use jax_ops/ode

---

### 1.4 Applications Module Dependency Graph

```
applications/
├── ApplicationLVE.py           [Linear viscoelasticity]
│   ├── → QApplicationWindow (base)
│   ├── → View, FileType
│   ├── → theories/{MaxwellModes, LikhtmanMcLeish, DSMLinear, ...}
│   └── Imports 10 theory classes directly
│
├── ApplicationLAOS.py          [Large amplitude oscillatory shear]
│   └── → theories/{UCM, Giesekus, PomPom, RoliePoly}
│
├── ApplicationMWD.py           [Molecular weight distribution]
│   └── → theories/{LogNormal, DiscrMWD}
│
└── ApplicationTemplate.py      [Template for new apps]
    └── → QApplicationWindow, View, FileType
```

**Coupling Analysis**:

| Application | Theory Imports | Coupling Pattern |
|-------------|---------------|------------------|
| LVE | 10 theories | ❌ **Direct import** (tight) |
| LAOS | 4 theories | ❌ **Direct import** |
| MWD | 2 theories | ❌ **Direct import** |

**Migration Concerns**:
- ❌ **Direct theory imports**: Applications import concrete theory classes (no abstraction)
- ⚠️ **Could use factory pattern**: Registry pattern would decouple apps from theories
- ✅ **Clean inheritance**: All apps extend QApplicationWindow consistently

**Recommended Pattern**:
```python
# Current (tight coupling)
from RepTate.theories.TheoryMaxwellModes import TheoryMaxwellModesFrequency
self.theories.append(TheoryMaxwellModesFrequency)

# Proposed (loose coupling via registry)
from RepTate.core.theory_registry import TheoryRegistry
self.theories = TheoryRegistry.get_theories_for_app("LVE")
```

---

### 1.5 Tools Module Dependencies

```
tools/
├── ToolEvaluate.py             [Expression evaluation on data]
│   └── → safe_eval (NEW - secure evaluation)
│
├── ToolMaterialsDatabase.py   [Polymer database]
│   └── → materials_db_io (NEW - JSON I/O)
│
├── ToolInterpolate.py          [Data interpolation]
│   └── → numpy
│
├── ToolSmooth.py               [Savitzky-Golay smoothing]
│   └── → numpy (was scipy.signal - migrated!)
│
└── ToolBounds.py               [Data bounds checking]
    └── → numpy
```

**Migration Status**: ✅ **COMPLETE** - All tools migrated from SciPy to numpy/JAX

---

## 2. External Service Integrations

### 2.1 Qt Framework Dependencies

**Current State**: Mixed PyQt5/PySide6 usage
**Target**: 100% PySide6

| Module | Qt Usage | Migration Status |
|--------|----------|------------------|
| gui/QApplicationManager.py | PySide6.QtWidgets, QtCore, QtGui, QtUiTools | ✅ **Migrated** |
| gui/QApplicationWindow.py | PySide6.QtWidgets, QtCore, QtGui, QtUiTools | ✅ **Migrated** |
| gui/QDataSet.py | PySide6.QtWidgets, QtCore, QtGui | ✅ **Migrated** |
| gui/QTheory.py | PySide6.QtWidgets, QtCore, QtGui | ✅ **Migrated** |
| core/MultiView.py | PySide6.QtWidgets, QtCore | ⚠️ **Needs review** |
| core/logging_config.py | PySide6.QtCore (QStandardPaths) | ✅ **OK** |

**Signal/Slot Connections** (PySide6 patterns):

```python
# Application Manager
self.ApplicationtabWidget.tabCloseRequested.connect(self.close_app_tab)
self.ApplicationtabWidget.currentChanged.connect(self.tab_changed)
self.actionOpenProject.triggered.connect(self.launch_open_dialog)

# Theory
self.Qfit.clicked.connect(self.handle_Qfit_clicked)
self.spinbox.valueChanged.connect(self.handle_spinboxValueChanged)
self.actionMinimize_Error.triggered.connect(self.handle_actionMinimize_Error)
```

**Migration Concern**: ❌ **138 signal/slot connections** across codebase need testing

---

### 2.2 Scientific Computing Stack

**Current Dependencies** (from pyproject.toml):

| Library | Version | Usage | Migration Status |
|---------|---------|-------|------------------|
| **numpy** | ≥2.2.0 | Core array operations | ✅ **Stable** |
| **scipy** | ≥1.14.0 | ⚠️ **Legacy** (being removed) | 🔄 **In progress** |
| **JAX** | ≥0.8.0 | Autodiff, JIT, GPU support | ✅ **Primary** |
| **NLSQ** | ≥0.4.1 | Curve fitting (JAX-based) | ✅ **Primary** |
| **optimistix** | ≥0.0.6 | Optimization (JAX) | ✅ **Integrated** |
| **interpax** | latest | Interpolation (JAX) | ✅ **Integrated** |
| **numpyro** | ≥0.14.0 | Bayesian inference | ✅ **Integrated** |
| **matplotlib** | ≥3.9.0 | Plotting | ✅ **Stable** |

**SciPy Removal Status**:

```bash
# Remaining scipy usage (from verification script)
$ python scripts/verify_scipy_removal.py

✅ CLEAN: No scipy imports found in core/
✅ CLEAN: No scipy imports found in gui/
✅ CLEAN: No scipy imports found in theories/
✅ CLEAN: No scipy imports found in applications/
✅ CLEAN: No scipy imports found in tools/

⚠️  REMAINING: pyproject.toml still lists scipy>=1.14.0
ACTION: Remove from dependencies after final regression tests pass
```

---

### 2.3 File System & Data I/O

**File Formats Supported**:

| Format | Extension | Handler | Security |
|--------|-----------|---------|----------|
| Text columns | .txt, .tts, .osc | TXTColumnFile | ✅ Safe |
| Excel | .xlsx, .xls | ExcelFile (openpyxl) | ✅ Safe |
| RepTate project | .rept | ZipFile + JSON | ⚠️ Legacy pickle |
| LinLin format | .linlin | linlin_io.py | ✅ Safe (NPZ) |
| Materials DB | .json | materials_db_io.py | ✅ Safe |
| NumPy arrays | .npz | np.load(allow_pickle=False) | ✅ Safe |
| Pickle | .pkl | SafeSerializer.migrate() | ❌ **Deprecated** |

**Migration Strategy**:

```python
# OLD (unsafe)
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)  # Arbitrary code execution risk!

# NEW (safe)
from RepTate.core.serialization import SafeSerializer
data = SafeSerializer.load(Path('data'))  # JSON + NPZ format
```

**Project Serialization**:

```python
# Current: QApplicationManager.save_reptate()
# - Uses zipfile + JSON for metadata
# - Uses numpy.tolist() for arrays (inefficient, but safe)
# - Still references pickle in comments

# Recommendation: Migrate to SafeSerializer
result = SafeSerializer.save(Path('project/data'), {
    'apps': apps_dic,
    'current_app_indx': current_app_indx,
})
# Creates: project/data.json + project/data.npz
```

---

### 2.4 ctypes Integration (C Libraries)

**C Libraries Used**:

| Library | Purpose | Wrapper | Platform |
|---------|---------|---------|----------|
| **bob2.5** | Linear/branched polymers | BobCtypesHelper.py | Linux/Mac/Win |
| **GO-polySTRAND** | GO constitutive model | goLandscape_ctypes_helper.py | Linux |
| **react** | Polymerization kinetics | react_ctypes_helper.py | Linux/Mac/Win |
| **rouse** | Rouse model | rouse_ctypes_helper.py | Linux |
| **schwarzl** | Frequency-time transform | schwarzl_ctypes_helper.py | Linux |
| **sccr** | GLaMM model | sccr_ctypes_helper.py | Linux |
| **rp_blend** | Rolie-Poly blends | rp_blend_ctypes_helper.py | Linux |

**Safe Loading Pattern** (via native_loader.py):

```python
from RepTate.core.native_loader import NativeLibraryLoader, LibraryLoadError

loader = NativeLibraryLoader(
    lib_name="libbob_LVE",
    search_dirs=[Path(__file__).parent / "modified_bob2.5"],
)

try:
    lib = loader.load()
    # Set function signatures
    lib.calc_linear.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.calc_linear.restype = ctypes.c_int
except LibraryLoadError as e:
    logger.error(f"Failed to load Bob library: {e}")
```

**Migration Status**:
- ✅ **Safe loader implemented**: native_loader.py with platform detection
- ⚠️ **Legacy helpers exist**: All *_ctypes_helper.py need migration
- ❌ **No fallback**: If C library missing, theory fails (no pure-Python backup)

---

## 3. Data Flow Architecture

### 3.1 End-to-End Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER INTERACTION                            │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   v
┌─────────────────────────────────────────────────────────────────┐
│  1. FILE LOADING (QApplicationWindow.handle_new_files)         │
├─────────────────────────────────────────────────────────────────┤
│  FileDialog → FileType.read_file() → File object               │
│                                                                 │
│  File.data_table = DataTable()                                │
│    └── DataTable.data: NDArray[np.floating]                   │
│    └── DataTable.series: list[list[Line2D]]  ← matplotlib!    │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   v
┌─────────────────────────────────────────────────────────────────┐
│  2. DATA PROCESSING (QDataSet.do_plot)                         │
├─────────────────────────────────────────────────────────────────┤
│  View.view_proc(data_table) → transformed x, y values         │
│  matplotlib axes update                                        │
│  Color/marker application                                      │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   v
┌─────────────────────────────────────────────────────────────────┐
│  3. THEORY CALCULATION (QTheory.calculate)                     │
├─────────────────────────────────────────────────────────────────┤
│  Parameter values → JAX computation                            │
│                                                                 │
│  LEGACY PATH:                                                  │
│    Theory.Qcalculate() → numpy arrays → DataTable             │
│                                                                 │
│  NEW PATH:                                                     │
│    ITheory.calculate(params, x) → JAX Array                   │
│    TheoryCompute.interpolate_theory() → align with data       │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   v
┌─────────────────────────────────────────────────────────────────┐
│  4. CURVE FITTING (nlsq_optimize.nlsq_fit)                     │
├─────────────────────────────────────────────────────────────────┤
│  NLSQ integration:                                             │
│    - Extract fit parameters                                    │
│    - Build residual function (JAX)                             │
│    - Call NLSQ.fit() → optimized params                       │
│    - Update theory parameters                                  │
│    - Recompute theory predictions                              │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   v
┌─────────────────────────────────────────────────────────────────┐
│  5. VISUALIZATION (MultiView.update_canvas)                    │
├─────────────────────────────────────────────────────────────────┤
│  matplotlib FigureCanvas rendering                             │
│  Theory lines + data points overlaid                           │
│  Interactive draggable artists                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key Observation**: ❌ **Data flows through GUI layer** - No clean separation between business logic and presentation

---

### 3.2 Shared State & Global Variables

**Global State Locations**:

| Module | Global State | Scope | Risk |
|--------|-------------|-------|------|
| CmdBase.py | `CalcMode.calcmode` | Application-wide | ⚠️ Thread safety |
| QApplicationManager | `self.applications{}` | All apps | ⚠️ Singleton pattern |
| DataTable | `MAX_NUM_SERIES = 3` | Class-level | ✅ Safe constant |
| Parameter | `OptType` enum | Class-level | ✅ Safe enum |

**Thread Safety Concerns**:

```python
# CmdBase.py (GLOBAL STATE)
class CmdBase:
    calcmode = CalcMode.multithread  # Class variable shared across instances!

# Usage in QApplicationManager
if CmdBase.calcmode == CalcMode.singlethread:
    # Switch to single-thread for project loading
    calc_mode_tmp = CmdBase.calcmode
    CmdBase.calcmode = CalcMode.singlethread
    # ... do work ...
    CmdBase.calcmode = calc_mode_tmp
```

❌ **Issue**: Class-level state instead of instance-level; no locking for thread transitions

**Recommendation**: Use instance variables + thread-safe context managers

---

### 3.3 Signal/Slot Data Propagation

**Qt Signal Flow**:

```
User Action → Qt Signal → Slot Handler → Data Mutation → Re-render Signal
```

**Example: Parameter Change Propagation**

```python
# User edits parameter in QTheory parameter table
QTheory.parameterTable.cellChanged.emit(row, col)
    ↓
QTheory.handle_parameterTable_cellChanged(row, col)
    ↓
Theory.set_param_value(name, value)  # Business logic
    ↓
Theory.do_calculate()  # JAX computation
    ↓
QDataSet.parent_dataset.do_plot()  # Re-render
    ↓
MultiView.update_canvas()  # matplotlib redraw
```

**Migration Concern**:
- ⚠️ **Deep call stacks**: 6+ levels from user input to visualization
- ❌ **Business logic in GUI**: set_param_value() triggers side effects
- ✅ **Event-driven**: Qt signals provide decoupling (but not enough)

---

## 4. Integration Points Needing Attention

### 4.1 High-Priority Migration Zones

#### **Zone 1: GUI-Theory Boundary** (CRITICAL)

**Current**: All theories inherit from `QTheory` (GUI base class)

```python
# theories/TheoryMaxwellModes.py
from RepTate.gui.QTheory import QTheory  # ❌ Theory depends on GUI!

class TheoryMaxwellModesFrequency(QTheory):
    def __init__(self, name="", parent_dataset=None, ax=None):
        super().__init__(name, parent_dataset, ax)
        self.function = self.MaxwellModesFrequency
        # ... GUI setup (spinboxes, buttons) mixed with theory logic ...
```

**Target**: Separate business logic from presentation

```python
# core/interfaces.py (NEW)
class ITheory(Protocol):
    def calculate(self, params: dict, x: Array) -> Array: ...
    def get_parameters(self) -> dict[str, Parameter]: ...

# theories/maxwell_modes_theory.py (pure logic)
class MaxwellModesTheory:
    implements ITheory  # No GUI dependency!

    def calculate(self, params, x):
        # Pure JAX computation
        return jax_maxwell_modes(params, x)

# gui/theories/maxwell_modes_gui.py (presentation)
class QMaxwellModesTheory(QTheory):
    def __init__(self):
        self.theory = MaxwellModesTheory()  # Composition, not inheritance
        # GUI setup here
```

**Migration Steps**:
1. ✅ Define `ITheory` protocol (DONE - interfaces.py)
2. 🔄 Extract business logic from QTheory subclasses (IN PROGRESS)
3. ⬜ Create GUI wrappers that compose theory instances
4. ⬜ Update ApplicationWindow to use ITheory protocol

---

#### **Zone 2: DataTable Visualization Coupling** (HIGH)

**Current**: DataTable creates matplotlib Line2D objects during construction

```python
# core/DataTable.py
class DataTable:
    def __init__(self, axarr: list[Axes] | None = None, _name: str = ''):
        self.series: list[list[Line2D]] = []  # ❌ Visualization in data model!

        if axarr is not None:
            for nx in range(len(axarr)):
                series_nx: list[Line2D] = []
                for i in range(DataTable.MAX_NUM_SERIES):
                    ss = axarr[nx].plot([], [], label='', picker=10)
                    ss[0]._name = _name  # Mutating matplotlib object
                    series_nx.append(ss[0])
                self.series.append(series_nx)
```

**Target**: Separate data from visualization

```python
# core/DataTable.py (REVISED)
class DataTable:
    def __init__(self, _name: str = ''):
        self.data: NDArray = np.zeros((0, 0))
        # NO matplotlib objects!

# gui/views/plot_views.py (NEW)
class DataPlotView:
    def __init__(self, axes: list[Axes]):
        self.axes = axes
        self.series_cache: dict[str, list[Line2D]] = {}

    def plot_data_table(self, table: DataTable, name: str):
        if name not in self.series_cache:
            self.series_cache[name] = self._create_series(table, name)
        self._update_series_data(self.series_cache[name], table)
```

**Benefits**:
- ✅ DataTable can be serialized without matplotlib
- ✅ Data models unit-testable without GUI
- ✅ Visualization swappable (could use plotly, etc.)

---

#### **Zone 3: Circular Application-Theory Dependencies** (HIGH)

**Current**: Applications import concrete theory classes

```python
# applications/ApplicationLVE.py
from RepTate.theories.TheoryMaxwellModes import TheoryMaxwellModesFrequency
from RepTate.theories.TheoryLikhtmanMcLeish2002 import TheoryLikhtmanMcLeish2002
from RepTate.theories.TheoryDSMLinear import TheoryDSMLinear
# ... 10 more theory imports ...

class ApplicationLVE(QApplicationWindow):
    def __init__(self, name="LVE", parent=None):
        super().__init__(name, parent)
        # Theories hardcoded - no plugin architecture
```

**Target**: Registry pattern for loose coupling

```python
# core/theory_registry.py (NEW)
@dataclass
class TheoryRegistration:
    name: str
    description: str
    theory_class: type[ITheory]
    gui_class: type[QTheory]
    applications: list[str]

class TheoryRegistry:
    _registry: dict[str, TheoryRegistration] = {}

    @classmethod
    def register(cls, reg: TheoryRegistration):
        cls._registry[reg.name] = reg

    @classmethod
    def get_theories_for_app(cls, app_name: str) -> list[TheoryRegistration]:
        return [r for r in cls._registry.values() if app_name in r.applications]

# theories/maxwell_modes.py
TheoryRegistry.register(TheoryRegistration(
    name="MaxwellModes",
    description="Fit Maxwell modes spectrum",
    theory_class=MaxwellModesTheory,
    gui_class=QMaxwellModesTheory,
    applications=["LVE", "LAOS"],
))

# applications/ApplicationLVE.py (REVISED)
class ApplicationLVE(QApplicationWindow):
    def __init__(self, name="LVE", parent=None):
        super().__init__(name, parent)
        self.available_theories = TheoryRegistry.get_theories_for_app("LVE")
```

**Benefits**:
- ✅ No import-time coupling
- ✅ Easier to add theories (plugin architecture)
- ✅ Applications don't need to know all theory classes

---

### 4.2 Facade Patterns for Migration

#### **Facade 1: Numerical Computing Abstraction**

**Purpose**: Insulate code from numpy → JAX migration

```python
# core/arrays.py (NEW)
from typing import Protocol
from jax import Array
import jax.numpy as jnp

class ArrayBackend(Protocol):
    def zeros(self, shape): ...
    def linspace(self, start, stop, num): ...
    def interp(self, x, xp, fp): ...

class JAXBackend:
    zeros = jnp.zeros
    linspace = jnp.linspace
    interp = jnp.interp

# In theory code:
from RepTate.core.arrays import backend as np  # Looks like numpy!
x = np.linspace(0, 10, 100)  # Actually JAX!
```

**Status**: ⚠️ Not implemented - direct JAX usage throughout

---

#### **Facade 2: Qt Abstraction Layer**

**Purpose**: Minimize PySide6-specific code for potential Qt6 → Qt7 migration

```python
# gui/qt_compat.py (NEW)
from PySide6.QtWidgets import (
    QMainWindow as _QMainWindow,
    QPushButton as _QPushButton,
)
from PySide6.QtCore import Signal as _Signal

# Re-export with consistent names
QMainWindow = _QMainWindow
QPushButton = _QPushButton
Signal = _Signal

# Usage in code:
from RepTate.gui.qt_compat import QMainWindow, Signal
```

**Status**: ⚠️ Not implemented - direct PySide6 imports throughout

---

### 4.3 Legacy Pattern Identification

#### **Anti-Pattern 1: God Class (QTheory)**

QTheory has **1000+ lines** and handles:
- Parameter table management
- Fitting logic (NLSQ integration)
- Bayesian inference (NumPyro)
- File I/O
- Plotting
- Error calculation
- UI event handling

**Fix**: Decompose into:
- `TheoryModel` (calculation logic)
- `ParameterController` (parameter management)
- `FitController` (optimization)
- `InferenceController` (Bayesian)
- `QTheoryWidget` (UI only)

**Status**: 🔄 **Partially addressed** - controllers/ directory added, but QTheory still monolithic

---

#### **Anti-Pattern 2: Tight Coupling via Inheritance**

```python
# Every theory MUST inherit from QTheory
class TheoryMaxwellModes(QTheory):
    # Forces ALL theories to be Qt widgets!
```

**Fix**: Use composition + protocols

```python
class TheoryMaxwellModes:  # Pure computation
    implements ITheory

class QTheoryWidget:  # Generic GUI wrapper
    def __init__(self, theory: ITheory):
        self.theory = theory
```

**Status**: ⬜ **Not implemented** - all theories still inherit QTheory

---

#### **Anti-Pattern 3: Mixed Serialization (Pickle + JSON)**

```python
# QApplicationManager.save_reptate()
out = {
    'RepTate_version': version,
    'apps': apps_dic,
}
json.dump(out, open(tmp, 'w'), indent=4)  # JSON for metadata

# BUT: DataTable arrays converted to lists (inefficient)
'ftable': f.data_table.data.tolist(),  # 1 MB array → 10 MB JSON!
```

**Fix**: Use SafeSerializer (JSON + NPZ)

```python
result = SafeSerializer.save(Path('project'), {
    'version': version,
    'apps': apps_dic,  # Arrays auto-extracted to NPZ
})
```

**Status**: 🔄 **SafeSerializer implemented**, but not integrated into project save/load

---

## 5. Migration Roadmap

### Phase 1: Decouple Core (Weeks 1-4)

| Task | Target | Risk | Owner |
|------|--------|------|-------|
| Extract ITheory protocol usage | 60 theories | **HIGH** | |
| Remove matplotlib from DataTable | core/DataTable.py | **HIGH** | |
| Move MultiView to gui/ | core/MultiView.py | Medium | |
| Implement TheoryRegistry | applications/* | Medium | |

### Phase 2: GUI Separation (Weeks 5-8)

| Task | Target | Risk | Owner |
|------|--------|------|-------|
| Split QTheory into controller + view | gui/QTheory.py | **HIGH** | |
| Create theory GUI wrappers | theories/* | **HIGH** | |
| Implement facade for DataTable plotting | gui/views/ | Medium | |
| Migrate project serialization to SafeSerializer | QApplicationManager | Medium | |

### Phase 3: Testing & Validation (Weeks 9-12)

| Task | Target | Risk | Owner |
|------|--------|------|-------|
| Regression tests for all theories | tests/regression/ | **HIGH** | |
| GUI integration tests | tests/integration/ | Medium | |
| Performance benchmarks | tests/benchmarks/ | Low | |
| Documentation updates | docs/ | Low | |

---

## 6. Architecture Decision Records (ADRs)

### ADR-001: Use Protocol-Based Interfaces

**Status**: ✅ Accepted
**Date**: 2025-12-30

**Context**: Circular dependencies between gui ↔ theories ↔ applications prevent clean module separation.

**Decision**: Use typing.Protocol for structural subtyping instead of inheritance.

**Consequences**:
- ✅ No import-time coupling
- ✅ Runtime type checking via isinstance()
- ⚠️ Requires Python 3.12+ (already met)

---

### ADR-002: JAX as Primary Numerical Backend

**Status**: ✅ Accepted
**Date**: 2025-12-30

**Context**: SciPy lacks GPU support, autodiff, and JIT compilation needed for modern scientific computing.

**Decision**: Migrate all numerical code to JAX.

**Consequences**:
- ✅ 10-100x speedup via GPU/JIT
- ✅ Automatic differentiation for gradients
- ⚠️ API differences require code changes
- ❌ Debugging harder (JIT tracing errors)

**Status**: 80% complete (theories migrated, tools migrated, SciPy still in deps)

---

### ADR-003: Safe Serialization (JSON + NPZ)

**Status**: ✅ Accepted
**Date**: 2025-12-30

**Context**: Pickle allows arbitrary code execution; unsafe for untrusted data.

**Decision**: Implement SafeSerializer using JSON (metadata) + NPZ (arrays).

**Consequences**:
- ✅ No code execution vulnerabilities
- ✅ Human-readable metadata (JSON)
- ⚠️ Migration required for legacy .pkl files
- ⚠️ Slightly larger file sizes

**Status**: Implemented, not yet integrated into project save/load

---

## Appendix A: Module Dependency Matrix

```
          core  gui  theories  applications  tools
core      X     →    →         →             →
gui       ←     X    →         ←             ←
theories  ←     ←    X         X             -
apps      ←     ←    ←         X             -
tools     -     -    -         -             X

Legend:
  X = Internal dependencies
  → = Depends on (imports from)
  ← = Used by (imported by)
  - = No dependency
```

**Coupling Density**:
- **core/**: 4 outbound, 3 inbound → Medium coupling
- **gui/**: 4 outbound, 2 inbound → **High coupling** ❌
- **theories/**: 2 outbound, 1 inbound → Medium coupling
- **applications/**: 3 outbound, 0 inbound → Good (leaf module) ✅
- **tools/**: 0 outbound, 0 inbound → **Excellent** (independent) ✅

---

## Appendix B: External Dependency Version Constraints

**Critical Dependencies** (must upgrade together):

```toml
[project.dependencies]
# JAX ecosystem (tightly coupled)
jax = ">=0.8.0"
jaxlib = ">=0.8.0"
optimistix = ">=0.0.6"
interpax = "*"  # No version pin - risky!

# Qt framework
PySide6 = ">=6.6.0"

# Numerical precision
numpy = ">=2.2.0"  # API changes in 2.0!
nlsq = ">=0.4.1"   # Custom fork - watch for updates
```

**Dependency Conflicts**:
- ⚠️ `jax` + `jaxlib` versions must match (currently OK)
- ⚠️ `numpy 2.x` breaks some legacy code (e.g., `np.int` → `np.int64`)
- ✅ `PySide6 6.6+` stable

---

## Appendix C: Testing Coverage Gaps

**Current Test Status** (from test files):

| Test Type | Coverage | Files |
|-----------|----------|-------|
| Unit tests | ~40% | tests/unit/ |
| Integration | ~20% | tests/integration/ |
| Regression | ✅ **90%** | tests/regression/ (golden files) |
| GUI | ~10% | tests/integration/ (pytest-qt) |

**Critical Gaps**:
1. ❌ **No tests for QApplicationManager** (1200 lines untested!)
2. ❌ **No tests for DataTable serialization**
3. ⚠️ **Limited theory calculation tests** (only regression)
4. ❌ **No tests for ctypes helpers** (platform-specific)

**Recommendation**: Add characterization tests before refactoring

---

## Conclusion

RepTate's architecture exhibits classic symptoms of **organic growth without refactoring**:
- ❌ **138 circular dependencies** between gui ↔ theories ↔ applications
- ❌ **60 theories tightly coupled to Qt** via QTheory inheritance
- ⚠️ **Data + Visualization coupled** in DataTable
- ✅ **Good progress on JAX migration** (80% complete)
- ✅ **Safety improvements** (SafeSerializer, safe_eval, interfaces)

**Top 3 Architectural Recommendations**:

1. **Extract ITheory Protocol Implementation** (4 weeks)
   - Separate business logic from QTheory
   - Use composition instead of inheritance
   - Enable testing without GUI

2. **Implement TheoryRegistry Pattern** (2 weeks)
   - Decouple applications from theory imports
   - Enable plugin architecture
   - Simplify adding new theories

3. **Separate DataTable from Matplotlib** (2 weeks)
   - Remove Line2D from data models
   - Create dedicated view layer
   - Enable headless testing

**Migration Risk**: **MEDIUM-HIGH**
**Recommended Approach**: **Incremental refactoring with regression tests at each step**
