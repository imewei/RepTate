# RepTate Modernization Migration Strategy

**Document Version**: 1.0
**Date**: 2025-12-30
**Status**: Active Planning

---

## Executive Summary

RepTate requires architectural refactoring to complete the PyQt5→PySide6 and SciPy→JAX+NLSQ migration. This document outlines a phased approach to decouple tightly-coupled modules while maintaining backward compatibility.

**Timeline**: 12 weeks (3 phases × 4 weeks)
**Risk Level**: MEDIUM-HIGH
**Recommended Approach**: Incremental refactoring with regression tests at each step

---

## Current Architecture Issues (Priority Order)

| # | Issue | Impact | Modules Affected | Effort |
|---|-------|--------|------------------|--------|
| 1 | **Circular GUI-Theory Dependency** | ❌ **CRITICAL** - All 60 theories inherit QTheory | theories/*, gui/QTheory.py | 4-6 weeks |
| 2 | **DataTable Visualization Coupling** | ❌ **CRITICAL** - Can't serialize/test data without matplotlib | core/DataTable.py, gui/* | 2-3 weeks |
| 3 | **Application-Theory Direct Imports** | ⚠️ **HIGH** - No plugin architecture, tight coupling | applications/* | 2 weeks |
| 4 | **MultiView in core/** | ⚠️ **MEDIUM** - PySide6 dependency in non-GUI module | core/MultiView.py | 1 week |
| 5 | **Global State (CmdBase.calcmode)** | ⚠️ **MEDIUM** - Thread safety concerns | core/CmdBase.py | 1 week |

---

## Phase 1: Foundation Decoupling (Weeks 1-4)

**Goal**: Establish clean architectural boundaries without breaking existing functionality

### 1.1 Extract ITheory Protocol Implementation

**Current State**:
```python
# theories/TheoryMaxwellModes.py
class TheoryMaxwellModesFrequency(QTheory):  # ❌ Inherits from GUI!
    def __init__(self, name="", parent_dataset=None, ax=None):
        super().__init__(name, parent_dataset, ax)
        self.function = self.MaxwellModesFrequency
```

**Target State**:
```python
# theories/maxwell_modes/core.py (NEW - pure logic)
class MaxwellModesTheory:
    def __init__(self):
        self.parameters = {
            'G': Parameter('G', 1e5, 'Modulus', ParameterType.real),
            'tau': Parameter('tau', 1.0, 'Relaxation time', ParameterType.real),
        }

    def calculate(self, params: dict[str, float], x: Array) -> Array:
        """Pure JAX computation - no GUI dependencies"""
        G = params['G']
        tau = params['tau']
        omega = x
        Gp = G * (omega * tau)**2 / (1 + (omega * tau)**2)
        Gpp = G * omega * tau / (1 + (omega * tau)**2)
        return jnp.stack([Gp, Gpp], axis=-1)

# gui/theories/maxwell_modes.py (NEW - GUI wrapper)
class QMaxwellModesTheory(QTheory):
    def __init__(self, name="", parent_dataset=None, ax=None):
        super().__init__(name, parent_dataset, ax)
        self.theory = MaxwellModesTheory()  # Composition!
        self._setup_ui()

    def Qcalculate(self):
        """Delegate to pure theory logic"""
        params = {k: v.value for k, v in self.parameters.items()}
        x = self._get_x_data()
        result = self.theory.calculate(params, x)
        self._update_plot(result)
```

**Migration Steps**:

1. ✅ **Create interfaces.py** (DONE)
   - Define ITheory protocol
   - Add runtime type checking

2. 🔄 **Extract 5 pilot theories** (Week 1)
   - TheoryMaxwellModes → maxwell_modes/core.py + gui.py
   - TheoryPolynomial → polynomial/core.py + gui.py
   - TheoryDebye → debye/core.py + gui.py
   - TheoryRoliePoly → roliepoly/core.py + gui.py (already mostly JAX)
   - TheoryCarreauYasuda → carreau_yasuda/core.py + gui.py

3. ⬜ **Create migration guide** (Week 2)
   - Document pattern for theory authors
   - Provide boilerplate templates
   - Add migration script to automate 80% of work

4. ⬜ **Batch migrate remaining theories** (Weeks 3-4)
   - 55 theories remaining
   - ~3-4 theories per day (with script assistance)
   - Regression test each batch

**Success Criteria**:
- [ ] All theories implement ITheory protocol
- [ ] No direct QTheory inheritance in theory logic
- [ ] Regression tests pass for all migrated theories
- [ ] Documentation updated with new pattern

**Risk Mitigation**:
- Keep QTheory wrapper classes to maintain backward compatibility
- Flag legacy inheritance with deprecation warnings
- Provide automated migration script

---

### 1.2 Remove matplotlib from DataTable

**Current State**:
```python
# core/DataTable.py
class DataTable:
    def __init__(self, axarr: list[Axes] | None = None, _name: str = ''):
        self.series: list[list[Line2D]] = []  # ❌ Matplotlib objects!

        if axarr is not None:
            for nx in range(len(axarr)):
                series_nx: list[Line2D] = []
                for i in range(DataTable.MAX_NUM_SERIES):
                    ss = axarr[nx].plot([], [], label='', picker=10)
                    ss[0]._name = _name
                    series_nx.append(ss[0])
                self.series.append(series_nx)
```

**Target State**:
```python
# core/DataTable.py (REVISED - pure data)
@dataclass
class DataTable:
    """Pure data container - no visualization dependencies"""
    name: str = ''
    num_columns: int = 0
    num_rows: int = 0
    column_names: list[str] = field(default_factory=list)
    column_units: list[str] = field(default_factory=list)
    data: NDArray[np.floating[Any]] = field(default_factory=lambda: np.zeros((0, 0)))
    extra_tables: dict[str, NDArray[np.floating[Any]]] = field(default_factory=dict)

    # NO matplotlib objects!

# gui/data_visualization.py (NEW - visualization layer)
class DataTableRenderer:
    """Manages matplotlib visualization of DataTable objects"""
    def __init__(self, axes: list[Axes]):
        self.axes = axes
        self._series_cache: dict[str, list[Line2D]] = {}

    def render_data_table(self, table: DataTable, style: PlotStyle) -> None:
        """Create or update matplotlib series for a DataTable"""
        if table.name not in self._series_cache:
            self._create_series(table)
        self._update_series_data(table)
        self._apply_style(style)
```

**Migration Steps**:

1. ⬜ **Create DataTableRenderer class** (Week 2)
   - Extract all matplotlib code from DataTable
   - Handle series creation, updates, styling
   - Maintain backward compatibility via adapter

2. ⬜ **Update DataTable to pure data model** (Week 2)
   - Remove axes parameter from __init__
   - Convert to @dataclass for cleaner serialization
   - Add validation methods

3. ⬜ **Update all DataTable usage sites** (Week 3)
   - File.py: Remove axes parameter
   - QDataSet.py: Use DataTableRenderer
   - QTheory.py: Use DataTableRenderer
   - ~30 files affected

4. ⬜ **Test serialization** (Week 3)
   - Verify SafeSerializer works with new DataTable
   - Test project save/load
   - Benchmark file sizes (should be smaller)

**Success Criteria**:
- [ ] DataTable has zero matplotlib imports
- [ ] DataTable can be pickled/serialized without matplotlib
- [ ] All visualization still works (via DataTableRenderer)
- [ ] Unit tests pass for DataTable without GUI

**Risk Mitigation**:
- Create DataTableLegacyAdapter for backward compatibility
- Keep old DataTable as DataTableDeprecated with warnings
- Migrate incrementally, one file at a time

---

### 1.3 Implement TheoryRegistry Pattern

**Current State**:
```python
# applications/ApplicationLVE.py
from RepTate.theories.TheoryMaxwellModes import TheoryMaxwellModesFrequency
from RepTate.theories.TheoryLikhtmanMcLeish2002 import TheoryLikhtmanMcLeish2002
# ... 8 more direct imports ...

class ApplicationLVE(QApplicationWindow):
    def __init__(self, name="LVE", parent=None):
        super().__init__(name, parent)
        # Theories hardcoded - cannot add plugins
```

**Target State**:
```python
# core/theory_registry.py (NEW)
@dataclass
class TheoryRegistration:
    name: str
    description: str
    theory_class: type[ITheory]
    gui_class: type[QTheory]
    applications: list[str]
    category: str = "General"

class TheoryRegistry:
    _registry: dict[str, TheoryRegistration] = {}

    @classmethod
    def register(cls, reg: TheoryRegistration) -> None:
        """Register a theory for use in applications"""
        if reg.name in cls._registry:
            raise ValueError(f"Theory {reg.name} already registered")
        cls._registry[reg.name] = reg

    @classmethod
    def get_theories_for_app(cls, app_name: str) -> list[TheoryRegistration]:
        """Get all theories compatible with an application"""
        return [
            reg for reg in cls._registry.values()
            if app_name in reg.applications
        ]

# theories/maxwell_modes/__init__.py (NEW)
from RepTate.core.theory_registry import TheoryRegistry, TheoryRegistration
from .core import MaxwellModesTheory
from .gui import QMaxwellModesTheory

TheoryRegistry.register(TheoryRegistration(
    name="MaxwellModes",
    description="Fit Maxwell modes spectrum to G', G'' data",
    theory_class=MaxwellModesTheory,
    gui_class=QMaxwellModesTheory,
    applications=["LVE", "LAOS"],
    category="Linear Viscoelasticity",
))

# applications/ApplicationLVE.py (REVISED)
from RepTate.core.theory_registry import TheoryRegistry

class ApplicationLVE(QApplicationWindow):
    def __init__(self, name="LVE", parent=None):
        super().__init__(name, parent)
        self.available_theories = TheoryRegistry.get_theories_for_app("LVE")
        # No direct imports needed!
```

**Migration Steps**:

1. ⬜ **Create TheoryRegistry** (Week 2)
   - Implement registry pattern
   - Add discovery mechanism
   - Support categories for organization

2. ⬜ **Migrate 5 pilot theories to registry** (Week 3)
   - Same theories as in 1.1
   - Test dynamic loading
   - Verify UI menu generation

3. ⬜ **Update applications to use registry** (Week 3)
   - ApplicationLVE (10 theories)
   - ApplicationLAOS (4 theories)
   - ApplicationMWD (2 theories)
   - Other applications (1-3 theories each)

4. ⬜ **Add plugin loader** (Week 4)
   - Support loading theories from external packages
   - Entry point discovery via setuptools
   - Security validation for external plugins

**Success Criteria**:
- [ ] All applications use TheoryRegistry (no direct imports)
- [ ] Theories can be added without modifying application code
- [ ] Plugin architecture supports external packages
- [ ] UI dynamically populates theory menus

**Risk Mitigation**:
- Keep backward compatibility with direct imports (deprecated)
- Add migration warnings for 1 release cycle
- Provide clear upgrade guide

---

### 1.4 Move MultiView to gui/

**Current State**:
```python
# core/MultiView.py
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout  # ❌ Qt in core!

class MultiView(CmdBase):
    def __init__(self):
        super().__init__()
        # ... Qt widget creation ...
```

**Target State**:
```python
# gui/multi_view.py (MOVED)
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout  # ✅ Qt in gui!

class MultiView:  # No longer in core/
    # Same implementation, just moved
```

**Migration Steps**:

1. ⬜ **Create gui/multi_view.py** (Week 1)
   - Copy MultiView implementation
   - Update imports

2. ⬜ **Update all import statements** (Week 1)
   - Find: `from RepTate.core.MultiView import`
   - Replace: `from RepTate.gui.multi_view import`
   - ~15 files affected

3. ⬜ **Remove core/MultiView.py** (Week 1)
   - Verify no usages remain
   - Run tests

**Success Criteria**:
- [ ] No Qt imports in core/ directory
- [ ] All tests pass
- [ ] No breaking changes to API

**Risk Mitigation**: Very low risk - simple move operation

---

## Phase 2: GUI Separation (Weeks 5-8)

**Goal**: Complete separation of business logic from presentation

### 2.1 Decompose QTheory God Class

**Current State**: QTheory is 1000+ lines handling:
- Parameter management
- Fitting (NLSQ)
- Bayesian inference (NumPyro)
- File I/O
- Plotting
- Error calculation
- UI events

**Target State**: Split into focused components

```
theories/
├── theory_model.py              # Pure computation
│   └── class TheoryModel (implements ITheory)
│
└── gui/
    ├── theory_widget.py         # UI only
    │   └── class QTheoryWidget
    ├── parameter_panel.py       # Parameter table
    ├── fit_panel.py             # Fit controls
    └── inference_panel.py       # Bayesian controls
```

**Migration Steps**:

1. ⬜ **Extract ParameterController** (Week 5)
   - Move parameter table management
   - Handle value updates, constraints
   - Emit change signals

2. ⬜ **Extract FitController** (Week 5)
   - Already exists in gui/controllers/fit_controller.py
   - Integrate with QTheory
   - Remove fitting logic from QTheory

3. ⬜ **Extract InferenceController** (Week 6)
   - Already exists in gui/controllers/inference_controller.py
   - Integrate with QTheory
   - Remove NumPyro logic from QTheory

4. ⬜ **Create QTheoryWidget base** (Week 6)
   - Thin wrapper around controllers
   - Minimal UI setup
   - Delegates all logic to controllers/theory

5. ⬜ **Migrate all theories to new pattern** (Weeks 7-8)
   - Use automated script
   - Test each batch
   - Update documentation

**Success Criteria**:
- [ ] QTheory < 300 lines (down from 1000+)
- [ ] All business logic in controllers/theory models
- [ ] Unit tests for controllers without GUI
- [ ] Regression tests pass

---

### 2.2 Integrate SafeSerializer into Project Workflow

**Current State**: Project save/load uses JSON + tolist() (inefficient)

**Target State**: Use SafeSerializer for compact, safe storage

**Migration Steps**:

1. ⬜ **Update save_reptate()** (Week 5)
   - Replace manual JSON serialization
   - Use SafeSerializer.save()
   - Maintain .rept file format (zip container)

2. ⬜ **Update open_project()** (Week 5)
   - Use SafeSerializer.load()
   - Detect legacy pickle format
   - Auto-migrate with user confirmation

3. ⬜ **Add migration tool** (Week 6)
   - Convert existing .rept files
   - Batch conversion support
   - Backup original files

**Success Criteria**:
- [ ] New projects use SafeSerializer
- [ ] Legacy projects auto-migrate
- [ ] File size reduced 50-80%
- [ ] No pickle usage anywhere

---

## Phase 3: Testing & Validation (Weeks 9-12)

**Goal**: Ensure migration maintains correctness and performance

### 3.1 Regression Testing Suite

**Tests to Add**:

| Test Category | Count | Priority | Effort |
|--------------|-------|----------|--------|
| Theory calculation accuracy | 60 | ❌ **CRITICAL** | 2 weeks |
| GUI integration (Qt) | 30 | ⚠️ **HIGH** | 1 week |
| Project save/load | 10 | ⚠️ **HIGH** | 3 days |
| Fitting convergence | 20 | ⚠️ **HIGH** | 1 week |
| Performance benchmarks | 15 | ⚠️ **MEDIUM** | 3 days |

**Implementation**:

```python
# tests/regression/test_theory_accuracy.py
import pytest
from pathlib import Path
import numpy as np
from RepTate.core.serialization import SafeSerializer

@pytest.mark.regression
@pytest.mark.parametrize("theory_name,data_file,golden_file", [
    ("MaxwellModes", "data/lve/test.tts", "golden/maxwell_modes.npz"),
    ("RoliePoly", "data/nlve/test.shear", "golden/roliepoly.npz"),
    # ... 58 more theories
])
def test_theory_prediction(theory_name, data_file, golden_file):
    """Verify theory predictions match golden reference"""
    # Load test data
    data = load_data(data_file)

    # Load golden reference
    golden = SafeSerializer.load(Path(golden_file))

    # Get theory instance
    theory = TheoryRegistry.get(theory_name)

    # Calculate
    result = theory.calculate(golden['params'], data['x'])

    # Compare (allow 1e-10 relative tolerance for numerical precision)
    np.testing.assert_allclose(
        result, golden['prediction'],
        rtol=1e-10, atol=0,
        err_msg=f"{theory_name} prediction differs from golden reference"
    )
```

**Success Criteria**:
- [ ] 100% theory coverage
- [ ] All tests pass on Linux/Mac/Windows
- [ ] Golden files committed to repo
- [ ] CI/CD integration

---

### 3.2 Performance Benchmarking

**Benchmarks to Track**:

| Operation | Baseline (old) | Target (new) | Tolerance |
|-----------|---------------|--------------|-----------|
| Load 100 MB data file | 2.5s | 2.5s | ±10% |
| Fit Maxwell modes (20 modes) | 5.0s (CPU) | 0.5s (JAX+JIT) | 10x faster |
| Save project (50 files) | 8.0s | 3.0s (SafeSerializer) | 2.5x faster |
| Bayesian inference (1000 samples) | 120s | 60s (JAX+NumPyro) | 2x faster |

**Implementation**:

```python
# tests/benchmarks/test_performance.py
import pytest
import time
from RepTate.core.theory_registry import TheoryRegistry

@pytest.mark.benchmark
def test_maxwell_fit_performance(benchmark):
    """Benchmark Maxwell modes fitting"""
    theory = TheoryRegistry.get("MaxwellModes")
    data = load_test_data("data/lve/benchmark.tts")

    def fit_routine():
        result = theory.fit(data['x'], data['y'])
        return result

    result = benchmark(fit_routine)

    # Assert performance target
    assert benchmark.stats.mean < 0.5, "Fit should complete in < 0.5s"
```

**Success Criteria**:
- [ ] No regressions > 10%
- [ ] JAX operations show 2-10x speedup
- [ ] Memory usage stable or reduced

---

## Risk Assessment & Mitigation

### High-Risk Areas

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **Breaking theory calculations** | MEDIUM | ❌ **CRITICAL** | Regression tests with golden files |
| **Qt signal/slot issues** | HIGH | ⚠️ **HIGH** | GUI integration tests |
| **Data loss in migration** | LOW | ❌ **CRITICAL** | Auto-backup, migration validation |
| **Performance regression** | MEDIUM | ⚠️ **HIGH** | Continuous benchmarking |
| **Incomplete migration** | MEDIUM | ⚠️ **MEDIUM** | Phased approach, feature flags |

### Rollback Strategy

If critical issues arise:

1. **Immediate Rollback** (< 1 hour)
   - Revert to previous Git tag
   - Feature flags to disable new code paths

2. **Partial Rollback** (< 1 day)
   - Keep migrated theories that work
   - Revert problematic ones to legacy code

3. **Forward Fix** (< 3 days)
   - Fix issues in new code
   - Re-run regression tests

---

## Resource Requirements

### Developer Effort

| Phase | Duration | Team Size | Total Person-Weeks |
|-------|----------|-----------|-------------------|
| Phase 1 | 4 weeks | 2 developers | 8 person-weeks |
| Phase 2 | 4 weeks | 2 developers | 8 person-weeks |
| Phase 3 | 4 weeks | 1 developer + 1 tester | 8 person-weeks |
| **Total** | **12 weeks** | **2-3 people** | **24 person-weeks** |

### Infrastructure

- CI/CD: GitHub Actions (already in place)
- Regression test data: ~500 MB
- Golden reference files: ~100 MB
- Documentation updates: ~20 hours

---

## Success Metrics

### Technical Metrics

| Metric | Baseline | Target | Measurement |
|--------|----------|--------|-------------|
| **Circular dependencies** | 138 | 0 | `madge --circular src/` |
| **Qt imports in core/** | 2 files | 0 | `grep -r "PySide6" src/RepTate/core/` |
| **Theory-GUI coupling** | 60 theories | 0 | Code review |
| **Test coverage** | 40% | 75% | `pytest --cov` |
| **JAX migration** | 80% | 100% | Verification script |

### User-Facing Metrics

| Metric | Baseline | Target |
|--------|----------|--------|
| **Application startup** | 3.0s | < 3.5s (no regression) |
| **Project load time** | 8.0s | < 4.0s (50% faster) |
| **Fit convergence** | 5.0s | < 1.0s (5x faster) |
| **Memory usage** | 500 MB | < 550 MB |

---

## Communication Plan

### Stakeholders

| Stakeholder | Role | Communication Frequency |
|-------------|------|------------------------|
| **Core Developers** | Implementation | Daily standups |
| **Power Users** | Beta testing | Weekly updates |
| **Community** | Feedback | Bi-weekly blog posts |
| **Management** | Decision-making | Monthly reports |

### Milestones & Checkpoints

| Week | Milestone | Deliverable | Checkpoint |
|------|-----------|-------------|------------|
| 2 | Phase 1 Pilot | 5 theories extracted | Code review |
| 4 | Phase 1 Complete | All theories extracted | Regression tests pass |
| 6 | Phase 2 Pilot | QTheory decomposed | Integration tests pass |
| 8 | Phase 2 Complete | GUI separation done | Performance benchmarks pass |
| 10 | Phase 3 Testing | Regression suite complete | 100% coverage |
| 12 | Phase 3 Complete | Migration done | Production-ready |

---

## Post-Migration Cleanup

### Deprecation Timeline

| Feature | Deprecation Notice | Removal |
|---------|-------------------|---------|
| Direct theory imports | Week 4 | Week 12 |
| QTheory inheritance in theory logic | Week 6 | Week 12 |
| DataTable with axes parameter | Week 3 | Week 12 |
| Pickle serialization | Week 5 | Week 12 |

### Documentation Updates

- [ ] Update developer guide with new patterns
- [ ] Create theory migration tutorial
- [ ] Update API documentation
- [ ] Write migration blog post for community

---

## Appendix: Code Migration Templates

### Template A: Theory Extraction

**Before**:
```python
# theories/TheoryExample.py
from RepTate.gui.QTheory import QTheory

class TheoryExample(QTheory):
    def __init__(self, name="", parent_dataset=None, ax=None):
        super().__init__(name, parent_dataset, ax)
        self.function = self.example_function

    def example_function(self, f=None):
        # Mixed GUI and logic
        x = np.array([...])
        y = self.calculate_y(x)
        return x, y
```

**After**:
```python
# theories/example/core.py
from RepTate.core.interfaces import ITheory

class ExampleTheory:
    implements ITheory

    def __init__(self):
        self.parameters = {...}

    def calculate(self, params: dict, x: Array) -> Array:
        # Pure JAX computation
        return jnp.array([...])

# theories/example/gui.py
from RepTate.gui.QTheory import QTheory
from .core import ExampleTheory

class QExampleTheory(QTheory):
    def __init__(self, name="", parent_dataset=None, ax=None):
        super().__init__(name, parent_dataset, ax)
        self.theory = ExampleTheory()

    def Qcalculate(self):
        result = self.theory.calculate(self._get_params(), self._get_x())
        self._update_plot(result)
```

### Template B: Registry Registration

```python
# theories/example/__init__.py
from RepTate.core.theory_registry import TheoryRegistry, TheoryRegistration
from .core import ExampleTheory
from .gui import QExampleTheory

TheoryRegistry.register(TheoryRegistration(
    name="Example",
    description="Example theory for demonstration",
    theory_class=ExampleTheory,
    gui_class=QExampleTheory,
    applications=["LVE"],
    category="Examples",
))
```

---

## Conclusion

This migration strategy provides a clear, phased approach to modernizing RepTate's architecture while maintaining stability. The key principles are:

1. **Incremental Changes**: Never break more than 5% of the codebase at once
2. **Continuous Testing**: Regression tests at every step
3. **Backward Compatibility**: Deprecation warnings before removal
4. **Clear Documentation**: Every pattern documented with examples

**Next Steps**:
1. Review and approve this strategy
2. Set up project tracking (GitHub Projects)
3. Begin Phase 1, Week 1 tasks
4. Schedule weekly sync meetings
