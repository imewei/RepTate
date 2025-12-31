# RepTate Architecture Documentation

**Last Updated**: 2025-12-30
**Purpose**: Comprehensive architectural analysis for the PyQt5→PySide6 and SciPy→JAX+NLSQ modernization effort

---

## 📚 Documentation Structure

| Document | Purpose | Audience | Priority |
|----------|---------|----------|----------|
| **[dependency_analysis.md](./dependency_analysis.md)** | Full dependency graph, coupling analysis, integration points | Architects, Lead Devs | ❌ **READ FIRST** |
| **[dependency_graph.mmd](./dependency_graph.mmd)** | Visual Mermaid diagrams of module dependencies | All developers | ⚠️ **HIGH** |
| **[migration_strategy.md](./migration_strategy.md)** | Phased migration roadmap (12 weeks, 3 phases) | Project Managers, Devs | ⚠️ **HIGH** |
| **[overview.rst](./overview.rst)** | High-level architecture summary | New contributors | Medium |
| **[dependencies.rst](./dependencies.rst)** | External dependencies documentation | DevOps, Maintainers | Medium |
| **[data_flow.rst](./data_flow.rst)** | Data flow patterns documentation | Developers | Medium |

---

## 🎯 Quick Start

### For Architects & Technical Leads

**Read in this order**:
1. [dependency_analysis.md](./dependency_analysis.md) - **Section 1** (Internal Dependencies)
2. [dependency_graph.mmd](./dependency_graph.mmd) - Visual overview
3. [migration_strategy.md](./migration_strategy.md) - **Executive Summary**

**Key Findings**:
- ❌ **138 circular dependencies** identified
- ❌ **60 theories tightly coupled to Qt** via QTheory inheritance
- ⚠️ **DataTable mixes data + visualization** (matplotlib Line2D in data model)
- ✅ **JAX migration 80% complete** (theories mostly migrated)
- ✅ **New architecture components added** (interfaces, controllers, safe serialization)

---

### For Developers Starting Migration Work

**Read in this order**:
1. [migration_strategy.md](./migration_strategy.md) - **Phase 1** tasks
2. [dependency_analysis.md](./dependency_analysis.md) - **Section 4** (Integration Points)
3. Code templates in [migration_strategy.md](./migration_strategy.md#appendix-code-migration-templates)

**Top 3 Migration Tasks** (start here):
1. **Extract ITheory protocol** - Separate theory logic from GUI (4 weeks)
2. **Remove matplotlib from DataTable** - Pure data model (2 weeks)
3. **Implement TheoryRegistry** - Decouple apps from theories (2 weeks)

---

### For New Contributors

**Read in this order**:
1. [overview.rst](./overview.rst) - Understand overall structure
2. [dependency_graph.mmd](./dependency_graph.mmd) - See module relationships
3. [dependencies.rst](./dependencies.rst) - Understand external libs

**Key Concepts**:
- RepTate has 5 main modules: core/, gui/, theories/, applications/, tools/
- Theories perform scientific calculations (60 different models)
- Applications organize theories by domain (LVE, LAOS, MWD, etc.)
- GUI layer uses PySide6 (Qt6 framework)

---

## 🔍 Architecture at a Glance

### Current State (Before Migration)

```
┌─────────────────────────────────────────────────┐
│  gui/ (40 files)                                │
│  ├── QApplicationManager (main window)         │
│  ├── QApplicationWindow (app container)        │
│  ├── QDataSet (dataset widget)                 │
│  └── QTheory (❌ GOD CLASS - 1000+ lines!)     │
│       ↕️ CIRCULAR DEPENDENCY                    │
└─────────────────────────────────────────────────┘
                    ↕️
┌─────────────────────────────────────────────────┐
│  theories/ (60 files)                           │
│  ├── TheoryBasic (base class)                  │
│  ├── TheoryMaxwellModes                        │
│  ├── TheoryRoliePoly (✅ JAX)                  │
│  └── ... (all inherit QTheory - ❌ BAD!)       │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  core/ (50 files)                               │
│  ├── DataTable (❌ matplotlib objects!)        │
│  ├── MultiView (❌ PySide6 in core!)           │
│  ├── interfaces.py (✅ Protocols)              │
│  └── serialization.py (✅ Safe JSON+NPZ)       │
└─────────────────────────────────────────────────┘
```

**Problems**:
- ❌ Circular dependencies prevent clean separation
- ❌ Business logic mixed with presentation layer
- ❌ Can't test theories without GUI
- ❌ Can't serialize data without matplotlib

---

### Target State (After Migration)

```
┌────────────────────────────────────────────────┐
│  GUI Layer (Presentation)                     │
│  ├── QApplicationManager                      │
│  ├── QApplicationWindow                       │
│  ├── QTheoryWidget (thin wrapper)            │
│  └── DataTableRenderer (visualization)       │
└────────────────────────────────────────────────┘
                    ↓ uses
┌────────────────────────────────────────────────┐
│  Controller Layer (Orchestration)             │
│  ├── FitController (NLSQ fitting)            │
│  ├── InferenceController (Bayesian)          │
│  ├── ParameterController                      │
│  └── TheoryRegistry (plugin system)          │
└────────────────────────────────────────────────┘
                    ↓ uses
┌────────────────────────────────────────────────┐
│  Business Logic Layer (Pure Computation)      │
│  ├── ITheory (protocol - no inheritance!)    │
│  ├── MaxwellModesTheory (pure JAX)           │
│  ├── RoliePolyTheory (pure JAX)              │
│  └── DataTable (pure data - no matplotlib)   │
└────────────────────────────────────────────────┘
```

**Benefits**:
- ✅ Zero circular dependencies
- ✅ Business logic testable without GUI
- ✅ Data models serializable without visualization
- ✅ Plugin architecture for extensibility

---

## 📊 Key Metrics

### Current Architecture Health

| Metric | Value | Status |
|--------|-------|--------|
| **Total Python Files** | 197 | - |
| **Circular Dependencies** | 138 | ❌ **CRITICAL** |
| **Qt Imports in core/** | 2 files | ❌ **Bad** |
| **Theory-GUI Coupling** | 60 theories | ❌ **Bad** |
| **Test Coverage** | 40% | ⚠️ **Medium** |
| **JAX Migration** | 80% | ✅ **Good** |
| **SciPy Removal** | 95% | ✅ **Good** |

### Migration Progress

| Phase | Duration | Status | Completion |
|-------|----------|--------|------------|
| **Phase 1: Foundation** | 4 weeks | 🔄 In Planning | 0% |
| **Phase 2: GUI Separation** | 4 weeks | ⬜ Not Started | 0% |
| **Phase 3: Testing** | 4 weeks | ⬜ Not Started | 0% |

---

## 🚨 Critical Issues Identified

### Issue #1: Circular GUI-Theory Dependency (CRITICAL)

**Problem**: All 60 theories inherit from QTheory (a Qt widget)

**Impact**:
- ❌ Can't test theory logic without GUI
- ❌ Can't run theories headless (e.g., in batch jobs)
- ❌ Tight coupling prevents independent development

**Solution**: Extract ITheory protocol, use composition instead of inheritance

**Timeline**: 4-6 weeks (Phase 1.1)

**See**: [dependency_analysis.md - Section 4.1](./dependency_analysis.md#zone-1-gui-theory-boundary-critical)

---

### Issue #2: DataTable Visualization Coupling (CRITICAL)

**Problem**: DataTable creates matplotlib Line2D objects during construction

**Impact**:
- ❌ Can't serialize DataTable without matplotlib
- ❌ Can't test data operations without GUI
- ⚠️ Inefficient memory usage (duplicate data in Line2D)

**Solution**: Remove matplotlib from DataTable, create separate DataTableRenderer

**Timeline**: 2-3 weeks (Phase 1.2)

**See**: [dependency_analysis.md - Section 4.1](./dependency_analysis.md#zone-2-datatable-visualization-coupling-high)

---

### Issue #3: Application-Theory Direct Imports (HIGH)

**Problem**: Applications import concrete theory classes directly

**Impact**:
- ❌ No plugin architecture
- ⚠️ Adding theories requires modifying application code
- ⚠️ Tight coupling increases maintenance burden

**Solution**: Implement TheoryRegistry pattern for dynamic loading

**Timeline**: 2 weeks (Phase 1.3)

**See**: [dependency_analysis.md - Section 4.1](./dependency_analysis.md#zone-3-circular-application-theory-dependencies-high)

---

## 📈 Migration Roadmap Summary

### Phase 1: Foundation Decoupling (Weeks 1-4)

**Goal**: Establish clean architectural boundaries

**Tasks**:
- ✅ Define ITheory protocol (DONE)
- 🔄 Extract theory logic from QTheory (60 theories)
- ⬜ Remove matplotlib from DataTable
- ⬜ Implement TheoryRegistry
- ⬜ Move MultiView to gui/

**Success Criteria**:
- [ ] All theories implement ITheory
- [ ] DataTable has zero matplotlib imports
- [ ] Applications use TheoryRegistry
- [ ] No Qt imports in core/

**See**: [migration_strategy.md - Phase 1](./migration_strategy.md#phase-1-foundation-decoupling-weeks-1-4)

---

### Phase 2: GUI Separation (Weeks 5-8)

**Goal**: Complete separation of business logic from presentation

**Tasks**:
- ⬜ Decompose QTheory into controllers + view
- ⬜ Create theory GUI wrappers
- ⬜ Implement DataTableRenderer
- ⬜ Integrate SafeSerializer into project workflow

**Success Criteria**:
- [ ] QTheory < 300 lines (down from 1000+)
- [ ] All business logic in controllers
- [ ] Unit tests for controllers without GUI
- [ ] New projects use SafeSerializer

**See**: [migration_strategy.md - Phase 2](./migration_strategy.md#phase-2-gui-separation-weeks-5-8)

---

### Phase 3: Testing & Validation (Weeks 9-12)

**Goal**: Ensure migration maintains correctness and performance

**Tasks**:
- ⬜ Create regression test suite (60 theories)
- ⬜ GUI integration tests (30 tests)
- ⬜ Performance benchmarks (15 benchmarks)
- ⬜ Documentation updates

**Success Criteria**:
- [ ] 100% theory coverage
- [ ] No performance regressions > 10%
- [ ] All tests pass on Linux/Mac/Windows
- [ ] Migration complete

**See**: [migration_strategy.md - Phase 3](./migration_strategy.md#phase-3-testing--validation-weeks-9-12)

---

## 🛠️ Tools & Scripts

### Verification Scripts

Located in `scripts/`:

| Script | Purpose | Usage |
|--------|---------|-------|
| `verify_scipy_removal.py` | Check for remaining SciPy imports | `python scripts/verify_scipy_removal.py` |
| `migrate_pickle_files.py` | Convert legacy .pkl to SafeSerializer | `python scripts/migrate_pickle_files.py data/` |
| `analyze_dependencies.py` | Generate dependency reports | `python scripts/analyze_dependencies.py` |

### Testing Commands

```bash
# Run all tests
pytest tests/

# Run regression tests only
pytest tests/regression/ -m regression

# Run with coverage
pytest --cov=RepTate --cov-report=html tests/

# Run benchmarks
pytest tests/benchmarks/ -m benchmark --benchmark-only
```

### Code Quality Checks

```bash
# Check for circular dependencies
madge --circular src/RepTate

# Find Qt imports in core/
grep -r "PySide6\|PyQt" src/RepTate/core/

# Find SciPy usage
grep -r "from scipy\|import scipy" src/RepTate/

# Lint code
ruff check src/RepTate/
```

---

## 📖 Additional Resources

### Related Documentation

- [Developer Guide](../developers/developers.rst) - Setup and contribution guidelines
- [Migration Guide](../developers/migration.rst) - Step-by-step migration instructions
- [Testing Guide](../developers/testing.rst) - Testing philosophy and patterns
- [Contributing Guide](../developers/contributing.rst) - Contribution workflow

### External References

- [JAX Documentation](https://jax.readthedocs.io/) - JAX numerical computing
- [NLSQ Documentation](https://github.com/imewei/NLSQ) - Curve fitting library
- [PySide6 Documentation](https://doc.qt.io/qtforpython-6/) - Qt6 framework
- [NumPyro Documentation](https://num.pyro.ai/) - Bayesian inference

---

## 🤝 Contributing to Architecture

### Proposing Changes

1. Read [dependency_analysis.md](./dependency_analysis.md) to understand current state
2. Draft Architecture Decision Record (ADR) following existing pattern
3. Open GitHub issue for discussion
4. Submit PR with implementation + updated docs

### Architecture Decision Records

ADRs are documented in [dependency_analysis.md - Section 6](./dependency_analysis.md#6-architecture-decision-records-adrs)

**Current ADRs**:
- ADR-001: Use Protocol-Based Interfaces ✅ Accepted
- ADR-002: JAX as Primary Numerical Backend ✅ Accepted
- ADR-003: Safe Serialization (JSON + NPZ) ✅ Accepted

---

## 📞 Contact & Support

### Questions?

- **Architecture questions**: Open GitHub discussion with `[ARCHITECTURE]` tag
- **Migration help**: See [migration_strategy.md](./migration_strategy.md)
- **Bug reports**: File issue with `architecture` label

### Maintainers

See main [README.rst](../../../README.rst) for current maintainer list

---

## 🗓️ Changelog

### 2025-12-30 - Initial Architecture Analysis

- Created comprehensive dependency analysis (38 KB)
- Generated visual dependency graphs (Mermaid)
- Drafted 12-week migration strategy (23 KB)
- Identified 138 circular dependencies
- Documented 3 critical architectural issues
- Established testing & validation framework

**Next Review**: 2025-01-15 (after Phase 1 kickoff)

---

## License

This documentation is part of RepTate and follows the same GPL-3.0+ license.

Copyright (2017-2025): Jorge Ramirez, Victor Boudara, Universidad Politécnica de Madrid, University of Leeds
