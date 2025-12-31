# Changelog

All notable changes to RepTate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Feature flags system for gradual rollout of new implementations
- Safe serialization using JSON/NPZ (replaces pickle)
- JAX-based curve fitting with NLSQ library
- Bayesian inference with NumPyro NUTS sampler
- Controller-based GUI architecture (decomposed from god classes)
- Comprehensive regression test suite with golden master tests
- Native library loader with ctypes helpers
- Path utilities for cross-platform compatibility
- Temporary file utilities with automatic cleanup
- Verification scripts for legacy code removal

### Changed
- Migrated from PyQt5 to PySide6 for Qt bindings
- Replaced scipy.optimize with JAX-based optimization
- Decomposed QApplicationWindow into specialized controllers:
  - FileIOController for file operations
  - ViewCoordinator for plot management
  - DatasetManager for dataset operations
  - ParameterController for parameter handling
  - TheoryCompute for theory calculations
  - MenuManager for menu operations

### Deprecated
- SciPy imports in core modules (use JAX/interpax equivalents)
  - `scipy.integrate` → `jax.experimental.ode` or `interpax`
  - `scipy.interpolate` → `interpax`
  - `scipy.signal` → `jax.scipy.signal`
  - `scipy.optimize` → `RepTate.core.fitting.nlsq_optimize`
- Pickle file format (migrate to JSON/NPZ using `scripts/migrate_pickle_files.py`)
- Direct eval() usage (use `RepTate.core.safe_eval` module)
- Direct method access on god classes (use controller-based API)

### Removed
- None yet (removal scheduled for future releases per deprecation timeline)

### Fixed
- Security: Removed arbitrary code execution vulnerabilities from eval/pickle usage
- Improved error handling in file I/O operations
- Cross-platform path handling issues
- Memory leaks in GUI event handlers

### Security
- Implemented safe expression evaluation (no eval/exec)
- Removed pickle deserialization (no arbitrary code execution)
- Added input validation for all user-provided expressions
- Secured file I/O with path traversal prevention

## [1.0.0] - 2024-XX-XX

### Added
- Initial release with modernization branch
- JAX ecosystem integration (jax >= 0.8.0, nlsq >= 0.4.1, numpyro >= 0.14.0)
- Python 3.12+ support with type hints
- Comprehensive test infrastructure

### Migration Notes

For detailed migration instructions, see:
- User Guide: `docs/source/user_guide/migration.rst`
- Developer Guide: `docs/source/developers/migration.rst`
- API Changes: `docs/source/api/migration_guide.md`

**Migrating from pickle to JSON/NPZ:**
```bash
python scripts/migrate_pickle_files.py /path/to/your/data/
```

**Updating imports for SciPy removal:**
```python
# OLD
from scipy.interpolate import interp1d

# NEW
from interpax import interp1d
```

**Using new controller-based API:**
```python
# OLD
app = QApplicationWindow()
app.load_files(file_list)

# NEW
from RepTate.gui.controllers import ApplicationController
app = ApplicationController()
app.file_io.load_files(file_list)
```

---

## Version Comparison Links

- [Unreleased](https://github.com/jorge-ramirez-upm/RepTate/compare/v1.0.0...HEAD)
- [1.0.0](https://github.com/jorge-ramirez-upm/RepTate/releases/tag/v1.0.0)
