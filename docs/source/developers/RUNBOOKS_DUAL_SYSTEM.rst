=============================================
RepTate Dual-System Operation Runbooks
=============================================

This document provides operational runbooks for managing the dual-system architecture during the modernization period, including feature flag management, switching between implementations, and troubleshooting.

.. contents:: Table of Contents
   :local:
   :depth: 3

Overview
========

During the modernization period, RepTate runs with both legacy and modern implementations in parallel. Feature flags control which implementation is active, allowing safe rollout and instant rollback.

**Dual-System Period:** Current (2025-12-31) until ~April 2026 (estimated)

**Key Principles:**

- Both implementations must work at all times
- Feature flags control which is active
- Instant rollback via environment variables
- No database migrations (file-based state only)

Feature Flag System
===================

Understanding Feature Flags
----------------------------

Feature flags are boolean toggles that control which implementation is used. They have:

- **Default value:** True/False (in code)
- **Environment override:** REPTATE_<FLAG_NAME>=true|false
- **Scope:** Process-wide (set at startup)

Available Flags
---------------

**Current Flags** (``src/RepTate/core/feature_flags.py``):

.. code-block:: python

    FEATURES = {
        'USE_SAFE_EVAL': True,              # Safe expression evaluator
        'USE_SAFE_SERIALIZATION': True,     # JSON/NPZ serialization
        'USE_JAX_OPTIMIZATION': True,       # JAX-based optimization
    }

**Flag Status:**

+----------------------------+----------+--------------+----------------+
| Flag                       | Default  | Legacy Path  | Modern Path    |
+============================+==========+==============+================+
| USE_SAFE_EVAL              | True     | eval()       | safe_eval      |
+----------------------------+----------+--------------+----------------+
| USE_SAFE_SERIALIZATION     | True     | pickle       | JSON/NPZ       |
+----------------------------+----------+--------------+----------------+
| USE_JAX_OPTIMIZATION       | True     | scipy        | JAX/NLSQ       |
+----------------------------+----------+--------------+----------------+

Checking Current Flag State
----------------------------

.. code-block:: bash

    # Python REPL
    python3 -c "from RepTate.core.feature_flags import get_all_flags; print(get_all_flags())"

    # Expected output:
    # {'USE_SAFE_EVAL': True, 'USE_SAFE_SERIALIZATION': True, 'USE_JAX_OPTIMIZATION': True}

.. code-block:: python

    # In Python code
    from RepTate.core.feature_flags import get_flag_info

    for flag_name, info in get_flag_info().items():
        print(f"{flag_name}:")
        print(f"  Default: {info['default']}")
        print(f"  Current: {info['current']}")
        print(f"  Description: {info['description']}")
        print(f"  Env Var: {info['env_var']}")

Runbook 1: Switching Between Implementations
=============================================

Scenario: Enable Legacy Implementation
---------------------------------------

**When to use:**

- Modern implementation has a critical bug
- Need to compare behavior (debugging)
- User reports issue with modern path

**Steps:**

1. **Set environment variable:**

   .. code-block:: bash

       # Disable modern safe_eval (use legacy eval)
       export REPTATE_USE_SAFE_EVAL=false

       # Disable modern serialization (use legacy pickle)
       export REPTATE_USE_SAFE_SERIALIZATION=false

       # Disable modern JAX optimization (use legacy scipy)
       export REPTATE_USE_JAX_OPTIMIZATION=false

2. **Verify flag state:**

   .. code-block:: bash

       python3 -c "from RepTate.core.feature_flags import get_all_flags; print(get_all_flags())"

       # Expected output:
       # {'USE_SAFE_EVAL': False, 'USE_SAFE_SERIALIZATION': False, 'USE_JAX_OPTIMIZATION': False}

3. **Start RepTate:**

   .. code-block:: bash

       python -m RepTate

4. **Test functionality:**

   - Load data files
   - Fit a theory (to test optimization)
   - Save results (to test serialization)
   - Verify expected behavior

**Rollback:**

.. code-block:: bash

    # Re-enable modern implementations
    unset REPTATE_USE_SAFE_EVAL
    unset REPTATE_USE_SAFE_SERIALIZATION
    unset REPTATE_USE_JAX_OPTIMIZATION

    # Restart RepTate
    python -m RepTate

Scenario: Selective Rollback
-----------------------------

**When to use:**

- Only one modern component has issues
- Want to test specific modernization

**Example: Rollback only serialization:**

.. code-block:: bash

    # Disable only serialization (keep others modern)
    export REPTATE_USE_SAFE_SERIALIZATION=false

    # Verify
    python3 -c "from RepTate.core.feature_flags import get_all_flags; print(get_all_flags())"

    # Expected output:
    # {'USE_SAFE_EVAL': True, 'USE_SAFE_SERIALIZATION': False, 'USE_JAX_OPTIMIZATION': True}

    # Start RepTate
    python -m RepTate

**Clean up:**

.. code-block:: bash

    unset REPTATE_USE_SAFE_SERIALIZATION

Scenario: Temporary Testing
----------------------------

**When to use:**

- Quick test of legacy path
- No permanent configuration change

**One-liner:**

.. code-block:: bash

    # Test with legacy eval (single session)
    REPTATE_USE_SAFE_EVAL=false python -m RepTate

    # Test with legacy serialization
    REPTATE_USE_SAFE_SERIALIZATION=false python -m RepTate

    # Test with legacy optimization
    REPTATE_USE_JAX_OPTIMIZATION=false python -m RepTate

Runbook 2: Troubleshooting Common Issues
=========================================

Issue: "Cannot serialize type" Error
-------------------------------------

**Symptom:**

.. code-block:: text

    TypeError: Cannot serialize type: function

**Cause:**

Modern SafeSerializer doesn't support functions/classes (security restriction).

**Solution 1: Convert to Supported Type**

.. code-block:: python

    # BAD
    data = {
        "transform": lambda x: x ** 2,  # ERROR
    }

    # GOOD
    data = {
        "transform_name": "power_law",
        "transform_params": {"exponent": 2.0},
    }

**Solution 2: Temporary Rollback to Pickle**

.. code-block:: bash

    # If conversion is complex, temporarily use pickle
    export REPTATE_USE_SAFE_SERIALIZATION=false
    python -m RepTate

    # File a bug to fix the serialization issue
    # Re-enable after fix is merged

Issue: "Disallowed operation" in safe_eval
-------------------------------------------

**Symptom:**

.. code-block:: text

    ValueError: Disallowed operation: attribute access (method)

**Cause:**

Expression contains operations not in whitelist (e.g., ``obj.method()``, ``arr[0]``).

**Solution 1: Rewrite Expression**

.. code-block:: python

    # BAD: Attribute access not allowed
    expr = "math.sqrt(x)"  # ERROR: attribute access

    # GOOD: Use whitelisted function
    expr = "sqrt(x)"  # OK

    # BAD: Subscript not allowed
    expr = "arr[0] + arr[1]"  # ERROR: subscript

    # GOOD: Use separate variables
    expr = "arr0 + arr1"  # OK
    variables = {"arr0": arr[0], "arr1": arr[1]}

**Solution 2: Temporary Rollback to eval**

.. code-block:: bash

    # If rewriting is complex, temporarily use eval
    export REPTATE_USE_SAFE_EVAL=false
    python -m RepTate

    # File a bug to add needed operation to whitelist (if safe)
    # Re-enable after whitelist is updated

Issue: JAX "TracerArrayConversionError"
----------------------------------------

**Symptom:**

.. code-block:: text

    TracerArrayConversionError: Attempted boolean conversion of traced array

**Cause:**

Python control flow (``if``, ``while``) on traced JAX arrays.

**Solution 1: Use jnp.where**

.. code-block:: python

    # BAD
    @jit
    def bad_function(x):
        if x > 0:  # ERROR with JAX
            return x ** 2
        else:
            return -x

    # GOOD
    @jit
    def good_function(x):
        return jnp.where(x > 0, x ** 2, -x)

**Solution 2: Use lax.cond**

.. code-block:: python

    from jax import lax

    @jit
    def good_function(x):
        return lax.cond(
            x > 0,
            lambda x: x ** 2,  # True branch
            lambda x: -x,      # False branch
            x
        )

**Solution 3: Temporary Rollback to SciPy**

.. code-block:: bash

    # If fixing is complex, temporarily use scipy
    export REPTATE_USE_JAX_OPTIMIZATION=false
    python -m RepTate

    # File a bug to fix JAX incompatibility
    # Re-enable after fix is merged

Issue: Numerical Differences Between Implementations
-----------------------------------------------------

**Symptom:**

Results differ between legacy and modern implementations (beyond numerical precision).

**Diagnosis:**

1. **Quantify difference:**

   .. code-block:: python

       import numpy as np

       legacy_result = ...  # Run with legacy flag
       modern_result = ...  # Run with modern flag

       abs_diff = np.abs(legacy_result - modern_result)
       rel_diff = abs_diff / np.abs(legacy_result)

       print(f"Max absolute difference: {np.max(abs_diff)}")
       print(f"Max relative difference: {np.max(rel_diff)}")

2. **Check if within tolerance:**

   .. code-block:: python

       # Typical tolerances
       assert np.allclose(modern_result, legacy_result, rtol=1e-10, atol=1e-12)

**Solution 1: Expected (Numerical Precision)**

If differences are small (< 1e-10 relative error), this is expected due to:

- Different algorithms (NLSQ vs scipy.optimize)
- Different compilation (JIT vs interpreted)
- Different linear algebra backends

**Action:** Document the difference, update tests to use appropriate tolerance.

**Solution 2: Unexpected (Bug)**

If differences are large (> 1e-6 relative error):

1. File a bug with reproduction steps
2. Temporarily rollback to legacy implementation
3. Investigate root cause

.. code-block:: bash

    # Rollback while investigating
    export REPTATE_USE_JAX_OPTIMIZATION=false
    python -m RepTate

Issue: Performance Regression
------------------------------

**Symptom:**

Modern implementation is slower than legacy.

**Diagnosis:**

.. code-block:: python

    import time
    from RepTate.core.fitting.nlsq_fit import fit_data

    # Time modern implementation
    start = time.time()
    result = fit_data(theory_func, xdata, ydata, p0)
    modern_time = time.time() - start

    print(f"Modern JAX time: {modern_time:.4f}s")

**Causes and Solutions:**

**1. JIT Compilation Overhead (First Run)**

.. code-block:: python

    # First run: slow (includes compilation)
    result1 = fit_data(...)  # ~2s

    # Second run: fast (compiled)
    result2 = fit_data(...)  # ~0.3s

**Solution:** Warm up JIT cache before timing:

.. code-block:: python

    # Warm up
    _ = fit_data(theory_func, xdata, ydata, p0)

    # Now time
    start = time.time()
    result = fit_data(theory_func, xdata, ydata, p0)
    time_taken = time.time() - start

**2. Small Dataset**

JAX has overhead that's only amortized for larger datasets.

**Solution:** JAX optimization only makes sense for datasets with:

- 100+ data points, or
- 10+ fit parameters, or
- Iterative algorithms (>10 iterations)

For tiny datasets (<50 points), legacy scipy may be faster.

**3. Actual Performance Bug**

If JAX is slower even after JIT compilation and for large datasets:

1. Profile the code
2. File a performance bug
3. Temporarily rollback

.. code-block:: bash

    export REPTATE_USE_JAX_OPTIMIZATION=false
    python -m RepTate

Runbook 3: Migrating User Data
===============================

Scenario: Convert Existing .pkl Files
--------------------------------------

**When:** User has legacy pickle files that need conversion.

**Steps:**

1. **Backup original files:**

   .. code-block:: bash

       cp -r user_data/ user_data_backup/

2. **Run migration script:**

   .. code-block:: bash

       # Migrate single file
       python scripts/migrate_pickle_files.py user_data/experiment.pkl

       # Migrate directory
       python scripts/migrate_pickle_files.py user_data/

3. **Verify conversion:**

   .. code-block:: bash

       # Check that .json and .npz files were created
       ls user_data/
       # Expected: experiment.json, experiment.npz, experiment.pkl.bak

4. **Test loading:**

   .. code-block:: python

       from RepTate.core.serialization import SafeSerializer
       from pathlib import Path

       # Load converted data
       data = SafeSerializer.load(Path("user_data/experiment"))

       # Verify contents
       print(data.keys())

5. **Archive backups:**

   Once verified, archive .pkl.bak files:

   .. code-block:: bash

       mkdir user_data/legacy_backups/
       mv user_data/*.pkl.bak user_data/legacy_backups/

Scenario: Batch Migration
--------------------------

**When:** Many users need to migrate files.

**Script:**

.. code-block:: bash

    #!/bin/bash
    # migrate_all.sh

    # Find all .pkl files
    find /path/to/user_data -name "*.pkl" -type f | while read pkl_file; do
        echo "Migrating: $pkl_file"
        python scripts/migrate_pickle_files.py "$pkl_file"

        if [ $? -eq 0 ]; then
            echo "  ✓ Success"
        else
            echo "  ✗ Failed"
        fi
    done

    echo "Migration complete!"

**Run:**

.. code-block:: bash

    chmod +x migrate_all.sh
    ./migrate_all.sh

Runbook 4: Monitoring Dual-System Health
=========================================

Scenario: Check Implementation Usage
-------------------------------------

**Goal:** Understand which implementation is being used in production.

**Logging:**

Add logging to track feature flag usage:

.. code-block:: python

    # Add to application startup
    import logging
    from RepTate.core.feature_flags import get_all_flags

    logger = logging.getLogger(__name__)

    def log_feature_flags():
        """Log current feature flag state."""
        flags = get_all_flags()
        logger.info("Feature Flags:")
        for name, enabled in flags.items():
            status = "MODERN" if enabled else "LEGACY"
            logger.info(f"  {name}: {status}")

    # Call at startup
    log_feature_flags()

**Analysis:**

.. code-block:: bash

    # Check logs for flag usage
    grep "Feature Flags" reptate.log

    # Expected output:
    # Feature Flags:
    #   USE_SAFE_EVAL: MODERN
    #   USE_SAFE_SERIALIZATION: MODERN
    #   USE_JAX_OPTIMIZATION: MODERN

Scenario: Verify Both Paths Work
---------------------------------

**Goal:** Ensure both legacy and modern implementations remain functional.

**Automated Test:**

.. code-block:: python

    # tests/integration/test_dual_system.py

    import pytest
    import os
    from RepTate.core.feature_flags import is_enabled

    def test_both_serialization_paths():
        """Test both pickle and JSON/NPZ serialization."""
        from RepTate.core.serialization import SafeSerializer
        from pathlib import Path
        import numpy as np

        data = {"value": 1.0, "array": np.array([1, 2, 3])}

        # Test modern path
        os.environ['REPTATE_USE_SAFE_SERIALIZATION'] = 'true'
        result = SafeSerializer.save(Path("/tmp/test_modern"), data)
        loaded_modern = SafeSerializer.load(Path("/tmp/test_modern"))

        # Test legacy path (if still available)
        if 'USE_SAFE_SERIALIZATION' in os.environ:
            os.environ['REPTATE_USE_SAFE_SERIALIZATION'] = 'false'
            # Load with legacy if available
            # ... (test legacy path)

        # Cleanup
        os.environ.pop('REPTATE_USE_SAFE_SERIALIZATION', None)

**Run:**

.. code-block:: bash

    pytest tests/integration/test_dual_system.py -v

Runbook 5: Deprecation Timeline
================================

Scenario: Plan Deprecation
---------------------------

**Timeline:**

+---------------+-------------------------+----------------------------------+
| Phase         | Timeline                | Actions                          |
+===============+=========================+==================================+
| Current       | 2025-12-31              | Both paths functional            |
+---------------+-------------------------+----------------------------------+
| Rollout       | Jan 2026 - Mar 2026     | Modern default, legacy available |
+---------------+-------------------------+----------------------------------+
| Deprecation   | Apr 2026 - Jun 2026     | Deprecation warnings added       |
+---------------+-------------------------+----------------------------------+
| Removal       | Jul 2026 (6 months)     | Legacy code removed              |
+---------------+-------------------------+----------------------------------+

**Rollout Phase (Current):**

- Modern implementations default (USE_*=True)
- Legacy available via environment variables
- Both paths tested in CI/CD

**Deprecation Phase (Apr 2026):**

Add deprecation warnings:

.. code-block:: python

    import warnings
    from RepTate.core.feature_flags import is_enabled

    if not is_enabled('USE_SAFE_EVAL'):
        warnings.warn(
            "Legacy eval() is deprecated and will be removed in RepTate 3.0 (July 2026). "
            "Please report any issues with safe_eval.",
            DeprecationWarning,
            stacklevel=2
        )

**Removal Phase (Jul 2026):**

Remove legacy code and feature flags:

.. code-block:: python

    # Remove legacy code blocks
    # Before:
    if is_enabled('USE_SAFE_EVAL'):
        result = safe_eval(expr, variables)
    else:
        result = eval(expr, {}, variables)  # REMOVE THIS

    # After:
    result = safe_eval(expr, variables)  # Only modern path

Scenario: Communicate Deprecation
----------------------------------

**Channels:**

1. **Release Notes:**

   .. code-block:: text

       RepTate 2.9.0 (April 2026)
       ---------------------------

       **Deprecation Notice:**

       The following legacy implementations are deprecated and will be removed in RepTate 3.0 (July 2026):

       - eval() expression evaluator (use safe_eval)
       - pickle serialization (use JSON/NPZ)
       - scipy.optimize curve fitting (use JAX/NLSQ)

       To continue using legacy implementations temporarily, set environment variables:

           REPTATE_USE_SAFE_EVAL=false
           REPTATE_USE_SAFE_SERIALIZATION=false
           REPTATE_USE_JAX_OPTIMIZATION=false

       Please report any issues with modern implementations before July 2026.

2. **User Documentation:**

   Update migration guide with:

   - Timeline for removal
   - How to report issues
   - How to temporarily use legacy (if needed)

3. **In-App Warnings:**

   .. code-block:: python

       # Show dialog on startup if using legacy
       if not is_enabled('USE_SAFE_EVAL'):
           QMessageBox.warning(
               self,
               "Deprecation Warning",
               "You are using deprecated legacy eval(). "
               "This will be removed in RepTate 3.0 (July 2026). "
               "Please report any issues with safe_eval."
           )

Runbook 6: Rollback Procedures
===============================

Scenario: Emergency Rollback
-----------------------------

**When:** Critical bug found in modern implementation.

**Steps:**

1. **Identify affected component:**

   - Serialization: USE_SAFE_SERIALIZATION
   - Expression eval: USE_SAFE_EVAL
   - Optimization: USE_JAX_OPTIMIZATION

2. **Set environment variable:**

   .. code-block:: bash

       # Example: Rollback serialization
       export REPTATE_USE_SAFE_SERIALIZATION=false

3. **Verify rollback:**

   .. code-block:: bash

       python3 -c "from RepTate.core.feature_flags import get_all_flags; print(get_all_flags())"

4. **Document issue:**

   Create GitHub issue with:

   - Steps to reproduce
   - Expected vs actual behavior
   - Environment (OS, Python version, RepTate version)

5. **Communicate to users:**

   .. code-block:: text

       Subject: Temporary Rollback for [Component]

       We've identified an issue with [modern component]. As a temporary workaround:

       1. Set environment variable: REPTATE_USE_[FLAG]=false
       2. Restart RepTate

       We're working on a fix and will update when available.

6. **Fix and re-enable:**

   Once fixed:

   .. code-block:: bash

       unset REPTATE_USE_SAFE_SERIALIZATION

Scenario: Gradual Re-enablement
--------------------------------

**After fixing a rolled-back component:**

1. **Test fix thoroughly:**

   .. code-block:: bash

       pytest tests/ -v

2. **Enable for developers first:**

   .. code-block:: bash

       # Developer testing
       export REPTATE_USE_SAFE_SERIALIZATION=true
       python -m RepTate

3. **Enable for beta users:**

   Update docs:

   .. code-block:: text

       Beta testers: Please test the fixed serialization by ensuring
       REPTATE_USE_SAFE_SERIALIZATION is not set (or set to true).

4. **Monitor for issues:**

   Wait 1-2 weeks, monitor GitHub issues.

5. **Enable for all users:**

   Release new version with fix, modern implementation as default.

Appendix: Quick Reference
==========================

Feature Flag Cheat Sheet
-------------------------

.. code-block:: bash

    # Check current flags
    python3 -c "from RepTate.core.feature_flags import get_all_flags; print(get_all_flags())"

    # Disable safe_eval (use legacy eval)
    export REPTATE_USE_SAFE_EVAL=false

    # Disable safe serialization (use legacy pickle)
    export REPTATE_USE_SAFE_SERIALIZATION=false

    # Disable JAX optimization (use legacy scipy)
    export REPTATE_USE_JAX_OPTIMIZATION=false

    # Re-enable all (remove overrides)
    unset REPTATE_USE_SAFE_EVAL
    unset REPTATE_USE_SAFE_SERIALIZATION
    unset REPTATE_USE_JAX_OPTIMIZATION

Common Commands
---------------

.. code-block:: bash

    # Test with all legacy
    REPTATE_USE_SAFE_EVAL=false REPTATE_USE_SAFE_SERIALIZATION=false REPTATE_USE_JAX_OPTIMIZATION=false python -m RepTate

    # Test with all modern (default)
    python -m RepTate

    # Migrate pickle files
    python scripts/migrate_pickle_files.py data.pkl

    # Verify SciPy removal
    python scripts/verify_scipy_removal.py

Contact Information
-------------------

**For Issues:**

- GitHub: https://github.com/jorge-ramirez-upm/RepTate/issues
- Documentation: ``docs/source/developers/``

**For Modernization Questions:**

- See: :doc:`MIGRATION_GUIDE_DETAILED`
- See: :doc:`MODERNIZATION_ARCHITECTURE`
