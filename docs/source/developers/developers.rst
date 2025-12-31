======================
RepTate for developers
======================

This section provides comprehensive documentation for developers working on RepTate, including architecture, migration guides, and best practices.

.. contents:: Table of Contents
   :local:
   :depth: 2

Getting Started
===============

New to RepTate development? Start here:

.. toctree::
   :maxdepth: 2

   ONBOARDING_GUIDE
   contributing
   testing

Modernization Documentation
============================

RepTate is undergoing a modernization effort to eliminate security vulnerabilities,
improve performance, and adopt modern scientific computing practices. These documents
provide comprehensive guidance:

.. toctree::
   :maxdepth: 2

   MODERNIZATION_ARCHITECTURE
   MIGRATION_GUIDE_DETAILED
   RUNBOOKS_DUAL_SYSTEM
   LESSONS_LEARNED

**Quick References:**

- ``MODERNIZATION_SUMMARY.md`` - Migration status dashboard (root directory)
- ``TECHNICAL_DEBT_INVENTORY.md`` - Detailed technical debt analysis (root directory)
- ``COMPONENT_READINESS_MATRIX.md`` - Component readiness scores (root directory)

Migration Guides
================

Guides for specific migration tasks:

.. toctree::
   :maxdepth: 2

   migration
   scipy_jax_migration_guide
   progressive_rollout

Architecture and Design
=======================

Understanding RepTate's architecture:

.. toctree::
   :maxdepth: 2

   ../architecture/overview
   ../architecture/dependencies
   ../architecture/data_flow

Legacy Documentation
====================

Original documentation (being updated):

.. toctree::
   :maxdepth: 2

   functionality
   python_c_interface
   code
   callgraphGUI
   todo

Quick Navigation
================

**For New Developers:**

1. Start with :doc:`ONBOARDING_GUIDE` - Get up to speed in 30 minutes
2. Read :doc:`../architecture/overview` - Understand the architecture
3. Follow :doc:`contributing` - Make your first contribution

**For Modernization Work:**

1. Read :doc:`MODERNIZATION_ARCHITECTURE` - Comprehensive architecture overview
2. Follow :doc:`MIGRATION_GUIDE_DETAILED` - Using modern infrastructure
3. Use :doc:`RUNBOOKS_DUAL_SYSTEM` - Operational procedures

**For Troubleshooting:**

1. Check :doc:`RUNBOOKS_DUAL_SYSTEM` - Common issues and solutions
2. See :doc:`LESSONS_LEARNED` - Past challenges and solutions
3. Review ``TECHNICAL_DEBT_INVENTORY.md`` - Known limitations

Documentation Status
====================

**Modernization Documentation:** ✓ Complete (2025-12-31)

- Architecture overview
- Migration guides (JAX, SafeSerializer, safe_eval)
- Operational runbooks
- Lessons learned
- Developer onboarding

**Next Updates:**

- API reference documentation (ongoing)
- Tutorial videos (planned)
- Example gallery (planned)

Contributing to Documentation
==============================

Documentation is as important as code. When contributing:

1. **Update docs with code changes** - Keep docs in sync
2. **Add examples** - Show how to use new features
3. **Document decisions** - Explain why, not just what
4. **Review docs in PRs** - Treat docs as first-class deliverable

**Documentation Sources:**

- Sphinx RST files: ``docs/source/``
- Markdown files: Root directory (``MODERNIZATION_SUMMARY.md``, etc.)
- Code docstrings: Inline documentation

See Also
========

- `RepTate Homepage <http://github.com/jorge-ramirez-upm/RepTate>`_
- `GitHub Issues <https://github.com/jorge-ramirez-upm/RepTate/issues>`_
- `GitHub Discussions <https://github.com/jorge-ramirez-upm/RepTate/discussions>`_
