==========================================================================================
RepTate: Rheology of Entangled Polymers: Toolkit for the Analysis of Theory and Experiment
==========================================================================================

|Build Status| |Documentation| |License| |Python|

.. |Build Status| image:: https://github.com/jorge-ramirez-upm/RepTate/workflows/CI/badge.svg
   :target: https://github.com/jorge-ramirez-upm/RepTate/actions

.. |Documentation| image:: https://readthedocs.org/projects/reptate/badge/?version=latest
   :target: https://reptate.readthedocs.io/en/latest/

.. |License| image:: https://img.shields.io/badge/License-GPLv3+-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0

.. |Python| image:: https://img.shields.io/badge/python-3.12+-blue.svg
   :target: https://www.python.org/downloads/

RepTate is a free and open source rheology software package for analyzing polymer dynamics
experiments with theoretical models.

Features
--------

- **Modern GUI** - PySide6-based graphical interface
- **Cross-platform** - Windows, Linux, and macOS
- **JAX-first computation** - Modern numerical kernels with GPU/TPU support
- **Bayesian inference** - NumPyro NUTS integration for uncertainty quantification
- **Curve fitting** - NLSQ-based deterministic optimization
- **Extensible** - Add custom theories, applications, and tools

Quick Start
-----------

**Installation:**

.. code-block:: bash

   pip install reptate

**From source:**

.. code-block:: bash

   git clone https://github.com/jorge-ramirez-upm/RepTate.git
   cd RepTate
   pip install -e .

**Run:**

.. code-block:: bash

   python -m RepTate

Documentation
-------------

- Full documentation: https://reptate.readthedocs.io/
- Installation guide: https://reptate.readthedocs.io/installation.html
- Developer guide: https://reptate.readthedocs.io/developers/developers.html

Applications
------------

RepTate includes applications for:

- **LVE** - Linear viscoelasticity
- **NLVE** - Non-linear viscoelastic flows
- **TTS** - Time-temperature superposition
- **MWD** - Molecular weight distribution
- **Gt** - Relaxation modulus
- **Creep** - Creep compliance
- **React** - Reactor simulation
- **SANS** - Small angle neutron scattering
- **Dielectric** - Dielectric spectroscopy
- **LAOS** - Large amplitude oscillatory shear
- **Crystal** - Crystallization

Technology Stack
----------------

==================== ========================================
Component            Technology
==================== ========================================
GUI Framework        PySide6 >= 6.6.0
Numerical Computing  JAX >= 0.8.0, NumPy >= 2.2.0
Curve Fitting        NLSQ >= 0.4.1
Bayesian Inference   NumPyro >= 0.14.0
Interpolation        interpax
Optimization         optimistix >= 0.0.6
Plotting             Matplotlib >= 3.9.0
==================== ========================================

Contributing
------------

Contributions are welcome! See our `Contributing Guide <https://reptate.readthedocs.io/developers/contributing.html>`_.

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

Cite RepTate
------------

Boudara V.A.H, Daniel J. Read and Jorge Ramírez, "RepTate rheology software:
Toolkit for the analysis of theories and experiments", Journal of Rheology 64,
709 (2020). https://doi.org/10.1122/8.0000002

License
-------

RepTate is licensed under the `GNU General Public License v3.0 or later <LICENSE>`_.

Authors
-------

- Jorge Ramirez (jorge.ramirez AT upm.es)
- Victor Boudara (victor.boudara AT gmail.com)

Screenshots
-----------

.. image:: docs/source/images/RepTate_LVE.png
    :width: 400pt
    :align: center

.. image:: docs/source/images/RepTate_React.png
    :width: 400pt
    :align: center
