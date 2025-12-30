#!/usr/bin/env python3
"""Headless smoke tests for theory math without Qt initialization."""
from __future__ import annotations

import os
import sys
import types

import numpy as np


def _install_pyside6_stubs() -> None:
    if "PySide6" in sys.modules:
        return

    pyside6 = types.ModuleType("PySide6")
    pyside6.__version__ = "0.0.0"
    qtwidgets = types.ModuleType("PySide6.QtWidgets")
    qtcore = types.ModuleType("PySide6.QtCore")
    qtgui = types.ModuleType("PySide6.QtGui")
    qtuitools = types.ModuleType("PySide6.QtUiTools")

    class _Dummy:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __call__(self, *args, **kwargs):
            return None

        def __getattr__(self, name):
            return _Dummy()

    class _Signal:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def connect(self, *args, **kwargs) -> None:
            return None

    for name in (
        "QWidget",
        "QMainWindow",
        "QAction",
        "QDialog",
        "QLabel",
        "QLineEdit",
        "QComboBox",
        "QCheckBox",
        "QVBoxLayout",
        "QHBoxLayout",
        "QGridLayout",
        "QPushButton",
        "QTableWidget",
        "QTableWidgetItem",
        "QTreeWidget",
        "QTreeWidgetItem",
        "QTabWidget",
        "QFrame",
        "QSizePolicy",
        "QToolBar",
        "QToolButton",
        "QMenu",
        "QSpinBox",
        "QInputDialog",
        "QMessageBox",
        "QFileDialog",
    ):
        setattr(qtwidgets, name, _Dummy)
    qtwidgets.__getattr__ = lambda name: _Dummy

    for name in ("QSize", "QObject", "QEvent"):
        setattr(qtcore, name, _Dummy)
    qtcore.Signal = _Signal
    qtcore.QStandardPaths = type(
        "QStandardPaths",
        (),
        {
            "AppDataLocation": 0,
            "DocumentsLocation": 1,
            "CacheLocation": 2,
            "writableLocation": staticmethod(lambda *args, **kwargs: ""),
        },
    )
    class _Version:
        def segments(self):
            return (0, 0, 0)

        def toString(self):
            return "0.0.0"

    qtcore.QLibraryInfo = type(
        "QLibraryInfo",
        (),
        {"version": staticmethod(lambda *args, **kwargs: _Version())},
    )
    qtcore.QVersionNumber = type(
        "QVersionNumber",
        (),
        {"toString": lambda self: "0.0.0"},
    )
    qtcore.__getattr__ = lambda name: _Dummy

    for name in ("QIcon",):
        setattr(qtgui, name, _Dummy)
    qtgui.__getattr__ = lambda name: _Dummy

    for name in ("QUiLoader",):
        setattr(qtuitools, name, _Dummy)

    def _load_ui_type(*args, **kwargs):
        ui_cls = type("UiStub", (), {})
        base_cls = type("UiBaseStub", (), {})
        return ui_cls, base_cls

    qtuitools.loadUiType = _load_ui_type
    qtuitools.__getattr__ = lambda name: _Dummy

    sys.modules["PySide6"] = pyside6
    sys.modules["PySide6.QtWidgets"] = qtwidgets
    sys.modules["PySide6.QtCore"] = qtcore
    sys.modules["PySide6.QtGui"] = qtgui
    sys.modules["PySide6.QtUiTools"] = qtuitools
    sys.modules.setdefault("PySide6.support", types.ModuleType("PySide6.support"))
    sys.modules.setdefault("shiboken6", types.ModuleType("shiboken6"))
    sys.modules.setdefault("shibokensupport", types.ModuleType("shibokensupport"))

    sys.modules.setdefault("RepTate.gui.Theory_rc", types.ModuleType("RepTate.gui.Theory_rc"))


def _install_matplotlib_stubs() -> None:
    if "matplotlib" in sys.modules:
        return

    mpl = types.ModuleType("matplotlib")

    class _Dummy:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __call__(self, *args, **kwargs):
            return None

        def __getattr__(self, name):
            return _Dummy()

    mpl.__version__ = "0.0.0"
    mpl.use = lambda *args, **kwargs: None
    mpl.__path__ = []
    mpl.rcParams = {"figure.dpi": 100}
    mpl.__getattr__ = lambda name: _Dummy

    pyplot = types.ModuleType("matplotlib.pyplot")
    pyplot.matplotlib = mpl
    pyplot.rcParams = {"figure.dpi": 100}
    pyplot.__getattr__ = lambda name: _Dummy

    figure = types.ModuleType("matplotlib.figure")
    figure.Figure = _Dummy

    backends = types.ModuleType("matplotlib.backends")
    backends.__path__ = []
    backend_qt5agg = types.ModuleType("matplotlib.backends.backend_qt5agg")
    backend_qt5agg.FigureCanvasQTAgg = _Dummy
    backend_qt5agg.NavigationToolbar2QT = _Dummy
    backend_qtagg = types.ModuleType("matplotlib.backends.backend_qtagg")
    backend_qtagg.FigureCanvasQTAgg = _Dummy
    backend_qtagg.FigureCanvas = _Dummy
    backend_qtagg.NavigationToolbar2QT = _Dummy

    ticker = types.ModuleType("matplotlib.ticker")
    ticker.AutoMinorLocator = _Dummy
    patheffects = types.ModuleType("matplotlib.patheffects")
    patheffects.Stroke = _Dummy
    patheffects.Normal = _Dummy
    gridspec = types.ModuleType("matplotlib.gridspec")
    gridspec.GridSpec = _Dummy

    sys.modules["matplotlib"] = mpl
    sys.modules["matplotlib.pyplot"] = pyplot
    sys.modules["matplotlib.figure"] = figure
    sys.modules["matplotlib.backends"] = backends
    sys.modules["matplotlib.backends.backend_qt5agg"] = backend_qt5agg
    sys.modules["matplotlib.backends.backend_qtagg"] = backend_qtagg
    sys.modules["matplotlib.ticker"] = ticker
    sys.modules["matplotlib.patheffects"] = patheffects
    sys.modules["matplotlib.gridspec"] = gridspec

    mpl.pyplot = pyplot
    mpl.ticker = ticker
    mpl.patheffects = patheffects
    mpl.gridspec = gridspec
    mpl.backends = backends


def _assert_finite(arr: np.ndarray, label: str) -> None:
    if not np.all(np.isfinite(arr)):
        raise AssertionError(f"{label} contains non-finite values")


def test_pets() -> None:
    from RepTate.theories.TheoryPETS import TheoryPETS
    from RepTate.core.jax_ops.ode import rk4_integrate

    theory = TheoryPETS.__new__(TheoryPETS)
    theory.stop_theory_flag = False
    theory.RD_MAX = 1e6

    vec = np.array([0.5, 1.0, 1.0, 1.0, 0.1, 1.0, 1.0, 0.05], dtype=float)
    p = np.array([10.0, 0.01, 10.0, 100.0, 1.0, 100.0, 0.01, 1.0, -0.5, 0.1])
    out = np.asarray(theory.sigmadot_shear(vec, 0.0, p), dtype=float)
    if out.shape != (8,):
        raise AssertionError("TheoryPETS.sigmadot_shear output shape mismatch")
    _assert_finite(out, "TheoryPETS.sigmadot_shear output")

    t = np.linspace(0.0, 0.1, 5)
    traj = rk4_integrate(theory.sigmadot_shear, vec, t, p)
    if traj.shape != (len(t), len(vec)):
        raise AssertionError("TheoryPETS RK4 trajectory shape mismatch")


def test_pompom() -> None:
    from RepTate.theories.TheoryPomPom import TheoryPomPom
    from RepTate.core.jax_ops.ode import rk4_integrate

    theory = TheoryPomPom.__new__(TheoryPomPom)
    theory.stop_theory_flag = False

    p = [2.0, 1.0, 1.0, 0.1]
    out = float(theory.sigmadot_shear(1.2, 0.0, p))
    if not np.isfinite(out):
        raise AssertionError("TheoryPomPom.sigmadot_shear returned non-finite value")

    t = np.linspace(0.0, 0.2, 6)
    traj = rk4_integrate(theory.sigmadot_shear, 1.0, t, p)
    if traj.shape != (len(t), 1):
        raise AssertionError("TheoryPomPom RK4 trajectory shape mismatch")


def test_rolie_poly() -> None:
    from RepTate.theories.TheoryRoliePoly import TheoryRoliePoly
    from RepTate.core.jax_ops.ode import rk4_integrate
    from RepTate.theories.theory_helpers import FeneMode

    theory = TheoryRoliePoly.__new__(TheoryRoliePoly)
    theory.stop_theory_flag = False
    theory.read_gdot_action = type("ReadGdotStub", (), {"isChecked": lambda self: False})()
    theory.with_fene = FeneMode.none

    vec = np.array([1.0, 1.0, 0.0], dtype=float)
    p = [10.0, 100.0, 1.0, 1.0, -0.5, 0.1]
    out = np.asarray(theory.sigmadot_shear(vec, 0.0, p), dtype=float)
    if out.shape != (3,):
        raise AssertionError("TheoryRoliePoly.sigmadot_shear output shape mismatch")
    _assert_finite(out, "TheoryRoliePoly.sigmadot_shear output")

    t = np.linspace(0.0, 0.1, 5)
    traj = rk4_integrate(theory.sigmadot_shear, vec, t, p)
    if traj.shape != (len(t), len(vec)):
        raise AssertionError("TheoryRoliePoly RK4 trajectory shape mismatch")


def main() -> int:
    os.environ.setdefault("MPLBACKEND", "Agg")
    _install_matplotlib_stubs()
    _install_pyside6_stubs()
    sys.path.insert(0, "src")

    tests = [
        ("TheoryPETS", test_pets),
        ("TheoryPomPom", test_pompom),
        ("TheoryRoliePoly", test_rolie_poly),
    ]
    failures = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:
            import traceback

            failures.append((name, exc, traceback.format_exc()))
    if failures:
        for name, exc, tb in failures:
            print(f"[FAIL] {name}: {exc}")
            print(tb)
        return 1
    print("[OK] Headless theory smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
