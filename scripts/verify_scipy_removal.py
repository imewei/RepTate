#!/usr/bin/env python3
"""Verify scipy imports have been removed from core modules.

This script checks that scipy.optimize, scipy.interpolate, and scipy.linalg
are not imported in the core RepTate modules, ensuring the JAX migration
is complete.

Usage:
    python scripts/verify_scipy_removal.py

Exit codes:
    0: All checks pass
    1: Disallowed scipy imports found
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Final

# Directories to check (core modules that should be scipy-free)
CORE_DIRS: Final[tuple[str, ...]] = (
    "src/RepTate/core",
    "src/RepTate/theories",
    "src/RepTate/gui",
)

# Directories to skip (tools/applications may still use scipy)
SKIP_DIRS: Final[tuple[str, ...]] = (
    "src/RepTate/tools",
    "src/RepTate/applications",
)

# Scipy submodules that should be replaced
DISALLOWED_SCIPY: Final[tuple[str, ...]] = (
    "scipy.optimize",
    "scipy.interpolate",
    "scipy.linalg",
)

# Allowed scipy imports (for transition period)
ALLOWED_SCIPY: Final[tuple[str, ...]] = (
    "scipy.signal",  # savgol_filter in tools
    "scipy.integrate",  # odeint in tools
)


class SciPyImportFinder(ast.NodeVisitor):
    """AST visitor to find scipy imports."""

    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath
        self.violations: list[tuple[int, str]] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._check_import(node.lineno, alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self._check_import(node.lineno, node.module)
        self.generic_visit(node)

    def _check_import(self, lineno: int, module: str) -> None:
        for disallowed in DISALLOWED_SCIPY:
            if module.startswith(disallowed):
                self.violations.append((lineno, module))


def check_file(filepath: Path) -> list[tuple[int, str]]:
    """Check a single file for disallowed scipy imports."""
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, str(filepath))
        finder = SciPyImportFinder(filepath)
        finder.visit(tree)
        return finder.violations
    except SyntaxError:
        return []


def main() -> int:
    """Main entry point."""
    project_root = Path(__file__).parent.parent
    all_violations: list[tuple[Path, int, str]] = []

    for dir_pattern in CORE_DIRS:
        dir_path = project_root / dir_pattern
        if not dir_path.exists():
            continue

        for py_file in dir_path.rglob("*.py"):
            # Skip files in allowed directories
            skip = False
            for skip_dir in SKIP_DIRS:
                if str(py_file).startswith(str(project_root / skip_dir)):
                    skip = True
                    break
            if skip:
                continue

            violations = check_file(py_file)
            for lineno, module in violations:
                all_violations.append((py_file, lineno, module))

    if all_violations:
        print("Disallowed scipy imports found:")
        print("-" * 60)
        for filepath, lineno, module in sorted(all_violations):
            rel_path = filepath.relative_to(project_root)
            print(f"{rel_path}:{lineno}: {module}")
        print("-" * 60)
        print(f"Total: {len(all_violations)} violations")
        print("\nThese imports should be replaced with:")
        print("  scipy.optimize -> RepTate.core.fitting.nlsq_optimize")
        print("  scipy.interpolate -> interpax")
        print("  scipy.linalg -> jax.numpy.linalg")
        return 1

    print("All scipy removal checks pass!")
    print(f"Checked directories: {', '.join(CORE_DIRS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
