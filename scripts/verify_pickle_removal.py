#!/usr/bin/env python3
"""Verify pickle imports have been removed from core modules.

This script checks that pickle is not imported in core RepTate modules,
ensuring the safe serialization migration is complete.

Usage:
    python scripts/verify_pickle_removal.py

Exit codes:
    0: All checks pass
    1: Disallowed pickle imports found
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Final

# Directories to check (core modules that should be pickle-free)
CORE_DIRS: Final[tuple[str, ...]] = (
    "src/RepTate/core",
    "src/RepTate/theories",
    "src/RepTate/gui",
    "src/RepTate/applications",
)

# Files allowed to have pickle imports (migration and legacy support)
ALLOWED_FILES: Final[tuple[str, ...]] = (
    "src/RepTate/core/serialization.py",  # Has controlled pickle usage for migration
    "scripts/migrate_pickle_files.py",  # Migration tool
)


class PickleImportFinder(ast.NodeVisitor):
    """AST visitor to find pickle imports."""

    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath
        self.violations: list[tuple[int, str]] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if "pickle" in alias.name.lower():
                self.violations.append((node.lineno, alias.name))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and "pickle" in node.module.lower():
            self.violations.append((node.lineno, node.module))
        self.generic_visit(node)


def check_file(filepath: Path) -> list[tuple[int, str]]:
    """Check a single file for disallowed pickle imports."""
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, str(filepath))
        finder = PickleImportFinder(filepath)
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
            # Skip allowed files
            rel_path = py_file.relative_to(project_root)
            if str(rel_path) in ALLOWED_FILES:
                continue

            # Skip auto-generated resource files
            if py_file.name.endswith("_rc.py") or py_file.name.endswith("_ui.py"):
                continue

            violations = check_file(py_file)
            for lineno, module in violations:
                all_violations.append((py_file, lineno, module))

    if all_violations:
        print("Disallowed pickle imports found:")
        print("-" * 60)
        for filepath, lineno, module in sorted(all_violations):
            rel_path = filepath.relative_to(project_root)
            print(f"{rel_path}:{lineno}: {module}")
        print("-" * 60)
        print(f"Total: {len(all_violations)} violations")
        print("\nPickle imports should only exist in:")
        print("  - src/RepTate/core/serialization.py (migration support)")
        print("  - scripts/migrate_pickle_files.py (migration tool)")
        print("\nUse SafeSerializer for all save/load operations:")
        print("  from RepTate.core.serialization import SafeSerializer")
        return 1

    print("All pickle removal checks pass!")
    print(f"Checked directories: {', '.join(CORE_DIRS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
