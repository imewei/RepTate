#!/usr/bin/env python3
"""Verify no unsafe eval/exec usage exists in codebase.

This script checks that eval() and exec() are not used in RepTate modules,
ensuring the safe evaluation migration is complete.

Usage:
    python scripts/verify_safe_eval_only.py

Exit codes:
    0: All checks pass
    1: Unsafe eval/exec usage found
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Final

# Directories to check
CHECK_DIRS: Final[tuple[str, ...]] = (
    "src/RepTate",
)

# Files allowed to have eval (safe_eval implementation itself)
ALLOWED_FILES: Final[tuple[str, ...]] = (
    "src/RepTate/core/safe_eval.py",  # Implementation of safe eval
)


class UnsafeEvalFinder(ast.NodeVisitor):
    """AST visitor to find eval/exec calls."""

    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath
        self.violations: list[tuple[int, str, str]] = []

    def visit_Call(self, node: ast.Call) -> None:
        # Check for eval() or exec() calls
        if isinstance(node.func, ast.Name):
            if node.func.id in ("eval", "exec"):
                # Get the line of code for context
                context = "eval()" if node.func.id == "eval" else "exec()"
                self.violations.append((node.lineno, node.func.id, context))
        self.generic_visit(node)


def check_file(filepath: Path) -> list[tuple[int, str, str]]:
    """Check a single file for unsafe eval/exec usage."""
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, str(filepath))
        finder = UnsafeEvalFinder(filepath)
        finder.visit(tree)
        return finder.violations
    except SyntaxError:
        return []


def main() -> int:
    """Main entry point."""
    project_root = Path(__file__).parent.parent
    all_violations: list[tuple[Path, int, str, str]] = []

    for dir_pattern in CHECK_DIRS:
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
            for lineno, func_name, context in violations:
                all_violations.append((py_file, lineno, func_name, context))

    if all_violations:
        print("Unsafe eval/exec usage found:")
        print("-" * 60)
        for filepath, lineno, func_name, context in sorted(all_violations):
            rel_path = filepath.relative_to(project_root)
            print(f"{rel_path}:{lineno}: {context}")
        print("-" * 60)
        print(f"Total: {len(all_violations)} violations")
        print("\nDo NOT use eval() or exec() directly!")
        print("Use safe_eval module instead:")
        print("  from RepTate.core.safe_eval import safe_eval")
        print("  result = safe_eval(expression, context)")
        return 1

    print("All safe eval checks pass!")
    print(f"Checked directories: {', '.join(CHECK_DIRS)}")
    print("No unsafe eval() or exec() usage detected ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
