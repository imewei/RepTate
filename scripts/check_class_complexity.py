#!/usr/bin/env python3
"""Check for god classes (classes exceeding LOC threshold).

This script identifies classes that exceed complexity thresholds,
helping enforce architectural guidelines.

Usage:
    python scripts/check_class_complexity.py [--threshold LOC] [--fail-on-violation]

Exit codes:
    0: All classes within threshold
    1: God classes found (if --fail-on-violation specified)
"""
from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

# Default threshold for maximum class LOC
DEFAULT_THRESHOLD: Final[int] = 1000

# Target thresholds for known god classes (decomposition goals)
TARGET_THRESHOLDS: Final[dict[str, int]] = {
    "QApplicationWindow": 800,
    "QTheory": 600,
    "QDataSet": 500,
    "QApplicationManager": 400,
}

# Directories to check
CHECK_DIRS: Final[tuple[str, ...]] = (
    "src/RepTate/gui",
    "src/RepTate/core",
    "src/RepTate/theories",
    "src/RepTate/applications",
    "src/RepTate/tools",
)


@dataclass
class ClassInfo:
    """Information about a class."""

    name: str
    filepath: Path
    start_line: int
    end_line: int
    loc: int
    methods: int


class ClassAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze class complexity."""

    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath
        self.classes: list[ClassInfo] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Count methods
        method_count = sum(
            1 for item in node.body if isinstance(item, ast.FunctionDef)
        )

        # Calculate LOC (end_lineno - lineno gives total lines including whitespace)
        loc = (node.end_lineno or node.lineno) - node.lineno + 1

        class_info = ClassInfo(
            name=node.name,
            filepath=self.filepath,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            loc=loc,
            methods=method_count,
        )
        self.classes.append(class_info)

        # Continue visiting nested classes
        self.generic_visit(node)


def analyze_file(filepath: Path) -> list[ClassInfo]:
    """Analyze a single file for class complexity."""
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, str(filepath))
        analyzer = ClassAnalyzer(filepath)
        analyzer.visit(tree)
        return analyzer.classes
    except SyntaxError:
        return []


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Check for god classes")
    parser.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_THRESHOLD,
        help=f"Maximum LOC per class (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--fail-on-violation",
        action="store_true",
        help="Exit with code 1 if violations found",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all classes, not just violations",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    all_classes: list[ClassInfo] = []

    # Analyze all files
    for dir_pattern in CHECK_DIRS:
        dir_path = project_root / dir_pattern
        if not dir_path.exists():
            continue

        for py_file in dir_path.rglob("*.py"):
            # Skip auto-generated files
            if py_file.name.endswith("_rc.py") or py_file.name.endswith("_ui.py"):
                continue

            classes = analyze_file(py_file)
            all_classes.extend(classes)

    # Find violations
    violations: list[ClassInfo] = []
    for cls in all_classes:
        # Check against specific target threshold if exists
        threshold = TARGET_THRESHOLDS.get(cls.name, args.threshold)
        if cls.loc > threshold:
            violations.append(cls)

    # Report results
    if violations:
        print("God classes found (exceeding complexity threshold):")
        print("-" * 80)
        print(f"{'Class':<30} {'File':<40} {'LOC':>6} {'Methods':>8}")
        print("-" * 80)

        for cls in sorted(violations, key=lambda c: c.loc, reverse=True):
            rel_path = cls.filepath.relative_to(project_root)
            threshold = TARGET_THRESHOLDS.get(cls.name, args.threshold)
            print(
                f"{cls.name:<30} {str(rel_path):<40} {cls.loc:>6} {cls.methods:>8}"
            )
            print(f"  └─ Threshold: {threshold} LOC (current: {cls.loc} LOC)")

        print("-" * 80)
        print(f"Total violations: {len(violations)}")
        print("\nTarget thresholds for known god classes:")
        for name, target in TARGET_THRESHOLDS.items():
            print(f"  {name}: {target} LOC")

        if args.fail_on_violation:
            return 1

    elif args.show_all:
        print("All classes analyzed:")
        print("-" * 80)
        print(f"{'Class':<30} {'File':<40} {'LOC':>6} {'Methods':>8}")
        print("-" * 80)

        for cls in sorted(all_classes, key=lambda c: c.loc, reverse=True):
            rel_path = cls.filepath.relative_to(project_root)
            print(
                f"{cls.name:<30} {str(rel_path):<40} {cls.loc:>6} {cls.methods:>8}"
            )

        print("-" * 80)
        print(f"Total classes: {len(all_classes)}")

    else:
        print(f"All class complexity checks pass! (threshold: {args.threshold} LOC)")
        print(f"Analyzed {len(all_classes)} classes across {len(CHECK_DIRS)} directories")

    return 0


if __name__ == "__main__":
    sys.exit(main())
