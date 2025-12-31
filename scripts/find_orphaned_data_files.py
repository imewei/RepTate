#!/usr/bin/env python3
"""Find orphaned data files that may be legacy artifacts.

This script identifies data files that are no longer referenced in code,
helping clean up legacy formats and test artifacts.

Usage:
    python scripts/find_orphaned_data_files.py [--delete] [--dry-run]

Exit codes:
    0: Success (orphaned files found or not)
    1: Error during analysis
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

# File patterns to check for orphaned files
DATA_PATTERNS: Final[tuple[str, ...]] = (
    "*.pkl",
    "*.pickle",
    "*.dat.old",
    "*.dat.backup",
    "*.bak",
    "*.tmp",
)

# Directories to scan
SCAN_DIRS: Final[tuple[str, ...]] = (
    "data",
    "tests/regression/golden",
    "tests/integration/fixtures",
    "examples",
)

# Minimum age (days) before considering file orphaned
MIN_AGE_DAYS: Final[int] = 180  # 6 months


def is_referenced_in_code(filepath: Path, project_root: Path) -> bool:
    """Check if file is referenced in any Python code."""
    filename = filepath.name
    stem = filepath.stem

    # Search for references in Python files
    for py_file in project_root.rglob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8")
            # Check for filename or stem references
            if filename in content or stem in content:
                return True
        except (UnicodeDecodeError, PermissionError):
            continue

    return False


def get_file_age_days(filepath: Path) -> int:
    """Get age of file in days since last modification."""
    mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
    age = datetime.now() - mtime
    return age.days


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Find orphaned data files"
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete orphaned files (use with caution!)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    parser.add_argument(
        "--min-age",
        type=int,
        default=MIN_AGE_DAYS,
        help=f"Minimum age in days to consider orphaned (default: {MIN_AGE_DAYS})",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    orphaned_files: list[tuple[Path, int, int]] = []  # (path, age_days, size_bytes)

    print("Scanning for orphaned data files...")
    print(f"Minimum age: {args.min_age} days")
    print("-" * 80)

    for dir_pattern in SCAN_DIRS:
        dir_path = project_root / dir_pattern
        if not dir_path.exists():
            continue

        for pattern in DATA_PATTERNS:
            for data_file in dir_path.rglob(pattern):
                age_days = get_file_age_days(data_file)

                # Skip recent files
                if age_days < args.min_age:
                    continue

                # Check if referenced in code
                if not is_referenced_in_code(data_file, project_root):
                    size_bytes = data_file.stat().st_size
                    orphaned_files.append((data_file, age_days, size_bytes))

    if not orphaned_files:
        print("No orphaned files found ✓")
        return 0

    # Report findings
    total_size = sum(size for _, _, size in orphaned_files)
    print(f"Found {len(orphaned_files)} orphaned files:")
    print("-" * 80)
    print(f"{'File':<60} {'Age (days)':>12} {'Size':>10}")
    print("-" * 80)

    for filepath, age_days, size_bytes in sorted(
        orphaned_files, key=lambda x: x[1], reverse=True
    ):
        rel_path = filepath.relative_to(project_root)
        size_str = f"{size_bytes:,} bytes" if size_bytes < 1024 else f"{size_bytes / 1024:.1f} KB"
        print(f"{str(rel_path):<60} {age_days:>12} {size_str:>10}")

    print("-" * 80)
    print(f"Total: {len(orphaned_files)} files, {total_size / 1024:.1f} KB")

    # Handle deletion
    if args.delete or args.dry_run:
        print("\n" + "=" * 80)
        if args.dry_run:
            print("DRY RUN: Would delete the following files:")
        else:
            print("WARNING: Deleting orphaned files...")

        for filepath, _, _ in orphaned_files:
            if args.dry_run:
                print(f"  Would delete: {filepath.relative_to(project_root)}")
            else:
                try:
                    filepath.unlink()
                    print(f"  Deleted: {filepath.relative_to(project_root)}")
                except OSError as e:
                    print(f"  ERROR deleting {filepath}: {e}")

        if args.dry_run:
            print("\nNo files were actually deleted (dry run mode)")
        else:
            print(f"\nDeleted {len(orphaned_files)} orphaned files")

    else:
        print("\nTo delete these files, run with --delete flag")
        print("(Recommend using --dry-run first to preview)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
