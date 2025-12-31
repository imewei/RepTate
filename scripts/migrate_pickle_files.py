#!/usr/bin/env python3
"""Script to migrate legacy pickle files to safe JSON/NPZ format.

This script finds all pickle files in a directory (recursively) and
converts them to the new safe serialization format.

Task: T013 [US1]

Usage:
    python scripts/migrate_pickle_files.py /path/to/data
    python scripts/migrate_pickle_files.py /path/to/data --dry-run
    python scripts/migrate_pickle_files.py /path/to/data --verbose

Options:
    --dry-run   Show what would be migrated without making changes
    --verbose   Show detailed progress information
    --help      Show this help message
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Final

# Add src directory to path for standalone usage
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from RepTate.core.serialization import migrate_pickle

# Pickle file extensions to migrate
PICKLE_EXTENSIONS: Final[tuple[str, ...]] = (".pkl", ".pickle", ".p")

logger = logging.getLogger(__name__)


def find_pickle_files(directory: Path) -> list[Path]:
    """Find all pickle files in a directory recursively.

    Args:
        directory: Root directory to search

    Returns:
        List of paths to pickle files
    """
    pickle_files: list[Path] = []

    for ext in PICKLE_EXTENSIONS:
        pickle_files.extend(directory.rglob(f"*{ext}"))

    # Sort for consistent ordering
    pickle_files.sort()

    return pickle_files


def migrate_file(pickle_path: Path, dry_run: bool = False) -> bool:
    """Migrate a single pickle file.

    Args:
        pickle_path: Path to pickle file
        dry_run: If True, don't actually migrate

    Returns:
        True if migration succeeded (or would succeed), False otherwise
    """
    if dry_run:
        logger.info("Would migrate: %s", pickle_path)
        return True

    try:
        new_path = migrate_pickle(pickle_path)
        logger.info("Migrated: %s -> %s", pickle_path, new_path)
        return True
    except Exception as e:
        logger.error("Failed to migrate %s: %s", pickle_path, e)
        return False


def main() -> int:
    """Main entry point for the migration script.

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    parser = argparse.ArgumentParser(
        description="Migrate legacy pickle files to safe JSON/NPZ format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Migrate all pickle files in data directory
    python scripts/migrate_pickle_files.py ./data

    # Preview what would be migrated
    python scripts/migrate_pickle_files.py ./data --dry-run

    # Migrate with verbose output
    python scripts/migrate_pickle_files.py ./data --verbose
        """,
    )

    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing pickle files to migrate",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without making changes",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed progress information",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
    )

    # Validate directory
    if not args.directory.exists():
        logger.error("Directory does not exist: %s", args.directory)
        return 1

    if not args.directory.is_dir():
        logger.error("Not a directory: %s", args.directory)
        return 1

    # Find pickle files
    pickle_files = find_pickle_files(args.directory)

    if not pickle_files:
        logger.info("No pickle files found in %s", args.directory)
        return 0

    logger.info("Found %d pickle file(s) to migrate", len(pickle_files))

    if args.dry_run:
        logger.info("DRY RUN - no changes will be made")

    # Migrate files
    success_count = 0
    error_count = 0

    for pickle_path in pickle_files:
        if migrate_file(pickle_path, dry_run=args.dry_run):
            success_count += 1
        else:
            error_count += 1

    # Summary
    logger.info("")
    logger.info("Migration complete:")
    logger.info("  Succeeded: %d", success_count)
    logger.info("  Failed: %d", error_count)

    if args.dry_run:
        logger.info("(DRY RUN - no actual changes were made)")

    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
