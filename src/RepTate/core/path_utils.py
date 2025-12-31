"""Path validation utilities for traversal protection.

This module provides secure path handling to prevent path traversal attacks
(e.g., using '../' to escape allowed directories).

All paths are resolved to absolute paths and validated to be within the
specified base directory. Symlinks are resolved before validation.

Usage:
    from RepTate.core.path_utils import SafePath

    safe = SafePath('/data/user_files')

    # Safe usage
    path = safe.validate(Path('subdir/file.txt'))  # OK
    path = safe.join('subdir', 'file.txt')  # OK

    # Attack prevention
    safe.validate(Path('../etc/passwd'))  # Raises ValueError
    safe.join('..', '..', 'etc', 'passwd')  # Raises ValueError
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


class PathTraversalError(ValueError):
    """Raised when a path escapes the allowed base directory."""

    def __init__(self, path: Path, base_dir: Path) -> None:
        self.path = path
        self.base_dir = base_dir
        super().__init__(
            f"Path traversal detected: '{path}' escapes base directory '{base_dir}'"
        )


@dataclass
class SafePath:
    """Path operations with traversal protection.

    All path operations validate that the resulting path is within the
    configured base directory. Symlinks are resolved before validation
    to prevent symlink-based traversal attacks.

    Attributes:
        base_dir: Allowed base directory (resolved to absolute path)

    Invariants:
        - All validated paths are within base_dir
        - Symlinks are resolved before validation
        - base_dir is always an absolute path

    Examples:
        >>> import tempfile
        >>> import os
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     safe = SafePath(tmpdir)
        ...     # Valid path within base
        ...     path = safe.join('subdir', 'file.txt')
        ...     str(path).startswith(tmpdir)
        True
    """

    base_dir: Path = field(default_factory=Path.cwd)

    def __post_init__(self) -> None:
        """Resolve base_dir to absolute path."""
        # Convert to Path if string was passed
        if isinstance(self.base_dir, str):
            object.__setattr__(self, 'base_dir', Path(self.base_dir))

        # Resolve to absolute path
        resolved = self.base_dir.resolve()
        object.__setattr__(self, 'base_dir', resolved)

    def _is_within_base(self, resolved_path: Path) -> bool:
        """Check if a resolved path is within the base directory.

        Args:
            resolved_path: An already-resolved absolute path

        Returns:
            True if path is within base_dir, False otherwise
        """
        try:
            resolved_path.relative_to(self.base_dir)
            return True
        except ValueError:
            return False

    def validate(self, path: Path | str) -> Path:
        """Ensure path is within base directory.

        Args:
            path: Path to validate (can be relative or absolute)

        Returns:
            Resolved absolute path within base_dir

        Raises:
            PathTraversalError: If path escapes base_dir

        Examples:
            >>> import tempfile
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     safe = SafePath(tmpdir)
            ...     # Relative path is resolved against base
            ...     path = safe.validate('subdir/file.txt')
            ...     path.is_absolute()
            True
        """
        if isinstance(path, str):
            path = Path(path)

        # If path is relative, resolve against base_dir
        if not path.is_absolute():
            full_path = self.base_dir / path
        else:
            full_path = path

        # Resolve symlinks and normalize
        resolved = full_path.resolve()

        # Check if within base directory
        if not self._is_within_base(resolved):
            raise PathTraversalError(path, self.base_dir)

        return resolved

    def join(self, *parts: str) -> Path:
        """Safely join path components.

        Joins the given path components to the base directory and
        validates the result is within the base directory.

        Args:
            *parts: Path components to join

        Returns:
            Resolved absolute path within base_dir

        Raises:
            PathTraversalError: If resulting path escapes base_dir

        Examples:
            >>> import tempfile
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     safe = SafePath(tmpdir)
            ...     path = safe.join('a', 'b', 'c.txt')
            ...     path.name
            'c.txt'
        """
        combined = Path(*parts) if parts else Path('.')
        return self.validate(combined)

    def exists(self, path: Path | str) -> bool:
        """Check if a path exists within the base directory.

        Args:
            path: Path to check (validated before existence check)

        Returns:
            True if path exists and is within base_dir, False otherwise

        Raises:
            PathTraversalError: If path escapes base_dir
        """
        validated = self.validate(path)
        return validated.exists()

    def is_file(self, path: Path | str) -> bool:
        """Check if path is a file within the base directory.

        Args:
            path: Path to check (validated before check)

        Returns:
            True if path is a file within base_dir, False otherwise

        Raises:
            PathTraversalError: If path escapes base_dir
        """
        validated = self.validate(path)
        return validated.is_file()

    def is_dir(self, path: Path | str) -> bool:
        """Check if path is a directory within the base directory.

        Args:
            path: Path to check (validated before check)

        Returns:
            True if path is a directory within base_dir, False otherwise

        Raises:
            PathTraversalError: If path escapes base_dir
        """
        validated = self.validate(path)
        return validated.is_dir()

    def list_dir(self, path: Path | str = '.') -> list[Path]:
        """List directory contents within the base directory.

        Args:
            path: Directory path relative to base_dir (default: base_dir itself)

        Returns:
            List of paths within the directory

        Raises:
            PathTraversalError: If path escapes base_dir
            NotADirectoryError: If path is not a directory
        """
        validated = self.validate(path)
        if not validated.is_dir():
            raise NotADirectoryError(f"'{validated}' is not a directory")
        return list(validated.iterdir())
