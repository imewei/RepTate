"""Secure temporary file utilities.

This module provides secure temporary file and directory creation with:
- Restricted permissions (0o600 for files, 0o700 for directories)
- Automatic cleanup on exit (via context managers)
- Platform-appropriate defaults

Usage:
    from RepTate.core.temp_utils import SecureTempFile, SecureTempDir

    # Temporary file with automatic cleanup
    with SecureTempFile(suffix='.json') as tmp_path:
        tmp_path.write_text('{"data": 123}')
        # File is readable only by owner

    # File is deleted when context exits

    # Temporary directory with automatic cleanup
    with SecureTempDir(prefix='reptate_') as tmp_dir:
        (tmp_dir / 'file.txt').write_text('data')
        # Directory is accessible only by owner

    # Directory and contents deleted when context exits
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


# Secure permission modes
FILE_MODE: int = 0o600  # Owner read/write only
DIR_MODE: int = 0o700   # Owner read/write/execute only


@dataclass
class SecureTempFile:
    """Context manager for secure temporary files.

    Creates a temporary file with restricted permissions (0o600) that is
    automatically deleted when the context exits.

    Attributes:
        suffix: File suffix (e.g., '.json', '.npz')
        prefix: File prefix (e.g., 'reptate_')
        dir: Directory to create temp file in (default: system temp)
        delete: Whether to delete file on exit (default: True)

    Examples:
        >>> with SecureTempFile(suffix='.txt') as path:
        ...     path.write_text('test')
        ...     path.exists()
        True
        >>> # File is deleted after context exit
    """

    suffix: str = ''
    prefix: str = 'reptate_'
    dir: Path | str | None = None
    delete: bool = True

    _path: Path | None = field(default=None, init=False, repr=False)

    def __enter__(self) -> Path:
        """Create the temporary file and return its path."""
        # Create temp file with restricted permissions
        # os.open with O_CREAT | O_EXCL ensures atomic creation
        dir_path = str(self.dir) if self.dir else None

        fd, path_str = tempfile.mkstemp(
            suffix=self.suffix,
            prefix=self.prefix,
            dir=dir_path
        )

        try:
            # Set restricted permissions
            os.fchmod(fd, FILE_MODE)
        finally:
            os.close(fd)

        self._path = Path(path_str)
        return self._path

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None
    ) -> None:
        """Delete the temporary file if delete=True."""
        if self.delete and self._path is not None and self._path.exists():
            try:
                self._path.unlink()
            except OSError:
                # Best effort cleanup
                pass


@dataclass
class SecureTempDir:
    """Context manager for secure temporary directories.

    Creates a temporary directory with restricted permissions (0o700) that is
    automatically deleted (including contents) when the context exits.

    Attributes:
        suffix: Directory suffix
        prefix: Directory prefix (e.g., 'reptate_')
        dir: Parent directory (default: system temp)
        delete: Whether to delete directory on exit (default: True)

    Examples:
        >>> with SecureTempDir(prefix='test_') as path:
        ...     (path / 'file.txt').write_text('data')
        ...     path.is_dir()
        True
        >>> # Directory and contents deleted after context exit
    """

    suffix: str | None = None
    prefix: str | None = 'reptate_'
    dir: Path | str | None = None
    delete: bool = True

    _path: Path | None = field(default=None, init=False, repr=False)

    def __enter__(self) -> Path:
        """Create the temporary directory and return its path."""
        dir_path = str(self.dir) if self.dir else None

        path_str = tempfile.mkdtemp(
            suffix=self.suffix,
            prefix=self.prefix,
            dir=dir_path
        )

        self._path = Path(path_str)

        # Set restricted permissions
        os.chmod(self._path, DIR_MODE)

        return self._path

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None
    ) -> None:
        """Delete the temporary directory and contents if delete=True."""
        if self.delete and self._path is not None and self._path.exists():
            try:
                shutil.rmtree(self._path)
            except OSError:
                # Best effort cleanup
                pass


@contextmanager
def secure_temp_file(
    suffix: str = '',
    prefix: str = 'reptate_',
    dir: Path | str | None = None,
    delete: bool = True
) -> Iterator[Path]:
    """Context manager function for secure temporary files.

    Functional alternative to SecureTempFile class.

    Args:
        suffix: File suffix (e.g., '.json')
        prefix: File prefix (default: 'reptate_')
        dir: Directory to create temp file in
        delete: Whether to delete file on exit

    Yields:
        Path to the temporary file

    Examples:
        >>> with secure_temp_file(suffix='.json') as path:
        ...     path.write_text('{}')
        ...     path.exists()
        True
    """
    with SecureTempFile(suffix=suffix, prefix=prefix, dir=dir, delete=delete) as path:
        yield path


@contextmanager
def secure_temp_dir(
    suffix: str | None = None,
    prefix: str | None = 'reptate_',
    dir: Path | str | None = None,
    delete: bool = True
) -> Iterator[Path]:
    """Context manager function for secure temporary directories.

    Functional alternative to SecureTempDir class.

    Args:
        suffix: Directory suffix
        prefix: Directory prefix (default: 'reptate_')
        dir: Parent directory
        delete: Whether to delete directory on exit

    Yields:
        Path to the temporary directory

    Examples:
        >>> with secure_temp_dir(prefix='test_') as path:
        ...     (path / 'data.txt').touch()
        ...     path.is_dir()
        True
    """
    with SecureTempDir(suffix=suffix, prefix=prefix, dir=dir, delete=delete) as path:
        yield path


class TempFileManager:
    """Manages multiple temporary files with cleanup on program exit.

    Provides a registry for temporary files that need to be cleaned up
    when the program exits, even if not explicitly deleted.

    Usage:
        manager = TempFileManager()
        path = manager.create_file(suffix='.json')
        # Use path...
        # Cleaned up at program exit or when cleanup() is called
    """

    def __init__(self) -> None:
        self._files: list[Path] = []
        self._dirs: list[Path] = []
        atexit.register(self.cleanup)

    def create_file(
        self,
        suffix: str = '',
        prefix: str = 'reptate_',
        dir: Path | str | None = None
    ) -> Path:
        """Create a tracked temporary file.

        Args:
            suffix: File suffix
            prefix: File prefix
            dir: Directory for temp file

        Returns:
            Path to the created temporary file
        """
        dir_path = str(dir) if dir else None

        fd, path_str = tempfile.mkstemp(
            suffix=suffix,
            prefix=prefix,
            dir=dir_path
        )

        try:
            os.fchmod(fd, FILE_MODE)
        finally:
            os.close(fd)

        path = Path(path_str)
        self._files.append(path)
        return path

    def create_dir(
        self,
        suffix: str | None = None,
        prefix: str | None = 'reptate_',
        dir: Path | str | None = None
    ) -> Path:
        """Create a tracked temporary directory.

        Args:
            suffix: Directory suffix
            prefix: Directory prefix
            dir: Parent directory

        Returns:
            Path to the created temporary directory
        """
        dir_path = str(dir) if dir else None

        path_str = tempfile.mkdtemp(
            suffix=suffix,
            prefix=prefix,
            dir=dir_path
        )

        path = Path(path_str)
        os.chmod(path, DIR_MODE)
        self._dirs.append(path)
        return path

    def cleanup(self) -> None:
        """Remove all tracked temporary files and directories."""
        # Clean up files first
        for path in self._files:
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass
        self._files.clear()

        # Then clean up directories
        for path in self._dirs:
            if path.exists():
                try:
                    shutil.rmtree(path)
                except OSError:
                    pass
        self._dirs.clear()
