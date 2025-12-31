"""Native library loader with path validation.

This module provides secure loading of native libraries (shared objects,
DLLs) with validation to prevent loading from untrusted locations.

Security measures:
- Path traversal prevention
- Validation that file exists and has expected format
- Optional signature/checksum verification (for future extension)

Usage:
    from RepTate.core.native_loader import NativeLoader, LibraryLoadError

    loader = NativeLoader(allowed_dirs=['/app/libs', '/usr/lib'])

    try:
        lib = loader.load('libcompute.so')
        result = lib.compute_function(data)
    except LibraryLoadError as e:
        print(f"Failed to load library: {e}")
"""

from __future__ import annotations

import ctypes
import platform
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final


# Platform-specific library extensions
PLATFORM_EXTENSIONS: Final[dict[str, tuple[str, ...]]] = {
    'Linux': ('.so', '.so.*'),
    'Darwin': ('.dylib', '.so'),
    'Windows': ('.dll', '.pyd'),
}

# ELF magic number for Linux shared objects
ELF_MAGIC: Final[bytes] = b'\x7fELF'

# Mach-O magic numbers for macOS
MACHO_MAGIC_32: Final[bytes] = b'\xfe\xed\xfa\xce'
MACHO_MAGIC_64: Final[bytes] = b'\xfe\xed\xfa\xcf'
MACHO_MAGIC_32_REV: Final[bytes] = b'\xce\xfa\xed\xfe'
MACHO_MAGIC_64_REV: Final[bytes] = b'\xcf\xfa\xed\xfe'
MACHO_FAT_MAGIC: Final[bytes] = b'\xca\xfe\xba\xbe'
MACHO_FAT_MAGIC_64: Final[bytes] = b'\xca\xfe\xba\xbf'

# PE magic for Windows DLLs
PE_MAGIC: Final[bytes] = b'MZ'


class LibraryLoadError(Exception):
    """Raised when a native library cannot be loaded."""
    pass


class LibraryNotFoundError(LibraryLoadError):
    """Raised when a library file does not exist."""
    pass


class InvalidLibraryError(LibraryLoadError):
    """Raised when a file is not a valid native library."""
    pass


class UntrustedPathError(LibraryLoadError):
    """Raised when a library path is outside allowed directories."""
    pass


def _get_platform() -> str:
    """Get the current platform name."""
    return platform.system()


def _get_expected_extensions() -> tuple[str, ...]:
    """Get the expected library extensions for the current platform."""
    return PLATFORM_EXTENSIONS.get(_get_platform(), ('.so',))


def _validate_library_format(path: Path) -> None:
    """Validate that a file has the expected native library format.

    Args:
        path: Path to the library file

    Raises:
        InvalidLibraryError: If file is not a valid native library
    """
    try:
        with open(path, 'rb') as f:
            header = f.read(64)  # Read enough for all format checks
    except OSError as e:
        raise InvalidLibraryError(f"Cannot read library file: {e}")

    if len(header) < 4:
        raise InvalidLibraryError(f"File too small to be a native library: {path}")

    current_platform = _get_platform()

    if current_platform == 'Linux':
        if not header.startswith(ELF_MAGIC):
            raise InvalidLibraryError(
                f"Not a valid ELF file (expected Linux shared object): {path}"
            )

    elif current_platform == 'Darwin':
        magic = header[:4]
        valid_macho = (
            MACHO_MAGIC_32, MACHO_MAGIC_64,
            MACHO_MAGIC_32_REV, MACHO_MAGIC_64_REV,
            MACHO_FAT_MAGIC, MACHO_FAT_MAGIC_64
        )
        if magic not in valid_macho:
            raise InvalidLibraryError(
                f"Not a valid Mach-O file (expected macOS library): {path}"
            )

    elif current_platform == 'Windows':
        if not header.startswith(PE_MAGIC):
            raise InvalidLibraryError(
                f"Not a valid PE file (expected Windows DLL): {path}"
            )

        # Additional PE validation: check for PE signature
        if len(header) >= 64:
            # PE header offset is at bytes 60-63 (little-endian)
            pe_offset = struct.unpack_from('<I', header, 60)[0]
            if pe_offset > 0 and pe_offset + 4 <= len(header):
                pe_sig = header[pe_offset:pe_offset + 4]
                if pe_sig != b'PE\x00\x00':
                    raise InvalidLibraryError(
                        f"Invalid PE signature in file: {path}"
                    )


@dataclass
class NativeLoader:
    """Secure loader for native libraries.

    Validates that libraries are loaded from allowed directories and
    have the expected file format for the current platform.

    Attributes:
        allowed_dirs: List of directories from which libraries can be loaded.
                     If empty, only absolute paths within allowed_dirs work.
        validate_format: Whether to validate the library file format.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     loader = NativeLoader(allowed_dirs=[tmpdir])
        ...     # Would load from tmpdir if library existed
    """

    allowed_dirs: list[Path] = field(default_factory=list)
    validate_format: bool = True

    def __post_init__(self) -> None:
        """Resolve allowed directories to absolute paths."""
        self.allowed_dirs = [
            Path(d).resolve() for d in self.allowed_dirs
        ]

    def _is_allowed_path(self, path: Path) -> bool:
        """Check if a path is within an allowed directory.

        Args:
            path: Resolved absolute path to check

        Returns:
            True if path is within an allowed directory
        """
        for allowed_dir in self.allowed_dirs:
            try:
                path.relative_to(allowed_dir)
                return True
            except ValueError:
                continue
        return False

    def _find_library(self, name: str) -> Path:
        """Find a library by name in allowed directories.

        Args:
            name: Library name (with or without extension)

        Returns:
            Resolved absolute path to the library

        Raises:
            LibraryNotFoundError: If library cannot be found
        """
        # If name is an absolute path, validate it directly
        name_path = Path(name)
        if name_path.is_absolute():
            resolved = name_path.resolve()
            if not resolved.exists():
                raise LibraryNotFoundError(f"Library not found: {name}")
            if not self._is_allowed_path(resolved):
                raise UntrustedPathError(
                    f"Library path '{resolved}' is outside allowed directories"
                )
            return resolved

        # Search in allowed directories
        extensions = _get_expected_extensions()

        for allowed_dir in self.allowed_dirs:
            # Try exact name first
            candidate = allowed_dir / name
            if candidate.exists():
                return candidate.resolve()

            # Try with platform extensions
            for ext in extensions:
                if ext.endswith('*'):
                    # Handle versioned extensions like .so.*
                    base_ext = ext[:-1]  # Remove *
                    for file in allowed_dir.glob(f"{name}{base_ext}*"):
                        if file.is_file():
                            return file.resolve()
                else:
                    candidate = allowed_dir / f"{name}{ext}"
                    if candidate.exists():
                        return candidate.resolve()

        raise LibraryNotFoundError(
            f"Library '{name}' not found in allowed directories: "
            f"{[str(d) for d in self.allowed_dirs]}"
        )

    def validate(self, path: Path | str) -> Path:
        """Validate a library path without loading it.

        Args:
            path: Path to the library file

        Returns:
            Validated absolute path

        Raises:
            LibraryNotFoundError: If file does not exist
            UntrustedPathError: If path is outside allowed directories
            InvalidLibraryError: If file is not a valid library format
        """
        path = Path(path)
        resolved = path.resolve()

        if not resolved.exists():
            raise LibraryNotFoundError(f"Library not found: {path}")

        if not resolved.is_file():
            raise InvalidLibraryError(f"Not a file: {path}")

        if not self._is_allowed_path(resolved):
            raise UntrustedPathError(
                f"Library path '{resolved}' is outside allowed directories"
            )

        if self.validate_format:
            _validate_library_format(resolved)

        return resolved

    def load(self, name: str) -> ctypes.CDLL:
        """Load a native library securely.

        Searches for the library in allowed directories, validates the
        file format, and loads it using ctypes.

        Args:
            name: Library name or path

        Returns:
            Loaded ctypes.CDLL object

        Raises:
            LibraryNotFoundError: If library cannot be found
            UntrustedPathError: If library is outside allowed directories
            InvalidLibraryError: If file is not a valid library format
            LibraryLoadError: If library cannot be loaded
        """
        library_path = self._find_library(name)

        if self.validate_format:
            _validate_library_format(library_path)

        try:
            return ctypes.CDLL(str(library_path))
        except OSError as e:
            raise LibraryLoadError(f"Failed to load library '{library_path}': {e}")

    def add_allowed_dir(self, directory: Path | str) -> None:
        """Add a directory to the allowed list.

        Args:
            directory: Directory path to add
        """
        resolved = Path(directory).resolve()
        if resolved not in self.allowed_dirs:
            self.allowed_dirs.append(resolved)

    def is_valid_library(self, path: Path | str) -> bool:
        """Check if a path points to a valid, loadable library.

        Args:
            path: Path to check

        Returns:
            True if the library is valid and can be loaded
        """
        try:
            self.validate(path)
            return True
        except LibraryLoadError:
            return False
