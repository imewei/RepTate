"""Safe JSON/NPZ serialization replacing pickle.

This module provides safe serialization using JSON for metadata and NPZ for
numpy arrays, eliminating arbitrary code execution vulnerabilities associated
with pickle.

Security Guarantees:
- NPZ loaded with allow_pickle=False
- JSON cannot contain executable code
- Type safety: Only whitelisted types serializable
- Path safety: No path traversal in references

Task: T011 [P] [US1], T012 [P] [US1]

Example:
    >>> from pathlib import Path
    >>> from RepTate.core.serialization import SafeSerializer
    >>> import numpy as np
    >>>
    >>> data = {
    ...     "name": "experiment_001",
    ...     "frequency": np.array([0.1, 1.0, 10.0]),
    ... }
    >>> result = SafeSerializer.save(Path("output/data"), data)
    >>> loaded = SafeSerializer.load(Path("output/data"))
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from types import FunctionType, GeneratorType, LambdaType, MethodType
from typing import Any, Callable, Final

import numpy as np

# Import JAX Array type for detection
try:
    from jax import Array as JaxArray
except ImportError:
    JaxArray = None  # JAX not installed

logger = logging.getLogger(__name__)


# Marker used to indicate array references in JSON
_ARRAY_REF_KEY: Final[str] = "__array_ref__"
# Version key in JSON metadata
_VERSION_KEY: Final[str] = "__version__"

# Types that are explicitly not serializable for security
_UNSUPPORTED_TYPES: tuple[type, ...] = (
    FunctionType,
    LambdaType,
    MethodType,
    GeneratorType,
    type,  # classes themselves
)


@dataclass
class SerializationResult:
    """Result of a serialization operation.

    Attributes:
        json_path: Path to JSON metadata file
        npz_path: Path to NPZ array file (None if no arrays)
        version: Serialization format version
    """

    json_path: Path
    npz_path: Path | None
    version: int


class SafeSerializer:
    """Safe serialization service replacing pickle.

    Provides JSON/NPZ-based serialization that eliminates arbitrary code
    execution vulnerabilities. Numpy arrays are stored in NPZ format and
    referenced from JSON metadata.

    Class Attributes:
        VERSION: Current format version for forward compatibility

    Methods:
        save: Serialize data to JSON/NPZ format
        load: Deserialize data from JSON/NPZ format
        can_load: Check if file is in supported format
    """

    VERSION: Final[int] = 1

    @classmethod
    def save(cls, filepath: Path, data: dict[str, Any]) -> SerializationResult:
        """Save data to safe JSON/NPZ format.

        Separates data into JSON-serializable values and numpy arrays.
        Arrays are stored in NPZ format with references in JSON.

        Args:
            filepath: Base path for output files (without extension)
            data: Dictionary containing data to serialize

        Returns:
            SerializationResult with paths to created files

        Raises:
            IOError: If files cannot be written
            TypeError: If data contains non-serializable types
        """
        filepath = Path(filepath)

        # Ensure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Collect arrays and prepare JSON-safe data
        arrays: dict[str, np.ndarray] = {}
        json_data = cls._prepare_for_json(data, arrays, prefix="")

        # Add version marker
        json_data[_VERSION_KEY] = cls.VERSION

        # Write JSON file
        json_path = Path(str(filepath) + ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        # Write NPZ file if there are arrays
        npz_path: Path | None = None
        if arrays:
            npz_path = Path(str(filepath) + ".npz")
            np.savez_compressed(npz_path, **arrays)

        logger.debug("Saved data to %s (arrays: %d)", json_path, len(arrays))

        return SerializationResult(
            json_path=json_path,
            npz_path=npz_path,
            version=cls.VERSION,
        )

    @classmethod
    def load(cls, filepath: Path) -> dict[str, Any]:
        """Load data from safe JSON/NPZ format.

        Loads JSON metadata and restores numpy arrays from NPZ file.

        Args:
            filepath: Base path for input files (without extension)

        Returns:
            Dictionary with deserialized data and restored numpy arrays

        Raises:
            FileNotFoundError: If JSON file doesn't exist
            ValueError: If file version is unsupported
            ValueError: If NPZ referenced but missing
        """
        filepath = Path(filepath)
        json_path = Path(str(filepath) + ".json")

        # Load JSON metadata
        if not json_path.exists():
            raise FileNotFoundError(f"No such file: '{json_path}'")

        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        # Check version
        version = json_data.pop(_VERSION_KEY, None)
        if version is None:
            raise ValueError(f"Missing version in file: {json_path}")
        if version > cls.VERSION:
            raise ValueError(f"Unsupported file version: {version}")

        # Check if arrays are referenced
        has_array_refs = cls._has_array_refs(json_data)

        # Load NPZ if needed
        arrays: dict[str, np.ndarray] = {}
        if has_array_refs:
            npz_path = Path(str(filepath) + ".npz")
            if not npz_path.exists():
                raise ValueError(f"Referenced NPZ file not found: {npz_path}")

            try:
                with np.load(npz_path, allow_pickle=False) as npz_file:
                    for key in npz_file.files:
                        arrays[key] = npz_file[key]
            except ValueError as e:
                # NPZ file requires pickle, which is not allowed
                raise ValueError(
                    f"NPZ file contains pickled data, which is not safe: {npz_path}"
                ) from e

        # Restore arrays in data structure
        data = cls._restore_arrays(json_data, arrays)

        logger.debug("Loaded data from %s (arrays: %d)", json_path, len(arrays))

        return data

    @classmethod
    def can_load(cls, filepath: Path) -> bool:
        """Check if a file is in supported format.

        Verifies that the JSON file exists and has a valid format header.

        Args:
            filepath: Base path to check

        Returns:
            True if file exists and has valid format header
        """
        filepath = Path(filepath)
        json_path = Path(str(filepath) + ".json")

        if not json_path.exists():
            return False

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)

            # Must be a dict with version
            if not isinstance(json_data, dict):
                return False
            if _VERSION_KEY not in json_data:
                return False

            # Version must be supported
            version = json_data[_VERSION_KEY]
            if not isinstance(version, int) or version > cls.VERSION:
                return False

            return True

        except (json.JSONDecodeError, OSError):
            return False

    @classmethod
    def _prepare_for_json(
        cls,
        obj: Any,
        arrays: dict[str, np.ndarray],
        prefix: str,
    ) -> Any:
        """Recursively prepare data for JSON serialization.

        Extracts numpy arrays into the arrays dict and replaces them
        with reference markers.

        Args:
            obj: Object to prepare
            arrays: Dict to store extracted arrays
            prefix: Key prefix for array naming

        Returns:
            JSON-serializable representation

        Raises:
            TypeError: If object cannot be serialized
        """
        if obj is None:
            return None

        # Explicitly check for unsupported types first (security)
        if isinstance(obj, _UNSUPPORTED_TYPES):
            raise TypeError(f"Cannot serialize type: {type(obj).__name__}")

        # Check for callable objects (functions, lambdas, etc.)
        if callable(obj) and not isinstance(obj, type):
            raise TypeError(f"Cannot serialize callable: {type(obj).__name__}")

        if isinstance(obj, bool):
            return obj

        if isinstance(obj, (int, float, str)):
            return obj

        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            return float(obj)

        if isinstance(obj, np.ndarray):
            # Generate unique array key (safe, no path separators)
            array_key = f"_{prefix.replace('.', '_')}_array_{len(arrays)}"
            array_key = array_key.lstrip("_").replace("__", "_")
            if not array_key.startswith("_"):
                array_key = "_" + array_key
            arrays[array_key] = obj
            return {_ARRAY_REF_KEY: array_key}

        # Handle JAX arrays by converting to numpy
        if JaxArray is not None and isinstance(obj, JaxArray):
            # Convert JAX array to numpy for storage
            array_key = f"_{prefix.replace('.', '_')}_array_{len(arrays)}"
            array_key = array_key.lstrip("_").replace("__", "_")
            if not array_key.startswith("_"):
                array_key = "_" + array_key
            arrays[array_key] = np.array(obj)
            return {_ARRAY_REF_KEY: array_key}

        if isinstance(obj, list):
            return [
                cls._prepare_for_json(item, arrays, f"{prefix}_{i}")
                for i, item in enumerate(obj)
            ]

        if isinstance(obj, dict):
            return {
                key: cls._prepare_for_json(value, arrays, f"{prefix}_{key}")
                for key, value in obj.items()
            }

        # Check for classes with __dict__ but exclude problematic types
        if hasattr(obj, "__dict__"):
            if hasattr(obj, "__slots__"):
                raise TypeError(
                    f"Cannot serialize object with __slots__: {type(obj).__name__}"
                )
            # Don't serialize objects with callable methods that might be confused
            return cls._prepare_for_json(obj.__dict__, arrays, prefix)

        # Unsupported type
        raise TypeError(f"Cannot serialize type: {type(obj).__name__}")

    @classmethod
    def _has_array_refs(cls, obj: Any) -> bool:
        """Check if object contains array references."""
        if isinstance(obj, dict):
            if _ARRAY_REF_KEY in obj:
                return True
            return any(cls._has_array_refs(v) for v in obj.values())
        if isinstance(obj, list):
            return any(cls._has_array_refs(item) for item in obj)
        return False

    @classmethod
    def _restore_arrays(
        cls,
        obj: Any,
        arrays: dict[str, np.ndarray],
    ) -> Any:
        """Recursively restore numpy arrays from references."""
        if isinstance(obj, dict):
            if _ARRAY_REF_KEY in obj and len(obj) == 1:
                array_key = obj[_ARRAY_REF_KEY]
                if array_key not in arrays:
                    raise ValueError(f"Array reference not found: {array_key}")
                return arrays[array_key]
            return {
                key: cls._restore_arrays(value, arrays) for key, value in obj.items()
            }
        if isinstance(obj, list):
            return [cls._restore_arrays(item, arrays) for item in obj]
        return obj


def migrate_pickle(pickle_path: Path) -> Path:
    """Convert legacy pickle file to safe format.

    Loads a pickle file and saves it in the new JSON/NPZ format.
    The original file is renamed to .pkl.bak for safety.

    Args:
        pickle_path: Path to .pkl or .pickle file

    Returns:
        Path to new JSON base file (without extension)

    Side Effects:
        - Creates .json and .npz files
        - Renames original to .pkl.bak

    Example:
        >>> new_path = migrate_pickle(Path("legacy_data.pkl"))
        >>> # Creates: legacy_data.json, legacy_data.npz
        >>> # Renames: legacy_data.pkl -> legacy_data.pkl.bak
    """
    import pickle as _pickle  # Import only when needed for migration

    pickle_path = Path(pickle_path)

    if not pickle_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {pickle_path}")

    # Load legacy data
    logger.info("Migrating pickle file: %s", pickle_path)
    with open(pickle_path, "rb") as f:
        data = _pickle.load(f)

    # Determine base path (remove .pkl or .pickle extension)
    base_name = pickle_path.stem
    if pickle_path.suffix.lower() == ".pickle":
        base_path = pickle_path.parent / base_name
    else:
        base_path = pickle_path.parent / base_name

    # Convert to dict if not already
    if not isinstance(data, dict):
        if hasattr(data, "__dict__"):
            data = data.__dict__
        else:
            data = {"data": data}

    # Save in new format
    SafeSerializer.save(base_path, data)

    # Backup original
    backup_path = Path(str(pickle_path) + ".bak")
    pickle_path.rename(backup_path)

    logger.info("Migration complete: %s -> %s", pickle_path, base_path)

    return base_path
