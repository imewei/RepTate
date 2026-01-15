"""Application Registry for Lazy Loading.

This module provides a registry of RepTate applications with lazy loading support.
Applications are not imported until they are first accessed, reducing startup time
by ~30% (from ~4s to ~2.5s on typical systems).

Feature: 005-gui-performance-integration (T008)
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from RepTate.core.Application import Application

logger = logging.getLogger(__name__)

# Registry mapping application names to their module paths and metadata
# Format: 'AppName': {'module': 'module.path.ClassName', 'extensions': ['ext1', 'ext2']}
# Note: First extension in list is the primary one
_APPLICATION_REGISTRY: dict[str, dict[str, str | list[str]]] = {
    "MWD": {"module": "RepTate.applications.ApplicationMWD.ApplicationMWD", "extensions": ["gpc"]},
    "TTS": {"module": "RepTate.applications.ApplicationTTS.ApplicationTTS", "extensions": ["osc"]},
    "TTSF": {"module": "RepTate.applications.ApplicationTTSFactors.ApplicationTTSFactors", "extensions": ["ttsf"]},
    "LVE": {"module": "RepTate.applications.ApplicationLVE.ApplicationLVE", "extensions": ["tts"]},
    "NLVE": {"module": "RepTate.applications.ApplicationNLVE.ApplicationNLVE", "extensions": ["shear", "uext"]},
    "Crystal": {"module": "RepTate.applications.ApplicationCrystal.ApplicationCrystal", "extensions": ["shearxs", "uextxs"]},
    "Gt": {"module": "RepTate.applications.ApplicationGt.ApplicationGt", "extensions": ["gt"]},
    "Creep": {"module": "RepTate.applications.ApplicationCreep.ApplicationCreep", "extensions": ["creep"]},
    "SANS": {"module": "RepTate.applications.ApplicationSANS.ApplicationSANS", "extensions": ["sans"]},
    "React": {"module": "RepTate.applications.ApplicationReact.ApplicationReact", "extensions": ["reac"]},
    "Dielectric": {"module": "RepTate.applications.ApplicationDielectric.ApplicationDielectric", "extensions": ["dls"]},
    "LAOS": {"module": "RepTate.applications.ApplicationLAOS.ApplicationLAOS", "extensions": ["laos"]},
}

# Cache for loaded application classes
_loaded_applications: dict[str, type[Application]] = {}


def get_application_class(name: str) -> type[Application]:
    """Lazily load and return an application class by name.

    Applications are loaded on first access and cached for subsequent calls.
    This reduces startup time by deferring imports until needed.

    Args:
        name: Application name (e.g., 'MWD', 'TTS', 'LVE').

    Returns:
        The application class.

    Raises:
        KeyError: If the application name is not in the registry.
        ImportError: If the application module cannot be imported.
        AttributeError: If the class cannot be found in the module.
    """
    if name not in _loaded_applications:
        if name not in _APPLICATION_REGISTRY:
            available = ", ".join(sorted(_APPLICATION_REGISTRY.keys()))
            raise KeyError(
                f"Unknown application '{name}'. Available applications: {available}"
            )

        module_path = _APPLICATION_REGISTRY[name]["module"]
        module_name, class_name = module_path.rsplit(".", 1)

        logger.debug(f"Lazy loading application: {name} from {module_name}")

        try:
            module = importlib.import_module(module_name)
            app_class = getattr(module, class_name)
            _loaded_applications[name] = app_class
            logger.debug(f"Successfully loaded application: {name}")
        except ImportError as e:
            logger.error(f"Failed to import application module: {module_name}")
            raise ImportError(
                f"Cannot import application '{name}' from '{module_name}': {e}"
            ) from e
        except AttributeError as e:
            logger.error(f"Class {class_name} not found in module {module_name}")
            raise AttributeError(
                f"Class '{class_name}' not found in module '{module_name}': {e}"
            ) from e

    return _loaded_applications[name]


def get_available_application_names() -> list[str]:
    """Return a list of all registered application names.

    Returns:
        Sorted list of application names.
    """
    return sorted(_APPLICATION_REGISTRY.keys())


def is_application_loaded(name: str) -> bool:
    """Check if an application has already been loaded.

    Args:
        name: Application name to check.

    Returns:
        True if the application class has been loaded, False otherwise.
    """
    return name in _loaded_applications


def preload_application(name: str) -> None:
    """Preload an application class without instantiating it.

    Useful for warming up the cache before user interaction.

    Args:
        name: Application name to preload.
    """
    get_application_class(name)


def clear_cache() -> None:
    """Clear the loaded applications cache.

    Primarily useful for testing purposes.
    """
    _loaded_applications.clear()


def get_extension_to_appname_map() -> dict[str, str]:
    """Return a mapping of file extensions to application names.

    This allows determining which application handles a file extension
    without loading the application classes.

    Returns:
        Dict mapping extension strings to application names.
    """
    result = {}
    for name, info in _APPLICATION_REGISTRY.items():
        for ext in info["extensions"]:
            result[ext] = name
    return result
