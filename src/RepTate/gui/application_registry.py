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

# Registry mapping application names to their module paths
# Format: 'AppName': 'module.path.ClassName'
_APPLICATION_REGISTRY: dict[str, str] = {
    "MWD": "RepTate.applications.ApplicationMWD.ApplicationMWD",
    "TTS": "RepTate.applications.ApplicationTTS.ApplicationTTS",
    "TTSF": "RepTate.applications.ApplicationTTSFactors.ApplicationTTSFactors",
    "LVE": "RepTate.applications.ApplicationLVE.ApplicationLVE",
    "NLVE": "RepTate.applications.ApplicationNLVE.ApplicationNLVE",
    "Crystal": "RepTate.applications.ApplicationCrystal.ApplicationCrystal",
    "Gt": "RepTate.applications.ApplicationGt.ApplicationGt",
    "Creep": "RepTate.applications.ApplicationCreep.ApplicationCreep",
    "SANS": "RepTate.applications.ApplicationSANS.ApplicationSANS",
    "React": "RepTate.applications.ApplicationReact.ApplicationReact",
    "Dielectric": "RepTate.applications.ApplicationDielectric.ApplicationDielectric",
    "LAOS": "RepTate.applications.ApplicationLAOS.ApplicationLAOS",
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

        module_path = _APPLICATION_REGISTRY[name]
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
