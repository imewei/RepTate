"""Model registry for existing RepTate theory modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class ModelSpec:
    """Lightweight descriptor for a model class discovered in theories/."""

    model_id: str
    module: str
    class_name: str


class ModelRegistry:
    """Registry of available RepTate models with discovery helpers."""

    def __init__(self, specs: list[ModelSpec]) -> None:
        self._specs = {spec.model_id: spec for spec in specs}

    @classmethod
    def from_theories(cls) -> "ModelRegistry":
        """Create a registry by discovering all models in the theories/ directory.

        Automatically scans the RepTate theories directory for Theory classes
        and builds a registry of available models. This is the primary factory
        method for initializing the model registry.

        Returns:
            ModelRegistry: A new registry populated with all discovered theory models.
        """
        return cls(discover_theory_models())

    def list_models(self) -> list[ModelSpec]:
        """Return all registered models sorted by model ID.

        Returns:
            list[ModelSpec]: List of model specifications sorted alphabetically by model_id.
        """
        return sorted(self._specs.values(), key=lambda spec: spec.model_id)

    def get(self, model_id: str) -> ModelSpec:
        """Retrieve a model specification by its unique identifier.

        Args:
            model_id: The unique identifier for the model (typically the class name).

        Returns:
            ModelSpec: The model specification containing module and class information.

        Raises:
            KeyError: If the model_id is not found in the registry.
        """
        return self._specs[model_id]


def discover_theory_models(theories_path: Path | None = None) -> list[ModelSpec]:
    """Discover all theory model classes in the RepTate theories directory.

    Scans Python files matching the pattern 'Theory*.py' in the theories directory
    and extracts all class names that inherit from theory base classes. Each
    discovered class is wrapped in a ModelSpec descriptor.

    Args:
        theories_path: Optional custom path to the theories directory. If None,
            defaults to RepTate/theories/ relative to this module.

    Returns:
        list[ModelSpec]: List of model specifications for all discovered theory classes.
            Returns an empty list if the theories directory does not exist.
    """
    package_root = Path(__file__).resolve().parents[3]
    theories_dir = theories_path or (package_root / "theories")
    if not theories_dir.exists():
        return []

    specs: list[ModelSpec] = []
    for path in sorted(theories_dir.glob("Theory*.py")):
        class_names = _find_theory_classes(path)
        module_name = f"RepTate.theories.{path.stem}"
        for class_name in class_names:
            specs.append(
                ModelSpec(
                    model_id=class_name,
                    module=module_name,
                    class_name=class_name,
                )
            )
    return specs


def _find_theory_classes(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    candidates = re.findall(r"^class\\s+(Theory\\w+)\\s*\\(", text, re.MULTILINE)
    return sorted(set(candidates))
