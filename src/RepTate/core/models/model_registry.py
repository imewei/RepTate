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
        return cls(discover_theory_models())

    def list_models(self) -> list[ModelSpec]:
        return sorted(self._specs.values(), key=lambda spec: spec.model_id)

    def get(self, model_id: str) -> ModelSpec:
        return self._specs[model_id]


def discover_theory_models(theories_path: Path | None = None) -> list[ModelSpec]:
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
