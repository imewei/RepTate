"""Viewmodel for deterministic fit results."""

from __future__ import annotations

from dataclasses import dataclass

from RepTate.core.models import FitResultRecord


@dataclass(frozen=True)
class FitViewModel:
    result: FitResultRecord

    @property
    def status(self) -> str:
        return self.result.status

    @property
    def parameters(self) -> dict[str, float]:
        return self.result.parameter_estimates
