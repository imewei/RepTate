"""Viewmodel for uncertainty summaries and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

from RepTate.core.models import PosteriorResultRecord


@dataclass(frozen=True)
class PosteriorViewModel:
    result: PosteriorResultRecord

    @property
    def status(self) -> str:
        return self.result.status

    @property
    def summaries(self) -> dict[str, object]:
        return self.result.summary_statistics
