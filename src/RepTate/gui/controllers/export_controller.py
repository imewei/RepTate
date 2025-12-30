"""Controller for export workflow."""

from __future__ import annotations

from typing import Iterable

from RepTate.core.models.results import FitResultRecord, PosteriorResultRecord
from RepTate.gui.exports.export_service import export_results


class ExportController:
    """Bridge UI export requests to the export service."""

    def export(
        self,
        output_dir: str,
        *,
        fit_record: FitResultRecord,
        posterior_record: PosteriorResultRecord | None,
        figures: Iterable[object] | None = None,
    ) -> list[str]:
        return export_results(
            output_dir,
            fit_record=fit_record,
            posterior_record=posterior_record,
            figures=figures,
        )
