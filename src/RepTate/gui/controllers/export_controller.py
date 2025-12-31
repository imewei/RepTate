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
        """Export fit results, posterior samples, and optional figures to disk.

        Delegates to the export service to serialize fit and posterior records
        as JSON files and save matplotlib figures as PNG files in the specified
        output directory.

        Args:
            output_dir: Directory path where exported artifacts will be saved.
                Directory will be created if it does not exist.
            fit_record: Deterministic fit results containing optimized parameters,
                residuals, and metadata.
            posterior_record: Optional Bayesian inference results containing MCMC
                samples and diagnostics. If None, only fit results are exported.
            figures: Optional iterable of matplotlib figure objects to save as
                PNG files. Each figure must have a savefig method.

        Returns:
            List of absolute file paths to all exported artifacts (JSON files
            and PNG images), in the order they were created.
        """
        return export_results(
            output_dir,
            fit_record=fit_record,
            posterior_record=posterior_record,
            figures=figures,
        )
