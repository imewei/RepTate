"""Export pipeline for plots, results, and raw posterior traces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from RepTate.core.models.results import FitResultRecord, PosteriorResultRecord


def export_results(
    output_dir: str | Path,
    *,
    fit_record: FitResultRecord,
    posterior_record: PosteriorResultRecord | None,
    figures: Iterable[object] | None = None,
) -> list[str]:
    """Export fit/posterior artifacts and optional plot figures.

    Serializes fit results and posterior samples to JSON files and saves
    matplotlib figures as PNG images in the specified output directory.
    Creates the directory if it does not exist.

    Args:
        output_dir: Directory path (string or Path object) where all exported
            artifacts will be saved. Parent directories are created if needed.
        fit_record: Deterministic fit results containing optimized parameters,
            residuals, dataset/model IDs, and optimization metadata.
        posterior_record: Optional Bayesian inference results containing MCMC
            samples and diagnostics. If None, only fit results are exported.
        figures: Optional iterable of matplotlib figure objects to save as PNG
            files. Each figure must have a savefig method. Figures are numbered
            sequentially (plot_1.png, plot_2.png, etc.).

    Returns:
        List of absolute file paths (as strings) to all successfully exported
        artifacts, in creation order. Typically includes:
            - {result_id}_fit.json
            - {result_id}_posterior.json (if posterior_record provided)
            - plot_1.png, plot_2.png, ... (if figures provided)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifacts: list[str] = []

    fit_path = output_path / f"{fit_record.result_id}_fit.json"
    fit_path.write_text(json.dumps(fit_record.__dict__, indent=2, sort_keys=True))
    artifacts.append(str(fit_path))

    if posterior_record:
        posterior_path = output_path / f"{posterior_record.result_id}_posterior.json"
        posterior_path.write_text(
            json.dumps(posterior_record.__dict__, indent=2, sort_keys=True)
        )
        artifacts.append(str(posterior_path))

    if figures:
        for idx, fig in enumerate(figures):
            fig_path = output_path / f"plot_{idx + 1}.png"
            if hasattr(fig, "savefig"):
                fig.savefig(fig_path)
                artifacts.append(str(fig_path))

    return artifacts
