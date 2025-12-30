"""Benchmark export timing for SC-004."""

from __future__ import annotations

import time
from pathlib import Path

from RepTate.gui.exports.export_service import export_results
from RepTate.core.models.results import FitResultRecord, PosteriorResultRecord


def main() -> None:
    fit_record = FitResultRecord(
        result_id="bench_fit",
        dataset_id="bench_dataset",
        model_id="bench_model",
        parameter_estimates={"p0": 1.0},
        diagnostics={},
        residuals=[0.0],
        execution_context={"backend": "jax"},
        status="completed",
    )
    posterior_record = PosteriorResultRecord(
        result_id="bench_post",
        fit_result_id="bench_fit",
        sample_traces={"p0": [1.0, 1.0]},
        summary_statistics={"p0": {"mean": 1.0}},
        chain_metadata={},
        resume_state={},
        status="completed",
    )
    output_dir = Path("tests/regression/_export_bench")
    start = time.perf_counter()
    export_results(
        output_dir,
        fit_record=fit_record,
        posterior_record=posterior_record,
        figures=None,
    )
    duration = time.perf_counter() - start
    print(f"Export timing: {duration:.3f}s")


if __name__ == "__main__":
    main()
