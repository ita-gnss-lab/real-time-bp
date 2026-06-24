"""Generate the full-dataset sliding latency heatmap CSV outputs."""

from __future__ import annotations

from loguru import logger

from real_time_bp.scripts.sliding.common import load_amplitude_tf_estimates
from real_time_bp.scripts.sliding.heatmap import aggregate_heatmap_cases, compute_heatmap
from real_time_bp.utils.io import find_dataset_paths
from real_time_bp.utils.params import load_params
from real_time_bp.utils.progress import configure_logging
from real_time_bp.utils.stage import (
    csv_stage_dir,
    l1_wavelength_from_params,
    parallel_workers_from_params,
    write_sorted_csv,
)

STAGE_DIR = "04_sliding"
LATENCY_HEATMAP_CASES_CSV = "sliding_latency_heatmap_cases.csv"
LATENCY_HEATMAP_CSV = "sliding_latency_heatmap.csv"


def main() -> None:
    """Run the expensive full-dataset M/L heatmap stage."""
    configure_logging()
    logger.info("Starting full-dataset sliding heatmap stage")
    params = load_params()
    paths = find_dataset_paths(params)
    wavelength_m = l1_wavelength_from_params(params)
    tf_estimates = load_amplitude_tf_estimates()
    out_dir = csv_stage_dir(STAGE_DIR)
    logger.info("Using {} parallel worker(s)", parallel_workers_from_params(params))

    case_rows = compute_heatmap(params, paths, wavelength_m, tf_estimates)
    write_sorted_csv(
        case_rows,
        out_dir / LATENCY_HEATMAP_CASES_CSV,
        ["tracking_case_id", "block_len", "latency_samples"],
    )
    write_sorted_csv(
        aggregate_heatmap_cases(case_rows),
        out_dir / LATENCY_HEATMAP_CSV,
        ["block_len", "latency_samples"],
    )
    logger.info("Finished full-dataset sliding heatmap stage")


if __name__ == "__main__":
    main()
