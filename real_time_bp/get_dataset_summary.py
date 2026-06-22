from __future__ import annotations

from loguru import logger
import pandas as pd

from real_time_bp.stage_utils import (
    csv_stage_dir,
    residual_doppler_grid_hz,
    tf_candidate_grid,
    write_csv,
)
from real_time_bp.utils.io import find_dataset_paths, load_case_index
from real_time_bp.utils.params import load_params
from real_time_bp.utils.progress import configure_logging


def main() -> None:
    configure_logging()
    logger.info("Starting dataset summary stage")
    params = load_params()
    out_dir = csv_stage_dir("01_dataset")
    paths = find_dataset_paths(params)
    logger.info("Using dataset root {}", paths.root)
    case_index = load_case_index(paths)

    # Paper: Results Section III.1. These summaries are
    # the source tables for carrier-to-noise, severity, T_F, and dataset-path
    # descriptions before the analysis stages run.
    if "input_c_n0_db_hz" not in case_index.columns:
        case_index["input_c_n0_db_hz"] = pd.NA
    if "rhof_veff_ratio" not in case_index.columns:
        case_index["rhof_veff_ratio"] = pd.NA

    summary = (
        case_index.groupby(["severity", "frequency"], dropna=False)
        .agg(
            n_cases=("severity", "size"),
            n_c_n0=("input_c_n0_db_hz", "nunique"),
            min_c_n0_db_hz=("input_c_n0_db_hz", "min"),
            max_c_n0_db_hz=("input_c_n0_db_hz", "max"),
            min_true_fresnel_time_scale_s=("rhof_veff_ratio", "min"),
            max_true_fresnel_time_scale_s=("rhof_veff_ratio", "max"),
        )
        .reset_index()
    )
    dataset_paths = pd.DataFrame(
        [
            {
                "dataset_root": str(paths.root),
                "layout": paths.layout,
                "diagnostics_csv": str(paths.diagnostics_csv),
                "n_measurement_csvs": len(paths.measurement_csvs),
            }
        ]
    )
    # Paper: Methodology Section II.4 and Results
    # Section III.2; persist the exact candidate grid so plots and
    # tables can be regenerated without recomputing configuration logic.
    tf_candidates = pd.DataFrame({"fresnel_time_candidate_s": tf_candidate_grid(params)})

    write_csv(dataset_paths, out_dir / "dataset_paths.csv")
    write_csv(case_index, out_dir / "case_index.csv")
    write_csv(summary, out_dir / "dataset_summary.csv")
    write_csv(residual_doppler_grid_hz(params), out_dir / "residual_doppler_grid.csv")
    write_csv(tf_candidates, out_dir / "fresnel_time_candidate_grid.csv")
    logger.info("Finished dataset summary stage")


if __name__ == "__main__":
    main()
