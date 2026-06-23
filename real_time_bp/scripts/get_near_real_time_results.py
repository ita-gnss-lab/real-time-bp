"""Generate near-real-time BP result CSV outputs."""

from __future__ import annotations

from loguru import logger
import pandas as pd

from real_time_bp.scripts.near_real_time.common import (
    NEAR_REAL_TIME_TRACKING_COLUMNS,
    load_amplitude_tf_estimates,
)
from real_time_bp.scripts.near_real_time.diagnostics import (
    compute_residual_doppler_metrics,
    compute_var_residual_diagnostics,
)
from real_time_bp.scripts.near_real_time.heatmap import compute_heatmap
from real_time_bp.scripts.near_real_time.metrics import (
    compute_practical_case_task,
    iter_practical_tasks,
    select_forecast_case_id,
)
from real_time_bp.utils.io import count_cases, find_dataset_paths, iter_case_dataframes
from real_time_bp.utils.parallel import ordered_parallel_map
from real_time_bp.utils.params import load_params
from real_time_bp.utils.progress import configure_logging, progress_iter
from real_time_bp.utils.representatives import load_representative_case_ids
from real_time_bp.utils.stage import (
    csv_chunksize_from_params,
    csv_stage_dir,
    l1_wavelength_from_params,
    optional_int,
    parallel_workers_from_params,
    write_sorted_csv,
)

STAGE_DIR = "04_near_real_time"
VAR_ORDERS_CSV = "near_real_time_var_orders.csv"
VAR_RESIDUAL_SUMMARY_CSV = "near_real_time_var_residual_summary.csv"
VAR_FORECAST_UTILITY_CSV = "near_real_time_var_forecast_utility.csv"
PRACTICAL_METRICS_CSV = "near_real_time_practical_metrics.csv"
PHASE_EXAMPLE_CSV = "near_real_time_phase_example.csv"
FORECAST_EXAMPLE_CSV = "near_real_time_forecast_example.csv"
LATENCY_HEATMAP_CSV = "near_real_time_latency_heatmap.csv"
RESIDUAL_DOPPLER_METRICS_CSV = "near_real_time_residual_doppler_metrics.csv"
VAR_RESIDUAL_DIAGNOSTICS_CSV = "near_real_time_var_residual_diagnostics.csv"


def main() -> None:
    """Run the near-real-time stage and write practical, heatmap, and diagnostic CSVs.

    Main paper:
        Results Section III.3 consumes the amplitude-standard-deviation T_F
        estimate and evaluates practical near-real-time BP, residual Doppler
        sensitivity, and the M/L latency heatmap.
    """
    configure_logging()
    logger.info("Starting near-real-time results stage")
    params = load_params()
    experiments = params.get("experiments", {})
    out_dir = csv_stage_dir(STAGE_DIR)
    paths = find_dataset_paths(params)
    logger.info("Using dataset root {}", paths.root)
    wavelength_m = l1_wavelength_from_params(params)
    max_cases = optional_int(experiments.get("max_cases_per_condition"))
    chunksize = csv_chunksize_from_params(params)
    tf_estimates = load_amplitude_tf_estimates()
    representative_case_ids = set(load_representative_case_ids().values())
    forecast_case_id = select_forecast_case_id(params, paths, tf_estimates)
    if forecast_case_id is not None:
        logger.info("Using tracking case {} for the VAR forecast example", forecast_case_id)
    workers = parallel_workers_from_params(params)
    logger.info("Using {} parallel worker(s)", workers)
    total_cases = count_cases(
        paths,
        severities=experiments.get("severities"),
        max_cases_per_condition=max_cases,
    )

    var_order_rows: list[dict] = []
    var_residual_summary_rows: list[dict] = []
    var_forecast_utility_rows: list[dict] = []
    practical_rows: list[dict] = []
    example_rows: list[dict] = []
    forecast_rows: list[dict] = []

    case_iter = iter_case_dataframes(
        paths,
        severities=experiments.get("severities"),
        max_cases_per_condition=max_cases,
        chunksize=chunksize,
        columns=NEAR_REAL_TIME_TRACKING_COLUMNS,
    )
    task_iter = iter_practical_tasks(
        case_iter,
        params,
        wavelength_m,
        tf_estimates,
        representative_case_ids,
        forecast_case_id,
    )
    result_iter = ordered_parallel_map(
        compute_practical_case_task,
        task_iter,
        workers=workers,
    )
    for result in progress_iter(
        result_iter,
        desc="near-real-time cases",
        total=total_cases,
        unit="case",
    ):
        if result is None:
            continue
        var_order_rows.append(result["var_order_row"])
        var_residual_summary_rows.extend(result["var_residual_summary_rows"])
        var_forecast_utility_rows.extend(result["var_forecast_utility_rows"])
        practical_rows.append(result["practical_row"])
        example_rows.extend(result["example_rows"])
        forecast_rows.extend(result["forecast_rows"])
    residual_metrics = compute_residual_doppler_metrics(
        params,
        paths,
        wavelength_m,
        tf_estimates,
        representative_case_ids,
    )
    residual_diagnostics = compute_var_residual_diagnostics(
        params,
        paths,
        wavelength_m,
        representative_case_ids,
    )

    heatmap_df = compute_heatmap(params, paths, wavelength_m, tf_estimates)
    write_sorted_csv(
        pd.DataFrame(var_order_rows),
        out_dir / VAR_ORDERS_CSV,
        ["tracking_case_id"],
    )
    write_sorted_csv(
        pd.DataFrame(var_residual_summary_rows),
        out_dir / VAR_RESIDUAL_SUMMARY_CSV,
        ["tracking_case_id", "requested_whiteness_lag"],
    )
    write_sorted_csv(
        pd.DataFrame(var_forecast_utility_rows),
        out_dir / VAR_FORECAST_UTILITY_CSV,
        ["tracking_case_id", "forecast_horizon_samples"],
    )
    write_sorted_csv(
        pd.DataFrame(practical_rows),
        out_dir / PRACTICAL_METRICS_CSV,
        ["tracking_case_id"],
    )
    write_sorted_csv(
        pd.DataFrame(example_rows),
        out_dir / PHASE_EXAMPLE_CSV,
        ["tracking_case_id", "sample_index"],
    )
    write_sorted_csv(
        pd.DataFrame(forecast_rows),
        out_dir / FORECAST_EXAMPLE_CSV,
        ["tracking_case_id", "latency_samples", "sample_index"],
    )
    write_sorted_csv(
        heatmap_df,
        out_dir / LATENCY_HEATMAP_CSV,
        ["tracking_case_id", "block_len", "latency_samples"],
    )
    write_sorted_csv(
        residual_metrics,
        out_dir / RESIDUAL_DOPPLER_METRICS_CSV,
        ["tracking_case_id", "residual_doppler_error_hz"],
    )
    write_sorted_csv(
        residual_diagnostics,
        out_dir / VAR_RESIDUAL_DIAGNOSTICS_CSV,
        ["tracking_case_id", "component", "diagnostic", "sample_index", "lag"],
    )
    logger.info("Finished near-real-time results stage")


if __name__ == "__main__":
    main()
