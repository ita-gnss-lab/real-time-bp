"""Generate sliding BP result CSV outputs."""

from __future__ import annotations

from loguru import logger
import pandas as pd

from real_time_bp.scripts.sliding.common import (
    SLIDING_TRACKING_COLUMNS,
    load_amplitude_tf_estimates,
)
from real_time_bp.scripts.sliding.metrics import (
    compute_practical_case_task,
    iter_practical_tasks,
    select_forecast_case_ids,
)
from real_time_bp.utils.io import count_cases, find_dataset_paths, iter_case_dataframes
from real_time_bp.utils.parallel import ordered_parallel_map
from real_time_bp.utils.params import load_params
from real_time_bp.utils.progress import configure_logging, progress_iter
from real_time_bp.utils.quality import exclude_loss_of_lock_enabled
from real_time_bp.utils.stage import (
    csv_chunksize_from_params,
    csv_stage_dir,
    l1_wavelength_from_params,
    optional_int,
    parallel_workers_from_params,
    write_sorted_csv,
)

STAGE_DIR = "04_sliding"
VAR_ORDERS_CSV = "sliding_var_orders.csv"
VAR_RESIDUAL_SUMMARY_CSV = "sliding_var_residual_summary.csv"
VAR_FORECAST_UTILITY_CSV = "sliding_var_forecast_utility.csv"
PRACTICAL_METRICS_CSV = "sliding_practical_metrics.csv"
PHASE_EXAMPLE_CSV = "sliding_phase_example.csv"
FORECAST_EXAMPLE_CSV = "sliding_forecast_example.csv"


def main() -> None:
    """Run the sliding stage and write practical result CSVs.

    Main paper:
        Results Section III.3 consumes the amplitude-standard-deviation T_F
        estimate and evaluates practical sliding BP. The expensive
        full-dataset M/L heatmap is kept in a separate DVC stage.
    """
    configure_logging()
    logger.info("Starting sliding results stage")
    params = load_params()
    experiments = params.get("experiments", {})
    out_dir = csv_stage_dir(STAGE_DIR)
    paths = find_dataset_paths(params)
    logger.info("Using dataset root {}", paths.root)
    wavelength_m = l1_wavelength_from_params(params)
    max_cases = optional_int(experiments.get("max_cases_per_condition"))
    chunksize = csv_chunksize_from_params(params)
    tf_estimates = load_amplitude_tf_estimates()
    forecast_case_ids = select_forecast_case_ids(params, paths, tf_estimates)
    if forecast_case_ids:
        logger.info("Using VAR forecast examples {}", forecast_case_ids)
    workers = parallel_workers_from_params(params)
    exclude_loss_of_lock = exclude_loss_of_lock_enabled(params)
    logger.info("Using {} parallel worker(s)", workers)
    raw_total_cases = count_cases(
        paths,
        severities=experiments.get("severities"),
        max_cases_per_condition=max_cases,
        exclude_loss_of_lock=False,
    )
    total_cases = count_cases(
        paths,
        severities=experiments.get("severities"),
        max_cases_per_condition=max_cases,
        exclude_loss_of_lock=exclude_loss_of_lock,
    )
    if raw_total_cases is not None and total_cases is not None:
        logger.info(
            "Sliding cases: {} after loss-of-lock filtering from {} raw case(s)",
            total_cases,
            raw_total_cases,
        )
    logger.info(
        "Sliding practical setting: N_init={}, M={}, L={}",
        int(experiments.get("n_init_samples", 4096)),
        int(params.get("sliding", {}).get("practical_fft_m", 8192)),
        int(params.get("sliding", {}).get("practical_latency_samples", 4095)),
    )

    var_order_rows: list[dict] = []
    var_residual_summary_rows: list[dict] = []
    var_forecast_utility_rows: list[dict] = []
    practical_rows: list[dict] = []
    forecast_rows: list[dict] = []

    case_iter = iter_case_dataframes(
        paths,
        severities=experiments.get("severities"),
        max_cases_per_condition=max_cases,
        chunksize=chunksize,
        columns=SLIDING_TRACKING_COLUMNS,
        exclude_loss_of_lock=exclude_loss_of_lock,
    )
    task_iter = iter_practical_tasks(
        case_iter,
        params,
        wavelength_m,
        tf_estimates,
        set(),
        forecast_case_ids,
        include_phase_examples=False,
    )
    result_iter = ordered_parallel_map(
        compute_practical_case_task,
        task_iter,
        workers=workers,
    )
    for result in progress_iter(
        result_iter,
        desc="sliding cases",
        total=total_cases,
        unit="case",
    ):
        if result is None:
            continue
        var_order_rows.append(result["var_order_row"])
        var_residual_summary_rows.extend(result["var_residual_summary_rows"])
        var_forecast_utility_rows.extend(result["var_forecast_utility_rows"])
        practical_rows.append(result["practical_row"])
        forecast_rows.extend(result["forecast_rows"])

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
        pd.DataFrame(forecast_rows),
        out_dir / FORECAST_EXAMPLE_CSV,
        ["tracking_case_id", "latency_samples", "sample_index"],
    )
    logger.info("Finished sliding results stage")


if __name__ == "__main__":
    main()
