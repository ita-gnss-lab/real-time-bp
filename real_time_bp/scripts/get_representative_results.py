"""Generate lightweight representative-example CSV outputs for paper figures."""

from __future__ import annotations

from loguru import logger
import numpy as np
import pandas as pd

from real_time_bp.scripts.get_cycle_slip_results import (
    REPRESENTATIVE_COLUMNS,
    REPRESENTATIVES_CSV,
    TRACE_COLUMNS,
    TRACES_CSV,
    TVD_TRACKING_COLUMNS,
    _compute_tvd_trace_rows,
    _representative_rank_key,
    _representative_rows,
)
from real_time_bp.scripts.get_cycle_slip_results import (
    STAGE_DIR as CYCLE_STAGE_DIR,
)
from real_time_bp.scripts.get_cycle_slip_results import SUMMARY_CSV as TVD_SUMMARY_CSV
from real_time_bp.scripts.get_filter_bank_results import (
    FILTER_BANK_TRACKING_COLUMNS,
    _compute_case_filter_bank,
    _score_curve_n_init_samples,
)
from real_time_bp.scripts.get_filter_bank_results import (
    REPRESENTATIVE_TRACES_CSV as FILTER_BANK_REPRESENTATIVE_TRACES_CSV,
)
from real_time_bp.scripts.get_filter_bank_results import (
    STAGE_DIR as FILTER_BANK_STAGE_DIR,
)
from real_time_bp.scripts.near_real_time.common import (
    NEAR_REAL_TIME_TRACKING_COLUMNS,
    load_amplitude_tf_estimates,
)
from real_time_bp.scripts.near_real_time.metrics import compute_practical_case
from real_time_bp.utils.io import find_dataset_paths, iter_case_dataframes
from real_time_bp.utils.params import load_params
from real_time_bp.utils.progress import configure_logging, progress_iter
from real_time_bp.utils.stage import (
    csv_chunksize_from_params,
    csv_stage_dir,
    l1_wavelength_from_params,
    tf_candidate_grid,
    write_csv,
    write_sorted_csv,
)

NEAR_REAL_TIME_STAGE_DIR = "04_near_real_time"
FILTER_BANK_REPRESENTATIVE_SCORE_CURVES_CSV = "filter_bank_representative_score_curves.csv"
NEAR_REAL_TIME_PHASE_EXAMPLE_CSV = "near_real_time_phase_example.csv"


def main() -> None:
    """Write representative TVD, filter-bank, and near-real-time example CSVs.

    Main paper:
        The representative figures are diagnostic examples, not full-dataset
        metrics. Keeping them in a separate DVC stage lets the rank in
        `params.yaml` change without invalidating the all-case result stages.
    """
    configure_logging()
    logger.info("Starting representative result stage")
    params = load_params()
    representatives = params.get("representatives", {})
    paths = find_dataset_paths(params)
    wavelength_m = l1_wavelength_from_params(params)
    chunksize = csv_chunksize_from_params(params)
    selected_rows = select_representative_rows(params)
    selected_case_ids = {
        int(row["tracking_case_id"])
        for row in selected_rows
        if pd.notna(row.get("tracking_case_id"))
    }
    logger.info("Selected representative case ids {}", sorted(selected_case_ids))

    cycle_dir = csv_stage_dir(CYCLE_STAGE_DIR)
    write_csv(
        pd.DataFrame(selected_rows, columns=REPRESENTATIVE_COLUMNS),
        cycle_dir / REPRESENTATIVES_CSV,
    )
    tvd_trace_rows = compute_tvd_trace_rows(
        params,
        paths,
        selected_case_ids,
        wavelength_m,
        chunksize,
    )
    validate_selected_cases(tvd_trace_rows, selected_case_ids, "TVD traces")
    write_sorted_csv(
        pd.DataFrame(tvd_trace_rows, columns=TRACE_COLUMNS),
        cycle_dir / TRACES_CSV,
        ["tracking_case_id", "sample_index"],
    )

    score_rows, filter_trace_rows = compute_filter_bank_representative_rows(
        params,
        paths,
        selected_case_ids,
        wavelength_m,
        chunksize,
    )
    validate_selected_cases(filter_trace_rows, selected_case_ids, "filter-bank traces")
    filter_dir = csv_stage_dir(FILTER_BANK_STAGE_DIR)
    write_sorted_csv(
        pd.DataFrame(score_rows),
        filter_dir / FILTER_BANK_REPRESENTATIVE_SCORE_CURVES_CSV,
        ["tracking_case_id", "n_init_samples", "fresnel_time_candidate_s"],
    )
    write_sorted_csv(
        pd.DataFrame(filter_trace_rows),
        filter_dir / FILTER_BANK_REPRESENTATIVE_TRACES_CSV,
        ["tracking_case_id", "sample_index"],
    )

    phase_rows = compute_near_real_time_phase_rows(
        params,
        paths,
        selected_case_ids,
        wavelength_m,
        chunksize,
    )
    validate_selected_cases(phase_rows, selected_case_ids, "near-real-time phase traces")
    write_sorted_csv(
        pd.DataFrame(phase_rows),
        csv_stage_dir(NEAR_REAL_TIME_STAGE_DIR) / NEAR_REAL_TIME_PHASE_EXAMPLE_CSV,
        ["tracking_case_id", "sample_index"],
    )
    logger.info(
        "Finished representative result stage using ranks {}",
        representatives.get("case_rank_by_severity", {}),
    )


def select_representative_rows(params: dict) -> list[dict]:
    """Select representative cases from the already generated TVD summary table."""
    summary_path = csv_stage_dir(CYCLE_STAGE_DIR) / TVD_SUMMARY_CSV
    if not summary_path.exists() or summary_path.stat().st_size == 0:
        return []
    summary_df = pd.read_csv(summary_path)
    candidates_by_severity: dict[str, list[tuple[tuple[float, ...], dict]]] = {}
    for summary_row in summary_df.to_dict("records"):
        rank_key = _representative_rank_key(summary_row)
        if rank_key is None:
            continue
        severity = str(summary_row.get("severity"))
        candidates_by_severity.setdefault(severity, []).append((rank_key, summary_row))
    rank_experiments = {
        **params.get("experiments", {}),
        "representative_case_rank_by_severity": params.get("representatives", {}).get(
            "case_rank_by_severity",
            {},
        ),
    }
    return _representative_rows(candidates_by_severity, rank_experiments)


def compute_tvd_trace_rows(
    params: dict,
    paths,
    selected_case_ids: set[int],
    wavelength_m: float,
    chunksize: int,
) -> list[dict]:
    """Compute TVD traces only for selected representative cases."""
    if not selected_case_ids:
        return []
    experiments = params.get("experiments", {})
    tvd_cfg = params.get("tvd", {})
    rows: list[dict] = []
    case_iter = iter_case_dataframes(
        paths,
        severities=experiments.get("severities"),
        max_cases_per_condition=None,
        chunksize=chunksize,
        columns=TVD_TRACKING_COLUMNS,
    )
    for metadata, case_df in progress_iter(
        iter_selected_cases(case_iter, selected_case_ids),
        desc="representative TVD cases",
        total=len(selected_case_ids),
        unit="case",
    ):
        rows.extend(
            _compute_tvd_trace_rows(
                metadata,
                case_df,
                float(wavelength_m),
                int(experiments.get("n_init_samples", 4096)),
                float(tvd_cfg.get("fade_threshold_db", -10.0)),
                float(tvd_cfg.get("allowed_jump_weight", 1.0)),
                float(tvd_cfg.get("blocked_jump_weight", 10000.0)),
            )
        )
    return rows


def compute_filter_bank_representative_rows(
    params: dict,
    paths,
    selected_case_ids: set[int],
    wavelength_m: float,
    chunksize: int,
) -> tuple[list[dict], list[dict]]:
    """Compute filter-bank score curves and phase traces for selected cases."""
    if not selected_case_ids:
        return [], []
    experiments = params.get("experiments", {})
    candidates = tf_candidate_grid(params)
    score_n_init_samples = _score_curve_n_init_samples(experiments)
    score_rows: list[dict] = []
    trace_rows: list[dict] = []
    case_iter = iter_case_dataframes(
        paths,
        severities=experiments.get("severities"),
        max_cases_per_condition=None,
        chunksize=chunksize,
        columns=FILTER_BANK_TRACKING_COLUMNS,
    )
    for metadata, case_df in progress_iter(
        iter_selected_cases(case_iter, selected_case_ids),
        desc="representative filter-bank cases",
        total=len(selected_case_ids),
        unit="case",
    ):
        result = _compute_case_filter_bank(
            metadata,
            case_df,
            params,
            candidates,
            float(wavelength_m),
            residual_doppler_error_hz=0.0,
            n_init=int(experiments.get("n_init_samples", 4096)),
            score_n_init_samples=score_n_init_samples,
            include_trace=True,
        )
        score_rows.extend(result["score_rows"])
        trace_rows.extend(result["trace_rows"])
    return score_rows, trace_rows


def compute_near_real_time_phase_rows(
    params: dict,
    paths,
    selected_case_ids: set[int],
    wavelength_m: float,
    chunksize: int,
) -> list[dict]:
    """Compute near-real-time phase examples only for selected cases."""
    if not selected_case_ids:
        return []
    experiments = params.get("experiments", {})
    tf_estimates = load_amplitude_tf_estimates()
    rows: list[dict] = []
    case_iter = iter_case_dataframes(
        paths,
        severities=experiments.get("severities"),
        max_cases_per_condition=None,
        chunksize=chunksize,
        columns=NEAR_REAL_TIME_TRACKING_COLUMNS,
    )
    for metadata, case_df in progress_iter(
        iter_selected_cases(case_iter, selected_case_ids),
        desc="representative near-real-time cases",
        total=len(selected_case_ids),
        unit="case",
    ):
        case_id = int(metadata.get("tracking_case_id", 0) or 0)
        estimated_tf = tf_estimates.get(case_id, float(metadata.get("rhof_veff_ratio", np.nan)))
        if not np.isfinite(estimated_tf):
            continue
        result = compute_practical_case(
            metadata,
            case_df,
            params,
            float(wavelength_m),
            float(estimated_tf),
            include_phase_example=True,
            include_forecast_example=False,
        )
        rows.extend(result["example_rows"])
    return rows


def iter_selected_cases(case_iter, selected_case_ids: set[int]):
    """Yield dataset cases whose tracking ids are in `selected_case_ids`."""
    yielded: set[int] = set()
    for metadata, case_df in case_iter:
        case_id = int(metadata.get("tracking_case_id", 0) or 0)
        if case_id not in selected_case_ids:
            continue
        yielded.add(case_id)
        yield metadata, case_df
        if yielded == selected_case_ids:
            return


def validate_selected_cases(rows: list[dict], selected_case_ids: set[int], label: str) -> None:
    """Raise a clear error when representative ids are absent from raw data."""
    present = {
        int(row["tracking_case_id"])
        for row in rows
        if row.get("tracking_case_id") is not None and pd.notna(row.get("tracking_case_id"))
    }
    missing = sorted(selected_case_ids - present)
    if missing:
        raise RuntimeError(
            f"Missing representative case ids in {label}: {missing}. "
            "Regenerate the core CSV stages once so the TVD summary matches the current dataset."
        )


if __name__ == "__main__":
    main()
