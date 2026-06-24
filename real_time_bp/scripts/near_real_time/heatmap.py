"""Latency and block-length heatmap generation for near-real-time BP."""

from __future__ import annotations

from loguru import logger
import numpy as np
import pandas as pd

from real_time_bp.methodology.backpropagation import median_align_phase, unwrap_phase
from real_time_bp.methodology.metrics import cycle_slips_per_minute, phase_rmse_rad
from real_time_bp.methodology.near_real_time import (
    select_var_order_bic,
    sliding_window_bp_with_var_forecast,
)
from real_time_bp.methodology.signal_model import (
    estimate_sample_interval_s,
    estimated_scintillation_signal,
    true_refractive_phase_rad,
)
from real_time_bp.scripts.near_real_time.common import (
    NEAR_REAL_TIME_TRACKING_COLUMNS,
    quality_valid_case_ids,
    valid_eval_indices,
)
from real_time_bp.utils.io import count_cases, iter_case_dataframes, load_case_index
from real_time_bp.utils.parallel import ordered_parallel_map
from real_time_bp.utils.params import fixed_step_values
from real_time_bp.utils.progress import progress_iter
from real_time_bp.utils.quality import evaluate_analysis_quality
from real_time_bp.utils.representatives import SEVERITY_ORDER
from real_time_bp.utils.stage import (
    case_keys,
    csv_chunksize_from_params,
    is_loss_of_lock,
    optional_int,
    parallel_workers_from_params,
)


def compute_heatmap(
    params: dict, paths, wavelength_m: float, tf_estimates: dict[int, float]
) -> pd.DataFrame:
    """Compute full-dataset near-real-time M/L heatmap rows.

    Main paper:
        Results Figure 11 evaluates FFT length M and latency L over all
        accepted cases and all valid time samples. For each M, the valid
        latency domain is 0 <= L <= M-M_c-1, producing the triangular heatmap
        domain.

    Args:
        params: Parsed `params.yaml` dictionary.
        paths: Resolved dataset paths.
        wavelength_m: Carrier wavelength in meters.
        tf_estimates: Fresnel-time estimates keyed by tracking case id.

    Returns:
        Per-case DataFrame of phase RMSE and cycle-slip metrics for each M/L
        cell.
    """
    experiments = params.get("experiments", {})
    near_cfg = params.get("near_real_time", {})
    rows: list[dict] = []
    m_grid = _m_grid_from_params(near_cfg)
    latency_fractions = _latency_fractions_from_params(near_cfg)
    chunksize = csv_chunksize_from_params(params)
    workers = parallel_workers_from_params(params)
    total_cases = count_cases(paths, severities=experiments.get("severities"))
    heatmap_cells_per_case = _heatmap_cell_count(m_grid, latency_fractions)
    _log_heatmap_plan(total_cases, m_grid, latency_fractions)
    logger.info(
        "Full heatmap grid: {} raw case(s), {} FFT lengths, {} M/L cells per accepted case, no time thinning, {} worker(s)",
        total_cases,
        len(m_grid),
        heatmap_cells_per_case,
        workers,
    )
    case_iter = iter_case_dataframes(
        paths,
        severities=experiments.get("severities"),
        max_cases_per_condition=None,
        chunksize=chunksize,
        columns=NEAR_REAL_TIME_TRACKING_COLUMNS,
    )
    task_iter = (
        (
            metadata,
            case_df,
            params,
            wavelength_m,
            tf_estimates,
            m_grid,
            latency_fractions,
        )
        for metadata, case_df in case_iter
    )
    result_iter = ordered_parallel_map(
        _compute_heatmap_case_task,
        task_iter,
        workers=workers,
    )
    for case_rows in progress_iter(
        result_iter,
        desc="heatmap cases",
        total=total_cases,
        unit="case",
    ):
        rows.extend(case_rows)
    return pd.DataFrame(rows)


def aggregate_heatmap_cases(case_rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-case heatmap rows into the figure heatmap table.

    Args:
        case_rows: Per-case rows returned by `compute_heatmap`.

    Returns:
        DataFrame with one row per severity, M, and latency cell.
    """
    if case_rows.empty:
        return pd.DataFrame()
    group_cols = ["severity", "block_len", "latency_samples", "latency_fraction"]
    return (
        case_rows.groupby(group_cols, dropna=False)
        .agg(
            refractive_phase_rmse_rad=("refractive_phase_rmse_rad", "median"),
            mean_refractive_phase_rmse_rad=("refractive_phase_rmse_rad", "mean"),
            median_refractive_phase_rmse_rad=("refractive_phase_rmse_rad", "median"),
            cycle_slips_per_minute=("cycle_slips_per_minute", "median"),
            mean_cycle_slips_per_minute=("cycle_slips_per_minute", "mean"),
            median_cycle_slips_per_minute=("cycle_slips_per_minute", "median"),
            n_cases=("tracking_case_id", "nunique"),
        )
        .reset_index()
    )


def _compute_heatmap_case_task(task: tuple) -> list[dict]:
    """Compute all heatmap cells for one accepted tracking case.

    Main paper:
        Methodology Eq. (20) fits one VAR model per case and reuses it across
        all M/L heatmap cells. Methodology Eq. (18) defines each sliding-window
        BP evaluation for the chosen FFT length M and latency L.
    """
    (
        metadata,
        case_df,
        params,
        wavelength_m,
        tf_estimates,
        m_grid,
        latency_fractions,
    ) = task
    experiments = params.get("experiments", {})
    near_cfg = params.get("near_real_time", {})
    case_id = int(metadata.get("tracking_case_id", 0) or 0)
    estimated_tf = tf_estimates.get(case_id, float(metadata.get("rhof_veff_ratio", np.nan)))
    if not np.isfinite(estimated_tf):
        return []
    quality = evaluate_analysis_quality(metadata, case_df, params, float(wavelength_m))
    if not quality.passed:
        return []
    sample_interval_s = estimate_sample_interval_s(case_df)
    signal = estimated_scintillation_signal(case_df, float(wavelength_m))
    true_phase = true_refractive_phase_rad(case_df, float(wavelength_m))
    n_init = min(int(experiments.get("n_init_samples", 4096)), len(case_df))
    features = np.column_stack([np.real(signal[:n_init]), np.imag(signal[:n_init])])
    _, var_result = select_var_order_bic(
        features,
        int(near_cfg.get("var_order_min", 1)),
        int(near_cfg.get("var_order_max", 24)),
    )
    logger.info(
        "Heatmap case {} {}: {} cells, {} sliding-BP evaluation(s)",
        case_id,
        metadata.get("severity"),
        _heatmap_cell_count(m_grid, latency_fractions),
        _case_heatmap_evaluation_count(len(case_df), m_grid, latency_fractions),
    )
    rows = []
    for block_len in m_grid:
        future_context = int(block_len) - int(block_len) // 2 - 1
        eval_indices = valid_eval_indices(len(case_df), int(block_len))
        for latency_fraction, latency in _latency_grid(future_context, latency_fractions):
            sliding = sliding_window_bp_with_var_forecast(
                signal,
                sample_interval_s,
                float(estimated_tf),
                int(block_len),
                int(latency),
                eval_indices,
                var_result,
            )
            phase = median_align_phase(unwrap_phase(sliding), true_phase, n_init)
            mask = np.zeros(len(case_df), dtype=bool)
            mask[eval_indices] = True
            mask[:n_init] = False
            slip_count, slips_per_min = cycle_slips_per_minute(
                true_phase[mask],
                phase[mask],
                sample_interval_s,
            )
            rows.append(
                {
                    **case_keys(metadata),
                    "block_len": int(block_len),
                    "latency_samples": int(latency),
                    "latency_fraction": float(latency_fraction),
                    "refractive_phase_rmse_rad": phase_rmse_rad(true_phase[mask], phase[mask]),
                    "cycle_slip_count": slip_count,
                    "cycle_slips_per_minute": slips_per_min,
                }
            )
    return rows


def _log_heatmap_plan(
    total_cases: int,
    m_grid: np.ndarray,
    latency_fractions: np.ndarray,
) -> None:
    """Log the heatmap workload before the expensive stage begins."""
    logger.info("Heatmap M grid: {}", [int(value) for value in m_grid])
    logger.info(
        "Heatmap latency fractions: {}",
        [round(float(value), 3) for value in latency_fractions],
    )
    logger.info(
        "Heatmap upper-bound workload: {} raw case(s), {} cell(s)/case",
        total_cases,
        _heatmap_cell_count(m_grid, latency_fractions),
    )
    for block_len in m_grid:
        future_context = int(block_len) - int(block_len) // 2 - 1
        latency_count = len(_latency_grid(future_context, latency_fractions))
        logger.info(
            "Heatmap M={}: {} latency cell(s); valid samples depend on each case length",
            int(block_len),
            latency_count,
        )


def _case_heatmap_evaluation_count(
    n_samples: int,
    m_grid: np.ndarray,
    latency_fractions: np.ndarray,
) -> int:
    """Count sliding-BP evaluations for one case over the full heatmap grid."""
    total = 0
    for block_len in m_grid:
        future_context = int(block_len) - int(block_len) // 2 - 1
        total += valid_eval_indices(int(n_samples), int(block_len)).size * len(
            _latency_grid(future_context, latency_fractions)
        )
    return int(total)


def select_diagnostic_case_ids(params: dict, paths) -> set[int] | None:
    """Select representative case ids for lightweight diagnostic stages.

    Main paper:
        Residual-Doppler and VAR diagnostics are supporting checks, so a small
        number of representative C/N0 cases are selected per severity.

    Args:
        params: Parsed `params.yaml` dictionary.
        paths: Resolved dataset paths.

    Returns:
        Selected case ids, an empty set when no cases match, or None to use all
        cases.
    """
    near_cfg = params.get("near_real_time", {})
    max_cases = optional_int(near_cfg.get("diagnostic_max_cases_per_severity", 1))
    if max_cases is None:
        return None

    experiments = params.get("experiments", {})
    case_index = load_case_index(paths)
    if case_index.empty or "tracking_case_id" not in case_index.columns:
        return None
    case_index = case_index.drop_duplicates("tracking_case_id").copy()
    if "loss_of_lock" in case_index.columns:
        case_index = case_index[
            ~case_index.apply(lambda row: is_loss_of_lock(row.to_dict()), axis=1)
        ]
    severity_filter = set(experiments.get("severities") or [])
    if severity_filter and "severity" in case_index.columns:
        case_index = case_index[case_index["severity"].astype(str).isin(severity_filter)]
    if case_index.empty:
        return set()

    target_c_n0 = float(experiments.get("representative_c_n0_db_hz", 30.0))
    if "input_c_n0_db_hz" in case_index.columns:
        case_index["_c_n0_distance"] = (
            pd.to_numeric(case_index["input_c_n0_db_hz"], errors="coerce") - target_c_n0
        ).abs()
    else:
        case_index["_c_n0_distance"] = 0.0
    if "severity" in case_index.columns:
        case_index["_severity_order"] = pd.Categorical(
            case_index["severity"].astype(str),
            categories=SEVERITY_ORDER,
            ordered=True,
        )
    sort_columns = [
        column
        for column in ["_severity_order", "severity", "_c_n0_distance", "tracking_case_id"]
        if column in case_index.columns
    ]
    case_index = case_index.sort_values(sort_columns)
    valid_case_ids = quality_valid_case_ids(
        params,
        paths,
        set(case_index["tracking_case_id"].astype(int).to_numpy()),
    )
    case_index = case_index[case_index["tracking_case_id"].astype(int).isin(valid_case_ids)]
    if case_index.empty:
        return set()
    selected = case_index.groupby("severity", dropna=False, sort=False).head(max_cases)
    return {int(case_id) for case_id in selected["tracking_case_id"].to_numpy()}


def iter_selected_diagnostic_cases(case_iter, selected_case_ids: set[int] | None):
    """Yield only selected diagnostic cases from a case iterator.

    Args:
        case_iter: Iterator over `(metadata, case_df)` tracking cases.
        selected_case_ids: Case ids to keep, or None to yield every case.

    Yields:
        Tracking case metadata and samples.
    """
    if selected_case_ids is None:
        yield from case_iter
        return

    yielded: set[int] = set()
    for metadata, case_df in case_iter:
        case_id = int(metadata.get("tracking_case_id", 0) or 0)
        if case_id not in selected_case_ids:
            continue
        yielded.add(case_id)
        yield metadata, case_df
        if yielded == selected_case_ids:
            return


def _heatmap_cell_count(m_grid: np.ndarray, latency_fractions: np.ndarray) -> int:
    total = 0
    for block_len in m_grid:
        future_context = int(block_len) - int(block_len) // 2 - 1
        total += len(_latency_grid(future_context, latency_fractions))
    return total


def _m_grid_from_params(near_cfg: dict) -> np.ndarray:
    return fixed_step_values(
        near_cfg.get("fft_m_values"),
        default=[512, 1024, 1536, 2048],
        dtype=int,
    )


def _latency_fractions_from_params(near_cfg: dict) -> np.ndarray:
    return fixed_step_values(
        near_cfg.get("heatmap_latency_fractions"),
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        dtype=float,
    )


def _latency_grid(future_context: int, latency_fractions: np.ndarray) -> list[tuple[float, int]]:
    pairs: list[tuple[float, int]] = []
    seen: set[int] = set()
    for fraction in latency_fractions:
        clipped = float(np.clip(fraction, 0.0, 1.0))
        latency = int(round(clipped * int(future_context)))
        if latency in seen:
            continue
        seen.add(latency)
        pairs.append((clipped, latency))
    return pairs
