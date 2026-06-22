from __future__ import annotations

from loguru import logger
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import pacf

from real_time_bp.config import CSV_RESULTS_DIR
from real_time_bp.methodology.backpropagation import (
    batch_backpropagate,
    median_align_phase,
    unwrap_phase,
)
from real_time_bp.methodology.filter_bank import AMPLITUDE_SCORE_NAME
from real_time_bp.methodology.metrics import cycle_slips_per_minute, phase_rmse_rad
from real_time_bp.methodology.near_real_time import (
    select_var_order_bic,
    sliding_window_bp_with_var_forecast,
)
from real_time_bp.methodology.signal_model import (
    estimate_sample_interval_s,
    estimated_scintillation_phase_rad,
    estimated_scintillation_signal,
    true_refractive_phase_rad,
    true_scintillation_phase_rad,
    true_scintillation_signal,
)
from real_time_bp.representatives import (
    SEVERITY_ORDER,
    load_representative_case_ids,
)
from real_time_bp.stage_utils import (
    csv_chunksize_from_params,
    csv_stage_dir,
    l1_wavelength_from_params,
    optional_int,
    parallel_workers_from_params,
    residual_doppler_grid_hz,
    write_csv,
)
from real_time_bp.utils.io import (
    count_cases,
    find_dataset_paths,
    iter_case_dataframes,
    load_case_index,
)
from real_time_bp.utils.parallel import ordered_parallel_map
from real_time_bp.utils.params import load_params
from real_time_bp.utils.progress import configure_logging, progress_iter

NEAR_REAL_TIME_TRACKING_COLUMNS = {
    "carrier_phase_est_m",
    "los_phase_m",
    "scint_phase_m",
    "time_sec",
    "tracked_intensity",
    "true_intensity",
    "true_refractive_phase_m",
}


def main() -> None:
    configure_logging()
    logger.info("Starting near-real-time results stage")
    params = load_params()
    experiments = params.get("experiments", {})
    out_dir = csv_stage_dir("04_near_real_time")
    paths = find_dataset_paths(params)
    logger.info("Using dataset root {}", paths.root)
    wavelength_m = l1_wavelength_from_params(params)
    max_cases = optional_int(experiments.get("max_cases_per_condition"))
    chunksize = csv_chunksize_from_params(params)
    tf_estimates = _load_amplitude_tf_estimates()
    representative_case_ids = set(load_representative_case_ids().values())
    workers = parallel_workers_from_params(params)
    logger.info("Using {} parallel worker(s)", workers)
    # Paper: Results Section III.3. This stage consumes
    # the amplitude-standard-deviation T_F estimate and evaluates practical
    # near-real-time BP plus the M/L latency heatmap.
    total_cases = count_cases(
        paths,
        severities=experiments.get("severities"),
        max_cases_per_condition=max_cases,
    )

    var_order_rows: list[dict] = []
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
    task_iter = _iter_practical_tasks(
        case_iter,
        params,
        wavelength_m,
        tf_estimates,
        representative_case_ids,
    )
    result_iter = ordered_parallel_map(
        _compute_practical_case_task,
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
        practical_rows.append(result["practical_row"])
        example_rows.extend(result["example_rows"])
        forecast_rows.extend(result["forecast_rows"])
    residual_metrics = _compute_residual_doppler_metrics(
        params,
        paths,
        wavelength_m,
        tf_estimates,
        representative_case_ids,
    )
    residual_diagnostics = _compute_var_residual_diagnostics(
        params,
        paths,
        wavelength_m,
        representative_case_ids,
    )

    heatmap_df = _compute_heatmap(params, paths, wavelength_m, tf_estimates)
    write_csv(
        _sort_by_severity(pd.DataFrame(var_order_rows), ["tracking_case_id"]),
        out_dir / "near_real_time_var_orders.csv",
    )
    write_csv(
        _sort_by_severity(pd.DataFrame(practical_rows), ["tracking_case_id"]),
        out_dir / "near_real_time_practical_metrics.csv",
    )
    write_csv(
        _sort_by_severity(pd.DataFrame(example_rows), ["tracking_case_id", "sample_index"]),
        out_dir / "near_real_time_phase_example.csv",
    )
    write_csv(
        _sort_by_severity(
            pd.DataFrame(forecast_rows),
            ["tracking_case_id", "latency_samples", "sample_index"],
        ),
        out_dir / "near_real_time_forecast_example.csv",
    )
    write_csv(
        _sort_by_severity(heatmap_df, ["tracking_case_id", "block_len", "latency_samples"]),
        out_dir / "near_real_time_latency_heatmap.csv",
    )
    write_csv(
        _sort_by_severity(
            residual_metrics,
            ["tracking_case_id", "residual_doppler_error_hz"],
        ),
        out_dir / "near_real_time_residual_doppler_metrics.csv",
    )
    write_csv(
        _sort_by_severity(
            residual_diagnostics,
            ["tracking_case_id", "component", "diagnostic", "sample_index", "lag"],
        ),
        out_dir / "near_real_time_var_residual_diagnostics.csv",
    )
    logger.info("Finished near-real-time results stage")


def _load_amplitude_tf_estimates() -> dict[int, float]:
    # Paper: Results Section III.3 explicitly uses the
    # amplitude standard-deviation baseline from Section II.4.
    path = CSV_RESULTS_DIR / "03_filter_bank" / "filter_bank_estimates.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "score_name" in df.columns:
        df = df[df["score_name"].eq(AMPLITUDE_SCORE_NAME)]
    return {
        int(row.tracking_case_id): float(row.estimated_fresnel_time_scale_s)
        for row in df.itertuples()
    }


def _iter_practical_tasks(
    case_iter,
    params: dict,
    wavelength_m: float,
    tf_estimates: dict[int, float],
    representative_case_ids: set[int],
):
    experiments = params.get("experiments", {})
    example_severities: set[str] = set()
    for metadata, case_df in case_iter:
        severity = str(metadata.get("severity"))
        case_id = int(metadata.get("tracking_case_id", 0) or 0)
        # Paper: Figures 9 and 13 use representative forecast/phase plots by
        # severity; metrics still use every eligible case.
        include_examples = (
            case_id in representative_case_ids
            or (
                not representative_case_ids
                and _is_representative_case(metadata, experiments)
                and severity not in example_severities
            )
        )
        if include_examples:
            example_severities.add(severity)
        yield metadata, case_df, params, wavelength_m, tf_estimates, include_examples


def _compute_practical_case_task(task: tuple) -> dict[str, object] | None:
    metadata, case_df, params, wavelength_m, tf_estimates, include_examples = task
    case_id = int(metadata.get("tracking_case_id", 0) or 0)
    estimated_tf = tf_estimates.get(case_id, float(metadata.get("rhof_veff_ratio", np.nan)))
    if not np.isfinite(estimated_tf):
        return None
    return _compute_practical_case(
        metadata,
        case_df,
        params,
        float(wavelength_m),
        float(estimated_tf),
        include_examples=bool(include_examples),
    )


def _compute_practical_case(
    metadata: dict,
    case_df: pd.DataFrame,
    params: dict,
    wavelength_m: float,
    estimated_tf: float,
    include_examples: bool = False,
    residual_doppler_error_hz: float = 0.0,
) -> dict[str, object]:
    experiments = params.get("experiments", {})
    near_cfg = params.get("near_real_time", {})
    n_init = int(experiments.get("n_init_samples", 4096))
    block_len = int(near_cfg.get("practical_fft_m", 8192))
    latency = int(near_cfg.get("practical_latency_samples", block_len // 2 - 1))
    sample_interval_s = estimate_sample_interval_s(case_df)
    # Paper: Methodology Eq. (13); both
    # batch and windowed BP operate on the same cleaned complex signal.
    signal = estimated_scintillation_signal(case_df, wavelength_m, residual_doppler_error_hz)
    true_signal = true_scintillation_signal(case_df, wavelength_m)
    estimated_scintillation_phase = estimated_scintillation_phase_rad(
        case_df,
        wavelength_m,
        residual_doppler_error_hz,
    )
    true_scintillation_phase = true_scintillation_phase_rad(case_df, wavelength_m)
    true_phase = true_refractive_phase_rad(case_df, wavelength_m)
    # Paper: Methodology Section II.5; fit VAR on the same N_init
    # segment used for T_F estimation.
    features = np.column_stack([np.real(signal[:n_init]), np.imag(signal[:n_init])])
    var_order, var_result = select_var_order_bic(
        features,
        int(near_cfg.get("var_order_min", 1)),
        int(near_cfg.get("var_order_max", 24)),
    )
    eval_indices = _valid_eval_indices(len(case_df), block_len)
    # Paper: Figure 12 practical comparison. Batch BP is the noncausal reference;
    # near-real-time BP uses a finite M window and latency L.
    batch = batch_backpropagate(signal, sample_interval_s, estimated_tf)
    batch_phase = median_align_phase(unwrap_phase(batch), true_phase, n_init)
    sliding = sliding_window_bp_with_var_forecast(
        signal,
        sample_interval_s,
        estimated_tf,
        block_len,
        latency,
        eval_indices,
        var_result,
    )
    sliding_phase = median_align_phase(unwrap_phase(sliding), true_phase, n_init)
    eval_mask = np.zeros(len(case_df), dtype=bool)
    eval_mask[eval_indices] = True
    eval_mask[:n_init] = False
    batch_slips, batch_slips_per_min = cycle_slips_per_minute(
        true_phase[eval_mask],
        batch_phase[eval_mask],
        sample_interval_s,
    )
    sliding_slips, sliding_slips_per_min = cycle_slips_per_minute(
        true_phase[eval_mask],
        sliding_phase[eval_mask],
        sample_interval_s,
    )
    practical_row = {
        # Paper: Figure 12; these metrics feed
        # batch-vs-near-real-time MRMSE and CS/min comparisons.
        **_case_keys(metadata),
        "estimated_fresnel_time_scale_s": estimated_tf,
        "residual_doppler_error_hz": float(residual_doppler_error_hz),
        "var_order_bic": var_order,
        "block_len": block_len,
        "latency_samples": latency,
        "batch_refractive_phase_rmse_rad": phase_rmse_rad(true_phase[eval_mask], batch_phase[eval_mask]),
        "near_real_time_refractive_phase_rmse_rad": phase_rmse_rad(
            true_phase[eval_mask],
            sliding_phase[eval_mask],
        ),
        "batch_cycle_slip_count": batch_slips,
        "batch_cycle_slips_per_minute": batch_slips_per_min,
        "near_real_time_cycle_slip_count": sliding_slips,
        "near_real_time_cycle_slips_per_minute": sliding_slips_per_min,
    }
    var_order_row = {
        **_case_keys(metadata),
        "var_order_bic": var_order,
        "var_bic": float(var_result.bic),
        "var_aic": float(var_result.aic),
    }
    example_rows = (
        _phase_example_rows(
            metadata,
            case_df,
            params,
            signal,
            sample_interval_s,
            estimated_tf,
            var_result,
            true_scintillation_phase,
            estimated_scintillation_phase,
            true_phase,
            batch_phase,
        )
        if include_examples
        else []
    )
    forecast_rows = (
        _forecast_example_rows(metadata, case_df, signal, true_signal, var_result, block_len)
        if include_examples
        else []
    )
    return {
        "var_order_row": var_order_row,
        "practical_row": practical_row,
        "example_rows": example_rows,
        "forecast_rows": forecast_rows,
    }


def _compute_heatmap(params: dict, paths, wavelength_m: float, tf_estimates: dict[int, float]) -> pd.DataFrame:
    experiments = params.get("experiments", {})
    near_cfg = params.get("near_real_time", {})
    rows: list[dict] = []
    selected_case_ids = _select_heatmap_case_ids(params, paths)
    m_grid = _m_grid_from_params(near_cfg)
    # Paper: Figure 11; for each M, valid
    # latency is 0 <= L <= M-M_c-1, producing the triangular heatmap domain.
    latency_fractions = _latency_fractions_from_params(near_cfg)
    heatmap_eval_samples = max(int(near_cfg.get("heatmap_eval_samples", 32)), 1)
    chunksize = csv_chunksize_from_params(params)
    workers = parallel_workers_from_params(params)
    total_cases = (
        len(selected_case_ids)
        if selected_case_ids is not None
        else count_cases(paths, severities=experiments.get("severities"))
    )
    heatmap_cells_per_case = _heatmap_cell_count(m_grid, latency_fractions)
    logger.info(
        "Heatmap grid: {} selected case(s), {} FFT lengths, {} M/L cells per case, {} eval sample(s) per cell",
        total_cases,
        len(m_grid),
        heatmap_cells_per_case,
        heatmap_eval_samples,
    )
    case_iter = iter_case_dataframes(
        paths,
        severities=experiments.get("severities"),
        max_cases_per_condition=None,
        chunksize=chunksize,
        columns=NEAR_REAL_TIME_TRACKING_COLUMNS,
    )
    selected_case_iter = _iter_selected_heatmap_cases(case_iter, selected_case_ids)
    task_iter = (
        (
            metadata,
            case_df,
            params,
            wavelength_m,
            tf_estimates,
            m_grid,
            latency_fractions,
            heatmap_eval_samples,
        )
        for metadata, case_df in selected_case_iter
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


def _compute_heatmap_case_task(task: tuple) -> list[dict]:
    (
        metadata,
        case_df,
        params,
        wavelength_m,
        tf_estimates,
        m_grid,
        latency_fractions,
        heatmap_eval_samples,
    ) = task
    experiments = params.get("experiments", {})
    near_cfg = params.get("near_real_time", {})
    case_id = int(metadata.get("tracking_case_id", 0) or 0)
    estimated_tf = tf_estimates.get(case_id, float(metadata.get("rhof_veff_ratio", np.nan)))
    if not np.isfinite(estimated_tf):
        return []
    sample_interval_s = estimate_sample_interval_s(case_df)
    signal = estimated_scintillation_signal(case_df, float(wavelength_m))
    true_phase = true_refractive_phase_rad(case_df, float(wavelength_m))
    n_init = min(int(experiments.get("n_init_samples", 4096)), len(case_df))
    # Paper: Methodology Eq. (28); one VAR fit per case is reused over
    # all M/L heatmap cells.
    features = np.column_stack([np.real(signal[:n_init]), np.imag(signal[:n_init])])
    _, var_result = select_var_order_bic(
        features,
        int(near_cfg.get("var_order_min", 1)),
        int(near_cfg.get("var_order_max", 24)),
    )
    rows = []
    for block_len in m_grid:
        future_context = int(block_len) - int(block_len) // 2 - 1
        eval_indices = _valid_eval_indices(len(case_df), int(block_len))
        eval_indices = _thin_eval_indices(eval_indices, max_points=int(heatmap_eval_samples))
        for latency_fraction, latency in _latency_grid(future_context, latency_fractions):
            # Paper: Methodology Eq. (26); this evaluates a
            # single heatmap cell for the chosen FFT length M and latency L.
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
                    **_case_keys(metadata),
                    "block_len": int(block_len),
                    "latency_samples": int(latency),
                    "latency_fraction": float(latency_fraction),
                    "refractive_phase_rmse_rad": phase_rmse_rad(true_phase[mask], phase[mask]),
                    "cycle_slip_count": slip_count,
                    "cycle_slips_per_minute": slips_per_min,
                }
            )
    return rows


def _select_heatmap_case_ids(params: dict, paths) -> set[int] | None:
    near_cfg = params.get("near_real_time", {})
    max_cases = optional_int(near_cfg.get("heatmap_max_cases_per_severity"))
    if max_cases is None:
        return None

    experiments = params.get("experiments", {})
    case_index = load_case_index(paths)
    if case_index.empty or "tracking_case_id" not in case_index.columns:
        return None
    case_index = case_index.drop_duplicates("tracking_case_id").copy()
    severity_filter = set(experiments.get("severities") or [])
    if severity_filter and "severity" in case_index.columns:
        case_index = case_index[case_index["severity"].astype(str).isin(severity_filter)]
    if case_index.empty:
        return set()

    # Paper: Figure 11 is a latency/M diagnostic, not a carrier-to-noise sweep.
    # Use representative C/N0 cases so heatmap_max_cases_per_severity means
    # one (or N) case(s) per severity rather than one per severity and C/N0 bin.
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
    selected = case_index.groupby("severity", dropna=False, sort=False).head(max_cases)
    return {int(case_id) for case_id in selected["tracking_case_id"].to_numpy()}


def _iter_selected_heatmap_cases(case_iter, selected_case_ids: set[int] | None):
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
    values = near_cfg.get("fft_m_values")
    if values:
        return np.asarray(sorted({int(value) for value in values}), dtype=int)
    return np.linspace(
        int(near_cfg.get("fft_m_min", 512)),
        int(near_cfg.get("fft_m_max", 8192)),
        int(near_cfg.get("fft_m_count", 16)),
        dtype=int,
    )


def _latency_fractions_from_params(near_cfg: dict) -> np.ndarray:
    values = near_cfg.get("heatmap_latency_fractions")
    if values:
        return np.asarray(sorted({float(value) for value in values}), dtype=float)
    return np.linspace(0.0, 1.0, 8)


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


def _sort_by_severity(df: pd.DataFrame, extra_columns: list[str]) -> pd.DataFrame:
    if df.empty or "severity" not in df.columns:
        return df
    ordered = pd.Categorical(df["severity"].astype(str), categories=SEVERITY_ORDER, ordered=True)
    sort_columns = [
        column
        for column in ["_severity_order", *extra_columns]
        if column == "_severity_order" or column in df.columns
    ]
    return (
        df.assign(_severity_order=ordered)
        .sort_values(sort_columns)
        .drop(columns="_severity_order")
    )


def _valid_eval_indices(n_samples: int, block_len: int) -> np.ndarray:
    # Paper: Methodology Eq. (26); edge samples without the full
    # past/future block support are excluded from near-real-time evaluation.
    half = int(block_len) // 2
    future_context = int(block_len) - half - 1
    return np.arange(half, max(half, int(n_samples) - future_context), dtype=int)


def _thin_eval_indices(indices: np.ndarray, max_points: int) -> np.ndarray:
    # Implementation note for Results heatmaps: thinning keeps the M/L sweep
    # tractable while preserving the triangular grid definition.
    if indices.size <= max_points:
        return indices
    positions = np.linspace(0, indices.size - 1, int(max_points), dtype=int)
    return indices[positions]


def _phase_example_rows(
    metadata: dict,
    case_df,
    params: dict,
    signal: np.ndarray,
    sample_interval_s: float,
    estimated_tf: float,
    var_result,
    true_scintillation_phase: np.ndarray,
    estimated_scintillation_phase: np.ndarray,
    true_phase: np.ndarray,
    batch_phase: np.ndarray,
) -> list[dict]:
    near_cfg = params.get("near_real_time", {})
    configs = _example_ml_configs(near_cfg)
    stride = max(len(case_df) // 2500, 1)
    plot_indices = np.arange(0, len(case_df), stride, dtype=int)
    config_phases = []
    for block_len, latency in configs:
        valid_indices = _valid_eval_indices(len(case_df), int(block_len))
        eval_indices = np.intersect1d(valid_indices, plot_indices, assume_unique=True)
        sliding = sliding_window_bp_with_var_forecast(
            signal,
            sample_interval_s,
            float(estimated_tf),
            int(block_len),
            int(latency),
            eval_indices,
            var_result,
        )
        phase = median_align_phase(unwrap_phase(sliding), true_phase, int(params.get("experiments", {}).get("n_init_samples", 4096)))
        config_phases.append((int(block_len), int(latency), phase))

    rows = []
    for block_len, latency, phase in config_phases:
        configuration = f"M={block_len}, L={latency}"
        for idx in plot_indices:
            rows.append(
                {
                    **_case_keys(metadata),
                    "configuration": configuration,
                    "block_len": int(block_len),
                    "latency_samples": int(latency),
                    "sample_index": int(idx),
                    "time_sec": float(case_df["time_sec"].iloc[idx]),
                    "true_scintillation_phase_rad": float(true_scintillation_phase[idx]),
                    "estimated_scintillation_phase_rad": float(estimated_scintillation_phase[idx]),
                    "true_refractive_phase_rad": float(true_phase[idx]),
                    "batch_refractive_phase_rad": float(batch_phase[idx]),
                    "near_real_time_refractive_phase_rad": float(phase[idx]),
                    "true_scintillation_minus_refractive_rad": float(
                        true_scintillation_phase[idx] - true_phase[idx]
                    ),
                    "estimated_scintillation_minus_refractive_rad": float(
                        estimated_scintillation_phase[idx] - true_phase[idx]
                    ),
                    "batch_refractive_phase_error_rad": float(batch_phase[idx] - true_phase[idx]),
                    "near_real_time_refractive_phase_error_rad": float(phase[idx] - true_phase[idx]),
                }
            )
    return rows


def _forecast_example_rows(metadata, case_df, signal, true_signal, var_result, block_len) -> list[dict]:
    # Paper: Figure 9; export true vs VAR-forecasted real/imaginary samples for
    # several latency values. The practical L may equal the full future context,
    # where no future samples remain to forecast, so this figure uses a sweep.
    center = len(case_df) // 2
    half = int(block_len) // 2
    future_context = int(block_len) - half - 1
    k_ar = int(var_result.k_ar)
    if future_context <= 0:
        return []
    features = np.column_stack([np.real(signal), np.imag(signal)])
    rows = []
    for latency in _forecast_example_latencies(future_context):
        observed_stop = center + int(latency)
        forecast_steps = future_context - int(latency)
        if observed_stop + 1 - k_ar < 0 or forecast_steps <= 0:
            continue
        history = features[observed_stop + 1 - k_ar : observed_stop + 1]
        forecast = var_result.forecast(history, steps=forecast_steps)
        future_axis = np.arange(observed_stop + 1, observed_stop + 1 + forecast_steps)
        true_tail = true_signal[future_axis]
        estimated_tail = signal[future_axis]
        forecast_start_sample = int(observed_stop + 1)
        forecast_start_time = float(case_df["time_sec"].iloc[forecast_start_sample])
        for sample, true_value, estimated_value, forecast_value in zip(
            future_axis,
            true_tail,
            estimated_tail,
            forecast[:, 0] + 1j * forecast[:, 1],
        ):
            rows.append(
                {
                    **_case_keys(metadata),
                    "target_sample_index": int(center),
                    "sample_index": int(sample),
                    "time_sec": float(case_df["time_sec"].iloc[int(sample)]),
                    "latency_samples": int(latency),
                    "forecast_start_sample_index": forecast_start_sample,
                    "forecast_start_time_sec": forecast_start_time,
                    "true_real": float(np.real(true_value)),
                    "true_imag": float(np.imag(true_value)),
                    "estimated_real": float(np.real(estimated_value)),
                    "estimated_imag": float(np.imag(estimated_value)),
                    "forecast_real": float(np.real(forecast_value)),
                    "forecast_imag": float(np.imag(forecast_value)),
                }
            )
    return rows


def _forecast_example_latencies(future_context: int) -> np.ndarray:
    max_forecast_latency = max(int(future_context) - 1, 0)
    return np.unique(
        np.linspace(
            0,
            max_forecast_latency,
            min(4, max_forecast_latency + 1),
            dtype=int,
        )
    )


def _example_ml_configs(near_cfg: dict) -> list[tuple[int, int]]:
    configs = []
    for item in near_cfg.get("example_ml_configs", []) or []:
        block_len = int(item.get("block_len"))
        latency = int(item.get("latency_samples"))
        future_context = block_len - block_len // 2 - 1
        configs.append((block_len, int(np.clip(latency, 0, future_context))))
    if configs:
        return configs
    block_len = int(near_cfg.get("practical_fft_m", 8192))
    latency = int(near_cfg.get("practical_latency_samples", block_len // 2 - 1))
    return [(block_len, latency)]


def _compute_residual_doppler_metrics(
    params: dict,
    paths,
    wavelength_m: float,
    tf_estimates: dict[int, float],
    representative_case_ids: set[int],
) -> pd.DataFrame:
    rows: list[dict] = []
    selected_case_ids = representative_case_ids or (_select_heatmap_case_ids(params, paths) or set())
    if not selected_case_ids:
        return pd.DataFrame()
    chunksize = csv_chunksize_from_params(params)
    case_iter = iter_case_dataframes(
        paths,
        severities=params.get("experiments", {}).get("severities"),
        max_cases_per_condition=None,
        chunksize=chunksize,
        columns=NEAR_REAL_TIME_TRACKING_COLUMNS,
    )
    cases = list(_iter_selected_heatmap_cases(case_iter, selected_case_ids))
    residual_grid = residual_doppler_grid_hz(params)
    for metadata, case_df in progress_iter(
        cases,
        desc="near-real-time residual cases",
        total=len(cases),
        unit="case",
    ):
        case_id = int(metadata.get("tracking_case_id", 0) or 0)
        estimated_tf = tf_estimates.get(case_id, float(metadata.get("rhof_veff_ratio", np.nan)))
        if not np.isfinite(estimated_tf):
            continue
        for residual_row in residual_grid.itertuples():
            result = _compute_practical_case(
                metadata,
                case_df,
                params,
                float(wavelength_m),
                float(estimated_tf),
                include_examples=False,
                residual_doppler_error_hz=float(residual_row.residual_doppler_error_hz),
            )
            rows.append(result["practical_row"])
    return pd.DataFrame(rows)


def _compute_var_residual_diagnostics(
    params: dict,
    paths,
    wavelength_m: float,
    representative_case_ids: set[int],
) -> pd.DataFrame:
    selected_case_ids = representative_case_ids or (_select_heatmap_case_ids(params, paths) or set())
    if not selected_case_ids:
        return pd.DataFrame()
    chunksize = csv_chunksize_from_params(params)
    case_iter = iter_case_dataframes(
        paths,
        severities=params.get("experiments", {}).get("severities"),
        max_cases_per_condition=None,
        chunksize=chunksize,
        columns=NEAR_REAL_TIME_TRACKING_COLUMNS,
    )
    rows: list[dict] = []
    experiments = params.get("experiments", {})
    near_cfg = params.get("near_real_time", {})
    for metadata, case_df in _iter_selected_heatmap_cases(case_iter, selected_case_ids):
        signal = estimated_scintillation_signal(case_df, float(wavelength_m))
        n_init = min(int(experiments.get("n_init_samples", 4096)), len(case_df))
        features = np.column_stack([np.real(signal[:n_init]), np.imag(signal[:n_init])])
        _, var_result = select_var_order_bic(
            features,
            int(near_cfg.get("var_order_min", 1)),
            int(near_cfg.get("var_order_max", 24)),
        )
        residuals = np.asarray(var_result.resid, dtype=float)
        start_sample = int(var_result.k_ar)
        for component_idx, component in enumerate(["real", "imag"]):
            values = residuals[:, component_idx]
            scale = np.nanstd(values)
            standardized = (values - np.nanmean(values)) / scale if scale > 0 else values - np.nanmean(values)
            for offset, value in enumerate(standardized):
                sample_index = start_sample + offset
                rows.append(
                    {
                        **_case_keys(metadata),
                        "component": component,
                        "diagnostic": "residual",
                        "sample_index": int(sample_index),
                        "time_sec": float(case_df["time_sec"].iloc[sample_index]),
                        "value": float(value),
                        "lag": np.nan,
                        "pacf": np.nan,
                    }
                )
            max_lag = min(40, max(1, values.size // 4))
            pacf_values = pacf(standardized, nlags=max_lag, method="ywm")
            for lag, value in enumerate(pacf_values):
                rows.append(
                    {
                        **_case_keys(metadata),
                        "component": component,
                        "diagnostic": "pacf",
                        "sample_index": np.nan,
                        "time_sec": np.nan,
                        "value": np.nan,
                        "lag": int(lag),
                        "pacf": float(value),
                    }
                )
    return pd.DataFrame(rows)


def _is_representative_case(metadata: dict, experiments: dict) -> bool:
    c_n0 = metadata.get("input_c_n0_db_hz")
    target_c_n0 = experiments.get("representative_c_n0_db_hz", 30.0)
    return bool(np.isfinite(c_n0) and abs(float(c_n0) - float(target_c_n0)) < 1e-9)


def _case_keys(metadata: dict) -> dict:
    return {
        "tracking_case_id": metadata.get("tracking_case_id"),
        "example_idx": metadata.get("example_idx"),
        "c_n0_idx": metadata.get("c_n0_idx"),
        "severity": metadata.get("severity"),
        "frequency": metadata.get("frequency"),
        "svid": metadata.get("svid"),
        "input_c_n0_db_hz": metadata.get("input_c_n0_db_hz"),
        "init_doppler_error_hz": metadata.get("init_doppler_error_hz"),
    }


if __name__ == "__main__":
    main()
