"""Near-real-time practical metric generation for representative and all cases."""

from __future__ import annotations

import numpy as np
import pandas as pd

from real_time_bp.methodology.backpropagation import (
    batch_backpropagate,
    median_align_phase,
    unwrap_phase,
)
from real_time_bp.methodology.metrics import phase_rmse_rad
from real_time_bp.methodology.near_real_time import (
    select_var_order_bic,
    sliding_window_bp_with_var_forecast,
)
from real_time_bp.methodology.signal_model import (
    detrended_intensity_db,
    estimate_sample_interval_s,
    estimated_scintillation_phase_rad,
    estimated_scintillation_signal,
    true_refractive_phase_rad,
    true_scintillation_phase_rad,
    true_scintillation_signal,
)
from real_time_bp.methodology.tvd import tvd_cycle_slips_per_minute
from real_time_bp.scripts.near_real_time.common import quality_valid_case_ids, valid_eval_indices
from real_time_bp.scripts.near_real_time.examples import (
    forecast_example_rows,
    phase_example_rows,
)
from real_time_bp.utils.io import load_case_index
from real_time_bp.utils.quality import evaluate_analysis_quality
from real_time_bp.utils.representatives import SEVERITY_ORDER
from real_time_bp.utils.stage import (
    case_keys,
    is_loss_of_lock,
    is_representative_case,
)


def iter_practical_tasks(
    case_iter,
    params: dict,
    wavelength_m: float,
    tf_estimates: dict[int, float],
    representative_case_ids: set[int],
    forecast_case_id: int | dict[str, int] | None = None,
    *,
    include_phase_examples: bool = True,
):
    """Yield practical BP tasks with representative-example flags.

    Main paper:
        Results Figure 13 uses the same representative cases as the TVD
        examples, while metrics use every eligible case. Results Figure 9 uses
        a slightly noisier strong-scintillation case for the VAR forecast
        example.

    Args:
        case_iter: Iterator over `(metadata, case_df)` tracking cases.
        params: Parsed `params.yaml` dictionary.
        wavelength_m: Carrier wavelength in meters.
        tf_estimates: Fresnel-time estimates keyed by tracking case id.
        representative_case_ids: Explicit case ids selected for examples.
        forecast_case_id: Optional preselected tracking case id, or one case id
            per severity, for the VAR forecast example.
        include_phase_examples: Whether representative phase-example rows are
            requested by the caller.

    Yields:
        Tuples consumed by `compute_practical_case_task`.
    """
    experiments = params.get("experiments", {})
    phase_example_severities: set[str] = set()
    forecast_example_severities: set[str] = set()
    forecast_case_ids = forecast_case_id if isinstance(forecast_case_id, dict) else {}
    near_cfg = params.get("near_real_time", {})
    forecast_c_n0 = near_cfg.get("forecast_example_c_n0_db_hz")
    for metadata, case_df in case_iter:
        severity = str(metadata.get("severity"))
        case_id = int(metadata.get("tracking_case_id", 0) or 0)
        include_phase_example = include_phase_examples and (
            case_id in representative_case_ids
            or (
                not representative_case_ids
                and is_representative_case(metadata, experiments)
                and severity not in phase_example_severities
            )
        )
        if include_phase_example:
            phase_example_severities.add(severity)

        include_forecast_example = False
        if forecast_case_ids:
            include_forecast_example = case_id == int(forecast_case_ids.get(severity, -1))
        elif forecast_case_id is not None:
            include_forecast_example = case_id == int(forecast_case_id)
        elif severity not in forecast_example_severities:
            c_n0 = metadata.get("input_c_n0_db_hz")
            if forecast_c_n0 is None:
                include_forecast_example = is_representative_case(metadata, experiments)
            else:
                include_forecast_example = bool(
                    np.isfinite(c_n0) and float(c_n0) >= float(forecast_c_n0)
                )
        if include_forecast_example:
            forecast_example_severities.add(severity)
        yield (
            metadata,
            case_df,
            params,
            wavelength_m,
            tf_estimates,
            include_phase_example,
            include_forecast_example,
        )


def select_forecast_case_ids(
    params: dict,
    paths,
    tf_estimates: dict[int, float] | None = None,
) -> dict[str, int]:
    """Select robust weak and strong cases for the VAR forecast figure.

    Args:
        params: Parsed `params.yaml` dictionary.
        paths: Resolved dataset paths.
        tf_estimates: Optional Fresnel-time estimates keyed by tracking case id.

    Returns:
        Mapping from severity name to tracking case id closest to the
        requested forecast C/N0.
    """
    case_index = load_case_index(paths)
    if case_index.empty or "tracking_case_id" not in case_index.columns:
        return {}
    experiments = params.get("experiments", {})
    near_cfg = params.get("near_real_time", {})
    candidates = case_index.drop_duplicates("tracking_case_id").copy()
    if "loss_of_lock" in candidates.columns:
        candidates = candidates[
            ~candidates.apply(lambda row: is_loss_of_lock(row.to_dict()), axis=1)
        ]
    severity_filter = [str(value) for value in (experiments.get("severities") or [])]
    if not severity_filter:
        severity_filter = list(SEVERITY_ORDER)
    if severity_filter and "severity" in candidates:
        candidates = candidates[candidates["severity"].astype(str).isin(severity_filter)]
    if tf_estimates:
        candidates = candidates[candidates["tracking_case_id"].astype(int).isin(set(tf_estimates))]
    if candidates.empty:
        return {}
    valid_case_ids = quality_valid_case_ids(
        params,
        paths,
        set(candidates["tracking_case_id"].astype(int).to_numpy()),
    )
    candidates = candidates[candidates["tracking_case_id"].astype(int).isin(valid_case_ids)]
    if candidates.empty:
        return {}
    target_c_n0 = near_cfg.get("forecast_example_c_n0_db_hz")
    if target_c_n0 is not None and "input_c_n0_db_hz" in candidates:
        candidates["_c_n0_distance"] = (
            pd.to_numeric(candidates["input_c_n0_db_hz"], errors="coerce") - float(target_c_n0)
        ).abs()
    else:
        candidates["_c_n0_distance"] = 0.0
    selected_case_ids: dict[str, int] = {}
    for severity in severity_filter:
        if "severity" in candidates:
            severity_candidates = candidates[candidates["severity"].astype(str).eq(severity)]
        else:
            severity_candidates = candidates
        if severity_candidates.empty:
            continue
        selected = severity_candidates.sort_values(["_c_n0_distance", "tracking_case_id"]).iloc[0]
        selected_case_ids[str(severity)] = int(selected["tracking_case_id"])
    return selected_case_ids


def select_forecast_case_id(
    params: dict,
    paths,
    tf_estimates: dict[int, float] | None = None,
) -> int | None:
    """Return the strong forecast case id for backward-compatible callers.

    Args:
        params: Parsed `params.yaml` dictionary.
        paths: Resolved dataset paths.
        tf_estimates: Optional Fresnel-time estimates keyed by tracking case id.

    Returns:
        Strong case id when available, otherwise the first selected severity.
    """
    selected = select_forecast_case_ids(params, paths, tf_estimates)
    if not selected:
        return None
    if "strong" in selected:
        return selected["strong"]
    return next(iter(selected.values()))


def compute_practical_case_task(task: tuple) -> dict[str, object] | None:
    """Process one practical near-real-time case task.

    Args:
        task: Tuple produced by `iter_practical_tasks`.

    Returns:
        Result dictionary from `compute_practical_case`, or None when no
        finite Fresnel-time estimate is available.
    """
    (
        metadata,
        case_df,
        params,
        wavelength_m,
        tf_estimates,
        include_phase_example,
        include_forecast_example,
    ) = task
    case_id = int(metadata.get("tracking_case_id", 0) or 0)
    estimated_tf = tf_estimates.get(case_id, float(metadata.get("rhof_veff_ratio", np.nan)))
    if not np.isfinite(estimated_tf):
        return None
    quality = evaluate_analysis_quality(metadata, case_df, params, float(wavelength_m))
    if not quality.passed:
        return None
    return compute_practical_case(
        metadata,
        case_df,
        params,
        float(wavelength_m),
        float(estimated_tf),
        include_phase_example=bool(include_phase_example),
        include_forecast_example=bool(include_forecast_example),
    )


def compute_practical_case(
    metadata: dict,
    case_df: pd.DataFrame,
    params: dict,
    wavelength_m: float,
    estimated_tf: float,
    include_phase_example: bool = False,
    include_forecast_example: bool = False,
    residual_doppler_error_hz: float = 0.0,
) -> dict[str, object]:
    """Compute practical near-real-time metrics for one tracking case.

    Main paper:
        Methodology Eq. (13) defines the cleaned complex signal used by both
        batch and windowed BP. Methodology Section II.5 fits VAR on the same
        N_init segment used for T_F estimation. Results Figure 12 compares
        noncausal batch BP against finite-M, finite-latency near-real-time BP.

    Args:
        metadata: Metadata for the tracking case.
        case_df: Tracking samples for one case.
        params: Parsed `params.yaml` dictionary.
        wavelength_m: Carrier wavelength in meters.
        estimated_tf: Estimated Fresnel time scale in seconds.
        include_phase_example: Whether to include phase-example rows.
        include_forecast_example: Whether to include forecast-example rows.
        residual_doppler_error_hz: Optional residual Doppler sensitivity term.

    Returns:
        Dictionary containing VAR-order, practical-metric, phase-example, and
        forecast-example rows.
    """
    experiments = params.get("experiments", {})
    near_cfg = params.get("near_real_time", {})
    tvd_cfg = params.get("tvd", {})
    n_init = int(experiments.get("n_init_samples", 4096))
    block_len = int(near_cfg.get("practical_fft_m", 8192))
    latency = int(near_cfg.get("practical_latency_samples", block_len // 2 - 1))
    sample_interval_s = estimate_sample_interval_s(case_df)
    signal = estimated_scintillation_signal(case_df, wavelength_m, residual_doppler_error_hz)
    true_signal = true_scintillation_signal(case_df, wavelength_m)
    estimated_scintillation_phase = estimated_scintillation_phase_rad(
        case_df,
        wavelength_m,
        residual_doppler_error_hz,
    )
    intensity_db = detrended_intensity_db(case_df)
    time_sec = case_df["time_sec"].to_numpy(dtype=float)
    true_scintillation_phase = true_scintillation_phase_rad(case_df, wavelength_m)
    # Plot-only convention: align the displayed tracked scintillation phase to
    # truth with the same post-N_init median rule used for BP phase estimates.
    # This does not alter the complex signal used by BP.
    aligned_estimated_scintillation_phase = median_align_phase(
        estimated_scintillation_phase,
        true_scintillation_phase,
        n_init,
    )
    true_phase = true_refractive_phase_rad(case_df, wavelength_m)
    features = np.column_stack([np.real(signal[:n_init]), np.imag(signal[:n_init])])
    var_order, var_result = select_var_order_bic(
        features,
        int(near_cfg.get("var_order_min", 1)),
        int(near_cfg.get("var_order_max", 24)),
    )
    eval_indices = valid_eval_indices(len(case_df), block_len)
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
    batch_slips, batch_slips_per_min = tvd_cycle_slips_per_minute(
        batch_phase - true_phase,
        intensity_db,
        time_sec,
        float(tvd_cfg.get("fade_threshold_db", -10.0)),
        float(tvd_cfg.get("allowed_jump_weight", 1.0)),
        float(tvd_cfg.get("blocked_jump_weight", 10000.0)),
        mask=eval_mask,
    )
    sliding_slips, sliding_slips_per_min = tvd_cycle_slips_per_minute(
        sliding_phase - true_phase,
        intensity_db,
        time_sec,
        float(tvd_cfg.get("fade_threshold_db", -10.0)),
        float(tvd_cfg.get("allowed_jump_weight", 1.0)),
        float(tvd_cfg.get("blocked_jump_weight", 10000.0)),
        mask=eval_mask,
    )
    batch_mrmse_phi_r = phase_rmse_rad(true_phase[eval_mask], batch_phase[eval_mask])
    near_real_time_mrmse_phi_r = phase_rmse_rad(
        true_phase[eval_mask],
        sliding_phase[eval_mask],
    )
    tracked_mrmse_phi_s = phase_rmse_rad(
        true_phase[eval_mask],
        aligned_estimated_scintillation_phase[eval_mask],
    )
    practical_row = {
        **case_keys(metadata),
        "estimated_fresnel_time_scale_s": estimated_tf,
        "residual_doppler_error_hz": float(residual_doppler_error_hz),
        "n_init_samples": int(n_init),
        "var_order_bic": var_order,
        "block_len": block_len,
        "latency_samples": latency,
        "batch_mrmse_phi_r_rad": batch_mrmse_phi_r,
        "near_real_time_mrmse_phi_r_rad": near_real_time_mrmse_phi_r,
        "tracked_mrmse_phi_s_rad": tracked_mrmse_phi_s,
        "batch_refractive_phase_rmse_rad": batch_mrmse_phi_r,
        "near_real_time_refractive_phase_rmse_rad": near_real_time_mrmse_phi_r,
        "batch_cycle_slip_count": batch_slips,
        "batch_cs_per_min": batch_slips_per_min,
        "batch_cycle_slips_per_minute": batch_slips_per_min,
        "near_real_time_cycle_slip_count": sliding_slips,
        "near_real_time_cs_per_min": sliding_slips_per_min,
        "near_real_time_cycle_slips_per_minute": sliding_slips_per_min,
    }
    var_order_row = {
        **case_keys(metadata),
        "var_order_bic": var_order,
        "var_bic": float(var_result.bic),
        "var_aic": float(var_result.aic),
    }
    var_residual_summary_rows = _var_residual_summary_rows(
        metadata,
        var_result,
        near_cfg,
    )
    var_forecast_utility_rows = _var_forecast_utility_rows(
        metadata,
        signal,
        var_result,
        n_init,
        near_cfg,
    )
    example_rows = (
        phase_example_rows(
            metadata,
            case_df,
            params,
            signal,
            sample_interval_s,
            estimated_tf,
            var_result,
            true_scintillation_phase,
            aligned_estimated_scintillation_phase,
            true_phase,
            batch_phase,
        )
        if include_phase_example
        else []
    )
    forecast_rows = (
        forecast_example_rows(
            metadata,
            case_df,
            signal,
            true_signal,
            var_result,
            int(near_cfg.get("forecast_example_block_len", block_len)),
            [int(value) for value in near_cfg.get("forecast_example_latency_samples", [0])],
        )
        if include_forecast_example
        else []
    )
    return {
        "var_order_row": var_order_row,
        "practical_row": practical_row,
        "example_rows": example_rows,
        "forecast_rows": forecast_rows,
        "var_residual_summary_rows": var_residual_summary_rows,
        "var_forecast_utility_rows": var_forecast_utility_rows,
    }


def _var_residual_summary_rows(metadata: dict, var_result, near_cfg: dict) -> list[dict]:
    """Summarize residual whiteness and autocorrelation effect size.

    Main paper:
        Results Section III.3 treats VAR as a practical short-horizon
        predictor. These diagnostics report whether predictable residual
        structure remains, without relying on Gaussianity as a requirement.
    """
    residuals = np.asarray(var_result.resid, dtype=float)
    max_acf_lag = int(near_cfg.get("var_whiteness_test_lags", 24))
    acf_energy = _mean_component_acf_energy(residuals, max_acf_lag)
    rows = []
    for requested_lag in near_cfg.get("var_diagnostic_whiteness_lags", [24]):
        effective_lag = max(int(requested_lag), int(var_result.k_ar) + 1)
        pvalue = _whiteness_pvalue(var_result, effective_lag)
        rows.append(
            {
                **case_keys(metadata),
                "var_order_bic": int(var_result.k_ar),
                "requested_whiteness_lag": int(requested_lag),
                "effective_whiteness_lag": int(effective_lag),
                "whiteness_pvalue": pvalue,
                "minus_log10_whiteness_pvalue": _minus_log10(pvalue),
                "residual_acf_energy": acf_energy,
            }
        )
    return rows


def _var_forecast_utility_rows(
    metadata: dict,
    signal: np.ndarray,
    var_result,
    n_init: int,
    near_cfg: dict,
) -> list[dict]:
    """Compare VAR forecasts against persistence over the full dataset.

    Main paper:
        Methodology Section II.5 uses the VAR only to fill unavailable future
        samples. The forecast RMSE ratio checks whether that fill is better
        than the simplest persistence baseline at each horizon.
    """
    horizons = sorted(
        {
            int(value)
            for value in near_cfg.get("var_forecast_horizons_samples", [1, 4, 16, 64, 256])
            if int(value) > 0
        }
    )
    if not horizons:
        return []
    max_horizon = max(horizons)
    features = np.column_stack([np.real(signal), np.imag(signal)])
    k_ar = int(var_result.k_ar)
    origin_start = max(int(n_init), k_ar - 1)
    origin_stop = features.shape[0] - max_horizon - 1
    if origin_stop <= origin_start:
        return []
    max_eval = max(int(near_cfg.get("var_forecast_eval_samples", 32)), 1)
    origins = np.linspace(origin_start, origin_stop, max_eval, dtype=int)
    origins = np.unique(origins)
    squared_var_errors = {horizon: [] for horizon in horizons}
    squared_persistence_errors = {horizon: [] for horizon in horizons}
    for origin in origins:
        history = features[origin - k_ar + 1 : origin + 1]
        forecast = var_result.forecast(history, steps=max_horizon)
        current = features[origin]
        for horizon in horizons:
            target = features[origin + horizon]
            squared_var_errors[horizon].append(
                float(np.sum((target - forecast[horizon - 1]) ** 2))
            )
            squared_persistence_errors[horizon].append(float(np.sum((target - current) ** 2)))

    rows = []
    for horizon in horizons:
        var_rmse = float(np.sqrt(np.nanmean(squared_var_errors[horizon])))
        persistence_rmse = float(np.sqrt(np.nanmean(squared_persistence_errors[horizon])))
        ratio = var_rmse / persistence_rmse if persistence_rmse > 0 else np.nan
        rows.append(
            {
                **case_keys(metadata),
                "var_order_bic": int(var_result.k_ar),
                "forecast_horizon_samples": int(horizon),
                "forecast_eval_origins": int(origins.size),
                "var_forecast_rmse": var_rmse,
                "persistence_forecast_rmse": persistence_rmse,
                "var_to_persistence_rmse_ratio": ratio,
            }
        )
    return rows


def _mean_component_acf_energy(residuals: np.ndarray, max_lag: int) -> float:
    energies = []
    for component_idx in range(residuals.shape[1]):
        values = residuals[:, component_idx]
        values = values[np.isfinite(values)]
        values = values - np.nanmean(values)
        denom = float(np.dot(values, values))
        if denom <= 0:
            continue
        energy = 0.0
        for lag in range(1, min(int(max_lag), values.size - 1) + 1):
            rho = float(np.dot(values[:-lag], values[lag:]) / denom)
            energy += rho * rho
        energies.append(energy)
    return float(np.nanmean(energies)) if energies else np.nan


def _whiteness_pvalue(var_result, lag: int) -> float:
    try:
        return float(var_result.test_whiteness(nlags=int(lag), signif=0.05).pvalue)
    except Exception:
        return float("nan")


def _minus_log10(value: float) -> float:
    if not np.isfinite(value):
        return np.nan
    return float(-np.log10(max(float(value), np.finfo(float).tiny)))
