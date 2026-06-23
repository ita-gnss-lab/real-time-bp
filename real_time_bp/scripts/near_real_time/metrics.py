"""Near-real-time practical metric generation for representative and all cases."""

from __future__ import annotations

import numpy as np
import pandas as pd

from real_time_bp.methodology.backpropagation import (
    batch_backpropagate,
    median_align_phase,
    unwrap_phase,
)
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
from real_time_bp.scripts.near_real_time.common import valid_eval_indices
from real_time_bp.scripts.near_real_time.examples import (
    forecast_example_rows,
    phase_example_rows,
)
from real_time_bp.utils.stage import case_keys, is_representative_case


def iter_practical_tasks(
    case_iter,
    params: dict,
    wavelength_m: float,
    tf_estimates: dict[int, float],
    representative_case_ids: set[int],
):
    """Yield practical BP tasks with representative-example flags.

    Args:
        case_iter: Iterator over `(metadata, case_df)` tracking cases.
        params: Parsed `params.yaml` dictionary.
        wavelength_m: Carrier wavelength in meters.
        tf_estimates: Fresnel-time estimates keyed by tracking case id.
        representative_case_ids: Explicit case ids selected for examples.

    Yields:
        Tuples consumed by `compute_practical_case_task`.
    """
    experiments = params.get("experiments", {})
    example_severities: set[str] = set()
    for metadata, case_df in case_iter:
        severity = str(metadata.get("severity"))
        case_id = int(metadata.get("tracking_case_id", 0) or 0)
        # Paper: Figures 9 and 13 use representative forecast/phase plots by
        # severity; metrics still use every eligible case.
        include_examples = case_id in representative_case_ids or (
            not representative_case_ids
            and is_representative_case(metadata, experiments)
            and severity not in example_severities
        )
        if include_examples:
            example_severities.add(severity)
        yield metadata, case_df, params, wavelength_m, tf_estimates, include_examples


def compute_practical_case_task(task: tuple) -> dict[str, object] | None:
    """Process one practical near-real-time case task.

    Args:
        task: Tuple produced by `iter_practical_tasks`.

    Returns:
        Result dictionary from `compute_practical_case`, or None when no
        finite Fresnel-time estimate is available.
    """
    metadata, case_df, params, wavelength_m, tf_estimates, include_examples = task
    case_id = int(metadata.get("tracking_case_id", 0) or 0)
    estimated_tf = tf_estimates.get(case_id, float(metadata.get("rhof_veff_ratio", np.nan)))
    if not np.isfinite(estimated_tf):
        return None
    return compute_practical_case(
        metadata,
        case_df,
        params,
        float(wavelength_m),
        float(estimated_tf),
        include_examples=bool(include_examples),
    )


def compute_practical_case(
    metadata: dict,
    case_df: pd.DataFrame,
    params: dict,
    wavelength_m: float,
    estimated_tf: float,
    include_examples: bool = False,
    residual_doppler_error_hz: float = 0.0,
) -> dict[str, object]:
    """Compute practical near-real-time metrics for one tracking case.

    Args:
        metadata: Metadata for the tracking case.
        case_df: Tracking samples for one case.
        params: Parsed `params.yaml` dictionary.
        wavelength_m: Carrier wavelength in meters.
        estimated_tf: Estimated Fresnel time scale in seconds.
        include_examples: Whether to include phase and forecast example rows.
        residual_doppler_error_hz: Optional residual Doppler sensitivity term.

    Returns:
        Dictionary containing VAR-order, practical-metric, phase-example, and
        forecast-example rows.
    """
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
    eval_indices = valid_eval_indices(len(case_df), block_len)
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
        **case_keys(metadata),
        "estimated_fresnel_time_scale_s": estimated_tf,
        "residual_doppler_error_hz": float(residual_doppler_error_hz),
        "n_init_samples": int(n_init),
        "var_order_bic": var_order,
        "block_len": block_len,
        "latency_samples": latency,
        "batch_refractive_phase_rmse_rad": phase_rmse_rad(
            true_phase[eval_mask], batch_phase[eval_mask]
        ),
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
        **case_keys(metadata),
        "var_order_bic": var_order,
        "var_bic": float(var_result.bic),
        "var_aic": float(var_result.aic),
    }
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
            estimated_scintillation_phase,
            true_phase,
            batch_phase,
        )
        if include_examples
        else []
    )
    forecast_rows = (
        forecast_example_rows(metadata, case_df, signal, true_signal, var_result, block_len)
        if include_examples
        else []
    )
    return {
        "var_order_row": var_order_row,
        "practical_row": practical_row,
        "example_rows": example_rows,
        "forecast_rows": forecast_rows,
    }
