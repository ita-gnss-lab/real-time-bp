"""Representative near-real-time phase and VAR forecast output rows."""

from __future__ import annotations

import numpy as np

from real_time_bp.methodology.backpropagation import median_align_phase, unwrap_phase
from real_time_bp.methodology.near_real_time import sliding_window_bp_with_var_forecast
from real_time_bp.methodology.signal_model import detrended_intensity_db
from real_time_bp.methodology.tvd import tvd_integer_bias
from real_time_bp.scripts.near_real_time.common import valid_eval_indices
from real_time_bp.utils.output_rows import sample_indices, sampled_time_rows
from real_time_bp.utils.stage import case_keys


def phase_example_rows(
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
    """Build rows for the near-real-time phase example figure.

    Args:
        metadata: Metadata for the selected tracking case.
        case_df: Tracking samples for one case.
        params: Parsed `params.yaml` dictionary.
        signal: Complex tracked scintillation signal.
        sample_interval_s: Sample spacing in seconds.
        estimated_tf: Estimated Fresnel time scale in seconds.
        var_result: Fitted VAR result used for future-context forecasts.
        true_scintillation_phase: Truth propagated scintillation phase.
        estimated_scintillation_phase: LOS-corrected tracked phase estimate.
        true_phase: Truth refractive screen phase.
        batch_phase: Batch backpropagated refractive phase estimate.

    Returns:
        List of CSV rows containing sampled phase traces and TVD cycle-bias
        fits for each configured M/L pair.
    """
    near_cfg = params.get("near_real_time", {})
    tvd_cfg = params.get("tvd", {})
    configs = _example_ml_configs(near_cfg)
    plot_indices = np.fromiter(sample_indices(len(case_df)), dtype=int)
    intensity_db = detrended_intensity_db(case_df)
    batch_phase_error = batch_phase - true_phase
    batch_integer_cycle_bias = tvd_integer_bias(
        batch_phase_error,
        intensity_db,
        float(tvd_cfg.get("fade_threshold_db", -10.0)),
        float(tvd_cfg.get("allowed_jump_weight", 1.0)),
        float(tvd_cfg.get("blocked_jump_weight", 10000.0)),
    )
    config_phases = []
    for block_len, latency in configs:
        valid_indices = valid_eval_indices(len(case_df), int(block_len))
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
        phase = median_align_phase(
            unwrap_phase(sliding),
            true_phase,
            int(params.get("experiments", {}).get("n_init_samples", 4096)),
        )
        phase_error = phase - true_phase
        integer_cycle_bias = _finite_tvd_integer_bias(
            phase_error,
            intensity_db,
            tvd_cfg,
        )
        config_phases.append((int(block_len), int(latency), phase, integer_cycle_bias))

    rows = []
    for block_len, latency, phase, integer_cycle_bias in config_phases:
        configuration = f"M={block_len}, L={latency}"
        rows.extend(
            sampled_time_rows(
                case_df,
                {
                    **case_keys(metadata),
                    "configuration": configuration,
                    "block_len": int(block_len),
                    "latency_samples": int(latency),
                },
                lambda idx: {
                    "intensity_db": float(intensity_db[idx]),
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
                    "batch_integer_cycle_bias": int(batch_integer_cycle_bias[idx]),
                    "near_real_time_refractive_phase_error_rad": float(
                        phase[idx] - true_phase[idx]
                    ),
                    "near_real_time_integer_cycle_bias": _optional_float(integer_cycle_bias[idx]),
                },
            )
        )
    return rows


def _finite_tvd_integer_bias(
    phase_error: np.ndarray,
    intensity_db: np.ndarray,
    tvd_cfg: dict,
) -> np.ndarray:
    # Paper: Figure 13 uses sliding-window BP traces, which are only defined
    # where the selected M,L window can be evaluated. TVD is therefore fit on
    # finite trace samples only, preserving NaNs outside the sliding support.
    output = np.full(np.asarray(phase_error).shape, np.nan, dtype=float)
    finite = np.isfinite(phase_error) & np.isfinite(intensity_db)
    if np.count_nonzero(finite) < 2:
        return output
    output[finite] = tvd_integer_bias(
        np.asarray(phase_error)[finite],
        np.asarray(intensity_db)[finite],
        float(tvd_cfg.get("fade_threshold_db", -10.0)),
        float(tvd_cfg.get("allowed_jump_weight", 1.0)),
        float(tvd_cfg.get("blocked_jump_weight", 10000.0)),
    )
    return output


def _optional_float(value: float) -> float:
    return float(value) if np.isfinite(value) else np.nan


def forecast_example_rows(
    metadata, case_df, signal, true_signal, var_result, block_len
) -> list[dict]:
    """Build rows for the VAR forecast example figure.

    Args:
        metadata: Metadata for the selected tracking case.
        case_df: Tracking samples for one case.
        signal: Complex tracked scintillation signal.
        true_signal: Complex noiseless truth scintillation signal.
        var_result: Fitted VAR result used to forecast future samples.
        block_len: Sliding FFT block length that defines the future context.

    Returns:
        List of CSV rows comparing true, tracked, and forecasted real/imaginary
        samples after each forecast start.
    """
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
                    **case_keys(metadata),
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
    requested = np.asarray([0, 1024, 2048], dtype=int)
    return np.unique(requested[requested <= max_forecast_latency])


def _example_ml_configs(near_cfg: dict) -> list[tuple[int, int]]:
    configs = []
    for item in near_cfg.get("example_ml_configs", []) or []:
        block_len = int(item.get("block_len"))
        latency = int(item.get("latency_samples"))
        configs.append((block_len, latency))
    if configs:
        return configs
    block_len = int(near_cfg.get("practical_fft_m", 8192))
    latency = int(near_cfg.get("practical_latency_samples", block_len // 2 - 1))
    return [(block_len, latency)]
