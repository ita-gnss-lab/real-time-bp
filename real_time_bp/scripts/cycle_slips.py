"""TVD cycle-slip helpers used by representative and sliding result stages."""

from __future__ import annotations

import numpy as np
import pandas as pd

from real_time_bp.methodology.backpropagation import median_align_phase
from real_time_bp.methodology.signal_model import (
    detrended_intensity_db,
    estimated_scintillation_phase_rad,
    true_refractive_phase_rad,
    true_scintillation_phase_rad,
)
from real_time_bp.methodology.tvd import detect_cycle_slips_from_tvd
from real_time_bp.utils.output_rows import sampled_time_rows
from real_time_bp.utils.quality import evaluate_analysis_quality
from real_time_bp.utils.representatives import SEVERITY_ORDER, sort_by_severity
from real_time_bp.utils.stage import case_keys, is_loss_of_lock

TVD_TRACKING_COLUMNS = {
    "estimated_scintillation_phase_m",
    "scint_phase_m",
    "time_sec",
    "tracked_intensity",
    "true_refractive_phase_m",
}
TRACE_COLUMNS = [
    "tracking_case_id",
    "example_idx",
    "c_n0_idx",
    "severity",
    "frequency",
    "svid",
    "input_c_n0_db_hz",
    "init_doppler_error_hz",
    "loss_of_lock",
    "true_fresnel_time_scale_s",
    "sample_index",
    "time_sec",
    "intensity_db",
    "estimated_diffractive_phase_cycles",
    "true_diffractive_phase_cycles",
    "integer_cycle_bias",
    "is_fade_minimum",
]
STAGE_DIR = "02_cycle_slips"
TRACES_CSV = "tvd_example_traces.csv"


def compute_tvd_summary_row(
    metadata: dict,
    case_df,
    params: dict,
    wavelength_m: float,
    threshold_db: float,
    allowed_jump_weight: float,
    blocked_jump_weight: float,
) -> dict | None:
    """Compute one tracked-phase TVD cycle-slip summary row.

    Main paper:
        Methodology Eq. (12) supplies the LOS-corrected phase before removing
        the known refractive component for the diffractive-phase TVD input.
        Methodology Section II.2 and Breitsch et al. (2020, Section III.C)
        define slips as nonzero first differences of the rounded TVD
        cycle-bias staircase. Representative figures use this summary only to
        choose examples; all-case CS/min is computed in the sliding stage.
    """
    quality = evaluate_analysis_quality(metadata, case_df, params, float(wavelength_m))
    if not quality.passed:
        return None
    time_sec = case_df["time_sec"].to_numpy(dtype=float)
    intensity_db = detrended_intensity_db(case_df)
    estimated_phase = estimated_scintillation_phase_rad(case_df, float(wavelength_m))
    true_refractive = true_refractive_phase_rad(case_df, float(wavelength_m))
    diffractive_phase = estimated_phase - true_refractive
    fade_minima, _integer_cycle_bias, cycle_slip_indices = detect_cycle_slips_from_tvd(
        diffractive_phase,
        intensity_db,
        float(threshold_db),
        float(allowed_jump_weight),
        float(blocked_jump_weight),
    )
    metadata_keys = case_keys(
        metadata,
        include_loss_of_lock=True,
        include_true_fresnel_time=True,
    )
    duration_min = (time_sec[-1] - time_sec[0]) / 60.0 if time_sec.size > 1 else np.nan
    return {
        **metadata_keys,
        "n_samples": int(case_df.shape[0]),
        "n_fade_minima": int(fade_minima.size),
        "n_tvd_cycle_slips": int(cycle_slip_indices.size),
        "cycle_slips_per_minute": float(cycle_slip_indices.size / duration_min)
        if duration_min and duration_min > 0
        else np.nan,
        "diffractive_phase_slope_cycles_per_s": quality.slope_cycles_per_s,
        "diffractive_phase_span_cycles": quality.span_cycles,
        "analysis_quality_pass": quality.passed,
        "analysis_quality_reason": quality.reason,
    }


def _compute_tvd_trace_rows(
    metadata: dict,
    case_df,
    wavelength_m: float,
    n_init: int,
    threshold_db: float,
    allowed_jump_weight: float,
    blocked_jump_weight: float,
) -> list[dict]:
    intensity_db = detrended_intensity_db(case_df)
    estimated_phase = estimated_scintillation_phase_rad(case_df, float(wavelength_m))
    true_scintillation = true_scintillation_phase_rad(case_df, float(wavelength_m))
    true_refractive = true_refractive_phase_rad(case_df, float(wavelength_m))
    # Plot-only convention: remove the arbitrary carrier-phase constant from
    # the displayed scintillation phase. The event detector uses the same fade
    # minima because constant phase offsets do not change jump locations.
    aligned_estimated_phase = median_align_phase(
        estimated_phase,
        true_scintillation,
        min(int(n_init), len(case_df)),
    )
    estimated_diffractive_cycles = (aligned_estimated_phase - true_refractive) / (2.0 * np.pi)
    true_diffractive_cycles = (true_scintillation - true_refractive) / (2.0 * np.pi)
    fade_minima, integer_cycle_bias, _ = detect_cycle_slips_from_tvd(
        aligned_estimated_phase - true_refractive,
        intensity_db,
        float(threshold_db),
        float(allowed_jump_weight),
        float(blocked_jump_weight),
    )
    fade_minima_set = set(int(x) for x in fade_minima)
    metadata_keys = case_keys(
        metadata,
        include_loss_of_lock=True,
        include_true_fresnel_time=True,
    )
    return sampled_time_rows(
        case_df,
        metadata_keys,
        lambda idx: {
            "intensity_db": float(intensity_db[idx]),
            "estimated_diffractive_phase_cycles": float(estimated_diffractive_cycles[idx]),
            "true_diffractive_phase_cycles": float(true_diffractive_cycles[idx]),
            "integer_cycle_bias": int(integer_cycle_bias[idx]),
            "is_fade_minimum": idx in fade_minima_set,
        },
    )


def _representative_rank_key(summary_row: dict) -> tuple[float, ...] | None:
    quality_pass = summary_row.get("analysis_quality_pass", True)
    if isinstance(quality_pass, str):
        quality_pass = quality_pass.strip().lower() in {"1", "true", "yes"}
    if not bool(quality_pass):
        return None
    if is_loss_of_lock(summary_row):
        return None
    severity = str(summary_row.get("severity"))
    n_slips = int(summary_row.get("n_tvd_cycle_slips") or 0)
    slip_penalty = 1.0 if severity == "strong" and n_slips <= 0 else 0.0
    c_n0 = float(summary_row.get("input_c_n0_db_hz") or 0.0)
    slope = abs(float(summary_row.get("diffractive_phase_slope_cycles_per_s") or 0.0))
    case_id = float(summary_row.get("tracking_case_id") or 0.0)
    return (slip_penalty, -c_n0, slope, case_id)


def _representative_rows(
    candidates_by_severity: dict[str, list[tuple[tuple[float, ...], dict]]],
    experiments: dict,
) -> list[dict]:
    rows = []
    ranks = experiments.get("representative_case_rank_by_severity", {}) or {}
    for severity in SEVERITY_ORDER:
        candidates = sorted(candidates_by_severity.get(severity, []), key=lambda item: item[0])
        if not candidates:
            continue
        requested_rank = max(int(ranks.get(severity, 0)), 0)
        row = dict(candidates[min(requested_rank, len(candidates) - 1)][1])
        row["abs_diffractive_phase_slope_cycles_per_s"] = abs(
            float(row.get("diffractive_phase_slope_cycles_per_s") or 0.0)
        )
        rows.append(row)
    return sort_by_severity(pd.DataFrame(rows)).to_dict("records") if rows else []
