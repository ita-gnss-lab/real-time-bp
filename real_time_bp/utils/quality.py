"""Quality gates used to exclude loss-of-lock cases from analysis stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from real_time_bp.methodology.signal_model import (
    estimated_scintillation_phase_rad,
    true_refractive_phase_rad,
)
from real_time_bp.utils.stage import is_loss_of_lock


@dataclass(frozen=True)
class AnalysisQuality:
    """Phase-quality summary for one tracking case.

    Args:
        passed: True when the case should be included in analysis.
        reason: Empty for accepted cases, otherwise the rejection reason.
        slope_cycles_per_s: Linear slope of `hat_phi_s - phi_r` in cycles/s.
        span_cycles: Peak-to-peak span of `hat_phi_s - phi_r` in cycles.
    """

    passed: bool
    reason: str
    slope_cycles_per_s: float
    span_cycles: float


def exclude_loss_of_lock_enabled(params: dict[str, Any]) -> bool:
    """Read the metadata-level loss-of-lock cleanup flag.

    Args:
        params: Parsed `params.yaml` dictionary.

    Returns:
        True when stages should skip cases flagged as loss of lock before
        running expensive computations.
    """
    return bool(params.get("quality_control", {}).get("exclude_loss_of_lock", True))


def evaluate_analysis_quality(
    metadata: dict[str, Any],
    case_df: pd.DataFrame,
    params: dict[str, Any],
    wavelength_m: float,
) -> AnalysisQuality:
    """Decide whether a tracking case is usable for the paper analysis.

    Main paper:
        Methodology Section II.2 removes LOS dynamics and the known
        refractive phase to isolate the diffractive phase component
        `hat_phi_s - phi_r`. Cases with loss of carrier lock can produce large
        artificial ramps in this residual, so the analysis excludes both
        cases explicitly marked as `loss_of_lock` and cases whose residual
        phase slope or span is too large to represent scintillation.

    Args:
        metadata: Tracking-case metadata.
        case_df: Tracking samples for one case.
        params: Parsed `params.yaml` dictionary.
        wavelength_m: Carrier wavelength in meters.

    Returns:
        Analysis-quality decision and diagnostic phase metrics.
    """
    cfg = params.get("quality_control", {})
    slope_cycles_per_s, span_cycles = diffractive_phase_quality_metrics(
        case_df,
        wavelength_m,
    )
    if bool(cfg.get("exclude_loss_of_lock", True)) and is_loss_of_lock(metadata):
        return AnalysisQuality(False, "loss_of_lock", slope_cycles_per_s, span_cycles)

    max_abs_slope = cfg.get("max_abs_diffractive_phase_slope_cycles_per_s", None)
    if max_abs_slope is not None and np.isfinite(slope_cycles_per_s):
        if abs(float(slope_cycles_per_s)) > float(max_abs_slope):
            return AnalysisQuality(
                False, "diffractive_phase_slope", slope_cycles_per_s, span_cycles
            )

    max_span = cfg.get("max_diffractive_phase_span_cycles", None)
    if max_span is not None and np.isfinite(span_cycles):
        if float(span_cycles) > float(max_span):
            return AnalysisQuality(
                False, "diffractive_phase_span", slope_cycles_per_s, span_cycles
            )

    if not np.isfinite(slope_cycles_per_s) or not np.isfinite(span_cycles):
        return AnalysisQuality(False, "invalid_diffractive_phase", slope_cycles_per_s, span_cycles)

    return AnalysisQuality(True, "", slope_cycles_per_s, span_cycles)


def diffractive_phase_quality_metrics(
    case_df: pd.DataFrame,
    wavelength_m: float,
) -> tuple[float, float]:
    """Measure gross ramp behavior in `hat_phi_s - phi_r`.

    Args:
        case_df: Tracking samples for one case.
        wavelength_m: Carrier wavelength in meters.

    Returns:
        Pair `(slope_cycles_per_s, span_cycles)`.
    """
    time_sec = case_df["time_sec"].to_numpy(dtype=float)
    estimated_phase = estimated_scintillation_phase_rad(case_df, float(wavelength_m))
    true_refractive = true_refractive_phase_rad(case_df, float(wavelength_m))
    diffractive_phase_cycles = (estimated_phase - true_refractive) / (2.0 * np.pi)
    return (
        lisliding_slope_cycles_per_s(time_sec, diffractive_phase_cycles),
        phase_span_cycles(diffractive_phase_cycles),
    )


def lisliding_slope_cycles_per_s(time_sec: np.ndarray, phase_cycles: np.ndarray) -> float:
    """Fit the first-order phase slope in cycles/s.

    Args:
        time_sec: Sample times in seconds.
        phase_cycles: Phase samples in carrier cycles.

    Returns:
        Least-squares linear slope in cycles/s, or NaN when unavailable.
    """
    time_sec = np.asarray(time_sec, dtype=float)
    phase_cycles = np.asarray(phase_cycles, dtype=float)
    if time_sec.size < 2 or phase_cycles.size < 2:
        return float("nan")
    finite = np.isfinite(time_sec) & np.isfinite(phase_cycles)
    if finite.sum() < 2:
        return float("nan")
    time_rel = time_sec[finite] - time_sec[finite][0]
    return float(np.polyfit(time_rel, phase_cycles[finite], 1)[0])


def phase_span_cycles(phase_cycles: np.ndarray) -> float:
    """Compute finite peak-to-peak phase span in cycles.

    Args:
        phase_cycles: Phase samples in carrier cycles.

    Returns:
        Finite peak-to-peak span, or NaN when unavailable.
    """
    phase_cycles = np.asarray(phase_cycles, dtype=float)
    finite = phase_cycles[np.isfinite(phase_cycles)]
    if finite.size == 0:
        return float("nan")
    return float(np.nanmax(finite) - np.nanmin(finite))
