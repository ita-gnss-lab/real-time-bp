"""Shared inputs and small utilities for near-real-time result stages."""

from __future__ import annotations

import numpy as np
import pandas as pd

from real_time_bp.config import CSV_RESULTS_DIR
from real_time_bp.methodology.filter_bank import AMPLITUDE_SCORE_NAME

NEAR_REAL_TIME_TRACKING_COLUMNS = {
    "estimated_scintillation_phase_m",
    "scint_phase_m",
    "time_sec",
    "tracked_intensity",
    "true_intensity",
    "true_refractive_phase_m",
}


def load_amplitude_tf_estimates() -> dict[int, float]:
    """Load amplitude-score Fresnel-time estimates from filter-bank results.

    Returns:
        Mapping from tracking case identifier to estimated Fresnel time scale.
    """
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


def valid_eval_indices(n_samples: int, block_len: int) -> np.ndarray:
    """Return sample indices with full sliding-window support.

    Args:
        n_samples: Total number of samples in the case.
        block_len: Sliding FFT block length.

    Returns:
        Integer indices where the near-real-time BP window can be evaluated.
    """
    # Paper: Methodology Eq. (18); edge samples without the full
    # past/future block support are excluded from near-real-time evaluation.
    half = int(block_len) // 2
    future_context = int(block_len) - half - 1
    return np.arange(half, max(half, int(n_samples) - future_context), dtype=int)


def thin_eval_indices(indices: np.ndarray, max_points: int) -> np.ndarray:
    """Subsample evaluation indices for expensive diagnostic sweeps.

    Args:
        indices: Candidate evaluation indices.
        max_points: Maximum number of indices to keep.

    Returns:
        Original indices when already short enough, otherwise evenly spaced
        samples over the same span.
    """
    # Implementation note for Results heatmaps: thinning keeps the M/L sweep
    # tractable while preserving the triangular grid definition.
    if indices.size <= max_points:
        return indices
    positions = np.linspace(0, indices.size - 1, int(max_points), dtype=int)
    return indices[positions]


def near_real_time_residual_grid_hz(params: dict) -> pd.DataFrame:
    """Construct the residual Doppler grid for near-real-time sensitivity plots.

    Args:
        params: Parsed `params.yaml` dictionary.

    Returns:
        DataFrame with one `residual_doppler_error_hz` column.
    """
    near_cfg = params.get("near_real_time", {})
    values = np.linspace(
        float(near_cfg.get("residual_doppler_error_hz_min", 0.0)),
        float(near_cfg.get("residual_doppler_error_hz_max", 1.5)),
        int(near_cfg.get("residual_doppler_error_count", 16)),
    )
    return pd.DataFrame({"residual_doppler_error_hz": values})
