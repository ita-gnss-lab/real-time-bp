"""Shared inputs and small utilities for sliding result stages."""

from __future__ import annotations

import numpy as np
import pandas as pd

from real_time_bp.config import OUTPUTS_DIR
from real_time_bp.methodology.filter_bank import AMPLITUDE_SCORE_NAME
from real_time_bp.utils.io import iter_case_dataframes
from real_time_bp.utils.quality import (
    evaluate_analysis_quality,
    exclude_loss_of_lock_enabled,
)
from real_time_bp.utils.stage import csv_chunksize_from_params, l1_wavelength_from_params

SLIDING_TRACKING_COLUMNS = {
    "estimated_scintillation_phase_m",
    "scint_phase_m",
    "time_sec",
    "tracked_intensity",
    "true_intensity",
    "true_refractive_phase_m",
}


def quality_valid_case_ids(
    params: dict,
    paths,
    candidate_case_ids: set[int],
) -> set[int]:
    """Return candidate case ids accepted by the shared analysis quality gate.

    Args:
        params: Parsed `params.yaml` dictionary.
        paths: Resolved dataset paths.
        candidate_case_ids: Case ids considered for example selection.

    Returns:
        Candidate ids whose tracking samples pass `quality_control`.
    """
    if not candidate_case_ids:
        return set()
    wavelength_m = l1_wavelength_from_params(params)
    experiments = params.get("experiments", {})
    chunksize = csv_chunksize_from_params(params)
    exclude_loss_of_lock = exclude_loss_of_lock_enabled(params)
    valid: set[int] = set()
    seen: set[int] = set()
    case_iter = iter_case_dataframes(
        paths,
        severities=experiments.get("severities"),
        max_cases_per_condition=None,
        chunksize=chunksize,
        columns=SLIDING_TRACKING_COLUMNS,
        exclude_loss_of_lock=exclude_loss_of_lock,
    )
    for metadata, case_df in case_iter:
        case_id = int(metadata.get("tracking_case_id", 0) or 0)
        if case_id not in candidate_case_ids:
            continue
        seen.add(case_id)
        quality = evaluate_analysis_quality(metadata, case_df, params, float(wavelength_m))
        if quality.passed:
            valid.add(case_id)
        if seen == candidate_case_ids:
            break
    return valid


def load_amplitude_tf_estimates() -> dict[int, float]:
    """Load amplitude-score Fresnel-time estimates from filter-bank results.

    Main paper:
        Results Section III.3 explicitly uses the amplitude standard-deviation
        baseline from Methodology Section II.4.

    Returns:
        Mapping from tracking case identifier to estimated Fresnel time scale.
    """
    path = OUTPUTS_DIR / "03_filter_bank" / "filter_bank_estimates.csv"
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

    Main paper:
        Methodology Eq. (18) excludes edge samples without full past/future
        block support from sliding BP evaluation.

    Args:
        n_samples: Total number of samples in the case.
        block_len: Sliding FFT block length.

    Returns:
        Integer indices where the sliding BP window can be evaluated.
    """
    eval_slice = valid_eval_slice(n_samples, block_len)
    return np.arange(eval_slice.start, eval_slice.stop, dtype=int)


def valid_eval_slice(n_samples: int, block_len: int, start_index: int = 0) -> slice:
    """Return the contiguous analyzed segment for sliding-window support.

    Args:
        n_samples: Total number of samples in the case.
        block_len: Sliding FFT block length.
        start_index: First sample kept after initialization.

    Returns:
        Slice spanning samples with full sliding-window support and index at
        least `start_index`.
    """
    half = int(block_len) // 2
    future_context = int(block_len) - half - 1
    start = max(int(start_index), half)
    stop = max(start, int(n_samples) - future_context)
    return slice(start, stop)


def thin_eval_indices(indices: np.ndarray, max_points: int) -> np.ndarray:
    """Subsample evaluation indices for expensive diagnostic sweeps.

    Implementation:
        Results heatmaps can be expensive; thinning keeps the M/L sweep
        tractable while preserving the triangular grid definition.

    Args:
        indices: Candidate evaluation indices.
        max_points: Maximum number of indices to keep.

    Returns:
        Original indices when already short enough, otherwise evenly spaced
        samples over the same span.
    """
    if indices.size <= max_points:
        return indices
    positions = np.linspace(0, indices.size - 1, int(max_points), dtype=int)
    return indices[positions]
