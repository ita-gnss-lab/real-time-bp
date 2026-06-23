"""Filter-bank Fresnel-time scoring for batch backpropagation."""

from __future__ import annotations

import numpy as np

from real_time_bp.methodology.backpropagation import batch_backpropagate

AMPLITUDE_SCORE_NAME = "amplitude_std"


def compute_filter_bank_score(
    scintillation_signal: np.ndarray,
    sample_interval_s: float,
    fresnel_time_candidates_s: np.ndarray,
) -> np.ndarray:
    """Evaluate the amplitude-variance filter-bank score on each candidate.

    Args:
        scintillation_signal: Complex scintillation signal segment used for the
            filter-bank initialization sweep.
        sample_interval_s: Sample spacing in seconds.
        fresnel_time_candidates_s: Candidate Fresnel time scales in seconds.

    Returns:
        Score value for each Fresnel-time candidate.
    """
    amplitude_scores: list[float] = []
    for candidate in fresnel_time_candidates_s:
        # Paper: Methodology Section II.4. For each candidate
        # T_F, backpropagate the N_init segment and measure how close the
        # recovered screen-domain amplitude is to a constant-envelope field.
        backpropagated = batch_backpropagate(
            scintillation_signal,
            sample_interval_s,
            float(candidate),
        )
        # Paper Eq. (16): J(T_F) is
        # the standard deviation of the backpropagated amplitude.
        amplitude_scores.append(float(np.nanstd(np.abs(backpropagated))))

    return np.asarray(amplitude_scores, dtype=float)


def select_fresnel_time_estimate(
    candidates: np.ndarray,
    score_curve: np.ndarray,
) -> float:
    """Select the Fresnel-time estimate associated with the score minimum.

    Args:
        candidates: Candidate Fresnel time scales in seconds.
        score_curve: Score value for each Fresnel-time candidate.

    Returns:
        Estimated Fresnel time scale.
    """
    # Paper: Methodology Section II.4, Eq. (17). The T_F estimate is
    # the amplitude-score argmin over the candidate grid.
    return float(candidates[int(np.nanargmin(score_curve))])
