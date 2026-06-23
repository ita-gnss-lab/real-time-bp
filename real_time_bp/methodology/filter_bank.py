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

    Main paper:
        Methodology Section II.4 backpropagates the N_init segment for each
        candidate T_F and measures whether the recovered screen-domain
        amplitude is close to a constant-envelope field. Eq. (16) defines
        J(T_F) as the standard deviation of the backpropagated amplitude.

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
        backpropagated = batch_backpropagate(
            scintillation_signal,
            sample_interval_s,
            float(candidate),
        )
        amplitude_scores.append(float(np.nanstd(np.abs(backpropagated))))

    return np.asarray(amplitude_scores, dtype=float)


def select_fresnel_time_estimate(
    candidates: np.ndarray,
    score_curve: np.ndarray,
) -> float:
    """Select the Fresnel-time estimate associated with the score minimum.

    Main paper:
        Methodology Section II.4, Eq. (17), defines the T_F estimate as the
        candidate that minimizes the amplitude score J(T_F).

    Args:
        candidates: Candidate Fresnel time scales in seconds.
        score_curve: Score value for each Fresnel-time candidate.

    Returns:
        Estimated Fresnel time scale.
    """
    return float(candidates[int(np.nanargmin(score_curve))])
