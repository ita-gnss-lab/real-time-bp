from __future__ import annotations

import numpy as np

from real_time_bp.methodology.backpropagation import batch_backpropagate

AMPLITUDE_SCORE_NAME = "amplitude_std"
SCORE_NAMES = (AMPLITUDE_SCORE_NAME,)


def candidate_grid(min_s: float, max_s: float, count: int) -> np.ndarray:
    # Paper: Methodology Section II.4 defines the candidate
    # set T = {T_F,1, ..., T_F,P}; params.yaml fixes the grid limits/results.
    return np.linspace(float(min_s), float(max_s), int(count))


def compute_filter_bank_scores(
    scintillation_signal: np.ndarray,
    sample_interval_s: float,
    fresnel_time_candidates_s: np.ndarray,
) -> dict[str, np.ndarray]:
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
        # Paper Eq. (16): J_A(T_F) is
        # the standard deviation of the backpropagated amplitude.
        amplitude_scores.append(float(np.nanstd(np.abs(backpropagated))))

    amplitude = np.asarray(amplitude_scores, dtype=float)
    return {AMPLITUDE_SCORE_NAME: amplitude}


def select_fresnel_time_estimates(
    candidates: np.ndarray,
    score_curves: dict[str, np.ndarray],
) -> dict[str, float]:
    estimates: dict[str, float] = {}
    for name, values in score_curves.items():
        # Paper: Methodology Section II.4, Eq. (17). The T_F estimate is
        # the amplitude-score argmin over the candidate grid.
        estimates[name] = float(candidates[int(np.nanargmin(values))])
    return estimates
