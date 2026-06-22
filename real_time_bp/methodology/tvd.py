from __future__ import annotations

import numpy as np

from real_time_bp.methodology.constants import radians_to_cycles


def find_fade_minima(intensity_db: np.ndarray, threshold_db: float) -> np.ndarray:
    # Paper: Methodology Section II.2 cites the TVD cycle-slip
    # procedure of Breitsch et al. The allowed jump locations are local fade
    # minima below the configured -15 dB threshold.
    values = np.asarray(intensity_db, dtype=float)
    if values.size < 3:
        return np.array([], dtype=int)
    center = values[1:-1]
    minima = (center <= values[:-2]) & (center <= values[2:]) & (center <= float(threshold_db))
    return np.flatnonzero(minima) + 1


def tvd_weights(
    n_samples: int,
    fade_minima: np.ndarray,
    allowed_jump_weight: float,
    blocked_jump_weight: float,
) -> np.ndarray:
    # Paper/TVD implementation detail: the Breitsch-style weighted TVD uses
    # small penalty at fade minima and a large penalty elsewhere, so phase jumps
    # are encouraged only where the amplitude fade makes a transition plausible.
    weights = np.full(max(int(n_samples) - 1, 0), float(blocked_jump_weight), dtype=float)
    if weights.size == 0:
        return weights
    for minimum in np.asarray(fade_minima, dtype=int):
        weights[int(np.clip(minimum, 0, weights.size - 1))] = float(allowed_jump_weight)
    return weights


def weighted_tvd_fit(phase_cycles: np.ndarray, weights: np.ndarray) -> np.ndarray:
    # Paper: the methodology footnote points to proxTV for exact weighted TVD.
    # tv1w_1d solves the one-dimensional weighted total-variation denoising fit.
    try:
        from prox_tv import tv1w_1d
    except Exception as exc:  # pragma: no cover - depends on native prox-tv build.
        raise RuntimeError(
            "prox-tv is required for weighted TVD. Install system packages "
            "build-essential libblas-dev liblapack-dev liblapacke-dev, then run uv sync."
        ) from exc
    return tv1w_1d(np.asarray(phase_cycles, dtype=float), np.asarray(weights, dtype=float))


def quantize_tvd_cycle_bias(tvd_fit_cycles: np.ndarray) -> np.ndarray:
    # Paper/TVD post-processing: after fitting the slowly varying cycle-bias
    # staircase, the fit is rounded to integer cycles before differencing.
    fit = np.asarray(tvd_fit_cycles, dtype=float)
    finite = np.isfinite(fit)
    if not np.any(finite):
        return np.zeros_like(fit, dtype=int)
    fractional_bias = np.median(fit[finite] - np.rint(fit[finite]))
    return np.rint(fit - fractional_bias).astype(int)


def detect_cycle_slips_from_tvd(
    diffractive_phase_rad: np.ndarray,
    intensity_db: np.ndarray,
    threshold_db: float,
    allowed_jump_weight: float,
    blocked_jump_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Paper: Methodology Section II.2; Results
    # Section III.2 report cycle slips/minute from these detected
    # integer jumps. The input phase is converted to cycles because TVD models
    # cycle-bias jumps as integer-valued steps.
    phase_cycles = radians_to_cycles(np.asarray(diffractive_phase_rad, dtype=float))
    fade_minima = find_fade_minima(intensity_db, threshold_db)
    weights = tvd_weights(
        phase_cycles.size,
        fade_minima,
        allowed_jump_weight,
        blocked_jump_weight,
    )
    tvd_fit = weighted_tvd_fit(phase_cycles, weights)
    integer_cycle_bias = quantize_tvd_cycle_bias(tvd_fit)
    valid_minima = fade_minima[fade_minima < integer_cycle_bias.size - 1]
    jumps = integer_cycle_bias[valid_minima + 1] - integer_cycle_bias[valid_minima]
    cycle_slip_indices = valid_minima[jumps != 0]
    return fade_minima, integer_cycle_bias, cycle_slip_indices
