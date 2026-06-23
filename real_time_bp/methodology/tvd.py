"""Weighted total-variation denoising helpers for cycle-slip detection.

Warning:
    If code that depends on Breitsch et al. (2020, Section III.C) is moved out
    of this module, include that reference in the new code docstring if it is
    not already present there.
"""

from __future__ import annotations

import numpy as np

from real_time_bp.methodology.constants import radians_to_cycles


def find_fade_minima(intensity_db: np.ndarray, threshold_db: float) -> np.ndarray:
    """Find one amplitude-fade minimum per below-threshold fade segment.

    This implements the fade-minimum selection used by the main paper,
    Methodology Section II.2, for constraining TVD cycle-slip candidates.
    Breitsch et al. (2020, Section III.C) motivates restricting TVD jumps to
    fade minima, but does not specify the discrete minimum finder; here each
    contiguous below-threshold fade contributes one deepest sample.

    Args:
        intensity_db: Detrended intensity samples in dB.
        threshold_db: Maximum intensity value for a sample to belong to a fade.

    Returns:
        Integer sample indices of the deepest point in each below-threshold fade.

    Reference:
        B. Breitsch, Y. T. Morton, C. Rino, and D. Xu, "GNSS Carrier
        Phase Cycle Slips Due to Diffractive Ionosphere Scintillation:
        Simulation and Characterization," IEEE Trans. Aerosp. Electron. Syst.,
        vol. 56, no. 5, pp. 3632-3644, Oct. 2020,
        doi: 10.1109/TAES.2020.2979025.
    """
    values = np.asarray(intensity_db, dtype=float)
    below_threshold = np.isfinite(values) & (values <= float(threshold_db))
    if not np.any(below_threshold):
        return np.array([], dtype=int)

    padded = np.concatenate(([False], below_threshold, [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])

    minima = []
    for start, end in zip(starts, ends, strict=True):
        segment = values[start:end]
        # Use the center of a flat-bottom fade instead of np.argmin's first
        # occurrence, so quantized plateaus do not place the jump at the edge.
        deepest = np.flatnonzero(segment == np.min(segment))
        minima.append(start + int(deepest[deepest.size // 2]))
    return np.asarray(minima, dtype=int)


def tvd_weights(
    n_samples: int,
    fade_minima: np.ndarray,
    allowed_jump_weight: float,
    blocked_jump_weight: float,
) -> np.ndarray:
    """Build first-difference TVD weights from fade-minimum locations.

    This implements the weighted TVD constraint in the main paper,
    Methodology Section II.2: the first-difference penalty is small at fade
    minima and large elsewhere, so the fitted cycle-bias staircase is allowed
    to jump only at fade minima. The fade-minimum restriction follows Breitsch
    et al. (2020, Section III.C).

    Args:
        n_samples: Number of phase samples in the TVD fit.
        fade_minima: Sample indices where phase jumps are allowed.
        allowed_jump_weight: TVD penalty used at fade minima.
        blocked_jump_weight: TVD penalty used away from fade minima.

    Returns:
        Weight vector for the `n_samples - 1` first differences.

    Reference:
        B. Breitsch, Y. T. Morton, C. Rino, and D. Xu, "GNSS Carrier
        Phase Cycle Slips Due to Diffractive Ionosphere Scintillation:
        Simulation and Characterization," IEEE Trans. Aerosp. Electron. Syst.,
        vol. 56, no. 5, pp. 3632-3644, Oct. 2020,
        doi: 10.1109/TAES.2020.2979025.
    """
    weights = np.full(max(int(n_samples) - 1, 0), float(blocked_jump_weight), dtype=float)
    if weights.size == 0:
        return weights
    for minimum in np.asarray(fade_minima, dtype=int):
        weights[int(np.clip(minimum, 0, weights.size - 1))] = float(allowed_jump_weight)
    return weights


def weighted_tvd_fit(phase_cycles: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Solve the weighted one-dimensional TVD fit in phase cycles.

    This is the optimization step used in the main paper, Methodology
    Section II.2, to estimate the piecewise-constant integer cycle-bias signal.
    The numerical solver is proxTV's weighted one-dimensional TVD routine.

    Reference:
        A. Barbero and S. Sra, "Fast newton-type methods for total variation
        regularization," in Proceedings of the 28th International Conference
        on Machine Learning, ICML'11, Bellevue, Washington, USA: Omnipress,
        2011, pp. 313-320.

    Args:
        phase_cycles: Input phase error expressed in cycles.
        weights: First-difference TVD weights.

    Returns:
        Denoised phase-bias staircase in cycles.
    """
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
    """Round a TVD fit to an integer cycle-bias staircase.

    This implements the post-processing in the main paper, Methodology
    Section II.2: after TVD estimates the cycle-bias staircase, the fit is
    shifted by a robust fractional offset and rounded to integer cycles.
    Breitsch et al. (2020, Section III.C) also quantizes the TVD fit to
    identify integer cycle-slip jumps, but this implementation explicitly
    estimates and removes a global modulo-one-cycle fractional bias before
    rounding.

    Args:
        tvd_fit_cycles: Continuous TVD fit in cycles.

    Returns:
        Integer cycle-bias samples.
    """
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
    """Detect cycle slips by constraining TVD jumps to amplitude fades.

    This is the full cycle-slip detector from the main paper, Methodology
    Section II.2: phase is expressed in cycles, weighted TVD estimates the
    cycle-bias staircase, the staircase is quantized to integers, and nonzero
    first differences of the quantized signal are reported as cycle slips. The
    fade-minimum TVD weighting follows Breitsch et al. (2020, Section III.C).

    Args:
        diffractive_phase_rad: Diffractive phase estimate in radians.
        intensity_db: Detrended intensity samples in dB.
        threshold_db: Fade threshold in dB.
        allowed_jump_weight: TVD penalty used at fade minima.
        blocked_jump_weight: TVD penalty used outside fade minima.

    Returns:
        Tuple of fade-minimum indices, integer cycle bias, and detected
        cycle-slip indices.

    Reference:
        B. Breitsch, Y. T. Morton, C. Rino, and D. Xu, "GNSS Carrier
        Phase Cycle Slips Due to Diffractive Ionosphere Scintillation:
        Simulation and Characterization," IEEE Trans. Aerosp. Electron. Syst.,
        vol. 56, no. 5, pp. 3632-3644, Oct. 2020,
        doi: 10.1109/TAES.2020.2979025.
    """
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
    cycle_slip_indices = np.flatnonzero(np.diff(integer_cycle_bias) != 0)
    return fade_minima, integer_cycle_bias, cycle_slip_indices


def tvd_integer_bias(
    phase_rad: np.ndarray,
    intensity_db: np.ndarray,
    threshold_db: float,
    allowed_jump_weight: float,
    blocked_jump_weight: float,
) -> np.ndarray:
    """Return only the integer TVD cycle-bias estimate.

    This is the cycle-bias component of the detector in the main paper,
    Methodology Section II.2. Downstream figures use it when the integer
    staircase itself is needed rather than only the jump locations.

    Args:
        phase_rad: Phase error in radians.
        intensity_db: Detrended intensity samples in dB.
        threshold_db: Fade threshold in dB.
        allowed_jump_weight: TVD penalty used at fade minima.
        blocked_jump_weight: TVD penalty used outside fade minima.

    Returns:
        Integer cycle-bias samples from the weighted TVD fit.
    """
    _, integer_cycle_bias, _ = detect_cycle_slips_from_tvd(
        phase_rad,
        intensity_db,
        threshold_db,
        allowed_jump_weight,
        blocked_jump_weight,
    )
    return integer_cycle_bias
