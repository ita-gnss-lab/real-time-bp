from __future__ import annotations

import numpy as np


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    # Paper: Results Eq. (30) evaluates phase errors through a
    # wrapping operator W(.), so phase RMSE is not dominated by 2*pi branches.
    return (np.asarray(x, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(np.asarray(arrays[0]).shape, dtype=bool)
    for array in arrays:
        mask &= np.isfinite(array)
    return mask


def rmse(x: np.ndarray) -> float:
    # Paper: Results Section III.2 reports RMSE_TF and phase MRMSE.
    values = np.asarray(x, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values**2)))


def mae(x: np.ndarray) -> float:
    # Paper: Results Section III.2 requests MAE_TF by severity.
    values = np.asarray(x, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.mean(np.abs(values)))


def mape(reference: np.ndarray, estimate: np.ndarray) -> float:
    # Paper: Results Section III.2 requests MAPE_TF by severity.
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    mask = finite_mask(reference, estimate) & (np.abs(reference) > 0)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((estimate[mask] - reference[mask]) / reference[mask])) * 100.0)


def r2_score(reference: np.ndarray, estimate: np.ndarray) -> float:
    # Paper: Results Section III.2 compares filter-bank scores with
    # R^2_TF, using the true CPSSM Fresnel time scale as reference.
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    mask = finite_mask(reference, estimate)
    if np.sum(mask) < 2:
        return float("nan")
    residual_ss = float(np.sum((reference[mask] - estimate[mask]) ** 2))
    total_ss = float(np.sum((reference[mask] - np.mean(reference[mask])) ** 2))
    if total_ss == 0.0:
        return float("nan")
    return 1.0 - residual_ss / total_ss


def phase_rmse_rad(reference_phase_rad: np.ndarray, estimate_phase_rad: np.ndarray) -> float:
    # Paper: Results Eq. (30). This is the per-case wrapped phase
    # RMSE that is later averaged into MRMSE_phi_s.
    mask = finite_mask(reference_phase_rad, estimate_phase_rad)
    if not np.any(mask):
        return float("nan")
    return rmse(wrap_to_pi(estimate_phase_rad[mask] - reference_phase_rad[mask]))


def field_relative_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=complex)
    estimate = np.asarray(estimate, dtype=complex)
    mask = np.isfinite(reference) & np.isfinite(estimate)
    if not np.any(mask):
        return float("nan")
    denom = np.linalg.norm(reference[mask])
    if denom == 0.0:
        return float("nan")
    return float(np.linalg.norm(estimate[mask] - reference[mask]) / denom)


def cycle_slips_per_minute(
    reference_phase_rad: np.ndarray,
    estimate_phase_rad: np.ndarray,
    sample_interval_s: float,
    start_index: int = 0,
) -> tuple[int, float]:
    # Paper: Results Section III.2 defines CS/min over the samples
    # after N_init; jumps are changes in the rounded integer phase error.
    reference = np.asarray(reference_phase_rad, dtype=float)[start_index:]
    estimate = np.asarray(estimate_phase_rad, dtype=float)[start_index:]
    mask = finite_mask(reference, estimate)
    if np.sum(mask) < 2:
        return 0, float("nan")
    integer_cycle_error = np.rint((estimate[mask] - reference[mask]) / (2.0 * np.pi)).astype(int)
    count = int(np.count_nonzero(np.diff(integer_cycle_error)))
    duration_min = float(np.sum(mask) * sample_interval_s / 60.0)
    return count, float(count / duration_min) if duration_min > 0.0 else float("nan")
