"""Metric functions used by the result-generation DVC stages."""

from __future__ import annotations

import numpy as np


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """Wrap phase errors to the interval [-pi, pi).

    Main paper:
        Results Section III.2, Eq. (22), evaluates phase errors through the
        wrapping operator W(.), so phase RMSE is not dominated by 2*pi branches.

    Args:
        x: Phase values or phase errors in radians.

    Returns:
        Wrapped phase values in radians.
    """
    return (np.asarray(x, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    """Return positions that are finite in every provided array.

    Args:
        *arrays: Arrays with a common broadcast-free shape.

    Returns:
        Boolean mask that is true only where every input is finite.
    """
    mask = np.ones(np.asarray(arrays[0]).shape, dtype=bool)
    for array in arrays:
        mask &= np.isfinite(array)
    return mask


def rmse(x: np.ndarray) -> float:
    """Compute root-mean-square error after dropping non-finite samples.

    Main paper:
        Results Section III.2 reports RMSE_TF and phase MRMSE using this
        root-mean-square reduction.

    Args:
        x: Error samples.

    Returns:
        Root-mean-square value, or NaN when no finite samples exist.
    """
    values = np.asarray(x, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values**2)))


def mae(x: np.ndarray) -> float:
    """Compute mean absolute error after dropping non-finite samples.

    Main paper:
        Results Section III.2 reports MAE_TF by scintillation severity.

    Args:
        x: Error samples.

    Returns:
        Mean absolute value, or NaN when no finite samples exist.
    """
    values = np.asarray(x, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.mean(np.abs(values)))


def mape(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Compute mean absolute percentage error.

    Main paper:
        Results Section III.2 reports MAPE_TF by scintillation severity.

    Args:
        reference: Reference values used in the denominator.
        estimate: Estimated values compared with `reference`.

    Returns:
        Mean absolute percentage error in percent, or NaN when unavailable.
    """
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    mask = finite_mask(reference, estimate) & (np.abs(reference) > 0)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((estimate[mask] - reference[mask]) / reference[mask])) * 100.0)


def r2_score(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Compute the coefficient of determination for an estimate.

    Main paper:
        Results Section III.2 compares filter-bank estimates with R^2_TF,
        using the true CPSSM Fresnel time scale as the reference.

    Args:
        reference: Reference values.
        estimate: Estimated values.

    Returns:
        R-squared value, or NaN when it is not defined.
    """
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
    """Compute wrapped phase RMSE in radians.

    Main paper:
        Results Section III.2, Eq. (22), defines the per-case wrapped phase
        RMSE. Result plots average those per-case values into the reported
        MRMSE curves.

    Args:
        reference_phase_rad: Reference phase samples in radians.
        estimate_phase_rad: Estimated phase samples in radians.

    Returns:
        Root-mean-square wrapped phase error in radians.
    """
    mask = finite_mask(reference_phase_rad, estimate_phase_rad)
    if not np.any(mask):
        return float("nan")
    return rmse(wrap_to_pi(estimate_phase_rad[mask] - reference_phase_rad[mask]))
