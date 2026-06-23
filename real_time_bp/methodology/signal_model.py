"""Signal reconstruction helpers for tracked and truth scintillation phases."""

from __future__ import annotations

import numpy as np
import pandas as pd

from real_time_bp.methodology.constants import meters_to_radians


def estimate_sample_interval_s(case_df: pd.DataFrame) -> float:
    """Estimate the sample interval from a tracking case table.

    Main paper:
        Methodology Section II.2 writes the prompt stream at coherent
        integration steps kT; this recovers T from each case table.

    Args:
        case_df: Tracking samples for one case.

    Returns:
        Median spacing between unique sample times in seconds.
    """
    values = np.sort(case_df["time_sec"].dropna().unique())
    if values.size < 2:
        return float("nan")
    return float(np.median(np.diff(values)))


def estimated_scintillation_phase_rad(
    case_df: pd.DataFrame,
    wavelength_m: float,
    residual_doppler_error_hz: float = 0.0,
) -> np.ndarray:
    """Compute the LOS-corrected tracked scintillation phase estimate.

    Main paper:
        Methodology Section II.2, Eq. (12), defines the LOS-corrected tracked
        scintillation phase estimate. The `residual_doppler_error_hz` term is
        the explicit sensitivity perturbation used in Section III.3.

    Args:
        case_df: Tracking samples for one case.
        wavelength_m: Carrier wavelength in meters.
        residual_doppler_error_hz: Optional synthetic residual Doppler error
            subtracted for sensitivity studies.

    Returns:
        Estimated scintillation phase in radians.
    """
    # The compact dataset stores this LOS-corrected phase directly.
    phase_m = case_df["estimated_scintillation_phase_m"].to_numpy(dtype=float)
    phase_rad = meters_to_radians(phase_m, wavelength_m)
    time_sec = case_df["time_sec"].to_numpy(dtype=float)
    time_rel = time_sec - time_sec[0]
    return phase_rad - 2.0 * np.pi * float(residual_doppler_error_hz) * time_rel


def true_scintillation_phase_rad(case_df: pd.DataFrame, wavelength_m: float) -> np.ndarray:
    """Load the truth scintillation phase in radians.

    Main paper:
        Methodology Section II.3 uses `scint_phase_m` as the CPSSM truth phase
        for validation diagnostics.

    Args:
        case_df: Tracking samples for one case.
        wavelength_m: Carrier wavelength in meters.

    Returns:
        True propagated scintillation phase in radians.
    """
    return meters_to_radians(case_df["scint_phase_m"].to_numpy(dtype=float), wavelength_m)


def true_refractive_phase_rad(case_df: pd.DataFrame, wavelength_m: float) -> np.ndarray:
    """Load the truth refractive screen phase in radians.

    Main paper:
        Methodology Section II.1 names the latent screen phase phi_r[k]. The
        `true_refractive_phase_m` column is the ground truth BP target.

    Args:
        case_df: Tracking samples for one case.
        wavelength_m: Carrier wavelength in meters.

    Returns:
        True refractive phase in radians.
    """
    return meters_to_radians(
        case_df["true_refractive_phase_m"].to_numpy(dtype=float), wavelength_m
    )


def estimated_scintillation_signal(
    case_df: pd.DataFrame,
    wavelength_m: float,
    residual_doppler_error_hz: float = 0.0,
) -> np.ndarray:
    """Build the complex tracked scintillation signal used by BP.

    Main paper:
        Methodology Section II.2, Eq. (13), defines the BP input as the tracked
        scintillation field sqrt(I[k]) exp(j * phi_s_hat[k]).

    Args:
        case_df: Tracking samples for one case.
        wavelength_m: Carrier wavelength in meters.
        residual_doppler_error_hz: Optional residual Doppler sensitivity term.

    Returns:
        Complex signal with tracked amplitude and LOS-corrected phase.
    """
    intensity = case_df["tracked_intensity"].to_numpy(dtype=float)
    amplitude = np.sqrt(np.maximum(intensity, 0.0))
    phase = estimated_scintillation_phase_rad(
        case_df,
        wavelength_m,
        residual_doppler_error_hz,
    )
    return amplitude * np.exp(1j * phase)


def true_scintillation_signal(case_df: pd.DataFrame, wavelength_m: float) -> np.ndarray:
    """Build the noiseless truth scintillation signal.

    Main paper:
        Methodology Sections II.2-II.3 define the noiseless propagated
        scintillation field sqrt(I_true[k]) exp(j phi_s[k]), which is used for
        VAR forecast validation figures.

    Args:
        case_df: Tracking samples for one case.
        wavelength_m: Carrier wavelength in meters.

    Returns:
        Complex truth signal using true intensity and scintillation phase.
    """
    intensity = case_df["true_intensity"].to_numpy(dtype=float)
    amplitude = np.sqrt(np.maximum(intensity, 0.0))
    phase = true_scintillation_phase_rad(case_df, wavelength_m)
    return amplitude * np.exp(1j * phase)


def detrended_intensity_db(case_df: pd.DataFrame) -> np.ndarray:
    """Return tracked detrended intensity in decibels.

    Main paper:
        Methodology Section II.2, Eq. (10), uses detrended scintillation
        intensity I[k]. Plots and TVD convert that intensity to dB units.

    Args:
        case_df: Tracking samples for one case.

    Returns:
        Detrended tracked intensity in dB.
    """
    intensity = np.maximum(
        case_df["tracked_intensity"].to_numpy(dtype=float), np.finfo(float).tiny
    )
    return 10.0 * np.log10(intensity)
