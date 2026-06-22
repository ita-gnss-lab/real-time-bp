from __future__ import annotations

import numpy as np
import pandas as pd

from real_time_bp.methodology.constants import meters_to_radians


def estimate_sample_interval_s(case_df: pd.DataFrame) -> float:
    # Paper: Methodology Section II.2 writes the prompt stream
    # at coherent-integration steps kT; this recovers T from each case table.
    values = np.sort(case_df["time_sec"].dropna().unique())
    if values.size < 2:
        return float("nan")
    return float(np.median(np.diff(values)))


def estimated_scintillation_phase_rad(
    case_df: pd.DataFrame,
    wavelength_m: float,
    residual_doppler_error_hz: float = 0.0,
) -> np.ndarray:
    # Paper: Methodology Eq. (12).
    # The carrier-loop phase estimate is cleaned by subtracting the LOS phase.
    # Any residual_doppler_error_hz is an explicit Results sensitivity term,
    # not a fitted correction to hide scintillation-driven phase structure.
    carrier_phase_m = case_df["carrier_phase_est_m"].to_numpy(dtype=float)
    los_phase_m = case_df["los_phase_m"].to_numpy(dtype=float)
    phase_rad = meters_to_radians(carrier_phase_m - los_phase_m, wavelength_m)
    time_sec = case_df["time_sec"].to_numpy(dtype=float)
    time_rel = time_sec - time_sec[0]
    return phase_rad - 2.0 * np.pi * float(residual_doppler_error_hz) * time_rel


def true_scintillation_phase_rad(case_df: pd.DataFrame, wavelength_m: float) -> np.ndarray:
    # Paper validation quantity: scint_phase_m is the CPSSM truth phase used
    # in Methodology Section II.3 diagnostics.
    return meters_to_radians(case_df["scint_phase_m"].to_numpy(dtype=float), wavelength_m)


def true_refractive_phase_rad(case_df: pd.DataFrame, wavelength_m: float) -> np.ndarray:
    # Paper: Methodology Section II.1 names the latent screen
    # phase phi_r[k]; this column is the ground truth target for BP retrieval.
    return meters_to_radians(case_df["true_refractive_phase_m"].to_numpy(dtype=float), wavelength_m)


def estimated_scintillation_signal(
    case_df: pd.DataFrame,
    wavelength_m: float,
    residual_doppler_error_hz: float = 0.0,
) -> np.ndarray:
    # Paper: Methodology Eq. (13).
    # The BP input is sqrt(I[k]) * exp(j * phi_s_hat[k]).
    intensity = case_df["tracked_intensity"].to_numpy(dtype=float)
    amplitude = np.sqrt(np.maximum(intensity, 0.0))
    phase = estimated_scintillation_phase_rad(
        case_df,
        wavelength_m,
        residual_doppler_error_hz,
    )
    return amplitude * np.exp(1j * phase)


def true_scintillation_signal(case_df: pd.DataFrame, wavelength_m: float) -> np.ndarray:
    # Paper validation quantity for VAR forecast figures: this is the noiseless
    # propagated scintillation field sqrt(I_true[k]) exp(j phi_s[k]).
    intensity = case_df["true_intensity"].to_numpy(dtype=float)
    amplitude = np.sqrt(np.maximum(intensity, 0.0))
    phase = true_scintillation_phase_rad(case_df, wavelength_m)
    return amplitude * np.exp(1j * phase)


def true_screen_field(case_df: pd.DataFrame, wavelength_m: float) -> np.ndarray:
    # Paper: Methodology Eq. (1); before
    # propagation the screen-domain field is unit amplitude exp(j phi_r[k]).
    return np.exp(1j * true_refractive_phase_rad(case_df, wavelength_m))


def detrended_intensity_db(case_df: pd.DataFrame) -> np.ndarray:
    # Paper: Methodology Eq. (10). The dataset stores
    # the detrended scintillation intensity I[k]; plots and TVD use dB units.
    intensity = np.maximum(case_df["tracked_intensity"].to_numpy(dtype=float), np.finfo(float).tiny)
    return 10.0 * np.log10(intensity)
