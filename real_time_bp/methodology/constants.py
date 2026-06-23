"""Physical unit conversions used by the real-time backpropagation pipeline."""

from __future__ import annotations

import numpy as np


def carrier_wavelength_m(speed_of_light_m_s: float, carrier_frequency_hz: float) -> float:
    """Compute carrier wavelength from propagation speed and carrier frequency.

    Args:
        speed_of_light_m_s: Propagation speed in meters per second.
        carrier_frequency_hz: Carrier frequency in hertz.

    Returns:
        Carrier wavelength in meters.
    """
    # Paper: Results Section III.1 converts residual
    # geometric-rate errors to Doppler errors with Delta f_D = Delta G / lambda.
    return float(speed_of_light_m_s / carrier_frequency_hz)


def meters_to_radians(phase_m: np.ndarray, wavelength_m: float) -> np.ndarray:
    """Convert carrier phase stored in meters to radians.

    Args:
        phase_m: Phase values expressed as equivalent path length in meters.
        wavelength_m: Carrier wavelength in meters.

    Returns:
        Phase values expressed in radians.
    """
    # Paper: Methodology Section II.2 stores carrier phase in
    # metric units but defines scintillation phase in radians.
    return 2.0 * np.pi * np.asarray(phase_m, dtype=float) / float(wavelength_m)


def radians_to_cycles(phase_rad: np.ndarray) -> np.ndarray:
    """Convert phase from radians to cycles.

    Args:
        phase_rad: Phase values in radians.

    Returns:
        Phase values in cycles.
    """
    # Paper: Methodology Section II.2 uses integer-cycle jumps
    # for the TVD cycle-slip detector; 2*pi radians correspond to one cycle.
    return np.asarray(phase_rad, dtype=float) / (2.0 * np.pi)
