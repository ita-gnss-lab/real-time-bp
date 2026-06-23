"""Backpropagation kernels and phase-alignment utilities for the paper pipeline."""

from __future__ import annotations

import numpy as np
from scipy.fft import fft, ifft


def propagation_kernel(
    block_len: int, sample_interval_s: float, fresnel_time_scale_s: float
) -> np.ndarray:
    """Build the forward phase-screen propagation kernel.

    The kernel follows the main paper methodology for the dispersive
    phase-screen operator as a function of the Fresnel time scale.

    Args:
        block_len: Number of samples in the FFT block.
        sample_interval_s: Sample spacing in seconds.
        fresnel_time_scale_s: Fresnel time scale used by the propagation model.

    Returns:
        Complex all-pass frequency-domain propagation kernel.
    """
    # Paper: Methodology Eq. (3). This is the all-pass
    # dispersive phase-screen propagation kernel H[f_n, T_F].
    frequency_hz = np.fft.fftfreq(int(block_len), d=float(sample_interval_s))
    return np.exp(-0.5j * (2.0 * np.pi * frequency_hz * float(fresnel_time_scale_s)) ** 2)


def inverse_propagation_kernel(
    block_len: int,
    sample_interval_s: float,
    fresnel_time_scale_s: float,
) -> np.ndarray:
    """Build the inverse backpropagation kernel.

    Args:
        block_len: Number of samples in the FFT block.
        sample_interval_s: Sample spacing in seconds.
        fresnel_time_scale_s: Fresnel time scale used by the propagation model.

    Returns:
        Complex conjugate kernel used for radio backpropagation.
    """
    # Paper: Methodology Eq. (5). Radio BP is a matched filter, so it
    # applies the conjugate kernel H*[f_n, T_F_hat].
    return np.conj(propagation_kernel(block_len, sample_interval_s, fresnel_time_scale_s))


def batch_backpropagate(
    scintillation_signal: np.ndarray,
    sample_interval_s: float,
    fresnel_time_scale_s: float,
) -> np.ndarray:
    """Apply noncausal batch backpropagation to a complete scintillation signal.

    Args:
        scintillation_signal: Complex tracked scintillation signal.
        sample_interval_s: Sample spacing in seconds.
        fresnel_time_scale_s: Fresnel time scale used by the inverse kernel.

    Returns:
        Complex screen-domain field estimate for the full record.
    """
    # Paper: Methodology Eqs. (4)-(5).
    # This full-record batch operation is the noncausal reference used in
    # Results Section III.2 and Section III.3.
    signal = np.asarray(scintillation_signal, dtype=complex)
    kernel = inverse_propagation_kernel(signal.size, sample_interval_s, fresnel_time_scale_s)
    return ifft(fft(signal) * kernel)


def sliding_window_backpropagate_value_with_kernel(
    block: np.ndarray,
    inverse_kernel: np.ndarray,
    center_index: int,
) -> complex:
    """Backpropagate one block and retain the configured center sample.

    Args:
        block: Complex input samples for one local FFT window.
        inverse_kernel: Precomputed inverse propagation kernel for the block.
        center_index: Index of the sample retained from the inverse FFT result.

    Returns:
        Complex backpropagated sample at `center_index`.
    """
    # Paper: Methodology Eq. (18). When M and T_F are fixed, the inverse
    # propagation kernel is shared across all sliding-window centers.
    kernel = np.asarray(inverse_kernel, dtype=complex)
    return ifft(fft(block) * kernel)[center_index]


def unwrap_phase(z: np.ndarray) -> np.ndarray:
    """Unwrap the phase of finite complex samples while preserving NaNs.

    Args:
        z: Complex-valued samples whose phase should be unwrapped.

    Returns:
        Real-valued phase array with NaNs where the input was not finite.
    """
    # Paper: Methodology Eq. (7). We use
    # Itoh-style unwrapping (numpy.unwrap), as stated in Section II.1.
    values = np.asarray(z)
    phase = np.full(values.shape, np.nan, dtype=float)
    mask = np.isfinite(values)
    if np.any(mask):
        phase[mask] = np.unwrap(np.angle(values[mask]))
    return phase


def median_align_phase(
    phase: np.ndarray, reference: np.ndarray, start_index: int = 0
) -> np.ndarray:
    """Remove the constant phase ambiguity using a median reference offset.

    Args:
        phase: Estimated phase trace in radians.
        reference: Reference phase trace in radians.
        start_index: First sample included when estimating the constant offset.

    Returns:
        Phase trace shifted by the median finite error after `start_index`.
    """
    # Paper/results convention: BP recovers the screen phase up to an arbitrary
    # constant phase offset, so metrics align the median error after N_init.
    phase = np.asarray(phase, dtype=float)
    reference = np.asarray(reference, dtype=float)
    aligned = phase.copy()
    mask = np.isfinite(phase) & np.isfinite(reference)
    mask[:start_index] = False
    if np.any(mask):
        aligned -= np.median(phase[mask] - reference[mask])
    return aligned
