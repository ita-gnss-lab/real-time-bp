from __future__ import annotations

import numpy as np
from scipy.fft import fft, ifft


def propagation_kernel(block_len: int, sample_interval_s: float, fresnel_time_scale_s: float) -> np.ndarray:
    # Paper: Methodology Eq. (3). This is the all-pass
    # dispersive phase-screen propagation kernel H[f_n, T_F].
    frequency_hz = np.fft.fftfreq(int(block_len), d=float(sample_interval_s))
    return np.exp(-0.5j * (2.0 * np.pi * frequency_hz * float(fresnel_time_scale_s)) ** 2)


def inverse_propagation_kernel(
    block_len: int,
    sample_interval_s: float,
    fresnel_time_scale_s: float,
) -> np.ndarray:
    # Paper: Methodology Eq. (5). Radio BP is a matched filter, so it
    # applies the conjugate kernel H*[f_n, T_F_hat].
    return np.conj(propagation_kernel(block_len, sample_interval_s, fresnel_time_scale_s))


def batch_backpropagate(
    scintillation_signal: np.ndarray,
    sample_interval_s: float,
    fresnel_time_scale_s: float,
) -> np.ndarray:
    # Paper: Methodology Eqs. (4)-(5).
    # This full-record batch operation is the noncausal reference used in
    # Results Section III.2 and Section III.3.
    signal = np.asarray(scintillation_signal, dtype=complex)
    kernel = inverse_propagation_kernel(signal.size, sample_interval_s, fresnel_time_scale_s)
    return ifft(fft(signal) * kernel)


def sliding_window_backpropagate_value(
    block: np.ndarray,
    sample_interval_s: float,
    fresnel_time_scale_s: float,
    center_index: int,
) -> complex:
    # Paper: Methodology Eq. (26). The windowed method keeps
    # only the center sample M_c after applying the same inverse BP kernel.
    kernel = inverse_propagation_kernel(len(block), sample_interval_s, fresnel_time_scale_s)
    return sliding_window_backpropagate_value_with_kernel(block, kernel, center_index)


def sliding_window_backpropagate_value_with_kernel(
    block: np.ndarray,
    inverse_kernel: np.ndarray,
    center_index: int,
) -> complex:
    # Paper: Methodology Eq. (26). When M and T_F are fixed, the inverse
    # propagation kernel is shared across all sliding-window centers.
    kernel = np.asarray(inverse_kernel, dtype=complex)
    return ifft(fft(block) * kernel)[center_index]


def unwrap_phase(z: np.ndarray) -> np.ndarray:
    # Paper: Methodology Eq. (7). We use
    # Itoh-style unwrapping (numpy.unwrap), as stated in Section II.1.
    values = np.asarray(z)
    phase = np.full(values.shape, np.nan, dtype=float)
    mask = np.isfinite(values)
    if np.any(mask):
        phase[mask] = np.unwrap(np.angle(values[mask]))
    return phase


def median_align_phase(phase: np.ndarray, reference: np.ndarray, start_index: int = 0) -> np.ndarray:
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
