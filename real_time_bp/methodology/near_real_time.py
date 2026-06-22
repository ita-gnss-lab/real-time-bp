from __future__ import annotations

import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR

from real_time_bp.methodology.backpropagation import (
    inverse_propagation_kernel,
    sliding_window_backpropagate_value_with_kernel,
)


def select_var_order_bic(features: np.ndarray, min_order: int, max_order: int) -> tuple[int, object]:
    # Paper: Methodology Section II.5, Eqs. (28) and (29). The real/imaginary
    # scintillation features are fit with VAR(p), and p is selected by the BIC.
    best_order = int(min_order)
    best_result = None
    best_bic = float("inf")
    model = VAR(np.asarray(features, dtype=float))
    for order in range(int(min_order), int(max_order) + 1):
        try:
            result = model.fit(maxlags=order, ic=None, trend="c")
        except Exception:
            continue
        if np.isfinite(result.bic) and float(result.bic) < best_bic:
            best_bic = float(result.bic)
            best_order = int(result.k_ar)
            best_result = result
    if best_result is None:
        best_result = model.fit(maxlags=int(min_order), ic=None, trend="c")
        best_order = int(best_result.k_ar)
    return best_order, best_result


def forecast_complex_tail(var_result: object, history: np.ndarray, steps: int) -> np.ndarray:
    # Paper: Methodology Eq. (26); this supplies the unavailable
    # future samples psi_hat_{k,L}[r] when latency L is smaller than the full
    # noncausal future context.
    if int(steps) <= 0:
        return np.array([], dtype=complex)
    forecast = var_result.forecast(np.asarray(history, dtype=float), steps=int(steps))
    return forecast[:, 0] + 1j * forecast[:, 1]


def sliding_window_bp_with_var_forecast(
    scintillation_signal: np.ndarray,
    sample_interval_s: float,
    fresnel_time_scale_s: float,
    block_len: int,
    latency_samples: int,
    eval_indices: np.ndarray,
    var_result: object,
) -> np.ndarray:
    # Paper: Methodology Eq. (26). For each evaluation sample k,
    # build an M-sample block centered at M_c=floor(M/2), use measured samples
    # up to k+L, forecast the rest, and retain only the center BP output.
    signal = np.asarray(scintillation_signal, dtype=complex)
    features = np.column_stack([np.real(signal), np.imag(signal)])
    half = int(block_len) // 2
    future_context = int(block_len) - half - 1
    latency = int(np.clip(latency_samples, 0, future_context))
    forecast_steps = future_context - latency
    result = np.full(signal.size, np.nan + 1j * np.nan, dtype=complex)
    k_ar = int(var_result.k_ar)
    inverse_kernel = inverse_propagation_kernel(
        int(block_len),
        sample_interval_s,
        fresnel_time_scale_s,
    )

    for k in np.asarray(eval_indices, dtype=int):
        start = k - half
        observed_stop = k + latency
        block_stop = k + future_context
        if start < 0 or block_stop >= signal.size or observed_stop + 1 - k_ar < 0:
            continue
        prefix = signal[start : observed_stop + 1]
        if forecast_steps:
            # Paper: Methodology Eq. (29). The VAR model sees
            # only the history available at the delayed decision time k+L.
            history = features[observed_stop + 1 - k_ar : observed_stop + 1]
            tail = forecast_complex_tail(var_result, history, forecast_steps)
            block = np.concatenate([prefix, tail])
        else:
            block = signal[start : block_stop + 1]
        if block.size == int(block_len):
            result[k] = sliding_window_backpropagate_value_with_kernel(
                block,
                inverse_kernel,
                half,
            )
    return result


def full_noncausal_sliding_window_bp(
    scintillation_signal: np.ndarray,
    sample_interval_s: float,
    fresnel_time_scale_s: float,
    block_len: int,
    eval_indices: np.ndarray,
) -> np.ndarray:
    # Paper: Methodology Section II.5. L=M-M_c-1 is the limiting case
    # where the full future half-window is available, so no forecast is needed.
    half = int(block_len) // 2
    return sliding_window_bp_with_var_forecast(
        scintillation_signal,
        sample_interval_s,
        fresnel_time_scale_s,
        block_len,
        int(block_len) - half - 1,
        eval_indices,
        _PassthroughVar(scintillation_signal),
    )


class _PassthroughVar:
    # Helper for the full-noncausal limit above; it satisfies the VAR interface
    # but should not be interpreted as a model used in the Results section.
    def __init__(self, signal: np.ndarray):
        self.k_ar = 1
        self._features = np.column_stack([np.real(signal), np.imag(signal)])

    def forecast(self, history: np.ndarray, steps: int) -> np.ndarray:
        return np.repeat(history[-1:, :], int(steps), axis=0)
