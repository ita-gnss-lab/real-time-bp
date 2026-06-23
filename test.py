from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from real_time_bp.methodology.constants import radians_to_cycles
from real_time_bp.methodology.signal_model import (
    detrended_intensity_db,
    estimated_scintillation_phase_rad,
    true_refractive_phase_rad,
)
from real_time_bp.methodology.tvd import (
    find_fade_minima,
    quantize_tvd_cycle_bias,
    tvd_weights,
    weighted_tvd_fit,
)
from real_time_bp.utils.io import find_dataset_paths, iter_case_dataframes
from real_time_bp.utils.params import load_params
from real_time_bp.utils.stage import l1_wavelength_from_params

params = load_params()
paths = find_dataset_paths(params)
wavelength_m = l1_wavelength_from_params(params)
tvd_cfg = params.get("tvd", {})
case_id = 1700

metadata = None
case_df = None
for meta, df in iter_case_dataframes(paths, chunksize=75_000):
    if int(meta.get("tracking_case_id", -1)) == case_id:
        metadata = meta
        case_df = df.sort_values("time_sec").reset_index(drop=True)
        break
if case_df is None:
    raise SystemExit(f"case {case_id} not found")

time_sec = case_df["time_sec"].to_numpy(dtype=float)
time_rel = time_sec - time_sec[0]
intensity_db = detrended_intensity_db(case_df)
estimated_phase = estimated_scintillation_phase_rad(case_df, wavelength_m)
true_refractive = true_refractive_phase_rad(case_df, wavelength_m)
diffractive_phase = estimated_phase - true_refractive
diffractive_cycles = radians_to_cycles(diffractive_phase)
los_removed_cycles = (
    case_df["carrier_phase_est_m"].to_numpy(dtype=float)
    - case_df["los_phase_m"].to_numpy(dtype=float)
) / wavelength_m
true_refractive_cycles = radians_to_cycles(true_refractive)

fade_minima = find_fade_minima(intensity_db, float(tvd_cfg.get("fade_threshold_db", -15.0)))
weights = tvd_weights(
    diffractive_cycles.size,
    fade_minima,
    float(tvd_cfg.get("allowed_jump_weight", 1.0)),
    float(tvd_cfg.get("blocked_jump_weight", 10000.0)),
)
tvd_fit = weighted_tvd_fit(diffractive_cycles, weights)
integer_bias = quantize_tvd_cycle_bias(tvd_fit)
valid_minima = fade_minima[fade_minima < integer_bias.size - 1]
jumps = integer_bias[valid_minima + 1] - integer_bias[valid_minima]
slip_indices = valid_minima[jumps != 0]


def slope(y):
    finite = np.isfinite(time_rel) & np.isfinite(y)
    return (
        float(np.polyfit(time_rel[finite], y[finite], 1)[0]) if finite.sum() >= 2 else float("nan")
    )


phase_slope = slope(diffractive_cycles)
los_slope = slope(los_removed_cycles)
span = float(np.nanmax(diffractive_cycles) - np.nanmin(diffractive_cycles))

out_dir = Path("results/figures/debug")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"tvd_fit_case_{case_id}_doppler_corrected.png"

fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True, constrained_layout=True)
fig.suptitle(
    "TVD input, case {case_id} ({severity}, C/N0={cn0:.1f} dB-Hz, init Doppler={doppler:.3f} Hz)".format(
        case_id=case_id,
        severity=metadata.get("severity"),
        cn0=float(metadata.get("input_c_n0_db_hz")),
        doppler=float(metadata.get("init_doppler_error_hz")),
    )
)

axes[0].plot(
    time_rel,
    los_removed_cycles - los_removed_cycles[0],
    color="0.7",
    lw=1.0,
    label=f"LOS removed only, slope={los_slope:.6f} cycles/s",
)
axes[0].plot(
    time_rel,
    diffractive_cycles - diffractive_cycles[0],
    color="tab:blue",
    lw=1.2,
    label=f"minus true refractive, slope={phase_slope:.6f} cycles/s",
)
axes[0].set_ylabel("phase change [cycles]")
axes[0].legend(loc="best")

axes[1].plot(
    time_rel, diffractive_cycles, color="0.65", lw=0.9, label="corrected diffractive phase"
)
axes[1].step(time_rel, tvd_fit, where="post", color="tab:red", lw=1.2, label="weighted TVD fit")
axes[1].step(
    time_rel, integer_bias, where="post", color="0.1", lw=1.1, label="rounded integer bias"
)
if slip_indices.size:
    axes[1].scatter(
        time_rel[slip_indices],
        integer_bias[slip_indices],
        s=36,
        color="tab:purple",
        label="reported slips",
        zorder=5,
    )
axes[1].set_ylabel("cycles")
axes[1].set_title(
    f"diffractive span={span:.3f} cycles, fade minima={fade_minima.size}, reported slips={slip_indices.size}"
)
axes[1].legend(loc="best")

axes[2].plot(time_rel, intensity_db, color="tab:blue", lw=0.8)
axes[2].axhline(float(tvd_cfg.get("fade_threshold_db", -15.0)), color="0.25", ls="--", lw=1.0)
if fade_minima.size:
    axes[2].scatter(
        time_rel[fade_minima],
        intensity_db[fade_minima],
        s=20,
        color="tab:orange",
        label="fade minima",
        zorder=5,
    )
if slip_indices.size:
    axes[2].scatter(
        time_rel[slip_indices],
        intensity_db[slip_indices],
        s=32,
        color="tab:purple",
        label="reported slips",
        zorder=6,
    )
axes[2].set_ylabel("intensity [dB]")
axes[2].legend(loc="best")

jump_series = np.zeros_like(integer_bias, dtype=float)
if valid_minima.size:
    jump_series[valid_minima] = jumps
axes[3].stem(time_rel, jump_series, linefmt="tab:purple", markerfmt="o", basefmt="0.7")
axes[3].set_ylabel("jump [cycles]")
axes[3].set_xlabel("time since case start [s]")

for ax in axes:
    ax.grid(True, alpha=0.3)

fig.savefig(out_path, dpi=160)
plt.close(fig)
print(out_path)
print(
    "metadata",
    {
        k: metadata.get(k)
        for k in [
            "tracking_case_id",
            "severity",
            "input_c_n0_db_hz",
            "init_doppler_error_hz",
            "loss_of_lock",
        ]
    },
)
print("slope_los_removed_cycles_per_s", los_slope)
print("slope_diffractive_cycles_per_s", phase_slope)
print("span_diffractive_cycles", span)
print("fade_minima", fade_minima.size, "reported_slips", slip_indices.size)
