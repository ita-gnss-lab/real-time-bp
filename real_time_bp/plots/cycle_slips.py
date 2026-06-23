"""Cycle-slip and representative phase-retrieval figure generation."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from real_time_bp.plots.common import amplitude_only, column, placeholder, read_csv, remove_figure
from real_time_bp.utils.params import load_params
from real_time_bp.utils.plot_style import COLORS, grid_figure_size
from real_time_bp.utils.representatives import SEVERITY_ORDER, bool_series
from real_time_bp.utils.stage import figure_stage_dir


def plot_cycle_slips() -> None:
    """Render the combined TVD and representative phase-retrieval figure.

    Main paper:
        Results Figure 1 diagnostic support visualizes the TVD integer
        cycle-bias trace beside the fade minima that constrain allowed jumps.
    """
    out_dir = figure_stage_dir("02_cycle_slips")
    filter_dir = figure_stage_dir("03_filter_bank")
    params = load_params()
    threshold_db = float(params.get("tvd", {}).get("fade_threshold_db", -10.0))
    trace = read_csv("02_cycle_slips/tvd_example_traces.csv")
    filter_trace = amplitude_only(
        read_csv("03_filter_bank/filter_bank_representative_phase_traces.csv")
    )
    if trace.empty:
        placeholder(out_dir / "tvd_example_traces.pdf", "No TVD example trace data")
        remove_figure(filter_dir / "representative_phase_retrieval.pdf")
        return
    if "loss_of_lock" in trace.columns:
        no_lock_trace = trace[~bool_series(trace["loss_of_lock"])]
        if not no_lock_trace.empty:
            trace = no_lock_trace
    severities = list(SEVERITY_ORDER)
    if not any(str(value) in severities for value in trace["severity"].dropna().unique()):
        placeholder(out_dir / "tvd_example_traces.pdf", "No no-lock TVD example traces")
        remove_figure(filter_dir / "representative_phase_retrieval.pdf")
        return
    fig, axes = plt.subplots(
        3,
        len(severities) + 1,
        figsize=grid_figure_size(len(severities) + 1, 3, panel_height_cm=5.0),
        sharex="col",
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0] * len(severities) + [0.65]},
        squeeze=False,
    )
    legend_axes = axes[:, -1]
    for ax in legend_axes:
        ax.set_axis_off()
    intensity_values = pd.to_numeric(trace["intensity_db"], errors="coerce").dropna()
    if intensity_values.empty:
        intensity_ylim = None
    else:
        margin = 0.05 * max(float(intensity_values.max() - intensity_values.min()), 1.0)
        intensity_ylim = (
            float(intensity_values.min() - margin),
            float(intensity_values.max() + margin),
        )
    for col, severity in enumerate(severities):
        group = trace[trace["severity"].astype(str).eq(severity)]
        if group.empty:
            for ax in axes[:, col]:
                ax.text(0.5, 0.5, "No no-lock example", ha="center", va="center")
                ax.set_axis_off()
            axes[0, col].set_title(f"{severity} - no loss of lock")
            continue
        group = group[group["tracking_case_id"].eq(group["tracking_case_id"].iloc[0])]
        intensity_ax = axes[0, col]
        phase_ax = axes[1, col]
        error_ax = axes[2, col]
        filter_group = pd.DataFrame()
        if not filter_trace.empty:
            filter_group = filter_trace[
                filter_trace["tracking_case_id"].eq(group["tracking_case_id"].iloc[0])
            ]
            if filter_group.empty:
                filter_group = filter_trace[filter_trace["severity"].astype(str).eq(severity)]
                if not filter_group.empty:
                    filter_group = filter_group[
                        filter_group["tracking_case_id"].eq(
                            filter_group["tracking_case_id"].iloc[0]
                        )
                    ]
        intensity_ax.plot(
            group["time_sec"],
            group["intensity_db"],
            color=COLORS["truth"],
            linewidth=1.1,
            label="Detrended intensity",
        )
        minima = group[bool_series(group["is_fade_minimum"])]
        intensity_ax.scatter(
            minima["time_sec"],
            minima["intensity_db"],
            color=COLORS["truth"],
            s=9,
            alpha=0.65,
            label="Fade minima",
        )
        intensity_ax.axhline(
            threshold_db,
            color=COLORS["neutral"],
            linestyle=":",
            linewidth=1,
            label=rf"{abs(threshold_db):g} dB threshold",
        )
        intensity_ax.set_title(str(severity))
        intensity_ax.set_ylabel("Intensity [dB]")
        if intensity_ylim is not None:
            intensity_ax.set_ylim(*intensity_ylim)

        estimated_col = (
            "estimated_diffractive_phase_cycles"
            if "estimated_diffractive_phase_cycles" in group
            else "diffractive_phase_cycles"
        )
        if not filter_group.empty:
            true_scintillation_col = column(
                filter_group,
                "true_scintillation_phase_rad",
                "true_refractive_phase_rad",
            )
            estimated_scintillation_col = column(
                filter_group,
                "estimated_scintillation_phase_rad",
                "true_refractive_phase_rad",
            )
            batch_col = column(
                filter_group,
                "backpropagated_refractive_phase_rad",
                "backpropagated_phase_rad",
            )
            phase_ax.plot(
                filter_group["time_sec"],
                filter_group[true_scintillation_col],
                color=COLORS["true_scintillation"],
                linewidth=1.1,
                label=r"$\phi_s$",
            )
            phase_ax.plot(
                filter_group["time_sec"],
                filter_group[estimated_scintillation_col],
                color=COLORS["estimated_scintillation"],
                linestyle=":",
                linewidth=1.0,
                label=r"$\hat{\phi}_s$",
            )
            phase_ax.plot(
                filter_group["time_sec"],
                filter_group["true_refractive_phase_rad"],
                color=COLORS["truth"],
                linewidth=1.1,
                label=r"$\phi_r$",
            )
            phase_ax.plot(
                filter_group["time_sec"],
                filter_group[batch_col],
                color=COLORS["batch"],
                linestyle="--",
                linewidth=1.0,
                label=r"$\hat{\phi}_{r,\mathrm{batch}}$",
            )
        else:
            phase_ax.plot(
                group["time_sec"],
                2.0 * np.pi * group[estimated_col],
                color=COLORS["truth"],
                linewidth=1.0,
                label=r"$\hat{\phi}_s-\phi_r$",
            )
        phase_ax.set_ylabel("Phase [rad]")
        estimated_diff = _zero_reference_cycles(group[estimated_col])
        error_ax.plot(
            group["time_sec"],
            estimated_diff,
            color=COLORS["truth"],
            linewidth=1,
            label=r"$\hat{\phi}_s-\phi_r$",
        )
        if "true_diffractive_phase_cycles" in group:
            true_diff = _zero_reference_cycles(group["true_diffractive_phase_cycles"])
            error_ax.plot(
                group["time_sec"],
                true_diff,
                color=COLORS["true_scintillation"],
                linestyle="--",
                linewidth=1,
                label=r"$\phi_s-\phi_r$",
            )
        integer_cycle_bias = _zero_reference_cycles(group["integer_cycle_bias"])
        error_ax.step(
            group["time_sec"],
            integer_cycle_bias,
            where="mid",
            color=COLORS["tvd"],
            linestyle=":",
            linewidth=1.2,
            label=r"TVD $\hat{\phi}_s-\phi_r$",
        )
        if not filter_group.empty:
            batch_error_col = column(
                filter_group,
                "batch_refractive_phase_error_cycles",
                "phase_error_rad",
            )
            batch_error = filter_group[batch_error_col]
            if batch_error_col == "phase_error_rad":
                batch_error = batch_error / (2.0 * np.pi)
            batch_error = _zero_reference_cycles(batch_error)
            error_ax.plot(
                filter_group["time_sec"],
                batch_error,
                color=COLORS["batch"],
                linewidth=1.0,
                label=r"$\hat{\phi}_{r,\mathrm{batch}}-\phi_r$",
            )
            if "batch_integer_cycle_bias" in filter_group:
                batch_integer_cycle_bias = _zero_reference_cycles(
                    filter_group["batch_integer_cycle_bias"]
                )
                error_ax.step(
                    filter_group["time_sec"],
                    batch_integer_cycle_bias,
                    where="mid",
                    color=COLORS["tvd"],
                    linestyle=":",
                    linewidth=1.2,
                    label=r"TVD $\hat{\phi}_{r,\mathrm{batch}}-\phi_r$",
                )
        error_ax.set_xlabel("Time [sec]")
        error_ax.set_ylabel("Phase differences [cycles]")
    handles, labels = _unique_legend_items(axes[:, :-1].ravel())
    if handles:
        legend_axes[1].legend(handles, labels, loc="center left", frameon=False)
    fig.savefig(out_dir / "tvd_example_traces.pdf")
    remove_figure(filter_dir / "representative_phase_retrieval.pdf")
    plt.close(fig)


def _zero_reference_cycles(values) -> np.ndarray:
    series = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(series)
    if np.any(finite):
        series = series - series[np.flatnonzero(finite)[0]]
    return series


def _unique_legend_items(axes) -> tuple[list, list[str]]:
    handles = []
    labels = []
    seen = set()
    for ax in axes:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels, strict=True):
            if label in seen:
                continue
            seen.add(label)
            handles.append(handle)
            labels.append(label)
    return handles, labels
