"""Sliding figure generation from stage CSV outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from real_time_bp.plots.common import placeholder, read_csv, save_figure
from real_time_bp.utils.params import load_params
from real_time_bp.utils.plot_style import (
    COLORS,
    CONFIG_CMAP,
    FONT_SIZES_PT,
    HEATMAP_CMAP,
    SEVERITY_COLORS,
    figsize_cm,
    figure_size,
)
from real_time_bp.utils.representatives import SEVERITY_ORDER, severity_groups
from real_time_bp.utils.stage import figure_stage_dir


def plot_sliding() -> None:
    """Render VAR, latency, practical-metric, and forecast figures.

    Main paper:
        Results Section III.3, Figures 9-13, defines this plot group for VAR
        order distribution, practical metrics, M/L heatmap, and example
        forecasts.
    """
    out_dir = figure_stage_dir("04_sliding")
    var_orders = read_csv("04_sliding/sliding_var_orders.csv")
    practical = read_csv("04_sliding/sliding_practical_metrics.csv")
    heatmap = read_csv("04_sliding/sliding_latency_heatmap.csv")
    phase = read_csv("04_sliding/sliding_phase_example.csv")
    forecast = read_csv("04_sliding/sliding_forecast_example.csv")

    _plot_var_orders(var_orders, out_dir / "var_order_histogram.pdf")
    _plot_practical_metrics(practical, out_dir / "batch_vs_sliding_metrics.pdf")
    _plot_latency_heatmap(heatmap, out_dir / "latency_heatmap.pdf")
    _plot_phase_example(phase, out_dir / "sliding_phase_example.pdf")
    _plot_forecast_example(forecast, out_dir / "var_forecast_example.pdf")


def _plot_var_orders(df: pd.DataFrame, path: Path) -> None:
    """Plot the histogram of BIC-selected VAR orders.

    Main paper:
        Results Figure 10 reports the selected VAR orders over scintillation
        severities.
    """
    if df.empty:
        placeholder(path, "No VAR order data")
        return
    fig, ax = plt.subplots(figsize=figure_size("single"))
    for severity, group in severity_groups(df):
        weights = np.full(group.shape[0], 100.0 / max(group.shape[0], 1))
        ax.hist(
            group["var_order_bic"],
            bins=np.arange(0.5, 25.5, 1),
            weights=weights,
            alpha=0.45,
            color=SEVERITY_COLORS.get(str(severity), COLORS["sliding"]),
            label=severity,
        )
    ax.set_xlabel("BIC-selected VAR order")
    ax.set_ylabel("Relative Frequency")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.legend()
    save_figure(fig, path)


def _plot_practical_metrics(df: pd.DataFrame, path: Path) -> None:
    """Plot batch BP versus sliding BP practical metrics.

    Main paper:
        Results Figure 12 compares the noncausal batch BP reference with the
        practical sliding BP operating point. The first panel reports the
        mean across cases using MRMSE_phi_r^(a), where
        a is batch or sliding, and MRMSE_phi_s for the tracked baseline before
        BP.
        The second panel reports TVD-derived cycle-slip rates.
    """
    if df.empty:
        placeholder(path, "No practical sliding metric data")
        return
    fig, axes = plt.subplots(1, 2, figsize=figure_size("two_panel"), sharex=False)
    batch_col = "batch_rmse_phi_r_rad"
    sliding_col = "sliding_rmse_phi_r_rad"
    tracked_col = "tracked_rmse_phi_s_rad"
    batch_cs_col = "batch_cs_per_min"
    sliding_cs_col = "sliding_cs_per_min"
    tracked_cs_col = "tracked_cs_per_min"
    n_value = (
        int(df["n_init_samples"].dropna().iloc[0])
        if "n_init_samples" in df and df["n_init_samples"].notna().any()
        else None
    )
    m_value = (
        int(df["block_len"].dropna().iloc[0])
        if "block_len" in df and df["block_len"].notna().any()
        else None
    )
    l_value = (
        int(df["latency_samples"].dropna().iloc[0])
        if "latency_samples" in df and df["latency_samples"].notna().any()
        else None
    )
    for severity, group in severity_groups(df):
        metric_cols = [
            batch_col,
            sliding_col,
            tracked_col,
            batch_cs_col,
            sliding_cs_col,
            tracked_cs_col,
        ]
        clean = _mean_by_c_n0(group, metric_cols)
        color = SEVERITY_COLORS.get(str(severity), COLORS["sliding"])
        axes[0].plot(
            clean["input_c_n0_db_hz"],
            clean[tracked_col],
            color=color,
            linestyle=":",
            linewidth=1.2,
            label=rf"{severity} $\mathrm{{MRMSE}}_{{\phi_s}}$",
        )
        axes[0].plot(
            clean["input_c_n0_db_hz"],
            clean[batch_col],
            color=color,
            linestyle="--",
            alpha=0.75,
            label=rf"{severity} $\mathrm{{MRMSE}}_{{\phi_r}}^{{(\mathrm{{batch}})}}$",
        )
        axes[0].plot(
            clean["input_c_n0_db_hz"],
            clean[sliding_col],
            color=color,
            label=rf"{severity} $\mathrm{{MRMSE}}_{{\phi_r}}^{{(\mathrm{{sliding}})}}$",
        )
        axes[1].plot(
            clean["input_c_n0_db_hz"],
            clean[batch_cs_col],
            color=color,
            linestyle="--",
            alpha=0.75,
            label=rf"{severity} batch BP",
        )
        axes[1].plot(
            clean["input_c_n0_db_hz"],
            clean[sliding_cs_col],
            color=color,
            label=rf"{severity} sliding BP",
        )
        axes[1].plot(
            clean["input_c_n0_db_hz"],
            clean[tracked_cs_col],
            color=color,
            linestyle=":",
            linewidth=1.2,
            label=rf"{severity} TVD $\hat{{\phi}}_s-\phi_r$",
        )
    axes[0].set_xlabel(r"$C/N_0$ [dB-Hz]")
    axes[0].set_ylabel(r"$\mathrm{MRMSE}$ [rad]")
    axes[1].set_xlabel(r"$C/N_0$ [dB-Hz]")
    axes[1].set_ylabel("CS/min")
    for ax in axes:
        if n_value is not None and m_value is not None and l_value is not None:
            ax.set_title(rf"$N_{{\mathrm{{init}}}}={n_value}$, $M={m_value}$, $L={l_value}$")
    axes[0].legend()
    axes[1].legend()
    save_figure(fig, path)


def _plot_latency_heatmap(df: pd.DataFrame, path: Path) -> None:
    """Plot the latency and FFT-length heatmap.

    Main paper:
        Results Figure 11 evaluates FFT length M and latency L over the valid
        triangular domain; invalid cells are absent from the heatmap.
    """
    if df.empty:
        placeholder(path, "No latency heatmap data")
        return
    metric_col = "refractive_phase_rmse_rad"
    x_col = "latency_fraction"
    severities = [
        severity for severity in SEVERITY_ORDER if severity in set(df["severity"].astype(str))
    ]
    metric_values = pd.to_numeric(df[metric_col], errors="coerce")
    cs_values = pd.to_numeric(df["cycle_slips_per_minute"], errors="coerce")
    metric_vmin, metric_vmax = float(metric_values.min()), float(metric_values.max())
    cs_vmin, cs_vmax = float(cs_values.min()), float(cs_values.max())
    fig, axes = plt.subplots(
        2,
        len(severities),
        figsize=figure_size("heatmap_grid"),
        sharex=False,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    metric_image = None
    cs_image = None
    for col, severity in enumerate(severities):
        severity_df = df[df["severity"].astype(str).eq(severity)]
        metric_group = severity_df.groupby(["block_len", x_col], as_index=False)[metric_col].mean()
        cs_group = severity_df.groupby(["block_len", x_col], as_index=False)[
            "cycle_slips_per_minute"
        ].mean()
        metric_pivot = metric_group.pivot(index="block_len", columns=x_col, values=metric_col)
        cs_pivot = cs_group.pivot(
            index="block_len", columns=x_col, values="cycle_slips_per_minute"
        )
        for row, (pivot, vmin, vmax, title) in enumerate(
            [
                (
                    metric_pivot,
                    metric_vmin,
                    metric_vmax,
                    r"$\mathrm{MRMSE}_{\phi_r}^{(\mathrm{sliding})}$ [rad]",
                ),
                (cs_pivot, cs_vmin, cs_vmax, "CS/min"),
            ]
        ):
            ax = axes[row, col]
            image = ax.imshow(
                pivot.to_numpy(),
                aspect="auto",
                origin="lower",
                cmap=HEATMAP_CMAP,
                vmin=vmin,
                vmax=vmax,
            )
            if row == 0:
                metric_image = image
                ax.set_title(str(severity).capitalize())
            else:
                cs_image = image
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels(pivot.index.astype(int))
            xticks = np.linspace(0, len(pivot.columns) - 1, min(6, len(pivot.columns)), dtype=int)
            ax.set_xticks(xticks)
            if x_col == "latency_fraction":
                ax.set_xticklabels(
                    [f"{float(value):.2f}" for value in pivot.columns[xticks]], rotation=45
                )
            else:
                ax.set_xticklabels(pivot.columns[xticks].astype(int), rotation=45)
            if col == 0:
                ax.set_ylabel("FFT length M [samples]")
            if row == 1:
                ax.set_xlabel(
                    r"Latency fraction $L/(M-M_c-1)$"
                    if x_col == "latency_fraction"
                    else "Latency [samples]"
                )
            ax.text(
                0.02,
                0.96,
                title,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=FONT_SIZES_PT["annotation"],
                color="black",
            )
    if metric_image is not None:
        fig.colorbar(
            metric_image,
            ax=axes[0, :],
            label=r"$\mathrm{MRMSE}_{\phi_r}^{(\mathrm{sliding})}$ [rad]",
            fraction=0.03,
        )
    if cs_image is not None:
        fig.colorbar(cs_image, ax=axes[1, :], label="CS/min", fraction=0.03)
    save_figure(fig, path, tight=False)


def _plot_phase_example(df: pd.DataFrame, path: Path) -> None:
    """Plot representative phase traces for batch and sliding BP.

    Main paper:
        Results Figure 13 compares truth, batch BP, and sliding BP for
        representative cases.
    """
    if df.empty:
        placeholder(path, "No phase example data")
        return
    severities = [
        severity for severity in SEVERITY_ORDER if severity in set(df["severity"].astype(str))
    ]
    fig, axes = plt.subplots(
        1,
        len(severities) + 1,
        figsize=figsize_cm(8.4 * len(severities) + 3.8, 6.0),
        gridspec_kw={"width_ratios": [1.0] * len(severities) + [0.45]},
        sharex=False,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    plot_axes = axes[:-1]
    legend_ax = axes[-1]
    legend_ax.set_axis_off()
    legend_title = None
    for col, severity in enumerate(severities):
        ax = plot_axes[col]
        severity_group = df[df["severity"].astype(str).eq(severity)]
        case_id = severity_group["tracking_case_id"].iloc[0]
        severity_group = severity_group[severity_group["tracking_case_id"].eq(case_id)]
        if "block_len" in severity_group:
            severity_group = severity_group[
                pd.to_numeric(severity_group["block_len"], errors="coerce").eq(2048)
            ]
        if "latency_samples" in severity_group:
            severity_group = severity_group[
                pd.to_numeric(severity_group["latency_samples"], errors="coerce").isin([0, 512])
            ]
        if severity_group.empty:
            ax.text(0.5, 0.5, "No M=2048 examples", ha="center", va="center")
            ax.set_axis_off()
            continue
        n_init_value = (
            int(severity_group["n_init_samples"].dropna().iloc[0])
            if "n_init_samples" in severity_group
            and severity_group["n_init_samples"].notna().any()
            else None
        )
        m_value = (
            int(severity_group["block_len"].dropna().iloc[0])
            if "block_len" in severity_group and severity_group["block_len"].notna().any()
            else None
        )
        if legend_title is None and n_init_value is not None and m_value is not None:
            legend_title = rf"$N_{{\mathrm{{init}}}}={n_init_value}$, $M={m_value}$"
        first_config = (
            severity_group["configuration"].iloc[0] if "configuration" in severity_group else None
        )
        base = (
            severity_group[severity_group["configuration"].eq(first_config)]
            if first_config is not None
            else severity_group
        )
        batch_error_cycles = base["batch_refractive_phase_error_rad"] / (2.0 * np.pi)
        if "estimated_scintillation_minus_refractive_rad" in base:
            diffractive_cycles = base["estimated_scintillation_minus_refractive_rad"] / (
                2.0 * np.pi
            )
            finite_diffractive = np.isfinite(diffractive_cycles)
            if finite_diffractive.any():
                diffractive_cycles = diffractive_cycles - float(
                    np.nanmedian(diffractive_cycles[finite_diffractive])
                )
            ax.plot(
                base["time_sec"],
                diffractive_cycles,
                color=COLORS["estimated_scintillation"],
                linestyle="--",
                linewidth=1.0,
                label=r"$\hat{\phi}_s-\phi_r$",
            )
        sliding_col = "sliding_refractive_phase_error_rad"
        config_groups = severity_group.sort_values(["latency_samples", "sample_index"]).groupby(
            "latency_samples", sort=True
        )
        config_items = list(config_groups)
        config_cmap = plt.get_cmap(CONFIG_CMAP)
        config_colors = config_cmap(np.linspace(0.2, 0.85, len(config_items)))
        finite_times = []
        for color, (latency, config_group) in zip(config_colors, config_items):
            phase_error_cycles = config_group[sliding_col] / (2.0 * np.pi)
            finite = np.isfinite(phase_error_cycles)
            finite_times.extend(config_group.loc[finite, "time_sec"].to_list())
            ax.plot(
                config_group["time_sec"],
                phase_error_cycles,
                color=color,
                linewidth=1.0,
                label=rf"$\hat{{\phi}}_{{r,\mathrm{{sliding}}}}-\phi_r$, $L={int(latency)}$",
            )
        ax.plot(
            base["time_sec"],
            batch_error_cycles,
            color=COLORS["truth"],
            linewidth=1.1,
            zorder=6,
            label=r"$\hat{\phi}_{r,\mathrm{batch}}-\phi_r$",
        )
        if finite_times:
            ax.set_xlim(min(finite_times), max(finite_times))
        ax.set_title(str(severity).capitalize())
        ax.set_xlabel("Time [sec]")
        ax.set_ylabel("Phase differences [cycles]")
    handles, labels = _unique_legend_items(plot_axes)
    if handles:
        legend_ax.legend(handles, labels, loc="center left", title=legend_title)
    save_figure(fig, path, tight=False)


def _plot_forecast_example(df: pd.DataFrame, path: Path) -> None:
    """Plot true, tracked, and VAR-forecasted scintillation samples.

    Main paper:
        Results Figure 9 compares true and VAR-forecasted real/imaginary
        scintillation samples for a representative case.
    """
    if df.empty:
        placeholder(path, "No forecast example data")
        return
    params = load_params()
    plot_cfg = params.get("plots", {})
    sliding_cfg = params.get("sliding", {})
    zoom_start_s = plot_cfg.get("forecast_zoom_start_s")
    zoom_end_s = plot_cfg.get("forecast_zoom_end_s")
    requested_latencies = [
        int(value) for value in sliding_cfg.get("forecast_example_latency_samples", [])
    ]
    severities = [severity for severity in ["weak", "strong"] if severity in set(SEVERITY_ORDER)]
    fig, axes = plt.subplots(
        2,
        len(severities) + 1,
        figsize=figsize_cm(8.0 * len(severities) + 3.8, 8.2),
        gridspec_kw={"width_ratios": [1.0] * len(severities) + [0.48]},
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )
    plot_axes = axes[:, :-1]
    legend_axes = axes[:, -1]
    for ax in legend_axes:
        ax.set_axis_off()
    for col, severity in enumerate(severities):
        group = _forecast_case_group(df[df["severity"].astype(str).eq(severity)])
        if group.empty:
            for ax in plot_axes[:, col]:
                ax.text(0.5, 0.5, f"No {severity} forecast data", ha="center", va="center")
                ax.set_axis_off()
            continue
        latencies = sorted(
            pd.to_numeric(group["latency_samples"], errors="coerce").dropna().unique()
        )
        selected_latencies = [latency for latency in requested_latencies if latency in latencies]
        if not selected_latencies:
            selected_latencies = [int(latency) for latency in latencies]
        group = group[
            pd.to_numeric(group["latency_samples"], errors="coerce").isin(selected_latencies)
        ].sort_values(["latency_samples", "sample_index"])
        truth = group.drop_duplicates("sample_index").sort_values("sample_index")
        x_col = "time_sec" if "time_sec" in truth.columns else "sample_index"
        if x_col == "time_sec":
            start_s, end_s = _forecast_xlim(group, truth, x_col, zoom_start_s, zoom_end_s)
            truth = truth[truth[x_col].between(start_s, end_s)]
            group = group[group[x_col].between(start_s, end_s)]
        if truth.empty or group.empty:
            for ax in plot_axes[:, col]:
                ax.text(0.5, 0.5, "No samples in zoom window", ha="center", va="center")
                ax.set_axis_off()
            continue
        plot_axes[0, col].plot(
            truth[x_col],
            truth["true_real"],
            color=COLORS["truth"],
            linewidth=1.8,
            zorder=5,
            label=r"$\mathrm{Re}\{\psi\}$",
        )
        plot_axes[1, col].plot(
            truth[x_col],
            truth["true_imag"],
            color=COLORS["truth"],
            linewidth=1.8,
            zorder=5,
            label=r"$\mathrm{Im}\{\psi\}$",
        )
        if "estimated_real" in truth.columns:
            plot_axes[0, col].plot(
                truth[x_col],
                truth["estimated_real"],
                color=COLORS["estimated_signal"],
                linewidth=1.25,
                alpha=0.8,
                label=r"$\mathrm{Re}\{\hat{\psi}\}$",
            )
            plot_axes[1, col].plot(
                truth[x_col],
                truth["estimated_imag"],
                color=COLORS["estimated_signal"],
                linewidth=1.25,
                alpha=0.8,
                label=r"$\mathrm{Im}\{\hat{\psi}\}$",
            )
        forecast_groups = list(group.groupby("latency_samples", sort=True))
        forecast_cmap = plt.get_cmap(CONFIG_CMAP)
        forecast_colors = forecast_cmap(np.linspace(0.16, 0.86, len(forecast_groups)))
        for color, (latency, latency_group) in zip(forecast_colors, forecast_groups):
            forecast_real = latency_group.dropna(subset=["forecast_real"])
            forecast_imag = latency_group.dropna(subset=["forecast_imag"])
            plot_axes[0, col].plot(
                forecast_real[x_col],
                forecast_real["forecast_real"],
                "--",
                color=color,
                linewidth=1.5,
                label=rf"Forecast, $L={int(latency)}$",
            )
            plot_axes[1, col].plot(
                forecast_imag[x_col],
                forecast_imag["forecast_imag"],
                "--",
                color=color,
                linewidth=1.5,
                label=rf"Forecast, $L={int(latency)}$",
            )
            start = _forecast_start(latency_group)
            if start is not None:
                for ax in plot_axes[:, col]:
                    ax.axvline(start, color=color, linestyle=":", linewidth=1.0, alpha=0.8)
        plot_axes[0, col].set_title(str(severity).capitalize())
        for ax in plot_axes[:, col]:
            ax.set_xlabel("Time [sec]" if x_col == "time_sec" else "Sample")
            if x_col == "time_sec":
                ax.set_xlim(start_s, end_s)
    for ax in plot_axes[0, :]:
        ax.set_ylabel("Real part")
    for ax in plot_axes[1, :]:
        ax.set_ylabel("Imaginary part")
    for row, legend_ax in enumerate(legend_axes):
        handles, labels = _unique_legend_items(plot_axes[row, :])
        if handles:
            legend_ax.legend(handles, labels, loc="center left")
    save_figure(fig, path, tight=False)


def _forecast_case_group(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    source = df.copy()
    if "input_c_n0_db_hz" in source:
        ordered_cases = (
            source[["tracking_case_id", "input_c_n0_db_hz"]]
            .drop_duplicates("tracking_case_id")
            .sort_values(["input_c_n0_db_hz", "tracking_case_id"])
        )
        case_id = ordered_cases["tracking_case_id"].iloc[0]
    else:
        case_id = source["tracking_case_id"].sort_values().iloc[0]
    return source[source["tracking_case_id"].eq(case_id)].copy()


def _forecast_xlim(
    group: pd.DataFrame,
    truth: pd.DataFrame,
    x_col: str,
    configured_start: object,
    configured_end: object,
) -> tuple[float, float]:
    if configured_start is not None and configured_end is not None:
        return float(configured_start), float(configured_end)
    starts = (
        pd.to_numeric(group["forecast_start_time_sec"], errors="coerce").dropna()
        if "forecast_start_time_sec" in group.columns
        else pd.Series(dtype=float)
    )
    if starts.empty:
        return float(truth[x_col].min()), float(truth[x_col].max())
    start = float(starts.min())
    stop = min(float(truth[x_col].max()), float(starts.max()) + 10.0)
    if stop <= start:
        stop = float(truth[x_col].max())
    return start, stop


def _forecast_start(group: pd.DataFrame) -> float | None:
    if "forecast_start_time_sec" in group.columns:
        starts = pd.to_numeric(group["forecast_start_time_sec"], errors="coerce").dropna()
        if not starts.empty:
            return float(starts.iloc[0])
    if "forecast_start_sample_index" in group.columns:
        starts = pd.to_numeric(group["forecast_start_sample_index"], errors="coerce").dropna()
        if not starts.empty:
            return float(starts.iloc[0])
    return None


def _mean_by_c_n0(df: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    clean = df.copy()
    clean["input_c_n0_db_hz"] = pd.to_numeric(clean["input_c_n0_db_hz"], errors="coerce")
    for value_column in value_columns:
        clean[value_column] = pd.to_numeric(clean[value_column], errors="coerce")
    return (
        clean.dropna(subset=["input_c_n0_db_hz"])
        .groupby("input_c_n0_db_hz", as_index=False)[value_columns]
        .mean()
        .sort_values("input_c_n0_db_hz")
    )


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
