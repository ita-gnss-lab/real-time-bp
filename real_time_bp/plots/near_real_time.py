"""Near-real-time figure generation from stage CSV outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from real_time_bp.plots.common import column, placeholder, read_csv, remove_figure, save_figure
from real_time_bp.utils.plot_style import (
    COLORS,
    CONFIG_CMAP,
    FONT_SIZES_PT,
    HEATMAP_CMAP,
    SEVERITY_COLORS,
    figure_size,
    grid_figure_size,
)
from real_time_bp.utils.representatives import SEVERITY_ORDER, severity_groups
from real_time_bp.utils.stage import figure_stage_dir


def plot_near_real_time() -> None:
    """Render VAR, latency, practical-metric, and forecast figures.

    Main paper:
        Results Section III.3, Figures 9-13, defines this plot group for VAR
        order distribution, practical metrics, M/L heatmap, and example
        forecasts.
    """
    out_dir = figure_stage_dir("04_near_real_time")
    var_orders = read_csv("04_near_real_time/near_real_time_var_orders.csv")
    practical = read_csv("04_near_real_time/near_real_time_practical_metrics.csv")
    residual = read_csv("04_near_real_time/near_real_time_residual_doppler_metrics.csv")
    heatmap = read_csv("04_near_real_time/near_real_time_latency_heatmap.csv")
    phase = read_csv("04_near_real_time/near_real_time_phase_example.csv")
    forecast = read_csv("04_near_real_time/near_real_time_forecast_example.csv")
    diagnostics = read_csv("04_near_real_time/near_real_time_var_residual_diagnostics.csv")

    _plot_var_orders(var_orders, out_dir / "var_order_histogram.pdf")
    _plot_practical_metrics(practical, residual, out_dir / "batch_vs_near_real_time_metrics.pdf")
    _plot_latency_heatmap(heatmap, out_dir / "latency_heatmap.pdf")
    _plot_phase_example(phase, out_dir / "near_real_time_phase_example.pdf")
    _plot_forecast_example(forecast, out_dir / "var_forecast_example.pdf")
    _plot_var_residual_diagnostics(diagnostics, out_dir / "var_residual_diagnostics.pdf")
    remove_figure(out_dir / "var_residual_pacf.pdf")


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
    ax.set_ylabel("Relative Frequency (%)")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.legend()
    save_figure(fig, path)


def _plot_practical_metrics(df: pd.DataFrame, residual_df: pd.DataFrame, path: Path) -> None:
    """Plot batch BP versus near-real-time BP practical metrics.

    Main paper:
        Results Figure 12 compares the noncausal batch BP reference with the
        practical near-real-time BP operating point.
    """
    if df.empty:
        placeholder(path, "No practical near-real-time metric data")
        return
    fig, axes = plt.subplots(1, 2, figsize=figure_size("two_panel"), sharex=False)
    batch_col = column(df, "batch_refractive_phase_rmse_rad", "batch_phase_rmse_rad")
    near_col = column(
        df, "near_real_time_refractive_phase_rmse_rad", "near_real_time_phase_rmse_rad"
    )
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
    for severity, group in severity_groups(df):
        clean = (
            group.groupby("input_c_n0_db_hz", as_index=False)[[batch_col, near_col]]
            .median()
            .sort_values("input_c_n0_db_hz")
        )
        color = SEVERITY_COLORS.get(str(severity), COLORS["sliding"])
        axes[0].plot(
            clean["input_c_n0_db_hz"],
            clean[batch_col],
            color=color,
            linestyle="--",
            alpha=0.75,
            label=rf"{severity} batch BP",
        )
        axes[0].plot(
            clean["input_c_n0_db_hz"],
            clean[near_col],
            color=color,
            label=rf"{severity} sliding BP",
        )
    if not residual_df.empty:
        residual_batch_col = column(
            residual_df, "batch_refractive_phase_rmse_rad", "batch_phase_rmse_rad"
        )
        residual_near_col = column(
            residual_df,
            "near_real_time_refractive_phase_rmse_rad",
            "near_real_time_phase_rmse_rad",
        )
        for severity, group in severity_groups(residual_df):
            clean = (
                group.groupby("residual_doppler_error_hz", as_index=False)[
                    [residual_batch_col, residual_near_col]
                ]
                .median()
                .sort_values("residual_doppler_error_hz")
            )
            color = SEVERITY_COLORS.get(str(severity), COLORS["sliding"])
            axes[1].plot(
                clean["residual_doppler_error_hz"],
                clean[residual_batch_col],
                color=color,
                linestyle="--",
                alpha=0.75,
                label=rf"{severity} batch BP",
            )
            axes[1].plot(
                clean["residual_doppler_error_hz"],
                clean[residual_near_col],
                color=color,
                label=rf"{severity} sliding BP",
            )
    else:
        axes[1].text(0.5, 0.5, "Regenerate near-real-time CSVs", ha="center", va="center")
    axes[0].set_xlabel(r"$C/N_0$ [dB-Hz]")
    axes[1].set_xlabel(r"Residual Doppler error [Hz]")
    axes[1].set_xlim(0.0, 1.5)
    for ax in axes:
        ax.set_ylabel(r"MRMSE $\phi_r$ [rad]")
        if n_value is not None and m_value is not None:
            ax.set_title(rf"$N_{{\mathrm{{init}}}}={n_value}$, $M={m_value}$")
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
    metric_col = column(df, "refractive_phase_rmse_rad", "phase_rmse_rad")
    x_col = "latency_fraction" if "latency_fraction" in df.columns else "latency_samples"
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
        metric_group = severity_df.groupby(["block_len", x_col], as_index=False)[
            metric_col
        ].median()
        cs_group = severity_df.groupby(["block_len", x_col], as_index=False)[
            "cycle_slips_per_minute"
        ].median()
        metric_pivot = metric_group.pivot(index="block_len", columns=x_col, values=metric_col)
        cs_pivot = cs_group.pivot(
            index="block_len", columns=x_col, values="cycle_slips_per_minute"
        )
        for row, (pivot, vmin, vmax, title) in enumerate(
            [
                (metric_pivot, metric_vmin, metric_vmax, r"MRMSE $\phi_r$ [rad]"),
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
        fig.colorbar(metric_image, ax=axes[0, :], label=r"MRMSE $\phi_r$ [rad]", fraction=0.03)
    if cs_image is not None:
        fig.colorbar(cs_image, ax=axes[1, :], label="CS/min", fraction=0.03)
    save_figure(fig, path, tight=False)


def _plot_phase_example(df: pd.DataFrame, path: Path) -> None:
    """Plot representative phase traces for batch and near-real-time BP.

    Main paper:
        Results Figure 13 compares truth, batch BP, and near-real-time BP for
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
        len(severities),
        figsize=grid_figure_size(len(severities), 1, panel_height_cm=6.0),
        sharex=False,
    )
    axes = np.atleast_1d(axes)
    for col, severity in enumerate(severities):
        ax = axes[col]
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
        first_config = (
            severity_group["configuration"].iloc[0] if "configuration" in severity_group else None
        )
        base = (
            severity_group[severity_group["configuration"].eq(first_config)]
            if first_config is not None
            else severity_group
        )
        batch_error_col = column(base, "batch_refractive_phase_error_rad", "batch_phase_error_rad")
        batch_error_cycles = base[batch_error_col] / (2.0 * np.pi)
        # FIXME: There is still a phase bias between phi_s - phi_r and the zero-line. fix this. 
        if "estimated_scintillation_minus_refractive_rad" in base:
            ax.plot(
                base["time_sec"],
                base["estimated_scintillation_minus_refractive_rad"] / (2.0 * np.pi),
                color=COLORS["estimated_scintillation"],
                linestyle="--",
                linewidth=1.0,
                label=r"$\hat{\phi}_s-\phi_r$",
            )
        ax.plot(
            base["time_sec"],
            batch_error_cycles,
            color=COLORS["truth"],
            linewidth=1.1,
            label=r"$\hat{\phi}_{r,\mathrm{batch}}-\phi_r$",
        )
        near_col = column(
            severity_group,
            "near_real_time_refractive_phase_error_rad",
            "near_real_time_phase_error_rad",
        )
        config_groups = severity_group.sort_values(["latency_samples", "sample_index"]).groupby(
            "latency_samples", sort=True
        )
        config_items = list(config_groups)
        config_cmap = plt.get_cmap(CONFIG_CMAP)
        config_colors = config_cmap(np.linspace(0.2, 0.85, len(config_items)))
        finite_times = []
        for color, (latency, config_group) in zip(config_colors, config_items):
            phase_error_cycles = config_group[near_col] / (2.0 * np.pi)
            finite = np.isfinite(phase_error_cycles)
            finite_times.extend(config_group.loc[finite, "time_sec"].to_list())
            ax.plot(
                config_group["time_sec"],
                phase_error_cycles,
                color=color,
                linewidth=1.0,
                label=rf"$\hat{{\phi}}_{{r,\mathrm{{sliding}}}}-\phi_r$, $L={int(latency)}$",
            )
        if finite_times:
            ax.set_xlim(min(finite_times), max(finite_times))
        ax.set_title(str(severity).capitalize())
        ax.set_xlabel("Time [sec]")
        ax.set_ylabel("Phase differences [cycles]")
    axes[-1].legend(loc="best")
    save_figure(fig, path)


def _plot_forecast_example(df: pd.DataFrame, path: Path) -> None:
    """Plot true, tracked, and VAR-forecasted scintillation samples.

    Main paper:
        Results Figure 9 compares true and VAR-forecasted real/imaginary
        scintillation samples for a representative case.
    """
    if df.empty:
        placeholder(path, "No forecast example data")
        return
    strong = df[df["severity"].astype(str).eq("strong")]
    source = strong if not strong.empty else df
    if "input_c_n0_db_hz" in source:
        ordered_cases = (
            source[["tracking_case_id", "input_c_n0_db_hz"]]
            .drop_duplicates("tracking_case_id")
            .sort_values(["input_c_n0_db_hz", "tracking_case_id"])
        )
        case_id = ordered_cases["tracking_case_id"].iloc[0]
    else:
        case_id = source["tracking_case_id"].sort_values().iloc[0]
    group = source[source["tracking_case_id"].eq(case_id)].sort_values(
        ["latency_samples", "sample_index"]
    )
    group = group[pd.to_numeric(group["latency_samples"], errors="coerce").isin([0, 1024, 2048])]
    if group.empty:
        placeholder(path, "Regenerate forecast CSVs for L=0, 1024, 2048")
        return
    truth = group.drop_duplicates("sample_index").sort_values("sample_index")
    fig, axes = plt.subplots(2, 1, figsize=figure_size("two_row"), sharex=True)
    x_col = "time_sec" if "time_sec" in truth.columns else "sample_index"
    axes[0].plot(
        truth[x_col],
        truth["true_real"],
        color=COLORS["truth"],
        linewidth=1.8,
        zorder=5,
        label=r"$\mathrm{Re}\{\psi\}$",
    )
    axes[1].plot(
        truth[x_col],
        truth["true_imag"],
        color=COLORS["truth"],
        linewidth=1.8,
        zorder=5,
        label=r"$\mathrm{Im}\{\psi\}$",
    )
    if "estimated_real" in truth.columns:
        axes[0].plot(
            truth[x_col],
            truth["estimated_real"],
            color=COLORS["estimated_signal"],
            linewidth=1.2,
            alpha=0.75,
            label=r"$\mathrm{Re}\{\hat{\psi}\}$",
        )
        axes[1].plot(
            truth[x_col],
            truth["estimated_imag"],
            color=COLORS["estimated_signal"],
            linewidth=1.2,
            alpha=0.75,
            label=r"$\mathrm{Im}\{\hat{\psi}\}$",
        )
    forecast_cmap = plt.get_cmap(CONFIG_CMAP)
    latencies = sorted(pd.to_numeric(group["latency_samples"], errors="coerce").dropna().unique())
    forecast_colors = {
        latency: color
        for latency, color in zip(
            latencies, forecast_cmap(np.linspace(0.15, 0.85, len(latencies)))
        )
    }
    for latency, latency_group in group.groupby("latency_samples", sort=True):
        latency_x = "time_sec" if "time_sec" in latency_group.columns else "sample_index"
        forecast_color = forecast_colors.get(float(latency), COLORS["forecast"])
        axes[0].plot(
            latency_group[latency_x],
            latency_group["forecast_real"],
            "--",
            color=forecast_color,
            linewidth=1.3,
            label=f"Forecast L={int(latency)}",
        )
        axes[1].plot(
            latency_group[latency_x],
            latency_group["forecast_imag"],
            "--",
            color=forecast_color,
            linewidth=1.3,
            label=f"Forecast L={int(latency)}",
        )
        if "forecast_start_time_sec" in latency_group.columns:
            start = float(latency_group["forecast_start_time_sec"].iloc[0])
        elif "forecast_start_sample_index" in latency_group.columns:
            start = float(latency_group["forecast_start_sample_index"].iloc[0])
        else:
            continue
        for ax in axes:
            ax.axvline(start, color=COLORS["neutral"], linestyle=":", linewidth=1.0)
    axes[1].set_xlabel("Time [sec]" if x_col == "time_sec" else "Sample")
    axes[0].set_ylabel("Real part")
    axes[1].set_ylabel("Imaginary part")
    axes[0].legend(loc="upper right")
    axes[1].legend(loc="upper right")
    save_figure(fig, path)


def _plot_var_residual_diagnostics(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        placeholder(path, "No VAR residual diagnostic data")
        return
    fig, axes = plt.subplots(
        2,
        len(SEVERITY_ORDER),
        figsize=grid_figure_size(len(SEVERITY_ORDER), 2, panel_height_cm=5.2),
        sharex=False,
    )
    axes = np.atleast_2d(axes)
    residual_values = pd.to_numeric(
        df[df["diagnostic"].eq("residual")]["value"], errors="coerce"
    ).dropna()
    if residual_values.empty:
        residual_ylim = None
    else:
        max_abs = float(np.nanmax(np.abs(residual_values)))
        residual_ylim = (-1.05 * max_abs, 1.05 * max_abs)
    for col, severity in enumerate(SEVERITY_ORDER):
        severity_group = df[df["severity"].astype(str).eq(severity)]
        if severity_group.empty:
            for ax in axes[:, col]:
                ax.text(0.5, 0.5, "No diagnostic data", ha="center", va="center")
                ax.set_axis_off()
            continue
        case_id = severity_group["tracking_case_id"].iloc[0]
        severity_group = severity_group[severity_group["tracking_case_id"].eq(case_id)]
        color = COLORS["residual"]
        for row, component in enumerate(["real", "imag"]):
            ax = axes[row, col]
            component_group = severity_group[severity_group["component"].eq(component)]
            residuals = component_group[component_group["diagnostic"].eq("residual")]
            ax.plot(
                residuals["time_sec"],
                residuals["value"],
                color=color,
                linewidth=0.75,
                label=rf"$\mathrm{{{'Re' if component == 'real' else 'Im'}}}\{{\hat{{\psi}}\}}$ residual",
            )
            ax.axhline(0.0, color=COLORS["truth"], linewidth=0.8)
            if residual_ylim is not None:
                ax.set_ylim(*residual_ylim)
            if row == 0:
                ax.set_title(str(severity).capitalize())
            if col == 0:
                ax.set_ylabel(
                    r"$\mathrm{Re}\{\hat{\psi}\}$ residuals"
                    if component == "real"
                    else r"$\mathrm{Im}\{\hat{\psi}\}$ residuals"
                )
            if row == 1:
                ax.set_xlabel("Time [sec]")
    save_figure(fig, path)
