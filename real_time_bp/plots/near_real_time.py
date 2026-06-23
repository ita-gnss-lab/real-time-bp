"""Near-real-time figure generation from stage CSV outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from real_time_bp.plots.common import column, placeholder, read_csv, remove_figure, save_figure
from real_time_bp.utils.params import load_params
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
    var_residual_summary = read_csv("04_near_real_time/near_real_time_var_residual_summary.csv")
    var_forecast_utility = read_csv("04_near_real_time/near_real_time_var_forecast_utility.csv")
    practical = read_csv("04_near_real_time/near_real_time_practical_metrics.csv")
    residual = read_csv("04_near_real_time/near_real_time_residual_doppler_metrics.csv")
    heatmap = read_csv("04_near_real_time/near_real_time_latency_heatmap.csv")
    phase = read_csv("04_near_real_time/near_real_time_phase_example.csv")
    forecast = read_csv("04_near_real_time/near_real_time_forecast_example.csv")
    diagnostics = read_csv("04_near_real_time/near_real_time_var_residual_diagnostics.csv")

    _plot_var_orders(var_orders, out_dir / "var_order_histogram.pdf")
    _plot_var_model_diagnostics(
        var_residual_summary,
        var_forecast_utility,
        out_dir / "var_model_diagnostics.pdf",
    )
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
    ax.set_ylabel("Relative Frequency")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.legend()
    save_figure(fig, path)


def _plot_var_model_diagnostics(
    residual_summary: pd.DataFrame,
    forecast_utility: pd.DataFrame,
    path: Path,
) -> None:
    """Plot compact VAR adequacy and usefulness diagnostics.

    Main paper:
        Results Section III.3 uses VAR as a short-horizon engineering
        predictor, not as a complete scintillation process model. This figure
        therefore shows both residual whiteness and forecast utility relative
        to a persistence baseline.
    """
    if residual_summary.empty and forecast_utility.empty:
        placeholder(path, "No VAR model diagnostic data")
        return
    params = load_params()
    near_cfg = params.get("near_real_time", {})
    alpha = float(near_cfg.get("var_test_alpha", 0.05))
    plot_lag = int(near_cfg.get("var_diagnostic_plot_whiteness_lag", 24))
    fig, axes = plt.subplots(1, 2, figsize=figure_size("two_panel"))
    _draw_whiteness_distribution(residual_summary, axes[0], alpha, plot_lag)
    _draw_forecast_utility(forecast_utility, axes[1])
    save_figure(fig, path)


def _draw_whiteness_distribution(
    df: pd.DataFrame,
    ax,
    alpha: float,
    plot_lag: int,
) -> None:
    if df.empty:
        ax.text(0.5, 0.5, "No whiteness data", ha="center", va="center")
        ax.set_axis_off()
        return
    lag_values = pd.to_numeric(df["requested_whiteness_lag"], errors="coerce")
    selected = df[lag_values.eq(plot_lag)].copy()
    if selected.empty:
        selected = df[lag_values.eq(lag_values.max())].copy()
    values = []
    labels = []
    colors = []
    for severity in SEVERITY_ORDER:
        group = selected[selected["severity"].astype(str).eq(severity)]
        y = pd.to_numeric(group["minus_log10_whiteness_pvalue"], errors="coerce").dropna()
        if y.empty:
            continue
        values.append(y.to_numpy())
        labels.append(str(severity).capitalize())
        colors.append(SEVERITY_COLORS.get(severity, COLORS["sliding"]))
    if not values:
        ax.text(0.5, 0.5, "No finite whiteness p-values", ha="center", va="center")
        ax.set_axis_off()
        return
    box = ax.boxplot(values, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
        patch.set_edgecolor(color)
    for element in ["whiskers", "caps", "medians"]:
        for artist in box[element]:
            artist.set_color(COLORS["neutral"])
    ax.axhline(
        -np.log10(alpha),
        color=COLORS["truth"],
        linestyle="--",
        linewidth=1.0,
        label=rf"$\alpha={alpha:g}$",
    )
    lag = int(pd.to_numeric(selected["requested_whiteness_lag"], errors="coerce").dropna().iloc[0])
    ax.set_title(rf"Whiteness, lag {lag}")
    ax.set_ylabel(r"$-\log_{10}(p_{\mathrm{white}})$")
    ax.legend()


def _draw_forecast_utility(df: pd.DataFrame, ax) -> None:
    if df.empty:
        ax.text(0.5, 0.5, "No forecast utility data", ha="center", va="center")
        ax.set_axis_off()
        return
    for severity, group in severity_groups(df):
        clean = group.copy()
        clean["forecast_horizon_samples"] = pd.to_numeric(
            clean["forecast_horizon_samples"], errors="coerce"
        )
        clean["var_to_persistence_rmse_ratio"] = pd.to_numeric(
            clean["var_to_persistence_rmse_ratio"], errors="coerce"
        )
        clean = clean.dropna(subset=["forecast_horizon_samples", "var_to_persistence_rmse_ratio"])
        if clean.empty:
            continue
        summary = (
            clean.groupby("forecast_horizon_samples")["var_to_persistence_rmse_ratio"]
            .agg(
                median="median",
                q25=lambda values: np.nanquantile(values, 0.25),
                q75=lambda values: np.nanquantile(values, 0.75),
            )
            .reset_index()
            .sort_values("forecast_horizon_samples")
        )
        x = summary["forecast_horizon_samples"].to_numpy(dtype=float)
        color = SEVERITY_COLORS.get(str(severity), COLORS["sliding"])
        ax.plot(
            x,
            summary["median"],
            color=color,
            marker="o",
            linewidth=1.2,
            label=str(severity).capitalize(),
        )
        ax.fill_between(
            x,
            summary["q25"],
            summary["q75"],
            color=color,
            alpha=0.18,
            linewidth=0,
        )
    ax.axhline(1.0, color=COLORS["truth"], linestyle="--", linewidth=1.0)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Forecast horizon [samples]")
    ax.set_ylabel(r"RMSE$_{\mathrm{VAR}}$ / RMSE$_{\mathrm{pers.}}$")
    ax.set_title("Forecast utility")
    ax.legend()


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
        ax.set_ylabel(r"MRMSE $\hat{\phi}_r$ [rad]")
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
                (metric_pivot, metric_vmin, metric_vmax, r"MRMSE $\hat{\phi}_r$ [rad]"),
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
            metric_image, ax=axes[0, :], label=r"MRMSE $\hat{\phi}_r$ [rad]", fraction=0.03
        )
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
        len(severities) + 1,
        figsize=grid_figure_size(len(severities) + 1, 1, panel_height_cm=6.0),
        sharex=False,
        gridspec_kw={"width_ratios": [1.0] * len(severities) + [0.65]},
    )
    axes = np.atleast_1d(axes)
    legend_ax = axes[-1]
    legend_ax.set_axis_off()
    legend_title = None
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
        batch_error_col = column(base, "batch_refractive_phase_error_rad", "batch_phase_error_rad")
        batch_error_cycles = base[batch_error_col] / (2.0 * np.pi)
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
    handles, labels = _unique_legend_items(axes[:-1])
    if handles:
        legend_ax.legend(handles, labels, loc="center left", title=legend_title)
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
    params = load_params()
    plot_cfg = params.get("plots", {})
    near_cfg = params.get("near_real_time", {})
    zoom_start_s = plot_cfg.get("forecast_zoom_start_s")
    zoom_end_s = plot_cfg.get("forecast_zoom_end_s")
    requested_latencies = [
        int(value) for value in near_cfg.get("forecast_example_latency_samples", [])
    ]
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
    latencies = sorted(pd.to_numeric(group["latency_samples"], errors="coerce").dropna().unique())
    if not latencies:
        placeholder(path, "No forecast latency data")
        return
    selected_latencies = [latency for latency in requested_latencies if latency in latencies]
    if not selected_latencies:
        selected_latencies = [int(latency) for latency in latencies]
    group = group[
        pd.to_numeric(group["latency_samples"], errors="coerce").isin(selected_latencies)
    ]
    if group.empty:
        placeholder(path, "No forecast CSV rows for selected latencies")
        return
    truth = group.drop_duplicates("sample_index").sort_values("sample_index")
    x_col = "time_sec" if "time_sec" in truth.columns else "sample_index"
    if x_col == "time_sec":
        if zoom_start_s is None or zoom_end_s is None:
            zoom_start_s, zoom_end_s = _forecast_auto_xlim(group, truth, x_col)
        else:
            zoom_start_s = float(zoom_start_s)
            zoom_end_s = float(zoom_end_s)
        zoom_mask = truth[x_col].between(float(zoom_start_s), float(zoom_end_s))
        forecast_zoom_mask = group[x_col].between(float(zoom_start_s), float(zoom_end_s))
    else:
        zoom_mask = pd.Series(True, index=truth.index)
        forecast_zoom_mask = pd.Series(True, index=group.index)
    truth = truth[zoom_mask]
    group = group[forecast_zoom_mask]
    if truth.empty or group.empty:
        placeholder(path, "No forecast samples in the selected zoom window")
        return
    fig, axes = plt.subplots(1, 2, figsize=figure_size("two_panel"), sharex=True)
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
    forecast_groups = list(group.groupby("latency_samples", sort=True))
    forecast_colors = forecast_cmap(np.linspace(0.16, 0.86, len(forecast_groups)))
    for color, (latency, latency_group) in zip(forecast_colors, forecast_groups):
        axes[0].plot(
            latency_group[x_col],
            latency_group["forecast_real"],
            "--",
            color=color,
            linewidth=1.5,
            label=rf"Forecast, $L={int(latency)}$",
        )
        axes[1].plot(
            latency_group[x_col],
            latency_group["forecast_imag"],
            "--",
            color=color,
            linewidth=1.5,
            label=rf"Forecast, $L={int(latency)}$",
        )
        start = _forecast_start(latency_group)
        if start is not None:
            for ax in axes:
                ax.axvline(start, color=color, linestyle=":", linewidth=1.0, alpha=0.8)
    for ax in axes:
        ax.set_xlabel("Time [sec]" if x_col == "time_sec" else "Sample")
        if x_col == "time_sec":
            ax.set_xlim(float(zoom_start_s), float(zoom_end_s))
    axes[0].set_ylabel("Real part")
    axes[1].set_ylabel("Imaginary part")
    axes[0].set_title("Real")
    axes[1].set_title("Imaginary")
    axes[0].legend(loc="upper right")
    axes[1].legend(loc="upper right")
    save_figure(fig, path)


def _forecast_auto_xlim(
    group: pd.DataFrame, truth: pd.DataFrame, x_col: str
) -> tuple[float, float]:
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
        legend_title = _var_residual_legend_title(severity_group)
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
            ax.legend(loc="best", title=legend_title)
    save_figure(fig, path)


def _var_residual_legend_title(df: pd.DataFrame) -> str | None:
    parts = []
    if "n_init_samples" in df.columns and df["n_init_samples"].notna().any():
        parts.append(rf"$N_{{\mathrm{{init}}}}={int(df['n_init_samples'].dropna().iloc[0])}$")
    normality_pct = _pass_percentage(df, "var_normality_pass")
    whiteness_pct = _pass_percentage(df, "var_whiteness_pass")
    if normality_pct is not None:
        parts.append(f"Normality pass: {normality_pct:.0f}%")
    if whiteness_pct is not None:
        parts.append(f"Whiteness pass: {whiteness_pct:.0f}%")
    return "\n".join(parts) if parts else None


def _pass_percentage(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns:
        return None
    cases = df[["tracking_case_id", column]].drop_duplicates("tracking_case_id")
    values = pd.to_numeric(cases[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(100.0 * values.mean())


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
