from __future__ import annotations

from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from real_time_bp.config import CSV_RESULTS_DIR
from real_time_bp.methodology.filter_bank import AMPLITUDE_SCORE_NAME
from real_time_bp.representatives import (
    SEVERITY_COLORS,
    SEVERITY_ORDER,
    bool_series,
    ordered_severities,
    severity_groups,
)
from real_time_bp.stage_utils import figure_stage_dir
from real_time_bp.utils.params import load_params
from real_time_bp.utils.progress import configure_logging

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

HEATMAP_CMAP = "magma_r"
N_INIT_LINESTYLES = ("-", "--", ":")


def main() -> None:
    configure_logging()
    logger.info("Starting result plot stage")
    load_params()
    _plot_cycle_slips()
    _plot_filter_bank()
    _plot_near_real_time()
    logger.info("Finished result plot stage")


def _plot_cycle_slips() -> None:
    # Paper: Figure 1 diagnostic support. This visualizes the TVD integer
    # cycle-bias trace beside the fade minima that constrain allowed jumps.
    out_dir = figure_stage_dir("02_cycle_slips")
    trace = _read_csv("02_cycle_slips/tvd_example_traces.csv")
    if trace.empty:
        _placeholder(out_dir / "tvd_example_traces.pdf", "No TVD example trace data")
        return
    if "loss_of_lock" in trace.columns:
        no_lock_trace = trace[~bool_series(trace["loss_of_lock"])]
        if not no_lock_trace.empty:
            trace = no_lock_trace
    severities = list(SEVERITY_ORDER)
    if not any(str(value) in severities for value in trace["severity"].dropna().unique()):
        _placeholder(out_dir / "tvd_example_traces.pdf", "No no-lock TVD example traces")
        return
    fig, axes = plt.subplots(
        2,
        len(severities),
        figsize=(4.2 * len(severities), 5.2),
        sharex="col",
        constrained_layout=True,
        squeeze=False,
    )
    intensity_values = pd.to_numeric(trace["intensity_db"], errors="coerce").dropna()
    if intensity_values.empty:
        intensity_ylim = None
    else:
        margin = 0.05 * max(float(intensity_values.max() - intensity_values.min()), 1.0)
        intensity_ylim = (float(intensity_values.min() - margin), float(intensity_values.max() + margin))
    for col, severity in enumerate(severities):
        group = trace[trace["severity"].astype(str).eq(severity)]
        if group.empty:
            for ax in axes[:, col]:
                ax.text(0.5, 0.5, "No no-lock example", ha="center", va="center")
                ax.set_axis_off()
            axes[0, col].set_title(f"{severity} - no loss of lock")
            continue
        group = group[group["tracking_case_id"].eq(group["tracking_case_id"].iloc[0])]
        color = SEVERITY_COLORS.get(str(severity), "tab:blue")
        intensity_ax = axes[0, col]
        phase_ax = axes[1, col]
        intensity_ax.plot(
            group["time_sec"],
            group["intensity_db"],
            color=color,
            linewidth=1.1,
            label="Detrended intensity",
        )
        minima = group[bool_series(group["is_fade_minimum"])]
        intensity_ax.scatter(
            minima["time_sec"],
            minima["intensity_db"],
            color="k",
            s=9,
            alpha=0.65,
            label="Fade minima",
        )
        intensity_ax.axhline(-15.0, color="0.35", linestyle=":", linewidth=1, label="-15 dB threshold")
        intensity_ax.set_title(str(severity))
        intensity_ax.set_ylabel("Intensity [dB]")
        if intensity_ylim is not None:
            intensity_ax.set_ylim(*intensity_ylim)
        intensity_ax.legend(fontsize=7, loc="best")

        estimated_col = (
            "estimated_diffractive_phase_cycles"
            if "estimated_diffractive_phase_cycles" in group
            else "diffractive_phase_cycles"
        )
        phase_ax.plot(
            group["time_sec"],
            group[estimated_col],
            color="k",
            linewidth=1,
            label=r"$\hat{\phi}_s-\phi_r$",
        )
        if "true_diffractive_phase_cycles" in group:
            phase_ax.plot(
                group["time_sec"],
                group["true_diffractive_phase_cycles"],
                color="tab:blue",
                linestyle="--",
                linewidth=1,
                label=r"$\phi_s-\phi_r$",
            )
        phase_ax.step(
            group["time_sec"],
            group["integer_cycle_bias"],
            where="mid",
            color=color,
            linewidth=1.2,
            label="TVD integer bias",
        )
        phase_ax.set_xlabel("Time [sec]")
        phase_ax.set_ylabel("Phase [cycles]")
        phase_ax.legend(fontsize=7, loc="best")
    fig.savefig(out_dir / "tvd_example_traces.pdf")
    plt.close(fig)


def _plot_filter_bank() -> None:
    # Paper: Results Section III.2, Figures 2-8. This group now uses only the
    # amplitude-variability filter-bank baseline.
    out_dir = figure_stage_dir("03_filter_bank")
    scores = _read_csv("03_filter_bank/filter_bank_score_curves.csv")
    estimates = _amplitude_only(_read_csv("03_filter_bank/filter_bank_estimates.csv"))
    n_init = _amplitude_only(_read_csv("03_filter_bank/filter_bank_n_init_sensitivity.csv"))
    residual = _amplitude_only(_read_csv("03_filter_bank/filter_bank_residual_doppler_sensitivity.csv"))
    traces = _amplitude_only(_read_csv("03_filter_bank/filter_bank_representative_phase_traces.csv"))

    _plot_score_curves(scores, out_dir / "representative_filter_bank_score_curves.pdf")
    _plot_tf_error_boxplot(estimates, out_dir / "fresnel_time_error_boxplot.pdf")
    _plot_n_init_sensitivity(n_init, out_dir / "n_init_sensitivity.pdf")
    _plot_residual_sensitivity(residual, out_dir / "residual_doppler_sensitivity.pdf")
    _remove_figure(out_dir / "downstream_phase_metrics.pdf")
    _plot_representative_phase_traces(traces, out_dir / "representative_phase_retrieval.pdf")


def _plot_score_curves(scores: pd.DataFrame, path: Path) -> None:
    # Paper: Figure 2. Representative weak, moderate, and strong score curves
    # over the candidate T_F grid.
    if scores.empty:
        _placeholder(path, "No filter-bank score data")
        return
    severities = ordered_severities(scores)
    fig, axes = plt.subplots(1, len(severities), figsize=(4.2 * len(severities), 3.2), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, severity in zip(axes, severities):
        group = scores[scores["severity"].eq(severity)]
        case_id = group["tracking_case_id"].iloc[0]
        group = group[group["tracking_case_id"].eq(case_id)]
        n_init_values = sorted(pd.to_numeric(group["n_init_samples"], errors="coerce").dropna().unique())
        color_values = plt.cm.viridis(np.linspace(0.15, 0.9, max(len(n_init_values), 1)))
        for color, n_init in zip(color_values, n_init_values):
            n_group = group[pd.to_numeric(group["n_init_samples"], errors="coerce").eq(n_init)]
            ax.plot(
                n_group["fresnel_time_candidate_s"],
                n_group[AMPLITUDE_SCORE_NAME],
                color=color,
                linewidth=1.1,
                label=rf"$N_{{\mathrm{{init}}}}={int(n_init)}$",
            )
        if "true_fresnel_time_scale_s" in group:
            true_tf = pd.to_numeric(group["true_fresnel_time_scale_s"], errors="coerce").dropna()
            if not true_tf.empty:
                ax.axvline(float(true_tf.iloc[0]), color="k", linestyle="--", linewidth=1, label="True")
        ax.set_title(str(severity))
        ax.set_xlabel(r"$T_F$ [s]")
    axes[0].set_ylabel(r"$J\left(T_{\mathrm{F}}\right)$")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_tf_error_boxplot(estimates: pd.DataFrame, path: Path) -> None:
    # Paper: Figure 3. Selected T_F compared against true CPSSM T_F across the
    # dataset.
    if estimates.empty:
        _placeholder(path, "No filter-bank estimate data")
        return
    bins = np.linspace(
        float(estimates["true_fresnel_time_scale_s"].min()),
        float(estimates["true_fresnel_time_scale_s"].max()),
        13,
    )
    centers = 0.5 * (bins[:-1] + bins[1:])
    bin_width = float(np.median(np.diff(centers))) if centers.size > 1 else 0.1
    offsets = np.linspace(-0.27, 0.27, len(SEVERITY_ORDER)) * bin_width
    fig, ax = plt.subplots(figsize=(10, 4.5))
    legend_handles = []
    for severity_idx, severity in enumerate(SEVERITY_ORDER):
        group = estimates[estimates["severity"].astype(str).eq(severity)].copy()
        if group.empty:
            continue
        group["_tf_bin"] = pd.cut(
            group["true_fresnel_time_scale_s"],
            bins=bins,
            labels=False,
            include_lowest=True,
        )
        values = []
        positions = []
        for bin_idx, center in enumerate(centers):
            bin_values = group[group["_tf_bin"].eq(bin_idx)]["estimated_fresnel_time_scale_s"].dropna()
            if bin_values.empty:
                continue
            values.append(bin_values.to_numpy())
            positions.append(float(center + offsets[severity_idx]))
        if not values:
            continue
        color = SEVERITY_COLORS.get(severity, "tab:blue")
        box = ax.boxplot(
            values,
            positions=positions,
            widths=0.16 * bin_width,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
        )
        for patch in box["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.45)
            patch.set_edgecolor(color)
        for element in ["whiskers", "caps", "medians"]:
            for artist in box[element]:
                artist.set_color(color)
        legend_handles.append(plt.Line2D([0], [0], color=color, linewidth=4, alpha=0.65, label=severity))
    min_tf = float(estimates["true_fresnel_time_scale_s"].min())
    max_tf = float(estimates["true_fresnel_time_scale_s"].max())
    ax.plot([min_tf, max_tf], [min_tf, max_tf], "k--", linewidth=1, label="Identity")
    ax.set_xticks(centers)
    ax.set_xticklabels([f"{value:.2f}" for value in centers], rotation=45)
    ax.set_xlabel(r"True $T_F$ [s]")
    ax.set_ylabel(r"Estimated $\hat{T}_F$ [s]")
    ax.legend(handles=[*legend_handles, plt.Line2D([0], [0], color="k", linestyle="--", label="Identity")], fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_n_init_sensitivity(df: pd.DataFrame, path: Path) -> None:
    # Paper: Figure 4. N_init sensitivity used to justify the fixed
    # initialization length in later filter-bank and near-real-time experiments.
    if df.empty:
        _placeholder(path, "No initialization sensitivity data")
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharex=True)
    for severity, group in severity_groups(df):
        group = group.sort_values("n_init_samples")
        label = str(severity)
        color = SEVERITY_COLORS.get(str(severity), "tab:blue")
        axes[0].plot(group["n_init_samples"], group["rmse_fresnel_time_scale_s"], marker="o", color=color, label=label)
        axes[1].plot(group["n_init_samples"], group["r2_fresnel_time_scale"], marker="o", color=color, label=label)
    axes[0].set_ylabel(r"RMSE $T_F$ [s]")
    axes[1].set_ylabel(r"$R^2_{T_F}$")
    for ax in axes:
        ax.set_xlabel(r"$N_{\mathrm{init}}$ [samples]")
    axes[1].legend(fontsize=7, ncol=1)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_residual_sensitivity(df: pd.DataFrame, path: Path) -> None:
    # Paper: Figure 6. Sensitivity to residual Doppler Delta f_D after LOS
    # cleanup, using the sweep defined in Results Section III.1.
    if df.empty:
        _placeholder(path, "No residual Doppler sensitivity data")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
    if "n_init_samples" not in df.columns:
        df = df.assign(n_init_samples=np.nan)
    for severity, severity_group in severity_groups(df):
        color = SEVERITY_COLORS.get(str(severity), "tab:blue")
        for style_idx, (n_init, group) in enumerate(severity_group.groupby("n_init_samples", dropna=False)):
            group = group.sort_values("residual_doppler_error_hz")
            if pd.isna(n_init):
                label = str(severity)
            else:
                label = rf"{severity}, $N_{{\mathrm{{init}}}}={int(n_init)}$"
            linestyle = N_INIT_LINESTYLES[style_idx % len(N_INIT_LINESTYLES)]
            axes[0].plot(
                group["residual_doppler_error_hz"],
                group["rmse_fresnel_time_scale_s"],
                color=color,
                linestyle=linestyle,
                label=label,
            )
            axes[1].plot(
                group["residual_doppler_error_hz"],
                group["r2_fresnel_time_scale"],
                color=color,
                linestyle=linestyle,
                label=label,
            )
    axes[0].set_ylabel(r"RMSE $T_F$ [s]")
    axes[1].set_ylabel(r"$R^2_{T_F}$")
    for ax in axes:
        ax.set_xlabel(r"$\Delta f_D$ [Hz]")
    axes[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_downstream_summary(df: pd.DataFrame, path: Path) -> None:
    # Paper: Figure 8. Downstream metrics showing how the amplitude T_F
    # estimate affects phase-screen retrieval MRMSE and CS/min.
    if df.empty:
        _placeholder(path, "No downstream phase metric data")
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    summary = _sort_by_severity(df)
    x = np.arange(summary.shape[0])
    colors = [SEVERITY_COLORS.get(str(severity), "tab:blue") for severity in summary["severity"]]
    axes[0].bar(x, summary["mean_phase_rmse_rad"], color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(summary["severity"].astype(str), rotation=0)
    axes[0].set_ylabel(r"MRMSE $\phi_r$ [rad]")
    axes[1].bar(x, summary["mean_cycle_slips_per_minute"], color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(summary["severity"].astype(str), rotation=0)
    axes[1].set_ylabel("cycle slips / min")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_representative_phase_traces(df: pd.DataFrame, path: Path) -> None:
    # Paper: Figure 5. Representative phase-retrieval traces compare true
    # refractive phase with batch-BP retrieval for the baseline score.
    if df.empty:
        _placeholder(path, "No representative phase trace data")
        return
    severities = [severity for severity in SEVERITY_ORDER if severity in set(df["severity"].astype(str))]
    fig, axes = plt.subplots(2, len(severities), figsize=(4.2 * len(severities), 6.0), sharex="col")
    axes = np.atleast_2d(axes)
    for col, severity in enumerate(severities):
        group = df[df["severity"].astype(str).eq(severity)]
        case_id = group["tracking_case_id"].iloc[0]
        group = group[group["tracking_case_id"].eq(case_id)]
        color = SEVERITY_COLORS.get(str(severity), "tab:blue")
        true_scintillation_col = _column(
            group,
            "true_scintillation_phase_rad",
            "true_refractive_phase_rad",
        )
        estimated_scintillation_col = _column(
            group,
            "estimated_scintillation_phase_rad",
            "true_refractive_phase_rad",
        )
        batch_col = _column(group, "backpropagated_refractive_phase_rad", "backpropagated_phase_rad")
        axes[0, col].plot(group["time_sec"], group[true_scintillation_col], color="tab:blue", label=r"$\phi_s$")
        axes[0, col].plot(
            group["time_sec"],
            group[estimated_scintillation_col],
            color="tab:blue",
            linestyle=":",
            label=r"$\hat{\phi}_s$",
        )
        axes[0, col].plot(group["time_sec"], group["true_refractive_phase_rad"], color="k", label=r"$\phi_r$")
        axes[0, col].plot(
            group["time_sec"],
            group[batch_col],
            color=color,
            linestyle="--",
            label=r"$\hat{\phi}_{r,\mathrm{batch}}$",
        )
        axes[1, col].plot(group["time_sec"], np.zeros(group.shape[0]), color="k", label=r"$\phi_r-\phi_r$")
        axes[1, col].plot(
            group["time_sec"],
            group[true_scintillation_col] - group["true_refractive_phase_rad"],
            color="tab:blue",
            label=r"$\phi_s-\phi_r$",
        )
        axes[1, col].plot(
            group["time_sec"],
            group[estimated_scintillation_col] - group["true_refractive_phase_rad"],
            color="tab:blue",
            linestyle=":",
            label=r"$\hat{\phi}_s-\phi_r$",
        )
        axes[1, col].plot(
            group["time_sec"],
            group[batch_col] - group["true_refractive_phase_rad"],
            color=color,
            linestyle="--",
            label=r"$\hat{\phi}_{r,\mathrm{batch}}-\phi_r$",
        )
        axes[0, col].set_title(str(severity))
        axes[1, col].set_xlabel("Time [sec]")
    axes[0, 0].set_ylabel("Phase [rad]")
    axes[1, 0].set_ylabel("Phase differences [rad]")
    axes[0, -1].legend(fontsize=7, loc="best")
    axes[1, -1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_near_real_time() -> None:
    # Paper: Results Section III.3, Figures 9-13. Plot group for VAR order
    # distribution, practical metrics, M/L heatmap, and example forecasts.
    out_dir = figure_stage_dir("04_near_real_time")
    var_orders = _read_csv("04_near_real_time/near_real_time_var_orders.csv")
    practical = _read_csv("04_near_real_time/near_real_time_practical_metrics.csv")
    residual = _read_csv("04_near_real_time/near_real_time_residual_doppler_metrics.csv")
    heatmap = _read_csv("04_near_real_time/near_real_time_latency_heatmap.csv")
    phase = _read_csv("04_near_real_time/near_real_time_phase_example.csv")
    forecast = _read_csv("04_near_real_time/near_real_time_forecast_example.csv")
    diagnostics = _read_csv("04_near_real_time/near_real_time_var_residual_diagnostics.csv")

    _plot_var_orders(var_orders, out_dir / "var_order_histogram.pdf")
    _plot_practical_metrics(practical, residual, out_dir / "batch_vs_near_real_time_metrics.pdf")
    _plot_latency_heatmap(heatmap, out_dir / "latency_heatmap.pdf")
    _plot_phase_example(phase, out_dir / "near_real_time_phase_example.pdf")
    _plot_forecast_example(forecast, out_dir / "var_forecast_example.pdf")
    _plot_var_residual_diagnostics(diagnostics, out_dir / "var_residual_diagnostics.pdf")


def _plot_var_orders(df: pd.DataFrame, path: Path) -> None:
    # Paper: Figure 10. Histogram of BIC-selected VAR orders over scintillation
    # severities.
    if df.empty:
        _placeholder(path, "No VAR order data")
        return
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for severity, group in severity_groups(df):
        weights = np.full(group.shape[0], 100.0 / max(group.shape[0], 1))
        ax.hist(
            group["var_order_bic"],
            bins=np.arange(0.5, 25.5, 1),
            weights=weights,
            alpha=0.45,
            color=SEVERITY_COLORS.get(str(severity), "tab:blue"),
            label=severity,
        )
    ax.set_xlabel("BIC-selected VAR order")
    ax.set_ylabel("Relative Frequency [%]")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_practical_metrics(df: pd.DataFrame, residual_df: pd.DataFrame, path: Path) -> None:
    # Paper: Figure 12. Batch BP versus near-real-time BP at the practical M,L
    # operating point.
    if df.empty:
        _placeholder(path, "No practical near-real-time metric data")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharex=False)
    batch_col = _column(df, "batch_refractive_phase_rmse_rad", "batch_phase_rmse_rad")
    near_col = _column(df, "near_real_time_refractive_phase_rmse_rad", "near_real_time_phase_rmse_rad")
    for severity, group in severity_groups(df):
        clean = (
            group.groupby("input_c_n0_db_hz", as_index=False)[[batch_col, near_col]]
            .median()
            .sort_values("input_c_n0_db_hz")
        )
        color = SEVERITY_COLORS.get(str(severity), "tab:blue")
        axes[0].plot(clean["input_c_n0_db_hz"], clean[batch_col], color=color, linestyle="--", alpha=0.65)
        axes[0].plot(clean["input_c_n0_db_hz"], clean[near_col], color=color, label=severity)
    if not residual_df.empty:
        residual_batch_col = _column(residual_df, "batch_refractive_phase_rmse_rad", "batch_phase_rmse_rad")
        residual_near_col = _column(
            residual_df,
            "near_real_time_refractive_phase_rmse_rad",
            "near_real_time_phase_rmse_rad",
        )
        for severity, group in severity_groups(residual_df):
            clean = (
                group.groupby("residual_doppler_error_hz", as_index=False)[[residual_batch_col, residual_near_col]]
                .median()
                .sort_values("residual_doppler_error_hz")
            )
            color = SEVERITY_COLORS.get(str(severity), "tab:blue")
            axes[1].plot(
                clean["residual_doppler_error_hz"],
                clean[residual_batch_col],
                color=color,
                linestyle="--",
                alpha=0.65,
            )
            axes[1].plot(clean["residual_doppler_error_hz"], clean[residual_near_col], color=color, label=severity)
    else:
        axes[1].text(0.5, 0.5, "Regenerate near-real-time CSVs", ha="center", va="center")
    axes[0].set_xlabel(r"$C/N_0$ [dB-Hz]")
    axes[1].set_xlabel(r"Residual Doppler error [Hz]")
    for ax in axes:
        ax.set_ylabel(r"MRMSE $\phi_r$ [rad]")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_latency_heatmap(df: pd.DataFrame, path: Path) -> None:
    # Paper: Figure 11. Heatmap over FFT length M and latency L. Invalid
    # triangular-domain cells are absent.
    if df.empty:
        _placeholder(path, "No latency heatmap data")
        return
    metric_col = _column(df, "refractive_phase_rmse_rad", "phase_rmse_rad")
    x_col = "latency_fraction" if "latency_fraction" in df.columns else "latency_samples"
    group = df.groupby(["block_len", x_col], as_index=False)[metric_col].median()
    pivot = group.pivot(index="block_len", columns=x_col, values=metric_col)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", origin="lower", cmap=HEATMAP_CMAP)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(int))
    xticks = np.linspace(0, len(pivot.columns) - 1, min(8, len(pivot.columns)), dtype=int)
    ax.set_xticks(xticks)
    if x_col == "latency_fraction":
        ax.set_xticklabels([f"{float(value):.2f}" for value in pivot.columns[xticks]], rotation=45)
        ax.set_xlabel(r"Latency fraction $L/(M-M_c-1)$")
    else:
        ax.set_xticklabels(pivot.columns[xticks].astype(int), rotation=45)
        ax.set_xlabel("Latency [samples]")
    ax.set_ylabel("FFT length M [samples]")
    fig.colorbar(image, ax=ax, label=r"MRMSE $\phi_r$ [rad]")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_phase_example(df: pd.DataFrame, path: Path) -> None:
    # Paper: Figure 13. Representative phase comparison between truth, batch BP,
    # and near-real-time BP.
    if df.empty:
        _placeholder(path, "No phase example data")
        return
    severities = [severity for severity in SEVERITY_ORDER if severity in set(df["severity"].astype(str))]
    fig, axes = plt.subplots(2, len(severities), figsize=(4.5 * len(severities), 6.0), sharex="col")
    axes = np.atleast_2d(axes)
    for col, severity in enumerate(severities):
        severity_group = df[df["severity"].astype(str).eq(severity)]
        case_id = severity_group["tracking_case_id"].iloc[0]
        severity_group = severity_group[severity_group["tracking_case_id"].eq(case_id)]
        first_config = severity_group["configuration"].iloc[0] if "configuration" in severity_group else None
        base = (
            severity_group[severity_group["configuration"].eq(first_config)]
            if first_config is not None
            else severity_group
        )
        true_scintillation_col = _column(base, "true_scintillation_phase_rad", "true_refractive_phase_rad")
        estimated_scintillation_col = _column(base, "estimated_scintillation_phase_rad", "true_refractive_phase_rad")
        batch_col = _column(base, "batch_refractive_phase_rad", "batch_backpropagated_phase_rad")
        axes[0, col].plot(
            base["time_sec"],
            base[true_scintillation_col],
            color="tab:blue",
            linewidth=1.4,
            zorder=5,
            label=r"True $\phi_s$",
        )
        axes[0, col].plot(
            base["time_sec"],
            base["true_refractive_phase_rad"],
            color="k",
            linewidth=1.4,
            zorder=6,
            label=r"True $\phi_r$",
        )
        axes[0, col].plot(
            base["time_sec"],
            base[estimated_scintillation_col],
            color="tab:blue",
            linestyle=":",
            linewidth=0.9,
            alpha=0.75,
            label=r"Estimated $\hat{\phi}_s$",
        )
        axes[0, col].plot(
            base["time_sec"],
            base[batch_col],
            color="0.35",
            linestyle="--",
            linewidth=1.0,
            label=r"Batch $\hat{\phi}_{r,\mathrm{batch}}$",
        )
        axes[1, col].plot(
            base["time_sec"],
            base[true_scintillation_col] - base["true_refractive_phase_rad"],
            color="tab:blue",
            label=r"True $\phi_s-\phi_r$",
        )
        axes[1, col].plot(
            base["time_sec"],
            base[batch_col] - base["true_refractive_phase_rad"],
            color="0.35",
            linestyle="--",
            label=r"Batch $\hat{\phi}_{r,\mathrm{batch}}-\phi_r$",
        )
        near_col = _column(
            severity_group,
            "near_real_time_refractive_phase_rad",
            "near_real_time_backpropagated_phase_rad",
        )
        config_groups = (
            severity_group.groupby("configuration", sort=False)
            if "configuration" in severity_group
            else [(r"Near real time", severity_group)]
        )
        config_colors = plt.cm.plasma(np.linspace(0.2, 0.85, len(list(config_groups))))
        config_groups = (
            severity_group.groupby("configuration", sort=False)
            if "configuration" in severity_group
            else [(r"Near real time", severity_group)]
        )
        for color, (configuration, config_group) in zip(config_colors, config_groups):
            axes[0, col].plot(
                config_group["time_sec"],
                config_group[near_col],
                color=color,
                linewidth=0.9,
                label=rf"Near real time $\hat{{\phi}}_r$ ({configuration})",
            )
            axes[1, col].plot(
                config_group["time_sec"],
                config_group[near_col] - config_group["true_refractive_phase_rad"],
                color=color,
                linewidth=0.9,
                label=rf"Near real time $\hat{{\phi}}_r-\phi_r$ ({configuration})",
            )
        axes[0, col].set_title(str(severity))
        axes[1, col].set_xlabel("Time [sec]")
    axes[0, 0].set_ylabel("Phase [rad]")
    axes[1, 0].set_ylabel("Phase differences [rad]")
    axes[0, -1].legend(fontsize=6, loc="best")
    axes[1, -1].legend(fontsize=6, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_forecast_example(df: pd.DataFrame, path: Path) -> None:
    # Paper: Figure 9. True versus VAR-forecasted real/imaginary scintillation
    # samples for a representative case.
    if df.empty:
        _placeholder(path, "No forecast example data")
        return
    group = df[df["tracking_case_id"].eq(df["tracking_case_id"].iloc[0])].sort_values(
        ["latency_samples", "sample_index"]
    )
    truth = group.drop_duplicates("sample_index").sort_values("sample_index")
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    x_col = "time_sec" if "time_sec" in truth.columns else "sample_index"
    axes[0].plot(truth[x_col], truth["true_real"], "k", linewidth=1.4, zorder=5, label="True real")
    axes[1].plot(truth[x_col], truth["true_imag"], "k", linewidth=1.4, zorder=5, label="True imaginary")
    if "estimated_real" in truth.columns:
        axes[0].plot(truth[x_col], truth["estimated_real"], color="0.7", linewidth=0.8, label="Estimated real")
        axes[1].plot(truth[x_col], truth["estimated_imag"], color="0.7", linewidth=0.8, label="Estimated imaginary")
    for latency, latency_group in group.groupby("latency_samples"):
        latency_x = "time_sec" if "time_sec" in latency_group.columns else "sample_index"
        axes[0].plot(
            latency_group[latency_x],
            latency_group["forecast_real"],
            "--",
            linewidth=1,
            label=f"Forecast L={int(latency)}",
        )
        axes[1].plot(
            latency_group[latency_x],
            latency_group["forecast_imag"],
            "--",
            linewidth=1,
            label=f"Forecast L={int(latency)}",
        )
        if "forecast_start_time_sec" in latency_group.columns:
            start = float(latency_group["forecast_start_time_sec"].iloc[0])
        elif "forecast_start_sample_index" in latency_group.columns:
            start = float(latency_group["forecast_start_sample_index"].iloc[0])
        else:
            continue
        for ax in axes:
            ax.axvline(start, color="0.35", linestyle=":", linewidth=0.8)
    axes[1].set_xlabel("Time [sec]" if x_col == "time_sec" else "Sample")
    axes[0].set_ylabel("Real part")
    axes[1].set_ylabel("Imaginary part")
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_var_residual_diagnostics(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        _placeholder(path, "No VAR residual diagnostic data")
        return
    fig, axes = plt.subplots(2, len(SEVERITY_ORDER), figsize=(4.2 * len(SEVERITY_ORDER), 5.8), sharex=False)
    axes = np.atleast_2d(axes)
    for col, severity in enumerate(SEVERITY_ORDER):
        severity_group = df[df["severity"].astype(str).eq(severity)]
        if severity_group.empty:
            for ax in axes[:, col]:
                ax.text(0.5, 0.5, "No diagnostic data", ha="center", va="center")
                ax.set_axis_off()
            continue
        case_id = severity_group["tracking_case_id"].iloc[0]
        severity_group = severity_group[severity_group["tracking_case_id"].eq(case_id)]
        color = SEVERITY_COLORS.get(severity, "tab:blue")
        for row, component in enumerate(["real", "imag"]):
            ax = axes[row, col]
            component_group = severity_group[severity_group["component"].eq(component)]
            residuals = component_group[component_group["diagnostic"].eq("residual")]
            pacf_values = component_group[component_group["diagnostic"].eq("pacf")]
            ax.plot(
                residuals["time_sec"],
                residuals["value"],
                color=color,
                linewidth=0.75,
                label=f"{component.capitalize()} residual",
            )
            ax.axhline(0.0, color="k", linewidth=0.8)
            inset = ax.inset_axes([0.56, 0.53, 0.4, 0.4])
            inset.bar(pacf_values["lag"], pacf_values["pacf"], color=color, alpha=0.7, width=0.8)
            inset.axhline(0.0, color="k", linewidth=0.6)
            inset.set_title("PACF", fontsize=7)
            inset.tick_params(labelsize=6)
            if row == 0:
                ax.set_title(severity)
            if col == 0:
                ax.set_ylabel(f"{component.capitalize()} residual")
            if row == 1:
                ax.set_xlabel("Time [sec]")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _read_csv(relative_path: str) -> pd.DataFrame:
    path = CSV_RESULTS_DIR / relative_path
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def _amplitude_only(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "score_name" not in df.columns:
        return df
    return df[df["score_name"].eq(AMPLITUDE_SCORE_NAME)].copy()


def _ordered_severities(df: pd.DataFrame) -> list[str]:
    if df.empty or "severity" not in df.columns:
        return []
    present = {str(value) for value in df["severity"].dropna().unique()}
    ordered = [severity for severity in SEVERITY_ORDER if severity in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _severity_groups(df: pd.DataFrame):
    for severity in _ordered_severities(df):
        yield severity, df[df["severity"].astype(str).eq(severity)]


def _sort_by_severity(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "severity" not in df.columns:
        return df
    ordered = pd.Categorical(df["severity"].astype(str), categories=SEVERITY_ORDER, ordered=True)
    return df.assign(_severity_order=ordered).sort_values(["_severity_order", "severity"]).drop(
        columns="_severity_order"
    )


def _column(df: pd.DataFrame, preferred: str, fallback: str) -> str:
    return preferred if preferred in df.columns else fallback


def _remove_figure(path: Path) -> None:
    if path.exists():
        path.unlink()


def _placeholder(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.text(0.5, 0.5, text, ha="center", va="center")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
