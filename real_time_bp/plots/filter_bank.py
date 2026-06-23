"""Filter-bank figure generation from stage CSV outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from real_time_bp.methodology.filter_bank import AMPLITUDE_SCORE_NAME
from real_time_bp.plots.common import (
    amplitude_only,
    placeholder,
    read_csv,
    remove_figure,
    save_figure,
)
from real_time_bp.utils.plot_style import (
    COLORS,
    DOPPLER_CMAP,
    N_INIT_LINESTYLES,
    SEVERITY_COLORS,
    figure_size,
    grid_figure_size,
)
from real_time_bp.utils.representatives import SEVERITY_ORDER, ordered_severities, severity_groups
from real_time_bp.utils.stage import figure_stage_dir


def plot_filter_bank() -> None:
    """Render filter-bank score, T_F error, N_init, and Doppler figures."""
    # Paper: Results Section III.2, Figures 2-8. This group now uses only the
    # amplitude-variability filter-bank baseline.
    out_dir = figure_stage_dir("03_filter_bank")
    scores = read_csv("03_filter_bank/filter_bank_score_curves.csv")
    estimates = amplitude_only(read_csv("03_filter_bank/filter_bank_estimates.csv"))
    n_init = amplitude_only(read_csv("03_filter_bank/filter_bank_n_init_sensitivity.csv"))
    residual = amplitude_only(
        read_csv("03_filter_bank/filter_bank_residual_doppler_sensitivity.csv")
    )

    _plot_score_curves(scores, out_dir / "representative_filter_bank_score_curves.pdf")
    _plot_tf_error_boxplot(estimates, out_dir / "fresnel_time_error_boxplot.pdf")
    _plot_n_init_sensitivity(n_init, out_dir / "n_init_sensitivity.pdf")
    _plot_residual_sensitivity(residual, out_dir / "residual_doppler_sensitivity.pdf")
    remove_figure(out_dir / "downstream_phase_metrics.pdf")
    remove_figure(out_dir / "representative_phase_retrieval_legacy.pdf")


def _plot_score_curves(scores: pd.DataFrame, path: Path) -> None:
    # Paper: Figure 2. Representative weak, moderate, and strong score curves
    # over the candidate T_F grid.
    if scores.empty:
        placeholder(path, "No filter-bank score data")
        return
    severities = ordered_severities(scores)
    fig, axes = plt.subplots(
        1,
        len(severities),
        figsize=grid_figure_size(len(severities), 1, panel_height_cm=5.7),
        sharey=False,
    )
    axes = np.atleast_1d(axes)
    for ax, severity in zip(axes, severities):
        group = scores[scores["severity"].eq(severity)]
        case_id = group["tracking_case_id"].iloc[0]
        group = group[group["tracking_case_id"].eq(case_id)]
        n_init_values = sorted(
            pd.to_numeric(group["n_init_samples"], errors="coerce").dropna().unique()
        )
        if "residual_doppler_error_hz" not in group.columns:
            group = group.assign(residual_doppler_error_hz=0.0)
        doppler_values = sorted(
            pd.to_numeric(group["residual_doppler_error_hz"], errors="coerce").dropna().unique()
        )
        doppler_cmap = plt.get_cmap(DOPPLER_CMAP)
        doppler_colors = {
            doppler: color
            for doppler, color in zip(
                doppler_values, doppler_cmap(np.linspace(0.15, 0.9, len(doppler_values)))
            )
        }
        estimate_label_used = False
        for style_idx, n_init in enumerate(n_init_values):
            n_group = group[pd.to_numeric(group["n_init_samples"], errors="coerce").eq(n_init)]
            for doppler in doppler_values:
                curve = n_group[
                    pd.to_numeric(n_group["residual_doppler_error_hz"], errors="coerce").eq(
                        doppler
                    )
                ]
                if curve.empty:
                    continue
                color = doppler_colors[doppler]
                linestyle = N_INIT_LINESTYLES[style_idx % len(N_INIT_LINESTYLES)]
                label = (
                    rf"$J(T_{{\mathrm{{F}}}}),\ "
                    rf"\Delta f_D={doppler:g}\,\mathrm{{Hz}},\ "
                    rf"N_{{\mathrm{{init}}}}={int(n_init)}$"
                )
                estimated_tf = (
                    float(curve["estimated_fresnel_time_scale_s"].iloc[0])
                    if "estimated_fresnel_time_scale_s" in curve
                    else float(
                        curve.loc[curve[AMPLITUDE_SCORE_NAME].idxmin(), "fresnel_time_candidate_s"]
                    )
                )
                ax.axvline(
                    estimated_tf,
                    color=color,
                    linestyle=linestyle,
                    linewidth=0.8,
                    alpha=0.45,
                    label=r"$\hat{T}_{\mathrm{F}}$" if not estimate_label_used else None,
                )
                estimate_label_used = True
                ax.plot(
                    curve["fresnel_time_candidate_s"],
                    curve[AMPLITUDE_SCORE_NAME],
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.1,
                    label=label,
                )
        if "true_fresnel_time_scale_s" in group:
            true_tf = pd.to_numeric(group["true_fresnel_time_scale_s"], errors="coerce").dropna()
            if not true_tf.empty:
                ax.axvline(
                    float(true_tf.iloc[0]),
                    color=COLORS["truth"],
                    linestyle="--",
                    linewidth=1,
                    label=r"$T_{\mathrm{F}}$",
                )
        ax.set_title(str(severity))
        ax.set_xlabel(r"$T_{\mathrm{F}}$ [s]")
    axes[0].set_ylabel(r"$J\left(T_{\mathrm{F}}\right)$")
    axes[-1].legend()
    save_figure(fig, path)


def _plot_tf_error_boxplot(estimates: pd.DataFrame, path: Path) -> None:
    # Paper: Figure 3. Selected T_F compared against true CPSSM T_F across the
    # dataset.
    if estimates.empty:
        placeholder(path, "No filter-bank estimate data")
        return
    bins = np.linspace(
        float(estimates["true_fresnel_time_scale_s"].min()),
        float(estimates["true_fresnel_time_scale_s"].max()),
        13,
    )
    centers = 0.5 * (bins[:-1] + bins[1:])
    bin_width = float(np.median(np.diff(centers))) if centers.size > 1 else 0.1
    offsets = np.linspace(-0.27, 0.27, len(SEVERITY_ORDER)) * bin_width
    fig, ax = plt.subplots(figsize=figure_size("boxplot_grid"))
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
            bin_values = group[group["_tf_bin"].eq(bin_idx)][
                "estimated_fresnel_time_scale_s"
            ].dropna()
            if bin_values.empty:
                continue
            values.append(bin_values.to_numpy())
            positions.append(float(center + offsets[severity_idx]))
        if not values:
            continue
        color = SEVERITY_COLORS.get(severity, COLORS["sliding"])
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
        legend_handles.append(
            plt.Line2D([0], [0], color=color, linewidth=4, alpha=0.65, label=severity)
        )
    min_tf = float(estimates["true_fresnel_time_scale_s"].min())
    max_tf = float(estimates["true_fresnel_time_scale_s"].max())
    ax.plot(
        [min_tf, max_tf],
        [min_tf, max_tf],
        color=COLORS["truth"],
        linestyle="--",
        linewidth=1,
        label="Identity",
    )
    ax.set_xticks(centers)
    ax.set_xticklabels([f"{value:.2f}" for value in centers], rotation=45)
    ax.set_xlabel(r"$T_{\mathrm{F}}$ [s]")
    ax.set_ylabel(r"$\hat{T}_{\mathrm{F}}$ [s]")
    ax.legend(
        handles=[
            *legend_handles,
            plt.Line2D([0], [0], color=COLORS["truth"], linestyle="--", label="Identity"),
        ]
    )
    save_figure(fig, path)


def _plot_n_init_sensitivity(df: pd.DataFrame, path: Path) -> None:
    # Paper: Figure 4. N_init sensitivity used to justify the fixed
    # initialization length in later filter-bank and near-real-time experiments.
    if df.empty:
        placeholder(path, "No initialization sensitivity data")
        return
    fig, axes = plt.subplots(1, 2, figsize=figure_size("two_panel"), sharex=True)
    for severity, group in severity_groups(df):
        group = group.sort_values("n_init_samples")
        label = str(severity)
        color = SEVERITY_COLORS.get(str(severity), COLORS["sliding"])
        axes[0].plot(
            group["n_init_samples"],
            group["rmse_fresnel_time_scale_s"],
            marker="o",
            color=color,
            label=label,
        )
        axes[1].plot(
            group["n_init_samples"],
            group["r2_fresnel_time_scale"],
            marker="o",
            color=color,
            label=label,
        )
    axes[0].set_ylabel(r"RMSE $T_{\mathrm{F}}$ [s]")
    axes[1].set_ylabel(r"$R^2_{T_{\mathrm{F}}}$")
    for ax in axes:
        ax.set_xlabel(r"$N_{\mathrm{init}}$ [samples]")
    axes[1].legend(ncol=1)
    save_figure(fig, path)


def _plot_residual_sensitivity(df: pd.DataFrame, path: Path) -> None:
    # Paper: Figure 6. Sensitivity to residual Doppler Delta f_D after LOS
    # cleanup, using the sweep defined in Results Section III.1.
    if df.empty:
        placeholder(path, "No residual Doppler sensitivity data")
        return
    fig, ax_rmse = plt.subplots(figsize=figure_size("two_panel"))
    ax_r2 = ax_rmse.twinx()
    if "n_init_samples" not in df.columns:
        df = df.assign(n_init_samples=np.nan)
    for severity, severity_group in severity_groups(df):
        color = SEVERITY_COLORS.get(str(severity), COLORS["sliding"])
        for n_init, group in severity_group.groupby("n_init_samples", dropna=False):
            group = group.sort_values("residual_doppler_error_hz")
            n_label = "unknown" if pd.isna(n_init) else str(int(n_init))
            label = rf"{severity}, $N_{{\mathrm{{init}}}}={n_label}$"
            ax_rmse.plot(
                group["residual_doppler_error_hz"],
                group["rmse_fresnel_time_scale_s"],
                color=color,
                linestyle="-",
                label=label,
            )
            ax_r2.plot(
                group["residual_doppler_error_hz"],
                group["r2_fresnel_time_scale"],
                color=color,
                linestyle=":",
                linewidth=1.8,
            )
    ax_rmse.set_ylabel(r"RMSE $T_{\mathrm{F}}$ [s]")
    ax_r2.set_ylabel(r"$R^2_{T_{\mathrm{F}}}$")
    ax_rmse.set_xlabel(r"$\Delta f_D$ [Hz]")
    ax_rmse.legend(loc="best")
    save_figure(fig, path)
