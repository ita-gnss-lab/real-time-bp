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
    CONFIG_CMAP,
    N_INIT_LINESTYLES,
    SEVERITY_COLORS,
    grid_figure_size,
)
from real_time_bp.utils.representatives import SEVERITY_ORDER, ordered_severities, severity_groups
from real_time_bp.utils.stage import figure_stage_dir


def plot_filter_bank() -> None:
    """Render filter-bank score, T_F error, N_init, and Doppler figures.

    Main paper:
        Results Section III.2, Figures 2-8, define this figure group. The
        current pipeline uses only the amplitude-variability filter-bank
        baseline.
    """
    out_dir = figure_stage_dir("03_filter_bank")
    scores = read_csv("03_filter_bank/filter_bank_score_curves.csv")
    estimates = amplitude_only(read_csv("03_filter_bank/filter_bank_estimates.csv"))
    n_init = amplitude_only(read_csv("03_filter_bank/filter_bank_n_init_sensitivity.csv"))

    _plot_score_curves(scores, out_dir / "representative_filter_bank_score_curves.pdf")
    _plot_n_init_sensitivity(n_init, estimates, out_dir / "n_init_sensitivity.pdf")
    remove_figure(out_dir / "fresnel_time_error_boxplot.pdf")
    remove_figure(out_dir / "residual_doppler_sensitivity.pdf")
    remove_figure(out_dir / "downstream_phase_metrics.pdf")
    remove_figure(out_dir / "representative_phase_retrieval.pdf")
    remove_figure(out_dir / "representative_phase_retrieval_legacy.pdf")


def _plot_score_curves(scores: pd.DataFrame, path: Path) -> None:
    """Plot representative weak and strong score curves over the T_F grid.

    Main paper:
        Results Figure 2 displays J(T_F) against N_init for representative
        cases. Residual Doppler robustness is not part of this figure.
    """
    # TODO: Include the legends in a dedicated third column.
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
        if "residual_doppler_error_hz" in group.columns:
            residual = pd.to_numeric(group["residual_doppler_error_hz"], errors="coerce")
            group = group[residual.fillna(0.0).eq(0.0)]
        if group.empty:
            ax.text(0.5, 0.5, "No zero-Doppler score data", ha="center", va="center")
            ax.set_axis_off()
            continue
        n_init_per_case = (
            group.assign(
                _n_init=pd.to_numeric(group["n_init_samples"], errors="coerce"),
            )
            .dropna(subset=["_n_init"])
            .groupby("tracking_case_id")["_n_init"]
            .nunique()
            .sort_values(ascending=False)
        )
        if n_init_per_case.empty:
            ax.text(0.5, 0.5, "No N_init score data", ha="center", va="center")
            ax.set_axis_off()
            continue
        case_id = n_init_per_case.index[0]
        group = group[group["tracking_case_id"].eq(case_id)]
        n_init_values = sorted(
            pd.to_numeric(group["n_init_samples"], errors="coerce").dropna().unique()
        )
        n_init_cmap = plt.get_cmap(CONFIG_CMAP)
        n_init_colors = n_init_cmap(np.linspace(0.18, 0.86, max(len(n_init_values), 1)))
        for style_idx, n_init in enumerate(n_init_values):
            n_group = group[pd.to_numeric(group["n_init_samples"], errors="coerce").eq(n_init)]
            if n_group.empty:
                continue
            curve = n_group.sort_values("fresnel_time_candidate_s")
            color = n_init_colors[style_idx]
            linestyle = N_INIT_LINESTYLES[style_idx % len(N_INIT_LINESTYLES)]
            label = rf"$J(T_{{\mathrm{{F}}}}),\ N_{{\mathrm{{init}}}}={int(n_init)}$"
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
                label=rf"$\hat{{T}}_{{\mathrm{{F}}}},\ N_{{\mathrm{{init}}}}={int(n_init)}$",
            )
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
        ax.set_title(str(severity).capitalize())
        ax.set_xlabel(r"$T_{\mathrm{F}}$ [s]")
    axes[0].set_ylabel(r"$J\left(T_{\mathrm{F}}\right)$")
    axes[-1].legend()
    save_figure(fig, path)


def _draw_tf_error_boxplot(estimates: pd.DataFrame, ax) -> None:
    """Draw estimated T_F boxplots against true CPSSM T_F.

    Main paper:
        Results Figure 3 compares selected T_F estimates against the true CPSSM
        T_F across the dataset.
    """
    if estimates.empty:
        ax.text(0.5, 0.5, "No filter-bank estimate data", ha="center", va="center")
        ax.set_axis_off()
        return
    bins = np.linspace(
        float(estimates["true_fresnel_time_scale_s"].min()),
        float(estimates["true_fresnel_time_scale_s"].max()),
        13,
    )
    centers = 0.5 * (bins[:-1] + bins[1:])
    bin_width = float(np.median(np.diff(centers))) if centers.size > 1 else 0.1
    offsets = np.linspace(-0.27, 0.27, len(SEVERITY_ORDER)) * bin_width
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


def _plot_n_init_sensitivity(df: pd.DataFrame, estimates: pd.DataFrame, path: Path) -> None:
    """Plot N_init sensitivity and the T_F estimate distribution.

    Main paper:
        Results Figure 4 justifies the fixed initialization length used in
        later filter-bank and near-real-time experiments.
    """
    if df.empty:
        placeholder(path, "No initialization sensitivity data")
        return
    fig, axes = plt.subplots(
        1,
        3,
        figsize=grid_figure_size(3, 1, panel_height_cm=5.4),
        sharex=False,
    )
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
    for ax in axes[:2]:
        ax.set_xlabel(r"$N_{\mathrm{init}}$ [samples]")
    axes[1].legend(ncol=1)
    _draw_tf_error_boxplot(estimates, axes[2])
    save_figure(fig, path)
