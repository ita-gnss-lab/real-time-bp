"""Shared Matplotlib sizes, colors, and font settings for paper figures."""

from __future__ import annotations

import shutil

CM_PER_INCH = 2.54
BOX_BINS = 10
BOX_WIDTH_FACTOR = 0.7

FONT_SIZES_PT = {
    "base": 10,
    "title": 11,
    "label": 10,
    "tick": 9,
    "legend": 9,
    "annotation": 8,
}

OVERALL_SIZE_SCALE = 2
WIDTH_CM = 6
HEIGHT_CM = 3
SCALED_WIDTH_CM = OVERALL_SIZE_SCALE * WIDTH_CM
SCALED_HEIGHT_CM = OVERALL_SIZE_SCALE * HEIGHT_CM

FIGSIZES_CM = {
    "single": (SCALED_WIDTH_CM, SCALED_HEIGHT_CM),
    "two_panel": (SCALED_WIDTH_CM * 1.65, SCALED_HEIGHT_CM * 1.15),
    "two_row": (SCALED_WIDTH_CM * 1.65, SCALED_HEIGHT_CM * 1.8),
    "boxplot_grid": (SCALED_WIDTH_CM * 2.0, SCALED_HEIGHT_CM * 1.45),
    "heatmap_grid": (SCALED_WIDTH_CM * 2.35, SCALED_HEIGHT_CM * 2.45),
    "placeholder": (SCALED_WIDTH_CM, SCALED_HEIGHT_CM * 0.85),
}

SEVERITY_COLORS = {
    "weak": "#2ca02c",
    "strong": "#d62728",
}

COLORS = {
    "truth": "black",
    "neutral": "0.35",
    "true_scintillation": "#1f77b4",
    "estimated_scintillation": "#17becf",
    "batch": "#9467bd",
    "sliding": "#1f77b4",
    "estimated_signal": "#1f77b4",
    "forecast": "#9467bd",
    "tvd": "#d62728",
    "residual": "#1f77b4",
}

HEATMAP_CMAP = "magma_r"
DOPPLER_CMAP = "viridis"
CONFIG_CMAP = "plasma"
N_INIT_LINESTYLES = ("-", "--", ":")


def figsize_cm(width_cm: float, height_cm: float) -> tuple[float, float]:
    """Convert centimeters to a Matplotlib figsize tuple.

    Args:
        width_cm: Figure width in centimeters.
        height_cm: Figure height in centimeters.

    Returns:
        Figure size in inches.
    """
    return (float(width_cm) / CM_PER_INCH, float(height_cm) / CM_PER_INCH)


def figure_size(name: str) -> tuple[float, float]:
    """Look up a named figure size.

    Args:
        name: Key in `FIGSIZES_CM`.

    Returns:
        Figure size in inches.
    """
    return figsize_cm(*FIGSIZES_CM[name])


def grid_figure_size(
    columns: int,
    rows: int,
    *,
    panel_width_cm: float = 8.8,
    panel_height_cm: float = 4.8,
) -> tuple[float, float]:
    """Compute a grid figure size from panel dimensions.

    Args:
        columns: Number of subplot columns.
        rows: Number of subplot rows.
        panel_width_cm: Width allocated to each panel in centimeters.
        panel_height_cm: Height allocated to each panel in centimeters.

    Returns:
        Figure size in inches.
    """
    return figsize_cm(
        max(float(columns) * panel_width_cm, SCALED_WIDTH_CM), float(rows) * panel_height_cm
    )


def apply_plot_style() -> None:
    """Apply consistent Matplotlib style settings for all generated figures."""
    import matplotlib.pyplot as plt

    use_tex = shutil.which("pdflatex") is not None
    rc = {
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.size": FONT_SIZES_PT["base"],
        "axes.titlesize": FONT_SIZES_PT["title"],
        "axes.labelsize": FONT_SIZES_PT["label"],
        "xtick.labelsize": FONT_SIZES_PT["tick"],
        "ytick.labelsize": FONT_SIZES_PT["tick"],
        "legend.fontsize": FONT_SIZES_PT["legend"],
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
    if use_tex:
        rc.update(
            {
                "text.usetex": True,
                "text.latex.preamble": r"""
                \usepackage{amsmath,amssymb,newtxmath}
                \renewcommand{\rmdefault}{ptm}
                \renewcommand{\sfdefault}{phv}
                \renewcommand{\ttdefault}{cmtt}
                """,
            }
        )
    else:
        rc.update({"mathtext.fontset": "cm"})
    plt.rcParams.update(rc)
