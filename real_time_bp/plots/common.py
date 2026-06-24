"""Shared plotting utilities for reading operational CSVs and writing figures."""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from pandas.errors import EmptyDataError

from real_time_bp.config import OUTPUTS_DIR
from real_time_bp.methodology.filter_bank import AMPLITUDE_SCORE_NAME
from real_time_bp.utils.plot_style import figure_size


def read_csv(relative_path: str) -> pd.DataFrame:
    """Read an operational CSV file relative to the outputs directory.

    Args:
        relative_path: Path below `outputs`.

    Returns:
        Parsed DataFrame, or an empty DataFrame when the file is missing or empty.
    """
    path = OUTPUTS_DIR / relative_path
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def amplitude_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only amplitude-score rows from a score DataFrame.

    Args:
        df: DataFrame that may contain a `score_name` column.

    Returns:
        Filtered copy when `score_name` is present, otherwise the original frame.
    """
    if df.empty or "score_name" not in df.columns:
        return df
    return df[df["score_name"].eq(AMPLITUDE_SCORE_NAME)].copy()


def placeholder(path: Path, text: str) -> None:
    """Write a placeholder figure when a required CSV is empty.

    Args:
        path: Output figure path.
        text: Centered placeholder text.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figure_size("placeholder"))
    ax.text(0.5, 0.5, text, ha="center", va="center")
    ax.set_axis_off()
    save_figure(fig, path)


def save_figure(fig, path: Path, *, tight: bool = True) -> None:
    """Save a Matplotlib figure and close it.

    Args:
        fig: Matplotlib figure object.
        path: Output path.
        tight: Whether to apply `tight_layout` before saving.
    """
    if tight:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Tight layout not applied.*")
            fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
