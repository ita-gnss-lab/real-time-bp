"""Shared plotting utilities for reading CSV outputs and writing figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pandas.errors import EmptyDataError

from real_time_bp.config import CSV_RESULTS_DIR
from real_time_bp.methodology.filter_bank import AMPLITUDE_SCORE_NAME
from real_time_bp.utils.plot_style import figure_size


def read_csv(relative_path: str) -> pd.DataFrame:
    """Read a CSV result file relative to the results CSV directory.

    Args:
        relative_path: Path below `results/csv`.

    Returns:
        Parsed DataFrame, or an empty DataFrame when the file is missing or empty.
    """
    path = CSV_RESULTS_DIR / relative_path
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


def column(df: pd.DataFrame, preferred: str, fallback: str) -> str:
    """Select a preferred column name with a backward-compatible fallback.

    Args:
        df: DataFrame whose columns are inspected.
        preferred: New column name.
        fallback: Older column name used when `preferred` is absent.

    Returns:
        Column name that exists in `df`, preferring `preferred`.
    """
    return preferred if preferred in df.columns else fallback


def remove_figure(path: Path) -> None:
    """Remove a stale figure if it exists.

    Args:
        path: Figure path to remove.
    """
    if path.exists():
        path.unlink()


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
        fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
