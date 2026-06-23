"""Severity ordering and representative-case helpers for plots and stages."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from real_time_bp.config import CSV_RESULTS_DIR

SEVERITY_ORDER = ("weak", "strong")
SEVERITY_COLORS = {
    "weak": "tab:green",
    "strong": "tab:red",
}


def bool_series(series: pd.Series) -> pd.Series:
    """Convert common truthy string/numeric values to a boolean Series.

    Args:
        series: Series containing booleans, strings, or numeric values.

    Returns:
        Boolean Series with true for `1`, `true`, or `yes`.
    """
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def ordered_severities(df: pd.DataFrame) -> list[str]:
    """Return available severities in the configured paper order.

    Args:
        df: DataFrame with an optional `severity` column.

    Returns:
        Ordered weak/strong severity labels present in the DataFrame.
    """
    if df.empty or "severity" not in df.columns:
        return []
    present = {str(value) for value in df["severity"].dropna().unique()}
    return [severity for severity in SEVERITY_ORDER if severity in present]


def severity_groups(df: pd.DataFrame):
    """Yield DataFrame groups by severity in paper order.

    Args:
        df: DataFrame containing a `severity` column.

    Yields:
        Pairs of severity label and filtered DataFrame.
    """
    severity_values = df["severity"].astype(str)
    for severity in ordered_severities(df):
        yield severity, df[severity_values.eq(severity)]


def sort_by_severity(df: pd.DataFrame, extra_columns: list[str] | None = None) -> pd.DataFrame:
    """Sort and filter a DataFrame by weak/strong severity order.

    Args:
        df: DataFrame containing an optional `severity` column.
        extra_columns: Additional columns used after severity ordering.

    Returns:
        Sorted DataFrame with only configured severities, or the original
        DataFrame when severity is absent.
    """
    if df.empty or "severity" not in df.columns:
        return df
    extra_columns = extra_columns or []
    filtered = df[df["severity"].astype(str).isin(SEVERITY_ORDER)].copy()
    ordered = pd.Categorical(
        filtered["severity"].astype(str), categories=SEVERITY_ORDER, ordered=True
    )
    sort_columns = [
        column
        for column in ["_severity_order", *extra_columns]
        if column == "_severity_order" or column in filtered.columns
    ]
    return (
        filtered.assign(_severity_order=ordered)
        .sort_values(sort_columns)
        .drop(columns="_severity_order")
    )


def load_representative_case_ids(csv_dir: Path = CSV_RESULTS_DIR) -> dict[str, int]:
    """Load representative TVD case ids selected by the cycle-slip stage.

    Args:
        csv_dir: Root CSV results directory.

    Returns:
        Mapping from severity label to tracking case id.
    """
    path = csv_dir / "02_cycle_slips" / "tvd_representative_cases.csv"
    if not path.exists() or path.stat().st_size == 0:
        return {}
    df = pd.read_csv(path)
    if df.empty or "severity" not in df.columns or "tracking_case_id" not in df.columns:
        return {}
    return {
        str(row.severity): int(row.tracking_case_id)
        for row in sort_by_severity(df).itertuples()
        if str(row.severity) in SEVERITY_ORDER and pd.notna(row.tracking_case_id)
    }
