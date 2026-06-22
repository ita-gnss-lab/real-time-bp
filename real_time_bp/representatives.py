from __future__ import annotations

from pathlib import Path

import pandas as pd

from real_time_bp.config import CSV_RESULTS_DIR

SEVERITY_ORDER = ("weak", "moderate", "strong")
SEVERITY_COLORS = {
    "weak": "tab:green",
    "moderate": "tab:orange",
    "strong": "tab:red",
}


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def ordered_severities(df: pd.DataFrame) -> list[str]:
    if df.empty or "severity" not in df.columns:
        return []
    present = {str(value) for value in df["severity"].dropna().unique()}
    ordered = [severity for severity in SEVERITY_ORDER if severity in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def severity_groups(df: pd.DataFrame):
    severity_values = df["severity"].astype(str)
    for severity in ordered_severities(df):
        yield severity, df[severity_values.eq(severity)]


def sort_by_severity(df: pd.DataFrame, extra_columns: list[str] | None = None) -> pd.DataFrame:
    if df.empty or "severity" not in df.columns:
        return df
    extra_columns = extra_columns or []
    ordered = pd.Categorical(df["severity"].astype(str), categories=SEVERITY_ORDER, ordered=True)
    sort_columns = [
        column
        for column in ["_severity_order", *extra_columns]
        if column == "_severity_order" or column in df.columns
    ]
    return (
        df.assign(_severity_order=ordered)
        .sort_values(sort_columns)
        .drop(columns="_severity_order")
    )


def load_representative_case_ids(csv_dir: Path = CSV_RESULTS_DIR) -> dict[str, int]:
    path = csv_dir / "02_cycle_slips" / "tvd_representative_cases.csv"
    if not path.exists() or path.stat().st_size == 0:
        return {}
    df = pd.read_csv(path)
    if df.empty or "severity" not in df.columns or "tracking_case_id" not in df.columns:
        return {}
    return {
        str(row.severity): int(row.tracking_case_id)
        for row in sort_by_severity(df).itertuples()
        if pd.notna(row.tracking_case_id)
    }
