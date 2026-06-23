"""Dataset discovery and streaming readers for tracking-output CSVs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from real_time_bp.config import PROJECT_ROOT

TRACKING_BASE_REQUIRED_COLUMNS = {
    "tracking_case_id",
    "epoch_idx",
    "time_sec",
}
TRACKING_READ_DTYPES = {
    "tracking_case_id": "int64",
    "epoch_idx": "int32",
    "time_sec": "float64",
    "estimated_scintillation_phase_m": "float64",
    "scint_phase_m": "float64",
    "true_refractive_phase_m": "float64",
    "tracked_intensity": "float64",
    "true_intensity": "float64",
}


@dataclass(frozen=True)
class DatasetPaths:
    """Resolved paths for a supported tracking-output dataset layout."""

    root: Path
    layout: str
    diagnostics_csv: Path
    measurement_csvs: tuple[Path, ...]


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root when needed.

    Args:
        path: Absolute or project-relative path.

    Returns:
        Absolute resolved path.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def find_dataset_paths(params: dict) -> DatasetPaths:
    """Find the tracking-output dataset described by params.

    Args:
        params: Parsed `params.yaml` dictionary.

    Returns:
        Resolved dataset paths.

    Raises:
        FileNotFoundError: If no supported tracking-output layout is found.
    """
    dataset_cfg = params.get("dataset", {})
    root = resolve_path(dataset_cfg.get("output_dir", "data/raw/backpropagation"))
    tracking_root = _find_tracking_root(root)
    if tracking_root is None:
        raise FileNotFoundError(f"Unable to locate tracking dataset under {root}")
    return tracking_root


def _find_tracking_root(root: Path) -> DatasetPaths | None:
    if not root.exists():
        return None
    possible_roots = [
        root,
        root / "tracking_outputs",
        root / "tracking_outputs" / "main",
    ]
    possible_roots.extend(root.glob("*/tracking_outputs/main"))

    for candidate in possible_roots:
        if not candidate.exists():
            continue
        diagnostics = candidate / "diagnostics.csv"
        tracking_files = tuple(sorted(candidate.glob("tracking_measurements_*.csv")))
        if diagnostics.exists() and tracking_files:
            return DatasetPaths(candidate, "tracking_outputs", diagnostics, tracking_files)
    return None


def load_diagnostics(paths: DatasetPaths) -> pd.DataFrame:
    """Load diagnostics metadata for all tracking cases.

    Args:
        paths: Resolved dataset paths.

    Returns:
        Diagnostics DataFrame with integer identifier columns normalized.
    """
    diagnostics = pd.read_csv(paths.diagnostics_csv)
    for column in ["tracking_case_id", "example_idx", "c_n0_idx"]:
        if column in diagnostics:
            diagnostics[column] = pd.to_numeric(diagnostics[column], errors="coerce").astype(
                "Int64"
            )
    return diagnostics


def load_case_index(paths: DatasetPaths) -> pd.DataFrame:
    """Load one metadata row per tracking case.

    Args:
        paths: Resolved dataset paths.

    Returns:
        Case-index DataFrame containing only columns needed by stages.
    """
    diagnostics = load_diagnostics(paths)
    columns = [
        column
        for column in [
            "tracking_case_id",
            "example_idx",
            "c_n0_idx",
            "case_seed",
            "svid",
            "frequency",
            "severity",
            "input_c_n0_db_hz",
            "rhof_veff_ratio",
            "init_doppler_error_hz",
            "estimated_c_n0_db_hz_mean",
            "estimated_c_n0_db_hz_median",
            "loss_of_lock",
        ]
        if column in diagnostics.columns
    ]
    return diagnostics[columns].copy()


def count_cases(
    paths: DatasetPaths,
    severities: list[str] | None = None,
    max_cases_per_condition: int | None = None,
) -> int | None:
    """Count cases after applying severity and per-condition limits.

    Args:
        paths: Resolved dataset paths.
        severities: Optional severity labels to keep.
        max_cases_per_condition: Optional limit per severity/CN0 condition.

    Returns:
        Case count, or None when no case index is available.
    """
    case_index = load_case_index(paths)
    if case_index.empty:
        return None
    if "tracking_case_id" in case_index.columns:
        case_index = case_index.drop_duplicates("tracking_case_id")
    severity_filter = set(severities or [])
    if severity_filter and "severity" in case_index.columns:
        case_index = case_index[case_index["severity"].astype(str).isin(severity_filter)]
    if max_cases_per_condition is None:
        return int(case_index.shape[0])

    if "input_c_n0_db_hz" in case_index.columns:
        condition_columns = ["severity", "input_c_n0_db_hz"]
    else:
        condition_columns = ["severity"]
    return int(
        case_index.groupby(condition_columns, dropna=False)
        .size()
        .clip(upper=max_cases_per_condition)
        .sum()
    )


def iter_case_dataframes(
    paths: DatasetPaths,
    severities: list[str] | None = None,
    max_cases_per_condition: int | None = None,
    chunksize: int = 250_000,
    columns: set[str] | None = None,
) -> Iterator[tuple[dict, pd.DataFrame]]:
    """Stream tracking cases as individual DataFrames.

    Args:
        paths: Resolved dataset paths.
        severities: Optional severity labels to keep.
        max_cases_per_condition: Optional limit per severity/CN0 condition.
        chunksize: CSV chunk size used while reading measurement files.
        columns: Optional measurement columns to load in addition to metadata.

    Yields:
        Pairs of case metadata and a DataFrame containing one complete case.
    """
    diagnostics = load_diagnostics(paths)
    diagnostics_by_case = (
        diagnostics.set_index("tracking_case_id", drop=False)
        if "tracking_case_id" in diagnostics.columns
        else pd.DataFrame()
    )
    severity_filter = set(severities or [])
    yielded_by_condition: dict[tuple[str, float | str], int] = {}

    for csv_path in paths.measurement_csvs:
        for metadata, case_df in _iter_tracking_cases_from_csv(csv_path, chunksize, columns):
            if not diagnostics_by_case.empty and "tracking_case_id" in metadata:
                case_id = metadata["tracking_case_id"]
                if case_id in diagnostics_by_case.index:
                    metadata = {**metadata, **diagnostics_by_case.loc[case_id].to_dict()}
            if "severity" in metadata:
                severity = str(metadata["severity"])
            elif "severity" in case_df.columns:
                severity = str(case_df["severity"].iloc[0])
            else:
                severity = ""
            if severity_filter and severity not in severity_filter:
                continue
            c_n0 = metadata.get("input_c_n0_db_hz", np.nan)
            condition = (severity, float(c_n0) if pd.notna(c_n0) else "nan")
            if max_cases_per_condition is not None:
                if yielded_by_condition.get(condition, 0) >= int(max_cases_per_condition):
                    continue
                yielded_by_condition[condition] = yielded_by_condition.get(condition, 0) + 1
            yield metadata, case_df


def _iter_tracking_cases_from_csv(
    csv_path: Path,
    chunksize: int,
    columns: set[str] | None,
) -> Iterator[tuple[dict, pd.DataFrame]]:
    read_kwargs = _tracking_read_kwargs(columns)
    pending = pd.DataFrame()
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, **read_kwargs):
        chunk = _normalize_tracking_chunk(chunk)
        if not pending.empty:
            chunk = pd.concat([pending, chunk], ignore_index=True)
            pending = pd.DataFrame()
        case_ids = chunk["tracking_case_id"].to_numpy()
        if case_ids.size == 0:
            continue
        last_case_id = case_ids[-1]
        complete = chunk[chunk["tracking_case_id"] != last_case_id]
        pending = chunk[chunk["tracking_case_id"] == last_case_id].copy()
        for _, group in complete.groupby("tracking_case_id", sort=False):
            yield _metadata_from_case(group), group.reset_index(drop=True)
    if not pending.empty:
        yield _metadata_from_case(pending), pending.reset_index(drop=True)


def _tracking_read_kwargs(columns: set[str] | None) -> dict:
    if columns is None:
        return {"dtype": TRACKING_READ_DTYPES}
    selected = set(columns) | TRACKING_BASE_REQUIRED_COLUMNS
    return {
        "dtype": TRACKING_READ_DTYPES,
        "usecols": lambda column: column in selected,
    }


def _normalize_tracking_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    missing = TRACKING_BASE_REQUIRED_COLUMNS - set(chunk.columns)
    if missing:
        raise ValueError(f"Missing required tracking metadata columns: {sorted(missing)}")
    return chunk.sort_values(["tracking_case_id", "epoch_idx"]).reset_index(drop=True)


def _metadata_from_case(case_df: pd.DataFrame) -> dict:
    first = case_df.iloc[0]
    metadata = {"tracking_case_id": int(first["tracking_case_id"])}
    for column in ["example_idx", "c_n0_idx"]:
        if column in case_df.columns:
            metadata[column] = int(first[column])
    for column in ["svid", "frequency", "severity"]:
        if column in case_df.columns:
            metadata[column] = str(first[column])
    for column in ["input_c_n0_db_hz", "init_doppler_error_hz"]:
        if column in case_df.columns:
            metadata[column] = float(first[column])
    for column in ["rhof_veff_ratio"]:
        if column in case_df.columns:
            metadata[column] = float(first[column])
    return metadata
