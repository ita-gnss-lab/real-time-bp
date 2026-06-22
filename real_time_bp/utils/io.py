from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from real_time_bp.config import PROJECT_ROOT

TRACKING_REQUIRED_COLUMNS = {
    "tracking_case_id",
    "example_idx",
    "c_n0_idx",
    "epoch_idx",
    "time_sec",
    "svid",
    "frequency",
    "input_c_n0_db_hz",
    "init_doppler_error_hz",
    "carrier_phase_est_m",
    "los_phase_m",
    "scint_phase_m",
    "true_refractive_phase_m",
    "tracked_intensity",
    "estimated_c_n0_db_hz",
    "true_intensity",
    "severity",
}
TRACKING_BASE_REQUIRED_COLUMNS = {
    "tracking_case_id",
    "example_idx",
    "c_n0_idx",
    "epoch_idx",
    "time_sec",
    "svid",
    "frequency",
    "input_c_n0_db_hz",
    "init_doppler_error_hz",
    "severity",
}
TRACKING_READ_DTYPES = {
    "tracking_case_id": "int64",
    "example_idx": "int32",
    "c_n0_idx": "int32",
    "epoch_idx": "int32",
    "time_sec": "float64",
    "input_c_n0_db_hz": "float64",
    "init_doppler_error_hz": "float64",
    "carrier_phase_est_m": "float64",
    "los_phase_m": "float64",
    "scint_phase_m": "float64",
    "true_refractive_phase_m": "float64",
    "tracked_intensity": "float64",
    "estimated_c_n0_db_hz": "float64",
    "true_intensity": "float64",
    "rhof_veff_ratio": "float64",
}


@dataclass(frozen=True)
class DatasetPaths:
    root: Path
    layout: str
    diagnostics_csv: Path
    measurement_csvs: tuple[Path, ...]


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def find_dataset_paths(params: dict) -> DatasetPaths:
    dataset_cfg = params.get("dataset", {})
    candidates = [resolve_path(dataset_cfg.get("output_dir", "data/raw/backpropagation"))]
    for fallback in dataset_cfg.get("local_fallback_dirs", []) or []:
        candidates.append(resolve_path(fallback))

    for root in candidates:
        tracking_root = _find_tracking_root(root)
        if tracking_root is not None:
            return tracking_root
    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Unable to locate tracking dataset. Searched:\n{searched}")


def _find_tracking_root(root: Path) -> DatasetPaths | None:
    if not root.exists():
        return None
    possible_roots = [
        root,
        root / "tracking_outputs",
        root / "tracking_outputs" / "main",
        root / "observables",
        root / "observables" / "main",
    ]
    possible_roots.extend(root.glob("*/tracking_outputs/main"))
    possible_roots.extend(root.glob("*/observables/main"))

    for candidate in possible_roots:
        if not candidate.exists():
            continue
        diagnostics = candidate / "diagnostics.csv"
        tracking_files = tuple(sorted(candidate.glob("tracking_measurements_*.csv")))
        if diagnostics.exists() and tracking_files:
            return DatasetPaths(candidate, "tracking_outputs", diagnostics, tracking_files)
        observable_files = tuple(sorted(candidate.glob("observables_*.csv")))
        if diagnostics.exists() and observable_files:
            return DatasetPaths(candidate, "observables", diagnostics, observable_files)
    return None


def load_diagnostics(paths: DatasetPaths) -> pd.DataFrame:
    diagnostics = pd.read_csv(paths.diagnostics_csv)
    for column in ["tracking_case_id", "example_idx", "c_n0_idx"]:
        if column in diagnostics:
            diagnostics[column] = pd.to_numeric(diagnostics[column], errors="coerce").astype("Int64")
    return diagnostics


def load_case_index(paths: DatasetPaths) -> pd.DataFrame:
    diagnostics = load_diagnostics(paths)
    if paths.layout == "tracking_outputs":
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

    rows: list[dict] = []
    for csv_path in paths.measurement_csvs:
        head = pd.read_csv(csv_path, nrows=1)
        if head.empty:
            continue
        row = head.iloc[0].to_dict()
        matching = diagnostics[
            (diagnostics.get("city", "") == row.get("city"))
            & (diagnostics.get("svid", "") == row.get("svid"))
            & (diagnostics.get("frequency", "") == row.get("frequency"))
            & (diagnostics.get("severity", "") == row.get("severity"))
        ]
        meta = matching.iloc[0].to_dict() if not matching.empty else {}
        rows.append({**meta, **{k: row.get(k) for k in ["city", "svid", "frequency", "severity"]}})
    return pd.DataFrame(rows)


def count_cases(
    paths: DatasetPaths,
    severities: list[str] | None = None,
    max_cases_per_condition: int | None = None,
) -> int | None:
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
        case_index.groupby(condition_columns, dropna=False).size().clip(upper=max_cases_per_condition).sum()
    )


def iter_case_dataframes(
    paths: DatasetPaths,
    severities: list[str] | None = None,
    max_cases_per_condition: int | None = None,
    chunksize: int = 250_000,
    columns: set[str] | None = None,
) -> Iterator[tuple[dict, pd.DataFrame]]:
    diagnostics = load_diagnostics(paths)
    diagnostics_by_case = (
        diagnostics.set_index("tracking_case_id", drop=False)
        if "tracking_case_id" in diagnostics.columns
        else pd.DataFrame()
    )
    severity_filter = set(severities or [])
    yielded_by_condition: dict[tuple[str, float | str], int] = {}

    for csv_path in paths.measurement_csvs:
        if paths.layout == "tracking_outputs":
            iterator = _iter_tracking_cases_from_csv(csv_path, chunksize, columns)
        else:
            iterator = _iter_observable_cases_from_csv(csv_path, diagnostics, chunksize)

        for metadata, case_df in iterator:
            if not diagnostics_by_case.empty and "tracking_case_id" in metadata:
                case_id = metadata["tracking_case_id"]
                if case_id in diagnostics_by_case.index:
                    metadata = {**metadata, **diagnostics_by_case.loc[case_id].to_dict()}
            severity = str(metadata.get("severity", case_df["severity"].iloc[0]))
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


def _iter_observable_cases_from_csv(
    csv_path: Path,
    diagnostics: pd.DataFrame,
    chunksize: int,
) -> Iterator[tuple[dict, pd.DataFrame]]:
    key_columns = ["city", "svid", "frequency", "severity"]
    pending = pd.DataFrame()
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        if not pending.empty:
            chunk = pd.concat([pending, chunk], ignore_index=True)
            pending = pd.DataFrame()
        key_frame = chunk[key_columns].astype(str).agg("|".join, axis=1)
        last_key = key_frame.iloc[-1]
        complete = chunk[key_frame != last_key]
        pending = chunk[key_frame == last_key].copy()
        for _, group in complete.groupby(key_columns, sort=False):
            metadata = _metadata_from_observable_case(group, diagnostics)
            yield metadata, _normalize_observable_case(group, metadata)
    if not pending.empty:
        metadata = _metadata_from_observable_case(pending, diagnostics)
        yield metadata, _normalize_observable_case(pending, metadata)


def _normalize_tracking_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    missing = TRACKING_BASE_REQUIRED_COLUMNS - set(chunk.columns)
    if missing:
        raise ValueError(f"Missing required tracking metadata columns: {sorted(missing)}")
    return chunk.sort_values(["tracking_case_id", "epoch_idx"]).reset_index(drop=True)


def _metadata_from_case(case_df: pd.DataFrame) -> dict:
    first = case_df.iloc[0]
    metadata = {
        "tracking_case_id": int(first["tracking_case_id"]),
        "example_idx": int(first["example_idx"]),
        "c_n0_idx": int(first["c_n0_idx"]),
        "svid": str(first["svid"]),
        "frequency": str(first["frequency"]),
        "severity": str(first["severity"]),
        "input_c_n0_db_hz": float(first["input_c_n0_db_hz"]),
        "init_doppler_error_hz": float(first["init_doppler_error_hz"]),
    }
    for column in ["rhof_veff_ratio"]:
        if column in case_df.columns:
            metadata[column] = float(first[column])
    return metadata


def _metadata_from_observable_case(case_df: pd.DataFrame, diagnostics: pd.DataFrame) -> dict:
    first = case_df.iloc[0]
    metadata = {
        "city": str(first.get("city", "")),
        "svid": str(first["svid"]),
        "frequency": str(first["frequency"]),
        "severity": str(first["severity"]),
        "input_c_n0_db_hz": np.nan,
        "init_doppler_error_hz": np.nan,
    }
    if not diagnostics.empty:
        mask = (
            (diagnostics.get("city", "") == metadata["city"])
            & (diagnostics.get("svid", "") == metadata["svid"])
            & (diagnostics.get("frequency", "") == metadata["frequency"])
            & (diagnostics.get("severity", "") == metadata["severity"])
        )
        match = diagnostics[mask]
        if not match.empty:
            metadata.update(match.iloc[0].to_dict())
    return metadata


def _normalize_observable_case(case_df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    df = case_df.copy()
    df["tracking_case_id"] = int(metadata.get("tracking_case_id", 0) or 0)
    df["example_idx"] = int(metadata.get("example_idx", 0) or 0)
    df["c_n0_idx"] = int(metadata.get("c_n0_idx", 0) or 0)
    df["input_c_n0_db_hz"] = float(metadata.get("input_c_n0_db_hz", np.nan))
    df["init_doppler_error_hz"] = float(metadata.get("init_doppler_error_hz", np.nan))
    df["estimated_c_n0_db_hz"] = np.nan
    return df[
        [
            "tracking_case_id",
            "example_idx",
            "c_n0_idx",
            "epoch_idx",
            "time_sec",
            "svid",
            "frequency",
            "input_c_n0_db_hz",
            "init_doppler_error_hz",
            "carrier_phase_est_m",
            "los_phase_m",
            "elevation_deg",
            "scint_phase_m",
            "true_refractive_phase_m",
            "tracked_intensity",
            "estimated_c_n0_db_hz",
            "true_intensity",
            "severity",
        ]
    ].sort_values("epoch_idx").reset_index(drop=True)
