"""Shared stage helpers for paths, parameters, metadata, and CSV writing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger
import numpy as np
import pandas as pd

from real_time_bp.config import CSV_TABLES_DIR, FIGURES_DIR, OUTPUTS_DIR
from real_time_bp.methodology.constants import carrier_wavelength_m
from real_time_bp.utils.representatives import sort_by_severity


def ensure_dir(path: Path) -> Path:
    """Create a directory and return its path.

    Args:
        path: Directory path to create.

    Returns:
        The same path after ensuring it exists.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def csv_stage_dir(name: str) -> Path:
    """Return and create an operational CSV output stage directory.

    Args:
        name: Stage subdirectory below `outputs`.

    Returns:
        Created stage directory path.
    """
    return ensure_dir(OUTPUTS_DIR / name)


def csv_table_dir(name: str = "") -> Path:
    """Return and create a paper-facing CSV table directory.

    Args:
        name: Optional subdirectory below `results/csv_tables`.

    Returns:
        Created table directory path.
    """
    return ensure_dir(CSV_TABLES_DIR / name) if name else ensure_dir(CSV_TABLES_DIR)


def figure_stage_dir(name: str) -> Path:
    """Return and create a figure result stage directory.

    Args:
        name: Stage subdirectory below `results/figures`.

    Returns:
        Created figure directory path.
    """
    return ensure_dir(FIGURES_DIR / name)


def optional_int(value: Any) -> int | None:
    """Convert optional parameter values to integers.

    Args:
        value: Value from params that may be None, empty, NaN, or numeric.

    Returns:
        Integer value, or None when the value is intentionally unset.
    """
    if value is None or value == "":
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return int(value)


def l1_wavelength_from_params(params: dict[str, Any]) -> float:
    """Read GPS L1 wavelength from the constants parameter block.

    Main paper:
        Results Section III.1 uses the L1 wavelength to convert phase meters to
        radians/cycles and residual geometric-rate errors Delta G to residual
        Doppler errors Delta f_D.

    Args:
        params: Parsed `params.yaml` dictionary.

    Returns:
        L1 carrier wavelength in meters.
    """
    constants = params.get("constants", {})
    return carrier_wavelength_m(
        constants.get("speed_of_light_m_s", 299792458.0),
        constants.get("gps_l1_hz", 1575420000.0),
    )


def tf_candidate_grid(params: dict[str, Any]) -> np.ndarray:
    """Build the Fresnel-time candidate grid from experiment parameters.

    Main paper:
        Methodology Section II.4 defines the candidate set
        T={T_F,1,...,T_F,P} used by the filter-bank score.

    Args:
        params: Parsed `params.yaml` dictionary.

    Returns:
        Candidate Fresnel time scales in seconds.
    """
    experiments = params.get("experiments", {})
    return np.linspace(
        float(experiments.get("tf_candidate_min_s", 0.5)),
        float(experiments.get("tf_candidate_max_s", 2.0)),
        int(experiments.get("tf_candidate_count", 36)),
    )


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to CSV and log its shape.

    Args:
        df: DataFrame to write.
        path: Output CSV path.
    """
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
    logger.info("Wrote {} rows x {} columns to {}", df.shape[0], df.shape[1], path)


def write_sorted_csv(
    df: pd.DataFrame,
    path: Path,
    extra_columns: list[str] | None = None,
) -> None:
    """Sort by severity before writing a DataFrame to CSV.

    Args:
        df: DataFrame to sort and write.
        path: Output CSV path.
        extra_columns: Additional sort keys after severity.
    """
    write_csv(sort_by_severity(df, extra_columns), path)


def csv_chunksize_from_params(params: dict[str, Any]) -> int:
    """Read the CSV input chunk size from runtime parameters.

    Args:
        params: Parsed `params.yaml` dictionary.

    Returns:
        Positive CSV chunk size.
    """
    runtime = params.get("runtime", {})
    return max(int(runtime.get("csv_chunksize", 75_000)), 1)


def csv_write_batch_rows_from_params(params: dict[str, Any]) -> int:
    """Read the buffered CSV output batch size from runtime parameters.

    Args:
        params: Parsed `params.yaml` dictionary.

    Returns:
        Positive number of rows per buffered write.
    """
    runtime = params.get("runtime", {})
    return max(int(runtime.get("csv_write_batch_rows", 50_000)), 1)


def parallel_workers_from_params(params: dict[str, Any]) -> int:
    """Read the shared worker count from runtime parameters.

    Implementation:
        All DVC stages share this one process count so runtime tuning does not
        change the methodology semantics.

    Args:
        params: Parsed `params.yaml` dictionary.

    Returns:
        Positive worker count.
    """
    runtime = params.get("runtime", {})
    value = runtime.get("parallel_workers", 1)
    if value is None or value == "":
        return 1
    return max(int(value), 1)


def is_loss_of_lock(metadata: dict) -> bool:
    """Normalize loss-of-lock metadata to a boolean.

    Args:
        metadata: Case metadata dictionary.

    Returns:
        True when the metadata indicates loss of lock.
    """
    value = metadata.get("loss_of_lock", False)
    if value is None:
        return False
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def is_representative_case(metadata: dict, experiments: dict) -> bool:
    """Check whether a case matches the representative C/N0 setting.

    Args:
        metadata: Case metadata dictionary.
        experiments: Experiment parameter dictionary.

    Returns:
        True when the case C/N0 equals the configured representative C/N0.
    """
    c_n0 = metadata.get("input_c_n0_db_hz")
    target_c_n0 = experiments.get("representative_c_n0_db_hz", 30.0)
    return bool(np.isfinite(c_n0) and abs(float(c_n0) - float(target_c_n0)) < 1e-9)


def case_keys(
    metadata: dict,
    *,
    include_loss_of_lock: bool = False,
    include_true_fresnel_time: bool = False,
) -> dict:
    """Extract common metadata fields for result CSV rows.

    Args:
        metadata: Case metadata dictionary.
        include_loss_of_lock: Whether to include normalized lock status.
        include_true_fresnel_time: Whether to include truth T_F from metadata.

    Returns:
        Dictionary of common case-identification fields.
    """
    keys = {
        "tracking_case_id": metadata.get("tracking_case_id"),
        "example_idx": metadata.get("example_idx"),
        "c_n0_idx": metadata.get("c_n0_idx"),
        "severity": metadata.get("severity"),
        "frequency": metadata.get("frequency"),
        "svid": metadata.get("svid"),
        "input_c_n0_db_hz": metadata.get("input_c_n0_db_hz"),
        "init_doppler_error_hz": metadata.get("init_doppler_error_hz"),
    }
    if include_loss_of_lock:
        keys["loss_of_lock"] = is_loss_of_lock(metadata)
    if include_true_fresnel_time:
        keys["true_fresnel_time_scale_s"] = metadata.get("rhof_veff_ratio")
    return keys


class BufferedCsvWriter:
    """Buffered CSV writer that preserves a fixed column order."""

    def __init__(
        self,
        path: Path,
        *,
        columns: list[str],
        batch_rows: int,
    ) -> None:
        """Create a buffered CSV writer.

        Args:
            path: Output CSV path.
            columns: Fixed output column order.
            batch_rows: Number of rows accumulated before each disk write.
        """
        self.path = path
        self.columns = columns
        self.batch_rows = max(int(batch_rows), 1)
        self._buffer: list[dict] = []
        self._row_count = 0
        self._closed = False
        ensure_dir(path.parent)
        if path.exists():
            path.unlink()

    def __enter__(self) -> "BufferedCsvWriter":
        """Enter the writer context.

        Returns:
            This writer instance.
        """
        return self

    def __exit__(self, *args: object) -> None:
        """Flush and close the writer when leaving a context manager."""
        self.close()

    def write(self, row: dict) -> None:
        """Append one row to the buffer.

        Args:
            row: Row dictionary to write.
        """
        self._buffer.append(row)
        if len(self._buffer) >= self.batch_rows:
            self.flush()

    def write_many(self, rows: list[dict]) -> None:
        """Append multiple rows to the buffer.

        Args:
            rows: Row dictionaries to write.
        """
        for row in rows:
            self.write(row)

    def flush(self) -> None:
        """Write buffered rows to disk without closing the writer."""
        if not self._buffer:
            return
        frame = pd.DataFrame.from_records(self._buffer, columns=self.columns)
        frame.to_csv(
            self.path,
            mode="a",
            header=self._row_count == 0,
            index=False,
        )
        self._row_count += frame.shape[0]
        self._buffer.clear()

    def close(self) -> None:
        """Flush remaining rows and mark the writer as closed."""
        if self._closed:
            return
        self.flush()
        if self._row_count == 0:
            pd.DataFrame(columns=self.columns).to_csv(self.path, index=False)
        logger.info("Wrote {} rows to {}", self._row_count, self.path)
        self._closed = True
