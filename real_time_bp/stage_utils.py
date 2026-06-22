from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger
import numpy as np
import pandas as pd

from real_time_bp.config import CSV_RESULTS_DIR, FIGURES_DIR
from real_time_bp.methodology.constants import carrier_wavelength_m


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def csv_stage_dir(name: str) -> Path:
    return ensure_dir(CSV_RESULTS_DIR / name)


def figure_stage_dir(name: str) -> Path:
    return ensure_dir(FIGURES_DIR / name)


def optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return int(value)


def l1_wavelength_from_params(params: dict[str, Any]) -> float:
    # Paper: Results Section III.1; L1 wavelength is used
    # to convert phase meters to radians/cycles and Delta G to Delta f_D.
    constants = params.get("constants", {})
    return carrier_wavelength_m(
        constants.get("speed_of_light_m_s", 299792458.0),
        constants.get("gps_l1_hz", 1575420000.0),
    )


def residual_doppler_grid_hz(params: dict[str, Any]) -> pd.DataFrame:
    # Paper: Results Section III.1. The residual Doppler
    # sensitivity grid is specified as geometric-rate error Delta G in m/s and
    # converted to Delta f_D=Delta G/lambda for L1.
    experiments = params.get("experiments", {})
    wavelength = l1_wavelength_from_params(params)
    if "residual_doppler_error_hz_max" in experiments:
        residual_doppler_error_hz = np.linspace(
            float(experiments.get("residual_doppler_error_hz_min", 0.0)),
            float(experiments.get("residual_doppler_error_hz_max", 20.0)),
            int(experiments.get("residual_doppler_error_count", 21)),
        )
        geometric_error_mps = residual_doppler_error_hz * wavelength
    else:
        geometric_error_mps = np.linspace(
            float(experiments.get("residual_geometric_error_mps_min", 0.0)),
            float(experiments.get("residual_geometric_error_mps_max", 0.1)),
            int(experiments.get("residual_geometric_error_count", 15)),
        )
        residual_doppler_error_hz = geometric_error_mps / wavelength
    return pd.DataFrame(
        {
            "residual_geometric_error_mps": geometric_error_mps,
            "residual_doppler_error_hz": residual_doppler_error_hz,
        }
    )


def tf_candidate_grid(params: dict[str, Any]) -> np.ndarray:
    # Paper: Methodology Section II.4; this constructs the
    # candidate set T={T_F,1,...,T_F,P} used by every filter-bank score.
    experiments = params.get("experiments", {})
    return np.linspace(
        float(experiments.get("tf_candidate_min_s", 0.5)),
        float(experiments.get("tf_candidate_max_s", 2.0)),
        int(experiments.get("tf_candidate_count", 36)),
    )


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
    logger.info("Wrote {} rows x {} columns to {}", df.shape[0], df.shape[1], path)


def csv_chunksize_from_params(params: dict[str, Any]) -> int:
    runtime = params.get("runtime", {})
    return max(int(runtime.get("csv_chunksize", 75_000)), 1)


def csv_write_batch_rows_from_params(params: dict[str, Any]) -> int:
    runtime = params.get("runtime", {})
    return max(int(runtime.get("csv_write_batch_rows", 50_000)), 1)


def parallel_workers_from_params(params: dict[str, Any]) -> int:
    # Implementation note for DVC stages: all paper experiments share this one
    # process count so runtime tuning does not change methodology semantics.
    runtime = params.get("runtime", {})
    value = runtime.get("parallel_workers", 1)
    if value is None or value == "":
        return 1
    return max(int(value), 1)


class BufferedCsvWriter:
    def __init__(
        self,
        path: Path,
        *,
        columns: list[str],
        batch_rows: int,
    ) -> None:
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
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def write(self, row: dict) -> None:
        self._buffer.append(row)
        if len(self._buffer) >= self.batch_rows:
            self.flush()

    def write_many(self, rows: list[dict]) -> None:
        for row in rows:
            self.write(row)

    def flush(self) -> None:
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
        if self._closed:
            return
        self.flush()
        if self._row_count == 0:
            pd.DataFrame(columns=self.columns).to_csv(self.path, index=False)
        logger.info("Wrote {} rows to {}", self._row_count, self.path)
        self._closed = True
