"""Generate TVD cycle-slip CSV outputs for the paper results."""

from __future__ import annotations

from loguru import logger
import numpy as np
import pandas as pd

from real_time_bp.methodology.signal_model import (
    detrended_intensity_db,
    estimated_scintillation_phase_rad,
    true_refractive_phase_rad,
    true_scintillation_phase_rad,
)
from real_time_bp.methodology.tvd import detect_cycle_slips_from_tvd
from real_time_bp.utils.io import count_cases, find_dataset_paths, iter_case_dataframes
from real_time_bp.utils.output_rows import sampled_time_rows
from real_time_bp.utils.parallel import ordered_parallel_map
from real_time_bp.utils.params import load_params
from real_time_bp.utils.progress import configure_logging, progress_iter
from real_time_bp.utils.representatives import SEVERITY_ORDER, sort_by_severity
from real_time_bp.utils.stage import (
    BufferedCsvWriter,
    case_keys,
    csv_chunksize_from_params,
    csv_stage_dir,
    csv_write_batch_rows_from_params,
    is_loss_of_lock,
    l1_wavelength_from_params,
    optional_int,
    parallel_workers_from_params,
    write_csv,
)

TVD_TRACKING_COLUMNS = {
    "carrier_phase_est_m",
    "los_phase_m",
    "scint_phase_m",
    "time_sec",
    "tracked_intensity",
    "true_refractive_phase_m",
}
EVENT_COLUMNS = [
    "tracking_case_id",
    "example_idx",
    "c_n0_idx",
    "severity",
    "frequency",
    "svid",
    "input_c_n0_db_hz",
    "init_doppler_error_hz",
    "loss_of_lock",
    "true_fresnel_time_scale_s",
    "sample_index",
    "time_sec",
    "integer_cycle_bias",
    "integer_cycle_jump",
    "is_fade_minimum",
    "intensity_db",
]
SUMMARY_COLUMNS = [
    "tracking_case_id",
    "example_idx",
    "c_n0_idx",
    "severity",
    "frequency",
    "svid",
    "input_c_n0_db_hz",
    "init_doppler_error_hz",
    "loss_of_lock",
    "true_fresnel_time_scale_s",
    "n_samples",
    "n_fade_minima",
    "n_tvd_cycle_slips",
    "cycle_slips_per_minute",
    "diffractive_phase_slope_cycles_per_s",
    "diffractive_phase_span_cycles",
]
TRACE_COLUMNS = [
    "tracking_case_id",
    "example_idx",
    "c_n0_idx",
    "severity",
    "frequency",
    "svid",
    "input_c_n0_db_hz",
    "init_doppler_error_hz",
    "loss_of_lock",
    "true_fresnel_time_scale_s",
    "sample_index",
    "time_sec",
    "intensity_db",
    "diffractive_phase_cycles",
    "estimated_diffractive_phase_cycles",
    "true_diffractive_phase_cycles",
    "integer_cycle_bias",
    "is_fade_minimum",
]
REPRESENTATIVE_COLUMNS = [
    "tracking_case_id",
    "example_idx",
    "c_n0_idx",
    "severity",
    "frequency",
    "svid",
    "input_c_n0_db_hz",
    "init_doppler_error_hz",
    "loss_of_lock",
    "true_fresnel_time_scale_s",
    "n_tvd_cycle_slips",
    "cycle_slips_per_minute",
    "diffractive_phase_slope_cycles_per_s",
    "abs_diffractive_phase_slope_cycles_per_s",
    "diffractive_phase_span_cycles",
]
STAGE_DIR = "02_cycle_slips"
EVENTS_CSV = "tvd_cycle_slip_events.csv"
SUMMARY_CSV = "tvd_cycle_slip_summary.csv"
REPRESENTATIVES_CSV = "tvd_representative_cases.csv"
TRACES_CSV = "tvd_example_traces.csv"


def main() -> None:
    """Run the TVD cycle-slip stage and write events, summaries, and traces."""
    configure_logging()
    logger.info("Starting TVD cycle-slip stage")
    params = load_params()
    experiments = params.get("experiments", {})
    tvd_cfg = params.get("tvd", {})
    out_dir = csv_stage_dir(STAGE_DIR)
    paths = find_dataset_paths(params)
    logger.info("Using dataset root {}", paths.root)
    wavelength_m = l1_wavelength_from_params(params)
    max_cases = optional_int(experiments.get("max_cases_per_condition"))
    chunksize = csv_chunksize_from_params(params)
    write_batch_rows = csv_write_batch_rows_from_params(params)
    workers = parallel_workers_from_params(params)
    logger.info("Using {} parallel worker(s)", workers)
    # Paper: Methodology Section II.2. This DVC stage isolates
    # TVD cycle-slip detection from later BP/filter-bank experiments so Results
    # can report slips and export tikz/matplotlib-ready CSVs independently.
    total_cases = count_cases(
        paths,
        severities=experiments.get("severities"),
        max_cases_per_condition=max_cases,
    )

    case_iter = iter_case_dataframes(
        paths,
        severities=experiments.get("severities"),
        max_cases_per_condition=max_cases,
        chunksize=chunksize,
        columns=TVD_TRACKING_COLUMNS,
    )
    task_iter = _iter_tvd_tasks(case_iter, tvd_cfg, wavelength_m)
    result_iter = ordered_parallel_map(
        _compute_tvd_case,
        task_iter,
        workers=workers,
    )
    best_by_severity: dict[str, tuple[tuple[float, ...], dict]] = {}
    with (
        BufferedCsvWriter(
            out_dir / EVENTS_CSV,
            columns=EVENT_COLUMNS,
            batch_rows=write_batch_rows,
        ) as event_writer,
        BufferedCsvWriter(
            out_dir / SUMMARY_CSV,
            columns=SUMMARY_COLUMNS,
            batch_rows=write_batch_rows,
        ) as summary_writer,
    ):
        for result in progress_iter(
            result_iter,
            desc="TVD cases",
            total=total_cases,
            unit="case",
        ):
            event_writer.write_many(result["event_rows"])
            summary_row = result["summary_row"]
            summary_writer.write(summary_row)
            rank_key = _representative_rank_key(summary_row)
            if rank_key is not None:
                severity = str(summary_row.get("severity"))
                current = best_by_severity.get(severity)
                if current is None or rank_key < current[0]:
                    best_by_severity[severity] = (rank_key, summary_row)

    representative_rows = _representative_rows(best_by_severity)
    write_csv(
        pd.DataFrame(representative_rows, columns=REPRESENTATIVE_COLUMNS),
        out_dir / REPRESENTATIVES_CSV,
    )
    selected_case_ids = {
        int(row["tracking_case_id"])
        for row in representative_rows
        if row.get("tracking_case_id") is not None
    }
    trace_case_iter = iter_case_dataframes(
        paths,
        severities=experiments.get("severities"),
        max_cases_per_condition=None,
        chunksize=chunksize,
        columns=TVD_TRACKING_COLUMNS,
    )
    with BufferedCsvWriter(
        out_dir / TRACES_CSV,
        columns=TRACE_COLUMNS,
        batch_rows=write_batch_rows,
    ) as trace_writer:
        for trace_rows in _iter_representative_tvd_traces(
            trace_case_iter,
            selected_case_ids,
            float(wavelength_m),
            float(tvd_cfg.get("fade_threshold_db", -15.0)),
            float(tvd_cfg.get("allowed_jump_weight", 1.0)),
            float(tvd_cfg.get("blocked_jump_weight", 10000.0)),
        ):
            trace_writer.write_many(trace_rows)
    logger.info("Finished TVD cycle-slip stage")


def _iter_tvd_tasks(case_iter, tvd_cfg: dict, wavelength_m: float):
    for metadata, case_df in case_iter:
        yield (
            metadata,
            case_df,
            wavelength_m,
            float(tvd_cfg.get("fade_threshold_db", -15.0)),
            float(tvd_cfg.get("allowed_jump_weight", 1.0)),
            float(tvd_cfg.get("blocked_jump_weight", 10000.0)),
        )


def _compute_tvd_case(task: tuple) -> dict[str, object]:
    (
        metadata,
        case_df,
        wavelength_m,
        threshold_db,
        allowed_jump_weight,
        blocked_jump_weight,
    ) = task
    time_sec = case_df["time_sec"].to_numpy(dtype=float)
    intensity_db = detrended_intensity_db(case_df)
    # Paper: Methodology Eq. (12); subtract
    # the stored LOS phase before removing the known refractive component for
    # the diffractive-phase TVD input.
    estimated_phase = estimated_scintillation_phase_rad(case_df, float(wavelength_m))
    true_refractive = true_refractive_phase_rad(case_df, float(wavelength_m))
    diffractive_phase = estimated_phase - true_refractive
    diffractive_phase_cycles = diffractive_phase / (2.0 * np.pi)
    fade_minima, integer_cycle_bias, cycle_slip_indices = detect_cycle_slips_from_tvd(
        diffractive_phase,
        intensity_db,
        float(threshold_db),
        float(allowed_jump_weight),
        float(blocked_jump_weight),
    )
    fade_minima_set = set(int(x) for x in fade_minima)
    metadata_keys = case_keys(
        metadata,
        include_loss_of_lock=True,
        include_true_fresnel_time=True,
    )

    # Paper/TVD output convention: slips are recorded at fade minima where the
    # rounded TVD cycle-bias staircase changes by a nonzero integer.
    event_rows = []
    for idx in cycle_slip_indices:
        jump_to_idx = min(int(idx) + 1, integer_cycle_bias.size - 1)
        integer_cycle_jump = int(integer_cycle_bias[jump_to_idx] - integer_cycle_bias[int(idx)])
        event_rows.append(
            {
                **metadata_keys,
                "sample_index": int(idx),
                "time_sec": float(time_sec[idx]),
                "integer_cycle_bias": int(integer_cycle_bias[jump_to_idx]),
                "integer_cycle_jump": integer_cycle_jump,
                "is_fade_minimum": int(idx) in fade_minima_set,
                "intensity_db": float(intensity_db[idx]),
            }
        )

    duration_min = (time_sec[-1] - time_sec[0]) / 60.0 if time_sec.size > 1 else np.nan
    # Paper: Results Section III.2; this summary supports CS/min,
    # phase-slope diagnostics, and per-condition aggregation.
    summary_row = {
        **metadata_keys,
        "n_samples": int(case_df.shape[0]),
        "n_fade_minima": int(fade_minima.size),
        "n_tvd_cycle_slips": int(cycle_slip_indices.size),
        "cycle_slips_per_minute": float(cycle_slip_indices.size / duration_min)
        if duration_min and duration_min > 0
        else np.nan,
        "diffractive_phase_slope_cycles_per_s": _linear_slope_cycles_per_s(
            time_sec,
            diffractive_phase_cycles,
        ),
        "diffractive_phase_span_cycles": float(
            np.nanmax(diffractive_phase_cycles) - np.nanmin(diffractive_phase_cycles)
        )
        if diffractive_phase_cycles.size
        else np.nan,
    }

    return {
        "event_rows": event_rows,
        "summary_row": summary_row,
    }


def _iter_representative_tvd_traces(
    case_iter,
    selected_case_ids: set[int],
    wavelength_m: float,
    threshold_db: float,
    allowed_jump_weight: float,
    blocked_jump_weight: float,
):
    yielded: set[int] = set()
    for metadata, case_df in case_iter:
        case_id = int(metadata.get("tracking_case_id", 0) or 0)
        if case_id not in selected_case_ids:
            continue
        yield _compute_tvd_trace_rows(
            metadata,
            case_df,
            wavelength_m,
            threshold_db,
            allowed_jump_weight,
            blocked_jump_weight,
        )
        yielded.add(case_id)
        if yielded == selected_case_ids:
            return


def _compute_tvd_trace_rows(
    metadata: dict,
    case_df,
    wavelength_m: float,
    threshold_db: float,
    allowed_jump_weight: float,
    blocked_jump_weight: float,
) -> list[dict]:
    intensity_db = detrended_intensity_db(case_df)
    estimated_phase = estimated_scintillation_phase_rad(case_df, float(wavelength_m))
    true_scintillation = true_scintillation_phase_rad(case_df, float(wavelength_m))
    true_refractive = true_refractive_phase_rad(case_df, float(wavelength_m))
    estimated_diffractive_cycles = (estimated_phase - true_refractive) / (2.0 * np.pi)
    true_diffractive_cycles = (true_scintillation - true_refractive) / (2.0 * np.pi)
    fade_minima, integer_cycle_bias, _ = detect_cycle_slips_from_tvd(
        estimated_phase - true_refractive,
        intensity_db,
        float(threshold_db),
        float(allowed_jump_weight),
        float(blocked_jump_weight),
    )
    fade_minima_set = set(int(x) for x in fade_minima)
    metadata_keys = case_keys(
        metadata,
        include_loss_of_lock=True,
        include_true_fresnel_time=True,
    )
    return sampled_time_rows(
        case_df,
        metadata_keys,
        lambda idx: {
            "intensity_db": float(intensity_db[idx]),
            "diffractive_phase_cycles": float(estimated_diffractive_cycles[idx]),
            "estimated_diffractive_phase_cycles": float(estimated_diffractive_cycles[idx]),
            "true_diffractive_phase_cycles": float(true_diffractive_cycles[idx]),
            "integer_cycle_bias": int(integer_cycle_bias[idx]),
            "is_fade_minimum": idx in fade_minima_set,
        },
    )


def _representative_rank_key(summary_row: dict) -> tuple[float, ...] | None:
    if is_loss_of_lock(summary_row):
        return None
    severity = str(summary_row.get("severity"))
    n_slips = int(summary_row.get("n_tvd_cycle_slips") or 0)
    slip_penalty = 1.0 if severity == "strong" and n_slips <= 0 else 0.0
    c_n0 = float(summary_row.get("input_c_n0_db_hz") or 0.0)
    slope = abs(float(summary_row.get("diffractive_phase_slope_cycles_per_s") or 0.0))
    case_id = float(summary_row.get("tracking_case_id") or 0.0)
    return (slip_penalty, -c_n0, slope, case_id)


def _representative_rows(
    best_by_severity: dict[str, tuple[tuple[float, ...], dict]],
) -> list[dict]:
    rows = []
    for severity in SEVERITY_ORDER:
        if severity not in best_by_severity:
            continue
        row = dict(best_by_severity[severity][1])
        row["abs_diffractive_phase_slope_cycles_per_s"] = abs(
            float(row.get("diffractive_phase_slope_cycles_per_s") or 0.0)
        )
        rows.append(row)
    return sort_by_severity(pd.DataFrame(rows)).to_dict("records") if rows else []


def _linear_slope_cycles_per_s(time_sec: np.ndarray, phase_cycles: np.ndarray) -> float:
    if time_sec.size < 2 or phase_cycles.size < 2:
        return float("nan")
    time_rel = time_sec - time_sec[0]
    finite = np.isfinite(time_rel) & np.isfinite(phase_cycles)
    if finite.sum() < 2:
        return float("nan")
    return float(np.polyfit(time_rel[finite], phase_cycles[finite], 1)[0])


if __name__ == "__main__":
    main()
