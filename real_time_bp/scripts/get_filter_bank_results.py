"""Generate filter-bank Fresnel-time and batch-BP result CSV outputs."""

from __future__ import annotations

from loguru import logger
import numpy as np
import pandas as pd

from real_time_bp.methodology.backpropagation import (
    batch_backpropagate,
    median_align_phase,
    unwrap_phase,
)
from real_time_bp.methodology.filter_bank import (
    AMPLITUDE_SCORE_NAME,
    compute_filter_bank_score,
    select_fresnel_time_estimate,
)
from real_time_bp.methodology.metrics import (
    cycle_slips_per_minute,
    mae,
    mape,
    phase_rmse_rad,
    r2_score,
    rmse,
)
from real_time_bp.methodology.signal_model import (
    detrended_intensity_db,
    estimate_sample_interval_s,
    estimated_scintillation_phase_rad,
    estimated_scintillation_signal,
    true_refractive_phase_rad,
    true_scintillation_phase_rad,
)
from real_time_bp.methodology.tvd import tvd_integer_bias
from real_time_bp.utils.io import count_cases, find_dataset_paths, iter_case_dataframes
from real_time_bp.utils.output_rows import sampled_time_rows
from real_time_bp.utils.parallel import ordered_parallel_map
from real_time_bp.utils.params import fixed_step_values, load_params
from real_time_bp.utils.progress import configure_logging, progress_iter
from real_time_bp.utils.representatives import (
    severity_groups,
    sort_by_severity,
)
from real_time_bp.utils.stage import (
    case_keys,
    csv_chunksize_from_params,
    csv_stage_dir,
    is_representative_case,
    l1_wavelength_from_params,
    optional_int,
    parallel_workers_from_params,
    tf_candidate_grid,
    write_csv,
    write_sorted_csv,
)

FILTER_BANK_TRACKING_COLUMNS = {
    "estimated_scintillation_phase_m",
    "scint_phase_m",
    "time_sec",
    "tracked_intensity",
    "true_refractive_phase_m",
}
STAGE_DIR = "03_filter_bank"
SCORE_CURVES_CSV = "filter_bank_score_curves.csv"
ESTIMATES_CSV = "filter_bank_estimates.csv"
DOWNSTREAM_METRICS_CSV = "filter_bank_downstream_phase_metrics.csv"
REPRESENTATIVE_TRACES_CSV = "filter_bank_representative_phase_traces.csv"
TF_ERROR_METRICS_CSV = "filter_bank_tf_error_metrics.csv"
DOWNSTREAM_SUMMARY_CSV = "filter_bank_downstream_summary.csv"
N_INIT_SENSITIVITY_CSV = "filter_bank_n_init_sensitivity.csv"
BOXPLOT_ESTIMATES_CSV = "filter_bank_boxplot_estimates.csv"
RESIDUAL_DOPPLER_SENSITIVITY_CSV = "filter_bank_residual_doppler_sensitivity.csv"


def main() -> None:
    """Run the filter-bank stage and write score, estimate, and metric CSVs.

    Main paper:
        Results Section III.2 separates the amplitude score curve, T_F
        estimates, downstream phase metrics, and N_init sensitivity into
        independently plottable CSV outputs.
    """
    configure_logging()
    logger.info("Starting filter-bank results stage")
    params = load_params()
    experiments = params.get("experiments", {})
    out_dir = csv_stage_dir(STAGE_DIR)
    paths = find_dataset_paths(params)
    logger.info("Using dataset root {}", paths.root)
    wavelength_m = l1_wavelength_from_params(params)
    candidates = tf_candidate_grid(params)
    score_n_init_samples = _score_curve_n_init_samples(experiments)
    max_cases = optional_int(experiments.get("max_cases_per_condition"))
    chunksize = csv_chunksize_from_params(params)
    workers = parallel_workers_from_params(params)
    logger.info("Using {} parallel worker(s)", workers)
    total_cases = count_cases(
        paths,
        severities=experiments.get("severities"),
        max_cases_per_condition=max_cases,
    )

    score_rows: list[dict] = []
    estimate_rows: list[dict] = []
    downstream_rows: list[dict] = []

    case_iter = iter_case_dataframes(
        paths,
        severities=experiments.get("severities"),
        max_cases_per_condition=max_cases,
        chunksize=chunksize,
        columns=FILTER_BANK_TRACKING_COLUMNS,
    )
    task_iter = _iter_filter_bank_tasks(
        case_iter,
        params,
        candidates,
        wavelength_m,
        residual_doppler_error_hz=0.0,
        n_init=int(experiments.get("n_init_samples", 4096)),
        score_n_init_samples=score_n_init_samples,
        include_representative_traces=False,
        experiments=experiments,
    )
    result_iter = ordered_parallel_map(
        _compute_case_filter_bank_task,
        task_iter,
        workers=workers,
    )
    for result in progress_iter(
        result_iter,
        desc="filter-bank cases",
        total=total_cases,
        unit="case",
    ):
        score_rows.extend(result["score_rows"])
        estimate_rows.extend(result["estimate_rows"])
        downstream_rows.extend(result["downstream_rows"])

    write_sorted_csv(
        pd.DataFrame(score_rows),
        out_dir / SCORE_CURVES_CSV,
        ["tracking_case_id", "n_init_samples", "fresnel_time_candidate_s"],
    )
    estimates_df = sort_by_severity(pd.DataFrame(estimate_rows), ["tracking_case_id"])
    write_csv(estimates_df, out_dir / ESTIMATES_CSV)
    downstream_df = sort_by_severity(pd.DataFrame(downstream_rows), ["tracking_case_id"])
    write_csv(downstream_df, out_dir / DOWNSTREAM_METRICS_CSV)
    write_csv(_aggregate_tf_error_metrics(estimates_df), out_dir / TF_ERROR_METRICS_CSV)
    write_csv(_aggregate_downstream_metrics(downstream_df), out_dir / DOWNSTREAM_SUMMARY_CSV)
    write_csv(
        _compute_n_init_sensitivity(params, paths, candidates, wavelength_m),
        out_dir / N_INIT_SENSITIVITY_CSV,
    )
    write_csv(
        _compute_boxplot_estimates(params, paths, candidates, wavelength_m),
        out_dir / BOXPLOT_ESTIMATES_CSV,
    )
    stale_residual_csv = out_dir / RESIDUAL_DOPPLER_SENSITIVITY_CSV
    if stale_residual_csv.exists():
        stale_residual_csv.unlink()
    logger.info("Finished filter-bank results stage")


def _compute_case_filter_bank(
    metadata: dict,
    case_df: pd.DataFrame,
    params: dict,
    candidates: np.ndarray,
    wavelength_m: float,
    residual_doppler_error_hz: float,
    n_init: int,
    score_n_init_samples: list[int] | None = None,
    include_trace: bool = False,
) -> dict[str, list[dict]]:
    """Compute filter-bank score, T_F estimate, metrics, and trace rows.

    Main paper:
        Methodology Eq. (13) defines the estimated complex scintillation signal
        used by the filter bank. Methodology Section II.4 scores only the first
        N_init samples for each candidate T_F. Results Section III.2 uses the
        emitted rows for score curves, T_F error metrics, downstream phase
        metrics, and representative phase-retrieval traces.
    """
    sample_interval_s = estimate_sample_interval_s(case_df)
    n_init = min(int(n_init), len(case_df))
    signal = estimated_scintillation_signal(case_df, wavelength_m, residual_doppler_error_hz)
    estimated_scintillation_phase = estimated_scintillation_phase_rad(
        case_df,
        wavelength_m,
        residual_doppler_error_hz,
    )
    true_scintillation_phase = true_scintillation_phase_rad(case_df, wavelength_m)
    true_phase = true_refractive_phase_rad(case_df, wavelength_m)
    init_signal = signal[:n_init]
    primary_score_curve = compute_filter_bank_score(
        init_signal,
        sample_interval_s,
        candidates,
    )
    estimated_tf = select_fresnel_time_estimate(candidates, primary_score_curve)
    true_tf = float(metadata.get("rhof_veff_ratio", np.nan))
    base = case_keys(metadata)
    base.update(
        {
            "n_init_samples": n_init,
            "sample_interval_s": sample_interval_s,
            "residual_doppler_error_hz": float(residual_doppler_error_hz),
        }
    )

    score_rows = []
    score_curve_by_spec = {(int(n_init), float(residual_doppler_error_hz)): primary_score_curve}
    if include_trace:
        for score_n_init in score_n_init_samples or [n_init]:
            score_n_init = min(int(score_n_init), len(case_df))
            spec = (score_n_init, float(residual_doppler_error_hz))
            if spec in score_curve_by_spec:
                continue
            score_curve_by_spec[spec] = compute_filter_bank_score(
                signal[:score_n_init],
                sample_interval_s,
                candidates,
            )
    for (score_n_init, score_residual_doppler), score_curve in sorted(score_curve_by_spec.items()):
        estimated_score_tf = select_fresnel_time_estimate(candidates, score_curve)
        for i, candidate in enumerate(candidates):
            row = {
                **base,
                "true_fresnel_time_scale_s": true_tf,
                "n_init_samples": int(score_n_init),
                "residual_doppler_error_hz": float(score_residual_doppler),
                "estimated_fresnel_time_scale_s": estimated_score_tf,
                "candidate_index": i,
                "fresnel_time_candidate_s": float(candidate),
            }
            row[AMPLITUDE_SCORE_NAME] = float(score_curve[i])
            score_rows.append(row)

    estimate_rows = []
    downstream_rows = []
    trace_rows = []
    estimate_rows.append(
        {
            **base,
            "true_fresnel_time_scale_s": true_tf,
            "estimated_fresnel_time_scale_s": estimated_tf,
            "fresnel_time_error_s": estimated_tf - true_tf if np.isfinite(true_tf) else np.nan,
            "fresnel_time_abs_error_s": abs(estimated_tf - true_tf)
            if np.isfinite(true_tf)
            else np.nan,
            "selected_candidate_index": int(np.nanargmin(primary_score_curve)),
        }
    )
    backpropagated = batch_backpropagate(signal, sample_interval_s, estimated_tf)
    phase = median_align_phase(unwrap_phase(backpropagated), true_phase, n_init)
    phase_rmse = phase_rmse_rad(true_phase[n_init:], phase[n_init:])
    slip_count, slips_per_min = cycle_slips_per_minute(
        true_phase,
        phase,
        sample_interval_s,
        start_index=n_init,
    )
    downstream_rows.append(
        {
            **base,
            "true_fresnel_time_scale_s": true_tf,
            "estimated_fresnel_time_scale_s": estimated_tf,
            "phase_rmse_rad": phase_rmse,
            "cycle_slip_count": slip_count,
            "cycle_slips_per_minute": slips_per_min,
        }
    )
    if include_trace:
        intensity_db = detrended_intensity_db(case_df)
        tvd_cfg = params.get("tvd", {})
        # Plot-only convention: align the displayed tracked scintillation phase
        # to truth with the same post-N_init median rule used for BP estimates.
        # This only removes the arbitrary phase constant.
        aligned_estimated_scintillation_phase = median_align_phase(
            estimated_scintillation_phase,
            true_scintillation_phase,
            n_init,
        )
        batch_phase_error = phase - true_phase
        batch_integer_cycle_bias = tvd_integer_bias(
            batch_phase_error,
            intensity_db,
            float(tvd_cfg.get("fade_threshold_db", -10.0)),
            float(tvd_cfg.get("allowed_jump_weight", 1.0)),
            float(tvd_cfg.get("blocked_jump_weight", 10000.0)),
        )
        trace_rows.extend(
            sampled_time_rows(
                case_df,
                base,
                lambda idx: {
                    "intensity_db": float(intensity_db[idx]),
                    "true_scintillation_phase_rad": float(true_scintillation_phase[idx]),
                    "estimated_scintillation_phase_rad": float(
                        aligned_estimated_scintillation_phase[idx]
                    ),
                    "true_refractive_phase_rad": float(true_phase[idx]),
                    "backpropagated_refractive_phase_rad": float(phase[idx]),
                    "backpropagated_phase_rad": float(phase[idx]),
                    "true_scintillation_minus_refractive_rad": float(
                        true_scintillation_phase[idx] - true_phase[idx]
                    ),
                    "estimated_scintillation_minus_refractive_rad": float(
                        aligned_estimated_scintillation_phase[idx] - true_phase[idx]
                    ),
                    "phase_error_rad": float(phase[idx] - true_phase[idx]),
                    "batch_refractive_phase_error_rad": float(batch_phase_error[idx]),
                    "batch_refractive_phase_error_cycles": float(
                        batch_phase_error[idx] / (2.0 * np.pi)
                    ),
                    "batch_integer_cycle_bias": int(batch_integer_cycle_bias[idx]),
                },
            )
        )
    return {
        "score_rows": score_rows,
        "estimate_rows": estimate_rows,
        "downstream_rows": downstream_rows,
        "trace_rows": trace_rows,
    }


def _iter_filter_bank_tasks(
    case_iter,
    params: dict,
    candidates: np.ndarray,
    wavelength_m: float,
    *,
    residual_doppler_error_hz: float,
    n_init: int,
    score_n_init_samples: list[int] | None = None,
    include_representative_traces: bool = False,
    experiments: dict | None = None,
    representative_case_ids: set[int] | None = None,
):
    """Yield filter-bank case tasks with representative trace selection.

    Main paper:
        Results Figure 2 uses one representative score curve per severity at
        the configured carrier-to-noise ratio, while Section III.2 metrics use
        all eligible cases.
    """
    traced_severities: set[str] = set()
    experiments = experiments or {}
    representative_case_ids = representative_case_ids or set()
    for metadata, case_df in case_iter:
        severity = str(metadata.get("severity"))
        case_id = int(metadata.get("tracking_case_id", 0) or 0)
        include_trace = include_representative_traces and (
            case_id in representative_case_ids
            or (
                not representative_case_ids
                and is_representative_case(metadata, experiments)
                and severity not in traced_severities
            )
        )
        if include_trace:
            traced_severities.add(severity)
        yield (
            metadata,
            case_df,
            params,
            candidates,
            wavelength_m,
            residual_doppler_error_hz,
            n_init,
            score_n_init_samples,
            include_trace,
        )


def _compute_case_filter_bank_task(task: tuple) -> dict[str, list[dict]]:
    (
        metadata,
        case_df,
        params,
        candidates,
        wavelength_m,
        residual_doppler_error_hz,
        n_init,
        score_n_init_samples,
        include_trace,
    ) = task
    return _compute_case_filter_bank(
        metadata,
        case_df,
        params,
        candidates,
        float(wavelength_m),
        residual_doppler_error_hz=float(residual_doppler_error_hz),
        n_init=int(n_init),
        score_n_init_samples=score_n_init_samples,
        include_trace=bool(include_trace),
    )


def _compute_n_init_sensitivity(
    params: dict,
    paths,
    candidates: np.ndarray,
    wavelength_m: float,
) -> pd.DataFrame:
    """Compute the N_init sensitivity table.

    Main paper:
        Results Figure 4 sweeps N_init to justify the fixed initialization
        length used by the later filter-bank and near-real-time experiments.
    """
    experiments = params.get("experiments", {})
    rows: list[dict] = []
    max_cases = optional_int(experiments.get("n_init_sweep_max_cases_per_condition"))
    chunksize = csv_chunksize_from_params(params)
    workers = parallel_workers_from_params(params)
    n_init_values = fixed_step_values(
        experiments.get("n_init_sweep_samples"),
        default=[int(experiments.get("n_init_samples", 4096))],
        dtype=int,
    )
    for n_init in n_init_values:
        logger.info("Computing N_init sensitivity for N_init={}", n_init)
        estimate_rows: list[dict] = []
        downstream_rows: list[dict] = []
        case_iter = iter_case_dataframes(
            paths,
            severities=experiments.get("severities"),
            max_cases_per_condition=max_cases,
            chunksize=chunksize,
            columns=FILTER_BANK_TRACKING_COLUMNS,
        )
        task_iter = _iter_filter_bank_tasks(
            case_iter,
            params,
            candidates,
            wavelength_m,
            residual_doppler_error_hz=0.0,
            n_init=int(n_init),
            score_n_init_samples=None,
        )
        result_iter = ordered_parallel_map(
            _compute_case_filter_bank_task,
            task_iter,
            workers=workers,
        )
        total_cases = count_cases(
            paths,
            severities=experiments.get("severities"),
            max_cases_per_condition=max_cases,
        )
        for result in progress_iter(
            result_iter,
            desc=f"N_init={n_init} cases",
            total=total_cases,
            unit="case",
        ):
            estimate_rows.extend(result["estimate_rows"])
            downstream_rows.extend(result["downstream_rows"])
        estimates = pd.DataFrame(estimate_rows)
        downstream = pd.DataFrame(downstream_rows)
        if estimates.empty:
            continue
        for severity, group in severity_groups(estimates):
            downstream_group = downstream[downstream["severity"] == severity]
            rows.append(
                {
                    "n_init_samples": int(n_init),
                    "severity": severity,
                    "rmse_fresnel_time_scale_s": rmse(group["fresnel_time_error_s"].to_numpy()),
                    "r2_fresnel_time_scale": r2_score(
                        group["true_fresnel_time_scale_s"].to_numpy(),
                        group["estimated_fresnel_time_scale_s"].to_numpy(),
                    ),
                    "mean_cycle_slips_per_minute": float(
                        downstream_group["cycle_slips_per_minute"].mean()
                    )
                    if not downstream_group.empty
                    else np.nan,
                }
            )
    return sort_by_severity(pd.DataFrame(rows), ["n_init_samples"])


def _compute_boxplot_estimates(
    params: dict,
    paths,
    candidates: np.ndarray,
    wavelength_m: float,
) -> pd.DataFrame:
    """Compute per-case T_F estimates for the boxplot N_init setting.

    Main paper:
        Results Figure 4 uses the largest configured N_init in the boxplot
        column so the T_F estimate distribution is shown for the strongest
        initialization evidence available in the regenerated dataset.
    """
    experiments = params.get("experiments", {})
    n_init = int(
        experiments.get("n_init_samples_boxplot", experiments.get("n_init_samples", 4096))
    )
    chunksize = csv_chunksize_from_params(params)
    workers = parallel_workers_from_params(params)
    case_iter = iter_case_dataframes(
        paths,
        severities=experiments.get("severities"),
        max_cases_per_condition=optional_int(experiments.get("max_cases_per_condition")),
        chunksize=chunksize,
        columns=FILTER_BANK_TRACKING_COLUMNS,
    )
    task_iter = _iter_filter_bank_tasks(
        case_iter,
        params,
        candidates,
        wavelength_m,
        residual_doppler_error_hz=0.0,
        n_init=n_init,
        score_n_init_samples=None,
    )
    result_iter = ordered_parallel_map(
        _compute_case_filter_bank_task,
        task_iter,
        workers=workers,
    )
    total_cases = count_cases(
        paths,
        severities=experiments.get("severities"),
        max_cases_per_condition=optional_int(experiments.get("max_cases_per_condition")),
    )
    estimate_rows = []
    for result in progress_iter(
        result_iter,
        desc=f"boxplot N_init={n_init} cases",
        total=total_cases,
        unit="case",
    ):
        estimate_rows.extend(result["estimate_rows"])
    return sort_by_severity(pd.DataFrame(estimate_rows), ["tracking_case_id"])


def _score_curve_n_init_samples(experiments: dict) -> list[int]:
    values = experiments.get("score_curve_n_init_samples")
    if values:
        return sorted({int(value) for value in values})
    return [512, 1024, int(experiments.get("n_init_samples", 4096))]


def _aggregate_tf_error_metrics(estimates: pd.DataFrame) -> pd.DataFrame:
    if estimates.empty:
        return pd.DataFrame()
    rows = []
    for severity, group in severity_groups(estimates):
        rows.append(
            {
                "severity": severity,
                "mae_fresnel_time_scale_s": mae(group["fresnel_time_error_s"].to_numpy()),
                "mape_fresnel_time_scale_pct": mape(
                    group["true_fresnel_time_scale_s"].to_numpy(),
                    group["estimated_fresnel_time_scale_s"].to_numpy(),
                ),
                "rmse_fresnel_time_scale_s": rmse(group["fresnel_time_error_s"].to_numpy()),
                "r2_fresnel_time_scale": r2_score(
                    group["true_fresnel_time_scale_s"].to_numpy(),
                    group["estimated_fresnel_time_scale_s"].to_numpy(),
                ),
                "n_cases": int(group.shape[0]),
            }
        )
    return sort_by_severity(pd.DataFrame(rows), [])


def _aggregate_downstream_metrics(downstream: pd.DataFrame) -> pd.DataFrame:
    if downstream.empty:
        return pd.DataFrame()
    summary = (
        downstream.groupby("severity", dropna=False)
        .agg(
            mean_phase_rmse_rad=("phase_rmse_rad", "mean"),
            mean_cycle_slips_per_minute=("cycle_slips_per_minute", "mean"),
            n_cases=("phase_rmse_rad", "size"),
        )
        .reset_index()
    )
    return sort_by_severity(summary, [])


if __name__ == "__main__":
    main()
