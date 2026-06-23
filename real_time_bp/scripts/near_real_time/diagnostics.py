"""Near-real-time residual Doppler and VAR diagnostic CSV generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from real_time_bp.methodology.near_real_time import (
    select_var_order_bic,
    var_residual_test_summary,
)
from real_time_bp.methodology.signal_model import estimated_scintillation_signal
from real_time_bp.scripts.near_real_time.common import (
    NEAR_REAL_TIME_TRACKING_COLUMNS,
    near_real_time_residual_grid_hz,
)
from real_time_bp.scripts.near_real_time.heatmap import (
    iter_selected_heatmap_cases,
    select_heatmap_case_ids,
)
from real_time_bp.scripts.near_real_time.metrics import compute_practical_case
from real_time_bp.utils.io import iter_case_dataframes
from real_time_bp.utils.progress import progress_iter
from real_time_bp.utils.stage import case_keys, csv_chunksize_from_params


def compute_residual_doppler_metrics(
    params: dict,
    paths,
    wavelength_m: float,
    tf_estimates: dict[int, float],
    representative_case_ids: set[int],
) -> pd.DataFrame:
    """Evaluate practical BP metrics over residual Doppler errors.

    Args:
        params: Parsed `params.yaml` dictionary.
        paths: Resolved dataset paths.
        wavelength_m: Carrier wavelength in meters.
        tf_estimates: Fresnel-time estimates keyed by tracking case id.
        representative_case_ids: Case identifiers used for representative
            diagnostics.

    Returns:
        DataFrame of practical metric rows over the residual Doppler grid.
    """
    rows: list[dict] = []
    selected_case_ids = representative_case_ids or (
        select_heatmap_case_ids(params, paths) or set()
    )
    if not selected_case_ids:
        return pd.DataFrame()
    chunksize = csv_chunksize_from_params(params)
    case_iter = iter_case_dataframes(
        paths,
        severities=params.get("experiments", {}).get("severities"),
        max_cases_per_condition=None,
        chunksize=chunksize,
        columns=NEAR_REAL_TIME_TRACKING_COLUMNS,
    )
    cases = list(iter_selected_heatmap_cases(case_iter, selected_case_ids))
    residual_grid = near_real_time_residual_grid_hz(params)
    for metadata, case_df in progress_iter(
        cases,
        desc="near-real-time residual cases",
        total=len(cases),
        unit="case",
    ):
        case_id = int(metadata.get("tracking_case_id", 0) or 0)
        estimated_tf = tf_estimates.get(case_id, float(metadata.get("rhof_veff_ratio", np.nan)))
        if not np.isfinite(estimated_tf):
            continue
        for residual_row in residual_grid.itertuples():
            result = compute_practical_case(
                metadata,
                case_df,
                params,
                float(wavelength_m),
                float(estimated_tf),
                include_phase_example=False,
                include_forecast_example=False,
                residual_doppler_error_hz=float(residual_row.residual_doppler_error_hz),
            )
            rows.append(result["practical_row"])
    return pd.DataFrame(rows)


def compute_var_residual_diagnostics(
    params: dict,
    paths,
    wavelength_m: float,
    representative_case_ids: set[int],
) -> pd.DataFrame:
    """Compute standardized VAR residual diagnostics.

    Args:
        params: Parsed `params.yaml` dictionary.
        paths: Resolved dataset paths.
        wavelength_m: Carrier wavelength in meters.
        representative_case_ids: Case identifiers used for representative
            diagnostic plots.

    Returns:
        DataFrame containing residual samples for real and imaginary VAR
        components.
    """
    selected_case_ids = representative_case_ids or (
        select_heatmap_case_ids(params, paths) or set()
    )
    if not selected_case_ids:
        return pd.DataFrame()
    chunksize = csv_chunksize_from_params(params)
    case_iter = iter_case_dataframes(
        paths,
        severities=params.get("experiments", {}).get("severities"),
        max_cases_per_condition=None,
        chunksize=chunksize,
        columns=NEAR_REAL_TIME_TRACKING_COLUMNS,
    )
    rows: list[dict] = []
    experiments = params.get("experiments", {})
    near_cfg = params.get("near_real_time", {})
    for metadata, case_df in iter_selected_heatmap_cases(case_iter, selected_case_ids):
        signal = estimated_scintillation_signal(case_df, float(wavelength_m))
        n_init = min(int(experiments.get("n_init_samples", 4096)), len(case_df))
        features = np.column_stack([np.real(signal[:n_init]), np.imag(signal[:n_init])])
        _, var_result = select_var_order_bic(
            features,
            int(near_cfg.get("var_order_min", 1)),
            int(near_cfg.get("var_order_max", 24)),
        )
        test_summary = var_residual_test_summary(
            var_result,
            alpha=float(near_cfg.get("var_test_alpha", 0.05)),
            whiteness_lags=int(near_cfg.get("var_whiteness_test_lags", 24)),
        )
        residuals = np.asarray(var_result.resid, dtype=float)
        start_sample = int(var_result.k_ar)
        for component_idx, component in enumerate(["real", "imag"]):
            values = residuals[:, component_idx]
            scale = np.nanstd(values)
            standardized = (
                (values - np.nanmean(values)) / scale if scale > 0 else values - np.nanmean(values)
            )
            for offset, value in enumerate(standardized):
                sample_index = start_sample + offset
                rows.append(
                    {
                        **case_keys(metadata),
                        "component": component,
                        "diagnostic": "residual",
                        "n_init_samples": int(n_init),
                        "sample_index": int(sample_index),
                        "time_sec": float(case_df["time_sec"].iloc[sample_index]),
                        "value": float(value),
                        **test_summary,
                    }
                )
    return pd.DataFrame(rows)
