"""Parameter loading for DVC stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from real_time_bp.config import PARAMS_FILE


def load_params(path: Path = PARAMS_FILE) -> dict[str, Any]:
    """Load YAML parameters from disk.

    Args:
        path: Parameter file path.

    Returns:
        Parsed parameter dictionary, or an empty dictionary for an empty file.
    """
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def fixed_step_values(
    spec: Any,
    *,
    default: list[int | float],
    dtype: type = float,
) -> np.ndarray:
    """Build an inclusive fixed-step grid from params.

    Args:
        spec: Either a list of explicit values or a mapping with `min`, `max`,
            and `step`.
        default: Values used when `spec` is missing.
        dtype: Output dtype, typically `int` or `float`.

    Returns:
        Sorted unique values as a NumPy array.
    """
    if spec is None:
        values = np.asarray(default, dtype=dtype)
    elif isinstance(spec, dict):
        start = float(spec["min"])
        stop = float(spec["max"])
        step = float(spec["step"])
        if step <= 0:
            msg = "Fixed-step params require a positive step size."
            raise ValueError(msg)
        values = np.arange(start, stop + 0.5 * step, step, dtype=float)
        values = values[values <= stop + 1e-9]
        values = np.round(values, 10)
        if dtype is int:
            values = np.rint(values).astype(int)
        else:
            values = values.astype(dtype)
    else:
        values = np.asarray(spec, dtype=dtype)
    return np.asarray(sorted(set(values.tolist())), dtype=dtype)
