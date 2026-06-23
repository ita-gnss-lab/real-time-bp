"""Helpers for writing sampled time-series CSV rows."""

from __future__ import annotations

from collections.abc import Callable


def sample_indices(n_samples: int, max_points: int = 2500) -> range:
    """Return a stride-based index range capped by an approximate point count.

    Args:
        n_samples: Total number of samples.
        max_points: Approximate maximum number of returned samples.

    Returns:
        Range of sample indices with stride at least one.
    """
    stride = max(int(n_samples) // int(max_points), 1)
    return range(0, int(n_samples), stride)


def sampled_time_rows(
    case_df,
    base: dict,
    build_values: Callable[[int], dict],
    *,
    max_points: int = 2500,
) -> list[dict]:
    """Build sampled time-series rows with common case metadata.

    Args:
        case_df: Tracking samples containing a `time_sec` column.
        base: Metadata copied into every row.
        build_values: Callback that returns sample-specific values for an index.
        max_points: Approximate maximum number of rows to emit.

    Returns:
        List of row dictionaries suitable for CSV writing.
    """
    rows = []
    for idx in sample_indices(len(case_df), max_points=max_points):
        rows.append(
            {
                **base,
                "sample_index": int(idx),
                "time_sec": float(case_df["time_sec"].iloc[idx]),
                **build_values(int(idx)),
            }
        )
    return rows
