"""DVC-backed implementation of the real-time backpropagation paper pipeline."""

from __future__ import annotations

import os

for thread_env_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(thread_env_var, "1")
