"""Parameter loading for DVC stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
