from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from real_time_bp.config import PARAMS_FILE


def load_params(path: Path = PARAMS_FILE) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def get_nested(params: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = params
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current
