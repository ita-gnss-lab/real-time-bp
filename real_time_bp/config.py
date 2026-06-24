"""Project paths shared by DVC stages, analysis scripts, and plotting code."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAMS_FILE = PROJECT_ROOT / "params.yaml"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = PROJECT_ROOT / "results"
CSV_TABLES_DIR = RESULTS_DIR / "csv_tables"
FIGURES_DIR = RESULTS_DIR / "figures"
