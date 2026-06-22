from __future__ import annotations

from pathlib import Path

import kagglehub
from loguru import logger

from real_time_bp.config import PROJECT_ROOT
from real_time_bp.utils.params import load_params
from real_time_bp.utils.progress import configure_logging


def main() -> None:
    configure_logging()
    logger.info("Starting dataset download stage")
    params = load_params()
    dataset = params.get("dataset", {})
    output_dir = PROJECT_ROOT / dataset.get("output_dir", "data/raw/backpropagation")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading {} into {}", dataset.get("handle", "rodrigodlf/backpropagation"), output_dir)
    # Paper: Methodology Section II.3. DVC pins this as
    # the always-changed raw-data stage so all downstream Results CSVs are tied
    # to the current Kaggle dataset version.
    path = kagglehub.dataset_download(
        dataset.get("handle", "rodrigodlf/backpropagation"),
        force_download=bool(dataset.get("force_download", False)),
        output_dir=str(output_dir),
    )
    marker = output_dir / "_dataset_path.txt"
    marker.write_text(str(Path(path).resolve()) + "\n", encoding="utf-8")
    logger.info("Path to dataset files: {}", path)
    logger.info("Finished dataset download stage")


if __name__ == "__main__":
    main()
