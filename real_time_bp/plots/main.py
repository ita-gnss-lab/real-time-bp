"""Entry point for regenerating all paper figures from CSV results."""

from __future__ import annotations

from loguru import logger

from real_time_bp.plots.cycle_slips import plot_cycle_slips
from real_time_bp.plots.filter_bank import plot_filter_bank
from real_time_bp.plots.sliding import plot_sliding
from real_time_bp.utils.params import load_params
from real_time_bp.utils.plot_style import apply_plot_style
from real_time_bp.utils.progress import configure_logging


def main() -> None:
    """Apply plot styling and render every figure group."""
    apply_plot_style()
    configure_logging()
    logger.info("Starting result plot stage")
    load_params()
    plot_cycle_slips()
    plot_filter_bank()
    plot_sliding()
    logger.info("Finished result plot stage")


if __name__ == "__main__":
    main()
