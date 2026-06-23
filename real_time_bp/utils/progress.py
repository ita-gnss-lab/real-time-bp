"""Logging and progress-bar helpers for long-running DVC stages."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
import sys
from typing import TypeVar

from loguru import logger
from tqdm.auto import tqdm

T = TypeVar("T")

_LOGGING_CONFIGURED = False


def configure_logging() -> None:
    """Configure Loguru once with compact timestamps for stage output."""
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:HH:mm:ss} | {level:<7} | {message}",
    )
    _LOGGING_CONFIGURED = True


def progress_iter(
    iterable: Iterable[T],
    *,
    desc: str,
    total: int | None = None,
    unit: str = "it",
) -> Iterator[T]:
    """Wrap an iterable in a tqdm progress bar.

    Args:
        iterable: Values to yield.
        desc: Progress-bar label.
        total: Optional total count.
        unit: Unit label shown by tqdm.

    Yields:
        Values from `iterable`.
    """
    yield from tqdm(
        iterable,
        desc=desc,
        total=total,
        unit=unit,
        dynamic_ncols=True,
        mininterval=2.0,
        smoothing=0.1,
    )
