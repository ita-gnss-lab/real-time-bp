from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
import sys
from typing import TypeVar

from loguru import logger
from tqdm.auto import tqdm

T = TypeVar("T")

_LOGGING_CONFIGURED = False


def configure_logging() -> None:
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
    yield from tqdm(
        iterable,
        desc=desc,
        total=total,
        unit=unit,
        dynamic_ncols=True,
        mininterval=2.0,
        smoothing=0.1,
    )


@contextmanager
def progress_bar(*, desc: str, total: int | None = None, unit: str = "it"):
    bar = tqdm(
        desc=desc,
        total=total,
        unit=unit,
        dynamic_ncols=True,
        mininterval=2.0,
        smoothing=0.1,
    )
    try:
        yield bar
    finally:
        bar.close()
