from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def ordered_parallel_map(
    function: Callable[[T], R],
    iterable: Iterable[T],
    *,
    workers: int,
) -> Iterator[R]:
    worker_count = max(int(workers), 1)
    if worker_count == 1:
        for item in iterable:
            yield function(item)
        return

    pending_limit = worker_count
    iterator = iter(iterable)
    pending: deque[Future[R]] = deque()
    exhausted = False

    def fill_queue(executor: ProcessPoolExecutor) -> None:
        nonlocal exhausted
        while not exhausted and len(pending) < pending_limit:
            try:
                item = next(iterator)
            except StopIteration:
                exhausted = True
                return
            pending.append(executor.submit(function, item))

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        fill_queue(executor)
        while pending:
            future = pending.popleft()
            yield future.result()
            fill_queue(executor)
