import multiprocessing as mp
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor


class BackgroundTasks:
    """Calls of `fn` on each tuple of arguments in `iterable`, running in a pool of
    spawned processes while the caller does other work."""

    def __init__(self, n_workers: int, fn: Callable, iterable, initializer: Callable | None = None):
        self._executor = ProcessPoolExecutor(
            max_workers=n_workers, mp_context=mp.get_context("spawn"), initializer=initializer
        )
        self._futures = [self._executor.submit(fn, *args) for args in iterable]

    def wait(self) -> list:
        """The results in order, raising the first exception of a task if any."""
        try:
            return [future.result() for future in self._futures]
        finally:
            self._executor.shutdown(cancel_futures=True)

    def cancel(self) -> None:
        """Drop the tasks not started yet, without waiting for the others."""
        self._executor.shutdown(wait=False, cancel_futures=True)
