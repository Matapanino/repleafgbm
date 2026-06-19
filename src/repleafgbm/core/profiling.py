"""Internal per-phase wall-clock profiler for the fit/predict path.

Off by default and **not** part of the public API. The sklearn estimators build
a :class:`PhaseProfiler` only when the ``REPLEAFGBM_PROFILE`` environment
variable is set (see :func:`profiler_from_env`), thread it through the internal
booster/splitter, and expose the accumulated seconds as the fitted
``phase_seconds_`` attribute — consumed by ``benchmarks/gpu_profile.py`` to fill
its ``phase_seconds`` field (``docs/gpu_roadmap.md`` Phase 0).

When the env var is unset, ``profiler_from_env`` returns ``None`` and every
record site goes through :func:`timed`, which is a no-op for ``None`` — one
``is None`` check, no clock reads, no allocation — so the default training and
prediction paths are unchanged.

The idiom mirrors the CUDA backend's private transfer counters
(``backends/cuda_backend.py``): a tiny accumulator with ``add`` + snapshot,
adapted to seconds with a context-manager timer. Pure Python with no optional
dependencies, so it is import-safe from the native (Rust) path.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter

#: Environment-variable values that turn profiling on.
_TRUTHY = frozenset({"1", "true", "True", "TRUE", "yes", "on"})


class PhaseProfiler:
    """Accumulates wall-clock seconds per named phase.

    Holds an insertion-ordered ``{phase_name: seconds}`` mapping of plain
    floats, so a snapshot is trivially JSON- and pickle-serializable. Times are
    *accumulated*: recording the same name repeatedly — e.g. one ``"histogram"``
    entry per tree node — sums into a single total.
    """

    def __init__(self) -> None:
        self._seconds: dict[str, float] = {}

    def add(self, name: str, seconds: float) -> None:
        """Accumulate ``seconds`` under ``name``."""
        self._seconds[name] = self._seconds.get(name, 0.0) + float(seconds)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Time the wrapped block and accumulate it under ``name``."""
        t0 = perf_counter()
        try:
            yield
        finally:
            self.add(name, perf_counter() - t0)

    def as_dict(self) -> dict[str, float]:
        """A copy of the accumulated per-phase seconds."""
        return dict(self._seconds)


@contextmanager
def timed(profiler: PhaseProfiler | None, name: str) -> Iterator[None]:
    """Time the wrapped block under ``name`` when profiling is on; else no-op.

    Lets every record site read uniformly as ``with timed(self._profiler, ...)``
    whether or not profiling is enabled: when ``profiler is None`` the block runs
    with no clock reads (one branch), so per-node sites stay cheap on the default
    path.
    """
    if profiler is None:
        yield
        return
    t0 = perf_counter()
    try:
        yield
    finally:
        profiler.add(name, perf_counter() - t0)


def profiler_from_env() -> PhaseProfiler | None:
    """A fresh :class:`PhaseProfiler` when ``REPLEAFGBM_PROFILE`` is truthy.

    Returns ``None`` otherwise, which callers treat as "profiling disabled". The
    variable is read once per fit/predict to build a per-call instance, so there
    is no module-level mutable state.
    """
    if os.environ.get("REPLEAFGBM_PROFILE", "") in _TRUTHY:
        return PhaseProfiler()
    return None
