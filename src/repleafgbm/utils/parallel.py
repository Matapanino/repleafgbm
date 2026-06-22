"""Feature-parallel map for once-per-fit per-feature work (e.g. binning).

Several once-per-fit steps run an independent computation per feature: quantile
threshold search, bin assignment, column encoding. Each feature's work is a
NumPy C call (``np.unique`` / ``np.quantile`` / ``np.searchsorted``) that
*releases the GIL*, so dispatching the features across a thread pool gives real
parallelism on a multicore host.

The result is **bitwise-identical to the serial loop** by construction: every
feature runs the exact same NumPy call it would run serially, and results are
reassembled in feature order, so the output never depends on the thread count
or on scheduling. That keeps this a pure performance optimization — it does not
change any model output, and (because binning is upstream of the split backend)
it does not touch the NumPy/Rust histogram parity.

Pool size is resolved once per call from ``REPLEAFGBM_NUM_THREADS`` (falling
back to ``os.cpu_count()``), mirroring how :mod:`repleafgbm.core.profiling`
reads ``REPLEAFGBM_PROFILE`` — there is no module-level mutable state.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")

#: Below this many ``rows * features`` cells the thread-pool setup costs more
#: than it saves, so :func:`map_features` stays serial. This keeps small inputs
#: (and the test suite) overhead-free; it never affects the output, only timing.
PARALLEL_MIN_CELLS = 1 << 20


def resolve_n_threads(n_features: int) -> int:
    """Worker count for a feature-parallel step (never more than ``n_features``).

    Reads ``REPLEAFGBM_NUM_THREADS`` if set to a positive integer, else uses
    ``os.cpu_count()``. The value affects only speed, never the result.
    """
    env = os.environ.get("REPLEAFGBM_NUM_THREADS", "")
    if env:
        try:
            requested = int(env)
        except ValueError:
            requested = 0
        if requested >= 1:
            return max(1, min(requested, n_features))
    return max(1, min(os.cpu_count() or 1, n_features))


def map_features(
    fn: Callable[[int], T],
    n_features: int,
    *,
    work_cells: int,
    n_threads: int | None = None,
) -> list[T]:
    """Apply ``fn(j)`` for ``j`` in ``range(n_features)``, results in order.

    Runs serially when there is at most one feature, when the (requested or
    resolved) thread count is 1, or when ``work_cells`` is below
    :data:`PARALLEL_MIN_CELLS`; otherwise dispatches across a scoped
    ``ThreadPoolExecutor``. ``fn`` must be independent per feature (read shared
    inputs, write only its own returned value or a disjoint location) so the
    output is identical regardless of thread count or scheduling.

    Args:
        fn: Per-feature callable taking the feature index.
        n_features: Number of features to map over.
        work_cells: Rough total work size (e.g. ``n_rows * n_features``) used to
            gate parallelism; small inputs stay serial.
        n_threads: Override the resolved worker count (mainly for tests). ``1``
            forces the serial path.

    Returns:
        ``[fn(0), fn(1), ..., fn(n_features - 1)]``.
    """
    if n_threads is None:
        n_threads = resolve_n_threads(n_features)
    if n_features <= 1 or n_threads <= 1 or work_cells < PARALLEL_MIN_CELLS:
        return [fn(j) for j in range(n_features)]
    with ThreadPoolExecutor(max_workers=min(n_threads, n_features)) as executor:
        # executor.map preserves input order in the returned iterator.
        return list(executor.map(fn, range(n_features)))
