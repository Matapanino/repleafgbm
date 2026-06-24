"""Microbench: NumPy vs Rust ``partition_rows`` across node sizes.

Isolates the per-node row-partition kernel from a full fit to confirm the fused
single-pass Rust kernel beats NumPy's multi-pass boolean index. The native path
won at every node size measured (~3-5x numeric, ~10-15x categorical, down to 16
rows), which is why ``RustSplitBackend.partition_rows`` carries no min-rows
fallback gate. Stdout only, CPU-safe, runs in seconds.

    OMP_NUM_THREADS=1 RAYON_NUM_THREADS=8 python -m benchmarks.partition_microbench
"""

from __future__ import annotations

import time

import numpy as np

from repleafgbm.backends import NumPySplitBackend, RustSplitBackend
from repleafgbm.backends.base import SplitCandidate

_SIZES = [16, 64, 256, 1024, 4096, 16384, 65536, 262144]


def _best_us(fn, *args, repeats: int = 50) -> float:
    """Best (min) wall time in microseconds over ``repeats`` runs."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best * 1e6


def main() -> None:
    rng = np.random.default_rng(0)
    n_rows, n_features, missing_bin = 500_000, 200, 256
    binned = rng.integers(0, missing_bin, size=(n_rows, n_features)).astype(np.uint16)
    binned[rng.random((n_rows, n_features)) < 0.05] = missing_bin  # missing bin

    np_b, rs_b = NumPySplitBackend(), RustSplitBackend()
    rs_b._feature_major(binned)  # warm the transpose (a fit warms it via histogram)

    feature = 7
    numeric = SplitCandidate(feature=feature, bin=128, gain=0.0, n_left=0, n_right=0)
    cats = np.sort(rng.choice(missing_bin, size=32, replace=False)).astype(np.int64)
    categorical = SplitCandidate(
        feature=feature, bin=-1, gain=0.0, n_left=0, n_right=0, left_categories=cats
    )

    print(f"partition_rows microbench  ({n_rows:,} rows x {n_features} features, "
          f"5% missing; RAYON_NUM_THREADS affects only the Rust path)")
    for split, label in ((numeric, "numeric"), (categorical, "categorical")):
        print(f"\n== {label} split ==")
        print(f"{'rows':>9} {'numpy_us':>10} {'rust_us':>10} {'speedup':>9}")
        for size in _SIZES:
            rows = np.sort(
                rng.choice(n_rows, size=size, replace=False)
            ).astype(np.int64)
            t_np = _best_us(np_b.partition_rows, binned, rows, split, missing_bin)
            t_rs = _best_us(rs_b.partition_rows, binned, rows, split, missing_bin)
            print(f"{size:>9} {t_np:>10.1f} {t_rs:>10.1f} {t_np / t_rs:>8.2f}x")


if __name__ == "__main__":
    main()
