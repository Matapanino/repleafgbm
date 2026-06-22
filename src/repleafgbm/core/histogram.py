"""Feature binning for histogram-based split search.

Raw features are quantized once per ``fit`` into small integer bins. Split
search then works on bin indices, which keeps per-node cost linear in the
number of rows and bins. This mirrors LightGBM-style histogram training and
gives the future Rust/C++/CUDA backends a simple data layout to target:
a (n_rows, n_features) uint16 matrix plus per-feature threshold arrays.

Bin semantics for feature j with thresholds t[0] < ... < t[k-1]:

* bin b in [0, k-1]  -> x <= t[b] (and x > t[b-1] for b > 0)
* bin k              -> x > t[k-1]
* bin k + 1          -> missing (NaN). Missing always routes left in v0.
"""

from __future__ import annotations

import numpy as np

from repleafgbm.utils.parallel import map_features


def compute_bin_thresholds(
    X: np.ndarray, max_bins: int = 256, *, n_threads: int | None = None
) -> list[np.ndarray]:
    """Per-feature sorted candidate thresholds from quantiles of non-NaN values.

    Returns a list of float64 arrays (possibly empty for constant features).

    Each feature is independent, so the per-feature work (``np.unique`` /
    ``np.quantile``) is mapped across a thread pool
    (:func:`repleafgbm.utils.parallel.map_features`) and reassembled in feature
    order — bitwise-identical to the serial loop regardless of thread count.
    """
    n_rows, n_features = X.shape
    qs = np.linspace(0.0, 1.0, max_bins + 1)[1:-1]  # interior quantiles

    def _thresholds_for(j: int) -> np.ndarray:
        col = X[:, j]
        valid = col[~np.isnan(col)]
        if valid.size == 0:
            return np.empty(0, dtype=np.float64)
        uniq = np.unique(valid)
        if uniq.size <= max_bins:
            # Midpoints between consecutive unique values are exact candidates.
            return ((uniq[:-1] + uniq[1:]) / 2.0).astype(np.float64)
        return np.unique(np.quantile(valid, qs)).astype(np.float64)

    return map_features(
        _thresholds_for, n_features,
        work_cells=n_rows * n_features, n_threads=n_threads,
    )


def bin_features(
    X: np.ndarray, thresholds: list[np.ndarray], *, n_threads: int | None = None
) -> np.ndarray:
    """Quantize features into bin indices (uint16) following the semantics above.

    Per-feature binning is mapped across a thread pool; each feature computes
    its own contiguous column (avoiding cross-thread false sharing) and the
    columns are assembled in order, so the result is bitwise-identical to the
    serial loop regardless of thread count.
    """
    n_rows, n_features = X.shape

    def _bin_column(j: int) -> np.ndarray:
        t = thresholds[j]
        col = X[:, j]
        missing = np.isnan(col)
        # searchsorted with side="left": x <= t[b] -> bin b; x > t[-1] -> len(t).
        b = np.searchsorted(t, col, side="left")
        b[missing] = len(t) + 1
        return b.astype(np.uint16)

    columns = map_features(
        _bin_column, n_features,
        work_cells=n_rows * n_features, n_threads=n_threads,
    )
    binned = np.empty((n_rows, n_features), dtype=np.uint16)
    for j in range(n_features):
        binned[:, j] = columns[j]
    return binned


def missing_bin(thresholds_j: np.ndarray) -> int:
    """Bin index reserved for missing values of one feature."""
    return len(thresholds_j) + 1
